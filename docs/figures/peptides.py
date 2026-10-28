# Draw the figures of the peptides' page

# Libraries
import argparse, math, pprint, sys, pathlib, tempfile
import numpy as np
import torch
import mdtraj
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.patheffects import withStroke
from scipy.ndimage import gaussian_filter
from samplingdb import build
from samplingdb.metrics import histogram_2d, kmeans, subsample
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import blank, density_map, legend, note, pymol_render, save, save_gif, start_pymol

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=2000)
parser.add_argument('--n_background', type=int, default=100000)
parser.add_argument('--stride', type=int, default=10)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# The four peptides, in the order of the page's table
NAMES = ['ace-a-nme-T300', 'ace-aaa-nme-T300', 'aaa-T310', 'aaaaaa-T310']

# The colour of what a render points at inside a molecule, one shade that reads on both surfaces
ACCENT = '#8a6cf0'

# PyMOL's default vertical field of view, which fixes how many Angstrom a render spans
FIELD_OF_VIEW = 20.0


def pdb_string(topology, positions):
    """Writes one structure as the text of a PDB file

    Args:
        * topology (mdtraj.Topology): Topology
        * positions (numpy.ndarray of shape (n_atoms, 3)): Positions in Angstrom

    Returns:
        * pdb (str): Contents of the PDB file
    """
    with tempfile.TemporaryDirectory() as folder:
        path = folder + '/frame.pdb'
        mdtraj.Trajectory(positions[None] / 10.0, topology).save_pdb(path)
        return open(path).read()


def principal_frame(positions):
    """Returns the centre and the proper rotation putting a structure on its principal axes

    NOTE: the rotation is forced to determinant +1, since a reflection would draw the mirror image
    of the molecule, which is a different chirality.

    Args:
        * positions (numpy.ndarray of shape (n_atoms, 3)): Positions

    Returns:
        * centre (numpy.ndarray of shape (3,)): Centre of geometry
        * rotation (numpy.ndarray of shape (3, 3)): Columns along decreasing variance
    """
    centre = positions.mean(0)
    _, vectors = np.linalg.eigh(np.cov((positions - centre).T))
    rotation = vectors[:, ::-1].copy()
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1
    return centre, rotation


def render(topology, positions, width, height, extent, setup):
    """Ray-traces a structure already in its frame, looking down -z at a fixed scale

    Args:
        * topology (mdtraj.Topology): Topology
        * positions (numpy.ndarray of shape (n_atoms, 3)): Centred positions in Angstrom
        * width (int): Width of the image, in pixels
        * height (int): Height of the image, in pixels
        * extent (float): Height of the image, in Angstrom
        * setup (function): cmd -> None, the representation

    Returns:
        * image (numpy.ndarray of shape (height, width, 4)): RGBA image
    """
    distance = extent / (2.0 * math.tan(math.radians(FIELD_OF_VIEW / 2.0)))
    view = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -distance, 0, 0, 0, distance - 40, distance + 40, 1)
    return pymol_render(pdb_string(topology, positions), width, height, setup, view=view)


def project(points, width, height, extent):
    """Maps positions of a render to pixel columns and rows, as render() draws them

    Args:
        * points (numpy.ndarray of shape (n, 3)): Positions in Angstrom, in the render's frame
        * width (int): Width of the image, in pixels
        * height (int): Height of the image, in pixels
        * extent (float): Height of the image, in Angstrom

    Returns:
        * pixels (numpy.ndarray of shape (n, 2)): Column and row of every point
    """
    scale = height / extent
    return np.stack([(width - 1) / 2.0 + points[:, 0] * scale,
                     (height - 1) / 2.0 - points[:, 1] * scale], axis=-1)


def ball_and_stick(bonds=(), centre=None):
    """Returns the setup of a ball-and-stick render, with some bonds and one atom in the accent

    Args:
        * bonds (list of tuple of two ints): Atom indexes of the bonds to draw in the accent
            (default is ())
        * centre (int or None): Atom index drawn as a larger sphere in the accent
            (default is None)

    Returns:
        * setup (function): cmd -> None
    """
    def setup(cmd):
        cmd.hide('everything', 'mol')
        cmd.show('sticks', 'mol')
        cmd.show('spheres', 'mol')
        cmd.set('stick_radius', 0.14)
        cmd.set('sphere_scale', 0.24)
        cmd.set('sphere_scale', 0.17, 'mol and elem H')
        cmd.set_color('accent', list(to_rgb(ACCENT)))
        for i, j in bonds:
            pair = ('mol and id %d' % (i + 1), 'mol and id %d' % (j + 1))
            cmd.set_bond('stick_color', 'accent', *pair)
            cmd.set_bond('stick_radius', 0.24, *pair)
        if centre is not None:
            cmd.color('accent', 'mol and id %d' % (centre + 1))
            cmd.set('sphere_scale', 0.38, 'mol and id %d' % (centre + 1))
    return setup


def free_energy(values, x_edges, y_edges, periodic, smooth=1.0):
    """Estimates the free energy -log p of a 2D sample set on a grid, zero at its minimum

    Args:
        * values (numpy.ndarray of shape (n, 2)): The two coordinates
        * x_edges (numpy.ndarray of shape (n_x + 1,)): Edges along the first
        * y_edges (numpy.ndarray of shape (n_y + 1,)): Edges along the second
        * periodic (bool): Whether both coordinates are angles, which the smoothing wraps
        * smooth (float): Width of the Gaussian smoothing, in bins (default is 1.0)

    Returns:
        * energy (numpy.ndarray of shape (n_x, n_y)): Free energy in kT, inf where p is zero
    """
    counts = np.histogram2d(values[:, 0], values[:, 1], bins=[x_edges, y_edges])[0]
    counts = gaussian_filter(counts, smooth, mode='wrap' if periodic else 'constant')
    with np.errstate(divide='ignore'):
        return -np.log(counts / counts.max())


def free_energy_colours(theme, depth=1.0, top=8):
    """Returns one colour per 1 kT band from 0 to top, the lowest free energy last"""
    cmap = density_map(theme, depth=depth, levels=top + 1)
    return [cmap(k / top) for k in range(1, top + 1)]


def draw_free_energy(ax, energy, x_edges, y_edges, theme, depth=1.0, top=8):
    """Fills the 1 kT contours of a free energy up to top, leaving the rest as the page surface"""
    x = 0.5 * (x_edges[1:] + x_edges[:-1])
    y = 0.5 * (y_edges[1:] + y_edges[:-1])
    ax.contourf(x, y, (top - energy).T, levels=np.arange(0, top + 1),
                colors=free_energy_colours(theme, depth, top))


def free_energy_bar(fig, cax, theme, depth=1.0, top=8):
    """Draws the scale of draw_free_energy's contours"""
    colours = free_energy_colours(theme, depth, top)[::-1]
    mappable = ScalarMappable(norm=BoundaryNorm(np.arange(top + 1), top),
                              cmap=ListedColormap(colours))
    bar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    bar.set_ticks([0, top // 2, top])
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.set_label('free energy (kT)', fontsize=7.5)


def marginal(ax, edges, fill, test, train, theme, horizontal=False):
    """Draws one coordinate's marginal: the whole test split as a fill, the two scored sets on top

    Args:
        * ax (matplotlib.axes.Axes): Axis
        * edges (numpy.ndarray of shape (n_bins + 1,)): Bin edges
        * fill (numpy.ndarray): Values of the whole test split
        * test (numpy.ndarray): Values of the scored test rows
        * train (numpy.ndarray): Values of the train split
        * theme (dict): Theme
        * horizontal (bool): Whether the coordinate runs vertically, for a right-hand marginal
            (default is False)
    """
    orientation = 'horizontal' if horizontal else 'vertical'
    width = edges[1] - edges[0]
    density = [np.histogram(values, bins=edges)[0] / len(values) / width
               for values in [fill, test, train]]
    ax.stairs(density[0], edges, fill=True, color=density_map(theme)(0.45),
              orientation=orientation)
    ax.stairs(density[1], edges, color=theme['test'], linewidth=1.1, orientation=orientation)
    ax.stairs(density[2], edges, color=theme['train'], linewidth=1.1, orientation=orientation)
    peak = 1.1 * max(values.max() for values in density)
    if horizontal:
        ax.set_xlim(0.0, peak)
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylim(0.0, peak)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelbottom=False)


def basin(phi, psi):
    """Labels backbone angles in degrees as 0 for beta and PII, 1 for alpha-R, 2 for alpha-L"""
    alpha_r = (phi < 0) & (psi > -125) & (psi < 50)
    return np.where(phi > 0, 2, np.where(alpha_r, 1, 0))


# Build the four peptides, on the CPU and without compiling: every call here is a one-off
start_pymol()
targets = {name: build(name, compile=False) for name in NAMES}
aldp = targets['ace-a-nme-T300']
topologies = {name: mdtraj.Topology.from_openmm(target.topology)
              for name, target in targets.items()}

# One test structure per peptide, the one whose radius of gyration is the median
structures = {}
for name, target in targets.items():
    x = 10.0 * target.positions(target.test_samples(n=args.n_samples)).numpy()
    radius = np.sqrt(((x - x.mean(1, keepdims=True)) ** 2).sum(-1).mean(-1))
    frame = x[np.argsort(radius)[len(radius) // 2]]
    centre, rotation = principal_frame(frame)
    structures[name] = (frame - centre) @ rotation

# A common scale: the image height in Angstrom that fits the largest of the four
panel_width, panel_height, margin = 760, 420, 2.4
extent = max(max(np.ptp(x[:, 1]) + 2 * margin,
                 (np.ptp(x[:, 0]) + 2 * margin) * panel_height / panel_width)
             for x in structures.values())
print('molecules: %.1f Angstrom per image height' % extent)

# The atoms of Ace-A-Nme's two backbone torsions: phi turns about N-CA, psi about CA-C
phi_atoms, psi_atoms = aldp.torsion_indexes.tolist()
phi_bond, psi_bond = (phi_atoms[1], phi_atoms[2]), (psi_atoms[1], psi_atoms[2])
renders = {}
for name in NAMES:
    bonds = [phi_bond, psi_bond] if name == 'ace-a-nme-T300' else []
    renders[name] = render(topologies[name], structures[name], panel_width, panel_height, extent,
                           ball_and_stick(bonds))


def draw_molecules(theme):
    """Draws one test structure of each peptide, at one scale, with Ace-A-Nme's torsion axes"""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.6))
    for ax, name in zip(axes.flat, NAMES):
        target = targets[name]
        ax.imshow(renders[name], extent=(0, panel_width, panel_height, 0))
        blank(ax)
        ax.set_title(name, family='DejaVu Sans Mono')
        ax.text(0.5, -0.02, '%s, %d atoms, %d K' % (target.sequence, target.n_atoms,
                                                    int(target.temperature)),
                transform=ax.transAxes, ha='center', va='top', color=theme['muted'], fontsize=8)

    # Label the two rotatable bonds the Ramachandran plane is made of
    ax = axes[0, 0]
    pixels = project(structures['ace-a-nme-T300'], panel_width, panel_height, extent)
    middles = {label: 0.5 * (pixels[i] + pixels[j])
               for label, (i, j) in [(r'$\varphi$', phi_bond), (r'$\psi$', psi_bond)]}
    spread = np.sign(middles[r'$\varphi$'][0] - middles[r'$\psi$'][0])
    for label, middle in middles.items():
        side = spread if label == r'$\varphi$' else -spread
        ax.annotate(label, xy=middle, xytext=(middle[0] + 45 * side, middle[1] - 80),
                    ha='center', va='bottom', fontsize=12, color=theme['ink'],
                    arrowprops={'arrowstyle': '-', 'color': theme['muted'], 'linewidth': 0.8})
    return fig


save(draw_molecules, 'peptides_molecules')

# The splits of Ace-A-Nme and the panel that compares them
test = aldp.test_samples(n=args.n_samples)
train = aldp.train_samples(n=args.n_samples)
metrics, hist = aldp.compute_metrics(train, return_hist=True)
pprint.pprint({key: round(float(value), 4) for key, value in metrics.items()})

# The test split's Ramachandran free energy, from every structure of it
background = aldp.test_samples(n=args.n_background)
background_angles = np.rad2deg(aldp.torsions(background).numpy())
test_angles = np.rad2deg(aldp.torsions(test).numpy())
train_angles = np.rad2deg(aldp.torsions(train).numpy())
background_basins = np.bincount(basin(*background_angles.T), minlength=3) / len(background)
train_basins = np.bincount(basin(*train_angles.T), minlength=3) / len(train)
print('test split basins (beta/PII, alpha-R, alpha-L)', np.round(background_basins, 3))
print('train split basins (beta/PII, alpha-R, alpha-L)', np.round(train_basins, 3))
angle_edges = np.linspace(-180.0, 180.0, 91)
rama_energy = free_energy(background_angles, angle_edges, angle_edges, periodic=True)

# Where the basin labels of a Ramachandran plane sit
BASINS = [(r'$\beta$', -175.0, 105.0), (r'$\alpha_R$', -25.0, -45.0), (r'$\alpha_L$', 92.0, 30.0)]


def halo(theme):
    """Returns the outline that keeps a label or a marker readable on a filled density"""
    return [withStroke(linewidth=2.5, foreground=theme['surface'])]


def rama_axes(ax, equal=True):
    """Sets up a Ramachandran plane in degrees"""
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    if equal:
        ax.set_aspect('equal')


# The trajectory: every stride-th frame of the one 100 ps train trajectory, superposed on the
# backbone heavy atoms of the first and drawn in that frame's principal axes
frames = 10.0 * aldp.positions(train[args.stride - 1::args.stride]).numpy()
times = 0.05 * (np.arange(len(train))[args.stride - 1::args.stride] + 1)
trajectory = mdtraj.Trajectory(frames / 10.0, topologies['ace-a-nme-T300'])
trajectory.superpose(trajectory, 0, atom_indices=topologies['ace-a-nme-T300'].select('backbone'))
frames = 10.0 * trajectory.xyz
centre, rotation = principal_frame(frames[0])
frames = (frames - centre) @ rotation
frame_angles = np.rad2deg(aldp.torsions(torch.as_tensor(frames.reshape(len(frames), -1) / 10.0))
                          .numpy())
gif_width, gif_height = 520, 470
gif_extent = max(max(np.ptp(x[:, 1]) + 2 * margin,
                     (np.ptp(x[:, 0]) + 2 * margin) * gif_height / gif_width) for x in frames)
frame_renders = [render(topologies['ace-a-nme-T300'], x, gif_width, gif_height, gif_extent,
                        ball_and_stick([phi_bond, psi_bond])) for x in frames]
print('trajectory: %d frames over %.1f ps' % (len(frames), times[-1]))


def draw_trajectory_frame(theme, index):
    """Draws one frame: the train structure at that time, and its angles over the test split"""
    fig = plt.figure(figsize=(5.4, 2.75), layout='none')

    # The structure, rendered once and laid on either surface
    ax = fig.add_axes([0.0, 0.0, 0.5, 0.9])
    ax.imshow(frame_renders[index], extent=(0, gif_width, gif_height, 0))
    blank(ax)
    fig.text(0.02, 0.95, 'Ace-A-Nme, train split', color=theme['ink'], fontsize=9, va='top')
    fig.text(0.02, 0.84, 't = %5.1f ps' % times[index], color=theme['muted'], fontsize=8.5,
             va='top', family='DejaVu Sans Mono')

    # The angles, with a trail of the last 12 ps that fades and breaks where an angle wraps
    ax = fig.add_axes([0.6, 0.17, 0.34, 0.68])
    draw_free_energy(ax, rama_energy, angle_edges, angle_edges, theme, depth=0.6)
    rama_axes(ax)
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])
    ax.set_xlabel(r'$\varphi$ (deg)')
    ax.set_ylabel(r'$\psi$ (deg)')
    start = max(index - 24, 0)
    for k in range(start, index):
        step = np.abs(frame_angles[k + 1] - frame_angles[k])
        if (step < 180).all():
            alpha = 0.15 + 0.85 * (k - start + 1) / max(index - start, 1)
            ax.plot(frame_angles[k:k + 2, 0], frame_angles[k:k + 2, 1], color=theme['train'],
                    linewidth=1.4, alpha=alpha, solid_capstyle='round')
    ax.scatter(*frame_angles[index], s=34, color=theme['train'], edgecolors=theme['surface'],
               linewidths=1.0, zorder=3)
    for label, phi, psi in BASINS:
        ax.text(phi, psi, label, color=theme['ink'], fontsize=9, ha='left', va='center',
                path_effects=halo(theme))
    ax.set_title('test split free energy', fontsize=9)
    return fig


save_gif(draw_trajectory_frame, len(frames), 'peptides_trajectory', fps=12)

# The 32 x 32 grid js_ramachandran is computed on, from the same rows the panel scored
joint_edges = torch.linspace(-math.pi, math.pi, 33, dtype=torch.float64)
pair = aldp.ramachandran_pairs[0]
ref_joint = histogram_2d(aldp.torsions(test).double()[:, pair], joint_edges).numpy()
joint = histogram_2d(aldp.torsions(train).double()[:, pair], joint_edges).numpy()
print('occupancy recomputed %.4f' % (ref_joint > 0).mean())
marginal_edges = np.linspace(-180.0, 180.0, 65)


def draw_ramachandran(theme):
    """Draws the test split's Ramachandran free energy with both splits' torsion marginals, and
    the 32 x 32 cells js_ramachandran compares the two splits on"""
    fig, axes = plt.subplot_mosaic([['phi', 'numbers', '.'], ['centre', 'psi', 'grid']],
                                   figsize=(8.0, 4.7), width_ratios=[4.0, 1.0, 4.2],
                                   height_ratios=[1.0, 4.0])

    # The free energy of the whole test split, and the two marginals the KL and W1 compare
    ax = axes['centre']
    draw_free_energy(ax, rama_energy, angle_edges, angle_edges, theme)
    rama_axes(ax, equal=False)
    ax.set_xlabel(r'$\varphi$ (deg)')
    ax.set_ylabel(r'$\psi$ (deg)')
    for label, phi, psi in BASINS:
        ax.text(phi, psi, label, color=theme['ink'], fontsize=9, va='center',
                path_effects=halo(theme))
    free_energy_bar(fig, ax.inset_axes([0.55, 0.1, 0.4, 0.035]), theme)
    marginal(axes['phi'], marginal_edges, background_angles[:, 0], test_angles[:, 0],
             train_angles[:, 0], theme)
    axes['phi'].set_xlim(-180, 180)
    marginal(axes['psi'], marginal_edges, background_angles[:, 1], test_angles[:, 1],
             train_angles[:, 1], theme, horizontal=True)
    axes['psi'].set_ylim(-180, 180)
    ax = axes['numbers']
    blank(ax)
    note(ax, ['kl_phi1 %5.2f' % metrics['kl_phi1'], 'w1_phi1 %5.2f' % metrics['w1_phi1'],
              'kl_psi1 %5.2f' % metrics['kl_psi1'], 'w1_psi1 %5.2f' % metrics['w1_psi1']],
         theme, loc='lower left', size=6.5)

    # The 32 x 32 cells: the ones the test rows fill, and the train split's mass as dots
    ax = axes['grid']
    width = 360.0 / 32
    occupied = np.where(ref_joint > 0, 0.45, 0.0)
    ax.imshow(occupied.T, origin='lower', extent=(-180, 180, -180, 180), cmap=density_map(theme),
              vmin=0.0, vmax=1.0)
    centres = -180.0 + width * (np.arange(32) + 0.5)
    rows, cols = np.nonzero(joint)
    ax.scatter(centres[rows], centres[cols], s=3.0 + 60.0 * joint[rows, cols] / joint.max(),
               color=theme['train'], linewidths=0)
    for line in np.linspace(-180, 180, 33):
        ax.axvline(line, color=theme['grid'], linewidth=0.3)
        ax.axhline(line, color=theme['grid'], linewidth=0.3)
    rama_axes(ax)
    ax.set_xlabel(r'$\varphi$ (deg)')
    ax.set_title('the 32 x 32 cells scored', fontsize=9)
    note(ax, ['js_ramachandran        %.2f' % metrics['js_ramachandran'],
              'ramachandran_occupancy %.2f' % metrics['ramachandran_occupancy']], theme,
         loc='lower right', size=6.5)
    legend(fig, dict(theme, fill=density_map(theme)(0.45)),
           [('whole test split', 'fill', 'patch'),
                        ('test split, %d rows' % len(test), 'test', 'line'),
                        ('train split, %d rows' % len(train), 'train', 'line')])
    return fig


save(draw_ramachandran, 'peptides_ramachandran')

# The chirality centre seen down its bond to the carbonyl carbon, which points away from the
# viewer, so its other three bonds fan out around it; the mirror image reflects x -> -x
centre_atom, *partners = aldp.chirality_centres[0].tolist()
carbonyl = psi_atoms[2]
x = 10.0 * aldp.positions(test[:1]).numpy()[0]
away = x[centre_atom] - x[carbonyl]
away /= np.linalg.norm(away)
up = x[partners[0]] - x[centre_atom]
up -= (up @ away) * away
up /= np.linalg.norm(up)
frame_ = np.stack([np.cross(up, away), up, away], axis=-1)
left = (x - x[centre_atom]) @ frame_
right = left * np.array([-1.0, 1.0, 1.0])
pair_positions = torch.as_tensor(np.stack([left, right]).reshape(2, -1) / 10.0,
                                 dtype=torch.float32)
pair_energy = aldp.energy(pair_positions).numpy()
signs = aldp.chirality_signs(pair_positions).numpy()[:, 0]
test_signs = aldp.chirality_signs(test).numpy()[:, 0]
print('mirror pair energies (kT)', pair_energy, 'signs', signs,
      'test split sign -1 fraction %.4f' % (test_signs < 0).mean())
chiral_extent = 0.55 * extent
chiral_bonds = [(centre_atom, partner) for partner in partners]


def chirality_setup(cmd):
    """Draws the centre and its three bonds in full and fades every other atom"""
    ball_and_stick(chiral_bonds, centre=centre_atom)(cmd)
    others = 'mol and not id ' + '+'.join(str(i + 1) for i in [centre_atom] + partners)
    cmd.set('sphere_transparency', 0.6, others)
    cmd.set('stick_transparency', 0.6, others)


chiral_renders = [render(topologies['ace-a-nme-T300'], positions, panel_width, panel_height,
                         chiral_extent, chirality_setup) for positions in [left, right]]
atom_names = {partners[0]: 'N', partners[1]: 'H' + r'$\alpha$', partners[2]: r'C$\beta$'}


def draw_chirality(theme):
    """Draws a test structure and its mirror image, seen down the chirality centre's C-C bond"""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for ax, positions, image, sign, energy, label in zip(axes, [left, right], chiral_renders,
                                                         signs, pair_energy,
                                                         ['test structure', 'mirror image']):
        ax.imshow(image, extent=(0, panel_width, panel_height, 0))
        blank(ax)
        ax.set_title(label)
        pixels = project(positions, panel_width, panel_height, chiral_extent)
        middle = pixels[centre_atom]

        # The three bonded atoms, labelled outside the turn N -> H alpha -> C beta drawn around
        # the centre
        radius = 1.3 * max(np.linalg.norm(pixels[atom] - middle) for atom in partners)
        angles = []
        for atom in partners:
            direction = pixels[atom] - middle
            direction /= np.linalg.norm(direction)
            ax.text(*(middle + (radius + 34.0) * direction), atom_names[atom],
                    color=theme['ink'], fontsize=9, ha='center', va='center')
            angles.append(math.atan2(-direction[1], direction[0]))
        a, b, c = angles
        counter = math.sin(b - a) + math.sin(c - b) > 0
        sweep = a + (1.0 if counter else -1.0) * np.linspace(0.2, 4.0 * math.pi / 3.0 - 0.2, 80)
        arc = np.stack([middle[0] + radius * np.cos(sweep), middle[1] - radius * np.sin(sweep)], 1)
        ax.annotate('', xy=arc[-1], xytext=arc[-4],
                    arrowprops={'arrowstyle': '-|>', 'color': theme['ink'], 'linewidth': 1.0,
                                'mutation_scale': 9})
        ax.plot(*arc[:-3].T, color=theme['ink'], linewidth=1.0)
        ax.set_xlim(0, panel_width)
        ax.set_ylim(panel_height, 0)
        ax.text(0.5, -0.03, '%s, triple product %+d\nenergy %.2f kT'
                % ('counterclockwise' if counter else 'clockwise', sign, energy),
                transform=ax.transAxes, ha='center', va='top', color=theme['muted'], fontsize=8,
                family='DejaVu Sans Mono', linespacing=1.4)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.12, 0.88], color=theme['muted'], linewidth=1.0))
    return fig


save(draw_chirality, 'peptides_chirality')

# The TICA plane, and the k-means cells exactly as the panel fits them: the transport rows drawn
# by the panel's own subsample, then k-means on the test split's
model = aldp.tica_model()
generator = torch.Generator().manual_seed(0)
n = min(len(train), len(test), 2000)
cut = subsample(train, n, generator=generator)
ref_cut = subsample(test, n, generator=generator)
centres = kmeans(model(aldp.tica_features(ref_cut)).detach(), n_clusters=20).numpy()
background_tics = model(aldp.tica_features(background)).detach().numpy()
test_tics = model(aldp.tica_features(test)).detach().numpy()
train_tics = model(aldp.tica_features(train)).detach().numpy()
tic_lower = np.quantile(background_tics, 0.0005, axis=0)
tic_upper = np.quantile(background_tics, 0.9995, axis=0)
pad = 0.06 * (tic_upper - tic_lower)
tic_lower, tic_upper = tic_lower - pad, tic_upper + pad
x_edges = np.linspace(tic_lower[0], tic_upper[0], 121)
y_edges = np.linspace(tic_lower[1], tic_upper[1], 121)
tic_energy = free_energy(background_tics, x_edges, y_edges, periodic=False)
print('tica ranges', tic_lower, tic_upper, 'spread ratio %.0f'
      % (background_tics[:, 0].std() / background_tics[:, 1].std()))

# The cells: which centre is closest at every point of a fine grid, in the panel's coordinates.
# They split the plane along TIC 1 when every column of the grid falls in one cell
grid_x = np.linspace(tic_lower[0], tic_upper[0], 1600)
grid_y = np.linspace(tic_lower[1], tic_upper[1], 200)
points = np.stack(np.meshgrid(grid_x, grid_y, indexing='xy'), axis=-1)
cell = ((points[:, :, None, :] - centres[None, None]) ** 2).sum(-1).argmin(-1)
print('grid columns lying in a single cell: %.4f' % (cell == cell[:1]).all(0).mean())
borders = grid_x[1:][cell[len(grid_y) // 2, 1:] != cell[len(grid_y) // 2, :-1]]

# One test structure per basin, the one closest to that basin's median TICA coordinates
background_labels = basin(*background_angles.T)
insets = []
for label, (symbol, _, _) in enumerate(BASINS):
    members = np.nonzero(background_labels == label)[0]
    middle = np.median(background_tics[members], axis=0)
    scale = tic_upper - tic_lower
    pick = members[np.argmin((((background_tics[members] - middle) / scale) ** 2).sum(-1))]
    x = 10.0 * aldp.positions(background[pick:pick + 1]).numpy()[0]
    x_centre, x_rotation = principal_frame(x)
    image = render(topologies['ace-a-nme-T300'], (x - x_centre) @ x_rotation, 360, 270,
                   0.62 * extent, ball_and_stick())
    insets.append({'symbol': symbol, 'tic': background_tics[pick], 'image': image})
    print('inset', symbol, 'at', np.round(background_tics[pick], 4))


def draw_tica(theme):
    """Draws the test split's TICA free energy with both splits' marginals, the panel's k-means
    cells, and one structure per backbone basin"""
    fig, axes = plt.subplot_mosaic([['tic1', 'numbers'], ['centre', 'tic2']], figsize=(8.0, 4.8),
                                   width_ratios=[5.0, 1.0], height_ratios=[1.0, 4.0])
    ax = axes['centre']
    draw_free_energy(ax, tic_energy, x_edges, y_edges, theme)
    ax.scatter(*centres.T, s=14, marker='+', color=theme['ink'], linewidths=0.8,
               path_effects=halo(theme))
    for inset in insets:
        ax.scatter(*inset['tic'], s=22, facecolors='none', edgecolors=theme['ink'], linewidths=0.9,
                   path_effects=halo(theme))
        ax.annotate(inset['symbol'], xy=inset['tic'], xytext=(5, 7), textcoords='offset points',
                    color=theme['ink'], fontsize=9, va='center', path_effects=halo(theme))
    ax.set_xlim(tic_lower[0], tic_upper[0])
    ax.set_ylim(tic_lower[1], tic_upper[1])
    ax.set_xlabel('TIC 1')
    ax.set_ylabel('TIC 2')
    free_energy_bar(fig, ax.inset_axes([0.27, 0.08, 0.26, 0.035]), theme)

    # One structure per basin, in the empty band between the alpha-L island and the rest
    for k, inset in enumerate(sorted(insets, key=lambda inset: inset['tic'][0])):
        box = ax.inset_axes([0.24 + 0.155 * k, 0.6, 0.15, 0.36])
        box.imshow(inset['image'])
        blank(box)
        box.text(0.5, 0.0, inset['symbol'], transform=box.transAxes, color=theme['ink'],
                 fontsize=9, ha='center', va='top')

    # The marginals, with the cell borders on TIC 1, the one coordinate the cells split
    marginal(axes['tic1'], x_edges[::2], background_tics[:, 0], test_tics[:, 0], train_tics[:, 0],
             theme)
    axes['tic1'].set_xlim(tic_lower[0], tic_upper[0])
    for border in borders:
        axes['tic1'].axvline(border, color=theme['muted'], linewidth=0.5)
    marginal(axes['tic2'], y_edges[::2], background_tics[:, 1], test_tics[:, 1], train_tics[:, 1],
             theme, horizontal=True)
    axes['tic2'].set_ylim(tic_lower[1], tic_upper[1])
    ax = axes['numbers']
    blank(ax)
    note(ax, ['tica_w2    %.2f' % metrics['tica_w2'], 'tica_k_jsd %.3f' % metrics['tica_k_jsd']],
         theme, loc='lower left', size=6.5)
    legend(fig, dict(theme, fill=density_map(theme)(0.45)),
           [('whole test split', 'fill', 'patch'),
                        ('test split, %d rows' % len(test), 'test', 'line'),
                        ('train split, %d rows' % len(train), 'train', 'line'),
                        ('k-means cell border', 'muted', 'line')])
    return fig


save(draw_tica, 'peptides_tica')
