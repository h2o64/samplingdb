# Draw the banner of the README: every target of the catalog, one tile each, grouped by tier

# Libraries
import argparse, pprint, sys, pathlib, tempfile
import numpy as np
import torch
import mdtraj
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.ndimage import gaussian_filter
from samplingdb import ManyModes, build
from samplingdb.targets.bayesian import fetch_dataset
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import density_map, pymol_render, save, start_pymol

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=20000)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# The tiles of every family on a grid of three rows: (name, label, row, column, column span)
FAMILIES = [
    ('Synthetic', 'tier 0', 4, [
        ('gmm40', 'gmm40', 0, 0, 1), ('mos', 'mos', 0, 1, 1), ('twomodes', 'twomodes', 0, 2, 1),
        ('brackettwomodes', 'brackettwomodes', 0, 3, 1),
        ('manymodes-k10', 'manymodes-k10', 1, 0, 1), ('eightgaussians', 'eightgaussians', 1, 1, 1),
        ('rings', 'rings', 1, 2, 1), ('checkerboard', 'checkerboard', 1, 3, 1),
        ('funnel', 'funnel', 2, 0, 3), ('manywell', 'manywell', 2, 3, 1),
    ]),
    ('Particles', 'tier 1', 1, [
        ('dw4', 'dw4', 0, 0, 1), ('lj13', 'lj13', 1, 0, 1), ('lj55', 'lj55', 2, 0, 1),
    ]),
    ('Posteriors', 'tier 2', 3, [
        ('ionosphere', 'ionosphere', 0, 0, 1), ('sonar', 'sonar', 0, 1, 1),
        ('german-credit', 'german-credit', 0, 2, 1), ('breast-cancer', 'breast-cancer', 1, 0, 1),
        ('seeds', 'seeds', 1, 1, 1), ('lgcp', 'lgcp', 1, 2, 1),
        ('brownian', 'brownian', 2, 0, 2), ('lgcp-whitened', 'lgcp-whitened', 2, 2, 1),
    ]),
    ('Peptides', 'tier 3', 2, [
        ('ace-a-nme-T300', 'ace-a-nme-T300', 0, 0, 1), ('aaa-T310', 'aaa-T310', 0, 1, 1),
        ('ace-aaa-nme-T300', 'ace-aaa-nme-T300', 1, 0, 2),
        ('aaaaaa-T310', 'aaaaaa-T310', 2, 0, 2),
    ]),
    ('Glasses', 'tier 4', 1, [
        ('ka-ternary-N44-T1.0', 'ternary, T = 1.0', 0, 0, 1),
        ('ka-ternary-N44-T0.32', 'ternary, T = 0.32', 1, 0, 1),
        ('ka-binary-N64-T0.2', 'binary, T = 0.2', 2, 0, 1),
    ]),
]

# Geometry of the banner, in inches
TILE, GAP, FAMILY_GAP, LABEL, ROW_GAP, HEADER, MARGIN, INSET = 1.0, 0.1, 0.32, 0.2, 0.08, 0.5, 0.1, 0.05
DPI = 220

# The card behind every tile, one step off GitHub's page surface
CARDS = {'light': '#f6f8fa', 'dark': '#161b22'}

# Species colours of the glass page, so a glass looks the same here and there
SPECIES = {'light': ['#6250d6', '#1baf7a', '#eda100'], 'dark': ['#9085e9', '#199e70', '#c98500']}


def tile_size(span):
    """Returns the width of a tile spanning some columns, in inches"""
    return span * TILE + (span - 1) * GAP


def card_map(theme, depth=1.0):
    """Returns the density colormap with the card colour at zero instead of the page surface"""
    return density_map(dict(theme, surface=CARDS[theme['name']]), depth)


def levels(density):
    """Scales a density to [0, 1] for a ten-level map, each level a tenth of the peak

    NOTE: the peak is the 99.9% quantile, so Rings' integrable spike at the origin does not set it.
    """
    return np.clip(density / np.quantile(density, 0.999), 0.0, 1.0)


def grid_density(log_prob, x_window, y_window, n_x=300, n_y=300):
    """Evaluates an unnormalised 2D density on a grid, rows along y, from its log"""
    xs = torch.linspace(*x_window, n_x, dtype=torch.float64)
    ys = torch.linspace(*y_window, n_y, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1).reshape(-1, 2)
    values = log_prob(grid).reshape(n_y, n_x).numpy()
    return np.exp(values - values[np.isfinite(values)].max())


def sample_density(x, y, x_window, y_window, n_bins=120, smooth=2.0):
    """Estimates a 2D density from samples by a smoothed histogram, rows along y"""
    hist, _, _ = np.histogram2d(y, x, bins=n_bins, range=[y_window, x_window])
    return gaussian_filter(hist, smooth)


def pdb_string(topology, positions):
    """Writes one structure, positions in nanometres, as the text of a PDB file"""
    with tempfile.TemporaryDirectory() as folder:
        path = folder + '/frame.pdb'
        mdtraj.Trajectory(positions[None], topology).save_pdb(path)
        return open(path).read()


def cluster_pdb(positions):
    """Writes a cluster of identical particles, positions in Lennard-Jones units, as a PDB file

    One Lennard-Jones unit is drawn as 3.4 Angstrom, argon's sigma, so PyMOL's argon radius fits.
    """
    lines = ['HETATM%5d AR   LJ  A   1    %8.3f%8.3f%8.3f  1.00%6.2f          AR\n'
             % (index + 1, *(3.4 * position), 0.0) for index, position in enumerate(positions)]
    return ''.join(lines)


def fitted_render(pdb, width, height, setup, fill=0.9):
    """Ray-traces a structure so its longest axis spans a fraction of the tile's width

    PyMOL's orient puts the longest axis along x and its zoom fits the bounding sphere to the
    height, which leaves a wide tile mostly empty; the camera is moved in by the tile's aspect.

    Args:
        * pdb (str): Contents of a PDB file
        * width (int): Width of the image, in pixels
        * height (int): Height of the image, in pixels
        * setup (function): cmd -> None, applies the representation
        * fill (float): Fraction of the width the longest axis spans, read against PyMOL's
            bounding sphere, which is larger than the structure (default is 0.9)

    Returns:
        * image (numpy.ndarray of shape (height, width, 4)): RGBA image, values in [0, 1]
    """
    _, view = pymol_render(pdb, width // 4, height // 4, setup, buffer=0.0, return_view=True)
    view = list(view)
    factor = min(1.0, 1.0 / (fill * width / height)) if width > height else 1.0 / fill
    shift = -view[11] * (1.0 - factor)
    view[11] *= factor
    view[15], view[16] = max(view[15] - shift, 1.0), view[16] - shift
    return pymol_render(pdb, width, height, setup, view=tuple(view))


def make(name):
    """Builds a target by its name, ManyModes at any number of modes included"""
    if name.startswith('manymodes-k'):
        return ManyModes(n_modes=int(name[len('manymodes-k'):]))
    return build(name)


# Every 2D synthetic density, on the windows of the synthetic page, ManyModes with ten modes
tiles = {}
windows = {'gmm40': (-45.0, 45.0), 'mos': (-16.0, 16.0), 'twomodes': (-2.0, 2.0),
           'brackettwomodes': (-2.2, 2.2), 'manymodes-k10': (-11.0, 11.0),
           'eightgaussians': (-3.0, 3.0), 'rings': (-6.5, 6.5), 'checkerboard': (-4.5, 4.5)}
for name, window in windows.items():
    tiles[name] = {'kind': 'image', 'image': levels(grid_density(make(name).log_prob, window,
                                                                 window))}

# The funnel's first two coordinates, and ManyWell's first quartic coordinates of two wells, whose
# product shows four of the 2^16 modes with their unequal weights
funnel = build('funnel')
tiles['funnel'] = {'kind': 'image', 'image': levels(grid_density(
    lambda x: -0.5 * (x[:, 0] / 3.0) ** 2 - 0.5 * x[:, 0] - 0.5 * x[:, 1] ** 2 * torch.exp(-x[:, 0]),
    (-9.0, 9.0), (-12.0, 12.0), n_x=600, n_y=200))}
manywell = build('manywell')
tiles['manywell'] = {'kind': 'image', 'image': levels(grid_density(
    lambda x: manywell.log_prob_1d(x[:, 0]) + manywell.log_prob_1d(x[:, 1]),
    (-2.6, 2.6), (-2.6, 2.6)))}

# The particle systems: one reference configuration each
dw4 = build('dw4').test_samples(n=1)[0].reshape(4, 2).numpy()
tiles['dw4'] = {'kind': 'dw4', 'positions': dw4}
for name, n_particles in [('lj13', 13), ('lj55', 55)]:
    positions = build(name).test_samples(n=1)[0].reshape(n_particles, 3).numpy()
    tiles[name] = {'kind': 'cluster', 'positions': positions - positions.mean(0)}

# The logistic regressions: the pair of coefficients the posterior correlates most
for name in ['ionosphere', 'sonar', 'german-credit', 'breast-cancer']:
    x = build(name).test_samples(n=args.n_samples).numpy()
    correlation = np.abs(np.corrcoef(x.T))
    np.fill_diagonal(correlation, 0.0)
    i, j = np.unravel_index(np.argmax(correlation), correlation.shape)
    u, v = (x[:, i] - x[:, i].mean()) / x[:, i].std(), (x[:, j] - x[:, j].mean()) / x[:, j].std()
    tiles[name] = {'kind': 'image', 'image': levels(sample_density(u, v, (-4.0, 4.0),
                                                                  (-4.0, 4.0)))}

# Seeds: the hierarchical funnel between log tau and one random effect
seeds = build('seeds').test_samples(n=args.n_samples).numpy()
tiles['seeds'] = {'kind': 'image', 'image': levels(sample_density(
    seeds[:, -1], seeds[:, 4], (-1.0, 7.0), (-1.6, 1.0)))}

# Brownian motion: the posterior bands of the path and the twenty observations
brownian = build('brownian')
paths = brownian.test_samples(n=args.n_samples)[:, 2:].numpy()
tiles['brownian'] = {'kind': 'bands', 'quantiles': np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95],
                                                               axis=0),
                     'observed': brownian.observed.numpy(), 'y': brownian.y.numpy()}

# LGCP: the posterior mean intensity with the pines, and one draw in whitened coordinates
lgcp = build('lgcp')
field = lgcp.test_samples(n=2000)
intensity = torch.exp(field).mean(0).reshape(40, 40).numpy()
pines = np.loadtxt(fetch_dataset('pines.csv'), delimiter=',', skiprows=1, usecols=(1, 2))
tiles['lgcp'] = {'kind': 'field', 'image': levels(intensity), 'points': pines}
whitened = build('lgcp-whitened').test_samples(n=1)[0].reshape(40, 40).numpy()
tiles['lgcp-whitened'] = {'kind': 'field', 'image': np.clip((whitened + 3.0) / 6.0, 0.0, 1.0),
                          'points': None}

# The peptides: the most extended of the first 2000 reference structures, so the chain shows
for name, _, _, _, span in FAMILIES[3][3]:
    target = build(name, compile=False)
    x = target.test_samples(n=2000).reshape(2000, -1, 3)
    ends = torch.linalg.norm(x[:, 0] - x[:, -1], dim=-1)
    index = int(torch.argsort(ends)[int(0.9 * len(ends))])
    tiles[name] = {'kind': 'molecule', 'topology': mdtraj.Topology.from_openmm(target.topology),
                   'positions': x[index].numpy()}

# The glasses: one reference configuration in its periodic box
for name in ['ka-ternary-N44-T1.0', 'ka-ternary-N44-T0.32', 'ka-binary-N64-T0.2']:
    target = build(name)
    positions = target.test_samples(n=1)[0].reshape(-1, 2).numpy() % target.box
    tiles[name] = {'kind': 'glass', 'positions': positions, 'box': target.box,
                   'species': target.species.numpy(),
                   'diameters': target.sigma.diagonal()[target.species].numpy()}

# Render every molecule and cluster once, on a transparent background, at the tile's resolution
start_pymol()
for family, _, _, entries in FAMILIES:
    for name, _, _, _, span in entries:
        tile = tiles[name]
        width, height = int(2 * DPI * (tile_size(span) - 2 * INSET)), int(2 * DPI * (TILE - 2 * INSET))
        if tile['kind'] == 'cluster':
            radius = np.linalg.norm(tile['positions'], axis=-1)
            radius = (radius - radius.min()) / (radius.max() - radius.min())

            def setup(cmd, radius=radius):
                cmd.hide('everything')
                cmd.show('spheres')
                cmd.set('sphere_scale', 0.85)
                gnbu = plt.get_cmap('GnBu')
                for index, value in enumerate(radius):
                    cmd.set_color('shell%d' % index, list(gnbu(0.95 - 0.5 * value)[:3]))
                    cmd.color('shell%d' % index, 'mol and id %d' % (index + 1))

            tile['render'] = fitted_render(cluster_pdb(tile['positions']), width, height, setup,
                                           fill=0.88)
        elif tile['kind'] == 'molecule':

            def setup(cmd):
                cmd.hide('everything')
                cmd.show('sticks')
                cmd.show('spheres')
                cmd.set('stick_radius', 0.15)
                cmd.set('sphere_scale', 0.25)
                cmd.set('sphere_scale', 0.18, 'mol and elem H')

            tile['render'] = fitted_render(pdb_string(tile['topology'], tile['positions']), width,
                                           height, setup, fill=0.9 if span > 1 else 0.95)
        print('prepared', name)


def draw_tile(ax, tile, theme):
    """Draws one tile's content into an axis whose limits are [0, 1] x [0, 1]"""
    cmap = card_map(theme)
    if tile['kind'] == 'image':
        ax.imshow(tile['image'], origin='lower', extent=(0, 1, 0, 1), aspect='auto', cmap=cmap,
                  vmin=0.0, vmax=1.0)
    elif tile['kind'] == 'field':
        ax.imshow(tile['image'].T, origin='lower', extent=(0, 1, 0, 1), aspect='auto', cmap=cmap,
                  vmin=0.0, vmax=1.0, interpolation='nearest')
        if tile['points'] is not None:
            ax.scatter(tile['points'][:, 0], tile['points'][:, 1], s=1.2, color=theme['ink'],
                       linewidths=0)
    elif tile['kind'] in ['cluster', 'molecule']:
        ax.imshow(tile['render'], extent=(0, 1, 0, 1), aspect='auto', interpolation='antialiased')
    elif tile['kind'] == 'dw4':
        positions = tile['positions'] - tile['positions'].mean(0)
        scale = 0.38 / np.abs(positions).max()
        positions = 0.5 + scale * positions
        gnbu = plt.get_cmap('GnBu')
        for i in range(4):
            for j in range(i + 1, 4):
                ax.plot(*positions[[i, j]].T, color=theme['muted'], linewidth=0.6, zorder=1)
        for position in positions:
            ax.add_patch(Circle(position, 0.075, color=gnbu(0.8), zorder=2))
    elif tile['kind'] == 'bands':
        q = tile['quantiles']
        steps = np.linspace(0.06, 0.94, q.shape[1])
        low, high = q.min(), q.max()
        scale = lambda values: 0.1 + 0.8 * (values - low) / (high - low)
        gnbu = plt.get_cmap('GnBu')
        ax.axvspan(steps[10] - 0.015, steps[19] + 0.015, color=theme['grid'], alpha=0.35,
                   linewidth=0)
        ax.fill_between(steps, scale(q[0]), scale(q[4]), color=gnbu(0.45), linewidth=0)
        ax.fill_between(steps, scale(q[1]), scale(q[3]), color=gnbu(0.7), linewidth=0)
        ax.plot(steps, scale(q[2]), color=gnbu(0.95), linewidth=1.0)
        ax.scatter(steps[tile['observed']], scale(tile['y'][tile['observed']]), s=4,
                   color=theme['ink'], linewidths=0, zorder=3)
    elif tile['kind'] == 'glass':
        box, palette = tile['box'], SPECIES[theme['name']]
        order = np.argsort(-tile['diameters'])
        for index in order:
            for dx in [-box, 0.0, box]:
                for dy in [-box, 0.0, box]:
                    centre = (tile['positions'][index] + [dx, dy]) / box
                    ax.add_patch(Circle(centre, 0.5 * tile['diameters'][index] / box,
                                        facecolor=palette[tile['species'][index]],
                                        edgecolor=CARDS[theme['name']], linewidth=0.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_banner(theme):
    """Draws every target of the catalog as a tile on a card, grouped by family"""
    columns = sum(family[2] for family in FAMILIES)
    width = 2 * MARGIN + 0.15 + columns * TILE + (columns - len(FAMILIES)) * GAP \
        + (len(FAMILIES) - 1) * FAMILY_GAP
    height = 2 * MARGIN + HEADER + 3 * (TILE + LABEL) + 2 * ROW_GAP
    fig = plt.figure(figsize=(width, height), layout='none')
    left = MARGIN
    for family, tier, n_columns, entries in FAMILIES:

        # The family's name and tier, above its block
        fig.text(left / width, 1.0 - (MARGIN + 0.2) / height, family, fontsize=12,
                 fontweight='bold', color=theme['ink'], va='baseline')
        fig.text(left / width, 1.0 - (MARGIN + 0.4) / height, tier, fontsize=9,
                 color=theme['muted'], va='baseline')

        # Every tile: a rounded card, its content inset, and its name under it
        for name, label, row, column, span in entries:
            x = left + column * (TILE + GAP)
            y = height - MARGIN - HEADER - (row + 1) * TILE - row * (LABEL + ROW_GAP)
            w = tile_size(span)
            fig.patches.append(FancyBboxPatch(
                (x / width, y / height), w / width, TILE / height,
                boxstyle='round,pad=0,rounding_size=%f' % (0.06 / width),
                transform=fig.transFigure, facecolor=CARDS[theme['name']], linewidth=0,
                mutation_aspect=width / height, zorder=-1))
            ax = fig.add_axes(((x + INSET) / width, (y + INSET) / height,
                               (w - 2 * INSET) / width, (TILE - 2 * INSET) / height))
            draw_tile(ax, tiles[name], theme)
            fig.text((x + 0.5 * w) / width, (y - 0.04) / height, label, fontsize=8.5,
                     color=theme['muted'], ha='center', va='top')
        left += n_columns * TILE + (n_columns - 1) * GAP + FAMILY_GAP
    return fig


save(draw_banner, 'banner', colours=None)
