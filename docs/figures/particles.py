# Draw the figures of the particle systems' page

# Libraries
import argparse, pprint, sys, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from samplingdb import build
from samplingdb.targets.base import generator
from samplingdb.targets.kernels import random_walk_mh
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import blank, legend, pymol_render, save, save_gif, start_pymol

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=2048)
parser.add_argument('--n_chain_steps', type=int, default=100000)
parser.add_argument('--n_frames', type=int, default=100)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# A single chain is thousands of one-configuration calls, which threads only slow down
torch.set_num_threads(1)
start_pymol()

# Neighbours are counted out to the first minimum of the Lennard-Jones pair distribution, at 1.4 r
NEIGHBOUR_CUTOFF = 1.4

# Coordination numbers are drawn on one violet ramp, from an isolated particle to a full shell of
# 12; the renders are made once and shown on both surfaces, so the ramp is the same in both
COORDINATION = LinearSegmentedColormap.from_list('coordination', ['#d9d3f7', '#3b2d99'])
MAX_NEIGHBOURS = 12

# Angstrom per Lennard-Jones length in the renders, which with the scale below makes two spheres
# at the pair-minimum distance r just touch
ANGSTROM = 3.0
SPHERE_SCALE = 0.5 * ANGSTROM / 1.88


def coordination(positions):
    """Counts each particle's neighbours closer than NEIGHBOUR_CUTOFF

    Args:
        * positions (numpy.ndarray of shape (n_particles, 3)): One configuration

    Returns:
        * counts (numpy.ndarray of shape (n_particles,)): Neighbours per particle
    """
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    return (distances < NEIGHBOUR_CUTOFF).sum(-1) - 1


def cluster_pdb(positions):
    """Writes a cluster as a PDB string of argon atoms, in Angstrom"""
    lines = []
    for index, position in enumerate(ANGSTROM * positions):
        lines.append('HETATM%5d AR   LJ  A   1    %8.3f%8.3f%8.3f  1.00  0.00          AR\n'
                     % (index + 1, *position))
    return ''.join(lines)


def render_cluster(positions, size, view=None):
    """Ray-traces a cluster as spheres coloured by coordination number

    Args:
        * positions (numpy.ndarray of shape (n_particles, 3)): One configuration
        * size (int): Width and height of the image, in pixels
        * view (tuple or None): A PyMOL view to reuse (default is None)

    Returns:
        * image (numpy.ndarray of shape (size, size, 4)): RGBA image
        * view (tuple): The view used
    """
    counts = coordination(positions)

    def setup(cmd):
        cmd.hide('everything')
        cmd.show('spheres')
        cmd.set('sphere_scale', SPHERE_SCALE)
        for count in np.unique(counts):
            colour = COORDINATION(min(count, MAX_NEIGHBOURS) / MAX_NEIGHBOURS)
            cmd.set_color('cn_' + str(count), list(to_rgb(colour)))
            ids = '+'.join(str(index + 1) for index in np.nonzero(counts == count)[0])
            cmd.color('cn_' + str(count), 'mol and id ' + ids)

    return pymol_render(cluster_pdb(positions), size, size, setup, view=view, buffer=3.0,
                        return_view=True)


def kabsch(positions, reference):
    """Rotates a centred configuration onto a centred reference, particle for particle"""
    u, _, vt = np.linalg.svd(positions.T @ reference)
    sign = np.sign(np.linalg.det(u @ vt))
    return positions @ (u @ np.diag([1.0, 1.0, sign]) @ vt)


def principal_frame(positions):
    """Rotates a centred configuration onto its principal axes, for a canonical 2D view"""
    _, _, vt = np.linalg.svd(positions - positions.mean(0))
    return (positions - positions.mean(0)) @ vt.T


def outside_density(values, edges, x_window):
    """Bins values on the edges extended to a window, to show what falls outside them

    Args:
        * values (numpy.ndarray of shape (n,)): Values, all of them
        * edges (numpy.ndarray of shape (n_bins + 1,)): The panel's own edges
        * x_window (tuple of two floats): Extent of the plot

    Returns:
        * left (tuple): (density, edges) below the first edge, or None
        * right (tuple): (density, edges) above the last edge, or None
    """
    width = edges[1] - edges[0]
    sides = []
    for start, stop, step in [(edges[0], x_window[0], -width), (edges[-1], x_window[1], width)]:
        n_bins = int(np.ceil(abs(stop - start) / width))
        if n_bins == 0:
            sides.append(None)
            continue
        extra = np.sort(start + step * np.arange(n_bins + 1))
        counts, _ = np.histogram(values, bins=extra)
        sides.append((counts / len(values) / width, extra))
    return sides


# Build the three systems, their two splits and the train split's panel against the test split
systems = {}
for name in ['dw4', 'lj13', 'lj55']:
    target = build(name)
    test = target.test_samples(n=args.n_samples)
    train = target.train_samples(n=args.n_samples)
    metrics, hist = target.compute_metrics(train, ref_samples=test, return_hist=True)
    print(name, {key: round(float(value), 4) for key, value in metrics.items()})
    systems[name] = {
        'target': target, 'test': test, 'train': train, 'metrics': metrics, 'hist': hist,
        'values': {'test': {'dist': target.distances(test).reshape(-1).numpy(),
                            'energy': target.energy(test).numpy()},
                   'train': {'dist': target.distances(train).reshape(-1).numpy(),
                             'energy': target.energy(train).numpy()}},
    }

# A few DW-4 configurations of each split: the first test ones, and train states spread over the
# chain, each turned onto its principal axes
dw4 = systems['dw4']
n_shown = 6
picks = {'test': np.arange(n_shown),
         'train': np.linspace(0, args.n_samples - 1, n_shown).astype(int)}
dw4_shown = {split: [principal_frame(dw4[split][index].reshape(4, 2).numpy())
                     for index in picks[split]] for split in ['test', 'train']}

# One configuration of each split for the two clusters: the test one at the median energy, and
# the last state of the train chain. Each is oriented on its own principal axes, then all four are
# drawn from the camera distance of the widest, so they share one scale
cluster_positions, views = {}, {}
for name in ['lj13', 'lj55']:
    entry = systems[name]
    n_particles = entry['target'].n_particles
    energy = entry['values']['test']['energy']
    test_index = int(np.argsort(energy)[len(energy) // 2])
    for split, index in [('test', test_index), ('train', args.n_samples - 1)]:
        cluster_positions[name, split] = entry[split][index].reshape(n_particles, 3).numpy()
        _, views[name, split] = render_cluster(cluster_positions[name, split], 64)
distance = max(-view[11] for view in views.values())
cluster_renders = {}
for key, positions in cluster_positions.items():
    view = list(views[key])
    view[11], view[15], view[16] = -distance, distance - 100.0, distance + 100.0
    cluster_renders[key], _ = render_cluster(positions, 900, view=tuple(view))


def draw_configurations(theme):
    """Draws DW-4 configurations of both splits and one render of each cluster per split"""
    fig = plt.figure(figsize=(8.0, 5.3))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.05])

    # DW-4: four discs and their six pair distances
    axes = top.subplots(2, n_shown)
    for row, split in enumerate(['test', 'train']):
        for column, positions in enumerate(dw4_shown[split]):
            ax = axes[row, column]
            for i in range(4):
                for j in range(i + 1, 4):
                    ax.plot(positions[[i, j], 0], positions[[i, j], 1], color=theme['muted'],
                            linewidth=0.7, zorder=1)
            for position in positions:
                ax.add_patch(plt.Circle(position, 0.55, color=theme[split], zorder=2))
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.set_aspect('equal')
            blank(ax)
        axes[row, 0].set_ylabel(split + ' split', color=theme['ink'])
        axes[row, 0].yaxis.label.set_visible(True)
    top.suptitle('dw4: four particles in 2D, pair distances drawn', fontsize=10)

    # The two clusters, one render per split, with the coordination scale
    axes = bottom.subplots(1, 4)
    for ax, (name, split) in zip(axes, [('lj13', 'test'), ('lj13', 'train'),
                                        ('lj55', 'test'), ('lj55', 'train')]):
        ax.imshow(cluster_renders[name, split])
        blank(ax)
        ax.set_title(name + ', ' + split + ' split', family='DejaVu Sans Mono', fontsize=9)
    mappable = plt.cm.ScalarMappable(norm=Normalize(0, MAX_NEIGHBOURS), cmap=COORDINATION)
    bar = bottom.colorbar(mappable, ax=axes, orientation='horizontal', fraction=0.05,
                          shrink=0.4, aspect=30)
    bar.set_label('neighbours closer than 1.4 r', color=theme['ink'])
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=theme['muted'], labelcolor=theme['muted'])
    return fig


save(draw_configurations, 'particles_configurations')


def draw_histograms(theme):
    """Draws the distance and energy histograms of both splits on the panel's own edges"""
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.6), gridspec_kw={'width_ratios': [1, 1, 0.62]})
    for row, (name, entry) in enumerate(systems.items()):
        metrics = entry['metrics']
        for column, key in enumerate(['dist', 'energy']):
            ax = axes[row, column]
            hist = entry['hist'][key]
            edges = hist['edges'].numpy()
            width = edges[1] - edges[0]
            span = edges[-1] - edges[0]
            x_window = (edges[0] - 0.12 * span, edges[-1] + 0.12 * span)

            # Inside the edges, the histograms the panel compares, scaled back to the mass they
            # hold; outside them, the mass the panel drops
            ax.axvspan(x_window[0], edges[0], color=theme['grid'], alpha=0.5, linewidth=0)
            ax.axvspan(edges[-1], x_window[1], color=theme['grid'], alpha=0.5, linewidth=0)
            for split, values in [('test', hist['ref_hist']), ('train', hist['hist'])]:
                inside = values.numpy() / width
                if split == 'train':
                    inside = inside * (1.0 - metrics[key + '_out_of_range'])
                ax.stairs(inside, edges, color=theme[split], linewidth=1.1)
                for side in outside_density(entry['values'][split][key], edges, x_window):
                    if side is not None and side[0].any():
                        ax.stairs(side[0], side[1], color=theme[split], linewidth=1.1)
            ax.set_xlim(*x_window)
            peak = 3.0 * float(hist['ref_hist'].max()) / width
            if float(hist['hist'].max()) / width > peak:
                ax.set_ylim(0.0, peak)
                ax.text(0.98, 0.96, 'train peaks clipped', transform=ax.transAxes, ha='right',
                        va='top', fontsize=7, color=theme['ink'])
            else:
                ax.set_ylim(bottom=0.0)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            if row == 2:
                ax.set_xlabel('pair distance' if key == 'dist' else 'energy')
        axes[row, 0].set_ylabel(name, family='DejaVu Sans Mono')

        # The panel's numbers, train split against test split
        ax = axes[row, 2]
        blank(ax)
        rows = []
        for key in ['dist', 'energy']:
            rows += ['%-19s %6.3f' % (key + '_total_var', metrics[key + '_total_var']),
                     '%-19s %6.3f' % (key + '_out_of_range', metrics[key + '_out_of_range']),
                     '%-19s %6.3f' % (key + '_w1', metrics[key + '_w1']), '']
        ax.text(0.0, 0.5, '\n'.join(rows[:-1]), transform=ax.transAxes, fontsize=7,
                family='DejaVu Sans Mono', color=theme['ink'], va='center', linespacing=1.5)
    axes[0, 2].set_title('train against test', fontsize=9, loc='left')
    legend(fig, theme, [('test split', 'test', 'line'), ('train split', 'train', 'line'),
                        ('outside the edges, dropped by the panel', 'grid', 'patch')])
    return fig


save(draw_histograms, 'particles_histograms')

# The LJ-55 train split is one Metropolis chain from the lattice, burn-in included: rerun its
# recipe in full and keep evenly spaced states for the animation
lj55 = systems['lj55']['target']
start = lj55.initial_configuration()[None].clone()
trace = random_walk_mh(lj55.log_prob, start, n_steps=args.n_chain_steps,
                       step_size=lj55.train_step_size, generator=generator(0),
                       return_trace=True)[:, 0]
trace = lj55.preprocess(trace)
assert torch.allclose(trace[:args.n_samples], systems['lj55']['train'], atol=1e-5)
chain_energy = lj55.energy(trace).numpy()
steps = np.linspace(0, args.n_chain_steps - 1, args.n_frames).astype(int)
print('lj55 chain energy at the frames', np.round(chain_energy[steps[::10]], 1), 'max', chain_energy.max())

# Render every frame once, rotated onto the first so the cluster does not tumble, from one view
frames = []
reference = trace[0].reshape(55, 3).numpy().astype(np.float64)
view = None
for step in steps:
    positions = kabsch(trace[step].reshape(55, 3).numpy().astype(np.float64), reference)
    image, view = render_cluster(positions, 360, view=view)
    frames.append(image)
test_energy = systems['lj55']['values']['test']['energy']
energy_window = (-380.0, -215.0)
energy_bins = np.linspace(*energy_window, 67)


def draw_chain_frame(theme, index):
    """Draws one state of the LJ-55 train chain next to its energy so far"""
    index = min(index, len(frames) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.7),
                             gridspec_kw={'width_ratios': [1.0, 1.25, 0.32]})
    ax = axes[0]
    ax.imshow(frames[index])
    blank(ax)
    ax.set_title('step %d' % steps[index], family='DejaVu Sans Mono', fontsize=9)

    # The chain's energy so far, the current state marked
    ax = axes[1]
    ax.plot(np.arange(steps[index] + 1), chain_energy[:steps[index] + 1], color=theme['train'],
            linewidth=1.0)
    ax.plot(steps[index], chain_energy[steps[index]], 'o', color=theme['train'], markersize=4)
    ax.set_xlim(0, args.n_chain_steps)
    ax.set_ylim(*energy_window)
    ax.set_xticks([0, args.n_chain_steps // 2, args.n_chain_steps],
                  ['0', '%dk' % (args.n_chain_steps // 2000), '%dk' % (args.n_chain_steps // 1000)])
    ax.set_xlabel('Metropolis step')
    ax.set_ylabel('energy')
    ax.yaxis.grid(True, color=theme['grid'], linewidth=0.5)
    ax.set_title('train chain', fontsize=9)

    # The test split's energies, on the same axis
    ax = axes[2]
    ax.hist(test_energy, bins=energy_bins, orientation='horizontal', color=theme['test'],
            linewidth=0)
    ax.set_ylim(*energy_window)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_title('test split', fontsize=9)
    return fig


# The last state is held for a second before the loop restarts
save_gif(draw_chain_frame, len(frames) + 12, 'particles_train_chain', fps=12, dpi=100)
