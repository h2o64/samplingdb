# Draw the figures of the particle systems' page

# Libraries
import argparse, pprint, sys, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from samplingdb import build
from samplingdb.targets.base import generator
from samplingdb.targets.kernels import temperature_ladder
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import blank, legend, pymol_render, save, save_gif, start_pymol

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=None)
parser.add_argument('--chains', type=int, nargs='+', default=[0, 500, 999])
parser.add_argument('--trace_every', type=int, default=10)
parser.add_argument('--frame_every', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)
start_pymol()

# Chains, adaptation moves and thinning of the train recipe, whose split is laid out frame by frame
N_CHAINS, N_ADAPT, THIN = 1000, 1000, 10

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
        * edges (numpy.ndarray of shape (n_bins + 1,)): Edges spanning the panel's range
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
    target = build(name).to(args.device)
    train = target.train_samples(n=args.n_samples).cpu()
    target.cpu()
    test = target.test_samples(n=args.n_samples)
    metrics, hist = target.compute_metrics(train, ref_samples=test, return_hist=True)
    values = {split: {'dist': target.distances(x).reshape(-1).cpu().numpy(),
                      'energy': target.energy(x).cpu().numpy()}
              for split, x in [('test', test), ('train', train)]}
    systems[name] = {'target': target, 'test': test, 'train': train,
                     'metrics': metrics, 'hist': hist, 'values': values}

    # Print the counts, the panel and the energy ranges
    low, high = target.energy_histogram_range
    print(name, 'test', len(test), 'train', len(train))
    print(name, {key: round(float(value), 4) for key, value in metrics.items()})
    for split in ['test', 'train']:
        energy = values[split]['energy']
        print(name, split, 'energy range %.2f %.2f' % (energy.min(), energy.max()),
              'mean %.2f' % energy.mean(), 'below %.0f %.4f%%' % (low, 100 * (energy < low).mean()),
              'above %.0f %.4f%%' % (high, 100 * (energy > high).mean()))
    energy, train_energy = values['test']['energy'], values['train']['energy']
    outside = (energy < train_energy.min()) | (energy > train_energy.max())
    print(name, 'test energies outside the train range', int(outside.sum()), energy[outside])

# Locate the last kept frame of the train split and the temperature factor of every chain
for entry in systems.values():
    n_kept = len(entry['train']) // N_CHAINS
    entry['last'] = (n_kept - 1) * N_CHAINS
    entry['factors'] = temperature_ladder(N_CHAINS, entry['target'].train_temperatures).numpy()

# Pick the first DW-4 test configurations and the last train state of chains spread over the ladder
dw4 = systems['dw4']
n_shown = 6
dw4_chains = np.linspace(0, N_CHAINS - 1, n_shown).astype(int)
picks = {'test': np.arange(n_shown), 'train': dw4['last'] + dw4_chains}
dw4_shown = {split: [principal_frame(dw4[split][index].reshape(4, 2).numpy())
                     for index in picks[split]] for split in ['test', 'train']}
print('dw4 shown train factors', np.round(dw4['factors'][dw4_chains], 2))

# Render each cluster's median test configuration and the last state of its coldest and hottest
# train chain, at one scale
cluster_shown = []
for name in ['lj13', 'lj55']:
    entry = systems[name]
    energy = entry['values']['test']['energy']
    cluster_shown += [(name, 'test', int(np.argsort(energy)[len(energy) // 2]), None),
                      (name, 'train', entry['last'], entry['factors'][0]),
                      (name, 'train', entry['last'] + N_CHAINS - 1, entry['factors'][-1])]
cluster_positions, cluster_energies, views = [], [], []
for name, split, index, _ in cluster_shown:
    entry = systems[name]
    n_particles = entry['target'].n_particles
    cluster_positions.append(entry[split][index].reshape(n_particles, 3).numpy())
    cluster_energies.append(float(entry['values'][split]['energy'][index]))
    views.append(render_cluster(cluster_positions[-1], 64)[1])
print('cluster renders', [(name, split, round(energy, 2)) for (name, split, _, _), energy
                          in zip(cluster_shown, cluster_energies)])
distance = max(-view[11] for view in views)
cluster_renders = []
for positions, view in zip(cluster_positions, views):
    view = list(view)
    view[11], view[15], view[16] = -distance, distance - 100.0, distance + 100.0
    cluster_renders.append(render_cluster(positions, 900, view=tuple(view))[0])


def draw_configurations(theme):
    """Draws DW-4 configurations of both splits and three renders of each cluster"""
    fig = plt.figure(figsize=(8.0, 4.9))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 0.85])

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
    for ax, chain in zip(axes[1], dw4_chains):
        ax.set_xlabel('T × %.2f' % dw4['factors'][chain], family='DejaVu Sans Mono', fontsize=8,
                      color=theme['muted'])
    top.suptitle('dw4: four particles in 2D, pair distances drawn', fontsize=10)

    # The two clusters, with the coordination scale
    axes = bottom.subplots(1, len(cluster_renders))
    for ax, (name, split, _, factor), image, energy in zip(axes, cluster_shown, cluster_renders,
                                                           cluster_energies):
        ax.imshow(image)
        blank(ax)
        label = name + ', test' if factor is None else name + ', T × %.2f' % factor
        ax.set_title(label, family='DejaVu Sans Mono', fontsize=8)
        ax.set_xlabel('E = %.1f' % energy, family='DejaVu Sans Mono', fontsize=8,
                      color=theme['muted'])
    mappable = plt.cm.ScalarMappable(norm=Normalize(0, MAX_NEIGHBOURS), cmap=COORDINATION)
    bar = bottom.colorbar(mappable, ax=axes, orientation='horizontal', fraction=0.05,
                          shrink=0.4, aspect=30)
    bar.set_label('neighbours closer than 1.4 r', color=theme['ink'])
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=theme['muted'], labelcolor=theme['muted'])
    return fig


save(draw_configurations, 'particles_configurations')


def draw_histograms(theme):
    """Draws the distance and energy histograms of both splits within and beyond the panel's edges"""
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.6), gridspec_kw={'width_ratios': [1, 1, 0.62]})
    for row, (name, entry) in enumerate(systems.items()):
        metrics = entry['metrics']
        for column, key in enumerate(['dist', 'energy']):
            ax = axes[row, column]
            edges = entry['hist'][key]['edges'].cpu().numpy()
            span = edges[-1] - edges[0]
            x_window = (edges[0] - 0.12 * span, edges[-1] + 0.12 * span)

            # Inside the panel's edges, both splits on bins twice as fine as the panel's, each bin
            # the share of all the samples it holds; outside them, the mass the panel drops
            fine = np.linspace(edges[0], edges[-1], 2 * (len(edges) - 1) + 1)
            width = fine[1] - fine[0]
            ax.axvspan(x_window[0], edges[0], color=theme['grid'], alpha=0.5, linewidth=0)
            ax.axvspan(edges[-1], x_window[1], color=theme['grid'], alpha=0.5, linewidth=0)
            peaks = {}
            for split in ['test', 'train']:
                values = entry['values'][split][key]
                inside = np.histogram(values, bins=fine)[0] / len(values) / width
                peaks[split] = inside.max()
                ax.stairs(inside, fine, color=theme[split], linewidth=1.0)
                for side in outside_density(values, fine, x_window):
                    if side is not None and side[0].any():
                        ax.stairs(side[0], side[1], color=theme[split], linewidth=1.0)
            ax.set_xlim(*x_window)
            peak = 3.0 * peaks['test']
            if peaks['train'] > peak:
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

# Replay the adaptation moves of a few LJ-55 train chains, which the split does not keep
lj55 = systems['lj55']['target'].to(args.device)
chains = args.chains
factors = systems['lj55']['factors'][chains]
gen = generator(0, device=lj55.device)
x0 = lj55.initial_configuration() + 0.1 * torch.randn(
    N_CHAINS, lj55.dim, generator=gen, dtype=lj55.dtype, device=lj55.device)
moves = list(range(0, N_ADAPT + 1, args.trace_every))
states = [lj55.preprocess(x0[chains]).cpu()]
for move in moves[1:]:
    states.append(lj55.generate_train_samples(N_CHAINS, n_steps=move - 1, thin=1)[chains].cpu())

# Read the kept states off the split, after checking the replay against it
train = systems['lj55']['train']
n_kept = len(train) // N_CHAINS
kept = train[:n_kept * N_CHAINS].reshape(n_kept, N_CHAINS, lj55.dim)
replay = lj55.generate_train_samples(N_CHAINS, n_steps=N_ADAPT, thin=THIN).cpu()
assert torch.allclose(replay, kept[0], atol=1e-5)
states = torch.stack(states + list(kept[:, chains]))
moves = np.array(moves + [N_ADAPT + THIN * (j + 1) for j in range(n_kept)])
chain_energy = lj55.energy(states.reshape(-1, lj55.dim).to(lj55.device))
chain_energy = chain_energy.reshape(len(moves), len(chains)).cpu().numpy()

# Print the chains' energies
test_energy = systems['lj55']['values']['test']['energy']
train_energy = systems['lj55']['values']['train']['energy']
print('lj55 chains', chains, 'factors', np.round(factors, 3))
print('lj55 chain energy at the start', np.round(chain_energy[0], 1))
for k, chain in enumerate(chains):
    energy = chain_energy[moves > N_ADAPT, k]
    print('lj55 chain', chain, 'kept energy mean %.1f sd %.1f min %.1f max %.1f' % (
        energy.mean(), energy.std(), energy.min(), energy.max()),
        'above the test max %.0f%%' % (100 * (energy > test_energy.max()).mean()),
        'below the test min %.0f%%' % (100 * (energy < test_energy.min()).mean()))
print('lj55 test energy range %.1f %.1f, train split %.1f %.1f' % (
    test_energy.min(), test_energy.max(), train_energy.min(), train_energy.max()))

# Render every shown state, each chain rotated onto its own start
shown = np.nonzero(moves % args.frame_every == 0)[0]
positions = states.numpy().astype(np.float64).reshape(len(moves), len(chains), 55, 3)
frames = []
view = None
for index in shown:
    images = []
    for k in range(len(chains)):
        image, view = render_cluster(kabsch(positions[index, k], positions[0, k]), 300, view=view)
        images.append(image)
    frames.append(images)
energy_window = (-395.0, -195.0)
energy_bins = np.linspace(*energy_window, 101)
energy_width = energy_bins[1] - energy_bins[0]
split_density = {split: np.histogram(energy, bins=energy_bins)[0] / len(energy) / energy_width
                 for split, energy in [('test', test_energy), ('train', train_energy)]}
occupied = np.nonzero(split_density['train'])[0]
occupied = slice(occupied[0], occupied[-1] + 1)


def draw_chain_frame(theme, index):
    """Draws one state of each shown LJ-55 train chain next to its energy so far"""
    index = min(index, len(frames) - 1)
    state = shown[index]
    fig, axes = plt.subplots(len(chains), 3, figsize=(6.4, 1.95 * len(chains) + 0.4),
                             gridspec_kw={'width_ratios': [1.0, 1.55, 0.38]})
    for row, factor in enumerate(factors):
        ax = axes[row, 0]
        ax.imshow(frames[index][row])
        blank(ax)
        ax.set_title('T × %.2f' % factor, family='DejaVu Sans Mono', fontsize=9)

        # The chain's energy so far, the current state marked
        ax = axes[row, 1]
        ax.axvspan(0, N_ADAPT, color=theme['grid'], alpha=0.5, linewidth=0)
        ax.plot(moves[:state + 1], chain_energy[:state + 1, row], color=theme['train'],
                linewidth=1.0)
        if chain_energy[state, row] > energy_window[1]:
            ax.plot(moves[state], energy_window[1], '^', color=theme['train'], markersize=5,
                    clip_on=False)
        else:
            ax.plot(moves[state], chain_energy[state, row], 'o', color=theme['train'],
                    markersize=4)
        ax.set_xlim(0, moves[-1])
        ax.set_ylim(*energy_window)
        ax.set_ylabel('energy')
        ax.yaxis.grid(True, color=theme['grid'], linewidth=0.5)
        if row == len(chains) - 1:
            ax.set_xlabel('MALA move')
        else:
            ax.set_xticklabels([])

        # Both splits' energies, on the same axis
        ax = axes[row, 2]
        ax.stairs(split_density['test'], energy_bins, orientation='horizontal', fill=True,
                  color=theme['test'], linewidth=0)
        ax.stairs(split_density['train'][occupied], energy_bins[occupied.start:occupied.stop + 1],
                  orientation='horizontal', color=theme['train'], linewidth=1.0)
        ax.set_ylim(*energy_window)
        blank(ax)
    axes[0, 1].set_title('move %d' % moves[state], family='DejaVu Sans Mono', fontsize=9)
    legend(fig, theme, [('test split', 'test', 'patch'),
                        ('train chain, train split', 'train', 'line'),
                        ('adaptation, not kept', 'grid', 'patch')])
    return fig


# The last state is held for a second before the loop restarts
save_gif(draw_chain_frame, len(frames) + 12, 'particles_train_chain', fps=12, dpi=100)
