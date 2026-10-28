# Draw the figures of the glass formers' page

# Libraries
import argparse, pprint, sys, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba_array
from samplingdb import build
from samplingdb.targets.base import generator
from samplingdb.targets.physics import minimum_image, swap_monte_carlo
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import blank, legend, note, save, save_gif

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_train', type=int, default=512)
parser.add_argument('--n_ref', type=int, default=4096)
parser.add_argument('--n_chains', type=int, default=64)
parser.add_argument('--n_time_units', type=int, default=10000)
parser.add_argument('--n_frames', type=int, default=70)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# The three catalogued systems, with the short label each panel carries
SYSTEMS = {
    'ka-ternary-N44-T1.0': 'ternary, T = 1.0',
    'ka-ternary-N44-T0.32': 'ternary, T = 0.32',
    'ka-binary-N64-T0.2': 'binary, T = 0.2',
}

# Species colours, A then B then C: violet, aqua and yellow, every pair at least 8 OKLab Delta E x 100
# apart under simulated colour blindness in both modes, and none of them the test or train hue
SPECIES = {'light': ['#6250d6', '#1baf7a', '#eda100'], 'dark': ['#9085e9', '#199e70', '#c98500']}

# The far end of the |psi_6| ramp, one step past species A away from the page surface
DEEP_A = {'light': '#2e2470', 'dark': '#d9d4fb'}


def wrap(target, x):
    """Returns the positions of one configuration folded into the box, as (n_particles, 2)"""
    return np.mod(x.reshape(target.n_particles, 2), target.box)


def discs(ax, target, positions, colours, theme):
    """Draws particles as discs of diameter sigma_ii in the periodic box, images included

    Args:
        * ax (matplotlib.axes.Axes): Axis
        * target (PeriodicPairMixture): The target, for the box and the diameters
        * positions (numpy.ndarray of shape (n_particles, 2)): Positions inside the box
        * colours (list or numpy.ndarray of n_particles colours): Face colour of every particle
        * theme (dict): Theme
    """
    box = target.box
    diameters = target.sigma.diagonal()[target.species].cpu().numpy()
    shifts = np.array([[dx, dy] for dx in (-box, 0.0, box) for dy in (-box, 0.0, box)])
    offsets = (positions[None] + shifts[:, None]).reshape(-1, 2)
    sizes = np.tile(diameters, len(shifts))
    faces = np.tile(to_rgba_array(colours), (len(shifts), 1))

    # A ring of the page colour around every disc keeps touching neighbours apart
    collection = EllipseCollection(sizes, sizes, np.zeros(len(sizes)), units='xy', offsets=offsets,
                                   offset_transform=ax.transData, facecolors=faces,
                                   edgecolors=theme['surface'], linewidths=0.6)
    ax.add_collection(collection)
    ax.set_xlim(0.0, box)
    ax.set_ylim(0.0, box)
    ax.set_aspect('equal')
    blank(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(theme['muted'])
        spine.set_linewidth(0.6)


def species_colours(target, theme):
    """Returns the face colour of every particle, by species"""
    palette = SPECIES[theme['name']]
    return [palette[int(label)] for label in target.species]


def hexatic(target, x):
    """Evaluates the local hexatic order psi_6 of every particle over its six nearest neighbours

    Args:
        * target (PeriodicPairMixture): The target
        * x (torch.Tensor of shape (n, dim)): Configurations

    Returns:
        * psi (torch.Tensor of shape (n, n_particles)): psi_6 per particle, complex, modulus in
            [0, 1]
    """
    positions = target.positions(x)
    delta = minimum_image(positions[:, None] - positions[:, :, None], target.box)
    distances = torch.linalg.norm(delta, dim=-1)
    distances = distances + torch.diag(torch.full((target.n_particles,), float('inf'),
                                                  dtype=distances.dtype))
    nearest = distances.topk(6, dim=-1, largest=False).indices
    bonds = torch.gather(delta, 2, nearest[..., None].expand(-1, -1, -1, 2))
    angles = torch.atan2(bonds[..., 1], bonds[..., 0])
    return torch.complex(torch.cos(6 * angles), torch.sin(6 * angles)).mean(-1)


def swap_palette(theme, steps=10):
    """Returns the swap animation's fixed palette: every colour it draws, blended with the page

    NOTE: a palette fitted to the frames merges the thin train line into the large fills of species
    C, whose yellow is close to it, so this animation lists its own colours instead.

    Args:
        * theme (dict): Theme
        * steps (int): Blends of every colour with the surface (default is 10)

    Returns:
        * colours (list of numpy.ndarray of shape (3,)): RGB colours in [0, 1]
    """
    theme_ = dict(theme, **dict(zip(['A', 'B', 'C'], SPECIES[theme['name']])))
    surface = np.array(to_rgb(theme['surface']))
    return [surface + weight * (np.array(to_rgb(theme_[key])) - surface)
            for key in ['ink', 'muted', 'grid', 'test', 'train', 'A', 'B', 'C']
            for weight in np.linspace(0.0, 1.0, steps + 1)[1:]]


# Build the three systems, their two splits and their panels, with the floor of each panel: the
# first n_train reference rows scored against the next n_ref, as the train split is
systems = {}
for name in SYSTEMS:
    target = build(name)
    test = target.test_samples(n=args.n_train + args.n_ref)
    reference, held_out = test[args.n_train:], test[:args.n_train]
    train = target.train_samples(n=args.n_train)
    metrics, hist = target.compute_metrics(train, ref_samples=reference, return_hist=True)
    floor = target.compute_metrics(held_out, ref_samples=reference)
    systems[name] = {'target': target, 'test': held_out, 'train': train, 'metrics': metrics,
                     'hist': hist, 'floor': floor,
                     'energy': target.energy(train).numpy() / target.n_particles,
                     'ref_energy': target.energy(reference).numpy() / target.n_particles}
    print(name, 'train', {key: round(value, 4) for key, value in metrics.items()})
    print(name, 'floor', {key: round(value, 4) for key, value in floor.items()})

# The binary's test configuration: among the 16 closest to the median of the set in both the
# crystalline fraction of A (|psi_6| > 0.8) and the coherence of A's orientation (|mean psi_6|),
# the one whose B particles touch each other most, so the band they form is in view
binary = systems['ka-binary-N64-T0.2']
target = binary['target']
is_a, is_b = target.species == 0, target.species == 1


def order_statistics(x):
    """Returns the crystalline fraction of A, the coherence of A and the B-B contact fraction"""
    psi = hexatic(target, x)
    positions = target.positions(x)
    delta = minimum_image(positions[:, None] - positions[:, :, None], target.box)
    distances = torch.linalg.norm(delta, dim=-1)
    distances = distances + torch.diag(torch.full((target.n_particles,), float('inf')))
    nearest = distances.topk(6, dim=-1, largest=False).indices
    contacts = is_b[nearest][:, is_b].float().mean(dim=(1, 2))
    return (psi.abs()[:, is_a] > 0.8).float().mean(-1), psi[:, is_a].mean(-1).abs(), contacts


crystalline, coherence, contacts = order_statistics(binary['test'])
distance = ((crystalline - crystalline.median()).abs() / crystalline.std()
            + (coherence - coherence.median()).abs() / coherence.std())
typical = distance.argsort()[:16]
pick = int(typical[contacts[typical].argmax()])
binary['pick'] = pick
binary['psi_test'] = hexatic(target, binary['test'][pick:pick + 1])[0].abs().numpy()
binary['psi_train'] = hexatic(target, binary['train'][:1])[0].abs().numpy()
train_crystalline, train_coherence, train_contacts = order_statistics(binary['train'])
print('binary test:  crystalline A %.3f, coherence %.3f, B-B contacts %.3f (medians); picked %d '
      'at %.3f, %.3f, %.3f' % (crystalline.median(), coherence.median(), contacts.median(), pick,
                               crystalline[pick], coherence[pick], contacts[pick]))
print('binary train: crystalline A %.3f, coherence %.3f, B-B contacts %.3f (medians)'
      % (train_crystalline.median(), train_coherence.median(), train_contacts.median()))


def draw_configurations(theme):
    """Draws one test and one train configuration of each system, and the binary's |psi_6|

    NOTE: |psi_6| is one hue, species A's, the species that crystallises: from the grid colour,
    so a disordered particle stays visible on the page, to a deeper step of A than the species
    panels use, so the crystallite stands out from the band.
    """
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.6))
    hexatic_map = LinearSegmentedColormap.from_list('hexatic', [theme['grid'], theme['A'],
                                                                DEEP_A[theme['name']]])
    for column, (name, label) in enumerate(SYSTEMS.items()):
        entry = systems[name]
        target = entry['target']
        index = entry.get('pick', 0)
        for row, split in enumerate(['test', 'train']):
            x = entry[split][index if split == 'test' else 0]
            discs(axes[row, column], target, wrap(target, x.numpy()),
                  species_colours(target, theme), theme)
        axes[0, column].set_title(label)

    # The binary's two configurations again, every particle shaded by its hexatic order
    target = binary['target']
    for row, split in enumerate(['test', 'train']):
        x = binary[split][binary['pick'] if split == 'test' else 0].numpy()
        colours = hexatic_map(binary['psi_' + split])
        discs(axes[row, 3], target, wrap(target, x), colours, theme)
    axes[0, 3].set_title('binary, $|\\psi_6|$')
    mappable = plt.cm.ScalarMappable(cmap=hexatic_map, norm=plt.Normalize(0.0, 1.0))
    bar = fig.colorbar(mappable, ax=axes[:, 3], shrink=0.6, aspect=25, pad=0.03)
    bar.outline.set_visible(False)
    bar.set_ticks([0.0, 0.5, 1.0])
    axes[0, 0].set_ylabel('test split')
    axes[1, 0].set_ylabel('train split')
    legend(fig, theme, [('A', 'A', 'patch'), ('B', 'B', 'patch'),
                        ('C (ternary only)', 'C', 'patch')])
    return fig


def themed(draw):
    """Adds the species colours to the theme under 'A', 'B' and 'C', so legend() can name them"""
    def wrapped(theme):
        return draw(dict(theme, **dict(zip(['A', 'B', 'C'], SPECIES[theme['name']]))))
    return wrapped


save(themed(draw_configurations), 'glasses_configurations')


def draw_structure(theme):
    """Draws g(r) and the per-particle energy of both splits, with the panel's numbers"""
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.8))
    for column, (name, label) in enumerate(SYSTEMS.items()):
        entry = systems[name]
        target = entry['target']
        metrics, floor, hist = entry['metrics'], entry['floor'], entry['hist']

        # g(r) of both splits, and where the two differ most
        ax = axes[0, column]
        rdf = hist['rdf']
        centres = rdf['centres'].numpy()
        ax.plot(centres, rdf['ref_g'].numpy(), color=theme['test'], linewidth=1.2)
        ax.plot(centres, rdf['g'].numpy(), color=theme['train'], linewidth=1.0)
        worst = int((rdf['g'] - rdf['ref_g']).abs().argmax())
        ax.axvline(centres[worst], color=theme['muted'], linewidth=0.8)
        ax.set_xlim(0.5, max(target.box / 2, centres[worst] + 0.3))
        ax.set_xlabel('$r$')
        ax.set_title(label)
        note(ax, ['rdf_max_abs_err', 'train %5.2f' % metrics['rdf_max_abs_err'],
                  'floor %5.2f' % floor['rdf_max_abs_err']], theme, size=6.5)

        # Per-particle energies of both splits on shared display edges, the panel's own edges
        # shaded, so the train mass it drops stays in view
        ax = axes[1, column]
        energy, ref_energy = entry['energy'], entry['ref_energy']
        low = min(energy.min(), ref_energy.min())
        high = max(energy.max(), ref_energy.max())
        edges = np.linspace(low, high, 159)
        heights = [ax.hist(values, bins=edges, density=True, histtype='step', color=theme[key],
                           linewidth=1.0)[0].max()
                   for values, key in [(ref_energy, 'test'), (energy, 'train')]]
        panel_edges = hist['energy']['edges'].numpy()
        ax.axvspan(panel_edges[0], panel_edges[-1], color=theme['grid'], alpha=0.5, linewidth=0)
        ax.set_ylim(0.0, 1.75 * max(heights))
        ax.set_xlabel('energy per particle')
        ax.set_yticks([])
        rows = ['                    train floor']
        for key in ['energy_total_var', 'energy_w1', 'energy_out_of_range']:
            rows.append('%-19s %5.3f %5.3f' % (key, metrics[key], floor[key]))
        note(ax, rows, theme, size=6)
    axes[0, 0].set_ylabel('$g(r)$')
    axes[1, 0].set_ylabel('density')
    legend(fig, theme, [('test split', 'test', 'line'), ('train split', 'train', 'line'),
                        ('energy edges of the panel', 'grid', 'patch')])
    return fig


save(draw_structure, 'glasses_structure')

# The train recipe's start, run at T = 0.32 under the train recipe's kernel (displacement only)
# and under the reference generator's (swap Monte Carlo), frames at log-spaced times
glass = build('ka-ternary-N44-T0.32').to(args.device)
start = glass.lattice_configuration(args.n_chains, jitter=0.05, generator=generator(0))
times = np.unique(np.round(np.logspace(0, np.log10(args.n_time_units), args.n_frames)).astype(int))
times = np.concatenate([[0], times])
runs = {}
for key, p_swap in [('train', 0.0), ('test', 0.2)]:
    gen = torch.Generator(device=args.device).manual_seed(1)
    x = start.to(args.device)
    frames, energies = [], []
    for previous, time in zip(np.concatenate([[0], times[:-1]]), times):
        if time > previous:
            x = swap_monte_carlo(glass, x, n_time_units=int(time - previous), p_swap=p_swap,
                                 generator=gen, compile=args.device == 'cuda')
            x = x.reshape(args.n_chains, glass.n_particles, 2)
        flat = x.reshape(args.n_chains, -1)
        frames.append(flat[0].cpu().numpy())
        energies.append((glass.energy(flat) / glass.n_particles).cpu().numpy())
    runs[key] = {'frames': frames, 'energy': np.stack(energies)}
    print(key, 'final energy per particle %.4f +- %.4f' % (runs[key]['energy'][-1].mean(),
                                                           runs[key]['energy'][-1].std()))
glass = glass.cpu()
band = systems['ka-ternary-N44-T0.32']['ref_energy']
band_low, band_mid, band_high = np.quantile(band, [0.05, 0.5, 0.95])
print('test energy per particle: median %.4f, 90%% band [%.4f, %.4f]'
      % (band_mid, band_low, band_high))
for key in ['train', 'test']:
    mean = runs[key]['energy'].mean(-1)
    inside = np.nonzero(mean <= band_high)[0]
    print(key, 'mean energy at t = 2000: %.4f; first time inside the band: %s'
          % (mean[np.searchsorted(times, 2000)] if times[-1] >= 2000 else float('nan'),
             times[inside[0]] if len(inside) else 'never'))

# The energy axis shows the late part of the runs, both chains entering it from the lattice above
low = min(run['energy'].mean(-1).min() for run in runs.values())
top = -1.0
hold = 15


def draw_swap_frame(theme, index):
    """Draws one frame: the two chains' configurations, then their mean energies so far"""
    theme = dict(theme, **dict(zip(['A', 'B', 'C'], SPECIES[theme['name']])))
    index = min(index, len(times) - 1)
    fig = plt.figure(figsize=(6.0, 5.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.12)
    colours = species_colours(glass, theme)
    for column, (key, label) in enumerate([('train', 'displacement only (train recipe)'),
                                           ('test', 'swap Monte Carlo (reference)')]):
        ax = fig.add_subplot(grid[0, column])
        discs(ax, glass, wrap(glass, runs[key]['frames'][index]), colours, theme)
        ax.set_title(label, fontsize=9)

    # Mean energy per particle over the chains, against the test split's 90% band
    ax = fig.add_subplot(grid[1, :])
    ax.axhspan(band_low, band_high, color=theme['test'], alpha=0.18, linewidth=0)
    for key, label in [('train', 'displacement only'), ('test', 'swap')]:
        mean = runs[key]['energy'][1:index + 1].mean(-1)
        ax.plot(times[1:index + 1], mean, color=theme[key], linewidth=1.5)
        if index > 0 and mean[-1] < top:
            ax.text(times[index] * 1.15, mean[-1], label, color=theme['ink'], fontsize=7.5,
                    va='center')
    ax.axvline(2000, color=theme['muted'], linewidth=0.8)
    ax.text(2000, 1.02, 'train split ends', color=theme['muted'], fontsize=7.5, ha='center',
            va='bottom', transform=ax.get_xaxis_transform())
    ax.set_xscale('log')
    ax.set_xlim(1, times[-1] * 6)
    ax.set_ylim(min(low, band_low) - 0.05, top)
    ax.set_xlabel('time units (one is %d attempted moves)' % glass.n_particles)
    ax.set_ylabel('energy per particle')
    ax.text(0.99, 0.95, 't = %d' % times[index], transform=ax.transAxes, ha='right', va='top',
            fontsize=9, family='DejaVu Sans Mono')
    handles = [plt.Line2D([], [], color=theme['train'], linewidth=1.5, label='displacement only'),
               plt.Line2D([], [], color=theme['test'], linewidth=1.5, label='swap'),
               plt.Rectangle((0, 0), 1, 1, color=theme['test'], alpha=0.18,
                             label='test split, 90% of it')]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, handletextpad=0.4,
               columnspacing=1.6)
    return fig


save_gif(draw_swap_frame, len(times) + hold, 'glasses_swap', fps=10, palette=swap_palette)
