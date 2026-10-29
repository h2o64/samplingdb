# Draw the figures of the synthetic targets' page

# Libraries
import argparse, math, pprint, sys, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import ndtr
from samplingdb import ManyModes, build
from samplingdb.targets.base import generator
from samplingdb.targets.kernels import random_walk_mh
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import blank, density_map, legend, note, save

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=None)
parser.add_argument('--n_chain', type=int, default=10000)
parser.add_argument('--n_bins', type=int, default=100)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Make a Pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The 2D targets, each with the square window it is drawn in. ManyModes is drawn with ten modes,
# where the catalog builds three, so the figure shows what the family is about
WINDOWS = {
    'gmm40': (-45.0, 45.0), 'mos': (-16.0, 16.0), 'twomodes': (-2.0, 2.0),
    'brackettwomodes': (-2.2, 2.2), 'manymodes-k10': (-11.0, 11.0),
    'eightgaussians': (-3.0, 3.0), 'rings': (-6.5, 6.5), 'checkerboard': (-4.5, 4.5),
}


def make(name):
    """Builds a target by its name, ManyModes at any number of modes included"""
    if name.startswith('manymodes-k'):
        return ManyModes(n_modes=int(name[len('manymodes-k'):]))
    return build(name)


def log_density_grid(log_prob, x_window, y_window, n_grid=400):
    """Evaluates a 2D log density on a grid

    Args:
        * log_prob (function): x (n, 2) -> (n,), the unnormalised log density
        * x_window (tuple of two floats): Horizontal extent
        * y_window (tuple of two floats): Vertical extent
        * n_grid (int): Points per side (default is 400)

    Returns:
        * xs (numpy.ndarray of shape (n_grid,)): Horizontal grid
        * ys (numpy.ndarray of shape (n_grid,)): Vertical grid
        * values (numpy.ndarray of shape (n_grid, n_grid)): Rows along y, columns along x
    """
    xs = torch.linspace(*x_window, n_grid, dtype=torch.float64)
    ys = torch.linspace(*y_window, n_grid, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1).reshape(-1, 2)
    values = log_prob(grid.to(device)).reshape(n_grid, n_grid).cpu()
    return xs.numpy(), ys.numpy(), values.numpy()


def scaled(values):
    """Maps a log density to the density itself, scaled to [0, 1], for a ten-level image

    Each level of the image is then a tenth of the peak density.

    NOTE: the peak is the 99.9% quantile of the grid, so Rings' integrable spike at the origin does
    not set it.
    """
    density = np.exp(values - values[np.isfinite(values)].max())
    return np.clip(density / np.quantile(density, 0.999), 0.0, 1.0)


def grid_marginals(xs, ys, values):
    """Integrates a gridded log density into its two exact marginal densities

    Returns:
        * px (numpy.ndarray of shape (n_grid,)): Density of the horizontal coordinate
        * py (numpy.ndarray of shape (n_grid,)): Density of the vertical coordinate
    """
    density = np.exp(values - values[np.isfinite(values)].max())
    px, py = density.sum(0), density.sum(1)
    return px / np.trapezoid(px, xs), py / np.trapezoid(py, ys)


def joint(fig, spec, ratio=0.3):
    """Splits a grid cell into a main panel and two marginal panels, top and right

    Returns:
        * main (matplotlib.axes.Axes): Main panel
        * top (matplotlib.axes.Axes): Marginal of the horizontal coordinate
        * right (matplotlib.axes.Axes): Marginal of the vertical coordinate
        * corner (matplotlib.axes.Axes): The empty corner, blanked
    """
    inner = spec.subgridspec(2, 2, width_ratios=[1.0, ratio], height_ratios=[ratio, 1.0],
                             wspace=0.0, hspace=0.0)
    main = fig.add_subplot(inner[1, 0])
    top = fig.add_subplot(inner[0, 0], sharex=main)
    right = fig.add_subplot(inner[1, 1], sharey=main)
    corner = fig.add_subplot(inner[0, 1])
    for ax in [top, right, corner]:
        blank(ax)
    blank(main)
    return main, top, right, corner


def draw_density(ax, values, x_window, y_window, theme):
    """Draws a log density as a ten-level image"""
    ax.imshow(scaled(values), origin='lower', extent=(*x_window, *y_window), aspect='auto',
              cmap=density_map(theme), vmin=0.0, vmax=1.0)
    ax.set_xlim(*x_window)
    ax.set_ylim(*y_window)


def draw_marginal(ax, grid, exact, samples, edges, theme, vertical=False, ceiling=None):
    """Draws an exact marginal as a filled area and the two splits' histograms over it

    Args:
        * ax (matplotlib.axes.Axes): Axis
        * grid (numpy.ndarray of shape (n_grid,)): Where the exact density is evaluated
        * exact (numpy.ndarray of shape (n_grid,)): Exact density
        * samples (dict): 'test' and 'train', each a numpy.ndarray of shape (n,)
        * edges (numpy.ndarray of shape (n_bins + 1,)): Histogram edges
        * theme (dict): Theme
        * vertical (bool): Whether the marginal runs along the vertical axis (default is False)
        * ceiling (float or None): Top of the density axis; None fits everything
            (default is None)
    """
    fill = density_map(theme)(0.45)
    if vertical:
        ax.fill_betweenx(grid, 0.0, exact, color=fill, alpha=0.6, linewidth=0)
    else:
        ax.fill_between(grid, 0.0, exact, color=fill, alpha=0.6, linewidth=0)
    top = exact.max()
    for split in ['test', 'train']:
        hist, _ = np.histogram(samples[split], bins=edges, density=True)
        ax.stairs(hist, edges, color=theme[split], linewidth=1.0,
                  orientation='horizontal' if vertical else 'vertical')
        top = max(top, hist.max())
    top = 1.08 * top if ceiling is None else ceiling
    if vertical:
        ax.set_xlim(0.0, top)
    else:
        ax.set_ylim(0.0, top)


def weight_bars(ax, true, test, train, theme):
    """Draws the empirical mode weights of both splits against the true ones

    Args:
        * ax (matplotlib.axes.Axes): Axis
        * true (numpy.ndarray of shape (n_modes,)): True weights
        * test (numpy.ndarray of shape (n_modes,)): Empirical weights of the test split
        * train (numpy.ndarray of shape (n_modes,)): Empirical weights of the train split
        * theme (dict): Theme
    """
    index = np.arange(len(true))
    width = 0.4
    ax.bar(index - width / 2, test, width, color=theme['test'], linewidth=0)
    ax.bar(index + width / 2, train, width, color=theme['train'], linewidth=0)
    if np.allclose(true, true[0]):
        ax.axhline(true[0], color=theme['ink'], linewidth=1.2)
    else:
        ax.hlines(true, index - 0.45, index + 0.45, color=theme['ink'], linewidth=1.2)
    ax.set_xlim(-0.6, len(true) - 0.4)
    ax.set_ylim(0.0, 1.15 * max(true.max(), test.max(), train.max()))
    ax.set_xticks([])
    ax.yaxis.grid(True, color=theme['grid'], linewidth=0.5)
    ax.set_axisbelow(True)


# Build the eight 2D targets, their two splits, their exact marginals and their mode panels
targets = {}
for name, window in WINDOWS.items():
    target = make(name).to(device)
    test = target.test_samples(n=args.n_samples)
    train = target.train_samples(n=args.n_samples)
    test_metrics, test_hist = target.compute_metrics(test, compute_standard_metrics=False,
                                                     return_hist=True)
    train_metrics, train_hist = target.compute_metrics(train, compute_standard_metrics=False,
                                                       return_hist=True)
    xs, ys, values = log_density_grid(target.log_prob, window, window)
    px, py = grid_marginals(xs, ys, values)
    targets[name] = {
        'window': window, 'grid': xs, 'values': values, 'px': px, 'py': py,
        'test': test.cpu().numpy(), 'train': train.cpu().numpy(),
        'true': target.true_mode_weights[0].cpu().numpy(), 'test_hist': test_hist[0].cpu().numpy(),
        'train_hist': train_hist[0].cpu().numpy(), 'test_metrics': test_metrics,
        'train_metrics': train_metrics,
    }
    print(name, 'n test %d train %d' % (len(test), len(train)),
          'tv_weights test %.3f train %.3f' % (test_metrics['tv_weights'],
                                               train_metrics['tv_weights']))
    print('    weights true', np.round(targets[name]['true'], 3))
    print('    weights test', np.round(targets[name]['test_hist'], 3))
    print('    weights train', np.round(targets[name]['train_hist'], 3))


def draw_splits(theme):
    """Draws the eight 2D densities, both splits' marginals and both splits' mode weights"""
    fig = plt.figure(figsize=(8.0, 7.6))
    outer = fig.add_gridspec(4, 4, height_ratios=[1.0, 0.5, 1.0, 0.5])
    names = list(WINDOWS)
    for block, group in enumerate([names[:4], names[4:]]):
        for column, name in enumerate(group):
            entry = targets[name]
            main, top, right, _ = joint(fig, outer[2 * block, column])
            draw_density(main, entry['values'], entry['window'], entry['window'], theme)
            edges = np.linspace(*entry['window'], args.n_bins + 1)
            draw_marginal(top, entry['grid'], entry['px'],
                          {split: entry[split][:, 0] for split in ['test', 'train']}, edges, theme)
            draw_marginal(right, entry['grid'], entry['py'],
                          {split: entry[split][:, 1] for split in ['test', 'train']}, edges, theme,
                          vertical=True)
            top.set_title(name, family='DejaVu Sans Mono', loc='left')

            # The mode weights, with the panel's tv_weights for both splits
            ax = fig.add_subplot(outer[2 * block + 1, column])
            weight_bars(ax, entry['true'], entry['test_hist'], entry['train_hist'], theme)
            ax.set_xlabel('tv_weights  test %.2f\n           train %.2f'
                          % (entry['test_metrics']['tv_weights'],
                             entry['train_metrics']['tv_weights']),
                          family='DejaVu Sans Mono', fontsize=7, loc='left')
            if column == 0:
                ax.set_ylabel('mode weight')
    legend(fig, theme, [('target density', None, 'density'), ('exact marginal', None, 'fill'),
                        ('test split (exact draws)', 'test', 'line'),
                        ('train split', 'train', 'line'), ('true mode weight', 'ink', 'line')])
    return fig


save(draw_splits, 'synthetic_splits')

# GMM-40 under the README's Langevin loop and one random-walk Metropolis chain
gmm = build('gmm40').to(device)
gmm_test = gmm.test_samples(n=args.n_samples)
gen = generator(1, device=device)
ula = gmm.prior_scale * torch.randn(len(gmm_test), 2, generator=gen, dtype=torch.float64,
                                    device=device)
for step in range(500):
    ula = ula + 0.05 * gmm.score(ula) + 0.1 ** 0.5 * torch.randn(ula.shape, generator=gen,
                                                                 dtype=ula.dtype, device=device)
gmm = gmm.cpu()
chain = random_walk_mh(gmm.log_prob, gmm.means[:1].clone(), n_steps=50 * args.n_chain,
                       step_size=3.0, generator=generator(2), return_trace=True, thin=50)[:, 0]
coverage = []
for label, x in [('test split', gmm_test), ('Langevin from the prior', ula),
                 ('one Metropolis chain', chain)]:
    metrics, hist = gmm.compute_metrics(x, compute_standard_metrics=False, return_hist=True)
    coverage.append({'label': label, 'n': len(x), 'hist': hist[0].numpy(), 'metrics': metrics})
    print(label, 'n %d' % len(x), {key: round(float(value), 3) for key, value in metrics.items()},
          'modes below 5%% of 1/40: %d' % (hist[0] < 0.05 / gmm.n_modes).sum())
means = gmm.means.numpy()


def draw_mode_coverage(theme):
    """Draws three GMM-40 sample sets as mode-weight bubbles, sorted weights and coverage metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.6), gridspec_kw={'height_ratios': [1.0, 0.7]})
    true = 1.0 / gmm.n_modes
    ceiling = 1.75 * max(entry['hist'].max() for entry in coverage)
    area = 4000.0
    for column, entry in enumerate(coverage):
        key = 'test' if column == 0 else 'sample'

        # Every mode as a disc of area proportional to its empirical weight, in a ring of the true
        # one; a forgotten mode is an empty ring
        ax = axes[0, column]
        ax.scatter(means[:, 0], means[:, 1], s=area * entry['hist'], color=theme[key],
                   alpha=0.85, linewidths=0)
        ax.scatter(means[:, 0], means[:, 1], s=area * true, facecolors='none',
                   edgecolors=theme['ink'], linewidths=0.8)
        ax.set_xlim(-46.0, 46.0)
        ax.set_ylim(-46.0, 46.0)
        ax.set_aspect('equal')
        blank(ax)
        ax.set_title('%s\nn = %s' % (entry['label'], format(entry['n'], ',')))

        # The empirical weights sorted from the heaviest, against the true one
        ax = axes[1, column]
        order = np.sort(entry['hist'])[::-1]
        ax.bar(np.arange(gmm.n_modes), order, 0.75, color=theme[key], linewidth=0)
        ax.axhline(true, color=theme['ink'], linewidth=1.2)
        ax.axhline(0.05 * true, color=theme['muted'], linewidth=0.8)
        ax.set_ylim(0.0, ceiling)
        ax.set_xticks([])
        ax.set_xlabel('modes, heaviest first')
        ax.yaxis.grid(True, color=theme['grid'], linewidth=0.5)
        ax.set_axisbelow(True)
        metrics = entry['metrics']
        note(ax, ['emc                  %5.2f' % metrics['emc'],
                  'kl_weights           %5.2f' % metrics['kl_weights'],
                  'tv_weights           %5.2f' % metrics['tv_weights'],
                  'perc_forgotten_modes %5.2f' % metrics['perc_forgotten_modes']], theme, size=6.5)
    axes[1, 0].set_ylabel('mode weight')
    legend(fig, theme, [('empirical weight', 'muted', 'dot'), ('true weight 1/40', 'ink', 'line'),
                        ('forgotten below 5% of it', 'muted', 'line')])
    return fig


save(draw_mode_coverage, 'synthetic_mode_coverage')

# Neal's funnel: its first two coordinates, with the exact marginals of both
funnel = build('funnel').to(device)
funnel_splits = {'test': funnel.test_samples(n=args.n_samples).cpu().numpy(),
                 'train': funnel.train_samples(n=args.n_samples).cpu().numpy()}
funnel_metrics = {split: funnel.compute_metrics(torch.as_tensor(x), compute_standard_metrics=False)
                  for split, x in funnel_splits.items()}
for split, x in funnel_splits.items():
    print('funnel', split, 'n %d' % len(x), 'x1 range %.2f %.2f std %.2f' % (
        x[:, 0].min(), x[:, 0].max(), x[:, 0].std()),
          {key: round(float(value), 3) for key, value in funnel_metrics[split].items()})
x1_window, x2_window = (-13.0, 13.0), (-12.0, 12.0)


def funnel_log_prob(x):
    """Evaluates the exact log density of the funnel's first two coordinates"""
    return (-0.5 * (x[:, 0] / funnel.scale) ** 2 - 0.5 * x[:, 0]
            - 0.5 * x[:, 1] ** 2 * torch.exp(-x[:, 0]))


funnel_x1, funnel_x2, funnel_values = log_density_grid(funnel_log_prob, x1_window, x2_window)
funnel_p1 = np.exp(-0.5 * (funnel_x1 / funnel.scale) ** 2) / (funnel.scale * math.sqrt(2 * math.pi))

# The x_2 marginal mixes N(0, exp(x_1)) over x_1, a cusp at zero that a 2D grid cannot resolve, so it
# is integrated on its own: bin masses from the conditional CDFs, on a fine x_1 grid
x2_edges = np.linspace(*x2_window, 481)
x1_fine = np.linspace(-15.0, 15.0, 6001)
x1_weight = np.exp(-0.5 * (x1_fine / funnel.scale) ** 2)
x1_weight /= x1_weight.sum()
scale = np.exp(0.5 * x1_fine)
masses = (x1_weight[:, None] * np.diff(ndtr(x2_edges[None, :] / scale[:, None]), axis=1)).sum(0)
funnel_p2 = masses / np.diff(x2_edges)
funnel_c2 = 0.5 * (x2_edges[1:] + x2_edges[:-1])


def draw_funnel(theme):
    """Draws the funnel's density in (x_1, x_2) and both splits' marginals against the exact ones"""
    fig = plt.figure(figsize=(8.0, 3.9))
    outer = fig.add_gridspec(1, 1)
    main, top, right, corner = joint(fig, outer[0, 0], ratio=0.22)
    draw_density(main, funnel_values, x1_window, x2_window, theme)
    main.set_xticks([-12, -8, -4, 0, 4, 8, 12])
    main.set_yticks([-10, 0, 10])
    for spine in ['left', 'bottom']:
        main.spines[spine].set_visible(True)
    main.set_xlabel('$x_1$')
    main.set_ylabel('$x_2$')
    draw_marginal(top, funnel_x1, funnel_p1, {split: x[:, 0] for split, x in funnel_splits.items()},
                  np.linspace(*x1_window, 121), theme)
    draw_marginal(right, funnel_c2, funnel_p2,
                  {split: x[:, 1] for split, x in funnel_splits.items()},
                  np.linspace(*x2_window, 121), theme, vertical=True)
    rows = ['                  test  train']
    for key in ['ks_x1', 'tvd_x1', 'neck_var_rel_err']:
        rows.append('%-16s %5.2f %6.2f' % (key, funnel_metrics['test'][key],
                                          funnel_metrics['train'][key]))
    note(corner, rows, theme, loc='upper right', size=6.5)
    legend(fig, theme, [('target density', None, 'density'), ('exact marginal', None, 'fill'),
                        ('test split (exact draws)', 'test', 'line'),
                        ('train split', 'train', 'line')])
    return fig


save(draw_funnel, 'synthetic_funnel')

# ManyWell: one of its 16 double wells, the marginals pooled over wells, and the per-well weights
manywell = build('manywell').to(device)
well_splits = {'test': manywell.test_samples(n=args.n_samples),
               'train': manywell.train_samples(n=args.n_samples)}
well_panels = {split: manywell.compute_metrics(x, compute_standard_metrics=False,
                                               return_hist=True)
               for split, x in well_splits.items()}
for split, (metrics, hist) in well_panels.items():
    print('manywell', split, 'n %d' % len(well_splits[split]),
          {key: round(float(value), 4) for key, value in metrics.items()},
          'left mass per well %.3f to %.3f' % (hist[:, 0].min(), hist[:, 0].max()))
well_x1, well_x2 = (-2.6, 2.6), (-3.2, 3.2)
well_grid_x1, well_grid_x2, well_values = log_density_grid(
    lambda x: manywell.log_prob_1d(x[:, 0]) - 0.5 * x[:, 1] ** 2, well_x1, well_x2)
well_p1, well_p2 = grid_marginals(well_grid_x1, well_grid_x2, well_values)
pooled = {split: {'x1': x[:, 0::2].reshape(-1).cpu().numpy(),
                  'x2': x[:, 1::2].reshape(-1).cpu().numpy()}
          for split, x in well_splits.items()}


def draw_manywell(theme):
    """Draws one double well, the pooled marginals of both splits and the per-well left weight"""
    fig = plt.figure(figsize=(8.0, 3.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25])
    main, top, right, corner = joint(fig, outer[0, 0], ratio=0.3)
    draw_density(main, well_values, well_x1, well_x2, theme)
    for spine in ['left', 'bottom']:
        main.spines[spine].set_visible(True)
    main.set_xticks([-2, -1, 0, 1, 2])
    main.set_yticks([-2, 0, 2])
    main.set_xlabel('$x_1$ (top: pooled over the 16 wells)')
    main.set_ylabel('$x_2$')
    draw_marginal(top, well_grid_x1, well_p1, {split: pooled[split]['x1'] for split in pooled},
                  np.linspace(*well_x1, 161), theme)
    draw_marginal(right, well_grid_x2, well_p2, {split: pooled[split]['x2'] for split in pooled},
                  np.linspace(*well_x2, 161), theme, vertical=True)
    rows = ['        test  train']
    for key in ['tvd_x1', 'w1_x1']:
        rows.append('%-6s %5.2f %6.2f' % (key, well_panels['test'][0][key],
                                         well_panels['train'][0][key]))
    note(corner, rows, theme, loc='upper right', size=6)

    # The left-well weight of each of the 16 wells, the descriptor the coverage panel reads
    ax = fig.add_subplot(outer[0, 1])
    index = np.arange(manywell.n_wells)
    width = 0.4
    for offset, split in [(-width / 2, 'test'), (width / 2, 'train')]:
        ax.bar(index + offset, well_panels[split][1][:, 0].cpu().numpy(), width,
               color=theme[split], linewidth=0)
    true = float(manywell.true_mode_weights[0, 0])
    ax.axhline(true, color=theme['ink'], linewidth=1.2)
    ax.set_xticks(index[::3], [str(i + 1) for i in index[::3]])
    ax.set_xlabel('well')
    ax.set_ylabel('mass in the left well')
    ax.set_xlim(-0.6, manywell.n_wells - 0.4)
    ax.set_ylim(0.0, 0.75)
    ax.yaxis.grid(True, color=theme['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    rows = ['                     test  train']
    for key in ['emc', 'tv_weights', 'perc_forgotten_modes']:
        rows.append('%-20s %5.2f %6.2f' % (key, well_panels['test'][0][key],
                                          well_panels['train'][0][key]))
    note(ax, rows, theme, size=6.5)
    legend(fig, theme, [('target density', None, 'density'), ('exact marginal', None, 'fill'),
                        ('test split (exact draws)', 'test', 'line'),
                        ('train split', 'train', 'line'), ('true weight %.4f' % true, 'ink', 'line')])
    return fig


save(draw_manywell, 'synthetic_manywell')
