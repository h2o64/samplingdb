# Draw the figures of the Bayesian posteriors' page

# Libraries
import argparse, pprint, sys, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from samplingdb import build
from samplingdb.targets.bayesian import fetch_dataset
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import blank, density_map, legend, ramp, save, save_gif

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_test', type=int, default=None)
parser.add_argument('--n_train', type=int, default=None)
parser.add_argument('--n_frames', type=int, default=20)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Make a Pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The four logistic regressions, under their catalog names
LOGISTIC = ['ionosphere', 'sonar', 'german-credit', 'breast-cancer']


def count(n):
    """Writes a sample count with its thousands apart, 100 000 for 100000"""
    return format(n, ',').replace(',', ' ')


def splits(name):
    """Returns a target and its three sample sets: test and train draws, and the reference

    The test split is cut in two halves: the first is scored, which gives the self-comparison
    floor, and the second is the reference both sets are scored against, so the floor never
    compares a draw with itself.

    Args:
        * name (str): Catalog name

    Returns:
        * target (BayesianTarget): The posterior, on device
        * test (torch.Tensor of shape (n_test // 2, dim)): Test draws, scored
        * train (torch.Tensor of shape (n_train, dim)): Train draws, scored
        * reference (torch.Tensor of shape (n_test - n_test // 2, dim)): Test draws, scored against
    """
    target = build(name).to(device)
    rows = target.test_samples(n=args.n_test)
    half = len(rows) // 2
    return target, rows[:half], target.train_samples(n=args.n_train), rows[half:]


def panels(target, test, train, reference):
    """Scores both sets against the reference, histograms included

    Returns:
        * scored (dict): 'test' and 'train', each a (metrics, hist) pair
    """
    scored = {}
    for split, x in [('test', test), ('train', train)]:
        metrics, hist = target.compute_metrics(x, ref_samples=reference, return_hist=True)
        scored[split] = (metrics, {key: value.cpu().numpy() for key, value in hist.items()})
        print(target.name, split, len(x), 'draws against', len(reference),
              {key: round(float(value), 4) for key, value in metrics.items()})
    return scored


def metric_line(scored, keys):
    """Returns one line of metric values, test then train, for under a figure"""
    return '   '.join(['%s  test %.2f  train %.2f' % (key, scored['test'][0][key],
                                                    scored['train'][0][key]) for key in keys])


def binned_image(x, y, x_window, y_window, n_grid=200, smooth=1.5):
    """Estimates a 2D density on a grid by a fine histogram, smoothed and scaled to a maximum of one

    Args:
        * x (numpy.ndarray of shape (n,)): Horizontal coordinate of the draws
        * y (numpy.ndarray of shape (n,)): Vertical coordinate of the draws
        * x_window (tuple of two floats): Horizontal extent
        * y_window (tuple of two floats): Vertical extent
        * n_grid (int): Bins per side (default is 200)
        * smooth (float): Standard deviation of the Gaussian smoothing, in bins (default is 1.5)

    Returns:
        * image (numpy.ndarray of shape (n_grid, n_grid)): Rows along y, columns along x
    """
    hist = np.histogram2d(y, x, bins=n_grid, range=[y_window, x_window])[0]
    image = gaussian_filter(hist, smooth)
    return image / image.max()


def marginal(x, edges):
    """Returns a histogram as a density over all the draws, so mass outside the edges is lost"""
    return np.histogram(x, bins=edges)[0] / (len(x) * np.diff(edges))


def joint(axes, theme, reference, sets, image, x_window, y_window, n_bins=150):
    """Draws a joint plot: the reference density in the centre, both splits in the marginals

    The centre is the density of the reference draws alone; the test and train splits are read in
    the two marginals, against the reference's own marginal drawn as a light fill. Every marginal
    is normalised over all its draws, so a split that spills out of the window shows less area.

    Args:
        * axes (tuple of three matplotlib.axes.Axes): Centre, top marginal, right marginal
        * theme (dict): Theme
        * reference (numpy.ndarray of shape (n, 2)): Reference draws
        * sets (dict): 'test' and 'train', each a numpy.ndarray of shape (m, 2)
        * image (numpy.ndarray): The reference density on the window, from binned_image
        * x_window (tuple of two floats): Horizontal extent
        * y_window (tuple of two floats): Vertical extent
        * n_bins (int): Bins per marginal (default is 150)
    """
    centre, top, right = axes
    centre.imshow(image, origin='lower', extent=(*x_window, *y_window), cmap=density_map(theme),
                  vmin=0.0, vmax=1.0, aspect='auto')
    centre.set_xlim(*x_window)
    centre.set_ylim(*y_window)

    # The two marginals, the reference filled and both splits as outlines on the same edges
    fill = density_map(theme)(0.45)
    for column, ax, window in [(0, top, x_window), (1, right, y_window)]:
        edges = np.linspace(*window, n_bins + 1)
        orientation = 'vertical' if column == 0 else 'horizontal'
        hist = marginal(reference[:, column], edges)
        ax.stairs(hist, edges, fill=True, color=fill, linewidth=0, orientation=orientation)
        peak = hist.max()
        for split in ['test', 'train']:
            hist = marginal(sets[split][:, column], edges)
            ax.stairs(hist, edges, color=theme[split], linewidth=1.2, orientation=orientation)
            peak = max(peak, hist.max())
        if column == 0:
            ax.set_ylim(0.0, 1.05 * peak)
        else:
            ax.set_xlim(0.0, 1.05 * peak)
    top.set_xlim(*x_window)
    right.set_ylim(*y_window)
    top.set_xticks([])
    top.set_yticks([])
    right.set_xticks([])
    right.set_yticks([])
    top.spines['left'].set_visible(False)
    right.spines['bottom'].set_visible(False)


def joint_legend(fig, theme, reference, sets):
    """Draws the legend shared by the joint plots"""
    theme_ = dict(theme, reference=density_map(theme)(0.45))
    legend(fig, theme_, [('test split, %s draws' % count(len(sets['test'])), 'test', 'line'),
                         ('train split, %s draws' % count(len(sets['train'])), 'train', 'line'),
                         ('reference, %s other test draws' % count(len(reference)), 'reference',
                          'patch')])


# The moment panel of the four logistic regressions
moments = {}
for name in LOGISTIC:
    target, test, train, reference = splits(name)
    moments[name] = panels(target, test, train, reference)
n_moments = {'test': len(test), 'train': len(train)}


def draw_moments(theme):
    """Draws the per-coordinate mean and spread errors of both splits on the four datasets"""
    fig, axes = plt.subplots(4, 2, figsize=(8.0, 7.6), sharex='row')
    for row, name in enumerate(LOGISTIC):
        scored = moments[name]
        for column, key in enumerate(['mean', 'std']):
            ax = axes[row, column]
            ax.axhline(0.0, color=theme['grid'], linewidth=0.8)
            for split in ['test', 'train']:
                hist = scored[split][1]
                if key == 'mean':
                    error = (hist['mean'] - hist['ref_mean']) / hist['ref_std']
                else:
                    error = hist['std'] / hist['ref_std'] - 1.0
                index = np.arange(len(error))
                ax.scatter(index, error, s=9, color=theme[split], linewidths=0, zorder=3)

                # The worst coordinate, which is what the max_* metric reads
                worst = np.argmax(np.abs(error))
                ax.scatter(worst, error[worst], s=60, facecolors='none',
                           edgecolors=theme['ink'], linewidths=0.8, zorder=4)
            ax.yaxis.grid(True, color=theme['grid'], linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlim(-1, len(error))
            # The metrics the strip is read with, test then train
            keys = ['max_mean_err', 'mean_mean_err'] if key == 'mean' else ['max_std_ratio_err']
            width = max(len(metric) for metric in keys)
            ax.set_title('\n'.join(metric.ljust(width) + '  test %.2f  train %.2f'
                                   % (scored['test'][0][metric], scored['train'][0][metric])
                                   for metric in keys),
                         loc='left', family='DejaVu Sans Mono', fontsize=7)
        axes[row, 0].set_ylabel(name, family='DejaVu Sans Mono')
    axes[-1, 0].set_xlabel('coordinate (0 is the intercept)\n'
                           '(mean - reference mean) / reference sd')
    axes[-1, 1].set_xlabel('coordinate\nsd / reference sd - 1')
    legend(fig, theme, [('test split, %s draws' % count(n_moments['test']), 'test', 'dot'),
                        ('train split, %s draws' % count(n_moments['train']), 'train', 'dot')])
    return fig


save(draw_moments, 'posteriors_moments')

# Seeds: the neck of its funnel, log tau against the first random effect
seeds, test, train, reference = splits('seeds')
seeds_scored = panels(seeds, test, train, reference)
columns = [-1, 4]
seeds_sets = {'test': test[:, columns].cpu().numpy(), 'train': train[:, columns].cpu().numpy()}
seeds_reference = reference[:, columns].cpu().numpy()
seeds_window = (-3.2, 6.6), (-2.8, 1.8)
seeds_image = binned_image(seeds_reference[:, 0], seeds_reference[:, 1], *seeds_window)


def draw_seeds(theme):
    """Draws the Seeds funnel in (log tau, b_1), with both splits in the marginals"""
    fig, mosaic = plt.subplot_mosaic([['top', '.'], ['joint', 'right']], figsize=(8.0, 3.6),
                                     width_ratios=[1.0, 0.16], height_ratios=[0.28, 1.0])
    joint((mosaic['joint'], mosaic['top'], mosaic['right']), theme, seeds_reference, seeds_sets,
          seeds_image, *seeds_window)
    mosaic['joint'].set_xlabel(r'$\log \tau$')
    mosaic['joint'].set_ylabel('$b_1$')
    fig.supxlabel(metric_line(seeds_scored, ['log_tau_mean_err', 'log_tau_w1']),
                  family='DejaVu Sans Mono', fontsize=7)
    joint_legend(fig, theme, seeds_reference, seeds_sets)
    return fig


save(draw_seeds, 'posteriors_seeds')

# Brownian motion: the path, its unobserved middle, and the two scales
brownian, test, train, reference = splits('brownian')
brownian_scored = panels(brownian, test, train, reference)
brownian_sets = {'test': test.cpu().numpy(), 'train': train.cpu().numpy()}
brownian_reference = reference[:, :2].cpu().numpy()
scale_window = (-3.8, 0.6), (-7.6, 0.0)
scale_image = binned_image(brownian_reference[:, 0], brownian_reference[:, 1], *scale_window)
steps = np.arange(1, brownian.n_steps + 1)
observed = brownian.observed.cpu().numpy()
y = brownian.y.cpu().numpy()


def draw_brownian(theme):
    """Draws the posterior path of both splits against the data, and the joint of the scales"""
    fig, mosaic = plt.subplot_mosaic([['test', 'train', 'top', '.'],
                                      ['test', 'train', 'joint', 'right']], figsize=(8.0, 3.3),
                                     width_ratios=[1.15, 1.15, 1.0, 0.2],
                                     height_ratios=[0.28, 1.0])
    for split in ['test', 'train']:
        ax = mosaic[split]
        path = brownian_sets[split][:, 2:]
        low90, low50, median, high50, high90 = np.quantile(path, [0.05, 0.25, 0.5, 0.75, 0.95],
                                                           axis=0)
        ax.axvspan(10.5, 20.5, color=theme['grid'], alpha=0.5, linewidth=0)
        ax.fill_between(steps, low90, high90, color=theme[split], alpha=0.25, linewidth=0)
        ax.fill_between(steps, low50, high50, color=theme[split], alpha=0.45, linewidth=0)
        ax.plot(steps, median, color=theme[split], linewidth=1.5)
        ax.scatter(steps[observed], y[observed], s=12, color=theme['ink'], zorder=4)
        ax.set_xlim(0.5, 30.5)
        ax.set_ylim(-1.35, 0.55)
        ax.set_xlabel('step')
        ax.set_title(split + ' split')
        ax.text(15.5, 0.5, 'unobserved', ha='center', va='top', color=theme['muted'],
                fontsize=7.5)
    mosaic['test'].set_ylabel('$x_t$')
    mosaic['train'].set_yticklabels([])

    # The two log scales, the funnel of this posterior
    sets = {split: x[:, :2] for split, x in brownian_sets.items()}
    joint((mosaic['joint'], mosaic['top'], mosaic['right']), theme, brownian_reference, sets,
          scale_image, *scale_window)
    mosaic['top'].set_title('the two scales')
    mosaic['joint'].set_xlabel(r'$\log \alpha_{inn}$')
    mosaic['joint'].set_ylabel(r'$\log \alpha_{obs}$')
    fig.supxlabel(metric_line(brownian_scored, ['scale_mean_err', 'missing_middle_mean_err',
                                                'observed_mean_err']),
                  family='DejaVu Sans Mono', fontsize=7)
    theme_ = dict(theme, reference=density_map(theme)(0.45))
    legend(fig, theme_, [('test split, %s draws' % count(len(brownian_sets['test'])), 'test',
                          'patch'),
                         ('train split, %s draws' % count(len(brownian_sets['train'])), 'train',
                          'patch'),
                         ('observation', 'ink', 'dot'),
                         ('reference, %s other test draws' % count(len(brownian_reference)),
                          'reference', 'patch')])
    return fig


save(draw_brownian, 'posteriors_brownian')

# LGCP: the pines, and the posterior intensity field of both splits
lgcp, test, train, reference = splits('lgcp')
lgcp_scored = panels(lgcp, test, train, reference)
pines = np.loadtxt(fetch_dataset('pines.csv'), delimiter=',', skiprows=1, usecols=(1, 2))
n_grid = lgcp.n_grid


def grid(values):
    """Maps a row-major field to an image, rows along the second pine coordinate"""
    return values.reshape(n_grid, n_grid).T


lgcp_sets = {'test': test, 'train': train}
lgcp_fields = {}
for split, x in lgcp_sets.items():
    field = lgcp.positions(x)
    lgcp_fields[split] = {'intensity': grid(torch.exp(field).mean(0).cpu().numpy()),
                          'std': grid(field.std(0).cpu().numpy())}
intensity_norm = Normalize(0.0, np.quantile(np.concatenate(
    [entry['intensity'].ravel() for entry in lgcp_fields.values()]), 0.99))
std_norm = Normalize(*np.quantile(np.concatenate(
    [entry['std'].ravel() for entry in lgcp_fields.values()]), [0.01, 0.99]))


def pine_dots(ax, theme, size=5):
    """Draws the sapling locations, ringed in the surface colour so they read on any cell"""
    ax.scatter(pines[:, 0], pines[:, 1], s=size, color=theme['ink'], edgecolors=theme['surface'],
               linewidths=0.4, zorder=3)


def frame_axis(ax, theme, split, title):
    """Blanks a field's axis, titles it and outlines it in the colour of its split"""
    blank(ax)
    ax.set_title(title, color=theme['ink'])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(theme[split])
        spine.set_linewidth(1.5)


def draw_lgcp(theme):
    """Draws the posterior mean intensity and the spread of the field for both splits"""
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 3.0))
    images = []
    for column, (key, split) in enumerate([('intensity', 'test'), ('intensity', 'train'),
                                           ('std', 'test'), ('std', 'train')]):
        ax = axes[column]
        # The intensity is a density and is drawn as one; its spread is a plain magnitude
        if key == 'intensity':
            norm, cmap = intensity_norm, density_map(theme)
        else:
            norm, cmap = std_norm, ramp(theme, theme['ink']).resampled(10)
        images.append(ax.imshow(lgcp_fields[split][key], origin='lower', extent=(0, 1, 0, 1),
                                cmap=cmap, norm=norm, interpolation='nearest'))
        if key == 'intensity':
            pine_dots(ax, theme)
        frame_axis(ax, theme, split, split + ' split')
    fig.colorbar(images[0], ax=axes[:2], orientation='horizontal', shrink=0.8, aspect=30,
                 label='posterior mean intensity, saplings per unit area')
    fig.colorbar(images[2], ax=axes[2:], orientation='horizontal', shrink=0.8, aspect=30,
                 label='posterior sd of the log intensity')
    fig.supxlabel(metric_line(lgcp_scored, ['total_intensity_err', 'neighbour_correlation_err']),
                  family='DejaVu Sans Mono', fontsize=7)
    legend(fig, theme, [('test split, %s draws' % count(len(test)), 'test', 'patch'),
                        ('train split, %s draws' % count(len(train)), 'train', 'patch'),
                        ('pine sapling', 'ink', 'dot')])
    return fig


save(draw_lgcp, 'posteriors_lgcp')

# Single draws of the field from both splits
draws = {}
for split, x in lgcp_sets.items():
    rows = torch.linspace(0, len(x) - 1, args.n_frames).long()
    draws[split] = torch.exp(lgcp.positions(x[rows])).cpu().numpy()
draw_norm = Normalize(0.0, np.quantile(draws['test'], 0.99))


def draw_lgcp_frame(theme, index):
    """Draws one test draw and one train draw of the intensity field with the pines on top"""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
    for ax, split in zip(axes, ['test', 'train']):
        image = ax.imshow(grid(draws[split][index]), origin='lower', extent=(0, 1, 0, 1),
                          cmap=density_map(theme), norm=draw_norm, interpolation='nearest')
        pine_dots(ax, theme, size=7)
        frame_axis(ax, theme, split, '%s split, draw %d of %d' % (split, index + 1,
                                                                  args.n_frames))
    fig.colorbar(image, ax=axes, shrink=0.85, extend='max',
                 label='intensity, saplings per unit area')
    return fig


save_gif(draw_lgcp_frame, args.n_frames, 'posteriors_lgcp_draws', fps=2)
