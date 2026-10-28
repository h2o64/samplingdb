# Draw the figure of the metrics page: what the sliced Kolmogorov-Smirnov distance compares

# Libraries
import argparse, pprint, sys, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from samplingdb import build
from samplingdb.metrics.ks import compute_projected_ks
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import legend, note, save

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=2048)
parser.add_argument('--n_random_projections', type=int, default=128)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# TwoModes, whose train split has the two weights wrong: the test split is the reference side
target = build('twomodes')
test = target.test_samples(n=args.n_samples)
train = target.train_samples(n=args.n_samples)
metrics = target.compute_metrics(train)
print('ks of the train split', metrics['ks'])

# The 128 unit directions drawn the way compute_sliced_ks draws them, and the distance on each
gen = torch.Generator().manual_seed(0)
directions = torch.randn((args.n_random_projections, 2), generator=gen, dtype=torch.float64)
directions /= torch.linalg.norm(directions, dim=-1)[:, None]
per_direction = compute_projected_ks(train, test, directions).numpy()
print('mean over these directions', per_direction.mean())

# The direction drawn in detail: the one where the two sets differ most
best = int(np.argmax(per_direction))
direction = directions[best].numpy()
projected = {'test': (test.numpy() @ direction), 'train': (train.numpy() @ direction)}
grid = np.sort(np.concatenate(list(projected.values())))
cdfs = {split: np.searchsorted(np.sort(values), grid, side='right') / len(values)
        for split, values in projected.items()}
gap = np.abs(cdfs['test'] - cdfs['train'])
where = int(np.argmax(gap))


def draw_sliced_ks(theme):
    """Draws two sample sets, one projection of both, and the distance over every projection"""
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), gridspec_kw={'width_ratios': [1, 1.25, 1.1]})

    # The samples and three of the directions, the detailed one in ink
    ax = axes[0]
    for split in ['train', 'test']:
        values = test if split == 'test' else train
        ax.scatter(values[:, 0], values[:, 1], s=1.5, color=theme[split], alpha=0.5,
                   linewidths=0, rasterized=True)
    for index in [(best + 1) % len(directions), (best + 2) % len(directions)]:
        u = directions[index].numpy()
        ax.plot([-2.2 * u[0], 2.2 * u[0]], [-2.2 * u[1], 2.2 * u[1]], color=theme['muted'],
                linewidth=0.7)
    ax.plot([-2.2 * direction[0], 2.2 * direction[0]], [-2.2 * direction[1], 2.2 * direction[1]],
            color=theme['ink'], linewidth=1.2)
    ax.annotate('', xy=(1.6 * direction[0], 1.6 * direction[1]), xytext=(0.0, 0.0),
                arrowprops={'arrowstyle': '-|>', 'color': theme['ink'], 'linewidth': 1.2})
    ax.text(1.75 * direction[0] + 0.12, 1.75 * direction[1], '$u$', color=theme['ink'], fontsize=10)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.set_title('samples and directions')

    # The two empirical CDFs along the detailed direction, and their largest gap
    ax = axes[1]
    for split in ['test', 'train']:
        ax.step(grid, cdfs[split], where='post', color=theme[split], linewidth=1.3)
    ax.vlines(grid[where], cdfs['test'][where], cdfs['train'][where], color=theme['ink'],
              linewidth=1.5)
    ax.annotate('KS = %.3f' % gap[where], xy=(grid[where], 0.5 * (cdfs['test'][where]
                                                                 + cdfs['train'][where])),
                xytext=(10, -4), textcoords='offset points', color=theme['ink'], fontsize=8)
    ax.set_xlabel('projection on $u$')
    ax.set_ylabel('empirical CDF')
    ax.set_title('one projection, $u$')
    ax.yaxis.grid(True, color=theme['grid'], linewidth=0.5)
    ax.set_axisbelow(True)

    # The distance on every direction, against the angle of the direction
    ax = axes[2]
    angles = np.degrees(np.arctan2(directions[:, 1].numpy(), directions[:, 0].numpy())) % 180.0
    ax.scatter(angles, per_direction, s=8, color=theme['ink'], linewidths=0)
    ax.axhline(per_direction.mean(), color=theme['train'], linewidth=1.3)
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel('direction angle (degrees)')
    ax.set_ylabel('KS distance')
    ax.set_title('all %d projections' % len(directions))
    ax.yaxis.grid(True, color=theme['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    note(ax, ['ks %.3f' % metrics['ks']], theme, loc='lower left')
    legend(fig, theme, [('test split', 'test', 'dot'), ('train split', 'train', 'dot'),
                        ('ks, the mean over projections', 'train', 'line')])
    return fig


save(draw_sliced_ks, 'metrics_sliced_ks')
