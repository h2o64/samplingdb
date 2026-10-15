# Implementation of the sliced Kolmogorov-Smirnov distance

# Libraries
import torch
from .weights import normalize_weights

# Projections compared per pass, which bounds the sorted copies to this many times the sample count
PROJECTIONS_PER_PASS = 16


def projected_cdf(projected, mass, points):
    """Evaluates the weighted empirical CDF of every projection at the given points

    Args:
        * projected (torch.Tensor of shape (n_random_projections, n)): Projected samples
        * mass (torch.Tensor of shape (n,)): Weight of every sample
        * points (torch.Tensor of shape (n_random_projections, m)): Evaluation points

    Returns:
        * cdf (torch.Tensor of shape (n_random_projections, m)): The right-continuous CDF
    """
    values, order = torch.sort(projected, dim=-1)
    cumulative = torch.cumsum(mass[order], dim=-1)
    cumulative = torch.cat([torch.zeros_like(cumulative[:, :1]), cumulative], dim=-1)
    return cumulative.gather(-1, torch.searchsorted(values, points, right=True))


def compute_projected_ks(samples1, samples2, random_projs, weights=None):
    """Compute the exact two-sample Kolmogorov-Smirnov distance on every random projection

    Both empirical CDFs are step functions that change only at the pooled sample points, so the
    largest gap between them is attained at one of those points and the distance is exact.

    NOTE: a row with a non-finite coordinate has no position on any projection. Its mass is placed
    at whichever end of the line gives the larger distance, so the distance can only grow when rows
    of a set turn non-finite: half of the rows set to NaN reads at least 0.5.

    Args:
        * samples1 (torch.Tensor of shape (n, dim)): Samples from the first distribution
        * samples2 (torch.Tensor of shape (m, dim)): Samples from the second distribution
        * random_projs (torch.Tensor of shape (n_random_projections, dim)): Unit directions
        * weights (torch.Tensor of shape (m,) or None): Weights of samples2, normalized inside
            (default is None)

    Returns:
        * ks (torch.Tensor of shape (n_random_projections,)): Distance on every projection
    """
    # Mass of every row, in float64 so the cumulative sums reach one, with the non-finite rows
    # taken out. What they held, the first set's minus the second's, is the unplaced mass
    finite1 = torch.isfinite(samples1).all(dim=-1)
    finite2 = torch.isfinite(samples2).all(dim=-1)
    mass1 = normalize_weights(None, n=len(samples1), dtype=torch.float64, device=samples1.device)
    mass2 = normalize_weights(None if weights is None else weights.to(torch.float64),
                              n=len(samples2), dtype=torch.float64, device=samples2.device)
    mass1, mass2 = mass1 * finite1, mass2 * finite2
    unplaced = mass2.sum() - mass1.sum()

    # Project, sending the non-finite rows to +inf, where their zero mass changes no CDF
    projected1 = torch.where(finite1[:, None], samples1, 0.0) @ random_projs.T
    projected2 = torch.where(finite2[:, None], samples2, 0.0) @ random_projs.T
    projected1 = torch.where(finite1[:, None], projected1, float('inf')).T.contiguous()
    projected2 = torch.where(finite2[:, None], projected2, float('inf')).T.contiguous()

    # The gap at every pooled point, with the unplaced mass at either end of the line
    points = torch.cat([projected1, projected2], dim=-1)
    gap = projected_cdf(projected1, mass1, points) - projected_cdf(projected2, mass2, points)
    return torch.maximum(gap.abs(), (gap + unplaced).abs()).max(dim=-1).values


def compute_sliced_ks(samples1, samples2, weights=None, n_random_projections=128, generator=None):
    """Compute the sliced Kolmogorov-Smirnov distance

    The mean over random directions of the exact two-sample KS distance between the projected
    sets (see compute_projected_ks). Sorting makes it O(n log n) per direction and nothing is
    binned, so no mass is dropped wherever the samples sit.

    Args:
        * samples1 (torch.Tensor (batch_size, dim)): Samples from the first distribution
        * samples2 (torch.Tensor (batch_size, dim)): Samples from the second distribution
        * weights (torch.Tensor of shape (batch_size,)): Reweighting of the samples from the
            second distribution (default is None)
        * n_random_projections (int): Number of random projections (default is 128)
        * generator (torch.Generator or None): RNG of the projections (default is None)

    Returns:
        * ks (float): Sliced KS distance
    """
    # The projections are drawn on the CPU with the caller's generator, so a seeded run gives the
    # same directions wherever the samples live
    random_projs = torch.randn((n_random_projections, samples1.shape[-1]), generator=generator,
                               dtype=samples1.dtype).to(samples1.device)
    random_projs /= torch.linalg.norm(random_projs, dim=-1)[..., None]

    # A few projections at a time, which keeps the sorted copies linear in the sample count
    ks = torch.cat([compute_projected_ks(samples1, samples2, block, weights=weights)
                    for block in random_projs.split(PROJECTIONS_PER_PASS)])
    return ks.mean().item()
