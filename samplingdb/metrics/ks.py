# Implementation of the sliced Kolmogorov-Smirnov distance

# Libraries
import torch
from .weights import normalize_weights


def compute_random_proj_cdf(samples, random_projs, n_bins, min_x=None, max_x=None, weights=None,
                            return_min_max=False):
    """Compute the CDF of randomly 1D-projected samples

    Every projection is binned in one scatter rather than one torch.histogram call each, which is
    both vectorized and device-agnostic -- torch.histogram is CPU-only.

    Args:
        * samples (torch.Tensor of shape (batch_size, dim)): Samples to project
        * random_projs (torch.Tensor of shape (n_random_projections, dim)): Random 1D projections
        * n_bins (int): Number of bins for histogram computations
        * min_x, max_x (torch.Tensor of shape (n_random_projections,)): Ranges for histogram
            computation (default is None)
        * weights (torch.Tensor of shape (batch_size,)): Weights for histogram computation
            (default is None)
        * return_min_max (bool): Whether to return the ranges of histogram computations

    Returns:
        * random_proj_cdf (torch.Tensor of shape (n_random_projections, n_bins)): CDF of the
            projected samples
        if return_min_max:
            * min_x, max_x (torch.Tensor of shape (n_random_projections,)): Histogram ranges
    """
    n_projections = random_projs.shape[0]
    projected = samples @ random_projs.T
    if min_x is None or max_x is None:
        min_x, max_x = projected.min(dim=0).values, projected.max(dim=0).values

    # One bin index per (sample, projection), offset by the projection so a single scatter fills
    # every histogram at once
    width = (max_x - min_x).clamp_min(torch.finfo(samples.dtype).tiny) / n_bins
    index = ((projected - min_x) / width).floor().to(torch.int64)
    inside = (index >= 0) & (index < n_bins)
    index = index.clamp(0, n_bins - 1) + n_bins * torch.arange(n_projections, device=samples.device)

    mass = normalize_weights(weights, n=len(samples), dtype=samples.dtype,
                             device=samples.device).to(samples.dtype)
    mass = mass[:, None].expand_as(projected) * inside
    hist = torch.zeros(n_projections * n_bins, dtype=samples.dtype, device=samples.device)
    hist.scatter_add_(0, index.reshape(-1), mass.reshape(-1))
    hist = hist.reshape(n_projections, n_bins)

    random_proj_cdf = (hist / hist.sum(dim=-1, keepdim=True).clamp_min(1e-30)).cumsum(dim=-1)
    if return_min_max:
        return random_proj_cdf, min_x, max_x
    return random_proj_cdf


def compute_sliced_ks(samples1, samples2, weights=None, n_random_projections=128, n_bins=256,
                      generator=None):
    """Compute the sliced Kolmogorov-Smirnov distance

    WARNING: the histogram range is set by samples1, the reference side, so mass that samples2
    puts outside that range is dropped before the CDFs are normalized. Pass the ground-truth
    samples as samples1, and read a suspiciously small value together with the mass diagnostics
    of metrics.histogram.

    Args:
        * samples1 (torch.Tensor (batch_size, dim)): Samples from the first distribution
        * samples2 (torch.Tensor (batch_size, dim)): Samples from the second distribution
        * weights (torch.Tensor of shape (batch_size,)): Reweighting of the samples from the
            second distribution (default is None)
        * n_random_projections (int): Number of random projections (default is 128)
        * n_bins (int): Number of bins in the histogram computation (default is 256)
        * generator (torch.Generator or None): RNG of the projections (default is None)

    Returns:
        * ks (float): Approximation of the sliced KS distance
    """
    # The projections are drawn on the CPU with the caller's generator, so a seeded run gives the
    # same directions wherever the samples live
    random_projs = torch.randn((n_random_projections, samples1.shape[-1]), generator=generator,
                               dtype=samples1.dtype).to(samples1.device)
    random_projs /= torch.linalg.norm(random_projs, dim=-1)[..., None]

    samples1_cdf, min_x, max_x = compute_random_proj_cdf(samples1, random_projs, n_bins=n_bins,
                                                        return_min_max=True)
    samples2_cdf = compute_random_proj_cdf(samples2, random_projs, min_x=min_x, max_x=max_x,
                                          weights=weights, n_bins=n_bins)
    return torch.max(torch.abs(samples1_cdf - samples2_cdf), dim=-1).values.mean().item()
