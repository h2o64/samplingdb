# Weight normalization, the one thing every weighted metric needs

# Libraries
import torch


def normalize_weights(weights, n=None, dtype=None, device=None):
    """Normalize importance weights, treating None as uniform

    Args:
        * weights (torch.Tensor of shape (n,) or None): Unnormalized weights
        * n (int or None): Number of samples, required when weights is None (default is None)
        * dtype (torch.dtype or None): Output dtype when weights is None (default is None)
        * device (torch.device or None): Output device when weights is None (default is None)

    Returns:
        * weights (torch.Tensor of shape (n,)): Weights summing to one
    """
    if weights is None:
        return torch.full((n,), 1.0 / n, dtype=dtype, device=device)
    return weights / weights.sum()


def subsample(x, n, weights=None, generator=None):
    """Draws n rows of a weighted sample set as an unweighted one, for the quadratic metrics

    Without weights it is a uniformly random subset of n distinct rows, or every row when there are
    no more than n. With weights it is systematic resampling: n evenly spaced levels, shifted by
    one uniform draw, are read off the cumulative weights, so row i is drawn floor(n w_i) or
    floor(n w_i) + 1 times, n w_i in expectation, wherever it sits. A leading slice would score a
    chain's earliest states only, and lose all of the weight that lies past it.

    NOTE: systematic rather than multinomial resampling, so that equal weights over n rows return
    every row once and a weighted panel with uniform weights is the unweighted one. It also has
    the smaller variance of the two.

    Args:
        * x (torch.Tensor of shape (batch_size, *data_shape)): Samples
        * n (int): Number of rows to keep
        * weights (torch.Tensor of shape (batch_size,) or None): Unnormalized weights of the rows
            (default is None)
        * generator (torch.Generator or None): RNG of the draw, on the CPU (default is None)

    Returns:
        * x (torch.Tensor of shape (min(n, batch_size), *data_shape) or (n, *data_shape)):
            Unweighted rows
    """
    if weights is None:
        if len(x) <= n:
            return x
        index = torch.randperm(len(x), generator=generator)[:n]
    else:
        cumulative = torch.cumsum(weights.detach().to(torch.float64).cpu(), dim=0)
        offset = torch.rand((), generator=generator, dtype=torch.float64)
        levels = (offset + torch.arange(n, dtype=torch.float64)) / n * cumulative[-1]
        index = torch.searchsorted(cumulative, levels, right=True).clamp(max=len(x) - 1)
    return x[index.to(x.device)]
