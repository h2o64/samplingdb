# Wasserstein distances: the exact solver, its ground costs, and the sliced approximation

# Libraries
import math
import numpy as np
import torch
import ot as pot
from .weights import normalize_weights


def exact_wasserstein(cost, power=2, weights=None):
    """Solves the exact transport problem for a given cost matrix

    The one place a linear program is solved, whatever the ground cost was. A power of 2 means the
    cost matrix already holds SQUARED distances and the optimum is square-rooted here, which is
    the convention of the molecular reference implementation and the only one under which the
    result is a metric.

    NOTE: the network simplex is CPU-only in POT, so the cost matrix crosses the device boundary
    here. Everything that scales with the number of coordinates -- the cost itself -- stays on the
    device the samples are on.

    Args:
        * cost (torch.Tensor of shape (n, m)): Transport cost
        * power (int): 1 or 2; a power of 2 takes the square root of the optimum (default is 2)
        * weights (torch.Tensor of shape (m,) or None): Weights of the second set, uniform when
            None (default is None)

    Returns:
        * distance (float): The Wasserstein distance
    """
    assert power in [1, 2]
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    if weights is None:
        b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    else:
        b = normalize_weights(weights).detach().cpu().numpy().astype(np.float64)
    value = float(pot.emd2(a, b, cost.detach().cpu().numpy().astype(np.float64),
                           numItermax=int(1e7)))
    return math.sqrt(max(value, 0.0)) if power == 2 else value


def euclidean_cost(x0, x1, power=2):
    """Builds the Euclidean transport cost between two point clouds

    Args:
        * x0 (torch.Tensor of shape (n,...)): First set, flattened past the leading axis
        * x1 (torch.Tensor of shape (m,...)): Second set
        * power (int): 1 or 2 (default is 2)

    Returns:
        * cost (torch.Tensor of shape (n, m)): The cost matrix
    """
    assert power in [1, 2]

    # float64 whatever comes in. torch.cdist expands |x|^2 + |y|^2 - 2 x.y, which cancels
    # catastrophically for near-identical points, and a metric's floor is exactly where both sides
    # ARE near-identical. Measured on GMM40 with 256 samples: W(x, x) is 1.8e-07 from float64
    # samples, 3.9e-03 from float32 ones -- four orders of magnitude of lost resolution, right where
    # the panel needs it -- and 0.0 from float32 samples promoted here. So this is not a reason to
    # sample in float64: it is a reason for the metric to promote. The panel runs once over a
    # reference set; the force field runs 1e9 times, which is where the precision has to be bought.
    exact = torch.float64
    cost = torch.cdist(x0.reshape(len(x0), -1).detach().to(exact),
                       x1.reshape(len(x1), -1).detach().to(exact))
    return cost ** 2 if power == 2 else cost


def torus_cost(x0, x1, power=2):
    """Builds the transport cost between two sets of angle vectors on the torus

    The per-coordinate distance is min(|d|, 2 pi - |d|), raised to the power and summed over
    coordinates. Built on whichever device the inputs live on, which is the O(n m k) part; only
    the linear program itself is CPU-bound.

    Args:
        * x0 (torch.Tensor of shape (n, k)): First set of angles
        * x1 (torch.Tensor of shape (m, k)): Second set of angles
        * power (int): 1 or 2 (default is 2)

    Returns:
        * cost (torch.Tensor of shape (n, m)): The cost matrix
    """
    assert power in [1, 2]
    # float64 for the same reason euclidean_cost promotes, though this form is better conditioned:
    # it differences the coordinates directly rather than expanding a squared norm
    exact = torch.float64
    delta = (x0[:, None, :].to(exact) - x1[None, :, :].to(exact)).abs()
    delta = torch.minimum(delta, 2 * math.pi - delta)
    return (delta ** power).sum(-1)


def compute_wasserstein_distance(x0, x1, weights1=None, method='exact', reg=0.05, power=2):
    """Compute the (regularized) Wasserstein distance between two point clouds

    WARNING: for power 2 the cost is the SQUARED Euclidean distance and the optimum is rooted,
    which is what makes this a W2. Feeding an unsquared cost and rooting afterwards -- the
    convention of several flow-matching codebases -- gives sqrt(W1) under a W2 name; the molecular
    reference implementation squares first, so the whole corpus does.

    Args:
        * x0 (torch.Tensor (batch_size, dim)): Samples from the first distribution
        * x1 (torch.Tensor (batch_size, dim)): Samples from the second distribution
        * weights1 (torch.Tensor (batch_size,)): Normalized weights of the second distribution
            (default is None)
        * method (str): OT solver (either 'exact' or 'sinkhorn') (default is 'exact')
        * reg (float): Regularization factor (default is 0.05)
        * power (int): Power for the distance (default is 2)

    Returns:
        * w2 (float): (regularized) Wasserstein distance
    """
    assert method in ['exact', 'sinkhorn']
    cost = euclidean_cost(x0, x1, power=power)
    if method == 'exact':
        return exact_wasserstein(cost, power=power, weights=weights1)

    # The entropic solver has no numpy-side wrapper of its own, so it is run here on device
    a = torch.full((cost.shape[0],), 1.0 / cost.shape[0], device=cost.device, dtype=cost.dtype)
    b = (torch.full((cost.shape[1],), 1.0 / cost.shape[1], device=cost.device, dtype=cost.dtype)
         if weights1 is None else normalize_weights(weights1).to(cost))
    value = float(pot.sinkhorn2(a, b, cost, reg=reg, method='sinkhorn_log'))
    return math.sqrt(max(value, 0.0)) if power == 2 else value


def torus_wasserstein(x0, x1, power=2, weights=None):
    """Wasserstein distance between two sets of angles, on the torus

    Args:
        * x0 (torch.Tensor of shape (n, k)): Reference angles
        * x1 (torch.Tensor of shape (m, k)): Angles to score
        * power (int): 1 or 2 (default is 2)
        * weights (torch.Tensor of shape (m,) or None): Weights of x1 (default is None)

    Returns:
        * distance (float): The torus Wasserstein distance
    """
    return exact_wasserstein(torus_cost(x0, x1, power=power), power=power, weights=weights)


def compute_sliced_wasserstein_distance(x0, x1, weights1=None, n_projections=128, power=2,
                                        generator=None):
    """Compute the sliced Wasserstein distance

    Args:
        * x0 (torch.Tensor (batch_size, dim)): Samples from the first distribution
        * x1 (torch.Tensor (batch_size, dim)): Samples from the second distribution
        * weights1 (torch.Tensor (batch_size,)): Normalized weights of the second distribution
            (default is None)
        * n_projections (int): Number of random 1D projections (default is 128)
        * power (int): Power for the distance (default is 2)
        * generator (torch.Generator or None): RNG of the projections (default is None)

    Returns:
        * w2_sliced (float): Sliced Wasserstein distance
    """
    seed = None if generator is None else int(torch.randint(2 ** 31 - 1, (1,), generator=generator))
    value = pot.sliced_wasserstein_distance(
        x0.detach(), x1.detach(), b=None if weights1 is None else weights1.to(x1),
        n_projections=n_projections, p=power, seed=seed,
    )
    return float(value)


def wasserstein_distance_1d(u_values, v_values, u_weights=None, v_weights=None, power=1):
    """Compute the 1D Wasserstein distance between two 1D distributions

    Both quantile functions are step functions that only change at the pooled cumulative-weight
    levels, so the transport cost is the exact sum of |F_u^-1 - F_v^-1|^power over those segments.
    This is POT's emd2_1d without leaving the device, and it agrees with scipy to 1e-12 for
    power=1. islips' version instead interpolates the CDFs linearly and pairs each interval with
    the CDF value at its far end, which carries an O(1/n) bias -- 0.5745 against the exact 0.5707
    on 500 versus 400 standard normal draws. The metric feeds every panel in the corpus, so the
    bias was not worth keeping for the sake of a byte-identical port.

    Args:
        * u_values (torch.Tensor of shape (n,)): Samples from the first distribution
        * v_values (torch.Tensor of shape (m,)): Samples from the second distribution
        * u_weights (torch.Tensor of shape (n,)): Weights of u (default is None)
        * v_weights (torch.Tensor of shape (m,)): Weights of v (default is None)
        * power (int): 1 for W1, 2 for W2 (default is 1)

    Returns:
        * distance (torch.Tensor of shape ()): Wasserstein distance
    """
    assert power in [1, 2]
    u_values = u_values.flatten()
    v_values = v_values.flatten()

    if u_weights is None:
        u_weights = torch.ones_like(u_values) / len(u_values)
    else:
        u_weights = u_weights.flatten().to(u_values) / u_weights.sum()
    if v_weights is None:
        v_weights = torch.ones_like(v_values) / len(v_values)
    else:
        v_weights = v_weights.flatten().to(v_values) / v_weights.sum()

    # Sort both distributions and accumulate their weights
    u_sorted, u_indices = torch.sort(u_values)
    v_sorted, v_indices = torch.sort(v_values)
    u_cdf = torch.cumsum(u_weights[u_indices], dim=0)
    v_cdf = torch.cumsum(v_weights[v_indices], dim=0)

    # Cut [0, 1] at every level either CDF reaches, where both quantile functions are constant
    levels, _ = torch.sort(torch.cat([u_cdf, v_cdf]))
    widths = torch.diff(levels, prepend=torch.zeros_like(levels[:1]))
    u_index = torch.searchsorted(u_cdf.contiguous(), levels.contiguous()).clamp(max=len(u_cdf) - 1)
    v_index = torch.searchsorted(v_cdf.contiguous(), levels.contiguous()).clamp(max=len(v_cdf) - 1)

    # Integrate the gap between the quantile functions over those segments
    gap = (u_sorted[u_index] - v_sorted[v_index]).abs()
    if power == 2:
        return torch.sqrt(torch.sum(widths * gap ** 2))
    return torch.sum(widths * gap)
