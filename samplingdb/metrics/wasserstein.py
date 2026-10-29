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

    # float64 whatever comes in: torch.cdist expands |x|^2 + |y|^2 - 2 x.y, which cancels
    # catastrophically for the near-identical points that set a metric's floor, so float32 samples
    # are promoted here
    exact = torch.float64
    cost = torch.cdist(x0.reshape(len(x0), -1).detach().to(exact),
                       x1.reshape(len(x1), -1).detach().to(exact))
    return cost ** 2 if power == 2 else cost


def torus_cost(x0, x1, power=2):
    """Builds the transport cost between two sets of angle vectors on the torus

    The per-coordinate distance is min(d, 2 pi - d), with d the difference reduced modulo 2 pi so
    the angles need not lie in one period, raised to the power and summed over coordinates. Built
    on whichever device the inputs live on, which is the O(n m k) part; only the linear program
    itself is CPU-bound.

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
    delta = torch.remainder(x0[:, None, :].to(exact) - x1[None, :, :].to(exact), 2 * math.pi)
    delta = torch.minimum(delta, 2 * math.pi - delta)
    return (delta ** power).sum(-1)


def compute_wasserstein_distance(x0, x1, weights1=None, power=2):
    """Compute the Wasserstein distance between two point clouds, exactly

    WARNING: for power 2 the cost is the SQUARED Euclidean distance and the optimum is rooted,
    which is what makes this a W2. Feeding an unsquared cost and rooting afterwards -- the
    convention of several flow-matching codebases -- gives sqrt(W1) under a W2 name; the molecular
    reference implementation squares first, so the whole corpus does.

    Args:
        * x0 (torch.Tensor (batch_size, dim)): Samples from the first distribution
        * x1 (torch.Tensor (batch_size, dim)): Samples from the second distribution
        * weights1 (torch.Tensor (batch_size,)): Weights of the second distribution, normalized
            inside (default is None)
        * power (int): Power for the distance (default is 2)

    Returns:
        * w2 (float): Wasserstein distance
    """
    return exact_wasserstein(euclidean_cost(x0, x1, power=power), power=power, weights=weights1)


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
        * weights1 (torch.Tensor (batch_size,)): Weights of the second distribution, normalized
            inside (default is None)
        * n_projections (int): Number of random 1D projections (default is 128)
        * power (int): Power for the distance (default is 2)
        * generator (torch.Generator or None): RNG of the projections (default is None)

    Returns:
        * w2_sliced (float): Sliced Wasserstein distance
    """
    seed = None if generator is None else int(torch.randint(2 ** 31 - 1, (1,), generator=generator))
    b = None if weights1 is None else normalize_weights(weights1).to(x1)
    value = pot.sliced_wasserstein_distance(x0.detach(), x1.detach(), b=b,
                                            n_projections=n_projections, p=power, seed=seed)
    return float(value)


def wasserstein_distance_1d(u_values, v_values, u_weights=None, v_weights=None, power=1):
    """Compute the 1D Wasserstein distance between two 1D distributions

    Both quantile functions are step functions that only change at the pooled cumulative-weight
    levels, so the transport cost is the exact sum of |F_u^-1 - F_v^-1|^power over those segments.
    This is POT's emd2_1d without leaving the device, and it agrees with scipy to 1e-12 for
    power=1. It is exact where a linear interpolation of the CDFs carries an O(1/n) bias.

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


def circular_wasserstein_1d(u_values, v_values, u_weights=None, v_weights=None,
                            period=2 * math.pi):
    """Compute the 1D Wasserstein-1 distance between two distributions on a circle

    On the circle the transport cost is the minimum over a rotation c of the integral of
    |F_u - F_v - c|, which the median of F_u - F_v attains (Rabin, Delon and Gousseau 2011). Both
    CDFs are step functions, so the integral is an exact sum over the pooled sample points. A
    linear W1 on angles treats -pi and pi as the two ends of a segment, which inflates the
    distance of a torsion that sits on that cut, such as a backbone omega.

    Args:
        * u_values (torch.Tensor of shape (n,)): Angles of the first distribution, in radians
        * v_values (torch.Tensor of shape (m,)): Angles of the second distribution, in radians
        * u_weights (torch.Tensor of shape (n,)): Weights of u (default is None)
        * v_weights (torch.Tensor of shape (m,)): Weights of v (default is None)
        * period (float): Length of the circle (default is 2 pi)

    Returns:
        * distance (torch.Tensor of shape ()): Circular Wasserstein-1 distance
    """
    u_values = torch.remainder(u_values.flatten(), period)
    v_values = torch.remainder(v_values.flatten(), period)
    u_weights = torch.ones_like(u_values) / len(u_values) if u_weights is None \
        else u_weights.flatten().to(u_values) / u_weights.sum()
    v_weights = torch.ones_like(v_values) / len(v_values) if v_weights is None \
        else v_weights.flatten().to(v_values) / v_weights.sum()

    # F_u - F_v is constant between consecutive pooled points, and zero before the first and after
    # the last, which on the circle is one segment
    values = torch.cat([u_values, v_values])
    jumps = torch.cat([u_weights, -v_weights])
    order = torch.argsort(values)
    values, difference = values[order], torch.cumsum(jumps[order], dim=0)
    lengths = torch.cat([torch.diff(values), (values[:1] + period - values[-1:])])
    difference = torch.cat([difference[:-1], torch.zeros_like(difference[:1])])

    # The best rotation is the length-weighted median of the difference
    sorted_difference, index = torch.sort(difference)
    cumulative = torch.cumsum(lengths[index], dim=0)
    median = sorted_difference[torch.searchsorted(cumulative, cumulative[-1:] / 2).clamp(
        max=len(cumulative) - 1)]
    return torch.sum(lengths * (difference - median).abs())
