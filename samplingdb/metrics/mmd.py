# Implement the Maximum-Mean-Discrepancy (MMD) with the median bandwidth heuristic
# Based on DenisBless/variational_sampling_methods' mmd_median and ignite's MaximumMeanDiscrepancy

# Libraries
import torch
from .weights import normalize_weights


def pairwise_sq_distances(a, b=None):
    """Evaluates the squared Euclidean distances between two sets, and their squared norms

    Args:
        * a (torch.Tensor of shape (n, dim)): First set
        * b (torch.Tensor of shape (m, dim) or None): Second set; None compares a with itself
            (default is None)

    Returns:
        * distances (torch.Tensor of shape (n, m)): Squared distances
        * norms (torch.Tensor of shape (n,)): Squared norms of a's rows
    """
    norms_a = (a ** 2).sum(-1)
    if b is None:
        return norms_a[:, None] + norms_a[None, :] - 2.0 * (a @ a.T), norms_a
    return norms_a[:, None] + (b ** 2).sum(-1)[None, :] - 2.0 * (a @ b.T), norms_a


def gauss_kernel(pairwise_sq_matrix, bandwidth_sq):
    """Evaluates the RBF kernel of a squared-distance matrix at the given squared bandwidth"""
    return torch.exp(-pairwise_sq_matrix / bandwidth_sq / 2)


def compute_mmd(X, Y, weights=None):
    """Compute the Maximum-Mean-Discrepancy distance using the median bandwidth heuristic

    NOTE: with weights=None this is the unbiased U-statistic. With weights it is the weighted
    V-statistic, which is a different (biased) estimator -- the two are not directly comparable.

    NOTE: the median bandwidth is taken over the rows of Y with a positive weight only, so rows
    the weights discard change neither the kernel nor the value.

    WARNING: this is QUADRATIC in the sample count, in time and in memory, which is why the
    standard panel treats it as a costly metric and subsamples both sides before calling it.

    Args:
        * X (torch.Tensor (n, dim)): Samples from the first distribution
        * Y (torch.Tensor (m, dim)): Samples from the second distribution
        * weights (torch.Tensor of shape (m,)): Weights of Y, normalized inside (default is None)

    Returns:
        * mmd (float): MMD distance
    """
    n, m = X.shape[0], Y.shape[0]
    assert n >= 2 and m >= 2
    assert n == m or weights is not None

    d_sq_XX, _ = pairwise_sq_distances(X)
    d_sq_YY, _ = pairwise_sq_distances(Y)
    d_sq_XY, _ = pairwise_sq_distances(X, Y)

    # The bandwidth is the median over the two upper triangles and the whole cross block, over the
    # rows of Y that carry weight only. Taking the median of the masked matrices avoids
    # materialising two triu index tensors, which at the panel's cap would themselves be the
    # largest allocation of the call
    kept = torch.ones(m, dtype=torch.bool, device=Y.device) if weights is None else weights > 0
    upper_XX = d_sq_XX[torch.triu(torch.ones_like(d_sq_XX, dtype=torch.bool), diagonal=1)]
    upper_YY = d_sq_YY[torch.triu(kept[:, None] & kept[None, :], diagonal=1)]
    bandwidth_sq = torch.median(torch.cat([upper_XX, upper_YY, d_sq_XY[:, kept].reshape(-1)]))

    K_XX = gauss_kernel(d_sq_XX, bandwidth_sq)
    K_YY = gauss_kernel(d_sq_YY, bandwidth_sq)
    K_XY = gauss_kernel(d_sq_XY, bandwidth_sq)

    if weights is None:
        # The unbiased U-statistic drops the diagonal of each self-kernel, whose n and m terms are
        # exactly one each
        mmd = (K_XX.sum() - n) / (n * (n - 1))
        mmd = mmd + (K_YY.sum() - m) / (m * (m - 1))
        mmd = mmd - 2.0 * K_XY.mean()
    else:
        a = torch.full((n,), 1.0 / n, dtype=X.dtype, device=X.device)
        b = normalize_weights(weights).to(X)
        mmd = a @ K_XX @ a + b @ K_YY @ b - 2.0 * a @ K_XY @ b

    # A U-statistic can be slightly negative on identical samples, so the root is floored
    return float(torch.sqrt(torch.clamp(mmd, min=1e-20)))
