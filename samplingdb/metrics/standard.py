# The universal metric panel, and the split between its linear and quadratic halves

# Libraries
from .ks import compute_sliced_ks
from .mmd import compute_mmd
from .wasserstein import compute_sliced_wasserstein_distance, compute_wasserstein_distance
from .weights import normalize_weights

# Samples both sides of a quadratic metric are truncated to when one is asked for
MAX_COSTLY_SAMPLES = 2000


def compute_standard_metrics(samples, ref_samples, weights=None, skip_costly_metrics=True,
                             max_costly_samples=MAX_COSTLY_SAMPLES, generator=None):
    """Compute the target-agnostic metrics between two coordinate-space sample sets

    The panel has two cost classes, and skip_costly_metrics selects on exactly that:

    * LINEAR in the sample count -- the sliced KS distance and the sliced Wasserstein distance,
      both of which project onto random lines and sort. Always computed.
    * QUADRATIC in the sample count -- the median-bandwidth MMD, whose three kernel matrices are
      each n x n, and the exact and regularized transport distances, which are O(n^2) in memory
      and O(n^3) in time. Skipped by default, and truncated to max_costly_samples when asked for.

    WARNING: the MMD belongs to the second class and used to sit in the first, which made the
    default panel allocate 12 GB at n = 12000 and made it unusable at the catalog's default split
    size of 100000, where its three matrices alone would be 240 GB.

    NOTE: ref_samples is always the first argument of every underlying metric, so any range or
    bandwidth heuristic is set by the ground truth rather than by the samples being scored.

    Args:
        * samples (torch.Tensor of shape (batch_size, dim)): Samples to score
        * ref_samples (torch.Tensor of shape (n_ref, dim)): Reference samples
        * weights (torch.Tensor of shape (batch_size,)): Normalized weights of samples
            (default is None)
        * skip_costly_metrics (bool): Whether to skip the quadratic metrics (default is True)
        * max_costly_samples (int): Samples both sides of a quadratic metric are cut to
            (default is 2000)
        * generator (torch.Generator or None): RNG of the random projections (default is None)

    Returns:
        * metrics (dict): 'ks' and 'w2_sliced', plus 'mmd', 'w2', 'w2_reg' and the realised
            'n_costly_samples' when the quadratic metrics are not skipped
    """
    ret = {}
    ret['ks'] = compute_sliced_ks(ref_samples, samples, weights=weights, generator=generator)
    ret['w2_sliced'] = compute_sliced_wasserstein_distance(ref_samples, samples, weights1=weights,
                                                           generator=generator)
    if skip_costly_metrics:
        return ret

    # Both sides cut to the same count, which is reported rather than left implicit. The weights
    # of the kept samples are renormalised, since a transport marginal has to sum to one
    n = min(len(samples), len(ref_samples), max_costly_samples)
    cut, ref_cut = samples[:n], ref_samples[:n]
    cut_weights = None if weights is None else normalize_weights(weights[:n])
    ret['mmd'] = compute_mmd(ref_cut, cut, weights=cut_weights)
    ret['w2'] = compute_wasserstein_distance(ref_cut, cut, weights1=cut_weights)
    ret['w2_reg'] = compute_wasserstein_distance(ref_cut, cut, weights1=cut_weights,
                                                 method='sinkhorn')
    ret['n_costly_samples'] = float(n)
    return ret
