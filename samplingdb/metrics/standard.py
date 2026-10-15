# The universal metric panel, and the split between its linear and quadratic halves

# Libraries
from .ks import compute_sliced_ks
from .mmd import compute_mmd
from .wasserstein import compute_sliced_wasserstein_distance, compute_wasserstein_distance
from .weights import subsample

# Samples both sides of a quadratic metric are drawn down to when one is asked for
MAX_COSTLY_SAMPLES = 2000


def compute_standard_metrics(samples, ref_samples, weights=None, skip_costly_metrics=True,
                             max_costly_samples=MAX_COSTLY_SAMPLES, generator=None):
    """Compute the target-agnostic metrics between two coordinate-space sample sets

    The panel has two cost classes, and skip_costly_metrics selects on exactly that:

    * LINEAR in the sample count, up to a sort -- the sliced KS distance and the sliced
      Wasserstein distance, both of which project onto random lines and sort. Always computed.
    * QUADRATIC in the sample count -- the median-bandwidth MMD, whose three kernel matrices are
      each n x n, and the exact transport distance, which is O(n^2) in memory and O(n^3) in time.
      Skipped by default, and computed on n = min(len(samples), len(ref_samples),
      max_costly_samples) rows per side when asked for.

    Those n rows are drawn with the generator, never sliced off the front, so a chain is scored
    on all of its states rather than on its earliest ones. The reference side is a uniform subset
    without replacement. The scored side is one too when it is unweighted; weighted, it is n rows
    drawn in proportion to the weights by systematic resampling (see metrics.weights.subsample),
    so both quadratic metrics see two unweighted sets and the MMD is its U-statistic.

    NOTE: ref_samples is always the first argument of every underlying metric, so any bandwidth
    heuristic is set by the ground truth rather than by the samples being scored.

    Args:
        * samples (torch.Tensor of shape (batch_size, dim)): Samples to score
        * ref_samples (torch.Tensor of shape (n_ref, dim)): Reference samples
        * weights (torch.Tensor of shape (batch_size,)): Weights of the samples, normalized inside
            (default is None)
        * skip_costly_metrics (bool): Whether to skip the quadratic metrics (default is True)
        * max_costly_samples (int): Rows per side a quadratic metric is computed on at most
            (default is 2000)
        * generator (torch.Generator or None): RNG of the random projections and of the rows the
            quadratic metrics are computed on, on the CPU (default is None)

    Returns:
        * metrics (dict): 'ks' and 'w2_sliced', plus 'mmd', 'w2' and the realised
            'n_costly_samples' when the quadratic metrics are not skipped
    """
    ret = {}
    ret['ks'] = compute_sliced_ks(ref_samples, samples, weights=weights, generator=generator)
    ret['w2_sliced'] = compute_sliced_wasserstein_distance(ref_samples, samples, weights1=weights,
                                                           generator=generator)
    if skip_costly_metrics:
        return ret

    # Both sides drawn down to the same count, which is reported rather than left implicit
    n = min(len(samples), len(ref_samples), max_costly_samples)
    drawn = subsample(samples, n, weights=weights, generator=generator)
    ref_drawn = subsample(ref_samples, n, generator=generator)
    ret['mmd'] = compute_mmd(ref_drawn, drawn)
    ret['w2'] = compute_wasserstein_distance(ref_drawn, drawn)
    ret['n_costly_samples'] = float(n)
    return ret
