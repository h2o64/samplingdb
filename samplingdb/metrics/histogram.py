# Weighted histograms on shared bins, and the distances between them

# Libraries
import math
import torch
from .weights import normalize_weights

# Floor added to both histograms before any logarithm. It is also the smoothing the reference
# molecular implementation applies before its Jensen-Shannon divergence
EPS = 1e-10


def reference_edges(ref_values, n_bins=200, value_range=None):
    """Build bin edges spanning the REFERENCE values only

    WARNING: this is the protocol of jarridrb/DEM's interatomic-distance TVD -- n_bins
    equal-width bins spanning the reference values' own min and max. It is the convention the
    published numbers use, and it has a real consequence: whatever the compared samples put
    outside that range is dropped, then the survivors are renormalized, so the TVD understates
    the discrepancy. weighted_histogram() reports that dropped mass rather than hiding it.

    Args:
        * ref_values (torch.Tensor): Reference values, of any shape
        * n_bins (int): Number of bins (default is 200)
        * value_range (tuple of two floats or None): Explicit range, overriding the reference
            values' min and max (default is None)

    Returns:
        * edges (torch.Tensor of shape (n_bins + 1,)): Bin edges
    """
    if value_range is None:
        low, high = float(ref_values.min()), float(ref_values.max())
    else:
        low, high = value_range
    return torch.linspace(low, high, n_bins + 1, dtype=ref_values.dtype, device=ref_values.device)


def shared_edges(ref_values, values, n_bins=100, value_range=None):
    """Build bin edges spanning BOTH sets, so nothing is ever dropped

    This is the convention the molecular panel's plots are drawn with, and the reason it needs a
    second edge builder: reference_edges deliberately clips to the reference support to reproduce
    a published number, while these edges cover everything both sides contain.

    Args:
        * ref_values (torch.Tensor): Reference values, of any shape
        * values (torch.Tensor): Values to score, of any shape
        * n_bins (int): Number of bins (default is 100)
        * value_range (tuple of two floats or None): Explicit range (default is None)

    Returns:
        * edges (torch.Tensor of shape (n_bins + 1,)): Bin edges
    """
    if value_range is None:
        low = min(float(ref_values.min()), float(values.min()))
        high = max(float(ref_values.max()), float(values.max()))
    else:
        low, high = value_range
    return torch.linspace(low, high, n_bins + 1, dtype=ref_values.dtype, device=ref_values.device)


def bin_indexes(values, edges):
    """Returns the bin of every value, negative or n_bins outside, with the LAST bin closed

    numpy.histogram closes the final bin so the largest value lands inside it, and both edge
    builders put an edge exactly at that value, so leaving it half-open drops a sample from every
    panel. Everything genuinely outside the range keeps an out-of-range index for the caller.

    Args:
        * values (torch.Tensor): Values to bin, of any shape
        * edges (torch.Tensor of shape (n_bins + 1,)): Bin edges

    Returns:
        * index (torch.Tensor of the same shape as values): Bin index of every value
    """
    n_bins = len(edges) - 1
    index = torch.bucketize(values.contiguous(), edges, right=True) - 1
    return torch.where(values == edges[-1], torch.full_like(index, n_bins - 1), index)


def weighted_histogram(values, edges, weights=None, return_out_of_range=False):
    """Histogram of values on fixed edges, normalized to sum to one

    Bins are left-closed and the last one is closed on both sides, matching numpy.histogram.

    Args:
        * values (torch.Tensor of shape (n,) or (n, k)): Values to bin; when 2D, every row
            carries the weight of its sample
        * edges (torch.Tensor of shape (n_bins + 1,)): Bin edges
        * weights (torch.Tensor of shape (n,)): Per-sample weights, not necessarily normalized
            (default is None)
        * return_out_of_range (bool): Whether to also return the fraction of the mass that fell
            outside the edges (default is False)

    Returns:
        * hist (torch.Tensor of shape (n_bins,)): Normalized histogram, summing to one
        if return_out_of_range:
            * out_of_range (float): Fraction of the mass outside the edges
    """
    n_bins = len(edges) - 1
    values_2d = values if values.dim() > 1 else values[:, None]

    # Per-sample weight, broadcast over the entries of each row. The mass is float64 whatever the
    # values are, as in euclidean_cost: accumulated in float32 it sums to 1 - 6e-8 over a DW-4
    # panel, and out_of_range then reports dropped mass where none was dropped
    if weights is None:
        weight_2d = torch.full_like(values_2d, 1.0 / values_2d.numel(), dtype=torch.float64)
    else:
        weight_2d = normalize_weights(weights.to(device=values_2d.device, dtype=torch.float64))
        weight_2d = weight_2d[:, None].expand_as(values_2d) / values_2d.shape[1]

    index = bin_indexes(values_2d, edges)
    inside = (index >= 0) & (index < n_bins)

    # Accumulate
    hist = torch.zeros(n_bins, dtype=weight_2d.dtype, device=weight_2d.device)
    hist.scatter_add_(0, index[inside].reshape(-1), weight_2d[inside].reshape(-1))
    out_of_range = max(1.0 - float(hist.sum()), 0.0)
    hist = hist / hist.sum().clamp_min(EPS)

    if return_out_of_range:
        return hist, out_of_range
    return hist


def histogram_2d(values, edges, weights=None):
    """A weighted 2D histogram of a pair of coordinates on shared edges

    Args:
        * values (torch.Tensor of shape (n, 2)): The two coordinates
        * edges (torch.Tensor of shape (n_bins + 1,)): Shared bin edges
        * weights (torch.Tensor of shape (n,) or None): Per-sample weights (default is None)

    Returns:
        * hist (torch.Tensor of shape (n_bins, n_bins)): The histogram, summing to one when every
            point is inside the edges and to the fraction that is otherwise
    """
    n_bins = len(edges) - 1
    index = bin_indexes(values, edges)
    inside = ((index >= 0) & (index < n_bins)).all(dim=-1)
    flat = index[inside, 0] * n_bins + index[inside, 1]
    if weights is not None:
        weights = weights.to(device=values.device, dtype=torch.float64)
    mass = normalize_weights(weights, n=len(values), dtype=torch.float64, device=values.device)
    hist = torch.zeros(n_bins * n_bins, dtype=torch.float64, device=values.device)
    hist.scatter_add_(0, flat, mass[inside])
    return hist.reshape(n_bins, n_bins)


def histogram_tvd(hist, ref_hist):
    """Total variation distance between two normalized histograms, in [0, 1]"""
    return 0.5 * float((hist - ref_hist).abs().sum())


def histogram_kl(hist, ref_hist):
    """KL(ref || hist) between two normalized histograms, in nats, with a floor on both"""
    hist, ref_hist = hist + EPS, ref_hist + EPS
    return float((ref_hist * torch.log(ref_hist / hist)).sum())


def jensen_shannon(hist, ref_hist):
    """Squared Jensen-Shannon distance in base 2 between two histograms, in [0, 1]

    This is scipy's jensenshannon(..., base=2) ** 2, which is the convention the molecular
    reference implementation reports every one of its divergences in, so the whole corpus uses it
    rather than carrying two units for one quantity. Both sides are smoothed by EPS first, as
    upstream does, which is what keeps an empty reference cluster finite.

    Args:
        * hist, ref_hist (torch.Tensor of shape (k,)): Two distributions over the same support

    Returns:
        * divergence (float): The squared Jensen-Shannon distance, in [0, 1]
    """
    hist = (hist + EPS) / (hist.sum() + len(hist) * EPS)
    ref_hist = (ref_hist + EPS) / (ref_hist.sum() + len(ref_hist) * EPS)
    mixture = 0.5 * (hist + ref_hist)
    divergence = 0.5 * (histogram_kl(mixture, hist) + histogram_kl(mixture, ref_hist))
    return divergence / math.log(2.0)


def compare_histograms(values, ref_values, edges=None, n_bins=200, value_range=None, weights=None,
                       return_hist=False):
    """Bin two sets of values on shared edges and compare the resulting histograms

    Args:
        * values (torch.Tensor of shape (n,) or (n, k)): Values to score
        * ref_values (torch.Tensor of shape (m,) or (m, k)): Reference values
        * edges (torch.Tensor of shape (n_bins + 1,) or None): Shared bin edges; None derives
            them from the reference values (default is None)
        * n_bins (int): Number of bins when edges is None (default is 200)
        * value_range (tuple of two floats or None): Explicit range when edges is None
            (default is None)
        * weights (torch.Tensor of shape (n,)): Weights of values (default is None)
        * return_hist (bool): Whether to also return the raw histograms and their edges
            (default is False)

    WARNING: read 'tvd' together with 'out_of_range'. When ALL of the scored mass falls outside
    the shared edges the histogram is empty rather than renormalised, so 'tvd' reads 0.5 -- half
    the reference mass -- not 1, while 'out_of_range' reads 1.0. The dropped mass is the signal
    there, and that pair of numbers is what tells a diffuse sampler apart from a merely
    mis-shaped one.

    Returns:
        * metrics (dict): 'tvd', 'kl', 'js' and 'out_of_range', the last being the fraction of
            the scored mass that fell outside the shared edges
        if return_hist:
            * hist (dict): 'edges', 'hist' and 'ref_hist', the raw normalized histograms
    """
    if edges is None:
        edges = reference_edges(ref_values, n_bins=n_bins, value_range=value_range)
    hist, out_of_range = weighted_histogram(values, edges, weights=weights,
                                            return_out_of_range=True)
    ref_hist = weighted_histogram(ref_values, edges)
    metrics = {
        'tvd': histogram_tvd(hist, ref_hist),
        'kl': histogram_kl(hist, ref_hist),
        'js': jensen_shannon(hist, ref_hist),
        'out_of_range': out_of_range,
    }
    if return_hist:
        return metrics, {'edges': edges, 'hist': hist, 'ref_hist': ref_hist}
    return metrics
