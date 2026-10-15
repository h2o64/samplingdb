# The pair-distance and energy panel every structured target reports

# Libraries
from .histogram import compare_histograms
from .wasserstein import wasserstein_distance_1d
from .weights import normalize_weights


def compute_structure_metrics(distances, ref_distances, energy, ref_energy, weights=None,
                              distance_edges=None, n_distance_bins=200, energy_range=None,
                              n_energy_bins=100, return_hist=False):
    """Compare pair-distance and energy histograms against a reference set

    The three structured families -- particle clusters, glass formers and peptides -- are all
    read on the same two observables, so the panel is written once here and each target supplies
    only what is genuinely its own: the distance edges (derived from the reference set, or pinned
    by a periodic box), the energy range, and its own normalisation of the energy. A target whose
    energy is per particle or per atom passes it that way and says so;
    the key is 'energy_w1' either way, so one name means one quantity across the corpus.

    WARNING: whichever edges the caller passes decide whether mass can fall outside them, and
    'dist_out_of_range' / 'energy_out_of_range' are the numbers that say whether it did. A TVD
    against reference-derived edges understates the discrepancy of a diffuse sampler, because
    what lands outside is dropped and the survivors renormalised; read the pair together.

    Args:
        * distances (torch.Tensor of shape (n, n_pairs)): Pair distances of the scored samples
        * ref_distances (torch.Tensor of shape (m, n_pairs)): Pair distances of the reference
        * energy (torch.Tensor of shape (n,)): Energies of the scored samples
        * ref_energy (torch.Tensor of shape (m,)): Energies of the reference
        * weights (torch.Tensor of shape (n,)): Weights of the samples, normalized inside
            (default is None)
        * distance_edges (torch.Tensor of shape (n_bins + 1,) or None): Shared distance edges;
            None derives them from the reference distances (default is None)
        * n_distance_bins (int): Distance bins when distance_edges is None (default is 200)
        * energy_range (tuple of two floats or None): Explicit energy range; None derives it from
            the reference energies (default is None)
        * n_energy_bins (int): Energy bins (default is 100)
        * return_hist (bool): Whether to also return the raw histograms (default is False)

    Returns:
        * metrics (dict): 'dist_total_var', 'dist_out_of_range', 'dist_w1', 'energy_total_var',
            'energy_out_of_range' and 'energy_w1'
        if return_hist:
            * hist (dict): 'dist' and 'energy', each holding 'edges', 'hist' and 'ref_hist'
    """
    dist_metrics, dist_hist = compare_histograms(distances, ref_distances, edges=distance_edges,
                                                 n_bins=n_distance_bins, weights=weights,
                                                 return_hist=True)
    energy_metrics, energy_hist = compare_histograms(energy, ref_energy, n_bins=n_energy_bins,
                                                     value_range=energy_range, weights=weights,
                                                     return_hist=True)

    # The distance transport is over POOLED pairs, so every sample's weight is repeated once per
    # pair it contributes
    pair_weights = None
    if weights is not None:
        pair_weights = normalize_weights(weights).repeat_interleave(distances.shape[-1])
    metrics = {
        'dist_total_var': dist_metrics['tvd'],
        'dist_out_of_range': dist_metrics['out_of_range'],
        'dist_w1': float(wasserstein_distance_1d(ref_distances.reshape(-1), distances.reshape(-1),
                                                 v_weights=pair_weights)),
        'energy_total_var': energy_metrics['tvd'],
        'energy_out_of_range': energy_metrics['out_of_range'],
        'energy_w1': float(wasserstein_distance_1d(ref_energy, energy, v_weights=weights)),
    }
    if return_hist:
        return metrics, {'dist': dist_hist, 'energy': energy_hist}
    return metrics
