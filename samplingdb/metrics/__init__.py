# Metrics: the universal panel, and the shared machinery the target-specific ones are built from

# Libraries
from .weights import normalize_weights, subsample
from .histogram import (
    compare_histograms, histogram_2d, histogram_kl, histogram_tvd, jensen_shannon, reference_edges,
    shared_edges, weighted_histogram,
)
from .mmd import compute_mmd
from .ks import compute_sliced_ks
from .molecular import TicaModel, fit_tica, kmeans, kmeans_jsd, wrap_angles
from .standard import compute_standard_metrics
from .structure import compute_structure_metrics
from .wasserstein import (
    circular_wasserstein_1d, compute_sliced_wasserstein_distance, compute_wasserstein_distance,
    torus_wasserstein, wasserstein_distance_1d,
)
