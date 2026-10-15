# The molecular metric panel of transferable-samplers: TICA and the k-means divergences

# Libraries
import numpy as np
import torch
from .histogram import jensen_shannon
from .weights import normalize_weights

# Bins every upstream histogram uses
N_BINS = 100


def wrap_angles(angles):
    """Maps angles to their (sin, cos) pair, so a clustering metric sees no branch cut

    Args:
        * angles (torch.Tensor of shape (n, k)): Angles in radians

    Returns:
        * features (torch.Tensor of shape (n, 2 * k)): Sines then cosines
    """
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def kmeans(features, n_clusters=20, n_steps=100, seed=0):
    """Fits k-means on device, with a seeded k-means++ initialisation

    NOTE: this is Lloyd's algorithm in torch rather than a call into scikit-learn, so a cluster
    assignment is not bit-identical to upstream's, and the divergence built on it carries the
    initialisation noise every k-means metric has: on AAAAAA's TICA plane, eight seeds of this
    give 0.0662 to 0.0820 and eight random_states of scikit-learn give 0.0705 to 0.0825, the two
    ranges overlapping and both an order of magnitude above the 0.008 reference-versus-itself
    floor. The centres are returned so a caller can reuse exactly the ones a reference produced.

    Args:
        * features (torch.Tensor of shape (n, d)): Points to cluster
        * n_clusters (int): Number of clusters (default is 20)
        * n_steps (int): Maximum Lloyd iterations (default is 100)
        * seed (int): Seed of the initialisation (default is 0)

    Returns:
        * centres (torch.Tensor of shape (n_clusters, d)): Cluster centres
    """
    generator = torch.Generator(device='cpu').manual_seed(seed)
    first = int(torch.randint(len(features), (1,), generator=generator))
    centres = [features[first]]

    # k-means++: every further centre is drawn proportionally to its squared distance
    for _ in range(n_clusters - 1):
        distance = torch.cdist(features, torch.stack(centres)).min(dim=-1).values ** 2
        probability = (distance / distance.sum().clamp_min(1e-30)).cpu()
        centres.append(features[int(torch.multinomial(probability, 1, generator=generator))])
    centres = torch.stack(centres)

    # Lloyd's iterations, stopping as soon as no point changes cluster
    assignment = torch.cdist(features, centres).argmin(dim=-1)
    for _ in range(n_steps):
        for index in range(n_clusters):
            mask = assignment == index
            if bool(mask.any()):
                centres[index] = features[mask].mean(0)
        updated = torch.cdist(features, centres).argmin(dim=-1)
        if bool((updated == assignment).all()):
            break
        assignment = updated
    return centres


def cluster_populations(features, centres, weights=None):
    """Returns the (weighted) fraction of points nearest to each centre"""
    assignment = torch.cdist(features, centres).argmin(dim=-1)
    mass = normalize_weights(weights, n=len(features), dtype=features.dtype,
                             device=features.device)
    counts = torch.zeros(len(centres), dtype=features.dtype, device=features.device)
    return counts.scatter_add_(0, assignment, mass.to(features.dtype))


def kmeans_jsd(reference, samples, n_clusters=20, weights=None, seed=0, centres=None):
    """Jensen-Shannon divergence between two feature sets, discretised by k-means

    The clustering is fitted on the REFERENCE and both sets are assigned to it, which is what
    makes the number a comparison rather than a function of two independent clusterings. The
    divergence itself is metrics.histogram.jensen_shannon, i.e. upstream's squared JS distance in
    base 2 -- the same units every other divergence in the corpus reports.

    Args:
        * reference (torch.Tensor of shape (n, d)): Reference features
        * samples (torch.Tensor of shape (m, d)): Features to score
        * n_clusters (int): Number of clusters (default is 20)
        * weights (torch.Tensor of shape (m,) or None): Weights of samples (default is None)
        * seed (int): Seed of the k-means initialisation (default is 0)
        * centres (torch.Tensor or None): Pre-fitted centres to reuse (default is None)

    Returns:
        * divergence (float): The squared Jensen-Shannon distance, in [0, 1]
        * centres (torch.Tensor of shape (n_clusters, d)): The centres used
    """
    if centres is None:
        centres = kmeans(reference, n_clusters=n_clusters, seed=seed)
    reference_population = cluster_populations(reference, centres)
    sample_population = cluster_populations(samples, centres, weights=weights)
    return jensen_shannon(sample_population, reference_population), centres


class TicaModel(torch.nn.Module):
    """A fitted TICA projection, applied in torch

    Upstream's TicaModel: centre by the training mean, project onto the leading components. The
    fit is a one-off on reference data (fit_tica below); the projection is a matmul and runs
    wherever the samples are.
    """

    def __init__(self, mean, projection, dim=2):
        """Builds the projection

        Args:
            * mean (torch.Tensor of shape (n_features,)): Feature mean of the fitting set
            * projection (torch.Tensor of shape (n_features, n_components)): TICA components
            * dim (int): Number of components to keep (default is 2)
        """
        super().__init__()
        self.dim = dim
        self.register_buffer('mean', mean)
        self.register_buffer('projection', projection[:, :dim])

    def forward(self, features):
        """Projects features onto the leading TICA components"""
        return (features - self.mean) @ self.projection


def fit_tica(features, lagtime=100, dim=2):
    """Fits a Koopman-reweighted TICA model, then keeps only its projection

    NOTE: deeptime does the fit, exactly as upstream does (TICA with a KoopmanWeightingEstimator
    at the same lagtime), because reproducing Koopman reweighting by hand would be a second
    implementation of somebody else's estimator. It is a one-off on time-ordered reference data
    and the only place a non-torch dependency enters this module, which is why it ships in the
    [molecular] extra and is imported lazily here.

    WARNING: the features must be in TRAJECTORY ORDER. TICA is defined through a time-lagged
    covariance, so a shuffled reference set silently gives a meaningless projection.

    Args:
        * features (torch.Tensor of shape (n, n_features)): Time-ordered features
        * lagtime (int): Lag of the covariance, in frames (default is 100)
        * dim (int): Number of components to keep (default is 2)

    Returns:
        * model (TicaModel): The fitted projection
    """
    import deeptime

    array = features.detach().cpu().numpy().astype(np.float64)
    estimator = deeptime.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting = estimator.fit(array).fetch_model()
    fitted = deeptime.decomposition.TICA(dim=dim, lagtime=lagtime).fit(array, reweighting)
    model = fitted.fetch_model()
    return TicaModel(torch.as_tensor(model.mean_0, dtype=features.dtype),
                     torch.as_tensor(model.singular_vectors_left, dtype=features.dtype), dim=dim)
