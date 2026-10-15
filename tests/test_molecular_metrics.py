# The molecular metric primitives, against the numpy and scikit-learn versions they reproduce

# Libraries
import math
import numpy as np
import ot as pot
import torch
from scipy.spatial.distance import jensenshannon
from samplingdb.metrics.histogram import (
    histogram_2d, jensen_shannon, shared_edges, weighted_histogram,
)
from samplingdb.metrics.molecular import (
    N_BINS, TicaModel, cluster_populations, kmeans, kmeans_jsd, wrap_angles,
)
from samplingdb.metrics.wasserstein import (
    compute_wasserstein_distance, torus_cost, torus_wasserstein,
)
from samplingdb.targets.base import generator


def angles(n, shift=0.0, seed=0):
    """Returns n angle pairs in [-pi, pi), shifted around the circle"""
    x = torch.rand(n, 2, generator=generator(seed), dtype=torch.float64) * 2 * math.pi - math.pi
    return (x + shift + math.pi) % (2 * math.pi) - math.pi


def their_torus_wasserstein(x0, x1, power=2):
    """transferable-samplers' _torus_wasserstein_distance, verbatim in numpy"""
    x0, x1 = np.asarray(x0), np.asarray(x1)
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    delta = np.abs(x0[:, None] - x1[None, :])
    delta = np.minimum(delta, 2 * np.pi - delta)
    value = pot.emd2(a, b, (delta ** power).sum(-1), numItermax=int(1e9))
    return math.sqrt(value) if power == 2 else float(value)


def test_the_torus_distance_goes_the_short_way_round():
    """Two angles either side of the branch cut are close, which is the whole point of the metric"""
    left = torch.tensor([[-3.1]], dtype=torch.float64)
    right = torch.tensor([[3.1]], dtype=torch.float64)
    cost = torus_cost(left, right, power=1)
    assert abs(float(cost) - (2 * math.pi - 6.2)) < 1e-12

    # The same pair on the line would be 6.2 apart, so a Euclidean cost sees the opposite
    assert float(torch.cdist(left, right)) > float(cost)

    # And a distribution against a copy of itself is zero, however it straddles the cut
    x = angles(64)
    assert torus_wasserstein(x, x) < 1e-9


def test_torus_and_euclidean_transport_match_the_numpy_originals():
    """The cost matrices are built in torch, but the optimum must be the same number"""
    x0, x1 = angles(200, seed=0), angles(150, shift=1.0, seed=1)
    for power in [1, 2]:
        assert abs(torus_wasserstein(x0, x1, power=power)
                   - their_torus_wasserstein(x0, x1, power=power)) < 1e-9

    y0 = torch.randn(200, 3, generator=generator(2), dtype=torch.float64)
    y1 = torch.randn(150, 3, generator=generator(3), dtype=torch.float64) + 0.4
    a, b = pot.unif(200), pot.unif(150)
    cost = torch.cdist(y0, y1).numpy()
    assert abs(compute_wasserstein_distance(y0, y1, power=1) - pot.emd2(a, b, cost)) < 1e-9
    assert abs(compute_wasserstein_distance(y0, y1, power=2)
               - math.sqrt(pot.emd2(a, b, cost ** 2))) < 1e-9


def test_transport_honours_weights():
    """A weighted set must give exactly what the subset it puts its mass on gives"""
    x0, x1 = angles(120, seed=4), angles(80, shift=0.7, seed=5)
    weights = torch.zeros(80, dtype=torch.float64)
    weights[:40] = 2.0
    assert abs(torus_wasserstein(x0, x1, weights=weights)
               - torus_wasserstein(x0, x1[:40])) < 1e-9

    # Weights are normalised, so their scale cannot matter
    assert abs(torus_wasserstein(x0, x1, weights=weights)
               - torus_wasserstein(x0, x1, weights=weights / 7.3)) < 1e-9


def test_wrap_angles_removes_the_branch_cut():
    """sin and cos of every angle, which is what makes a Euclidean clustering of angles sensible"""
    x = angles(32)
    wrapped = wrap_angles(x)
    assert wrapped.shape == (32, 4)
    assert torch.allclose(wrapped[:, :2] ** 2 + wrapped[:, 2:] ** 2, torch.ones_like(x))

    # A full turn is the same point in this representation
    assert torch.allclose(wrap_angles(x + 2 * math.pi), wrapped, atol=1e-12)


def test_jensen_shannon_is_scipy_squared_in_base_two():
    """Upstream reports jensenshannon(...) ** 2, so the base and the square both matter"""
    p = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    q = torch.tensor([0.1, 0.1, 0.8], dtype=torch.float64)
    expected = jensenshannon(p.numpy(), q.numpy(), base=2) ** 2
    assert abs(jensen_shannon(p, q) - expected) < 1e-9

    # Bounded by one, zero only for equal distributions
    assert jensen_shannon(p, p) < 1e-12
    assert 0.0 < jensen_shannon(p, q) < 1.0
    one_hot = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    other = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    assert abs(jensen_shannon(one_hot, other) - 1.0) < 1e-6


def test_kmeans_partitions_well_separated_blobs():
    """Lloyd's algorithm in torch, checked on a problem whose answer is not ambiguous"""
    centres = torch.tensor([[-10.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=torch.float64)
    x = (centres[torch.arange(300) % 3]
         + 0.2 * torch.randn(300, 2, generator=generator(0), dtype=torch.float64))
    fitted = kmeans(x, n_clusters=3, seed=0)

    # Every true centre has a fitted one on top of it, in some order
    distance = torch.cdist(centres, fitted).min(dim=-1).values
    assert float(distance.max()) < 0.1

    # And the populations of a balanced set are balanced
    populations = cluster_populations(x, fitted)
    assert abs(float(populations.sum()) - 1.0) < 1e-12
    assert float(populations.min()) > 0.3


def test_kmeans_jsd_is_zero_for_a_copy_and_grows_with_a_shift():
    """The divergence is fitted on the reference, so both sets are read on one discretisation"""
    x = torch.randn(600, 2, generator=generator(1), dtype=torch.float64)
    y = torch.randn(600, 2, generator=generator(2), dtype=torch.float64)

    same, centres = kmeans_jsd(x, x, n_clusters=10)
    assert same < 1e-12
    assert centres.shape == (10, 2)
    independent = kmeans_jsd(x, y, n_clusters=10)[0]
    shifted = kmeans_jsd(x, y + 1.5, n_clusters=10)[0]
    assert independent < shifted

    # Reusing the reference's centres is what makes a run of samplers comparable
    assert abs(kmeans_jsd(x, y, centres=centres)[0] - independent) < 0.05

    # Dropping half the mass by weight matches passing that half
    weights = torch.cat([torch.ones(300), torch.zeros(300)]).to(torch.float64)
    assert abs(kmeans_jsd(x, y, weights=weights, centres=centres)[0]
               - kmeans_jsd(x, y[:300], centres=centres)[0]) < 1e-12


def test_kmeans_jsd_reproduces_scikit_learn_within_its_own_seed_spread():
    """Two k-means implementations never agree exactly; both must land in the same place"""
    from sklearn.cluster import KMeans

    x = torch.randn(500, 2, generator=generator(3), dtype=torch.float64)
    y = 1.3 * torch.randn(500, 2, generator=generator(4), dtype=torch.float64) + 0.5

    def theirs(state):
        fitted = KMeans(n_clusters=20, random_state=state).fit(x.numpy())
        p = np.bincount(fitted.labels_, minlength=20) + 1e-10
        q = np.bincount(fitted.predict(y.numpy()), minlength=20) + 1e-10
        return jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2

    mine = [kmeans_jsd(x, y, seed=seed)[0] for seed in range(4)]
    sk = [theirs(state) for state in range(4)]
    spread = max(max(mine) - min(mine), max(sk) - min(sk))
    assert abs(sum(mine) / len(mine) - sum(sk) / len(sk)) < 2 * spread + 0.01


def test_the_histograms_are_probabilities_on_shared_edges():
    """Shared edges are what makes two histograms comparable, and no mass may vanish"""
    reference = torch.randn(500, generator=generator(5), dtype=torch.float64)
    samples = torch.randn(400, generator=generator(6), dtype=torch.float64) + 0.5
    edges = shared_edges(reference, samples, n_bins=N_BINS)
    assert edges.shape == (N_BINS + 1,)
    assert float(edges[0]) <= min(float(reference.min()), float(samples.min()))
    assert float(edges[-1]) >= max(float(reference.max()), float(samples.max()))

    hist = weighted_histogram(samples, edges)
    assert abs(float(hist.sum()) - 1.0) < 1e-12
    assert torch.allclose(hist, torch.as_tensor(np.histogram(samples.numpy(), bins=edges.numpy())[0]
                                                / len(samples)))

    # Weights move mass between bins without creating any
    weights = torch.rand(400, generator=generator(7), dtype=torch.float64)
    weighted = weighted_histogram(samples, edges, weights=weights)
    assert abs(float(weighted.sum()) - 1.0) < 1e-12
    assert not torch.allclose(weighted, hist)

    # A value outside the edges is dropped rather than folded into an edge bin, and the survivors
    # are renormalised while the dropped fraction is reported -- one convention for every panel
    outside = torch.cat([samples, torch.full((100,), 1e6, dtype=torch.float64)])
    hist, out_of_range = weighted_histogram(outside, edges, return_out_of_range=True)
    assert abs(float(hist.sum()) - 1.0) < 1e-12
    assert abs(out_of_range - 100 / 500) < 1e-12


def test_the_2d_histogram_matches_numpy():
    """The joint histogram behind every Ramachandran and TICA plot"""
    x = torch.randn(700, 2, generator=generator(8), dtype=torch.float64)
    edges = shared_edges(x.reshape(-1), x.reshape(-1), n_bins=16)
    hist = histogram_2d(x, edges)
    expected = np.histogram2d(x[:, 0].numpy(), x[:, 1].numpy(),
                              bins=[edges.numpy(), edges.numpy()])[0] / len(x)
    assert hist.shape == (16, 16)
    assert abs(float(hist.sum()) - 1.0) < 1e-12
    assert torch.allclose(hist, torch.as_tensor(expected))

    # Weighting onto a subset is the same as histogramming that subset
    weights = torch.cat([torch.ones(300), torch.zeros(400)]).to(torch.float64)
    assert torch.allclose(histogram_2d(x, edges, weights=weights), histogram_2d(x[:300], edges))


def test_the_tica_model_is_a_centred_projection_that_moves_with_its_input():
    """It is applied wherever the samples are, so it must be a buffer-carrying module"""
    mean = torch.arange(5, dtype=torch.float64)
    projection = torch.randn(5, 3, generator=generator(9), dtype=torch.float64)
    model = TicaModel(mean, projection, dim=2)
    features = torch.randn(7, 5, generator=generator(10), dtype=torch.float64)

    assert model.projection.shape == (5, 2)
    assert torch.allclose(model(features), (features - mean) @ projection[:, :2])
    assert set(dict(model.named_buffers())) == {'mean', 'projection'}
