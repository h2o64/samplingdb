# The universal metric panel, and the binning it rests on

# Libraries
import math
import numpy as np
import torch
from samplingdb import GMM40, metrics
from samplingdb.targets.base import generator


def test_weighted_histogram_matches_numpy():
    """Bins are left-closed with the last one closed on both sides, as numpy.histogram has them"""
    values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 1.0, 2.5, 3.0, 4.0], dtype=torch.float64)
    edges = torch.linspace(0.0, 3.0, 4, dtype=torch.float64)
    hist, out_of_range = metrics.weighted_histogram(values, edges, return_out_of_range=True)

    expected, _ = np.histogram(values.numpy(), bins=edges.numpy())
    assert np.allclose(hist.numpy(), expected / expected.sum())

    # Two of the eight values fall outside [0, 3]
    assert abs(out_of_range - 2 / 8) < 1e-12


def test_weighted_histogram_uses_the_weights():
    """A weighted histogram reweights whole samples, spreading each weight over its own row"""
    values = torch.tensor([[0.5, 0.5], [2.5, 2.5]], dtype=torch.float64)
    edges = torch.linspace(0.0, 3.0, 4, dtype=torch.float64)
    weights = torch.tensor([3.0, 1.0], dtype=torch.float64)
    hist = metrics.weighted_histogram(values, edges, weights=weights)
    assert torch.allclose(hist, torch.tensor([0.75, 0.0, 0.25], dtype=torch.float64))

    # Uniform weights reproduce the unweighted histogram exactly
    assert torch.equal(metrics.weighted_histogram(values, edges),
                       metrics.weighted_histogram(values, edges, weights=torch.ones(2)))


def test_histogram_distances_vanish_on_identical_histograms():
    """TVD, KL and JS are all zero when both sides are the same histogram"""
    hist = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    assert metrics.histogram_tvd(hist, hist) == 0.0
    assert abs(metrics.histogram_kl(hist, hist)) < 1e-12
    assert abs(metrics.jensen_shannon(hist, hist)) < 1e-12

    # And TVD is 1 for disjoint support
    other = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    corner = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert abs(metrics.histogram_tvd(corner, other) - 1.0) < 1e-12


def test_reference_edges_reproduce_the_dem_total_variation():
    """compare_histograms is jarridrb/DEM's _compute_total_var, digit for digit"""
    gen = np.random.default_rng(0)
    ref = gen.normal(size=(500, 6))
    samples = gen.normal(loc=0.3, size=(500, 6))

    # The reference implementation, transcribed
    H_ref, x_ref = np.histogram(ref.reshape(-1), bins=200)
    H_samples, _ = np.histogram(samples.reshape(-1), bins=x_ref)
    expected = 0.5 * np.abs(H_ref / H_ref.sum() - H_samples / H_samples.sum()).sum()

    got = metrics.compare_histograms(torch.as_tensor(samples), torch.as_tensor(ref), n_bins=200)
    assert abs(got['tvd'] - expected) < 1e-12

    # And it reports the mass the reference implementation silently drops
    assert got['out_of_range'] > 0.0


def test_compare_histograms_returns_the_raw_histograms():
    """The non-scalar output: both normalised histograms and their shared edges"""
    ref = torch.randn(200, dtype=torch.float64)
    samples = torch.randn(200, dtype=torch.float64) + 1.0
    scalars, hist = metrics.compare_histograms(samples, ref, n_bins=32, return_hist=True)
    assert set(scalars) == {'tvd', 'kl', 'js', 'out_of_range'}
    assert hist['edges'].shape == (33,)
    assert hist['hist'].shape == (32,) and hist['ref_hist'].shape == (32,)
    assert abs(float(hist['ref_hist'].sum()) - 1.0) < 1e-12
    assert abs(metrics.histogram_tvd(hist['hist'], hist['ref_hist']) - scalars['tvd']) < 1e-12


def test_universal_metrics_vanish_on_identical_samples():
    """Every discrepancy is (numerically) zero when both sides are the same draw"""
    x = GMM40().test_samples(n=256)
    assert metrics.compute_mmd(x, x) < 1e-6
    assert metrics.compute_wasserstein_distance(x, x) < 1e-3
    assert metrics.compute_sliced_ks(x, x, generator=generator(0)) == 0.0
    assert metrics.compute_sliced_wasserstein_distance(x, x, generator=generator(0)) < 1e-3


def test_universal_metrics_grow_with_the_discrepancy():
    """Shifting one side increases every universal metric"""
    target = GMM40()
    x = target.test_samples(n=256, seed=1)
    y = target.test_samples(n=256, seed=2)
    shifted = y + 5.0
    for name, fn in [('mmd', metrics.compute_mmd),
                     ('w2', metrics.compute_wasserstein_distance),
                     ('w2_sliced', metrics.compute_sliced_wasserstein_distance)]:
        assert fn(x, shifted) > fn(x, y), name
    assert metrics.compute_sliced_ks(x, shifted, generator=generator(0)) > \
        metrics.compute_sliced_ks(x, y, generator=generator(0))


def test_universal_metrics_respect_the_weights():
    """Reweighting a two-cloud sample onto the reference cloud collapses every distance"""
    gen = generator(0)
    left = torch.randn(128, 2, generator=gen, dtype=torch.float64)
    right = torch.randn(128, 2, generator=gen, dtype=torch.float64) + 8.0
    x = torch.cat([left, right])
    weights = torch.cat([torch.ones(128), torch.zeros(128)]).to(torch.float64)

    reference = torch.randn(256, 2, generator=gen, dtype=torch.float64)
    for name, fn, kwarg in [('w2', metrics.compute_wasserstein_distance, 'weights1'),
                            ('mmd', metrics.compute_mmd, 'weights'),
                            ('w2_sliced', metrics.compute_sliced_wasserstein_distance, 'weights1')]:
        unweighted = fn(reference, x)
        assert fn(reference, x, **{kwarg: weights / weights.sum()}) < 0.5 * unweighted, name


def test_uniform_weights_agree_with_the_unweighted_path():
    """Explicit uniform weights change no OT or KS number"""
    target = GMM40()
    x = target.test_samples(n=256, seed=1)
    y = target.test_samples(n=256, seed=2)
    uniform = torch.full((256,), 1.0 / 256, dtype=torch.float64)

    assert abs(metrics.compute_wasserstein_distance(x, y)
               - metrics.compute_wasserstein_distance(x, y, weights1=uniform)) < 1e-9
    assert metrics.compute_sliced_ks(x, y, generator=generator(0)) == \
        metrics.compute_sliced_ks(x, y, weights=uniform, generator=generator(0))


def test_weighted_mmd_is_the_v_statistic():
    """MMD's two estimators are not interchangeable, and the weighted one is the biased V form

    The V-statistic keeps the diagonal self-similarity terms, so it sits above the unbiased
    U-statistic by O(1/n) in MMD^2 -- which is why compute_mmd only switches estimator when
    weights are actually given.
    """
    target = GMM40()
    x, y = target.test_samples(n=256, seed=1), target.test_samples(n=256, seed=2)
    uniform = torch.full((256,), 1.0 / 256, dtype=torch.float64)
    u_statistic = metrics.compute_mmd(x, y)
    v_statistic = metrics.compute_mmd(x, y, weights=uniform)
    assert v_statistic > u_statistic
    assert v_statistic ** 2 - u_statistic ** 2 < 4.0 / 256

    # Both stay far below a genuinely different distribution
    assert metrics.compute_mmd(x, y + 5.0) > 1.3 * v_statistic


def test_sliced_ks_is_reproducible_given_a_generator():
    """The random projections are seedable, so a reported KS number is reproducible"""
    target = GMM40()
    x, y = target.test_samples(n=128, seed=1), target.test_samples(n=128, seed=2)
    assert metrics.compute_sliced_ks(x, y, generator=generator(0)) == \
        metrics.compute_sliced_ks(x, y, generator=generator(0))


def test_wasserstein_1d_matches_scipy_and_survives_repeated_values():
    """The 1D distance, checked against scipy, including on the ties an MCMC chain produces"""
    from scipy.stats import wasserstein_distance
    gen = generator(0)
    u = torch.randn(500, generator=gen, dtype=torch.float64)
    v = torch.randn(400, generator=gen, dtype=torch.float64) + 0.5
    got = float(metrics.wasserstein_distance_1d(u, v))
    assert abs(got - wasserstein_distance(u.numpy(), v.numpy())) < 1e-12

    # Repeated values used to divide by zero and return nan
    tied = torch.cat([torch.zeros(50, dtype=torch.float64), torch.ones(50, dtype=torch.float64)])
    assert math.isfinite(float(metrics.wasserstein_distance_1d(tied, tied + 0.25)))

    # And weights are honoured
    weights = torch.cat([torch.ones(50), torch.zeros(50)]).to(torch.float64)
    assert abs(float(metrics.wasserstein_distance_1d(torch.zeros(10, dtype=torch.float64), tied,
                                                     v_weights=weights))) < 1e-9


def test_wasserstein_1d_is_exact_for_both_powers():
    """POT's emd2_1d, which the molecular energy panel is read against, to twelve digits"""
    import ot as pot
    gen = generator(0)
    u = torch.randn(500, generator=gen, dtype=torch.float64)
    v = 1.4 * torch.randn(400, generator=gen, dtype=torch.float64) + 0.5

    got = float(metrics.wasserstein_distance_1d(u, v, power=2))
    assert abs(got - math.sqrt(pot.emd2_1d(u.numpy(), v.numpy()))) < 1e-12
    got = float(metrics.wasserstein_distance_1d(u, v, power=1))
    assert abs(got - pot.emd2_1d(u.numpy(), v.numpy(), metric='euclidean')) < 1e-12

    # W2 dominates W1, and a distribution against itself is zero at both powers
    assert float(metrics.wasserstein_distance_1d(u, v, power=2)) > got
    for power in [1, 2]:
        assert float(metrics.wasserstein_distance_1d(u, u, power=power)) < 1e-12

    # Weighting v onto its first half is the same as passing that half alone
    weights = torch.cat([torch.ones(200), torch.zeros(200)]).to(torch.float64)
    for power in [1, 2]:
        whole = metrics.wasserstein_distance_1d(u, v, v_weights=weights, power=power)
        half = metrics.wasserstein_distance_1d(u, v[:200], power=power)
        assert abs(float(whole - half)) < 1e-12


def test_fully_out_of_range_samples_are_reported_not_renormalised():
    """All mass outside the edges gives out_of_range 1.0 and a tvd of 0.5, never a flattering one"""
    ref = torch.randn(200, dtype=torch.float64)
    far = ref + 1000.0
    scalars, hist = metrics.compare_histograms(far, ref, n_bins=32, return_hist=True)
    assert scalars['out_of_range'] == 1.0
    assert abs(scalars['tvd'] - 0.5) < 1e-12
    assert float(hist['hist'].sum()) == 0.0

    # And a partially out-of-range sample reports the fraction it lost
    half = torch.cat([ref[:100], ref[100:] + 1000.0])
    assert abs(metrics.compare_histograms(half, ref, n_bins=32)['out_of_range'] - 0.5) < 0.05


def test_the_autocorrelation_time_matches_an_ar1_process():
    """AR(1) has tau = (1 + phi) / (1 - phi) in closed form, which pins the estimator"""
    gen = generator(0)
    white = torch.randn(64, 20000, generator=gen, dtype=torch.float64)
    assert abs(metrics.integrated_autocorrelation_time(white) - 1.0) < 0.05

    for phi in [0.5, 0.8]:
        noise = torch.randn(64, 20000, generator=gen, dtype=torch.float64)
        x = torch.zeros(64, 20000, dtype=torch.float64)
        for step in range(1, 20000):
            x[:, step] = phi * x[:, step - 1] + noise[:, step]
        exact = (1 + phi) / (1 - phi)
        assert abs(metrics.integrated_autocorrelation_time(x[:, 1000:]) - exact) < 0.1 * exact


def test_the_effective_sample_size_reports_the_slowest_observable():
    """A set is worth what its worst coordinate says, not what its average does"""
    gen = generator(0)
    fast = torch.randn(32, 4000, generator=gen, dtype=torch.float64)
    slow = torch.zeros(32, 4000, dtype=torch.float64)
    noise = torch.randn(32, 4000, generator=gen, dtype=torch.float64)
    for step in range(1, 4000):
        slow[:, step] = 0.8 * slow[:, step - 1] + noise[:, step]

    ess = metrics.effective_sample_size({'fast': fast, 'slow': slow})
    assert ess['name'] == 'slow'
    assert abs(ess['tau'] - 9.0) < 1.5
    assert abs(ess['n_effective'] - 32 * 4000 / ess['tau']) < 1.0
    assert abs(ess['efficiency'] - 1.0 / ess['tau']) < 1e-12
    assert ess['tau_per_observable']['fast'] < ess['tau_per_observable']['slow']


def test_chain_mean_dispersion_agrees_with_the_autocorrelation_function():
    """A second estimate of tau from the spread of the chain means, with no lag window to choose

    It is a cross-check on the truncation rule, not a longer reach: both estimators are bounded by
    the chain length and both read low on a short series by the same amount.
    """
    gen = generator(0)

    # Independent chains: both read 1
    independent = torch.randn(128, 400, generator=gen, dtype=torch.float64)
    assert abs(metrics.chain_mean_dispersion({'x': independent})['tau'] - 1.0) < 0.35
    assert abs(metrics.effective_sample_size({'x': independent})['tau'] - 1.0) < 0.35

    # AR(1) with rho = 0.9, so tau = (1 + rho) / (1 - rho) = 19
    rho = 0.9
    series = torch.zeros(128, 1600, dtype=torch.float64)
    for step in range(1, series.shape[1]):
        noise = torch.randn(128, generator=gen, dtype=torch.float64)
        series[:, step] = rho * series[:, step - 1] + (1 - rho ** 2) ** 0.5 * noise
    series = series[:, 800:]

    dispersion = metrics.chain_mean_dispersion({'x': series})['tau']
    autocorrelation_time = metrics.effective_sample_size({'x': series})['tau']
    assert 12.0 < dispersion < 26.0
    assert abs(dispersion - autocorrelation_time) < 0.35 * autocorrelation_time

    # One chain stuck at its own mean is loss of independence too, and both must see it
    stuck = torch.randn(64, 400, generator=gen, dtype=torch.float64)
    stuck[0] = 6.0
    assert metrics.chain_mean_dispersion({'x': stuck})['tau'] > 10.0
    assert metrics.effective_sample_size({'x': stuck})['tau'] > 10.0
