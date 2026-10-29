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
    """All mass outside the edges gives out_of_range 1.0 and a tvd of 1, never a flattering one"""
    ref = torch.randn(200, dtype=torch.float64)
    far = ref + 1000.0
    scalars, hist = metrics.compare_histograms(far, ref, n_bins=32, return_hist=True)
    assert scalars['out_of_range'] == 1.0
    assert scalars['tvd'] == 1.0
    assert float(hist['hist'].sum()) == 0.0

    # Ranked last, after a sample with a sliver of mass inside the edges
    sliver = torch.cat([far[:199], ref[:1]])
    assert metrics.compare_histograms(sliver, ref, n_bins=32)['tvd'] < 1.0

    # And a partially out-of-range sample reports the fraction it lost
    half = torch.cat([ref[:100], ref[100:] + 1000.0])
    assert abs(metrics.compare_histograms(half, ref, n_bins=32)['out_of_range'] - 0.5) < 0.05


def test_the_circular_wasserstein_distance_ignores_where_the_circle_is_cut():
    """A rotation by 0.1 costs 0.1 wherever the angles sit, which a linear W1 gets wrong at +-pi"""
    gen = generator(0)
    u = 0.3 * torch.randn(4000, generator=gen, dtype=torch.float64)
    v = u + 0.1
    assert abs(float(metrics.circular_wasserstein_1d(u, v)) - 0.1) < 1e-9

    # Moved onto the cut, the circular distance does not change and the linear one does
    u_cut = torch.remainder(u, 2 * math.pi) - math.pi
    v_cut = torch.remainder(v, 2 * math.pi) - math.pi
    assert abs(float(metrics.circular_wasserstein_1d(u_cut, v_cut)) - 0.1) < 1e-9
    assert float(metrics.wasserstein_distance_1d(u_cut, v_cut)) > 0.5

    # Away from the cut and for small shifts it is the linear W1
    w = 0.3 * torch.randn(3000, generator=gen, dtype=torch.float64) + 0.2
    assert abs(float(metrics.circular_wasserstein_1d(u, w))
               - float(metrics.wasserstein_distance_1d(u, w))) < 1e-9

    # Integer weights are repeats
    weights = torch.randint(1, 4, (100,), generator=gen).to(torch.float64)
    repeated = torch.repeat_interleave(u[:100], weights.long())
    assert abs(float(metrics.circular_wasserstein_1d(u[:100], w, u_weights=weights))
               - float(metrics.circular_wasserstein_1d(repeated, w))) < 1e-9


def test_weights_are_normalized_inside_every_weighted_metric():
    """Scaling the weights by a constant changes nothing, on a direct call as in the panel"""
    gen = generator(0)
    x = torch.randn(300, 3, generator=gen, dtype=torch.float64)
    y = torch.randn(300, 3, generator=gen, dtype=torch.float64) + 0.5
    weights = torch.rand(300, generator=gen, dtype=torch.float64)
    for scale in [1e-3, 7.0]:
        assert abs(metrics.compute_mmd(x, y, weights=scale * weights)
                   - metrics.compute_mmd(x, y, weights=weights / weights.sum())) < 1e-12
        assert abs(metrics.compute_sliced_wasserstein_distance(
            x, y, weights1=scale * weights, generator=generator(1))
            - metrics.compute_sliced_wasserstein_distance(
                x, y, weights1=weights / weights.sum(), generator=generator(1))) < 1e-12
        assert abs(metrics.compute_wasserstein_distance(x, y, weights1=scale * weights)
                   - metrics.compute_wasserstein_distance(x, y, weights1=weights)) < 1e-12


def test_sliced_ks_is_the_exact_two_sample_ks_on_every_projection():
    """Each projection is scipy's ks_2samp, ties included, and the panel averages them"""
    from scipy.stats import ks_2samp
    from samplingdb.metrics.ks import compute_projected_ks
    gen = generator(0)
    x = torch.randn(500, 3, generator=gen, dtype=torch.float64)
    y = torch.randn(400, 3, generator=gen, dtype=torch.float64) + 0.2
    directions = torch.randn(6, 3, generator=gen, dtype=torch.float64)
    directions /= torch.linalg.norm(directions, dim=-1, keepdim=True)
    got = compute_projected_ks(x, y, directions)
    expected = [ks_2samp((x @ u).numpy(), (y @ u).numpy()).statistic for u in directions]
    assert np.allclose(got.numpy(), expected, atol=1e-12)

    # Rounded values tie within and across the two sets
    x_tied, y_tied = torch.round(2 * x) / 2, torch.round(2 * y) / 2
    axes = torch.eye(3, dtype=torch.float64)
    got = compute_projected_ks(x_tied, y_tied, axes)
    expected = [ks_2samp(x_tied[:, i].numpy(), y_tied[:, i].numpy()).statistic for i in range(3)]
    assert np.allclose(got.numpy(), expected, atol=1e-12)


def test_sliced_ks_sees_every_sample_wherever_it_is():
    """No range: mass moved far away and rows that are not finite both raise the distance"""
    gen = generator(0)
    x = torch.randn(1000, 2, generator=gen, dtype=torch.float64)
    y = torch.randn(1000, 2, generator=gen, dtype=torch.float64)
    clean = metrics.compute_sliced_ks(x, y, generator=generator(0))

    # A tenth of the mass 1000 units away is a tenth on every projection
    far = y.clone()
    far[:100] += 1000.0
    assert metrics.compute_sliced_ks(x, far, generator=generator(0)) > 0.1

    # Half of the rows not finite reads at least one half, never better than the rows before
    broken = y.clone()
    broken[::2, 0] = float('nan')
    broken[1::4, 1] = float('inf')
    assert metrics.compute_sliced_ks(x, broken, generator=generator(0)) >= 0.5 > clean

    # Whichever way the set was wrong: a set far to the left is not helped by losing rows
    left = y - 10.0
    lost = left.clone()
    lost[::2] = float('nan')
    assert metrics.compute_sliced_ks(x, lost, generator=generator(0)) >= \
        metrics.compute_sliced_ks(x, left, generator=generator(0)) - 1e-12


def test_the_weighted_mmd_ignores_rows_without_weight():
    """A zero-weight row changes neither the bandwidth nor the value"""
    gen = generator(0)
    x = torch.randn(200, 2, generator=gen, dtype=torch.float64)
    y = torch.randn(150, 2, generator=gen, dtype=torch.float64) + 0.3
    junk = 100.0 + torch.randn(150, 2, generator=gen, dtype=torch.float64)
    weights = torch.cat([torch.ones(150), torch.zeros(150)]).to(torch.float64)
    padded = metrics.compute_mmd(x, torch.cat([y, junk]), weights=weights)
    alone = metrics.compute_mmd(x, y, weights=torch.ones(150, dtype=torch.float64))
    assert abs(padded - alone) < 1e-12


def test_the_quadratic_metrics_draw_their_rows_rather_than_slice_them():
    """Weight that lies past row max_costly_samples is still what gets scored

    A leading slice kept the first 2000 rows, whose weights were all zero here: normalising them
    divided zero by zero and the transport solver failed.
    """
    target = GMM40()
    reference = target.test_samples(n=3000, seed=0)
    good = target.test_samples(n=1000, seed=1)
    junk = target.means[0] + torch.zeros(2000, target.dim, dtype=torch.float64)
    samples = torch.cat([junk, good])
    weights = torch.cat([torch.zeros(2000), torch.ones(1000)]).to(torch.float64)

    ret = metrics.compute_standard_metrics(samples, reference, weights=weights,
                                           skip_costly_metrics=False, generator=generator(0))
    assert ret['n_costly_samples'] == 2000
    assert all(math.isfinite(value) for value in ret.values())

    # And it reads like the good rows alone, far from the junk
    alone = metrics.compute_standard_metrics(good, reference, skip_costly_metrics=False,
                                             generator=generator(0))
    junk_only = metrics.compute_standard_metrics(junk, reference, skip_costly_metrics=False,
                                                 generator=generator(0))
    assert abs(ret['w2'] - alone['w2']) < 0.25 * alone['w2']
    assert ret['w2'] < 0.5 * junk_only['w2']


def test_subsample_is_uniform_without_weights_and_proportional_with_them():
    """Distinct rows without weights, and systematic resampling with them"""
    x = torch.arange(100, dtype=torch.float64)[:, None]
    drawn = metrics.subsample(x, 30, generator=generator(0))
    assert drawn.shape == (30, 1) and len(drawn.unique()) == 30
    assert torch.equal(metrics.subsample(x, 200), x)

    weights = torch.zeros(100, dtype=torch.float64)
    weights[[3, 97]] = torch.tensor([1.0, 3.0], dtype=torch.float64)
    drawn = metrics.subsample(x, 4000, weights=weights, generator=generator(0))
    assert set(drawn[:, 0].tolist()) == {3.0, 97.0}
    assert abs(int((drawn == 97.0).sum()) - 3000) <= 1

    # Equal weights over no more than n rows draw every row once, as the unweighted path does
    assert torch.equal(metrics.subsample(x, 100, weights=torch.full((100,), 0.01),
                                         generator=generator(0)), x)
