# Every target's metric panel, including the non-scalar histogram returns

# Libraries
import numpy as np
import pytest
import torch
from samplingdb import BracketTwoModes, DW4, Funnel, GMM40, LJ13, ManyModes, ManyWell, MoS, TwoModes
from samplingdb.targets.base import generator

# The linear half of the target-agnostic panel, which the synthetic targets always report
STANDARD_KEYS = {'ks', 'w2_sliced'}

# Its quadratic half, which is opt-in everywhere
COSTLY_KEYS = {'mmd', 'w2', 'w2_reg'}


@pytest.mark.parametrize('target_cls', [Funnel, GMM40, MoS, ManyWell, TwoModes, BracketTwoModes,
                                        ManyModes])
def test_exact_targets_score_their_own_samples_well(target_cls):
    """An exact draw scored against another exact draw gives small universal metrics"""
    target = target_cls()
    samples = target.test_samples(n=512, seed=1)
    ret = target.compute_metrics(samples, generator=generator(0))
    assert STANDARD_KEYS <= set(ret)
    assert all(np.isfinite(value) for value in ret.values())
    assert ret['ks'] < 0.2


@pytest.mark.parametrize('target_cls', [GMM40, MoS, TwoModes, BracketTwoModes, ManyModes])
def test_mixture_panel(target_cls):
    """The mixture panel: mode weights, their entropy, KL, TV and the collapsed fraction"""
    target = target_cls()
    samples = target.test_samples(n=4096, seed=1)
    ret, hist = target.compute_metrics(samples, compute_standard_metrics=False, return_hist=True)

    # The raw histogram is the empirical mode weights
    # One group for a flat mixture, so the panel's numbers are the flat ones
    assert hist.shape == (1, target.n_modes)
    assert abs(float(hist.sum()) - 1.0) < 1e-12
    assert {'emc', 'kl_weights', 'tv_weights', 'perc_forgotten_modes'} <= set(ret)

    # An exact draw recovers the true weights, so KL and TV are small and nothing is forgotten
    assert ret['kl_weights'] < 0.05
    assert ret['tv_weights'] < 0.1
    assert ret['perc_forgotten_modes'] == 0.0


def test_mixture_panel_catches_mode_collapse():
    """A sample sitting in one mode scores emc 0, tv 2 * (1 - w_0) and forgets the rest"""
    target = GMM40()
    collapsed = target.means[0][None].expand(512, target.dim).contiguous()
    ret = target.compute_metrics(collapsed, compute_standard_metrics=False)
    assert ret['emc'] == 0.0
    assert abs(ret['tv_weights'] - 2 * (1 - 1 / 40)) < 1e-9
    assert abs(ret['perc_forgotten_modes'] - 39 / 40) < 1e-9
    assert np.isinf(ret['kl_weights'])


def test_two_modes_panel_reports_the_heavy_mode_weight():
    """TwoModes adds the heavy mode's weight in percent and its absolute error"""
    target = TwoModes(dim=2)
    samples = target.test_samples(n=20000, seed=0)
    ret = target.compute_metrics(samples, compute_standard_metrics=False)
    assert abs(ret['mode_weight'] - 100 * 2 / 3) < 2.0
    assert ret['mode_weight_abs_err'] < 2.0

    # Sampling only the light mode gives the maximal error
    light = target.means[1][None].expand(512, target.dim).contiguous()
    ret = target.compute_metrics(light, compute_standard_metrics=False)
    assert ret['mode_weight'] == 0.0
    assert abs(ret['mode_weight_abs_err'] - 100 * 2 / 3) < 1e-9


def test_manywell_panel():
    """ManyWell's panel is per well, plus the pooled quartic marginal against its exact law"""
    target = ManyWell(dim=8)
    samples = target.test_samples(n=8192, seed=1)
    ret, hist = target.compute_metrics(samples, compute_standard_metrics=False, return_hist=True)

    # The raw histogram is the per-well mode weights
    assert hist.shape == (4, 2)
    assert {'emc', 'kl_weights', 'tv_weights', 'perc_forgotten_modes', 'tvd_x1',
            'w1_x1'} <= set(ret)
    assert ret['kl_weights'] < 0.01
    assert ret['tvd_x1'] < 0.1
    assert ret['x1_out_of_range'] < 1e-12

    # The train split lives in the deep well of every well, so the per-well panel flags it
    train = target.train_samples(n=2048)
    biased = target.compute_metrics(train, compute_standard_metrics=False)
    assert biased['emc'] < 0.02
    assert biased['tv_weights'] > 0.3
    assert biased['tvd_x1'] > ret['tvd_x1']


def test_funnel_panel_scores_the_analytic_marginal():
    """The funnel's panel needs no reference draw: the x_1 law is exactly N(0, 9)"""
    target = Funnel()
    samples = target.test_samples(n=20000, seed=1)
    ret, hist = target.compute_metrics(samples, compute_standard_metrics=False, return_hist=True)
    assert {'ks_x1', 'tvd_x1', 'x1_mean_err', 'x1_var_err', 'neck_var_rel_err'} <= set(ret)
    assert ret['ks_x1'] < 0.02
    assert ret['tvd_x1'] < 0.1
    assert abs(ret['x1_var_err']) < 0.5

    # The exact reference histogram comes from the analytic CDF and sums to one
    assert abs(float(hist['ref_hist'].sum()) - 1.0) < 1e-12

    # The short-chain train split is many sigma off on the tail
    train = target.train_samples(n=4096)
    biased = target.compute_metrics(train, compute_standard_metrics=False)
    assert biased['ks_x1'] > 10 * ret['ks_x1']
    assert biased['x1_var_err'] < -5.0


def test_particle_panel_binning():
    """The particle panel: distance and energy TVDs, their dropped mass, and the raw histograms"""
    target = DW4()

    # Stand in for the not-yet-hosted reference split with an exact-ish MCMC sample
    ref_samples = target.train_samples(n=4096)[2048:]
    samples = target.train_samples(n=4096)[2048:] + 0.05

    ret, hist = target.compute_metrics(samples, ref_samples=ref_samples,
                                       compute_standard_metrics=False, return_hist=True)
    assert {'dist_total_var', 'dist_out_of_range', 'dist_w1', 'energy_total_var',
            'energy_out_of_range', 'energy_w1'} <= set(ret)
    assert 0.0 <= ret['dist_total_var'] <= 1.0

    # The raw histograms come back with their shared edges
    assert hist['dist']['edges'].shape == (201,)
    assert hist['energy']['edges'].shape == (101,)
    assert abs(float(hist['dist']['ref_hist'].sum()) - 1.0) < 1e-12

    # The energy histogram uses the pinned range, not the samples' own
    assert float(hist['energy']['edges'][0]) == target.energy_histogram_range[0]
    assert float(hist['energy']['edges'][-1]) == target.energy_histogram_range[1]


def test_particle_panel_reports_dropped_mass():
    """Samples outside the reference range are reported rather than silently renormalised"""
    target = LJ13()
    ref_samples = target.train_samples(n=2048)[1024:]
    diffuse = 3.0 * ref_samples
    ret = target.compute_metrics(diffuse, ref_samples=ref_samples, compute_standard_metrics=False)
    assert ret['dist_out_of_range'] > 0.5
    assert ret['energy_out_of_range'] > 0.5
    assert ret['dist_w1'] > 1.0


def test_weights_reach_every_target_specific_metric():
    """Zeroing the weights of half a sample changes the panel exactly as dropping them would"""
    target = GMM40()
    good = target.test_samples(n=512, seed=1)
    collapsed = target.means[0][None].expand(512, target.dim).contiguous()
    samples = torch.cat([good, collapsed])
    weights = torch.cat([torch.ones(512), torch.zeros(512)]).to(torch.float64)

    weighted = target.compute_metrics(samples, weights=weights, compute_standard_metrics=False)
    unweighted = target.compute_metrics(good, compute_standard_metrics=False)
    for name, value in unweighted.items():
        assert abs(weighted[name] - value) < 1e-9, name


def test_costly_metrics_are_opt_in():
    """Every QUADRATIC metric only runs when asked for, the MMD included"""
    target = TwoModes(dim=2)
    samples = target.test_samples(n=256, seed=1)
    default = target.compute_metrics(samples, generator=generator(0))
    assert not (COSTLY_KEYS & set(default))
    ret = target.compute_metrics(samples, skip_costly_metrics=False, generator=generator(0))
    assert COSTLY_KEYS <= set(ret)
    assert ret['w2'] > 0.0
    assert ret['n_costly_samples'] == 256


def test_the_costly_metrics_are_capped_and_say_so():
    """Both sides of a quadratic metric are cut to max_costly_samples, which is reported

    The MMD builds three n x n kernel matrices, so an uncapped panel at the catalog's default
    split size of 100000 would ask for 240 GB. The cap is what keeps the panel callable at any n,
    and reporting the realised count is what keeps it from reading as a full-sample number.
    """
    target = GMM40()
    samples = target.test_samples(n=3000, seed=1)
    ret = target.compute_metrics(samples, skip_costly_metrics=False, generator=generator(0),
                                 compute_standard_metrics=True)
    assert ret['n_costly_samples'] == 2000

    # The linear half still reads every sample, so the cap is on the quadratic half alone
    assert STANDARD_KEYS <= set(ret)


@pytest.mark.parametrize('target_cls', [DW4, LJ13])
def test_the_standard_panel_is_synthetic_only(target_cls):
    """A structured target reports its own invariant panel and no raw-coordinate one

    A sliced KS or a transport distance between raw coordinates presumes the coordinates are
    directly comparable, which they are not for a system defined up to a translation. Asking for
    the panel on one of these is not an error, it simply has none -- and it used to raise from
    inside POT when the OT solvers were also asked for.
    """
    target = target_cls()
    reference = target.train_samples(n=512)
    ret = target.compute_metrics(reference[:256], ref_samples=reference[256:],
                                 skip_costly_metrics=False, generator=generator(0))
    assert not (STANDARD_KEYS & set(ret))
    assert not (COSTLY_KEYS & set(ret))
    assert 'dist_total_var' in ret and 'energy_w1' in ret
    assert not target.standard_panel and Funnel().standard_panel
