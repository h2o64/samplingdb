# The classical kernels must sample their target, and be reproducible given a seed

# Libraries
import math
import torch
from samplingdb import GMM40, metrics
from samplingdb.baselines import geometric_ladder, hmc, mala, parallel_tempering, random_walk_mh, \
    student_t2, ula
from samplingdb.targets.base import generator


def gaussian():
    """Returns the log-density and gradient of a standard Gaussian in dim 4"""
    log_prob = lambda x: -0.5 * (x ** 2).sum(-1)
    score = lambda x: -x
    return log_prob, score


def test_every_kernel_samples_a_standard_gaussian():
    """Mean 0, variance 1 per coordinate, from 2000 chains of 200 steps"""
    log_prob, score = gaussian()
    x0 = torch.randn(2000, 4, generator=generator(0), dtype=torch.float64)
    runs = {
        'random_walk_mh': random_walk_mh(log_prob, x0, 200, 1.2, generator=generator(1)),
        'ula': ula(score, x0, 200, 0.05, generator=generator(1)),
        'mala': mala(log_prob, score, x0, 200, 0.3, generator=generator(1)),
        'hmc': hmc(log_prob, score, x0, 50, 0.2, n_leapfrog=10, generator=generator(1)),
    }
    for name, x in runs.items():
        assert float(x.mean().abs()) < 0.1, name
        assert abs(float(x.var()) - 1.0) < 0.1, name


def test_kernels_are_reproducible_and_report_acceptance():
    """Same seed, same numbers; the acceptance rate is in a sane range"""
    log_prob, score = gaussian()
    x0 = torch.zeros(64, 4, dtype=torch.float64)
    first, acceptance = random_walk_mh(log_prob, x0, 50, 0.8, generator=generator(3),
                                       return_acceptance=True)
    second = random_walk_mh(log_prob, x0, 50, 0.8, generator=generator(3))
    assert torch.equal(first, second)
    assert 0.2 < float(acceptance) < 0.9

    _, mala_acceptance = mala(log_prob, score, x0, 50, 0.2, generator=generator(3),
                              return_acceptance=True)
    _, hmc_acceptance = hmc(log_prob, score, x0, 20, 0.2, generator=generator(3),
                            return_acceptance=True)
    assert float(mala_acceptance) > 0.5
    assert float(hmc_acceptance) > 0.5


def test_traces_are_thinned_as_asked():
    """return_trace gives one state every thin steps, per chain"""
    log_prob, score = gaussian()
    x0 = torch.zeros(8, 4, dtype=torch.float64)
    trace = mala(log_prob, score, x0, 100, 0.2, generator=generator(0), return_trace=True,
                 thin=10)
    assert trace.shape == (10, 8, 4)


def test_student_t2_matches_its_quantile_function():
    """The inverse-CDF draw is an exact Student-t(2), and it respects the generator"""
    x = student_t2((200000,), generator=generator(0))
    assert torch.equal(x, student_t2((200000,), generator=generator(0)))

    # The median is 0 and the quartiles are +-sqrt(2/3), the exact df = 2 values
    quantiles = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    assert abs(float(quantiles[0]) + math.sqrt(2 / 3)) < 0.02
    assert abs(float(quantiles[1])) < 0.02
    assert abs(float(quantiles[2]) - math.sqrt(2 / 3)) < 0.02


def test_hmc_is_a_usable_baseline_on_a_single_gmm40_mode():
    """A local kernel scores well on shape and badly on coverage -- the expected classical floor"""
    target = GMM40()
    x0 = target.means[0][None].expand(1024, target.dim).contiguous()
    x = hmc(target.log_prob, target.score, x0, 100, 0.2, generator=generator(0))
    ret = target.compute_metrics(x, compute_standard_metrics=False)
    assert ret['emc'] < 0.3
    assert abs(float((x - target.means[target.mode_labels(x)]).std()) - 1.0) < 0.2
    assert math.isfinite(metrics.compute_wasserstein_distance(target.test_samples(n=1024), x))


def test_parallel_tempering_samples_a_gaussian_and_thins_as_asked():
    """The cold rung is the target, and n_keep states come back one every thin sweeps"""
    log_prob, score = gaussian()
    beta = geometric_ladder(8, 0.1)
    x0 = torch.zeros(8, 256, 4, dtype=torch.float64)
    x, diagnostics = parallel_tempering(log_prob, score, x0, beta, n_sweeps=400, step_size=0.5,
                                        adapt_sweeps=200, generator=generator(0), n_keep=40,
                                        thin=10, return_diagnostics=True)

    assert x.shape == (40, 256, 4)
    assert abs(float(x.mean())) < 0.05
    assert abs(float(x.std()) - 1.0) < 0.05
    assert float(diagnostics['swap_acceptance'].min()) > 0.05


def test_round_trips_cannot_exceed_the_length_of_the_ladder():
    """One swap pass per sweep moves a configuration one rung, which bounds the count

    A round trip is 2 * (n_rungs - 1) rungs of travel and there is one swap pass per sweep, so
    n_rungs configurations cannot complete more than n_rungs * n_sweeps / (2 * (n_rungs - 1)) of
    them between them. The counter used to flag the RUNG a configuration had left rather than the
    configuration itself, which overcounted by 2.6x on dw4 and by 11x on a ladder that always
    swaps -- and 'round trips' is the diagnostic every generated reference set is gated on.
    """
    log_prob, score = gaussian()
    n_rungs, n_sweeps = 8, 400
    beta = geometric_ladder(n_rungs, 0.5)
    x0 = torch.zeros(n_rungs, 64, 4, dtype=torch.float64)
    _, diagnostics = parallel_tempering(log_prob, score, x0, beta, n_sweeps=n_sweeps,
                                        step_size=0.5, generator=generator(0), n_keep=1,
                                        return_diagnostics=True)

    # A near-identical ladder swaps almost every pass, so this bound is nearly attained
    bound = n_rungs * n_sweeps / (2.0 * (n_rungs - 1))
    assert float(diagnostics['swap_acceptance'].min()) > 0.5
    assert 0.0 < float(diagnostics['round_trips'].min())
    assert float(diagnostics['round_trips'].max()) <= bound
