# The Metropolis kernels must sample their target, and be reproducible given a seed

# Libraries
import math
import torch
from samplingdb import GMM40, metrics
from samplingdb.targets.base import generator
from samplingdb.targets.kernels import mala, random_walk_mh, student_t2
from nrpt import hmc


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


def test_student_t2_is_finite_at_the_ends_of_the_uniform(monkeypatch):
    """torch.rand can return exactly 0, which the quantile maps to -inf unless u is clamped"""
    for dtype in [torch.float32, torch.float64]:
        monkeypatch.setattr(torch, 'rand', lambda shape, **kwargs: torch.zeros(
            shape, dtype=kwargs['dtype']))
        low = student_t2((4,), dtype=dtype)
        monkeypatch.setattr(torch, 'rand', lambda shape, **kwargs: torch.full(
            shape, 1.0 - torch.finfo(kwargs['dtype']).eps / 4, dtype=kwargs['dtype']))
        high = student_t2((4,), dtype=dtype)
        assert torch.isfinite(low).all() and torch.isfinite(high).all(), dtype
        assert (low < -1e6).all() and (high > 1e3).all(), dtype


def test_hmc_is_a_usable_baseline_on_a_single_gmm40_mode():
    """A local kernel scores well on shape and badly on coverage -- the expected classical floor"""
    target = GMM40()
    x0 = target.means[0][None].expand(1024, target.dim).contiguous()
    x = hmc(target.log_prob, target.score, x0, 100, 0.2, generator=generator(0))
    ret = target.compute_metrics(x, compute_standard_metrics=False)
    assert ret['emc'] < 0.3
    assert abs(float((x - target.means[target.mode_labels(x)]).std()) - 1.0) < 0.2
    assert math.isfinite(metrics.compute_wasserstein_distance(target.test_samples(n=1024), x))
