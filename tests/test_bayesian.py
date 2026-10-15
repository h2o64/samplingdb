# The seven Bayesian posteriors: their densities, their Jacobians, and Stan's reference draws

# Libraries
import math
import numpy as np
import pytest
import torch
from samplingdb import Brownian, LGCP, LogisticRegression, Seeds
from samplingdb.data import cache_dir
from samplingdb.targets import score_autograd
from samplingdb.targets.base import generator
from samplingdb.targets.bayesian import (
    BROWNIAN_OBSERVED, LOGISTIC_DATASETS, SEEDS_N, SEEDS_R, VSM_CHECKSUMS, VSM_COMMIT,
    standardize_and_pad,
)

# Every target, and whether building it needs a dataset download
LOGISTIC = list(LOGISTIC_DATASETS)


def has_dataset(name):
    """Returns whether one Tier 2 dataset is already cached"""
    return (cache_dir() / 'vsm-data' / name).exists()


def logistic(name):
    """Builds one logistic regression, skipping when its dataset is not cached"""
    if not has_dataset(LOGISTIC_DATASETS[name]['file']):
        pytest.skip('the ' + name + ' dataset is not cached')
    return LogisticRegression(dataset=name)


def test_the_datasets_are_pinned_to_a_commit():
    """A moving branch would let the data change under a fixed name"""
    assert len(VSM_COMMIT) == 40
    assert len(VSM_CHECKSUMS) == 5
    for name, digest in VSM_CHECKSUMS.items():
        assert len(digest) == 64, name


def test_standardize_and_pad_puts_the_bias_first():
    """Coordinate 0 of every logistic regression is the intercept"""
    features = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    padded = standardize_and_pad(features)
    assert padded.shape == (3, 3)
    assert np.allclose(padded[:, 0], 1.0)

    # The first column is standardised, and the constant one is left alone rather than divided by 0
    assert abs(padded[:, 1].mean()) < 1e-12 and abs(padded[:, 1].std() - 1.0) < 1e-12
    assert np.isfinite(padded).all()
    assert np.allclose(padded[:, 2], 0.0)


def test_seeds_and_brownian_data_match_their_sources():
    """The inlined datasets, against Crowder (1978) and inference_gym"""
    assert len(SEEDS_R) == len(SEEDS_N) == 21
    assert all(r <= n for r, n in zip(SEEDS_R, SEEDS_N))
    assert SEEDS_R[:3] == [10, 23, 23] and SEEDS_N[:3] == [39, 62, 81]

    # inference_gym's series is 30 long with the middle ten unobserved
    assert len(BROWNIAN_OBSERVED) == 30
    missing = [i for i, value in enumerate(BROWNIAN_OBSERVED) if value is None]
    assert missing == list(range(10, 20))


@pytest.mark.parametrize('name', LOGISTIC)
def test_logistic_regression_matches_its_dimension_and_score(name):
    """dim is features + 1, and the score is X^T(y - sigmoid) minus the prior term"""
    target = logistic(name)
    assert target.dim == LOGISTIC_DATASETS[name]['dim']
    assert set(target.labels.tolist()) <= {0.0, 1.0}
    assert torch.allclose(target.features[:, 0], torch.ones(len(target.labels),
                                                            dtype=torch.float64))

    x = 0.3 * torch.randn(8, target.dim, generator=generator(0), dtype=torch.float64)
    assert torch.allclose(target.score(x), score_autograd(target, x), rtol=1e-9, atol=1e-9)

    # At w = 0 every logit is 0, so the likelihood is n log(1/2) and the score is X^T(y - 1/2)
    zero = torch.zeros(1, target.dim, dtype=torch.float64)
    expected = len(target.labels) * math.log(0.5) - 0.5 * target.dim * math.log(2 * math.pi)
    assert abs(float(target.log_prob(zero)) - expected) < 1e-9


def test_seeds_log_jacobian_is_present():
    """Dropping the log tau Jacobian is the standard error here, so pin it directly"""
    target = Seeds()
    assert target.dim == 26
    x = 0.3 * torch.randn(8, target.dim, generator=generator(0), dtype=torch.float64)
    assert torch.allclose(target.score(x), score_autograd(target, x), rtol=1e-9, atol=1e-9)

    # A target built without the Jacobian differs by exactly log tau, and its score by 1
    class NoJacobian(Seeds):
        def log_prob(self, x):
            return super().log_prob(x) - x[:, -1]

    without = NoJacobian()
    assert torch.allclose(target.log_prob(x) - without.log_prob(x), x[:, -1])
    assert torch.allclose(target.score(x)[:, -1] - score_autograd(without, x)[:, -1],
                          torch.ones(len(x), dtype=torch.float64))


def test_brownian_only_uses_the_observed_timesteps():
    """The middle ten locations are constrained by the walk alone"""
    target = Brownian()
    assert target.dim == 32
    assert int(target.observed.sum()) == 20

    x = 0.3 * torch.randn(8, target.dim, generator=generator(0), dtype=torch.float64)
    assert torch.allclose(target.score(x), score_autograd(target, x), rtol=1e-9, atol=1e-9)

    # Changing y where nothing is observed cannot change the density
    moved = Brownian()
    moved.y = moved.y.clone()
    moved.y[12] = 5.0
    assert torch.allclose(target.log_prob(x), moved.log_prob(x))


def test_lgcp_kernel_and_counts():
    """The 40 x 40 grid, the pinned kernel constants, and the binned pines"""
    if not has_dataset('pines.csv'):
        pytest.skip('the pines dataset is not cached')
    target = LGCP()
    assert target.dim == 1600
    assert abs(float(target.mean[0]) - (math.log(126.0) - 1.91 / 2)) < 1e-12
    assert abs(target.area - 1.0 / 1600) < 1e-12
    assert float(target.counts.sum()) == 126.0

    x = 0.3 * torch.randn(4, target.dim, generator=generator(0), dtype=torch.float64)
    assert torch.allclose(target.score(x), score_autograd(target, x), rtol=1e-9, atol=1e-9)

    # The whitened parameterisation is a different geometry, hence a different name. It is
    # catalogued because it is the only one Stan can sample in a sane budget
    whitened = LGCP(whitened=True)
    assert whitened.name != target.name
    assert torch.allclose(whitened.score(x), score_autograd(whitened, x), rtol=1e-9, atol=1e-9)

    # Both describe the same field: a whitened draw maps to the direct one
    field = x @ target.cholesky.T + target.mean
    assert torch.allclose(whitened.positions(x), field)


def test_test_samples_says_how_to_generate_them():
    """A missing reference set names the command rather than failing obscurely"""
    target = Seeds()
    if target.reference_path().exists():
        pytest.skip('the Stan reference draws are already generated')
    with pytest.raises(FileNotFoundError, match='gen/stan/generate.py'):
        target.test_samples()


@pytest.mark.parametrize('name', LOGISTIC + ['seeds', 'brownian'])
def test_the_stan_reference_is_usable_where_it_exists(name):
    """Stan's draws are in samplingdb's own coordinates, so the panel runs on them directly"""
    target = logistic(name) if name in LOGISTIC else {'seeds': Seeds, 'brownian': Brownian}[name]()
    if not target.reference_path().exists():
        pytest.skip('no Stan reference draws for ' + target.name)

    reference = target.test_samples()
    assert reference.shape[1] == target.dim
    assert len(reference) >= 4000
    assert torch.isfinite(target.log_prob(reference[:64])).all()

    # Two halves of Stan's own output agree, which is the floor the panel is read against
    floor = target.compute_metrics(reference[:2000], ref_samples=reference[2000:4000],
                                   compute_standard_metrics=False)
    assert floor['max_mean_err'] < 0.3
    assert floor['max_std_ratio_err'] < 0.3

    # The prior-initialised short chain is much worse on the same panel
    biased = target.compute_metrics(target.train_samples(n=512), ref_samples=reference[:2000],
                                    compute_standard_metrics=False)
    assert biased['max_mean_err'] > floor['max_mean_err']


def test_predictive_log_likelihood_needs_no_reference():
    """The one Tier 2 quantity computable without ground truth, since the data is on the target"""
    target = logistic('ionosphere')
    x = 0.1 * torch.randn(64, target.dim, generator=generator(0), dtype=torch.float64)
    value = target.predictive_log_likelihood(x)
    assert value < 0.0
    assert math.isfinite(value)

    # Weighting onto a single sample matches that sample's own likelihood
    weights = torch.zeros(64, dtype=torch.float64)
    weights[3] = 1.0
    single = target.predictive_log_likelihood(x[3:4])
    assert abs(target.predictive_log_likelihood(x, weights=weights) - single) < 1e-9


@pytest.mark.parametrize('name', LOGISTIC)
def test_logistic_reports_its_predictive_pair(name):
    """The two numbers a logistic regression is read on, neither needing reference draws"""
    target = logistic(name)
    x = 0.1 * torch.randn(64, target.dim, generator=generator(0), dtype=torch.float64)
    accuracy = target.predictive_accuracy(x)
    assert 0.0 <= accuracy <= 1.0

    # Predicting from the posterior beats predicting from a prior draw
    if target.reference_path().exists():
        reference = target.test_samples(n=2000)
        assert target.predictive_accuracy(reference) > accuracy
        assert target.predictive_log_likelihood(reference) > target.predictive_log_likelihood(x)

        # And the panel reports them, as an error against the reference's own value
        ret = target.compute_metrics(reference[:1000], ref_samples=reference[1000:],
                                     compute_standard_metrics=False)
        assert {'predictive_log_likelihood', 'predictive_log_likelihood_err',
                'predictive_accuracy'} <= set(ret)

        # Two halves of Stan's own draws agree on it; sonar is the noisiest of the four, being the
        # smallest dataset at the highest dimension (208 points, 61 parameters)
        assert ret['predictive_log_likelihood_err'] < 0.02


def test_seeds_panel_separates_the_funnel_parameter_from_the_fixed_effects():
    """log tau is the neck; the four fixed effects are easy, and an aggregate would hide that"""
    target = Seeds()
    if not target.reference_path().exists():
        pytest.skip('no Stan reference draws for seeds')
    reference = target.test_samples()
    floor = target.compute_metrics(reference[:2000], ref_samples=reference[2000:4000],
                                   compute_standard_metrics=False)
    biased = target.compute_metrics(target.train_samples(n=512), ref_samples=reference[:2000],
                                    compute_standard_metrics=False)
    assert {'log_tau_mean_err', 'log_tau_w1', 'max_fixed_effect_err'} <= set(floor)

    # The short chain misses the small-tau region, which is exactly what log_tau_mean_err sees
    assert floor['log_tau_mean_err'] < 0.1
    assert biased['log_tau_mean_err'] > 20 * floor['log_tau_mean_err']


def test_brownian_panel_splits_the_scales_from_the_unobserved_middle():
    """Three groups of coordinates, constrained by three different things"""
    target = Brownian()
    if not target.reference_path().exists():
        pytest.skip('no Stan reference draws for brownian')
    reference = target.test_samples()
    floor = target.compute_metrics(reference[:2000], ref_samples=reference[2000:4000],
                                   compute_standard_metrics=False)
    assert {'scale_mean_err', 'missing_middle_mean_err', 'observed_mean_err'} <= set(floor)
    for name in ['scale_mean_err', 'missing_middle_mean_err', 'observed_mean_err']:
        assert floor[name] < 0.15, name

    # The unobserved middle is the widest part of the posterior, which the reference itself shows
    spread = reference.std(0)
    assert float(spread[12:22].mean()) > float(spread[2:12].mean())


def test_lgcp_panel_reports_the_total_intensity_and_the_field_correlation():
    """A 1600-dimensional field is read on functionals, not on any one coordinate"""
    target = LGCP(whitened=True)
    if not target.reference_path().exists():
        pytest.skip('no Stan reference draws for the whitened LGCP')
    reference = target.test_samples()
    ret = target.compute_metrics(reference[:2000], ref_samples=reference[2000:4000],
                                 compute_standard_metrics=False)
    assert {'total_intensity', 'total_intensity_err', 'neighbour_correlation_err'} <= set(ret)

    # The total intensity is the number of trees the data holds, which is what pins it
    assert abs(ret['total_intensity'] - float(target.counts.sum())) < 5.0
    assert ret['total_intensity_err'] < 1.0

    # The biased split gets the total wrong by an order of magnitude more
    biased = target.compute_metrics(target.train_samples(n=512), ref_samples=reference[:2000],
                                    compute_standard_metrics=False)
    assert biased['total_intensity_err'] > 10 * ret['total_intensity_err']
