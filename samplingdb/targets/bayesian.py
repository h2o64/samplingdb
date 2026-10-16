# Tier 2 Bayesian posteriors, whose reference draws come from Stan

# Libraries
import math
import pickle
import numpy as np
import torch
from ..data import fetch_reference, fetch_url, read_rows
from ..metrics.weights import normalize_weights
from ..metrics.wasserstein import wasserstein_distance_1d
from .base import Target

# The datasets are published in the variational_sampling_methods hub, pinned to a commit
VSM_COMMIT = '32de62189abe07e96742735d91adbb959692d8c3'
VSM_URL = ('https://raw.githubusercontent.com/DenisBless/variational_sampling_methods/'
           + VSM_COMMIT + '/targets/data/')
VSM_CHECKSUMS = {
    'ionosphere_full.pkl': 'bcfc24ac32d840f04af7f890fc57a404b36e2ebb428b6a3884fbf42b3685bd5b',
    'sonar_full.pkl': '44832e63fe4d44a72ad6f91ccd086937f89014c9d5046bfe662d1b9101621b40',
    'german.data-numeric': '2752b044394958ab6dd193a0b56ca0f0b3a2d8bc7cb8c008e35a5e84bbec02f8',
    'breast_cancer.data': '4ba46e0483a3a4106f1dbd83b8a16dc85289da597d3d4a4ae03de07763a7c07b',
    'pines.csv': 'c57af2e9f2a38ed99a3884fbf81ad1b0c607adcf7749bad53a6df66ef313c89a',
}

# The four logistic-regression datasets: file, how to read it, and the expected dimension
LOGISTIC_DATASETS = {
    'ionosphere': {'file': 'ionosphere_full.pkl', 'dim': 35},
    'sonar': {'file': 'sonar_full.pkl', 'dim': 61},
    'german_credit': {'file': 'german.data-numeric', 'dim': 25},
    'breast_cancer': {'file': 'breast_cancer.data', 'dim': 31},
}

# The seeds germination data of Crowder (1978), as the hub inlines it
SEEDS_R = [10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3]
SEEDS_N = [39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7]
SEEDS_X1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SEEDS_X2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# inference_gym's brownian_motion_missing_middle_observations, with the middle ten unobserved
BROWNIAN_OBSERVED = [
    0.215926393866539, 0.11877139657735825, -0.0794544592499733, 0.03767748177051544,
    -0.2788584530353546, -0.14841559529304504, -0.32509058713912964, -0.22957904636859894,
    -0.44110897183418274, -0.09830781817436218,
    None, None, None, None, None, None, None, None, None, None,
    -0.878601610660553, -0.8373607397079468, -0.7384848594665527, -0.8939254283905029,
    -0.7774565815925598, -0.7023870944976807, -0.8777157068252563, -0.5185357332229614,
    -0.6948214173316956, -0.6202788949012756,
]


def softplus(x):
    """Evaluates log(1 + exp(x)) stably, which is where naive logistic likelihoods lose the tails"""
    return torch.nn.functional.softplus(x)


def standardize_and_pad(features):
    """Standardises every column and prepends a bias column

    Columns of zero variance keep their scale rather than dividing by zero, and the bias goes
    FIRST, so coordinate 0 of every logistic-regression target is the intercept.

    Args:
        * features (numpy.ndarray of shape (n, d)): Raw features

    Returns:
        * features (numpy.ndarray of shape (n, d + 1)): Standardised features with a bias column
    """
    mean, std = features.mean(axis=0), features.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return np.hstack([np.ones((len(features), 1)), (features - mean) / std])


def fetch_dataset(name):
    """Downloads one Tier 2 dataset from the pinned hub commit"""
    return fetch_url(VSM_URL + name, name, checksum=VSM_CHECKSUMS[name], folder='vsm-data')


class BayesianTarget(Target):
    """Base class for a Bayesian posterior sampled in an unconstrained parameterisation

    log_gamma(x) = log p(x) + sum_i log p(y_i | x) + the log-Jacobian of whatever transform makes
    the parameters unconstrained. Every subclass writes that Jacobian out: skipping it gives a
    wrong gradient at the boundary of a positivity constraint, which is the standard failure of a
    hand-rolled hierarchical posterior.

    WARNING: none of these has a known log Z, and none has a reference ensemble in the literature
    -- Blessing et al. 2024 report only ELBO/EUBO on them. The reference draws here are generated
    by Stan, by gen/stan/generate.py, and only written out once R-hat, ESS and divergence checks
    pass. They are published in this library's dataset repo, and test_samples() downloads them on
    first use.
    """

    truth_tier = 2
    reference_folder = 'stan'

    def test_samples(self, n=None, seed=None):
        """Returns Stan's reference draws, in the same unconstrained coordinates as log_prob

        The first call downloads the published draws (see data.fetch_reference), and later calls
        read the cache.

        Args:
            * n (int or None): Number of leading draws; None reads all (default is None)
            * seed (int or None): Unused, the reference set is a fixed file (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Reference draws

        Raises:
            * FileNotFoundError: When the draws are neither cached nor published, naming the
                command that generates them
            * ValueError: When n is more than the set holds, naming the command for a larger one
        """
        path = fetch_reference(
            self.reference_folder, self.reference_key(),
            'Generate them with: ./venv/bin/python gen/stan/generate.py --target ' + self.name
        )
        array = read_rows(path, n, './venv/bin/python gen/stan/generate.py --target ' + self.name
                          + ' --n_samples <n>')
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def generate_train_samples(self, n):
        """Runs the short-chain recipe

        A 500-step MALA chain from a prior draw, marked informal: it has no evaluation role, only a
        possible warm-start one. The observed data these posteriors condition on is on the target
        itself, which is this tier's real 'train' set.

        Args:
            * n (int): Number of samples, one per chain

        Returns:
            * x (torch.Tensor of shape (n, dim)): The biased samples
        """
        from ..baselines.classical import mala
        from .base import generator

        gen = generator(0)
        x0 = self.prior_scale * torch.randn(n, self.dim, generator=gen, dtype=self.dtype)
        return mala(self.log_prob, self.score, x0, n_steps=500, step_size=self.train_step_size,
                    generator=gen)

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False):
        """Replaces the panel with a per-coordinate moment comparison

        This is what a posterior is usually reported on: Stan gives means and standard deviations
        per parameter, and a sampler that gets those right on every coordinate has done the job.

        Arguments and returns are Target.compute_metrics'; the histogram is a dict of 'mean',
        'ref_mean', 'std' and 'ref_std', one entry per coordinate.
        """
        ref_samples = self.get_ref_samples(samples, ref_samples)
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                      compute_standard_metrics=compute_standard_metrics,
                                      skip_costly_metrics=skip_costly_metrics, generator=generator)
        weights = None if weights is None else normalize_weights(weights)
        mass = normalize_weights(weights, n=len(samples), dtype=samples.dtype,
                                 device=samples.device)
        mean = (mass[:, None] * samples).sum(0)
        variance = (mass[:, None] * (samples - mean) ** 2).sum(0)
        ref_mean, ref_std = ref_samples.mean(0), ref_samples.std(0)

        # Errors relative to the reference spread, so every coordinate is comparable
        ret['max_mean_err'] = float(((mean - ref_mean) / ref_std).abs().max())
        ret['mean_mean_err'] = float(((mean - ref_mean) / ref_std).abs().mean())
        ret['max_std_ratio_err'] = float((variance.sqrt() / ref_std - 1.0).abs().max())
        ret['w1_first_coordinate'] = float(wasserstein_distance_1d(ref_samples[:, 0], samples[:, 0],
                                                                  v_weights=weights))

        # Each posterior then adds the quantity ITS literature reports, which is never the
        # aggregate over coordinates: a hierarchical scale, an unobserved segment, a predictive
        ret.update(self.compute_parameter_metrics(samples, ref_samples, mean, ref_mean, ref_std,
                                                  weights=weights))
        if return_hist:
            return ret, {'mean': mean, 'ref_mean': ref_mean, 'std': variance.sqrt(),
                         'ref_std': ref_std}
        return ret

    def compute_parameter_metrics(self, samples, ref_samples, mean, ref_mean, ref_std,
                                  weights=None):
        """Returns the metrics on the parameters this particular posterior is judged on

        Empty by default; every subclass names its own, because the coordinates of a hierarchical
        posterior are not interchangeable and an aggregate over them hides the one that is hard.

        Args:
            * samples (torch.Tensor of shape (n, dim)): Samples to score
            * ref_samples (torch.Tensor of shape (m, dim)): Reference draws
            * mean (torch.Tensor of shape (dim,)): Weighted posterior mean of the samples
            * ref_mean (torch.Tensor of shape (dim,)): Reference posterior mean
            * ref_std (torch.Tensor of shape (dim,)): Reference posterior standard deviation
            * weights (torch.Tensor of shape (n,)): Weights of the samples (default is None)

        Returns:
            * metrics (dict): The per-posterior metrics
        """
        return {}


class LogisticRegression(BayesianTarget):
    """Bayesian logistic regression on one of the four UCI datasets

    log_gamma(w) = log N(w; 0, sigma_w^2 I) + sum_i [ y_i s_i - softplus(s_i) ],  s_i = w. u_i

    with standardised features, a prepended bias column, and sigma_w = 1. The parameters are
    unconstrained, so there is no Jacobian term.

    WARNING: the hub implements these four two different ways, and they are not the same target.
    Ionosphere and Sonar use the numpyro model above. German Credit and Breast Cancer use a
    hand-written one that scales the features by their standard deviation WITHOUT centring them,
    sets the prior scale to 10, and then omits the prior entirely (`log_posterior = log_likelihood
    # + log_prior`). samplingdb implements the documented model uniformly for all four, so its
    German Credit and Breast Cancer are proper posteriors rather than likelihoods -- and the two
    reference-draw files the hub ships for them (german_credit10k.npy,
    breastcancer_gt_with_lns_10k.npz) are draws from that other density and cannot be used to
    validate these.
    """

    train_step_size = 0.002
    prior_scale = 1.0

    def __init__(self, dataset='ionosphere', sigma_w=1.0, dtype=torch.float64):
        """Builds the logistic regression

        Args:
            * dataset (str): One of LOGISTIC_DATASETS' keys (default is 'ionosphere')
            * sigma_w (float): Prior standard deviation of every weight (default is 1.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        assert dataset in LOGISTIC_DATASETS

        # 1. Read the dataset and build the design matrix
        features, labels = load_logistic_dataset(dataset)
        dim = features.shape[1]
        assert dim == LOGISTIC_DATASETS[dataset]['dim']

        # Call the parent constructor
        super().__init__(dim, dataset.replace('_', '-'), dtype=dtype)

        # 2. Register the data, which is this tier's train split
        self.dataset = dataset
        self.sigma_w = sigma_w
        self.register_buffer('features', torch.as_tensor(features, dtype=dtype))
        self.register_buffer('labels', torch.as_tensor(labels, dtype=dtype))

    def spec(self):
        """Adds the prior width, the one constructor argument that moves the posterior"""
        spec = super().spec()
        spec['sigma_w'] = self.sigma_w
        return spec

    def logits(self, x):
        """Evaluates the linear predictor of every data point"""
        return x @ self.features.T

    def log_prob(self, x):
        """Evaluates log gamma(x), prior plus Bernoulli likelihood"""
        return self.chunked(self.log_prob_chunk, x, cost=len(self.labels))

    def log_prob_chunk(self, x):
        """Evaluates log gamma(x) on a batch small enough to hold every data point's logit"""
        logits = self.logits(x)
        likelihood = (self.labels * logits - softplus(logits)).sum(-1)
        prior = (-0.5 * (x / self.sigma_w) ** 2 - math.log(self.sigma_w)
                 - 0.5 * math.log(2 * math.pi)).sum(-1)
        return prior + likelihood

    def score(self, x):
        """Evaluates the score analytically: X^T (y - sigmoid(Xw)) minus the prior term"""
        return self.chunked(self.score_chunk, x, cost=len(self.labels))

    def score_chunk(self, x):
        """Evaluates the score on a batch small enough to hold every data point's logit"""
        residual = self.labels - torch.sigmoid(self.logits(x))
        return residual @ self.features - x / self.sigma_w ** 2

    def predictive_log_likelihood(self, x, weights=None):
        """Evaluates the held-out-style predictive log-likelihood of the observed data

        The one quantity computable on this tier without any reference samples, since it needs
        only the data already on the target.

        Args:
            * x (torch.Tensor of shape (n, dim)): Samples
            * weights (torch.Tensor of shape (n,)): Weights of the samples (default is None)

        Returns:
            * value (float): log of the weighted-average likelihood, per data point
        """
        logits = self.logits(x)
        per_sample = (self.labels * logits - softplus(logits)).sum(-1)
        mass = normalize_weights(weights, n=len(x), dtype=x.dtype, device=x.device)
        return float(torch.logsumexp(per_sample + torch.log(mass), dim=0)) / len(self.labels)

    def predictive_accuracy(self, x, weights=None):
        """Evaluates the accuracy of the posterior-mean predictive probability

        The companion of the predictive log-likelihood, and the number a logistic regression is
        actually reported with outside the sampling literature. It needs no reference draws either.

        Args:
            * x (torch.Tensor of shape (n, dim)): Samples
            * weights (torch.Tensor of shape (n,)): Weights of the samples (default is None)

        Returns:
            * accuracy (float): Fraction of data points the predictive labels correctly
        """
        mass = normalize_weights(weights, n=len(x), dtype=x.dtype, device=x.device)
        probability = (mass[:, None] * torch.sigmoid(self.logits(x))).sum(0)
        return float(((probability > 0.5).to(self.labels) == self.labels).to(x.dtype).mean())

    def compute_parameter_metrics(self, samples, ref_samples, mean, ref_mean, ref_std,
                                  weights=None):
        """Returns the two predictive numbers, which need no reference draws at all

        NOTE: predictive_log_likelihood is reported against the reference's own value rather than
        raw, so a bigger-is-better quantity becomes an error like everything else in the panel.
        """
        value = self.predictive_log_likelihood(samples, weights=weights)
        reference = self.predictive_log_likelihood(ref_samples)
        return {
            'predictive_log_likelihood': value,
            'predictive_log_likelihood_err': abs(value - reference),
            'predictive_accuracy': self.predictive_accuracy(samples, weights=weights),
        }


class Seeds(BayesianTarget):
    """The seeds random-effect logistic regression of Crowder (1978), dim 26

    tau ~ Gamma(0.01, 0.01);  a_0, a_1, a_2, a_12 ~ N(0, 10);  b_i ~ N(0, tau^-1/2)
    r_i ~ Binomial(N_i, sigmoid(a_0 + a_1 x1_i + a_2 x2_i + a_12 x1_i x2_i + b_i))

    Coordinates are ordered [a_0, a_1, a_2, a_12, b_1..b_21, log tau], and the last one is
    log tau rather than tau: the Gamma(0.01, 0.01) prior puts its mass over many orders of
    magnitude, so tau must be sampled on a log scale, and the log-Jacobian is part of log_gamma.

    NOTE: that vague Gamma prior is what makes this target a funnel -- small tau lets every b_i
    roam, large tau pins them at zero -- which is why the reference run raises Stan's adapt_delta
    rather than trusting the default.
    """

    train_step_size = 0.005
    prior_scale = 2.0
    n_groups = 21

    def __init__(self, dtype=torch.float64):
        """Builds the seeds posterior"""
        super().__init__(26, 'seeds', dtype=dtype)
        self.register_buffer('r', torch.as_tensor(SEEDS_R, dtype=dtype))
        self.register_buffer('n', torch.as_tensor(SEEDS_N, dtype=dtype))
        self.register_buffer('x1', torch.as_tensor(SEEDS_X1, dtype=dtype))
        self.register_buffer('x2', torch.as_tensor(SEEDS_X2, dtype=dtype))
        self.register_buffer('log_binomial', torch.as_tensor(
            [math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)
             for r, n in zip(SEEDS_R, SEEDS_N)], dtype=dtype))
        self.alpha, self.beta = 0.01, 0.01
        self.coefficient_scale = 10.0

        # The fixed-effect design matrix is a constant, so it is built once
        self.register_buffer('design', torch.stack(
            [torch.ones_like(self.x1), self.x1, self.x2, self.x1 * self.x2], dim=-1))

    def unpack(self, x):
        """Splits the flat vector into (coefficients, b, log_tau)"""
        return x[:, :4], x[:, 4:4 + self.n_groups], x[:, -1]

    def compute_parameter_metrics(self, samples, ref_samples, mean, ref_mean, ref_std,
                                  weights=None):
        """Returns the errors on log tau and on the four fixed effects, separately

        Every write-up of this model reports the random-effect scale and the four fixed effects,
        and they fail in different ways: the fixed effects are well identified and easy, while
        log tau is the neck of the funnel and is where a sampler that has not explored the small-tau
        region shows up. An aggregate over all 26 coordinates hides that, since 21 of them are the
        b_i, which follow whatever tau does.
        """
        return {
            'log_tau_mean_err': float((mean[-1] - ref_mean[-1]).abs() / ref_std[-1]),
            'log_tau_w1': float(wasserstein_distance_1d(ref_samples[:, -1], samples[:, -1],
                                                        v_weights=weights)),
            'max_fixed_effect_err': float(((mean[:4] - ref_mean[:4]) / ref_std[:4]).abs().max()),
        }

    def group_logits(self, coefficients, b):
        """Evaluates the logit of every group"""
        return coefficients @ self.design.T + b

    def log_prob(self, x):
        """Evaluates log gamma(x), including the log-Jacobian of log tau"""
        coefficients, b, log_tau = self.unpack(x)
        tau = torch.exp(log_tau)
        logits = self.group_logits(coefficients, b)

        # Gamma prior on tau, plus the Jacobian of tau = exp(log_tau)
        prior_tau = (self.alpha * math.log(self.beta) - math.lgamma(self.alpha)
                     + self.alpha * log_tau - self.beta * tau)
        prior_coefficients = (-0.5 * (coefficients / self.coefficient_scale) ** 2
                              - math.log(self.coefficient_scale)
                              - 0.5 * math.log(2 * math.pi)).sum(-1)
        prior_b = (0.5 * log_tau[:, None] - 0.5 * tau[:, None] * b ** 2
                   - 0.5 * math.log(2 * math.pi)).sum(-1)
        likelihood = (self.r * logits - self.n * softplus(logits) + self.log_binomial).sum(-1)
        return prior_tau + prior_coefficients + prior_b + likelihood

    def score(self, x):
        """Evaluates the score analytically, log tau included"""
        coefficients, b, log_tau = self.unpack(x)
        tau = torch.exp(log_tau)
        logits = self.group_logits(coefficients, b)
        residual = self.r - self.n * torch.sigmoid(logits)

        grad_coefficients = residual @ self.design - coefficients / self.coefficient_scale ** 2
        grad_b = residual - tau[:, None] * b
        grad_log_tau = (self.alpha - self.beta * tau + 0.5 * self.n_groups
                        - 0.5 * tau * (b ** 2).sum(-1))
        return torch.cat([grad_coefficients, grad_b, grad_log_tau[:, None]], dim=-1)


class Brownian(BayesianTarget):
    """inference_gym's Brownian motion with unknown scales and a missing middle, dim 32

    alpha_inn, alpha_obs ~ LogNormal(0, 2);  x_1 ~ N(0, alpha_inn);  x_i ~ N(x_{i-1}, alpha_inn)
    y_i ~ N(x_i, alpha_obs), observed for i in 1..10 and 21..30 only

    Coordinates are ordered [log alpha_inn, log alpha_obs, x_1..x_30]. Both scales are sampled on
    a log scale with the Jacobian included, and both LogNormal priors become N(0, 2^2) there.

    NOTE: the ten unobserved timesteps in the middle are what make this non-trivial -- they are
    constrained only by the random walk -- and the two unknown scales make it the tier's second
    funnel, alongside Seeds.

    NOTE: inference_gym publishes Stan-derived posterior means and standard deviations for this
    model, which gen/stan/generate.py checks its own run against: innovation scale 0.11985 +- sd
    0.04668, observation scale 0.10105 +- sd 0.04443.
    """

    train_step_size = 0.002
    prior_scale = 1.0
    n_steps = 30

    def __init__(self, dtype=torch.float64):
        """Builds the Brownian motion posterior"""
        super().__init__(32, 'brownian', dtype=dtype)
        observed = [value is not None for value in BROWNIAN_OBSERVED]
        values = [0.0 if value is None else value for value in BROWNIAN_OBSERVED]
        self.register_buffer('observed', torch.as_tensor(observed))
        self.register_buffer('y', torch.as_tensor(values, dtype=dtype))
        self.prior_log_scale = 2.0

    def unpack(self, x):
        """Splits the flat vector into (log_alpha_inn, log_alpha_obs, locations)"""
        return x[:, 0], x[:, 1], x[:, 2:]

    def compute_parameter_metrics(self, samples, ref_samples, mean, ref_mean, ref_std,
                                  weights=None):
        """Returns the errors on the two scales, the unobserved middle, and the observed locations

        inference_gym reports per-parameter expectations for this model, and the three groups are
        not equally hard: the observed locations are pinned by their data, the two scales are the
        funnel, and the ten unobserved middle steps are constrained only through the random walk,
        so they are where a sampler that has the scales slightly wrong shows it most.

        NOTE: what these should converge to is contested upstream. inference_gym's published Stan
        ground truth gives an innovation scale of 0.11985 and an observation scale of 0.10274,
        while a 10-million-step NUTS run reported in tensorflow/probability issue 1992 gives
        0.11526 and 0.10105 for the same model. Our own Stan run gives 0.1133 and 0.1137. The model
        text here matches theirs line for line (LogNormal(0, 2) on both scales, NaN-masked
        observations, 30 steps with ten missing) and our torch and Stan densities agree to 1e-14,
        so the remaining gap is in the observation array rather than in the density. Read these
        errors against OUR reference, which is the one in this cache.
        """
        middle = torch.as_tensor([index for index, value in enumerate(BROWNIAN_OBSERVED)
                                  if value is None], device=samples.device) + 2
        observed = torch.as_tensor([index for index, value in enumerate(BROWNIAN_OBSERVED)
                                    if value is not None], device=samples.device) + 2
        error = ((mean - ref_mean) / ref_std).abs()
        return {
            'scale_mean_err': float(error[:2].max()),
            'missing_middle_mean_err': float(error[middle].max()),
            'observed_mean_err': float(error[observed].max()),
        }

    def log_prob(self, x):
        """Evaluates log gamma(x), including both log-Jacobians"""
        log_inn, log_obs, locations = self.unpack(x)
        inn, obs = torch.exp(log_inn), torch.exp(log_obs)

        # LogNormal(0, 2) on each scale becomes N(0, 2^2) once the Jacobian is folded in
        prior_scales = (-0.5 * (log_inn / self.prior_log_scale) ** 2
                        - 0.5 * (log_obs / self.prior_log_scale) ** 2
                        - 2 * math.log(self.prior_log_scale) - math.log(2 * math.pi))

        # The random walk, first step from zero
        increments = torch.cat([locations[:, :1], locations[:, 1:] - locations[:, :-1]], dim=-1)
        walk = (-0.5 * (increments / inn[:, None]) ** 2 - log_inn[:, None]
                - 0.5 * math.log(2 * math.pi)).sum(-1)

        # The observations, only where they exist
        residual = (locations - self.y) * self.observed
        likelihood = ((-0.5 * (residual / obs[:, None]) ** 2 - log_obs[:, None]
                       - 0.5 * math.log(2 * math.pi)) * self.observed).sum(-1)
        return prior_scales + walk + likelihood

    def score(self, x):
        """Evaluates the score analytically, both log scales included"""
        log_inn, log_obs, locations = self.unpack(x)
        inn, obs = torch.exp(log_inn), torch.exp(log_obs)
        increments = torch.cat([locations[:, :1], locations[:, 1:] - locations[:, :-1]], dim=-1)
        residual = (locations - self.y) * self.observed

        # The walk pulls each location towards its neighbours, the data towards y
        scaled = increments / inn[:, None] ** 2
        grad_locations = -scaled + torch.cat([scaled[:, 1:], torch.zeros_like(scaled[:, :1])],
                                            dim=-1)
        grad_locations = grad_locations - residual / obs[:, None] ** 2

        # Each log scale gets its own Jacobian term, one per contributing Gaussian
        n_observed = self.observed.sum()
        grad_log_inn = (-log_inn / self.prior_log_scale ** 2 - self.n_steps
                        + (increments ** 2).sum(-1) / inn ** 2)
        grad_log_obs = (-log_obs / self.prior_log_scale ** 2 - n_observed
                        + (residual ** 2).sum(-1) / obs ** 2)
        return torch.cat([grad_log_inn[:, None], grad_log_obs[:, None], grad_locations], dim=-1)


class LGCP(BayesianTarget):
    """The log Gaussian Cox process on Finnish pine saplings, dim 1600

    log_gamma(x) = log N(x; mu, K) + sum_i [ x_i c_i - a exp(x_i) ]

    on a 40 x 40 grid, with K(m, n) = sigma^2 exp(-||m - n|| / (40 beta)), sigma^2 = 1.91,
    beta = 1/33, mu_i = log(126) - sigma^2 / 2 and a = 1/1600.

    NOTE: whitened=False, the default, is what Blessing et al. 2024 ran, and the whitened
    parameterisation x = L e + mu is a DIFFERENT geometry whose numbers do not compare -- it
    carries its own name for that reason. Both are catalogued, because the whitened one is the only
    one with a reference ensemble: the direct parameterisation needs a dense 1600 x 1600 triangular
    solve per gradient and a 300-draw Stan run did not finish in 20 minutes, while the whitened one
    took 151 s for 6000 draws at R-hat 1.0044.
    """

    train_step_size = 0.0005
    prior_scale = 1.0

    def __init__(self, n_grid=40, signal_variance=1.91, beta=1.0 / 33.0, whitened=False,
                 dtype=torch.float64):
        """Builds the LGCP posterior

        Args:
            * n_grid (int): Cells per dimension, so dim is n_grid^2 (default is 40)
            * signal_variance (float): sigma^2 of the kernel (default is 1.91)
            * beta (float): Correlation length, in units of n_grid (default is 1/33)
            * whitened (bool): Whether to sample in the whitened parameterisation
                (default is False)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        super().__init__(n_grid * n_grid, 'lgcp' + ('-whitened' if whitened else ''), dtype=dtype)

        # 1. The grid, its kernel and the Cholesky factor, computed once
        self.n_grid = n_grid
        self.whitened = whitened
        self.area = 1.0 / (n_grid * n_grid)
        axis = (torch.arange(n_grid, dtype=dtype) + 0.5) / n_grid
        grid = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), dim=-1).reshape(-1, 2)
        distance = torch.cdist(grid, grid)
        kernel = signal_variance * torch.exp(-distance / beta)
        self.register_buffer('cholesky', torch.linalg.cholesky(kernel))
        self.register_buffer('mean', torch.full((self.dim,), math.log(126.0) - signal_variance / 2,
                                                dtype=dtype))

        # 2. The binned counts of the pine saplings
        self.register_buffer('counts', torch.as_tensor(load_pines(n_grid), dtype=dtype))
        self.signal_variance = signal_variance
        self.beta = beta

    def spec(self):
        """Adds the kernel's variance and correlation length, which n_grid does not cover"""
        spec = super().spec()
        spec.update({'signal_variance': self.signal_variance, 'beta': self.beta,
                     'whitened': self.whitened})
        return spec

    def positions(self, x):
        """Maps the sampled coordinates to the latent field, which differ when whitened"""
        if self.whitened:
            return x @ self.cholesky.T + self.mean
        return x

    def compute_parameter_metrics(self, samples, ref_samples, mean, ref_mean, ref_std,
                                  weights=None):
        """Returns the total expected count and the field's nearest-neighbour correlation

        A 1600-dimensional field has no single interesting coordinate, so the two numbers reported
        here are the ones a point-process fit is read on: the total intensity sum_i a exp(x_i),
        which the 126 observed pines pin, and the mean product of neighbouring cells, which is what
        the kernel's correlation length controls. Both are functionals of the whole field, and a
        sampler can get every marginal mean right while getting the second one wrong.
        """
        mass = normalize_weights(weights, n=len(samples), dtype=samples.dtype,
                                 device=samples.device)
        field = self.positions(samples)
        ref_field = self.positions(ref_samples)

        intensity = float((mass * (self.area * torch.exp(field)).sum(-1)).sum())
        ref_intensity = float((self.area * torch.exp(ref_field)).sum(-1).mean())

        def neighbour_correlation(values, weight=None):
            grid = values.reshape(len(values), self.n_grid, self.n_grid)
            product = (grid[:, 1:] * grid[:, :-1]).mean(dim=(1, 2)) \
                + (grid[:, :, 1:] * grid[:, :, :-1]).mean(dim=(1, 2))
            if weight is None:
                return float(product.mean()) / 2.0
            return float((weight * product).sum()) / 2.0

        correlation = neighbour_correlation(field, mass)
        ref_correlation = neighbour_correlation(ref_field)
        return {
            'total_intensity': intensity,
            'total_intensity_err': abs(intensity - ref_intensity),
            'neighbour_correlation_err': abs(correlation - ref_correlation),
        }

    def log_prob(self, x):
        """Evaluates log gamma(x)"""
        field = self.positions(x)
        poisson = (field * self.counts - self.area * torch.exp(field)).sum(-1)
        if self.whitened:
            return -0.5 * (x ** 2).sum(-1) + poisson
        delta = torch.linalg.solve_triangular(self.cholesky, (x - self.mean).T, upper=False)
        return -0.5 * (delta ** 2).sum(0) + poisson

    def score(self, x):
        """Evaluates the score analytically"""
        field = self.positions(x)
        poisson = self.counts - self.area * torch.exp(field)
        if self.whitened:
            return -x + poisson @ self.cholesky
        delta = torch.cholesky_solve((x - self.mean).T, self.cholesky).T
        return -delta + poisson


def load_logistic_dataset(name):
    """Reads one logistic-regression dataset and returns (features, labels)

    Args:
        * name (str): One of LOGISTIC_DATASETS' keys

    Returns:
        * features (numpy.ndarray of shape (n, dim)): Standardised features with a bias column
        * labels (numpy.ndarray of shape (n,)): Labels in {0, 1}
    """
    path = fetch_dataset(LOGISTIC_DATASETS[name]['file'])
    if name in ['ionosphere', 'sonar']:
        with open(path, 'rb') as handle:
            features, labels = pickle.load(handle)
        labels = (np.asarray(labels) + 1) // 2
    elif name == 'german_credit':
        data = np.loadtxt(path)
        features, labels = data[:, :-1], data[:, -1] - 1
    else:
        data = np.loadtxt(path)
        features, labels = data[:, 2:], data[:, 1]
    return standardize_and_pad(np.asarray(features, dtype=np.float64)), \
        np.asarray(labels, dtype=np.float64)


def load_pines(n_grid):
    """Reads the Finnish pine sapling locations and bins them onto the grid

    NOTE: points on an upper edge fall into the lower bin, which is the convention the reference
    implementation uses and a source of off-by-one differences otherwise.

    Args:
        * n_grid (int): Cells per dimension

    Returns:
        * counts (numpy.ndarray of shape (n_grid * n_grid,)): Counts per cell, row-major
    """
    path = fetch_dataset('pines.csv')
    points = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(1, 2))
    index = np.clip((points * n_grid).astype(int), 0, n_grid - 1)
    counts = np.zeros((n_grid, n_grid))
    np.add.at(counts, (index[:, 0], index[:, 1]), 1.0)
    return counts.reshape(-1)
