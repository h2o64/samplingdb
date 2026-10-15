# Mixture targets, the corpus's mode-coverage benchmarks

# Libraries
import math
import torch
from ..data import load_asset
from ..baselines.classical import random_walk_mh, student_t2
from .base import ExactTarget, generator
from .modes import ModeCoverage

# Half of log(2 pi), the constant every Gaussian log-density below carries
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def log_prob_gaussian(x, mean, variance):
    """Evaluates the log-density of a diagonal Gaussian, broadcasting over leading axes

    Args:
        * x (torch.Tensor of shape (..., dim)): Evaluation points
        * mean (torch.Tensor of shape (..., dim)): Mean
        * variance (torch.Tensor of shape (..., dim)): Diagonal of the covariance

    Returns:
        * log_prob (torch.Tensor of shape (...,)): Log-density
    """
    log_prob = -0.5 * torch.sum(torch.square(x - mean) / variance, dim=-1)
    log_prob = log_prob - 0.5 * x.shape[-1] * math.log(2 * math.pi)
    return log_prob - 0.5 * torch.log(variance).sum(dim=-1)


# Every class here is a Tier 0 exact-draw toy in two dimensions: there is no sampler to speed up,
# so float32 would buy nothing and would cost the exactness checks that pin their specifications --
# unit isotropic components, modes on a circle with equal weights. float64 by decision, not by
# omission. The periodic pair mixtures that DO have a sampler live in physics.py and are float32.
class Mixture(ModeCoverage, ExactTarget):
    """Base class for the isotropic-component location mixtures

    A mixture's log_prob is a logsumexp over components, and its score is the responsibility-
    weighted average of the per-component scores -- both written out analytically here, since
    the responsibilities are a softmax of quantities log_prob already computes.

    Subclasses provide component_log_prob and component_score. The mode descriptor is the nearest
    component, so the coverage panel comes from ModeCoverage.
    """

    truth_tier = 0

    def __init__(self, dim, name, weights, means, dtype=torch.float64):
        """Builds the mixture

        Args:
            * dim (int): Dimension
            * name (str): Version tag
            * weights (torch.Tensor of shape (n_modes,)): Mixture weights, normalized here
            * means (torch.Tensor of shape (n_modes, dim)): Component locations
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # Call the parent constructor
        super().__init__(dim, name, log_Z=0.0, dtype=dtype)

        # Register everything to the buffer
        self.n_modes = weights.shape[0]
        self.register_buffer('weights', (weights / weights.sum()).to(dtype))
        self.register_buffer('means', means.to(dtype))

    def spec(self):
        """Adds the mixture weights and the component locations, which ARE the specification"""
        spec = super().spec()
        spec.update({'weights': self.weights, 'means': self.means})
        return spec

    @property
    def mode_weights(self):
        """Returns the true weight of every mode, which for a mixture is its mixture weights"""
        return self.weights

    def component_log_prob(self, x):
        """Evaluates log pi_k(x) for every component

        Args:
            * x (torch.Tensor of shape (n, dim)): Evaluation points

        Returns:
            * log_prob (torch.Tensor of shape (n, n_modes)): Per-component log-densities
        """
        raise NotImplementedError('component_log_prob is not implemented.')

    def component_score(self, x):
        """Evaluates grad log pi_k(x) for every component

        Args:
            * x (torch.Tensor of shape (n, dim)): Evaluation points

        Returns:
            * score (torch.Tensor of shape (n, n_modes, dim)): Per-component scores
        """
        raise NotImplementedError('component_score is not implemented.')

    def responsibilities(self, x, return_log_prob=False):
        """Evaluates the posterior probability of every component at x

        Args:
            * x (torch.Tensor of shape (n, dim)): Evaluation points
            * return_log_prob (bool): Whether to also return the mixture log-density
                (default is False)

        Returns:
            * probs (torch.Tensor of shape (n, n_modes)): Responsibilities, summing to one
            if return_log_prob:
                * log_prob (torch.Tensor of shape (n,)): Mixture log-density
        """
        log_probs = self.component_log_prob(x) + torch.log(self.weights)
        probs = torch.softmax(log_probs, dim=-1)
        if return_log_prob:
            return probs, torch.logsumexp(log_probs, dim=-1)
        return probs

    def log_prob(self, x):
        """Evaluates log gamma(x), normalised here so log Z = 0"""
        return self.chunked(self.log_prob_chunk, x, cost=self.n_modes)

    def log_prob_chunk(self, x):
        """Evaluates log gamma(x) on a batch small enough to hold its per-component densities"""
        return torch.logsumexp(self.component_log_prob(x) + torch.log(self.weights), dim=-1)

    def score(self, x):
        """Evaluates the score of the mixture at x, analytically"""
        return self.chunked(self.score_chunk, x, cost=self.n_modes * self.dim)

    def score_chunk(self, x):
        """Evaluates the score on a batch small enough to hold its per-component scores"""
        probs = self.responsibilities(x)
        return torch.sum(probs[..., None] * self.component_score(x), dim=1)

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the mixture at x, sharing the softmax"""
        return self.chunked(self.log_prob_and_grad_chunk, x, cost=self.n_modes * self.dim)

    def log_prob_and_grad_chunk(self, x):
        """Evaluates both on a batch, sharing the softmax between them"""
        probs, log_prob = self.responsibilities(x, return_log_prob=True)
        return log_prob, torch.sum(probs[..., None] * self.component_score(x), dim=1)

    def mode_labels(self, x):
        """Returns argmax_k pi_k(x), the mode descriptor that makes coverage computable"""
        return self.component_log_prob(x).argmax(dim=-1)

    def generate_train_samples(self, n):
        """Runs the one-chain-per-mode random-walk recipe

        Chains are spread evenly over the components, each starting at its own component's mean,
        and take 200 Metropolis steps of size 2.4 * min_scale / sqrt(dim) -- the optimal
        random-walk scaling for the narrowest component. Every mode is therefore VISITED, with the
        correct local shape, and none is missed. What is wrong is the PROPORTIONS: the chains are
        allocated by index rather than by weight, so a mode carrying 1% of the mass gets the same
        share of the samples as one carrying 50%, and no chain lives long enough to cross a gap and
        correct that. On TwoModes that is a 50/50 split against a true 2/3 : 1/3, i.e. a mode-weight
        TV of 0.33 where an exact draw of the same size gives 0.003.

        WARNING: for a mixture whose weights are EQUAL, allocating chains per mode is not an error
        -- it is stratified sampling, and its mode weights come out right by construction. GMM-40
        and MoS are in that case: measured mode-weight TV 0.13 for the train split against 0.15 for
        an exact draw of the same size, and W2 1.67 against 1.71. Their train split is biased only
        in that it contains no mode-to-mode dynamics whatsoever, every chain having stayed where it
        started. Read those two as cheap local data, not as a mode-proportion stress test; the
        unequal-weight mixtures (TwoModes, BracketTwoModes, ManyModes) are the stress test.

        Args:
            * n (int): Number of samples, one per chain

        Returns:
            * x (torch.Tensor of shape (n, dim)): The biased samples
        """
        gen = generator(0)
        component = torch.arange(n) % self.n_modes
        x0 = self.means.cpu()[component].clone()
        step_size = 2.4 * self.min_scale / math.sqrt(self.dim)
        return random_walk_mh(self.log_prob, x0, n_steps=200, step_size=step_size,
                              proposal=self.train_proposal, generator=gen)


class MOG(Mixture):
    """A mixture of Gaussians with diagonal covariances"""

    train_proposal = 'normal'

    def __init__(self, weights, means, variances, name='mog', prior_scale=None,
                 dtype=torch.float64):
        """Builds the mixture of Gaussians

        Args:
            * weights (torch.Tensor of shape (n_modes,)): Weights
            * means (torch.Tensor of shape (n_modes, dim)): Means
            * variances (torch.Tensor of shape (n_modes, dim)): Diagonals of the covariances
            * name (str): Short identifier (default is 'mog')
            * prior_scale (float or None): Scale a Gaussian reference should use; None covers the
                spread of the means (default is None)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # Call the parent constructor
        super().__init__(means.shape[-1], name, weights, means, dtype=dtype)

        # Register the covariances and the derived scales
        self.register_buffer('variances', variances.to(dtype))
        self.min_scale = float(variances.min().sqrt())
        if prior_scale is None:
            prior_scale = float(means.abs().max() + variances.max().sqrt())
        self.prior_scale = prior_scale

        # Build the distribution
        self.build_dist()

    def spec(self):
        """Adds the component covariances, which carry the scale and the conditioning"""
        spec = super().spec()
        spec['variances'] = self.variances
        return spec

    def build_dist(self):
        """Builds the inner torch.distributions.MixtureSameFamily object"""
        self.dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(self.weights),
            component_distribution=torch.distributions.Independent(
                base_distribution=torch.distributions.Normal(
                    loc=self.means, scale=torch.sqrt(self.variances), validate_args=False
                ),
                reinterpreted_batch_ndims=1, validate_args=False
            ), validate_args=False
        )

    def mean(self):
        """Returns the mean of the mixture"""
        return self.dist.mean

    def variance(self):
        """Returns the variance of the mixture"""
        return self.dist.variance

    def component_log_prob(self, x):
        """Evaluates the per-component diagonal-Gaussian log-densities"""
        return log_prob_gaussian(x[:, None, :], self.means, self.variances)

    def component_score(self, x):
        """Evaluates the per-component scores, -(x - mu_k) / var_k"""
        return -(x[:, None, :] - self.means) / self.variances

    def sample(self, sample_shape, generator=None):
        """Draws exact samples of shape sample_shape

        Args:
            * sample_shape (tuple): Shape of the draw, e.g. (n,)
            * generator (torch.Generator or None): RNG (default is None)

        Returns:
            * x (torch.Tensor of shape (*sample_shape, dim)): Exact samples
        """
        n = int(torch.tensor(sample_shape).prod())
        k = torch.multinomial(self.weights.cpu(), n, replacement=True, generator=generator)
        noise = torch.randn(n, self.dim, generator=generator, dtype=self.dtype)
        x = self.means.cpu()[k] + torch.sqrt(self.variances.cpu()[k]) * noise
        return x.reshape(*sample_shape, self.dim).to(self.device)


class GMM40(MOG):
    """The 40-component isotropic Gaussian mixture of Midgley et al. 2023 and Blessing et al. 2024

    log_gamma(x) = logsumexp_k [ log(1/40) + sum_j log N(x_j; mu_kj, 1) ]

    The components are ~1 std wide but spread over +-40, so this is a mode-finding test rather
    than a density-estimation test.

    WARNING: the mean matrix -- not the seed that drew it -- is the specification. JAX's
    threefry output has changed across major versions, so the matrix lives as a .npy asset and
    its sha256 is asserted in the test suite.
    """

    def __init__(self, dim=2, scale=1.0, means=None, prior_scale=40.0, dtype=torch.float64):
        """Builds GMM-40

        Args:
            * dim (int): Dimension; 2, 50 and 200 ship a pinned mean matrix (default is 2)
            * scale (float): Component scale (default is 1.0)
            * means (torch.Tensor of shape (40, dim) or None): Component means; None loads the
                pinned asset for this dim (default is None)
            * prior_scale (float): Reference-distribution scale (default is 40.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        if means is None:
            means = torch.as_tensor(load_asset('gmm40_means_dim' + str(dim) + '.npy'))
        weights = torch.ones(len(means), dtype=dtype)
        variances = torch.full_like(means.to(dtype), scale ** 2)
        super().__init__(weights, means, variances, name='gmm40',
                         prior_scale=prior_scale, dtype=dtype)


class MoS(Mixture):
    """The Student-t location mixture of Blessing et al. 2024, 10 components with df = 2

    WARNING: df = 2 means the variance is infinite. W2 and RBF-MMD are heavy-tailed here and
    need many samples; that is the point of the target.
    """

    train_proposal = 'student_t2'

    def __init__(self, dim=2, df=2.0, scale=1.0, means=None, prior_scale=10.0,
                 dtype=torch.float64):
        """Builds MoS

        Args:
            * dim (int): Dimension; 2, 50 and 200 ship a pinned mean matrix (default is 2)
            * df (float): Degrees of freedom (default is 2.0)
            * scale (float): Component scale (default is 1.0)
            * means (torch.Tensor of shape (10, dim) or None): Component means; None loads the
                pinned asset for this dim (default is None)
            * prior_scale (float): Reference-distribution scale (default is 10.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        if means is None:
            means = torch.as_tensor(load_asset('mos_means_dim' + str(dim) + '.npy'))
        weights = torch.ones(len(means), dtype=dtype)
        super().__init__(means.shape[-1], 'mos', weights, means, dtype=dtype)
        self.df = df
        self.scale = scale
        self.min_scale = scale
        self.prior_scale = prior_scale

    def spec(self):
        """Adds the degrees of freedom and the component scale"""
        spec = super().spec()
        spec.update({'df': self.df, 'scale': self.scale})
        return spec

    def component_log_prob(self, x):
        """Evaluates the per-component independent-Student-t log-densities"""
        delta = (x[:, None, :] - self.means) / self.scale
        constant = (
            math.lgamma(0.5 * (self.df + 1))
            - math.lgamma(0.5 * self.df)
            - 0.5 * math.log(self.df * math.pi)
            - math.log(self.scale)
        )
        return (constant - 0.5 * (self.df + 1) * torch.log1p(delta ** 2 / self.df)).sum(-1)

    def component_score(self, x):
        """Evaluates the per-component scores, -(df + 1) (x - mu_k) / (df scale^2 + (x - mu_k)^2)"""
        delta = x[:, None, :] - self.means
        return -(self.df + 1) * delta / (self.df * self.scale ** 2 + delta ** 2)

    def sample(self, sample_shape, generator=None):
        """Draws exact samples of shape sample_shape

        NOTE: exact sampling is implemented for df = 2, the corpus's only value.

        Args:
            * sample_shape (tuple): Shape of the draw, e.g. (n,)
            * generator (torch.Generator or None): RNG (default is None)

        Returns:
            * x (torch.Tensor of shape (*sample_shape, dim)): Exact samples
        """
        assert self.df == 2.0
        n = int(torch.tensor(sample_shape).prod())
        k = torch.multinomial(self.weights.cpu(), n, replacement=True, generator=generator)
        noise = student_t2((n, self.dim), generator=generator, dtype=self.dtype)
        x = self.means.cpu()[k] + self.scale * noise
        return x.reshape(*sample_shape, self.dim).to(self.device)


class TwoModes(MOG):
    """Two unequally weighted Gaussians, p = (2/3) N(-a 1_d, C) + (1/3) N(+a 1_d, C)

    The smallest possible test of whether a sampler gets mode PROPORTIONS right rather than
    just finding both modes, which is why its metric panel adds the weight of the heavy mode in
    percent and that weight's absolute error.
    """

    def __init__(self, dim=2, a=1.0, dtype=torch.float64):
        """Builds the two-mode mixture

        Args:
            * dim (int): Dimension of the mixture (default is 2)
            * a (float): Half distance between the modes (default is 1.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # 1. Mixture weights : 2/3, 1/3
        weights = torch.tensor([2.0, 1.0], dtype=dtype)
        # 2. Means : -a, a
        means = torch.stack([-a * torch.ones(dim, dtype=dtype), a * torch.ones(dim, dtype=dtype)])
        # 3. Variances, isotropic: BracketTwoModes below is where the conditioning lives
        variances = 0.05 * torch.ones_like(means)
        # Call the parent constructor
        super().__init__(weights, means, variances.contiguous(), name='twomodes', dtype=dtype)

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False):
        """Adds the heavy mode's weight in percent, and its absolute error

        Redundant with tv_weights for two modes -- it is that TV times fifty -- but it is the
        number this target is quoted with, so it is reported under its own name. Arguments and
        returns are Target.compute_metrics'.
        """
        ret, hist = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                            compute_standard_metrics=compute_standard_metrics,
                                            skip_costly_metrics=skip_costly_metrics,
                                            generator=generator, return_hist=True)
        ret['mode_weight'] = 100.0 * float(hist[0, 0])
        ret['mode_weight_abs_err'] = abs(ret['mode_weight'] - 100.0 * float(self.weights[0]))
        if return_hist:
            return ret, hist
        return ret


class BracketTwoModes(TwoModes):
    """Two modes whose diagonal covariances are each other's reverse

    Neither mode is a rescaling of the other -- the variance ramps from var_min to var_max along
    the coordinates in one and back down in the other -- so a sampler cannot fit both with one
    global step size. Same metric panel as TwoModes.
    """

    def __init__(self, dim=2, a=0.75, equilibrated=False, weight_max=None, var_min=0.01,
                 var_max=0.2, dtype=torch.float64):
        """Builds the bracketed two-mode mixture

        Args:
            * dim (int): Dimension of the mixture (default is 2)
            * a (float): Half distance between the modes (default is 0.75)
            * equilibrated (bool): Whether both modes carry equal weight (default is False)
            * weight_max (float or None): Weight of the first mode, overriding equilibrated
                (default is None)
            * var_min (float): Smallest per-coordinate variance (default is 0.01)
            * var_max (float): Largest per-coordinate variance (default is 0.2)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # 1. Means : -a, a
        means = torch.stack([-a * torch.ones(dim, dtype=dtype), a * torch.ones(dim, dtype=dtype)])
        # 2. Variances : a ramp, and the same ramp reversed
        variance_diag = torch.linspace(var_min, var_max, dim, dtype=dtype)
        variances = torch.stack([variance_diag, torch.flip(variance_diag, dims=(0,))])
        # 3. Weights
        if weight_max is not None:
            weights = torch.tensor([weight_max, 1.0 - weight_max], dtype=dtype)
        elif equilibrated:
            weights = torch.ones(2, dtype=dtype)
        else:
            weights = torch.tensor([2.0, 1.0], dtype=dtype)
        # Call the grand-parent constructor, TwoModes' own signature being a different one
        MOG.__init__(self, weights, means, variances, name='brackettwomodes', dtype=dtype)


class ManyModes(MOG):
    """n_modes Gaussians at uniformly drawn locations, with geometrically increasing weights

    The weight ratio between the lightest and heaviest mode is mixture_weight_factor, so mode
    coverage here is a question about weights and not only about locations.
    """

    def __init__(self, n_modes=3, dim=2, equilibrated=False, seed_loc=42,
                 mixture_weight_factor=3.0, var=0.1, dtype=torch.float64):
        """Builds the many-mode mixture

        Args:
            * n_modes (int): Number of modes (default is 3)
            * dim (int): Dimension of the mixture (default is 2)
            * equilibrated (bool): Whether every mode carries the same weight (default is False)
            * seed_loc (int): Seed of the mean draw, part of the specification (default is 42)
            * mixture_weight_factor (float): Ratio between the largest and smallest weight
                (default is 3.0)
            * var (float): Common per-coordinate variance (default is 0.1)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        gen = generator(seed_loc)
        # 1. Weights : geometrically increasing, or uniform
        base_log = 1.0 if equilibrated else mixture_weight_factor
        weights = torch.logspace(0.0, 1.0, n_modes, base=base_log, dtype=dtype)
        # 2. Means : uniformly chosen in [-n_modes, n_modes]^dim
        means = 2 * n_modes * torch.rand((n_modes, dim), generator=gen, dtype=dtype) - n_modes
        # 3. Variances
        variances = var * torch.ones_like(means)
        # Call the parent constructor
        super().__init__(weights, means, variances,
                         name='manymodes-k' + str(n_modes),
                         dtype=dtype)


class EightGaussians(MOG):
    """Eight equally weighted Gaussians on a circle

    WARNING: no two papers use the same parameters for this toy, so the ones here are a choice
    rather than a citation: radius 2, scale 0.1, equal weights. Both are constructor arguments,
    and the name carries neither -- a comparison across papers has to state them explicitly.
    """

    def __init__(self, radius=2.0, scale=0.1, dtype=torch.float64):
        """Builds the eight-Gaussian ring

        Args:
            * radius (float): Radius of the circle the means sit on (default is 2.0)
            * scale (float): Standard deviation of every component (default is 0.1)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        angle = 2 * math.pi * torch.arange(8, dtype=dtype) / 8
        means = radius * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        weights = torch.ones(8, dtype=dtype)
        variances = torch.full_like(means, scale ** 2)
        super().__init__(weights, means, variances, name='eightgaussians',
                         prior_scale=2.0 * radius, dtype=dtype)
