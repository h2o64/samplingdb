# Mixture targets, the corpus's mode-coverage benchmarks

# Libraries
import math
import torch
from ..metrics.weights import normalize_weights
from .kernels import student_t2
from .base import ExactTarget, generator
from .modes import ModeCoverage


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


def uniform_means(n_modes, dim, low, high, seed, min_distance=0.0):
    """Draws mixture locations uniformly in [low, high]^dim from a seeded torch generator

    The draw is made on the CPU and in float32 whatever the working dtype, so a float64 build holds
    the same locations as the default one and keeps its spec_id. The matrix it returns, not the
    seed, is the specification: tests/test_target_specs.py pins its sha256, so a change in torch's
    generator would fail there rather than move the density.

    NOTE: the candidates are kept in their order of draw, skipping any closer than min_distance to
    one already kept, so wherever the first n_modes candidates are far enough apart the locations
    are those of a plain draw.

    Args:
        * n_modes (int): Number of components
        * dim (int): Dimension
        * low (float): Lower end of every coordinate
        * high (float): Upper end of every coordinate
        * seed (int): Seed of the draw
        * min_distance (float): Smallest distance between two locations (default is 0.0)

    Returns:
        * means (torch.Tensor of shape (n_modes, dim)): Locations, in float32
    """
    u = torch.rand((100 * n_modes, dim), generator=generator(seed), dtype=torch.float32)
    candidates = low + (high - low) * u

    # Keep the candidates in order, skipping those too close to one already kept
    means = candidates[:1]
    for candidate in candidates[1:]:
        if len(means) == n_modes:
            break
        if float(torch.linalg.norm(means - candidate, dim=-1).min()) >= min_distance:
            means = torch.cat([means, candidate[None]])
    assert len(means) == n_modes
    return means


# Every class here is a Tier 0 exact-draw toy and works in float64, which the exactness checks that
# pin their specifications need. The periodic pair mixtures, which have a sampler, live in
# physics.py and are float32
class Mixture(ModeCoverage, ExactTarget):
    """Base class for the isotropic-component location mixtures

    A mixture's log_prob is a logsumexp over components, and its score is the responsibility-
    weighted average of the per-component scores -- both written out analytically here, since
    the responsibilities are a softmax of quantities log_prob already computes.

    Subclasses provide component_log_prob and component_score. The coverage panel comes from
    ModeCoverage, on the mean responsibility of every component (see compute_mode_weights).
    """

    truth_tier = 0

    def __init__(self, dim, name, weights, means, dtype=torch.float64):
        """Builds the mixture

        Args:
            * dim (int): Dimension
            * name (str): Version tag
            * weights (torch.Tensor of shape (n_modes,)): Mixture weights, normalized here in
                float64
            * means (torch.Tensor of shape (n_modes, dim)): Component locations
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # Call the parent constructor
        super().__init__(dim, name, log_Z=0.0, dtype=dtype)

        # 1. The constants that define the density, kept in float64 for spec() whatever the dtype
        self.n_modes = weights.shape[0]
        weights = weights.to(torch.float64)
        self.spec_constants = {'weights': weights / weights.sum(), 'means': means.to(torch.float64)}

        # 2. Register everything to the buffer, in the working dtype
        self.register_buffer('weights', self.spec_constants['weights'].to(dtype))
        self.register_buffer('means', means.to(dtype))

    def spec(self):
        """Adds the constants that ARE the specification, the weights and locations among them

        They are hashed in float64 whatever the working dtype, so a float32 build has the spec_id
        of the default one. Every subclass therefore computes them in float64 and lets the buffers
        take the working dtype.
        """
        spec = super().spec()
        spec.update(self.spec_constants)
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
        return self.chunked(self.log_prob_chunk, x, cost=self.n_modes * self.dim)

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
        """Returns argmax_k pi_k(x), the component each sample sits closest to"""
        return self.chunked(self.mode_labels_chunk, x, cost=self.n_modes * self.dim)

    def mode_labels_chunk(self, x):
        """Returns the mode labels of a batch small enough to hold its per-component densities"""
        return self.component_log_prob(x).argmax(dim=-1)

    def compute_mode_weights(self, samples, weights=None):
        """Returns the (weighted) mean of the component responsibilities, of shape (1, n_modes)

        Under exact draws the expected responsibility of component k is exactly its weight w_k,
        whatever the overlap between components, so an exact draw scores no floor on
        tv_weights beyond its sampling noise.

        NOTE: counts of mode_labels measure the mass of the region where each component dominates,
        which is not w_k where components overlap, so they differ from these weights.

        Args:
            * samples (torch.Tensor of shape (n, dim)): Samples
            * weights (torch.Tensor of shape (n,)): Weights of the samples (default is None)

        Returns:
            * hist (torch.Tensor of shape (1, n_modes)): Empirical mode weights
        """
        x = samples.to(self.weights)
        mass = normalize_weights(None if weights is None else weights.to(self.weights),
                                 n=len(x), dtype=self.dtype, device=x.device)
        probs = self.chunked(self.responsibilities, x, cost=self.n_modes * self.dim)
        return (mass @ probs)[None]

    @property
    def train_step_size(self):
        """Returns 2.4 min_scale / sqrt(dim), the optimal random-walk scale of the narrowest mode"""
        return 2.4 * self.min_scale / math.sqrt(self.dim)


class MOG(Mixture):
    """A mixture of Gaussians with diagonal covariances"""

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

        # Register the covariances, which carry the scale and the conditioning and are part of
        # the specification, and the derived scales
        self.spec_constants['variances'] = variances.to(torch.float64)
        self.register_buffer('variances', variances.to(dtype))
        self.min_scale = float(variances.min().sqrt())
        if prior_scale is None:
            prior_scale = float(means.abs().max() + variances.max().sqrt())
        self.prior_scale = prior_scale

    def component_log_prob(self, x):
        """Evaluates the per-component diagonal-Gaussian log-densities"""
        return log_prob_gaussian(x[:, None, :], self.means, self.variances)

    def component_score(self, x):
        """Evaluates the per-component scores, -(x - mu_k) / var_k"""
        return -(x[:, None, :] - self.means) / self.variances

    def sample_modes(self, modes, generator=None):
        """Draws one exact sample from each given component, on the device of modes"""
        noise = torch.randn(len(modes), self.dim, generator=generator, dtype=self.dtype,
                            device=modes.device)
        return self.means[modes] + torch.sqrt(self.variances[modes]) * noise

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

    WARNING: the means are drawn uniformly in [-40, 40]^dim by uniform_means with seed 0, in
    torch, so they are not the locations of the two papers, which were drawn with JAX. Numbers
    transfer through the construction, not through the exact locations.
    """

    def __init__(self, dim=2, scale=1.0, means=None, prior_scale=40.0, dtype=torch.float64):
        """Builds GMM-40

        Args:
            * dim (int): Dimension (default is 2)
            * scale (float): Component scale (default is 1.0)
            * means (torch.Tensor of shape (40, dim) or None): Component means; None draws them
                with uniform_means (default is None)
            * prior_scale (float): Reference-distribution scale (default is 40.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        if means is None:
            means = uniform_means(40, dim, -40.0, 40.0, seed=0)
        means = means.to(torch.float64)
        weights = torch.ones(len(means), dtype=torch.float64)
        variances = torch.full_like(means, scale ** 2)
        super().__init__(weights, means, variances, name='gmm40',
                         prior_scale=prior_scale, dtype=dtype)


class MoS(Mixture):
    """The Student-t location mixture of Blessing et al. 2024, 10 components with df = 2

    WARNING: df = 2 means the variance is infinite. W2 and RBF-MMD are heavy-tailed here and
    need many samples; that is the point of the target.

    NOTE: the means are drawn uniformly in [-10, 10]^dim by uniform_means with seed 0, in torch,
    no two closer than MIN_DISTANCE scales, so that the heavy tails do not merge two modes into
    one; see GMM40 on why they are not the paper's.
    """

    # Smallest distance between two means, in units of the component scale
    MIN_DISTANCE = 5.0

    def __init__(self, dim=2, df=2.0, scale=1.0, means=None, prior_scale=10.0,
                 dtype=torch.float64):
        """Builds MoS

        Args:
            * dim (int): Dimension (default is 2)
            * df (float): Degrees of freedom (default is 2.0)
            * scale (float): Component scale (default is 1.0)
            * means (torch.Tensor of shape (10, dim) or None): Component means; None draws them
                with uniform_means (default is None)
            * prior_scale (float): Reference-distribution scale (default is 10.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        if means is None:
            means = uniform_means(10, dim, -10.0, 10.0, seed=0,
                                  min_distance=self.MIN_DISTANCE * scale)
        means = means.to(torch.float64)
        weights = torch.ones(len(means), dtype=torch.float64)
        super().__init__(means.shape[-1], 'mos', weights, means, dtype=dtype)
        self.df = df
        self.scale = scale
        self.min_scale = scale
        self.prior_scale = prior_scale

    @property
    def train_step_size(self):
        """Returns half the component scale

        The df = 2 tails reach from one component to the next, so a random walk at the optimal
        scale would cross between them within the recipe's steps and undo its allocation.
        """
        return 0.5 * self.scale

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

    def sample_modes(self, modes, generator=None):
        """Draws one exact sample from each given component, on the device of modes

        A Student-t with df degrees of freedom is a standard normal over the root of an
        independent chi-square divided by df, and for an integer df that chi-square is a sum of df
        squared normals, so the draw respects generator at any integer df.
        """
        assert float(self.df).is_integer()
        shape = (len(modes), self.dim)
        normal = torch.randn(shape, generator=generator, dtype=self.dtype, device=modes.device)
        squares = torch.randn(shape + (int(self.df),), generator=generator, dtype=self.dtype,
                              device=modes.device) ** 2
        return self.means[modes] + self.scale * normal / torch.sqrt(squares.sum(-1) / self.df)

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
    just finding both modes: return_hist=True gives the empirical weight of each mode, and
    tv_weights is twice the error on the heavy one.
    """

    def __init__(self, dim=2, a=1.0, dtype=torch.float64):
        """Builds the two-mode mixture

        Args:
            * dim (int): Dimension of the mixture (default is 2)
            * a (float): Half distance between the modes (default is 1.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # 1. Mixture weights : 2/3, 1/3
        weights = torch.tensor([2.0, 1.0], dtype=torch.float64)
        # 2. Means : -a, a
        means = torch.stack([-a * torch.ones(dim, dtype=torch.float64),
                             a * torch.ones(dim, dtype=torch.float64)])
        # 3. Variances, isotropic: BracketTwoModes below is where the conditioning lives
        variances = 0.05 * torch.ones_like(means)
        # Call the parent constructor
        super().__init__(weights, means, variances, name='twomodes', dtype=dtype)


class BracketTwoModes(TwoModes):
    """Two modes whose diagonal covariances are each other's reverse

    Neither mode is a rescaling of the other -- the variance ramps from var_min to var_max along
    the coordinates in one and back down in the other -- so a sampler cannot fit both with one
    global step size. Same metric panel as TwoModes.
    """

    # Weights of the two modes, 2/3 : 1/3 as in TwoModes unless equilibrated
    unequal_weights = (2.0, 1.0)

    def __init__(self, dim=2, a=0.75, equilibrated=False, var_min=0.01, var_max=0.2,
                 dtype=torch.float64):
        """Builds the bracketed two-mode mixture

        Args:
            * dim (int): Dimension of the mixture (default is 2)
            * a (float): Half distance between the modes (default is 0.75)
            * equilibrated (bool): Whether both modes carry equal weight (default is False)
            * var_min (float): Smallest per-coordinate variance (default is 0.01)
            * var_max (float): Largest per-coordinate variance (default is 0.2)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # 1. Means : -a, a
        means = torch.stack([-a * torch.ones(dim, dtype=torch.float64),
                             a * torch.ones(dim, dtype=torch.float64)])
        # 2. Variances : a ramp, and the same ramp reversed
        variance_diag = torch.linspace(var_min, var_max, dim, dtype=torch.float64)
        variances = torch.stack([variance_diag, torch.flip(variance_diag, dims=(0,))])
        # 3. Weights
        weights = torch.tensor((1.0, 1.0) if equilibrated else self.unequal_weights,
                               dtype=torch.float64)
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
        # 1. Weights : geometrically increasing, or uniform
        base_log = 1.0 if equilibrated else mixture_weight_factor
        weights = torch.logspace(0.0, 1.0, n_modes, base=base_log, dtype=torch.float64)
        # 2. Means : uniformly chosen in [-n_modes, n_modes]^dim
        means = uniform_means(n_modes, dim, -n_modes, n_modes, seed_loc).to(torch.float64)
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
        angle = 2 * math.pi * torch.arange(8, dtype=torch.float64) / 8
        means = radius * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        weights = torch.ones(8, dtype=torch.float64)
        variances = torch.full_like(means, scale ** 2)
        super().__init__(weights, means, variances, name='eightgaussians',
                         prior_scale=2.0 * radius, dtype=dtype)
