# The 2D toys: concentric rings and a checkerboard

# Libraries
import math
import torch
from .kernels import random_walk_mh
from .base import ExactTarget, generator
from .mixtures import log_prob_gaussian
from .modes import ModeCoverage


class Rings(ModeCoverage, ExactTarget):
    """Concentric rings: a mixture of radii, uniform in angle

    In polar coordinates the radius is a 1D mixture of Gaussians at num_rad evenly spaced radii and
    the angle is uniform, so

        log gamma(x) = log sum_k w_k N(|x|; r_k, scale^2) - log(2 pi) - log |x|

    the last term being the polar Jacobian. The modes are RINGS rather than points, which is what
    makes this different from every other mixture in the catalog: a sampler has to spread mass
    around each ring as well as find it, and the mode descriptor is the nearest radius.

    NOTE: the weights are proportional to the radius by default (h2o64/sde_sampler_lrds' Rings with
    equilibrated=False), so the outer rings carry more mass -- which is what a uniform ring
    density in the plane would give, since the circumference grows with r.

    WARNING: the density diverges at the origin, where the Jacobian term -log|x| blows up. That is
    the specification, not a bug; it is integrable, and no sampler visits the origin.
    """

    truth_tier = 0

    def __init__(self, lower_radius=1.0, upper_radius=5.0, n_rings=3, scale=0.1,
                 equilibrated=False, prior_scale=5.0, dtype=torch.float64):
        """Builds the rings

        Args:
            * lower_radius (float): Radius of the innermost ring (default is 1.0)
            * upper_radius (float): Radius of the outermost ring (default is 5.0)
            * n_rings (int): Number of rings (default is 3)
            * scale (float): Width of every ring (default is 0.1)
            * equilibrated (bool): Whether every ring carries the same weight; False makes the
                weight proportional to the radius (default is False)
            * prior_scale (float): Scale a Gaussian reference should use (default is 5.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # Call the parent constructor
        super().__init__(2, 'rings' + ('-equal' if equilibrated else ''), log_Z=0.0, dtype=dtype)

        # 1. The radii and their weights, in float64 for spec() whatever the working dtype
        radii = torch.linspace(lower_radius, upper_radius, n_rings, dtype=torch.float64)
        weights = torch.ones(n_rings, dtype=torch.float64) if equilibrated else radii.clone()
        self.spec_constants = {'radii': radii, 'mode_weights': weights / weights.sum()}
        self.n_modes = n_rings
        self.scale = scale
        self.prior_scale = prior_scale

        # 2. Register everything to the buffer, in the working dtype
        self.register_buffer('radii', radii.to(dtype))
        self.register_buffer('variances', torch.full_like(radii, scale ** 2).to(dtype))
        self.register_buffer('mode_weights', self.spec_constants['mode_weights'].to(dtype))

    def spec(self):
        """Adds the ring radii, their width and their weights

        The radii and weights are hashed in float64 whatever the working dtype, so a float32 build
        has the spec_id of the default one.
        """
        spec = super().spec()
        spec.update({'scale': self.scale, **self.spec_constants})
        return spec

    def radius_log_prob(self, radius):
        """Evaluates the log-density of the radius mixture at |x|"""
        component = log_prob_gaussian(radius[:, None, None], self.radii[:, None],
                                      self.variances[:, None])
        return torch.logsumexp(component + torch.log(self.mode_weights), dim=-1)

    def radius_score(self, radius):
        """Evaluates the derivative of the radius mixture's log-density at |x|"""
        component = log_prob_gaussian(radius[:, None, None], self.radii[:, None],
                                      self.variances[:, None])
        probs = torch.softmax(component + torch.log(self.mode_weights), dim=-1)
        return -torch.sum(probs * (radius[:, None] - self.radii) / self.variances, dim=-1)

    def log_prob(self, x):
        """Evaluates log gamma(x), the radius mixture times a uniform angle"""
        radius = torch.linalg.norm(x, dim=-1)
        return self.radius_log_prob(radius) - math.log(2 * math.pi) - torch.log(radius)

    def score(self, x):
        """Evaluates the score analytically: the radial derivative, projected onto x / |x|"""
        radius = torch.linalg.norm(x, dim=-1, keepdim=True)
        radial = self.radius_score(radius.squeeze(-1))[:, None]
        return x * (radial / radius - 1.0 / radius ** 2)

    def sample(self, sample_shape, generator=None):
        """Draws exactly: a radius from the mixture, an angle uniformly, then the polar map"""
        n = int(torch.tensor(sample_shape).prod())
        component = torch.multinomial(self.mode_weights.cpu(), n, replacement=True,
                                      generator=generator)
        radius = self.radii.cpu()[component] + self.scale * torch.randn(
            n, generator=generator, dtype=self.dtype)
        angle = 2 * math.pi * torch.rand(n, generator=generator, dtype=self.dtype)
        x = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
        return x.to(self.device).reshape(*sample_shape, 2)

    def mode_labels(self, x):
        """Returns the index of the nearest ring, the mode descriptor of this target"""
        radius = torch.linalg.norm(x, dim=-1)
        return torch.argmin((radius[:, None] - self.radii).abs(), dim=-1)

    def generate_train_samples(self, n):
        """Runs the one-chain-per-ring recipe

        Chains are spread evenly over the rings, each starting at a uniformly drawn point of its
        own ring, and take 200 Metropolis steps of size scale -- enough to travel around a ring
        but far too small to cross the gap between two. Every ring is therefore covered, with the
        right width, and the ring weights are wrong in the same way the mixtures' are.

        Args:
            * n (int): Number of samples, one per chain

        Returns:
            * x (torch.Tensor of shape (n, 2)): The biased samples
        """
        gen = generator(0)
        radius = self.radii.cpu()[torch.arange(n) % self.n_modes]
        angle = 2 * math.pi * torch.rand(n, generator=gen, dtype=self.dtype)
        x0 = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
        return random_walk_mh(self.log_prob, x0, n_steps=200, step_size=self.scale, generator=gen)


class Checkerboard(ModeCoverage, ExactTarget):
    """A checkerboard of uniform squares, 2 x 2 each

    log gamma(x) = log sum_k w_k U(x; square_k), i.e. piecewise CONSTANT on the squares and -inf
    off them. With unequilibrated=True every second square carries three times the weight of its
    neighbours, so the target tests mode proportions as well as coverage.

    WARNING: the score is IDENTICALLY ZERO wherever the density is finite, so a score-based sampler
    gets no signal from this target at all. That is the density, not an implementation shortcut: a
    uniform mixture has no gradient. It is here as a coverage test for samplers that use log_prob
    -- Metropolis, SMC, importance sampling -- and log_prob is -inf off the support, which is why
    every evaluation point must come from the support or from a proposal that can be rejected.
    """

    truth_tier = 0

    def __init__(self, width=4, unequilibrated=True, prior_scale=4.0, dtype=torch.float64):
        """Builds the checkerboard

        Args:
            * width (int): Number of squares along the x axis (default is 4)
            * unequilibrated (bool): Whether every second square is three times heavier
                (default is True)
            * prior_scale (float): Scale a Gaussian reference should use (default is 4.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # Call the parent constructor
        super().__init__(2, 'checkerboard' + ('' if unequilibrated else '-equal'), log_Z=0.0,
                         dtype=dtype)

        # 1. The lower-left corner of every square, in h2o64/sde_sampler_lrds' layout
        corners = []
        for top in [4, 0]:
            corners += [(x, top - 2) for x in range(-2, -4 + 2 * width, 4)]
            corners += [(x, top - 4) for x in range(-4, -4 + 2 * width, 4)]
        low = torch.tensor(corners, dtype=dtype)
        self.width = width
        self.prior_scale = prior_scale
        self.n_modes = len(low)
        self.register_buffer('low', low)
        self.register_buffer('high', low + 2.0)

        # 2. Weights: every second square three times heavier, which is upstream's default
        weights = torch.ones(self.n_modes, dtype=dtype)
        if unequilibrated:
            weights[torch.arange(self.n_modes) % 2 == 0] *= 3.0
        self.register_buffer('mode_weights', weights / weights.sum())

    def spec(self):
        """Adds the squares themselves and their weights, which the 2D dim does not cover"""
        spec = super().spec()
        spec.update({'low': self.low, 'mode_weights': self.mode_weights})
        return spec

    def inside(self, x):
        """Returns whether each point lies in each square, as a boolean (n, n_modes) tensor"""
        return ((x[:, None, :] >= self.low) & (x[:, None, :] < self.high)).all(dim=-1)

    def log_prob(self, x):
        """Evaluates log gamma(x), piecewise constant on the squares and -inf off them

        WARNING: -inf off the support is deliberate, so a Metropolis proposal that leaves the
        board is rejected rather than accepted at a made-up density.
        """
        density = (self.inside(x).to(self.dtype) * self.mode_weights / 4.0).sum(dim=-1)
        return torch.log(density)

    def score(self, x):
        """Evaluates the score, which is exactly zero on a piecewise-constant density"""
        return torch.zeros_like(x)

    def sample(self, sample_shape, generator=None):
        """Draws exactly: pick a square by weight, then a uniform point inside it"""
        n = int(torch.tensor(sample_shape).prod())
        component = torch.multinomial(self.mode_weights.cpu(), n, replacement=True,
                                      generator=generator)
        unit = torch.rand(n, 2, generator=generator, dtype=self.dtype)
        x = self.low.cpu()[component] + 2.0 * unit
        return x.to(self.device).reshape(*sample_shape, 2)

    def mode_labels(self, x):
        """Returns the index of the square each point falls in, and 0 for points off the board"""
        return self.inside(x).to(self.dtype).argmax(dim=-1)

    def generate_train_samples(self, n):
        """Runs the one-chain-per-square recipe

        Chains are spread evenly over the squares, each starting at its own square's centre, and
        take 200 Metropolis steps of size 0.5. A step that leaves the board is rejected.

        NOTE: the squares meet at their CORNERS, and a 0.5 step crosses a corner diagonally often
        enough that the weights partly equilibrate rather than staying at the uniform allocation:
        measured heavy-to-light ratio 1.84 against the true 3.0 at n = 2048, i.e. tv_weights 0.21
        where an exact draw of the same size reads 0.06. So this split is a partly-relaxed chain,
        not a frozen one -- which is exactly the failure a real sampler with too short a run has.

        Args:
            * n (int): Number of samples, one per chain

        Returns:
            * x (torch.Tensor of shape (n, 2)): The biased samples
        """
        gen = generator(0)
        centre = self.low.cpu() + 1.0
        x0 = centre[torch.arange(n) % self.n_modes]
        return random_walk_mh(self.log_prob, x0, n_steps=200, step_size=0.5, generator=gen)
