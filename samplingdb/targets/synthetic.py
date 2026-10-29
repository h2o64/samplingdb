# Tier 0 synthetic targets with closed-form densities, analytic scores and exact ground truth

# Libraries
import math
import numpy as np
import torch
from scipy.integrate import quad
from .kernels import mala, random_walk_mh
from ..metrics.weights import normalize_weights
from ..metrics.histogram import histogram_tvd, weighted_histogram
from .base import ExactTarget, cycle, generator
from .modes import ModeCoverage

# Half of log(2 pi), the constant every Gaussian log-density below carries
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


class Funnel(ExactTarget):
    """Neal's funnel

    log_gamma(x) = log N(x_1; 0, scale^2) + sum_{i>=2} log N(x_i; 0, exp(x_1))

    WARNING: the reference implementation uses scale = 3, i.e. variance 9. Passing 9 where a
    scale is expected shifts log_gamma by log(3) = 1.0986 nats at the mode.

    WARNING: exp(x_1) overflows for x_1 > ~88 in float32, so the default dtype is float64.
    """

    truth_tier = 0
    train_recipe = 'uniform-neck'

    def __init__(self, dim=10, scale=3.0, prior_scale=3.0, dtype=torch.float64):
        """Builds the funnel

        Args:
            * dim (int): Dimension (default is 10, the corpus-wide choice)
            * scale (float): Scale -- not variance -- of the dominant coordinate (default is 3.0)
            * prior_scale (float): Scale a Gaussian reference should use (default is 3.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        # Call the parent constructor
        super().__init__(dim, 'funnel', log_Z=0.0, dtype=dtype)

        # Register the two pinned constants
        self.scale = scale
        self.prior_scale = prior_scale

    def spec(self):
        """Adds the scale of the dominant coordinate"""
        spec = super().spec()
        spec['scale'] = self.scale
        return spec

    def log_prob(self, x):
        """Evaluates log gamma(x), which is normalised here so log Z = 0"""
        x1, rest = x[:, 0], x[:, 1:]
        log_prob = -HALF_LOG_2PI - math.log(self.scale) - 0.5 * (x1 / self.scale) ** 2
        conditional = -HALF_LOG_2PI - 0.5 * x1[:, None] - 0.5 * rest ** 2 * torch.exp(-x1)[:, None]
        return log_prob + conditional.sum(-1)

    def score(self, x):
        """Evaluates the score of the funnel at x, analytically

        NOTE: the x_1 derivative has three terms: the prior, the log_sigma Jacobian of the d - 1
        conditionals, and their quadratic.
        """
        x1, rest = x[:, 0], x[:, 1:]
        precision = torch.exp(-x1)
        grad = torch.empty_like(x)
        grad[:, 0] = (-x1 / self.scale ** 2 - 0.5 * (self.dim - 1)
                      + 0.5 * precision * (rest ** 2).sum(-1))
        grad[:, 1:] = -rest * precision[:, None]
        return grad

    def sample(self, sample_shape, generator=None):
        """Draws exact samples of shape sample_shape

        Args:
            * sample_shape (tuple): Shape of the draw, e.g. (n,)
            * generator (torch.Generator or None): RNG (default is None)

        Returns:
            * x (torch.Tensor of shape (*sample_shape, dim)): Exact samples
        """
        n = int(torch.tensor(sample_shape).prod())
        x1 = self.scale * torch.randn(n, 1, generator=generator, dtype=self.dtype)
        rest = torch.exp(0.5 * x1) * torch.randn(n, self.dim - 1, generator=generator,
                                                 dtype=self.dtype)
        return torch.cat([x1, rest], dim=-1).reshape(*sample_shape, self.dim).to(self.device)

    def generate_train_samples(self, n, seed=0, x0=None, n_steps=200, step_size=0.5):
        """Runs the uniform-neck recipe: the whole funnel, with x_1 uniform instead of Gaussian

        Every chain starts with x_1 uniform on [-4 scale, 4 scale] and the other coordinates drawn
        from their exact conditional N(0, exp(x_1)), then takes n_steps random-walk Metropolis
        steps of size step_size. The steps move the chains in the wide end and hardly at all in
        the neck, so x_1 stays close to uniform: neck and wide end are both covered, and the
        marginal of x_1 is far wider than N(0, scale^2).

        Args:
            * n (int): Number of samples, one per chain
            * seed (int): Seed of the recipe (default is 0)
            * x0 (torch.Tensor of shape (m, dim) or None): Starting points, chain i starting from
                x0[i % m]; None draws them as above (default is None)
            * n_steps (int): Random-walk steps per chain (default is 200)
            * step_size (float): Proposal scale (default is 0.5)

        Returns:
            * x (torch.Tensor of shape (n, dim)): The biased samples
        """
        gen = generator(seed, device=self.device)
        if x0 is None:
            u = torch.rand(n, 1, generator=gen, dtype=self.dtype, device=self.device)
            x1 = 4 * self.scale * (2 * u - 1)
            rest = torch.exp(0.5 * x1) * torch.randn(n, self.dim - 1, generator=gen,
                                                     dtype=self.dtype, device=self.device)
            x0 = torch.cat([x1, rest], dim=-1)
        return random_walk_mh(self.log_prob, cycle(self.cast(x0), n), n_steps, step_size,
                              generator=gen)

    def x1_cdf(self, x1):
        """Evaluates the exact CDF of the dominant coordinate, N(0, scale^2)"""
        return 0.5 * torch.erfc(-x1 / (self.scale * math.sqrt(2.0)))

    def x1_reference_histogram(self, edges):
        """Returns the exact x_1 histogram on the given edges, from the analytic CDF

        Args:
            * edges (torch.Tensor of shape (n_bins + 1,)): Bin edges

        Returns:
            * hist (torch.Tensor of shape (n_bins,)): Exact bin masses, renormalised to sum to one
        """
        cdf = self.x1_cdf(edges)
        hist = cdf[1:] - cdf[:-1]
        return hist / hist.sum()

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False, n_bins=200, bound=None):
        """Adds a comparison of the dominant coordinate against its exact marginal

        The funnel has no modes to count; what a sampler gets wrong here is the calibration of
        x_1, whose marginal is exactly N(0, scale^2). So the target-specific panel is a KS
        distance and a histogram TVD against that analytic law -- no reference draw, no
        Monte-Carlo noise on the reference side. The sliced KS over ALL coordinates, which is what
        says anything about the neck the other d - 1 live in, is 'ks' in the universal panel.

        WARNING: neck_var_rel_err, the relative error on E[exp(x_1)], has a large and skewed floor:
        exp(x_1) is log-normal with a heavy right tail, so a few draws at large x_1 carry the mean.
        Read it against exact draws of the same size, never against 0.

        Args are Target.compute_metrics', plus:
            * n_bins (int): Number of bins of the histogram comparison (default is 200)
            * bound (float or None): Half-width of the histogram range; None uses 4 scales
                (default is None)

        Returns:
            * metrics (dict): The universal panel plus 'ks_x1', 'tvd_x1', 'x1_out_of_range',
                'x1_mean_err', 'x1_var_err' and 'neck_var_rel_err'
            if return_hist:
                * hist (dict): 'edges', 'hist' and 'ref_hist', the raw x_1 histograms
        """
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                      compute_standard_metrics=compute_standard_metrics,
                                      skip_costly_metrics=skip_costly_metrics, generator=generator)
        samples, weights = self.cast(samples), self.cast(weights)
        weights = None if weights is None else normalize_weights(weights)
        x1 = samples[:, 0].detach()
        mass = normalize_weights(weights, n=len(x1), dtype=self.dtype, device=x1.device)

        # Weighted KS distance against the analytic CDF, which the empirical one meets on both
        # sides of every jump
        order = torch.argsort(x1)
        right = torch.cumsum(mass[order], dim=0)
        left = right - mass[order]
        exact = self.x1_cdf(x1[order])
        ks = float(torch.maximum((right - exact).abs(), (left - exact).abs()).max())

        # Histogram TVD against the exact bin masses
        if bound is None:
            bound = 4.0 * self.scale
        edges = torch.linspace(-bound, bound, n_bins + 1, dtype=self.dtype, device=x1.device)
        hist, out_of_range = weighted_histogram(x1, edges, weights=weights,
                                                return_out_of_range=True)
        ref_hist = self.x1_reference_histogram(edges)

        # Moment errors, all three references being exact
        ret.update({
            'ks_x1': ks,
            'tvd_x1': histogram_tvd(hist, ref_hist),
            'x1_out_of_range': out_of_range,
            'x1_mean_err': float((mass * x1).sum()),
            'x1_var_err': float((mass * x1 ** 2).sum()) - self.scale ** 2,
            'neck_var_rel_err': (float((mass * torch.exp(x1)).sum())
                                 / math.exp(0.5 * self.scale ** 2) - 1.0),
        })
        if return_hist:
            return ret, {'edges': edges, 'hist': hist, 'ref_hist': ref_hist}
        return ret


class ManyWell(ModeCoverage, ExactTarget):
    """The product of d/2 independent 2D double wells (the coefficients of Wu et al. 2020)

    E_2D(x1, x2) = a*x1 + b*x1^2 + c*x1^4 + 0.5*x2^2,  a = -0.5, b = -6, c = 1

    Coordinates are interleaved: the even ones carry the quartic, the odd ones the Gaussian.

    The only target in the corpus that is simultaneously high-dimensional, has 2^(d/2) modes and
    an exact log Z. The two wells are NOT symmetric: the linear term tilts the potential, giving
    15.57% / 84.43% of the mass to the left / right well.

    NOTE: its mode descriptor has one GROUP PER WELL rather than one label over 2^(d/2) joint
    modes, because no finite sample can populate 2^16 of them while the per-well marginals are
    estimable and are what the literature reports. ModeCoverage handles both shapes; see there.
    """

    truth_tier = 0
    train_recipe = 'uniform-sides'

    def __init__(self, dim=32, a=-0.5, b=-6.0, c=1.0, bound=4.0, n_grid=200001, prior_scale=2.0,
                 dtype=torch.float64):
        """Builds ManyWell

        Args:
            * dim (int): Dimension, even (default is 32)
            * a (float): Linear coefficient of the 1D quartic (default is -0.5)
            * b (float): Quadratic coefficient (default is -6.0)
            * c (float): Quartic coefficient (default is 1.0)
            * bound (float): Truncation bound of the 1D marginal, where its density is ~e^-160
                (default is 4.0)
            * n_grid (int): Grid size of the inverse-CDF sampler (default is 200001)
            * prior_scale (float): Scale a Gaussian reference should use (default is 2.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
        """
        assert dim % 2 == 0

        # 1. The 1D quadrature that fixes log Z, the well masses and the moments
        log_Z_1d, well_mass = _double_well_normaliser(a, b, c, bound)
        n_wells = dim // 2
        super().__init__(dim, 'manywell', log_Z=n_wells * (log_Z_1d + HALF_LOG_2PI),
                         dtype=dtype)

        # 2. The pinned coefficients and the derived per-well quantities
        self.a, self.b, self.c = a, b, c
        self.bound = bound
        self.n_wells = n_wells
        self.n_modes = 2 ** n_wells
        self.log_Z_2d = log_Z_1d + HALF_LOG_2PI
        self.prior_scale = prior_scale
        self.register_buffer('mode_weights', torch.as_tensor(well_mass, dtype=dtype))

        # The left and right minima of the 1D quartic, the real roots of its derivative either
        # side of the barrier, where the train chains start
        roots = np.sort(np.roots([4 * c, 0.0, 2 * b, a]).real)
        self.well_minima = (float(roots[0]), float(roots[-1]))

        # 3. The inverse-CDF grid of the 1D marginal
        grid, cdf, moments = _double_well_grid(a, b, c, bound, n_grid)
        self.register_buffer('grid', torch.as_tensor(grid, dtype=dtype))
        self.register_buffer('cdf', torch.as_tensor(cdf, dtype=dtype))

        # 4. The per-coordinate moments, the quartic ones by quadrature and the Gaussian ones exact
        mean = torch.zeros(dim, dtype=dtype)
        variance = torch.ones(dim, dtype=dtype)
        mean[0::2], variance[0::2] = moments[0], moments[1]
        self.register_buffer('mean', mean)
        self.register_buffer('variance', variance)

    def spec(self):
        """Adds the quartic's coefficients and the truncation bound of its normaliser"""
        spec = super().spec()
        spec.update({'a': self.a, 'b': self.b, 'c': self.c, 'bound': self.bound})
        return spec

    def log_prob_1d(self, x1):
        """Evaluates the unnormalised log-density of one x1 coordinate"""
        return -(self.a * x1 + self.b * x1 ** 2 + self.c * x1 ** 4)

    def log_prob(self, x):
        """Evaluates log gamma(x) = sum over wells of -E_2D"""
        x1, x2 = x[:, 0::2], x[:, 1::2]
        return (self.log_prob_1d(x1) - 0.5 * x2 ** 2).sum(-1)

    def score(self, x):
        """Evaluates the score of ManyWell at x, analytically"""
        x1, x2 = x[:, 0::2], x[:, 1::2]
        grad = torch.empty_like(x)
        grad[:, 0::2] = -(self.a + 2 * self.b * x1 + 4 * self.c * x1 ** 3)
        grad[:, 1::2] = -x2
        return grad

    def sample(self, sample_shape, generator=None):
        """Draws exact samples of shape sample_shape, well by well

        Args:
            * sample_shape (tuple): Shape of the draw, e.g. (n,)
            * generator (torch.Generator or None): RNG (default is None)

        Returns:
            * x (torch.Tensor of shape (*sample_shape, dim)): Exact samples
        """
        n = int(torch.tensor(sample_shape).prod())
        u = torch.rand(n, self.n_wells, generator=generator, dtype=self.dtype)
        x = torch.empty(n, self.dim, dtype=self.dtype)
        x[:, 0::2] = _interpolate_inverse_cdf(u, self.cdf.cpu(), self.grid.cpu())
        x[:, 1::2] = torch.randn(n, self.n_wells, generator=generator, dtype=self.dtype)
        return x.reshape(*sample_shape, self.dim).to(self.device)

    def train_modes(self, n, generator=None):
        """Draws the side of every well for every train chain, left with a per-chain probability

        Each chain draws p uniformly on [0, 1] and then puts each of its wells on the left side
        with probability p, independently. Every well is on its light side half the time, and the
        number of left wells is uniform over 0..n_wells rather than binomial, which spreads the
        energy into both tails.

        Args:
            * n (int): Number of chains
            * generator (torch.Generator or None): RNG, on the target's device (default is None)

        Returns:
            * left (torch.Tensor of shape (n, n_wells)): Whether each well starts on the left
        """
        p = torch.rand(n, 1, generator=generator, dtype=self.dtype, device=self.device)
        u = torch.rand(n, self.n_wells, generator=generator, dtype=self.dtype, device=self.device)
        return u < p

    def generate_train_samples(self, n, seed=0, x0=None, n_steps=200, step_size=0.02):
        """Runs the uniform-sides recipe: both sides of every well, in the wrong proportions

        Every chain draws its wells' sides from train_modes, starts each quartic coordinate at the
        minimum of its side (well_minima) and each Gaussian coordinate at an exact N(0, 1) draw,
        then takes n_steps MALA steps of size step_size, which relax every coordinate inside its
        well without crossing the barrier. Both sides of every well are visited, in the wrong
        proportions.

        Args:
            * n (int): Number of samples, one per chain
            * seed (int): Seed of the recipe (default is 0)
            * x0 (torch.Tensor of shape (m, dim) or None): Starting points, chain i starting from
                x0[i % m]; None draws them as above (default is None)
            * n_steps (int): MALA steps per chain (default is 200)
            * step_size (float): MALA step size (default is 0.02)

        Returns:
            * x (torch.Tensor of shape (n, dim)): The biased samples
        """
        gen = generator(seed, device=self.device)
        if x0 is None:
            left = self.train_modes(n, generator=gen)
            x0 = torch.empty(n, self.dim, dtype=self.dtype, device=self.device)
            x0[:, 0::2] = torch.where(left, self.well_minima[0], self.well_minima[1])
            x0[:, 1::2] = torch.randn(n, self.n_wells, generator=gen, dtype=self.dtype,
                                      device=self.device)
        return mala(self.log_prob, self.score, cycle(self.cast(x0), n), n_steps, step_size,
                    generator=gen)

    def mode_labels(self, x):
        """Returns which side of each well every sample sits on, as a (n, n_wells) 0/1 tensor

        Bit i is 1 when well i sits in the deep (positive) side, so a row IS the sample's joint
        mode; torch.unique(labels, dim=0) counts how many of the 2^(d/2) modes were visited.
        """
        return (x[:, 0::2] > 0).to(torch.int64)

    def x1_reference_histogram(self, edges):
        """Returns the exact pooled x1 histogram on the given edges, from the quadrature grid

        Args:
            * edges (torch.Tensor of shape (n_bins + 1,)): Bin edges

        Returns:
            * hist (torch.Tensor of shape (n_bins,)): Exact bin masses, renormalised to sum to one
        """
        cdf = _interpolate_cdf(edges.to(self.grid), self.grid, self.cdf)
        hist = (cdf[1:] - cdf[:-1]).clamp_min(0.0)
        return hist / hist.sum()

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False, n_bins=200):
        """Adds the pooled quartic marginal to the per-well coverage panel

        Every well shares the same 1D law, so pooling the d/2 quartic coordinates gives a much
        tighter estimate of it than any single coordinate would -- and the reference side is
        exact, coming from the same quadrature that fixes log Z. 'w1_x1' is the integral of
        |F_n - F| between the pooled coordinates' weighted CDF and that exact CDF, computed in
        closed form on the quadrature grid, so no Monte-Carlo reference adds its own noise. On
        exact draws of n = 1000 it reads 0.0078 +- 0.0041 over 20 seeds.

        Args are Target.compute_metrics', plus:
            * n_bins (int): Number of bins of the histogram comparison (default is 200)

        Returns:
            * metrics (dict): The universal panel and the four coverage numbers, plus 'tvd_x1',
                'x1_out_of_range', 'w1_x1' and 'x1_mean_err'
            if return_hist:
                * hist (torch.Tensor of shape (n_wells, 2)): Empirical per-well mode weights
        """
        ret, hist = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                            compute_standard_metrics=compute_standard_metrics,
                                            skip_costly_metrics=skip_costly_metrics,
                                            generator=generator, return_hist=True)
        samples, weights = self.cast(samples), self.cast(weights)
        weights = None if weights is None else normalize_weights(weights)

        x1 = samples[:, 0::2].detach()
        edges = torch.linspace(-self.bound, self.bound, n_bins + 1, dtype=self.dtype,
                               device=x1.device)
        marginal, out_of_range = weighted_histogram(x1, edges, weights=weights,
                                                    return_out_of_range=True)
        ref_hist = self.x1_reference_histogram(edges).to(x1.device)
        mass = normalize_weights(weights, n=len(x1), dtype=self.dtype, device=x1.device)

        # The pooled quartic coordinates carry their sample's weight once per well
        pair_weights = mass.repeat_interleave(self.n_wells) / self.n_wells
        w1 = _w1_to_tabulated_cdf(x1.reshape(-1), pair_weights, self.grid.to(x1.device),
                                  self.cdf.to(x1.device))
        ret.update({
            'tvd_x1': histogram_tvd(marginal, ref_hist),
            'x1_out_of_range': out_of_range,
            'w1_x1': float(w1),
            'x1_mean_err': float((mass[:, None] * x1).sum() / self.n_wells) - float(self.mean[0]),
        })
        if return_hist:
            return ret, hist
        return ret


def _double_well_energy(x, a, b, c):
    """Evaluates the 1D quartic energy on a numpy array"""
    return a * x + b * x ** 2 + c * x ** 4


def _double_well_normaliser(a, b, c, bound):
    """Returns log Z of the 1D quartic marginal and the mass of its two wells

    Args:
        * a (float): Linear coefficient
        * b (float): Quadratic coefficient
        * c (float): Quartic coefficient
        * bound (float): Truncation bound of the quadrature

    Returns:
        * log_Z_1d (float): log of the 1D normaliser
        * well_mass (tuple of two floats): Mass left and right of the origin, summing to 1
    """
    def density(x):
        return math.exp(-_double_well_energy(x, a, b, c))

    left, _ = quad(density, -bound, 0.0, limit=500)
    right, _ = quad(density, 0.0, bound, limit=500)
    total = left + right
    return math.log(total), (left / total, right / total)


def _double_well_grid(a, b, c, bound, n_grid):
    """Returns the grid, the normalised CDF and the first two moments of the 1D quartic marginal"""
    grid = np.linspace(-bound, bound, n_grid)
    log_density = -_double_well_energy(grid, a, b, c)
    density = np.exp(log_density - log_density.max())
    density = density / np.trapezoid(density, grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    mean = np.trapezoid(density * grid, grid)
    variance = np.trapezoid(density * grid ** 2, grid) - mean ** 2
    return grid, cdf / cdf[-1], (float(mean), float(variance))


def _interpolate_cdf(x, grid, cdf):
    """Evaluates a tabulated CDF at x by linear interpolation"""
    index = torch.searchsorted(grid, x.contiguous()).clamp(1, len(grid) - 1)
    lo, hi = grid[index - 1], grid[index]
    weight = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return cdf[index - 1] + weight * (cdf[index] - cdf[index - 1])


def _w1_to_tabulated_cdf(values, weights, grid, cdf):
    """Evaluates the Wasserstein-1 distance between a weighted sample and a tabulated CDF

    W1 is the integral of |F_n - F|. Between consecutive points of the grid and the sample pooled
    together, F_n is constant and the interpolated F is linear, so every segment integrates in
    closed form, a crossing of the two included. Mass outside the grid, where F is 0 or 1, adds
    its distance to the nearest end.

    Args:
        * values (torch.Tensor of shape (n,)): Sample
        * weights (torch.Tensor of shape (n,)): Weights summing to one
        * grid (torch.Tensor of shape (n_grid,)): Increasing grid the CDF is tabulated on
        * cdf (torch.Tensor of shape (n_grid,)): CDF on the grid, from 0 to 1

    Returns:
        * w1 (torch.Tensor of shape ()): The distance
    """
    order = torch.argsort(values)
    values, weights = values[order], weights[order]
    outside = (weights * ((grid[0] - values).clamp_min(0.0)
                          + (values - grid[-1]).clamp_min(0.0))).sum()

    # F_n at every pooled point, right-continuous, and F interpolated there
    inside = values.clamp(grid[0], grid[-1])
    points = torch.sort(torch.cat([grid, inside])).values
    cumulative = torch.cat([torch.zeros_like(weights[:1]), weights.cumsum(0)])
    empirical = cumulative[torch.searchsorted(inside, points, right=True)]
    exact = _interpolate_cdf(points, grid, cdf)

    # On each segment F_n - F runs linearly from g0 to g1
    g0, g1 = empirical[:-1] - exact[:-1], empirical[:-1] - exact[1:]
    lengths = points[1:] - points[:-1]
    same_sign = lengths * 0.5 * (g0.abs() + g1.abs())
    crossing = lengths * 0.5 * (g0 ** 2 + g1 ** 2) / (g0.abs() + g1.abs()).clamp_min(1e-300)
    return torch.where(g0 * g1 >= 0, same_sign, crossing).sum() + outside


def _interpolate_inverse_cdf(u, cdf, grid):
    """Maps uniforms through a tabulated inverse CDF by linear interpolation

    Args:
        * u (torch.Tensor of shape (n, k)): Uniform draws
        * cdf (torch.Tensor of shape (n_grid,)): Increasing CDF values
        * grid (torch.Tensor of shape (n_grid,)): The grid the CDF was tabulated on

    Returns:
        * x (torch.Tensor of the same shape as u): The quantiles
    """
    index = torch.searchsorted(cdf, u.reshape(-1).contiguous()).clamp(1, len(cdf) - 1)
    lo, hi = cdf[index - 1], cdf[index]
    weight = (u.reshape(-1) - lo) / (hi - lo).clamp_min(torch.finfo(cdf.dtype).tiny)
    x = grid[index - 1] + weight * (grid[index] - grid[index - 1])
    return x.reshape(u.shape)
