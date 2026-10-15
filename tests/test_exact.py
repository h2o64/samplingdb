# The exact Brownian motion sampler must reproduce the posterior of its two log scales

# Libraries
import math, os, pathlib, subprocess, sys
import numpy as np
from samplingdb import Brownian

# The generator is a script rather than a module, so the test runs it as one
SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'gen' / 'exact' / 'brownian.py'


def log_scale_quadrature(target, n_grid=801):
    """Returns the posterior of the two log scales on a grid, the locations integrated out

    Written independently of the sampler's own marginal likelihood. Given the scales, the observed
    y_o ~ N(0, alpha_inn^2 M + alpha_obs^2 I) with M(i, j) = min(i, j) over the observed steps, so
    one eigendecomposition of M turns the likelihood at every grid point into a product of twenty
    univariate Gaussians. The grid spans [-12, 4] on both axes, like the sampler's bound search,
    and refining it from 801 to 2401 points per axis moves no moment in its sixth decimal.

    Args:
        * target (Brownian): The posterior
        * n_grid (int): Grid points per axis (default is 801)

    Returns:
        * log_inn (numpy.ndarray of shape (n_grid, n_grid)): log alpha_inn at every grid point
        * log_obs (numpy.ndarray of shape (n_grid, n_grid)): log alpha_obs at every grid point
        * weights (numpy.ndarray of shape (n_grid, n_grid)): Normalised posterior mass
    """
    # The observed steps and the eigenbasis of their random-walk covariance
    observed = target.observed.numpy()
    y = target.y.numpy()[observed]
    steps = np.arange(1, target.n_steps + 1)[observed]
    eigenvalues, eigenvectors = np.linalg.eigh(np.minimum(steps[:, None], steps[None, :])
                                               .astype(np.float64))
    projected = (eigenvectors.T @ y) ** 2

    # The log posterior on the grid, the N(0, 2^2) priors included
    grid = np.linspace(-12.0, 4.0, n_grid)
    log_inn, log_obs = np.meshgrid(grid, grid, indexing='ij')
    variance = np.exp(2 * log_inn)[..., None] * eigenvalues + np.exp(2 * log_obs)[..., None]
    log_posterior = (-0.5 * (projected / variance + np.log(variance)).sum(-1)
                     - 0.5 * (log_inn ** 2 + log_obs ** 2) / target.prior_log_scale ** 2)

    # Normalise the mass
    weights = np.exp(log_posterior - log_posterior.max())
    weights /= weights.sum()
    return log_inn, log_obs, weights


def test_the_exact_sampler_reproduces_the_quadrature_of_the_log_scales(tmp_path):
    """The draws of gen/exact/brownian.py match a quadrature of the scale marginal

    The draws are independent, so every tolerance is five Monte Carlo standard errors. At 50 000
    draws that is about 0.021 on the sd of log alpha_obs, whose left tail makes its sample sd
    noisy, against the 0.041 by which a NUTS run with clean diagnostics missed it (0.4264 against
    0.4675), and about 0.0013 on P(log alpha_obs < -5), against the 0.0028 by which the same run
    missed that (0.00069 against 0.0034).
    """
    target = Brownian()
    subprocess.run([sys.executable, str(SCRIPT), '--n_samples', '50000'], check=True,
                   env=dict(os.environ, SAMPLINGDB_CACHE=str(tmp_path)))
    draws = np.load(tmp_path / 'exact' / (target.reference_key() + '.npy'))
    n = len(draws)
    log_inn, log_obs, weights = log_scale_quadrature(target)

    # The first two moments of each log scale, with the delta-method error of a sample sd
    sds = []
    for column, axis in enumerate([log_inn, log_obs]):
        mean = float((weights * axis).sum())
        sds.append(math.sqrt(float((weights * (axis - mean) ** 2).sum())))
        x = draws[:, column]
        deviation = (x - x.mean()) ** 2
        assert abs(x.mean() - mean) < 5 * x.std() / math.sqrt(n)
        assert abs(x.std() - sds[-1]) < 5 * deviation.std() / math.sqrt(n) / (2 * x.std())

    # The quadrature reads the sd of log alpha_obs that docs/reference-sets.md quotes
    assert round(sds[1], 4) == 0.4675

    # The left tail of the observation scale, where the data stop constraining it
    tail = float(weights[log_obs < -5].sum())
    observed_tail = float((draws[:, 1] < -5).mean())
    assert abs(observed_tail - tail) < 5 * math.sqrt(tail * (1 - tail) / n)
