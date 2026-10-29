# Generate the Brownian motion reference set with an exact sampler

# Libraries
import argparse, math, pathlib, pprint, sys, time
import numpy as np
import torch
from samplingdb import Brownian
from samplingdb.data import cache_dir, sha256
from samplingdb.targets import spec_id
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from manifest import manifest

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=100000)
parser.add_argument('--batch', type=int, default=200000)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# The model is Gaussian in the 30 locations given the two scales, so the posterior factorises
# exactly: the two log scales from their marginal, then the locations from their Gaussian
# conditional. With x_1 ~ N(0, alpha_inn^2) and Gaussian increments, Cov(x_i, x_j) =
# alpha_inn^2 min(i, j), and the observed y_o ~ N(0, alpha_inn^2 M_oo + alpha_obs^2 I)
target = Brownian()
torch.manual_seed(args.seed)
observed = target.observed.nonzero(as_tuple=True)[0]
y = target.y[observed]
steps = torch.arange(1, target.n_steps + 1, dtype=torch.float64)
walk = torch.minimum(steps[:, None], steps[None, :])
walk_oo = walk[observed][:, observed]
walk_precision = torch.linalg.inv(walk)
mask = torch.diag(target.observed.to(torch.float64))
identity = torch.eye(len(observed), dtype=torch.float64)


def marginal_likelihood(log_inn, log_obs):
    """Returns log N(y_o; 0, alpha_inn^2 M_oo + alpha_obs^2 I), the locations integrated out"""
    covariance = (torch.exp(2 * log_inn)[:, None, None] * walk_oo
                  + torch.exp(2 * log_obs)[:, None, None] * identity)
    factor = torch.linalg.cholesky(covariance)
    solved = torch.cholesky_solve(y[None, :, None].expand(len(log_inn), -1, 1), factor)[..., 0]
    log_det = 2 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
    return -0.5 * (y * solved).sum(-1) - 0.5 * log_det - 0.5 * len(y) * math.log(2 * math.pi)


def conditional(log_inn, log_obs):
    """Returns the mean of the locations given the scales and the data, and the Cholesky factor of
    their precision, alpha_inn^-2 M^-1 + alpha_obs^-2 P_o

    NOTE: computed in precision form, because subtracting two covariances loses the smallest
    eigenvalues once the two scales are orders of magnitude apart.
    """
    precision = (torch.exp(-2 * log_inn)[:, None, None] * walk_precision
                 + torch.exp(-2 * log_obs)[:, None, None] * mask)
    factor = torch.linalg.cholesky(precision)
    right = torch.exp(-2 * log_obs)[:, None] * target.y * target.observed
    mean = torch.cholesky_solve(right[..., None], factor)[..., 0]
    return mean, factor


# 1. The density check every generator owes (AGENTS.md rule 7): the factorised density must equal
# the target's log_prob up to one constant, log Z, at arbitrary points
points = 2.0 * torch.randn(1000, 2, dtype=torch.float64)
locations = target.y + 0.3 * torch.randn(1000, target.n_steps, dtype=torch.float64)
value = marginal_likelihood(points[:, 0], points[:, 1])
mean, factor = conditional(points[:, 0], points[:, 1])
prior = -0.5 * (points / target.prior_log_scale).pow(2).sum(-1) \
    - 2 * math.log(target.prior_log_scale) - math.log(2 * math.pi)
gaussian = torch.distributions.MultivariateNormal(mean, precision_matrix=factor @ factor.mT)
factorised = prior + value + gaussian.log_prob(locations)
log_z = target.log_prob(torch.cat([points, locations], dim=-1)) - factorised
print('log Z %.3e, spread across 1000 points %.2e' % (log_z.mean(), log_z.std()))
assert float(log_z.std()) < 1e-8, 'the factorised density is not the target density'

# 2. The bound the rejection step needs: the marginal likelihood is maximised on a grid and then
# refined, and every accepted proposal is checked against it below, so an under-estimate cannot
# pass silently
grid = torch.linspace(-12.0, 4.0, 801, dtype=torch.float64)
log_inn, log_obs = torch.meshgrid(grid, grid, indexing='ij')
log_inn, log_obs = log_inn.reshape(-1), log_obs.reshape(-1)
values = torch.cat([marginal_likelihood(log_inn[i:i + 50000], log_obs[i:i + 50000])
                    for i in range(0, len(log_inn), 50000)])
best = torch.stack([log_inn[values.argmax()], log_obs[values.argmax()]]).requires_grad_(True)
optimiser = torch.optim.LBFGS([best], max_iter=200, line_search_fn='strong_wolfe')


def closure():
    optimiser.zero_grad()
    loss = -marginal_likelihood(best[None, 0], best[None, 1]).sum()
    loss.backward()
    return loss


optimiser.step(closure)
bound = float(marginal_likelihood(best[None, 0].detach(), best[None, 1].detach())) + 1e-6
print('marginal likelihood bound %.6f at log scales (%.4f, %.4f)' % (bound, *best.tolist()))

# 3. Rejection from the prior: propose both log scales from their N(0, 2^2) priors and accept with
# probability L(scales) / max L, which is exact whatever the shape of the posterior; then draw the
# locations from their exact conditional. The prior's tails are the posterior's in the direction
# where the data stop constraining alpha_obs, so the proposal covers that tail
start = time.time()
kept, proposed = [], 0
while sum(len(block) for block in kept) < args.n_samples:
    scales = target.prior_log_scale * torch.randn(args.batch, 2, dtype=torch.float64)
    value = marginal_likelihood(scales[:, 0], scales[:, 1])
    assert float(value.max()) <= bound, 'the likelihood bound was exceeded: refine it'
    accept = torch.rand(args.batch, dtype=torch.float64) < torch.exp(value - bound)
    proposed += args.batch

    # x = mean + L^-T z has covariance (L L^T)^-1, the conditional's
    mean, factor = conditional(scales[accept, 0], scales[accept, 1])
    noise = torch.randn(len(mean), target.n_steps, 1, dtype=torch.float64)
    offset = torch.linalg.solve_triangular(factor.mT, noise, upper=True)[..., 0]
    kept.append(torch.cat([scales[accept], mean + offset], dim=-1))
draws = torch.cat(kept)[:args.n_samples]
acceptance = sum(len(block) for block in kept) / proposed
print('%d draws from %d proposals (acceptance %.4f) in %.0f s'
      % (len(draws), proposed, acceptance, time.time() - start))

# What a sampler is scored on, against the quadrature of the scale marginal the bound came from
weights = torch.softmax(values, 0) * torch.exp(-0.5 * (log_inn ** 2 + log_obs ** 2)
                                               / target.prior_log_scale ** 2)
weights = weights / weights.sum()
for column, axis in enumerate([log_inn, log_obs]):
    exact_mean = float((weights * axis).sum())
    exact_sd = float((weights * (axis - exact_mean) ** 2).sum()) ** 0.5
    print('log scale %d: mean %.4f (quadrature %.4f), sd %.4f (quadrature %.4f)'
          % (column, draws[:, column].mean(), exact_mean, draws[:, column].std(), exact_sd))
print('P(log alpha_obs < -5) %.5f' % float((draws[:, 1] < -5).double().mean()))

# Save the draws next to a manifest naming the protocol
folder = cache_dir() / 'exact'
folder.mkdir(parents=True, exist_ok=True)
path = folder / (target.reference_key() + '.npy')
np.save(path, draws.numpy())
protocol = dict(config)
protocol.update({'sampler': 'rejection from the prior on the log scales, exact Gaussian locations',
                 'acceptance': acceptance, 'log_likelihood_bound': bound,
                 'log_normaliser': float(log_z.mean()), 'density_check_spread': float(log_z.std()),
                 'torch': torch.__version__})
(folder / (target.reference_key() + '_manifest.json')).write_text(manifest(
    target.name, spec_id(target.spec()), protocol, {path.name: sha256(path)}))
print('wrote %s  %s' % (path, tuple(draws.shape)))
