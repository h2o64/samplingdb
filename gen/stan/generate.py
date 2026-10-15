# Generate the Tier 2 reference posteriors with Stan

# Libraries
import argparse, json, math, pprint, time
import numpy as np
import torch
from cmdstanpy import CmdStanModel
from samplingdb import build as build_target
from samplingdb.data import cache_dir, manifest, sha256
from samplingdb.targets.bayesian import Brownian, LogisticRegression, Seeds

# Per-target sampler settings. The logistic posteriors are log-concave and need nothing special;
# Seeds and Brownian are funnels in their scale parameters, so adapt_delta goes up and the tree is
# allowed to grow (Stan's guidance: raise delta towards 0.9-0.95 for difficult geometry, and raise
# max_depth when the tree saturates). LGCP is 1600 dense dimensions, so it gets more chains.
#
# thin is what makes 100 000 stored draws worth ~100 000. At 10 000 unthinned draws (2026-10-09)
# the worst ESS_bulk was 9776-10402 for the four logistic regressions, 970 for Seeds, 633 for
# Brownian and 1224 of 2000 for LGCP, i.e. tau ~1, ~10, ~16 and ~1.6 draws; thin is about three
# times that, because thinning a fat-tailed autocorrelation does not divide tau by the factor.
SETTINGS = {
    'logistic': {'adapt_delta': 0.8, 'max_treedepth': 10, 'chains': 4, 'warmup': 1000,
                 'thin': 1},
    'seeds': {'adapt_delta': 0.95, 'max_treedepth': 12, 'chains': 4, 'warmup': 2000,
              'thin': 32},
    'brownian': {'adapt_delta': 0.95, 'max_treedepth': 12, 'chains': 4, 'warmup': 2000,
                 'thin': 48},
    'lgcp': {'adapt_delta': 0.8, 'max_treedepth': 10, 'chains': 16, 'warmup': 500, 'thin': 3},
}

# What a reference set has to clear before it is written out (Vehtari et al. 2021), and the share
# of its stored draws that have to be independent
MAX_RHAT = 1.01
MIN_EFFICIENCY = 0.9


def build(name):
    """Returns the (target, stan_file, data, settings) tuple for one Tier 2 target

    Args:
        * name (str): Its catalog name, which is also the file its test split reads

    Returns:
        * target (BayesianTarget): The torch target
        * stan_file (str): The Stan model, in gen/stan
        * data (dict): Its data block
        * settings (dict): A copy of its sampler settings
    """
    target = build_target(name)
    if isinstance(target, LogisticRegression):
        data = {
            'n_data': len(target.labels),
            'dim': target.dim,
            'features': target.features.tolist(),
            'labels': target.labels.to(torch.int64).tolist(),
            'sigma_w': target.sigma_w,
        }
        return target, 'logistic.stan', data, dict(SETTINGS['logistic'])
    if isinstance(target, Seeds):
        data = {
            'n_groups': target.n_groups,
            'r': target.r.to(torch.int64).tolist(),
            'n': target.n.to(torch.int64).tolist(),
            'x1': target.x1.tolist(),
            'x2': target.x2.tolist(),
            'alpha': target.alpha,
            'beta': target.beta,
            'coefficient_scale': target.coefficient_scale,
        }
        return target, 'seeds.stan', data, dict(SETTINGS['seeds'])
    if isinstance(target, Brownian):
        data = {
            'n_steps': target.n_steps,
            'y': target.y.tolist(),
            'observed': target.observed.to(torch.int64).tolist(),
            'prior_log_scale': target.prior_log_scale,
        }
        return target, 'brownian.stan', data, dict(SETTINGS['brownian'])
    data = {
        'dim': target.dim,
        'cholesky': target.cholesky.tolist(),
        'mu': target.mean.tolist(),
        'counts': target.counts.tolist(),
        'area': target.area,
        'whitened': int(target.whitened),
    }
    return target, 'lgcp.stan', data, dict(SETTINGS['lgcp'])


def check_log_prob(model, data, target, n=8):
    """Asserts Stan's log density equals the torch one, which is the anchor for this whole tier

    Both are the same unnormalised posterior in the same unconstrained coordinates, so they must
    agree constant included -- the Stan files are written with explicit target += for that reason.
    jacobian=False is correct here because the models declare their parameters unconstrained and
    add every Jacobian by hand, exactly as the torch targets do.

    NOTE: sig_figs=18 is load-bearing. Stan returns log_prob through a text file, and at the
    default 6 significant figures the round-trip error is ~4e-4 on a density of magnitude 400 --
    enough to look like a modelling difference. At 18 the two agree to 6e-14.

    Args:
        * model (CmdStanModel): The compiled model
        * data (dict): Its data block
        * target (BayesianTarget): The torch target
        * n (int): Number of random points to check (default is 8)

    Returns:
        * error (float): Largest absolute disagreement over the points checked
    """
    generator = torch.Generator().manual_seed(0)
    x = 0.3 * torch.randn(n, target.dim, generator=generator, dtype=torch.float64)
    names = stan_parameter_names(target)
    error = 0.0
    for row in x:
        params = unflatten(target, row, names)
        stan_value = float(model.log_prob(data=data, params=params, jacobian=False,
                                          sig_figs=18)['lp__'].iloc[0])
        error = max(error, abs(stan_value - float(target.log_prob(row[None]))))
    return error


def stan_parameter_names(target):
    """Returns the Stan parameter blocks of a target, in samplingdb's coordinate order"""
    if isinstance(target, LogisticRegression):
        return [('w', target.dim)]
    if isinstance(target, Seeds):
        return [('a', 4), ('b', target.n_groups), ('log_tau', 1)]
    if isinstance(target, Brownian):
        return [('log_inn', 1), ('log_obs', 1), ('locations', target.n_steps)]
    return [('x', target.dim)]


def unflatten(target, row, names):
    """Splits a flat coordinate vector into the dict Stan's log_prob expects"""
    params, offset = {}, 0
    for name, size in names:
        block = row[offset:offset + size]
        params[name] = float(block[0]) if size == 1 else block.tolist()
        offset += size
    return params


def flatten(draws, names, seed=0):
    """Concatenates Stan's per-parameter draws into samplingdb's coordinate order, then shuffles

    NOTE: the shuffle is load-bearing. cmdstanpy returns draws chain-blocked, so the first quarter
    of the array is chain 1 -- and every consumer of test_samples(n=...) would silently get the
    leading draws of one chain, which for a coordinate that mixes slowly is not a posterior sample
    at all. Two contiguous halves of a chain-blocked Brownian array disagreed by 44% on one
    coordinate's standard deviation, against 4% once permuted. The permutation is seeded, so the
    reference set stays reproducible.

    Args:
        * draws (CmdStanMCMC): The fit
        * names (list of tuple): Parameter blocks, in samplingdb's coordinate order
        * seed (int): Seed of the permutation (default is 0)

    Returns:
        * samples (numpy.ndarray of shape (n_draws, dim)): The draws, chain-interleaved
    """
    columns = []
    for name, size in names:
        value = draws.stan_variable(name)
        columns.append(value.reshape(len(value), -1) if size > 1 else value.reshape(-1, 1))
    samples = np.concatenate(columns, axis=-1)
    return samples[np.random.default_rng(seed).permutation(len(samples))]


def main():
    """Runs Stan on one Tier 2 target and writes its reference draws into the cache"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='ionosphere')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--chains', type=int, default=None)
    parser.add_argument('--thin', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--skip_log_prob_check', action='store_true')
    args = parser.parse_args()

    config = vars(args)
    pprint.pprint(config)

    # Build the target and its Stan twin from the same object, so they cannot drift apart
    target, stan_file, data, settings = build(args.target)
    for key in ['chains', 'thin', 'warmup']:
        if config[key] is not None:
            settings[key] = config[key]

    # Stan counts iterations before thinning, so each chain runs thin times what it keeps
    settings['draws'] = math.ceil(args.n_samples / settings['chains']) * settings['thin']
    print('target', target.name, 'dim', target.dim, 'settings', settings)

    model = CmdStanModel(stan_file='gen/stan/' + stan_file)

    # The anchor: Stan's log density must equal the torch one
    if not args.skip_log_prob_check:
        error = check_log_prob(model, data, target)
        print('max |stan log_prob - torch log_prob| = %.3e' % error)
        assert error < 1e-9, 'the Stan model and the torch target are not the same density'

    # Sample
    start = time.time()
    draws = model.sample(
        data=data, chains=settings['chains'], parallel_chains=settings['chains'],
        iter_warmup=settings['warmup'], iter_sampling=settings['draws'], thin=settings['thin'],
        seed=args.seed, adapt_delta=settings['adapt_delta'],
        max_treedepth=settings['max_treedepth'], show_progress=False,
    )
    elapsed = time.time() - start

    # Diagnose before writing anything: a reference set that failed its checks is worse than none.
    # The ESS is of the KEPT draws, since Stan summarises after thinning
    summary = draws.summary()
    parameters = summary.index[summary.index != 'lp__']
    rhat = float(summary.loc[parameters, 'R_hat'].max())
    ess = float(summary.loc[parameters, 'ESS_bulk'].min())
    divergences = int(sum(fit['divergent__'].sum() for fit in [draws.method_variables()]))
    samples = flatten(draws, stan_parameter_names(target)).astype(np.float64)
    efficiency = ess / len(samples)
    print('wall clock %.0f s   max R-hat %.4f   min ESS_bulk %.0f of %d kept (%.0f%%)   '
          'divergences %d' % (elapsed, rhat, ess, len(samples), 100 * efficiency, divergences))
    assert rhat < MAX_RHAT, 'R-hat %.4f exceeds %.2f' % (rhat, MAX_RHAT)
    assert efficiency > MIN_EFFICIENCY, (
        'ESS_bulk %.0f is %.0f%% of the %d kept draws: raise --thin to about %d'
        % (ess, 100 * efficiency, len(samples), math.ceil(settings['thin'] / efficiency)))
    assert divergences == 0, '%d divergent transitions' % divergences

    # Write the draws in samplingdb's coordinate order
    assert samples.shape[1] == target.dim
    folder = cache_dir() / 'stan'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (target.name + '.npy')
    np.save(path, samples)

    # A manifest, so the run behind a reference set is recoverable
    protocol = dict(settings)
    protocol.update({'seed': args.seed, 'cmdstan': model.exe_info().get('stan_version_string', ''),
                     'max_rhat': rhat, 'min_ess_bulk': ess, 'efficiency': efficiency,
                     'divergences': divergences,
                     'wall_clock_seconds': round(elapsed, 1), 'n_draws': len(samples)})
    with open(folder / (target.name + '_manifest.json'), 'w') as handle:
        handle.write(manifest(target.name, protocol, {path.name: sha256(path)}))
    print('wrote', path, json.dumps(protocol))

    # A first look at the posterior it produced
    print('posterior mean of the first four coordinates', samples.mean(0)[:4].round(4))
    print('posterior sd   of the first four coordinates', samples.std(0)[:4].round(4))


if __name__ == '__main__':
    main()
