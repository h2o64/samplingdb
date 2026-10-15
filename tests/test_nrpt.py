# Non-reversible parallel tempering must sample its target, tune its own schedule, and permute

# Libraries
import math
import torch
from samplingdb.targets.base import generator
from nrpt import GaussianReference, PotentialReference, hmc_explorer, linear_schedule, \
    non_reversible_parallel_tempering, update_schedule


def shifted_gaussian(dim=4, offset=2.0, shift=3.7):
    """Returns log_prob, score and the exact log Z of an off-centre unit Gaussian"""
    mean = offset * torch.ones(dim, dtype=torch.float64)

    def log_prob(x):
        return shift - 0.5 * ((x - mean) ** 2).sum(-1) - 0.5 * dim * math.log(2 * math.pi)

    return log_prob, lambda x: -(x - mean), shift


def run(dim=4, n_rungs=8, n_chains=32, n_rounds=8, seed=0):
    """Runs the sampler on the shifted Gaussian and returns everything it produced"""
    log_prob, score, log_z = shifted_gaussian(dim=dim)
    reference = GaussianReference(dim, scale=1.0)
    gen = generator(seed)
    x0 = reference.sample(n_rungs * n_chains, generator=gen).reshape(n_rungs, n_chains, dim)
    x, diagnostics = non_reversible_parallel_tempering(
        log_prob, score, x0, linear_schedule(n_rungs), n_rounds=n_rounds, reference=reference,
        step_size=0.3, generator=gen, return_diagnostics=True)
    return x, diagnostics, log_z


def test_it_samples_the_target_and_estimates_its_normaliser():
    """Mean, standard deviation and log Z, all of which are known in closed form here"""
    x, diagnostics, log_z = run()

    assert x.shape == (2 ** 8, 32, 4)
    assert abs(float(x.mean()) - 2.0) < 0.05
    assert abs(float(x.reshape(-1, 4).std(0).mean()) - 1.0) < 0.05

    # The stepping stone is against a normalised reference, so it is log Z rather than a ratio
    assert abs(diagnostics['log_normaliser'] - log_z) < 0.2


def test_the_schedule_adapts_towards_equal_rejection():
    """The optimal schedule is the equi-rejection one, so the spread across pairs has to shrink"""
    _, diagnostics, _ = run()
    rejection = diagnostics['rejection']
    first = diagnostics['rounds'][0]['rejection']

    # A linear schedule on this target is already close, so the test is that adaptation does not
    # make it worse and that the final spread is genuinely small
    assert float(rejection.max() - rejection.min()) <= float(first.max() - first.min()) + 1e-9
    assert float(rejection.max() - rejection.min()) < 0.05
    assert float(diagnostics['barrier']) > 0.0


def test_the_barrier_predicts_the_round_trip_rate_from_above():
    """1 / (2 + 2 Lambda) is what an ideal local kernel would reach, so it is an upper bound"""
    _, diagnostics, _ = run()

    assert diagnostics['round_trip_rate'] > 0.0
    assert diagnostics['round_trip_rate'] <= diagnostics['predicted_round_trip_rate'] + 0.05
    assert diagnostics['recommended_n_rungs'] >= 2


def test_round_trips_cannot_exceed_the_length_of_the_ladder():
    """One swap pass per scan moves a configuration one rung, which bounds the count"""
    n_rungs, n_rounds = 8, 8
    _, diagnostics, _ = run(n_rungs=n_rungs, n_rounds=n_rounds)

    # A round trip is 2 * (n_rungs - 1) rungs of travel, and the count sums over the n_rungs
    # configurations sharing one ladder
    bound = n_rungs * (2 ** n_rounds) / (2.0 * (n_rungs - 1))
    assert float(diagnostics['round_trips'].max()) <= bound


def test_a_swap_permutes_the_configurations_rather_than_rewriting_them():
    """With a local kernel that does nothing, every collected state must be an initial one

    This is the property that makes swapping annealing parameters equivalent to swapping states,
    and it is what the implementation relies on to leave the configurations untouched.
    """
    dim, n_rungs, n_chains = 3, 6, 8
    log_prob, score, _ = shifted_gaussian(dim=dim)
    reference = GaussianReference(dim, scale=1.0)
    gen = generator(0)
    x0 = reference.sample(n_rungs * n_chains, generator=gen).reshape(n_rungs, n_chains, dim)

    def frozen(log_prob, score, x, step_size, generator):
        return x, torch.zeros(len(x), dtype=x.dtype, device=x.device)

    # reference=None so that no rung is refreshed by an exact draw either
    x, diagnostics = non_reversible_parallel_tempering(
        log_prob, score, x0, linear_schedule(n_rungs, beta_min=0.05), explorer=frozen,
        n_rounds=4, reference=None, generator=gen, return_diagnostics=True)

    # Every state that came out is one of the states that went in, on its own chain
    for chain in range(n_chains):
        started = x0[:, chain]
        for state in x[:, chain]:
            assert bool((state[None] == started).all(-1).any())


def test_it_runs_without_a_reference_and_on_a_bounded_support():
    """reference=None is the pure tempering the corpus already uses, and -inf must not become NaN"""
    dim, n_rungs, n_chains = 2, 6, 16
    log_prob, score, _ = shifted_gaussian(dim=dim)

    # A hard wall, so log_prob is -inf on part of the reference's support
    def walled(x):
        return torch.where(x[..., 0] > -1.0, log_prob(x), torch.full_like(x[..., 0], -math.inf))

    gen = generator(0)
    x0 = torch.zeros(n_rungs, n_chains, dim, dtype=torch.float64)
    x, diagnostics = non_reversible_parallel_tempering(
        walled, score, x0, linear_schedule(n_rungs, beta_min=0.1),
        explorer=hmc_explorer(n_leapfrog=5), n_rounds=5, reference=None, step_size=0.5,
        generator=gen, return_diagnostics=True)

    assert torch.isfinite(x).all()
    assert math.isfinite(diagnostics['barrier'])
    assert bool((x[..., 0] > -1.0).all())


def test_update_schedule_equalises_the_cumulative_barrier():
    """Algorithm 2 places the rungs at equal heights of the cumulative barrier"""
    beta = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)

    # A barrier concentrated at the cold end, so the new schedule must crowd there
    rejection = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05], dtype=torch.float64)
    cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), rejection.cumsum(0)])
    updated = update_schedule(beta, cumulative)

    assert float(updated[0]) == 0.0 and float(updated[-1]) == 1.0
    assert bool((updated[1:] >= updated[:-1]).all())
    assert float(updated[1]) < float(beta[1])

    # Re-reading the same barrier at the new rungs gives equal increments, which is the fixed point
    heights = torch.tensor([float(torch.searchsorted(beta, b).clamp(1, 5)) for b in updated])
    assert len(heights) == 6

    # A flat barrier leaves any schedule alone
    flat = torch.zeros(6, dtype=torch.float64)
    assert torch.allclose(update_schedule(beta, flat), beta)


def test_step_size_calibration_finds_the_same_scale_from_any_start():
    """Acceptance saturates, so only a geometric search can travel orders of magnitude"""
    dim = 16
    log_prob = lambda x: -0.5 * x.pow(2).sum(-1)
    score = lambda x: -x
    x0 = torch.randn(6, 32, dim, generator=generator(0), dtype=torch.float64)

    finals = []
    for start in [1e-4, 1.0, 50.0]:
        _, diagnostics = non_reversible_parallel_tempering(
            log_prob, score, x0, linear_schedule(6, 0.1, dtype=torch.float64),
            explorer=hmc_explorer(n_steps=1, n_leapfrog=10), n_rounds=8, step_size=start,
            generator=generator(0), return_diagnostics=True)
        finals.append(float(diagnostics['step_size'].median()))

    # Five orders of magnitude of starting point collapse to a factor of two of each other
    assert max(finals) / min(finals) < 2.0
    assert 0.3 < min(finals) < 10.0


def test_a_sub_potential_reference_tempers_only_what_it_leaves_out():
    """Solute tempering: the swap statistic is the SOFT part alone, and beta = 1 is still the target

    The target is a Gaussian split into stiff and soft blocks, the stiff one being the reference.
    V = log gamma - log reference then depends on the soft block only, so the barrier must not move
    when the stiff block is made a hundred times stiffer -- which is the whole reason to do this on
    a peptide, where the stiff block is the bonds and the angles and carries most of Var(E).
    """
    dim, n_stiff, n_rungs = 8, 5, 6

    def path(stiff):
        """Builds the target, its score and the stiff reference for one stiff precision"""
        stiff_precision = torch.zeros(dim, dtype=torch.float64)
        stiff_precision[:n_stiff] = stiff
        soft_precision = torch.zeros(dim, dtype=torch.float64)
        soft_precision[n_stiff:] = 4.0
        total = stiff_precision + soft_precision
        reference = PotentialReference(lambda x: -0.5 * (stiff_precision * x.pow(2)).sum(-1),
                                       lambda x: -stiff_precision * x)
        return (lambda x: -0.5 * (total * x.pow(2)).sum(-1), lambda x: -total * x, reference)

    def sample(stiff):
        """Runs the sampler on that path and returns the cold rung and the diagnostics"""
        log_prob, score, reference = path(stiff)
        gen = generator(0)
        x0 = 0.05 * torch.randn(n_rungs, 64, dim, generator=gen, dtype=torch.float64)
        return non_reversible_parallel_tempering(
            log_prob, score, x0, linear_schedule(n_rungs, 0.2, dtype=torch.float64),
            explorer=hmc_explorer(n_steps=2, n_leapfrog=10), n_rounds=10, reference=reference,
            step_size=0.05, generator=gen, return_diagnostics=True)

    x, diagnostics = sample(100.0)

    # The cold rung is the FULL target, stiff coordinates included
    standard_deviation = x.reshape(-1, dim).std(0)
    assert abs(float(standard_deviation[0]) - 0.1) < 0.02
    assert abs(float(standard_deviation[-1]) - 0.5) < 0.05

    # A hundredfold stiffer stiff block leaves the barrier where it was
    _, stiffer = sample(10000.0)
    assert abs(stiffer['barrier'] - diagnostics['barrier']) < 0.15 * diagnostics['barrier']


def test_the_barrier_density_localises_the_barrier():
    """barrier_density integrates to Lambda and is the caloric curve's standard deviation"""
    _, diagnostics, _ = run()
    beta = diagnostics['beta']
    density = diagnostics['barrier_density']

    total = float((density * (beta[1:] - beta[:-1])).sum())
    assert abs(total - diagnostics['barrier']) < 1e-6

    # lambda(beta) = sqrt(Var_beta(V) / pi) for a Gaussian V, which this target's is
    predicted = diagnostics['value_std'][:-1] / math.sqrt(math.pi)
    assert abs(float((density / predicted).median()) - 1.0) < 0.35


def test_n_scans_sets_the_collection_round_independently_of_n_rounds():
    """The doubling ties warm-up length to collection length; n_scans is what unties them"""
    log_prob, score, _ = shifted_gaussian(dim=4)
    reference = GaussianReference(4, scale=1.0)
    gen = generator(0)
    x0 = reference.sample(6 * 32, generator=gen).reshape(6, 32, 4)

    for n_scans, thin, expected in [(None, 8, 2 ** 6 // 8), (400, 8, 50), (400, 25, 16)]:
        x, diagnostics = non_reversible_parallel_tempering(
            log_prob, score, x0, linear_schedule(6), n_rounds=6, reference=reference,
            step_size=0.3, generator=generator(0), thin=thin, n_scans=n_scans,
            return_diagnostics=True)
        assert len(x) == expected, (n_scans, thin, len(x))
        assert diagnostics['rounds'][-1]['n_scans'] == (n_scans or 2 ** 6)


def test_n_burn_is_discarded_from_the_front_of_the_collection_round():
    """A frozen run has no earlier rounds to serve as burn-in, so it needs its own"""
    log_prob, score, _ = shifted_gaussian(dim=4)
    reference = GaussianReference(4, scale=1.0)
    x0 = reference.sample(6 * 32, generator=generator(0)).reshape(6, 32, 4)

    kept, diagnostics = non_reversible_parallel_tempering(
        log_prob, score, x0, linear_schedule(6), n_rounds=1, reference=reference, step_size=0.3,
        generator=generator(0), thin=10, n_scans=200, n_burn=150, return_diagnostics=True)

    # 200 collected after 150 discarded, and every average is over the 200
    assert len(kept) == 20
    assert diagnostics['rounds'][-1]['n_scans'] == 200
    assert torch.isfinite(kept).all()

    # A burn-in longer than the collection is still burned in full, on top of it
    kept, diagnostics = non_reversible_parallel_tempering(
        log_prob, score, x0, linear_schedule(6), n_rounds=1, reference=reference, step_size=0.3,
        generator=generator(0), thin=10, n_scans=50, n_burn=150, return_diagnostics=True)
    assert len(kept) == 5
    assert diagnostics['rounds'][-1]['n_scans'] == 50


def test_a_frozen_run_reproduces_the_calibration_it_came_from():
    """Calibrate once, freeze, run any length: the whole basis of sharded generation

    The doubling costs a permanent 2x -- the discarded rounds sum to 2^n - 2 against a collection
    round of 2^n -- and the adaptation converges long before that, so a converged schedule and step
    size passed back in with n_rounds=1 should behave identically. It does, PROVIDED the start is one
    the frozen step size can move: a step size frozen at its converged value is permanently rejected
    on a high-energy configuration, and then nothing relaxes and no burn-in rescues it.
    """
    log_prob, score, _ = shifted_gaussian(dim=4)
    reference = GaussianReference(4, scale=1.0)
    x0 = reference.sample(6 * 32, generator=generator(0)).reshape(6, 32, 4)

    _, calibrated = non_reversible_parallel_tempering(
        log_prob, score, x0, linear_schedule(6), n_rounds=9, reference=reference, step_size=0.3,
        generator=generator(0), thin=8, return_diagnostics=True)

    _, frozen = non_reversible_parallel_tempering(
        log_prob, score, x0, calibrated['beta'], n_rounds=1, reference=reference,
        step_size=calibrated['step_size'], adaptation_gain=0.0, generator=generator(1), thin=8,
        n_scans=512, n_burn=128, return_diagnostics=True)

    # Nothing adapted: what went in came back out
    assert torch.equal(frozen['beta'], calibrated['beta'])
    assert torch.equal(frozen['step_size'], calibrated['step_size'])

    # And it is the same sampler, by the two numbers that describe one
    assert abs(frozen['barrier'] - calibrated['barrier']) < 0.15 * calibrated['barrier']
    assert abs(frozen['round_trip_rate'] - calibrated['round_trip_rate']) \
        < 0.3 * calibrated['round_trip_rate']


def test_a_frozen_run_continues_from_its_rung_state():
    """A run split into segments is the same chain as one run, which is what lets a shard save as it
    goes: the rung-ordered state puts every configuration back on the rung it left"""
    log_prob, score, _ = shifted_gaussian(dim=4)
    reference = GaussianReference(4, scale=1.0)
    x0 = reference.sample(6 * 32, generator=generator(0)).reshape(6, 32, 4)
    beta = linear_schedule(6)
    frozen = dict(n_rounds=1, reference=reference, step_size=0.3, adaptation_gain=0.0, thin=1,
                  return_diagnostics=True)

    kept, first = non_reversible_parallel_tempering(log_prob, score, x0, beta, n_scans=64,
                                                    generator=generator(1), **frozen)

    # The last kept state is the beta = 1 rung of the rung-ordered state, and the machine-ordered
    # one holds the same configurations in another order
    assert torch.equal(kept[-1], first['rung_state'][-1])
    assert torch.equal(first['rung_state'].sort(0).values, first['state'].sort(0).values)

    # Continuing from it keeps every rung in its ensemble: the cold rung's mean stays at the target's
    _, second = non_reversible_parallel_tempering(log_prob, score, first['rung_state'], beta,
                                                  n_scans=256, generator=generator(2), **frozen)
    assert abs(second['barrier'] - first['barrier']) < 0.3 * first['barrier']
    assert abs(float(second['rung_state'][-1].mean()) - 2.0) < 0.3


def test_independent_runs_concatenate_without_lengthening_the_correlation():
    """The exactness claim gen/merge_shards.py rests on: effective sample sizes ADD

    Shards are independent runs under different seeds, merged along the CHAIN axis. Independent
    chains cannot correlate with each other, so concatenating them cannot lengthen the correlation
    along the scan axis, and tau of the merged set has to match the shards'.
    """
    from diagnostics import effective_sample_size

    log_prob, score, _ = shifted_gaussian(dim=4)
    reference = GaussianReference(4, scale=1.0)
    x0 = reference.sample(6 * 32, generator=generator(0)).reshape(6, 32, 4)

    shards = [non_reversible_parallel_tempering(
        log_prob, score, x0, linear_schedule(6), n_rounds=8, reference=reference, step_size=0.3,
        generator=generator(seed), thin=4) for seed in [0, 1]]

    taus, effective = [], 0.0
    for shard in shards:
        ess = effective_sample_size({'x0': shard[..., 0].T})
        taus.append(ess['tau'])
        effective += ess['n_effective']

    merged = torch.cat(shards, dim=1)
    ess = effective_sample_size({'x0': merged[..., 0].T})
    assert merged.shape == (shards[0].shape[0], 2 * shards[0].shape[1], 4)
    assert abs(ess['tau'] - sum(taus) / 2) < 0.35 * sum(taus) / 2
    assert abs(ess['n_effective'] - effective) < 0.35 * effective


def test_a_fixed_schedule_adapts_the_step_sizes_only():
    """adapt_schedule=False keeps a measured ladder and still tunes every level's step size"""
    log_prob, score, _ = shifted_gaussian(dim=4)
    reference = GaussianReference(4, scale=1.0)
    x0 = reference.sample(6 * 32, generator=generator(0)).reshape(6, 32, 4)
    beta = torch.tensor([0.0, 0.05, 0.1, 0.3, 0.6, 1.0], dtype=torch.float64)
    _, diagnostics = non_reversible_parallel_tempering(
        log_prob, score, x0, beta, n_rounds=8, reference=reference, step_size=0.01,
        adapt_schedule=False, generator=generator(0), return_diagnostics=True)
    assert torch.equal(diagnostics['beta'], beta)
    assert float(diagnostics['step_size'].min()) > 0.05
    acceptance = diagnostics['explorer_acceptance']
    assert bool(((acceptance > 0.4) & (acceptance < 0.9)).all())
