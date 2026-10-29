# Non-reversible parallel tempering, round-based and self-tuning (Syed et al. 2022)

# Libraries
import math
import torch
from samplingdb.targets.kernels import metropolis, run_chains


def leapfrog(score, x, momentum, step_size, n_leapfrog, beta=1.0):
    """Integrates Hamilton's equations for beta * log_gamma

    Half a momentum kick at each end and a full one between, which is the standard leapfrog. beta
    may be a tensor broadcasting against x, so parallel tempering integrates every level at its
    own inverse temperature in one call.

    Args:
        * score (function): x -> grad log_gamma(x)
        * x (torch.Tensor of shape (..., dim)): Starting positions
        * momentum (torch.Tensor of the same shape): Starting momenta
        * step_size (float or torch.Tensor): Step size, broadcasting against x
        * n_leapfrog (int): Number of leapfrog steps
        * beta (float or torch.Tensor): Inverse temperature, broadcasting against x
            (default is 1.0)

    Returns:
        * x (torch.Tensor of the same shape): Final positions
        * momentum (torch.Tensor of the same shape): Final momenta
    """
    momentum = momentum + 0.5 * step_size * beta * score(x)
    for leap in range(n_leapfrog):
        x = x + step_size * momentum
        scale = step_size if leap < n_leapfrog - 1 else 0.5 * step_size
        momentum = momentum + scale * beta * score(x)
    return x, momentum


def hmc(log_prob, score, x0, n_steps, step_size, n_leapfrog=10, generator=None,
        return_trace=False, thin=1, return_acceptance=False, per_chain=False):
    """Runs Hamiltonian Monte Carlo with an identity mass matrix, one chain per row of x0

    Args:
        * log_prob (function): x (n_chains, dim) -> (n_chains,)
        * score (function): x (n_chains, dim) -> (n_chains, dim)
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting states
        * n_steps (int): Number of HMC iterations per chain
        * step_size (float): Leapfrog step size
        * n_leapfrog (int): Leapfrog steps per iteration (default is 10)
        * generator (torch.Generator or None): RNG (default is None)
        * return_trace (bool): Whether to return every kept state (default is False)
        * thin (int): Keep one state every thin steps when return_trace (default is 1)
        * return_acceptance (bool): Whether to also return the mean acceptance rate
            (default is False)
        * per_chain (bool): Whether that acceptance is per chain rather than averaged over them
            (default is False)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim), or (n_steps // thin, n_chains, dim) if
            return_trace): The final states, or the whole thinned trace
        if return_acceptance:
            * acceptance (torch.Tensor of shape ()): Mean acceptance rate
    """
    def move(state, gen):
        x = state['x']
        p = torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)
        y, q = leapfrog(score, x.clone(), p.clone(), step_size, n_leapfrog)

        # The Metropolis correction is on the Hamiltonian, so the kinetic terms are the ratio
        kinetic = 0.5 * (p ** 2).sum(-1) - 0.5 * (q ** 2).sum(-1)
        accept, x, log_p = metropolis(state, log_prob(y), y, log_ratio=kinetic, generator=gen)
        return {'x': x, 'log_p': log_p}, accept

    state = {'x': x0.clone(), 'log_p': log_prob(x0)}
    return run_chains(move, state, n_steps, generator=generator, return_trace=return_trace,
                      thin=thin, return_acceptance=return_acceptance, per_chain=per_chain)


class GaussianReference:
    """The isotropic Gaussian a target's prior_scale describes, as an annealing reference

    A reference makes the beta = 0 end of the path tractable: the local kernel there returns
    independent draws, so a round trip ends on a fresh start. Target.prior_scale exists for this.

    NOTE: this density is normalised, so a stepping-stone estimate against it is log Z rather than
    a ratio.
    """

    def __init__(self, dim, scale=1.0, dtype=torch.float64, device=None):
        """Builds the reference

        Args:
            * dim (int): Dimension
            * scale (float): Standard deviation per coordinate (default is 1.0)
            * dtype (torch.dtype): Working dtype (default is torch.float64)
            * device (torch.device or None): Device (default is None)
        """
        self.dim = dim
        self.scale = scale
        self.dtype = dtype
        self.device = device
        self.constant = -dim * (math.log(scale) + 0.5 * math.log(2 * math.pi))

    def log_prob(self, x):
        """Evaluates the normalised log-density"""
        return self.constant - 0.5 * (x / self.scale).pow(2).sum(-1)

    def score(self, x):
        """Evaluates the gradient of the log-density"""
        return -x / self.scale ** 2

    def sample(self, n, generator=None):
        """Draws n independent points

        NOTE: drawn on the generator's own device and moved afterwards, because the corpus seeds
        with a CPU torch.Generator (targets.base.generator) and a CPU generator cannot fill a
        cuda tensor.
        """
        noise = torch.randn(n, self.dim, generator=generator, dtype=self.dtype,
                            device=None if generator is None else generator.device)
        return (self.scale * noise).to(self.device)


class PotentialReference:
    """A sub-potential of the target as the beta = 0 end of the path, i.e. solute tempering

    The linear path is log gamma_beta = (1 - beta) log reference + beta log gamma. Setting the
    reference to exp(-E_stiff / kT) makes that

        log gamma_beta = -(E_stiff + beta * E_soft) / kT,

    which scales one part of the potential and leaves the rest at full strength. This is replica
    exchange with solute tempering (Wang, Friesner and Berne 2011), generalised to a chosen set of
    energy terms by gREST (Kamiya and Sugita 2018). The swap statistic becomes V = -E_soft / kT and
    the barrier density is sd(V) / sqrt(pi), so the stiff terms drop out of the ladder.

    For a peptide the stiff part is the bonds and the angles. They dominate the spread of the
    energy and carry none of the conformational barriers, so tempering the whole potential would
    spend levels on heating bond vibrations.

    WARNING: this reference has no exact sampler, so beta[0] must be > 0. The soft terms are scaled
    down and never switched off: without them a chain has no excluded volume and can pass through
    itself.
    """

    def __init__(self, log_prob, score):
        """Builds the reference from a sub-potential's log-density and its gradient

        Args:
            * log_prob (function): x (n, dim) -> (n,), the sub-potential's log-density
            * score (function): x (n, dim) -> (n, dim), its gradient
        """
        self.log_prob = log_prob
        self.score = score

    def sample(self, n, generator=None):
        """Not available: a sub-potential has no exact sampler, so run the path from beta > 0"""
        raise NotImplementedError('PotentialReference.sample is not implemented.')


def linear_schedule(n_rungs, beta_min=0.0, dtype=torch.float64, device=None):
    """Returns n_rungs inverse temperatures rising from beta_min to 1

    Uniform is what Algorithm 4 starts from, and the round-based adaptation replaces it within a
    round or two.

    Args:
        * n_rungs (int): Number of levels, including both ends
        * beta_min (float): Coldest end of the path; 0 anneals from the reference (default is 0.0)
        * dtype (torch.dtype): Working dtype (default is torch.float64)
        * device (torch.device or None): Device (default is None)

    Returns:
        * beta (torch.Tensor of shape (n_rungs,)): Increasing schedule, beta[-1] = 1
    """
    return torch.linspace(beta_min, 1.0, n_rungs, dtype=dtype, device=device)


def update_schedule(beta, cumulative):
    """Places the levels at equal increments of the cumulative communication barrier

    This is Algorithm 2 of Syed et al. 2022. The schedule that maximises the round-trip rate is
    the one whose adjacent swap rejection rates are all equal, and since the sum of the rejection
    rates is the global barrier whatever the schedule (their Corollary 2), that is the same as
    asking for Lambda(beta_k) = (k / N) Lambda. So the new schedule is the old cumulative barrier
    inverted at N + 1 equally spaced heights.

    NOTE: the paper interpolates the cumulative barrier with a Fritsch-Carlson monotone cubic;
    this uses a piecewise-linear interpolant, which is monotone by construction, inverts in closed
    form, and accepts the flat segment of a level pair that never rejects. The schedule is
    re-estimated every round with twice the data, so the smoothing of a cubic does not limit the
    result.

    Args:
        * beta (torch.Tensor of shape (n_rungs,)): Current increasing schedule
        * cumulative (torch.Tensor of shape (n_rungs,)): Cumulative barrier at each level,
            starting at 0 and non-decreasing

    Returns:
        * beta (torch.Tensor of the same shape): The updated schedule, with the ends unmoved
    """
    total = float(cumulative[-1])

    # A barrier of zero means every swap is accepted and no schedule is better than another
    if total < 1e-12:
        return beta.clone()

    n_rungs = len(beta)
    heights = torch.linspace(0.0, 1.0, n_rungs, dtype=beta.dtype, device=beta.device) * total

    # Invert the piecewise-linear cumulative barrier: find the segment each height lands in
    index = torch.searchsorted(cumulative.contiguous(), heights.contiguous())
    index = index.clamp(1, n_rungs - 1)
    lower, upper = index - 1, index
    span = (cumulative[upper] - cumulative[lower]).clamp_min(1e-12)
    weight = ((heights - cumulative[lower]) / span).clamp(0.0, 1.0)

    updated = beta[lower] + weight * (beta[upper] - beta[lower])
    updated[0], updated[-1] = beta[0], beta[-1]
    return updated


def hmc_explorer(n_steps=1, n_leapfrog=10):
    """Returns an HMC local kernel for non_reversible_parallel_tempering

    Args:
        * n_steps (int): HMC moves per scan (default is 1)
        * n_leapfrog (int): Leapfrog steps per move (default is 10)

    Returns:
        * explore (function): (log_prob, score, x, step_size, generator) -> (x, acceptance)
    """
    def explore(log_prob, score, x, step_size, generator):
        return hmc(log_prob, score, x, n_steps, step_size, n_leapfrog=n_leapfrog,
                   generator=generator, return_acceptance=True, per_chain=True)
    return explore


def dynamics_explorer(dynamics, n_steps=100, persist_velocities=True):
    """Returns an unadjusted molecular-dynamics local kernel, which makes this replica-exchange MD

    This is the local kernel for a molecule. hmc() uses unit masses on Cartesian coordinates, so
    its step size is set by the fastest hydrogen bond vibration, while the BAOAB integrator behind
    dynamics uses the real masses and a 1 fs step.

    The velocities are carried from scan to scan. Redrawing them every scan would be an Andersen
    thermostat with collision rate 1 / (n_steps * step_size), 10/ps at 100 fs per scan against
    the peptides' own friction of 0.3/ps. Carrying them is exact here because the tempering scales
    the potential and leaves the thermostat alone: every level is Maxwell at the same kT, so a swap
    needs no velocity rescaling, and the velocities stay with the machine like its configuration.

    WARNING: the move is unadjusted, so the timestep is a bias and not a rejection rate. Its
    acceptance is reported as 1, so pass adaptation_gain=0.0 to freeze the step size; otherwise
    the adaptation reads the acceptance as room to grow and raises the timestep until the
    integrator is wrong.

    Args:
        * dynamics (function): (x, n_steps, score, generator, velocities, return_velocities) -> x,
            normally functools.partial(run_batched_dynamics, target, step_size=0.001) with
            run_batched_dynamics from gen/md/dynamics.py
        * n_steps (int): Integration steps per scan (default is 100)
        * persist_velocities (bool): Whether to carry the velocities across scans rather than
            redraw them (default is True)

    Returns:
        * explore (function): (log_prob, score, x, step_size, generator) -> (x, acceptance)
    """
    state = {}

    def explore(log_prob, score, x, step_size, generator):
        velocities = state.get('velocities')
        if velocities is not None and velocities.shape != x.shape:
            velocities = None
        moved, velocities = dynamics(x, n_steps=n_steps, score=score, generator=generator,
                                     velocities=velocities, return_velocities=True)
        if persist_velocities:
            state['velocities'] = velocities
        return moved, torch.ones(len(x), dtype=x.dtype, device=x.device)
    return explore


def calibrate_step_size(explore, annealed_log_prob, annealed_score, x, step_size,
                        target_acceptance=0.65, generator=None, n_iterations=16):
    """Brackets a local-kernel step size per level by geometric search, before the rounds begin

    An acceptance rate gives the sign of a step size's error but not its size, because it
    saturates: 0.99 and 0.999 both mean "smaller than the scale of the target", and the damped
    between-round update cannot travel far enough to recover from a start that is orders of
    magnitude off. The search multiplies the step size by 2 while the sign of the error is
    unchanged, so it reaches 2^n_iterations, and takes a square root of the factor at every sign
    flip, so it then bisects in log space.

    WARNING: an unadjusted kernel reports an acceptance of 1 by construction, so calibrating one
    would raise its step size without bound. The caller gates this on adaptation_gain > 0, the
    same flag that turns the between-round adaptation off.

    Args:
        * explore (function): The local kernel, as the *_explorer factories return it
        * annealed_log_prob (function): Path log-density at the current per-row inverse temperatures
        * annealed_score (function): Its gradient
        * x (torch.Tensor of shape (n_rungs, n_chains, dim)): Current states
        * step_size (torch.Tensor of shape (n_rungs,)): Starting step size per level
        * target_acceptance (float): Acceptance to bracket (default is 0.65)
        * generator (torch.Generator or None): RNG (default is None)
        * n_iterations (int): Search iterations (default is 16)

    Returns:
        * step_size (torch.Tensor of shape (n_rungs,)): Calibrated step size per level
    """
    n_rungs, n_chains, dim = x.shape
    step_size = step_size.clone()
    factor = torch.full_like(step_size, 2.0)
    previous = None

    for _ in range(n_iterations):
        _, accepted = explore(annealed_log_prob, annealed_score, x.reshape(-1, dim),
                              step_size.repeat_interleave(n_chains)[:, None], generator)
        acceptance = accepted.reshape(n_rungs, n_chains).mean(-1)
        direction = torch.where(acceptance > target_acceptance, 1.0, -1.0)
        if previous is not None:
            factor = torch.where(direction != previous, factor.sqrt(), factor)
        step_size = step_size * factor ** direction
        previous = direction

    return step_size


def non_reversible_parallel_tempering(log_prob, score, x0, beta, explorer=None, n_rounds=10,
                                      reference=None, step_size=0.1, target_acceptance=0.65,
                                      adaptation_gain=1.0, adapt_schedule=True, generator=None,
                                      thin=1, n_scans=None, n_burn=0, return_diagnostics=False):
    """Runs non-reversible parallel tempering, tuning its own schedule between rounds

    Syed, Bouchard-Cote, Deligiannidis and Doucet 2022 (JRSS-B), Algorithms 1-4, which is what
    Pigeons.jl implements. It differs from reversible parallel tempering with a fixed schedule in
    three ways:

    1. The swap schedule is deterministic, alternating even and odd adjacent pairs (DEO). A
       configuration travels the ladder ballistically instead of diffusing, and the round-trip
       rate tends to 1 / (2 + 2 Lambda) as the ladder is refined, where a reversible scheme's
       tends to zero.
    2. The run is organised in rounds of doubling length, and the schedule is re-tuned between
       them from the measured communication barrier. The last round is half the total work, so
       adapting costs a factor of two, and the samples are collected only in that last round,
       where nothing adapts, so the output is exactly invariant.
    3. Swaps exchange the annealing parameters and leave the configurations in place. The path is
       linear in beta, so the acceptance depends on a configuration only through the scalar
       V = log_gamma - log_reference. A swap is a permutation of a (n_rungs, n_chains) index
       array, and the round-trip flags live on the machine that carries the configuration.

    The path is the linear one, log gamma_beta = (1 - beta) log reference + beta log_gamma, which
    for reference=None is beta * log_gamma, pure tempering, with beta running from beta[0] > 0.

    WARNING: beta must be increasing, beta[-1] = 1 being the target. This is the paper's
    convention.

    WARNING: a frozen run (adaptation_gain=0.0) has to start from equilibrated states, such as the
    'rung_state' of an earlier run. A step size frozen at its converged value is rejected on a
    high-energy start, so nothing relaxes and no burn-in recovers.

    NOTE: with reference=None the beta[0] level is a hot copy of the target that has no exact
    sampler, so a round trip returns a configuration that is decorrelated but not independent, and
    the stepping-stone number is log(Z(1) / Z(beta[0])) in place of log Z.

    Args:
        * log_prob (function): x (n, dim) -> (n,), the unnormalised target log-density
        * score (function): x (n, dim) -> (n, dim), its gradient
        * x0 (torch.Tensor of shape (n_rungs, n_chains, dim)): Starting states, one per machine
        * beta (torch.Tensor of shape (n_rungs,)): Initial increasing schedule
        * explorer (function or None): Local kernel from one of the *_explorer factories; None
            uses hmc_explorer() (default is None)
        * n_rounds (int): Number of rounds; round r runs 2^r scans, so the total is
            2^(n_rounds + 1) - 2 and the last round is half of it (default is 10)
        * reference (GaussianReference or None): Distribution at beta = 0; None tempers the target
            itself (default is None)
        * step_size (float or torch.Tensor of shape (n_rungs,)): Initial local-kernel step size
            per level (default is 0.1)
        * target_acceptance (float): Local-kernel acceptance the step sizes are tuned to
            (default is 0.65)
        * adaptation_gain (float): Strength of the between-round step-size update, damped by
            1 / sqrt(round) so late rounds settle, and scaled so that a fully saturated acceptance
            doubles the step size and a fully rejected one halves it; 0.0 freezes the step sizes
            (default is 1.0)
        * adapt_schedule (bool): Whether to re-place the levels between rounds; False keeps the
            given schedule and adapts the step sizes only (default is True)
        * generator (torch.Generator or None): RNG (default is None)
        * thin (int): Keep one state every thin scans of the last round (default is 1)
        * n_scans (int or None): Length of the final round, overriding the 2^n_rounds the doubling
            would give; None keeps the doubling (default is None). Nothing adapts in the final
            round, so a converged schedule and step size passed back in with n_rounds=1 give a
            frozen chain of any length and skip the discarded rounds, which double the cost of a
            doubling run
        * n_burn (int): Scans discarded from the front of the final round (default is 0). A
            frozen run needs a few autocorrelation times of it, since it has no earlier rounds to
            serve as burn-in and would otherwise measure its barrier on the descent into the
            ensemble
        * return_diagnostics (bool): Whether to also return the diagnostics dict (default is False)

    Returns:
        * x (torch.Tensor of shape (2^n_rounds // thin, n_chains, dim)): States of the beta = 1
            level, collected in the final round only
        if return_diagnostics:
            * diagnostics (dict): 'beta' the tuned schedule; 'barrier' the global communication
                barrier Lambda; 'rejection' and 'swap_acceptance' per adjacent pair;
                'explorer_acceptance' and 'step_size' per level; 'explorer_acceptance_per_chain'
                of shape (n_rungs, n_chains), which catches a single frozen chain that the
                per-level mean hides (a frozen chain accepts every swap, so its round-trip count
                looks ideal); 'value_mean' and 'value_std', the caloric curve of the path per
                level; 'barrier_density', the rejection per unit beta, which locates the
                bottleneck; 'round_trips' and 'restarts' per chain; 'round_trip_rate' against
                'predicted_round_trip_rate' = 1 / (2 + 2 Lambda); 'recommended_n_rungs' =
                2 Lambda; 'scans_per_round_trip', the observed cost of one machine's round trip,
                against 'ideal_scans_per_round_trip' = n_rungs (2 + 2 Lambda) and
                'ballistic_scans_per_round_trip' = 2 (n_rungs - 1), which together tell a badly
                configured run from one that is too short; 'log_normaliser' from the stepping
                stone; 'state' the final configurations in machine order; 'rung_state' the same
                configurations in level order, of shape (n_rungs, n_chains, dim), which is the x0
                that continues a frozen run; and 'rounds' the per-round history
    """
    n_rungs, n_chains, dim = x0.shape
    device, dtype = x0.device, x0.dtype
    explorer = hmc_explorer() if explorer is None else explorer
    beta = beta.to(dtype=dtype, device=device).clone()
    if not torch.is_tensor(step_size):
        step_size = torch.full((n_rungs,), float(step_size), dtype=dtype, device=device)
    step_size = step_size.to(dtype=dtype, device=device).clone()

    x = x0.clone()
    ladder = torch.arange(n_rungs, device=device)[:, None].expand(n_rungs, n_chains)

    # Two views of one permutation: which level each machine holds, and which machine holds each
    # level. The configurations are indexed by machine and never move; a swap edits these instead
    rung = ladder.clone()
    machine = ladder.clone()
    chains = torch.arange(n_chains, device=device)
    exact_reference = reference is not None and float(beta[0]) == 0.0

    # Round-trip bookkeeping, per machine, so it travels with the configuration it describes
    phase = torch.zeros(n_rungs, n_chains, dtype=torch.uint8, device=device)
    ones = torch.ones_like(phase)
    twos = 2 * ones

    def annealed(beta_row):
        """Returns the log-density and score of the path at these per-row inverse temperatures"""
        if reference is None:
            return (lambda z: beta_row[:, 0] * log_prob(z),
                    lambda z: beta_row * score(z))

        # Mask the tilt at beta = 0, where a target with bounded support would give 0 * (-inf),
        # a NaN that freezes that level for the whole run
        def annealed_log_prob(z):
            base = reference.log_prob(z)
            tilt = beta_row[:, 0] * (log_prob(z) - base)
            return base + torch.where(beta_row[:, 0] > 0, tilt, torch.zeros_like(base))

        def annealed_score(z):
            base = reference.score(z)
            tilt = beta_row * (score(z) - base)
            return base + torch.where(beta_row > 0, tilt, torch.zeros_like(base))

        return annealed_log_prob, annealed_score

    # Bracket the step sizes before round 1, so the damped between-round updates start from a good
    # guess
    if adaptation_gain > 0:
        calibration_log_prob, calibration_score = annealed(beta[rung].reshape(-1, 1))
        step_size = calibrate_step_size(explorer, calibration_log_prob, calibration_score, x,
                                        step_size, target_acceptance, generator)

    history = []
    kept = []
    for round_index in range(1, n_rounds + 1):
        collecting = round_index == n_rounds
        n_round_scans = n_scans if (collecting and n_scans is not None) else 2 ** round_index

        # Burn-in belongs to the collecting round alone, on top of its n_scans: every earlier round
        # is already discarded
        burn = n_burn if collecting else 0
        n_round_scans = n_round_scans + burn

        rejection = torch.zeros(n_rungs - 1, dtype=dtype, device=device)
        round_trips = torch.zeros(n_chains, dtype=dtype, device=device)
        restarts = torch.zeros(n_chains, dtype=dtype, device=device)
        acceptance_sum = torch.zeros(n_rungs, n_chains, dtype=dtype, device=device)
        acceptance_count = torch.zeros(n_rungs, n_chains, dtype=dtype, device=device)
        stepping = torch.full((n_rungs - 1,), -float('inf'), dtype=dtype, device=device)
        value_sum = torch.zeros(n_rungs, dtype=dtype, device=device)
        value_square_sum = torch.zeros(n_rungs, dtype=dtype, device=device)

        for scan in range(n_round_scans):
            # Reset the statistics at the end of the burn-in
            if scan == burn and burn > 0:
                rejection.zero_()
                round_trips.zero_()
                restarts.zero_()
                acceptance_sum.zero_()
                acceptance_count.zero_()
                value_sum.zero_()
                value_square_sum.zero_()
                stepping.fill_(-float('inf'))
                phase.zero_()

            # 1. Local exploration, every machine at the temperature and step size of its level
            beta_row = beta[rung].reshape(-1, 1)
            step_row = step_size[rung].reshape(-1, 1)
            annealed_log_prob, annealed_score = annealed(beta_row)
            moved, accepted = explorer(annealed_log_prob, annealed_score, x.reshape(-1, dim),
                                       step_row, generator)
            x = moved.reshape(n_rungs, n_chains, dim)

            # At beta = 0 the local kernel is an exact draw, K(0)(x, .) = pi_0, so a round trip
            # delivers an independent state, and a configuration outside the target's support is
            # replaced once it reaches the reference
            if exact_reference:
                x[machine[0], chains] = reference.sample(n_chains, generator=generator).to(x)

            # Attribute the acceptance to the level explored, since the step sizes are per level
            # and a machine changes level as the round goes on
            acceptance_sum.scatter_add_(0, rung, accepted.reshape(n_rungs, n_chains))
            acceptance_count.scatter_add_(0, rung, torch.ones_like(acceptance_sum))

            # 2. The swap statistic: one scalar per machine, whatever the dimension
            value = log_prob(x.reshape(-1, dim))
            if reference is not None:
                value = value - reference.log_prob(x.reshape(-1, dim))
            value = torch.gather(value.reshape(n_rungs, n_chains), 0, machine)

            # V per level, the caloric curve of the path, whose variance gives the barrier density
            # lambda(beta) = sqrt(Var_beta(V) / pi)
            value_sum += value.mean(-1)
            value_square_sum += value.pow(2).mean(-1)

            gap = (beta[1:] - beta[:-1])[:, None]
            log_alpha = torch.nan_to_num(gap * (value[:-1] - value[1:]), nan=-float('inf'))
            log_alpha = log_alpha.clamp(max=0.0)

            # Accumulate the rejection on every adjacent pair, including those this scan does not
            # propose, since the density is already evaluated
            rejection += (1.0 - log_alpha.exp()).mean(-1)

            # The stepping stone, accumulated in log space so a wide path cannot overflow
            stepping = torch.logaddexp(stepping, torch.logsumexp(gap * value[:-1], dim=-1))

            # 3. The DEO alternation: even adjacent pairs on even scans, odd pairs on odd ones
            lower = torch.arange(scan % 2, n_rungs - 1, 2, device=device)
            upper = lower + 1
            uniform = torch.rand(len(lower), n_chains, generator=generator, dtype=dtype,
                                 device=device)
            swapped = torch.log(uniform) < log_alpha[lower]

            held = machine[lower]
            machine[lower] = torch.where(swapped, machine[upper], held)
            machine[upper] = torch.where(swapped, held, machine[upper])
            rung.scatter_(0, machine, ladder)

            # 4. Round trips, as a three-state walk per machine: reached the reference, then the
            # target, then the reference again
            at_reference, at_target = rung == 0, rung == n_rungs - 1
            phase = torch.where(at_reference & (phase == 0), ones, phase)
            restart = at_target & (phase == 1)
            restarts += restart.to(dtype).sum(0)
            phase = torch.where(restart, twos, phase)
            complete = at_reference & (phase == 2)
            round_trips += complete.to(dtype).sum(0)
            phase = torch.where(complete, ones, phase)

            # 5. Collect, in the final round only, from whichever machine holds beta = 1
            if collecting and scan >= burn and (scan - burn + 1) % thin == 0:
                kept.append(x[machine[n_rungs - 1], torch.arange(n_chains, device=device)].clone())

        # Between rounds: re-tune the schedule from the barrier and the step sizes from acceptance
        counted = n_round_scans - burn
        rejection = rejection / counted
        cumulative = torch.cat([torch.zeros(1, dtype=dtype, device=device), rejection.cumsum(0)])
        barrier = float(cumulative[-1])
        acceptance_by_chain = acceptance_sum / acceptance_count.clamp_min(1.0)
        explorer_acceptance = acceptance_by_chain.mean(-1)
        value_mean = value_sum / counted
        value_std = (value_square_sum / counted - value_mean.pow(2)).clamp_min(0.0).sqrt()
        history.append({'round': round_index, 'n_scans': counted, 'barrier': barrier,
                        'value_mean': value_mean.clone(), 'value_std': value_std.clone(),
                        'beta': beta.clone(), 'rejection': rejection.clone(),
                        'explorer_acceptance': explorer_acceptance.clone(),
                        'step_size': step_size.clone(), 'round_trips': round_trips.clone(),
                        'restarts': restarts.clone(),
                        'round_trip_rate': float(round_trips.mean()) / counted,
                        'log_normaliser': float((stepping - math.log(counted * n_chains)).sum())})
        if not collecting:
            if adapt_schedule:
                beta = update_schedule(beta, cumulative)

            # Scale the update by the room left on the side it moves, so a saturated acceptance
            # doubles or halves the step size and the update vanishes at the target acceptance
            excess = explorer_acceptance - target_acceptance
            span = torch.where(excess > 0, 1.0 - target_acceptance, target_acceptance)
            step_size = step_size * torch.exp(
                math.log(2.0) * adaptation_gain / math.sqrt(round_index) * excess / span)

    out = torch.stack(kept)
    if return_diagnostics:
        diagnostics = {
            'beta': beta,
            'barrier': barrier,
            'rejection': rejection,
            'swap_acceptance': 1.0 - rejection,
            'value_mean': value_mean,
            'value_std': value_std,
            'barrier_density': rejection / (beta[1:] - beta[:-1]).clamp_min(1e-12),
            'explorer_acceptance': explorer_acceptance,
            'explorer_acceptance_per_chain': acceptance_by_chain,
            'step_size': step_size,
            'round_trips': round_trips,
            'restarts': restarts,
            'round_trip_rate': float(round_trips.mean()) / counted,
            'predicted_round_trip_rate': 1.0 / (2.0 + 2.0 * barrier),
            'recommended_n_rungs': max(2, int(math.ceil(2.0 * barrier))),
            'scans_per_round_trip': counted * n_rungs
            / max(float(round_trips.mean()), 1e-9),
            'ballistic_scans_per_round_trip': 2 * (n_rungs - 1),
            'ideal_scans_per_round_trip': n_rungs * (2.0 + 2.0 * barrier),
            'log_normaliser': history[-1]['log_normaliser'],
            'state': x.clone(),
            'rung_state': x[machine, chains].clone(),
            'rounds': history,
        }
        return out, diagnostics
    return out
