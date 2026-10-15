# Non-reversible parallel tempering, round-based and self-tuning (Syed et al. 2022)

# Libraries
import math
import torch
from .classical import hmc, mala, random_walk_mh


class GaussianReference:
    """The isotropic Gaussian a target's prior_scale describes, as an annealing reference

    A reference is what makes the beta = 0 end of the path tractable: the paper's analysis assumes
    the local kernel there returns INDEPENDENT draws, which is what turns a round trip into a
    genuinely fresh start rather than a long excursion of one chain. Target.prior_scale exists for
    exactly this and nothing else.

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


class HarmonicReference:
    """The Einstein crystal of Frenkel and Ladd 1984: particles tethered to a reference structure

    A zero-centred Gaussian is the wrong base distribution for anything with a hard core, and
    Lennard-Jones is the case in point: almost all of its mass sits on configurations where
    particles overlap, where the energy goes as r^-12, so the path from reference to target has to
    climb out of a region the target gives no weight at all. Measured on LJ-13, that is the
    difference between a global barrier of 11 and one of 2.

    The free-energy literature has used the harmonic (Einstein) crystal as the reference for
    exactly this reason since Frenkel and Ladd: each particle in its own quadratic well around a
    reference position. It is exactly sampleable, its normaliser is analytic, and it already has
    the target's local structure, so only the anharmonic part is left for the path to cover.

    NOTE: tethering to LABELLED positions breaks the permutation and translation symmetries the
    cluster energy has, so a stepping-stone estimate against this reference is a free energy
    relative to a symmetry-broken crystal, not an absolute log Z -- it is short by the log of the
    number of distinguishable relabellings. As an annealing reference that costs nothing, and the
    symmetry is restored as soon as beta leaves 0.

    Args:
        * centres (torch.Tensor of shape (dim,)): The reference configuration, flattened
        * scale (float): Spring width per coordinate, which should sit well below the nearest
            neighbour distance so the reference does not itself overlap particles (default is 0.1)
    """

    def __init__(self, centres, scale=0.1):
        """Builds the reference around a configuration"""
        self.centres = centres.reshape(-1)
        self.dim = len(self.centres)
        self.scale = scale
        self.dtype = centres.dtype
        self.device = centres.device
        self.constant = -self.dim * (math.log(scale) + 0.5 * math.log(2 * math.pi))

    def log_prob(self, x):
        """Evaluates the normalised log-density"""
        return self.constant - 0.5 * ((x - self.centres) / self.scale).pow(2).sum(-1)

    def score(self, x):
        """Evaluates the gradient of the log-density"""
        return -(x - self.centres) / self.scale ** 2

    def sample(self, n, generator=None):
        """Draws n independent points, jittered around the reference configuration"""
        noise = torch.randn(n, self.dim, generator=generator, dtype=self.dtype,
                            device=None if generator is None else generator.device)
        return self.centres + (self.scale * noise).to(self.device)


class PotentialReference:
    """A sub-potential of the target as the beta = 0 end of the path, i.e. solute tempering

    The linear path is log gamma_beta = (1 - beta) log reference + beta log gamma. Setting the
    reference to exp(-E_stiff / kT) makes that

        log gamma_beta = -(E_stiff + beta * E_soft) / kT,

    which scales one PART of the potential and leaves the rest at full strength. That is what
    replica exchange with solute tempering does (Wang, Friesner and Berne 2011) and what gREST
    generalises to a chosen set of energy terms (Kamiya and Sugita 2018), and the reason to want it
    is that the barrier is set by the variance of what is scaled: the swap statistic collapses to
    V = -E_soft / kT, so the stiff terms leave the ladder entirely.

    For a peptide the stiff part is the bonds and the angles, and they dominate the energy's spread
    while contributing none of the conformational barriers: sd(E) comes out within a few percent of
    sqrt((n_bonds + n_angles) / 2), which is what the stiff block alone would produce, and dropping
    it takes sd(V) down by 1.55x to 1.70x across these systems. Since the barrier density is
    sd(V) / sqrt(pi), tempering the whole potential spends a large part of the ladder heating bond
    vibrations -- which is the mechanism behind the square-root-of-system-size scaling of the
    replica count that solute tempering exists to break.

    WARNING: this reference has no exact sampler, so beta[0] must be > 0. That is the regime
    solute tempering is used in anyway: the soft terms are scaled down, not switched off, and
    switching them off would leave a chain with no excluded volume that can pass through itself.
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
    round or two, so there is nothing to gain from a cleverer initial guess.

    Args:
        * n_rungs (int): Number of rungs, including both ends
        * beta_min (float): Coldest end of the path; 0 anneals from the reference (default is 0.0)
        * dtype (torch.dtype): Working dtype (default is torch.float64)
        * device (torch.device or None): Device (default is None)

    Returns:
        * beta (torch.Tensor of shape (n_rungs,)): Increasing schedule, beta[-1] = 1
    """
    return torch.linspace(beta_min, 1.0, n_rungs, dtype=dtype, device=device)


def update_schedule(beta, cumulative):
    """Places the rungs at equal increments of the cumulative communication barrier

    This is Algorithm 2 of Syed et al. 2022. The schedule that maximises the round-trip rate is
    the one whose adjacent swap REJECTION rates are all equal, and since the sum of the rejection
    rates is the global barrier whatever the schedule (their Corollary 2), that is the same as
    asking for Lambda(beta_k) = (k / N) Lambda. So the new schedule is the old cumulative barrier
    inverted at N + 1 equally spaced heights.

    NOTE: the paper interpolates the cumulative barrier with a Fritsch-Carlson monotone cubic;
    this uses a monotone piecewise-linear interpolant instead. It is monotone by construction
    rather than by a spline condition, it inverts in closed form, it needs no strictly-increasing
    precondition (a rung pair that never rejects gives a flat segment, which a cubic fit refuses),
    and the schedule is re-estimated from scratch every round with twice the data, so the
    smoothing a cubic buys is not what limits the result.

    Args:
        * beta (torch.Tensor of shape (n_rungs,)): Current increasing schedule
        * cumulative (torch.Tensor of shape (n_rungs,)): Cumulative barrier at each rung, starting
            at 0 and non-decreasing

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


def mala_explorer(n_steps=1):
    """Returns a MALA local kernel for non_reversible_parallel_tempering"""
    def explore(log_prob, score, x, step_size, generator):
        return mala(log_prob, score, x, n_steps, step_size, generator=generator,
                    return_acceptance=True, per_chain=True)
    return explore


def random_walk_explorer(n_steps=1, proposal='normal'):
    """Returns a random-walk Metropolis local kernel, for a target with no usable score"""
    def explore(log_prob, score, x, step_size, generator):
        return random_walk_mh(log_prob, x, n_steps, step_size, proposal=proposal,
                              generator=generator, return_acceptance=True, per_chain=True)
    return explore


def dynamics_explorer(dynamics, n_steps=100, persist_velocities=True):
    """Returns an unadjusted molecular-dynamics local kernel, which makes this replica-exchange MD

    HMC is the wrong local kernel for a peptide and this was measured, not assumed: on chignolin,
    16 rungs over 310-775 K, the step size adapts down to 1.1e-5 and the cold-rung radius of
    gyration is 1.0270 +- 0.0006 at the first sweep and 1.0272 +- 0.0007 six hundred sweeps later.
    Nothing moves. The reason is that parallel_tempering's HMC uses unit masses on Cartesian
    coordinates, so the step size is set by the fastest hydrogen bond vibration in the molecule.
    A BAOAB integrator uses the real masses and a 1 fs step and does not have that problem.

    The velocities are CARRIED from scan to scan. Drawing them afresh every scan is a full momentum
    refresh, i.e. an Andersen thermostat whose collision rate is 1 / (n_steps * step_size): at the
    shipped 100 fs that is 10/ps against the peptides' own 0.3/ps, so the local kernel was being
    run thirty times over-damped and its structural diffusion suffered accordingly. Carrying them
    is also exactly correct here, and for a reason specific to how the tempering is applied: the
    potential is scaled, not the thermostat, so every rung is Maxwell at the same kT and a swap
    needs no velocity rescaling. The state travels with the machine, and machines never move.

    WARNING: the move is UNADJUSTED, so the timestep is a bias and not a rejection rate. Its
    acceptance is reported as 1 and the step size must therefore be frozen -- pass
    adaptation_gain=0.0, or the adaptation will read the acceptance as room to grow and raise the
    timestep until the integrator is wrong.

    Args:
        * dynamics (function): (x, n_steps, score, generator, velocities, return_velocities) -> x,
            normally functools.partial(target.run_batched_dynamics, step_size=0.001)
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
    """Brackets a local-kernel step size per rung by geometric search, before the rounds begin

    An acceptance rate says whether a step size is too large or too small but not by HOW MUCH,
    because it saturates: 0.99 and 0.999 look alike and both mean "smaller than the scale of the
    target". A Robbins-Monro update can therefore only travel gain * sum(1 / sqrt(round)) in log
    space -- a factor of about twenty over nine rounds even with a doubling normalisation -- and a
    step size that starts two orders of magnitude off never arrives. That is not hypothetical: it
    is why the two metallic glasses this library used to ship finished their runs at acceptance
    0.934 and 0.823 against a 0.65 target.

    A geometric search has no such limit. The factor is 2 while the sign of the error is unchanged,
    so the search travels 2^n_iterations, and it takes a square root at every sign flip, so it then
    bisects in log space and lands within a few percent.

    WARNING: an unadjusted kernel reports an acceptance of 1 by construction, so calibrating one
    would raise its step size without bound. The caller gates this on adaptation_gain > 0, which
    is the same flag that turns the between-round adaptation off.

    Args:
        * explore (function): The local kernel, as the *_explorer factories return it
        * annealed_log_prob (function): Path log-density at the current per-row inverse temperatures
        * annealed_score (function): Its gradient
        * x (torch.Tensor of shape (n_rungs, n_chains, dim)): Current states
        * step_size (torch.Tensor of shape (n_rungs,)): Starting step size per rung
        * target_acceptance (float): Acceptance to bracket (default is 0.65)
        * generator (torch.Generator or None): RNG (default is None)
        * n_iterations (int): Search iterations (default is 16)

    Returns:
        * step_size (torch.Tensor of shape (n_rungs,)): Calibrated step size per rung
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
                                      adaptation_gain=1.0, generator=None, thin=1, n_scans=None,
                                      n_burn=0, return_diagnostics=False):
    """Runs non-reversible parallel tempering, tuning its own schedule between rounds

    Syed, Bouchard-Cote, Deligiannidis and Doucet 2022 (JRSS-B), Algorithms 1-4, which is what
    Pigeons.jl implements. Three things separate it from the fixed-schedule parallel_tempering
    next door, and all three are why it is worth the extra cost:

    1. **The swap schedule is deterministic**, alternating even and odd adjacent pairs (DEO). A
       configuration then travels the ladder ballistically instead of diffusing, and the round
       trip rate tends to 1 / (2 + 2 Lambda) as the ladder is refined rather than to zero. The
       reversible scheme it replaces loses a factor N.
    2. **The run is organised in rounds of doubling length**, and the schedule is re-tuned between
       them from the measured communication barrier. Because the last round is half the total
       work, adapting costs a factor of two and no more -- and the samples are collected ONLY in
       that last round, where nothing is being adapted any more. That is what makes the output
       exactly invariant rather than approximately so.
    3. **Swaps exchange the annealing parameters, not the configurations.** The path is linear in
       beta, so the acceptance depends on a configuration only through the scalar
       V = log_gamma - log_reference. A swap is then a permutation of a (n_rungs, n_chains) index
       array, and the configurations never move: on chignolin that is 414 floats per replica per
       swap not moved, and it is also what makes the round-trip bookkeeping correct by
       construction, since the flags live on the machine that carries the configuration.

    The path is the linear one, log gamma_beta = (1 - beta) log reference + beta log_gamma, which
    for reference=None degenerates to beta * log_gamma -- the pure tempering the rest of the
    library uses, with beta running from beta[0] > 0 rather than from 0.

    WARNING: beta must be INCREASING, beta[-1] = 1 being the target. This is the paper's
    convention and the reverse of geometric_ladder's.

    NOTE: with reference=None the coldest rung is a hot copy of the target rather than something
    that can be sampled exactly, so a round trip returns a configuration that is decorrelated but
    not independent, and the stepping-stone number is log(Z(1) / Z(beta[0])) rather than log Z.

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
            per rung (default is 0.1)
        * target_acceptance (float): Local-kernel acceptance the step sizes are tuned to
            (default is 0.65)
        * adaptation_gain (float): Strength of the between-round step-size update, damped by
            1 / sqrt(round) so late rounds settle rather than oscillate, and scaled so that a
            fully saturated acceptance doubles the step size and a fully rejected one halves it.
            Measured on a Gaussian: an undamped gain of 2 drives rungs past HMC's stability limit
            and back every other round, ending with two rungs at zero acceptance, while this
            converges to a step size agreeing to three digits across every rung (default is 1.0)
        * generator (torch.Generator or None): RNG (default is None)
        * thin (int): Keep one state every thin scans of the last round (default is 1)
        * n_scans (int or None): Length of the FINAL round, overriding the 2^n_rounds the doubling
            would give. None keeps the doubling (default is None).

            This is what lets a converged sampler run for as long as you like. Nothing adapts in
            the final round -- that is what makes its output exactly invariant -- so a schedule and
            step size that have already stabilised can be passed straight back in with n_rounds=1
            and any n_scans, and the result is a frozen chain of that length. Adapting is otherwise
            a permanent 2x tax: the discarded rounds sum to 2^n_rounds - 2 against a collection
            round of 2^n_rounds, so warm-up always costs exactly as much as collection. Calibrate
            once, freeze, and every later run pays 1x.
        * n_burn (int): Scans discarded from the front of the FINAL round (default is 0).

            A frozen run needs this and the doubling structure does not, because there the earlier
            rounds were the burn-in. Without it a frozen run measures its own barrier off the
            descent into the ensemble: on dw4, restarting frozen from a cold Gaussian gave
            Lambda = 5.36 against the 1.77 the same schedule had converged to. It only has to be a
            few autocorrelation times, so at 10 tau it is a few percent of a long collection round
            against the 100% the doubling costs -- which is the whole point of freezing.
        * return_diagnostics (bool): Whether to also return the diagnostics dict (default is False)

    Returns:
        * x (torch.Tensor of shape (2^n_rounds // thin, n_chains, dim)): States of the beta = 1
            rung, collected in the final round only
        if return_diagnostics:
            * diagnostics (dict): 'beta' the tuned schedule, 'barrier' the global communication
                barrier Lambda, 'rejection' and 'swap_acceptance' per adjacent pair,
                'explorer_acceptance' and 'step_size' per rung,
                'explorer_acceptance_per_chain' of shape (n_rungs, n_chains), which is what
                catches ONE frozen chain among hundreds -- the per-rung mean hides it, and a chain
                whose configurations never move keeps the identical states every rung started from,
                so every swap it is offered is accepted and its round-trip count runs to the
                ballistic maximum while it contributes nothing, 'value_mean' and 'value_std'
                the caloric curve of the path per rung, 'barrier_density' the rejection per unit
                beta, which localises the bottleneck a scalar Lambda only sums over,
                'round_trips' and 'restarts' per chain, 'round_trip_rate' observed against
                'predicted_round_trip_rate' 1 / (2 + 2 Lambda), 'recommended_n_rungs' = 2 Lambda,
                'scans_per_round_trip' the observed cost of ONE machine's round trip against
                'ideal_scans_per_round_trip' = n_rungs (2 + 2 Lambda) and the
                'ballistic_scans_per_round_trip' floor of 2 (n_rungs - 1) that no ladder of this
                length can beat -- the three of them together are what say whether a run that
                reports no round trips was badly configured or merely too short,
                'log_normaliser' from the stepping stone, 'state' the final configurations of
                every machine -- which is what a frozen run has to start from, because a step size
                frozen at its converged value is permanently rejected on a high-energy start and
                then nothing relaxes and no burn-in can rescue it: measured on dw4, freezing from a
                cold start gave Lambda 4.42 and explorer acceptance 0.028 where the same schedule
                from an equilibrated state gave 1.65 and 0.644 against a calibrated 1.76 and 0.65 --
                and 'rounds' the per-round history
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

    # Two views of one permutation: which rung each machine holds, and which machine holds each
    # rung. The configurations are indexed by MACHINE and never move; a swap edits these instead
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

        # The where() is not cosmetic: a target with bounded support returns -inf outside it, and
        # at beta = 0 the path is the reference alone, so the product 0 * (-inf) that would
        # otherwise appear is a NaN that silently freezes the coldest rung for the whole run
        def annealed_log_prob(z):
            base = reference.log_prob(z)
            tilt = beta_row[:, 0] * (log_prob(z) - base)
            return base + torch.where(beta_row[:, 0] > 0, tilt, torch.zeros_like(base))

        def annealed_score(z):
            base = reference.score(z)
            tilt = beta_row * (score(z) - base)
            return base + torch.where(beta_row > 0, tilt, torch.zeros_like(base))

        return annealed_log_prob, annealed_score

    # Bracket the step sizes before round 1, so the damped between-round updates refine a good
    # guess rather than crawl towards a bad one from an order of magnitude away
    if adaptation_gain > 0:
        calibration_log_prob, calibration_score = annealed(beta[rung].reshape(-1, 1))
        step_size = calibrate_step_size(explorer, calibration_log_prob, calibration_score, x,
                                        step_size, target_acceptance, generator)

    history = []
    kept = []
    for round_index in range(1, n_rounds + 1):
        collecting = round_index == n_rounds
        n_round_scans = n_scans if (collecting and n_scans is not None) else 2 ** round_index

        # Burn-in belongs to the collecting round alone: every earlier round is already discarded
        burn = min(n_burn, n_round_scans - 1) if collecting else 0
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
            # Everything measured before this point described the approach to the ensemble, not the
            # ensemble, so it is thrown away rather than averaged in
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

            # 1. Local exploration, every machine at the temperature and step size of its rung
            beta_row = beta[rung].reshape(-1, 1)
            step_row = step_size[rung].reshape(-1, 1)
            annealed_log_prob, annealed_score = annealed(beta_row)
            moved, accepted = explorer(annealed_log_prob, annealed_score, x.reshape(-1, dim),
                                       step_row, generator)
            x = moved.reshape(n_rungs, n_chains, dim)

            # The paper's local kernel at beta = 0 is not a Markov move but an exact draw,
            # K(0)(x, .) = pi_0. That is what makes a round trip deliver an INDEPENDENT state
            # rather than a decorrelated one, and it also makes the ladder self-healing: a
            # configuration that starts outside the target's support is rejected all the way down
            # to the reference, where it is simply replaced
            if exact_reference:
                x[machine[0], chains] = reference.sample(n_chains, generator=generator).to(x)

            # Attribute the acceptance to the RUNG explored, not the machine, since the step sizes
            # are per rung and a machine changes rung as the round goes on
            acceptance_sum.scatter_add_(0, rung, accepted.reshape(n_rungs, n_chains))
            acceptance_count.scatter_add_(0, rung, torch.ones_like(acceptance_sum))

            # 2. The swap statistic: one scalar per machine, whatever the dimension
            value = log_prob(x.reshape(-1, dim))
            if reference is not None:
                value = value - reference.log_prob(x.reshape(-1, dim))
            value = torch.gather(value.reshape(n_rungs, n_chains), 0, machine)

            # V per RUNG, which is the caloric curve of the path: its variance is the barrier
            # density, lambda(beta) = sqrt(Var_beta(V) / pi), so a spike in it localises the
            # bottleneck that a total Lambda only sums over
            value_sum += value.mean(-1)
            value_square_sum += value.pow(2).mean(-1)

            gap = (beta[1:] - beta[:-1])[:, None]
            log_alpha = torch.nan_to_num(gap * (value[:-1] - value[1:]), nan=-float('inf'))
            log_alpha = log_alpha.clamp(max=0.0)

            # Rejection is accumulated on EVERY adjacent pair, including the ones this scan does
            # not propose: the density is already evaluated, and the barrier estimate that tunes
            # the schedule is what the whole round is for
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
            beta = update_schedule(beta, cumulative)

            # The update is scaled by the room LEFT on the side it is moving, so a saturated
            # acceptance means a full doubling rather than the exp(0.35) an additive rule gives.
            # Acceptance carries no information about HOW far a step size is from its mark once it
            # saturates -- 0.99 and 0.999 look alike -- so an additive rule can only travel
            # gain * sum(1/sqrt(r)) in log space, about 4x over nine rounds, which is not enough
            # to rescue a step size that started an order of magnitude too small. This travels 2^r
            # while saturated and still vanishes as the acceptance approaches its target, so the
            # damped convergence it replaces is unchanged near the mark
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
            'rounds': history,
        }
        return out, diagnostics
    return out
