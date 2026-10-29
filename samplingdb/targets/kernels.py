# Classical MCMC kernels the train-split recipes run, vectorized over chains

# Libraries
import math
import torch

# Multiplicative rate of the per-chain step-size adaptation: a step size moves by exp(+-0.05) per
# move, so it crosses two orders of magnitude in a few hundred moves and then hovers
ADAPTATION_RATE = 0.05


def student_t2(shape, generator=None, dtype=torch.float64, device=None):
    """Draws standard Student-t samples with 2 degrees of freedom by inverse CDF

    NOTE: df = 2 has a closed-form quantile, x = (2u - 1) / sqrt(2 u (1 - u)), so the draw
    respects a torch.Generator and stays bit-reproducible (torch's Gamma sampler does not). u is
    clamped to [tiny, 1 - eps] of the dtype, since torch.rand can return 0, which the quantile
    maps to -inf.

    Args:
        * shape (tuple): Output shape
        * generator (torch.Generator or None): RNG (default is None)
        * dtype (torch.dtype): Output dtype (default is torch.float64)
        * device (torch.device or None): Output device (default is None)

    Returns:
        * x (torch.Tensor of the given shape): Student-t(2) samples
    """
    u = torch.rand(shape, generator=generator, dtype=dtype, device=device)
    u = u.clamp(torch.finfo(dtype).tiny, 1.0 - torch.finfo(dtype).eps)
    return (2 * u - 1) / torch.sqrt(2 * u * (1 - u))


def run_chains(move, state, n_steps, generator=None, return_trace=False, thin=1,
               return_acceptance=False, per_chain=False, n_adapt=0, target_acceptance=0.5):
    """Drives any of the kernels below, which differ only in what one move does

    Every kernel is a Markov move on a state dict plus the same bookkeeping -- draw, propose,
    accept, adapt, keep the thinned trace, count the acceptances -- so the bookkeeping is written
    once here and each kernel supplies only its move.

    NOTE: during the first n_adapt moves every chain scales its own state['step_size'] by
    exp(ADAPTATION_RATE * (accepted - target_acceptance)), and nothing is traced. The step sizes
    are frozen afterwards, so every kept state comes from a fixed Metropolis kernel; the adaptation
    is burn-in, not part of the chain.

    Args:
        * move (function): (state, generator) -> (state, accepted), where state['x'] holds the
            positions, state['step_size'] the step size and accepted is a per-chain bool tensor or
            None for an unadjusted kernel
        * state (dict): Initial state, whose 'x' entry is (n_chains, dim)
        * n_steps (int): Number of moves per chain, the n_adapt ones included
        * generator (torch.Generator or None): RNG (default is None)
        * return_trace (bool): Whether to return every kept state instead of the last one
            (default is False)
        * thin (int): Keep one state every thin steps after the adaptation when return_trace
            (default is 1)
        * return_acceptance (bool): Whether to also return the mean acceptance rate
            (default is False)
        * per_chain (bool): Whether the acceptance is returned per chain rather than averaged
            over them, which is what an adaptation with one step size per chain needs
            (default is False)
        * n_adapt (int): Number of leading moves that adapt the step size of every chain and are
            not traced; state['step_size'] must then be of shape (n_chains, 1) (default is 0)
        * target_acceptance (float): Acceptance rate the adaptation aims at (default is 0.5)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim), or ((n_steps - n_adapt) // thin, n_chains,
            dim) if return_trace): The final states, or the whole thinned trace
        if return_acceptance:
            * acceptance (torch.Tensor of shape (), or (n_chains,) if per_chain): Acceptance rate
    """
    trace = []
    accepted = 0.0
    for step in range(n_steps):
        state, accept = move(state, generator)
        if accept is not None:
            rate = accept.to(state['x'].dtype)
            accepted += rate if per_chain else rate.mean()
        if step < n_adapt:
            state['step_size'] = state['step_size'] * torch.exp(
                ADAPTATION_RATE * (rate[:, None] - target_acceptance))
        elif return_trace and (step + 1 - n_adapt) % thin == 0:
            trace.append(state['x'].clone())

    out = torch.stack(trace) if return_trace else state['x']
    if return_acceptance:
        return out, accepted / n_steps
    return out


def metropolis(state, log_q, y, log_ratio=0.0, generator=None):
    """Accepts or rejects a proposal per chain, and returns the updated state

    Args:
        * state (dict): Current state, holding 'x' and 'log_p'
        * log_q (torch.Tensor of shape (n_chains,)): log gamma at the proposal
        * y (torch.Tensor of shape (n_chains, dim)): The proposal
        * log_ratio (torch.Tensor of shape (n_chains,) or float): log of the proposal ratio, zero
            for a symmetric proposal (default is 0.0)
        * generator (torch.Generator or None): RNG (default is None)

    Returns:
        * accept (torch.Tensor of shape (n_chains,)): Whether each chain moved
        * x (torch.Tensor of shape (n_chains, dim)): The updated positions
        * log_p (torch.Tensor of shape (n_chains,)): log gamma at the updated positions
    """
    x, log_p = state['x'], state['log_p']
    u = torch.rand(len(x), generator=generator, dtype=x.dtype, device=x.device)
    accept = torch.log(u) < (log_q - log_p + log_ratio)
    return accept, torch.where(accept[:, None], y, x), torch.where(accept, log_q, log_p)


def random_walk_mh(log_prob, x0, n_steps, step_size, proposal='normal', generator=None,
                   return_trace=False, thin=1, return_acceptance=False, per_chain=False,
                   n_adapt=0, target_acceptance=0.25):
    """Runs random-walk Metropolis-Hastings, one chain per row of x0

    Args:
        * log_prob (function): x (n_chains, dim) -> (n_chains,), the unnormalised log density
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting states
        * n_steps (int): Number of proposals per chain
        * step_size (float or torch.Tensor of shape (n_chains, 1)): Proposal scale
        * proposal (str): 'normal' or 'student_t2' (default is 'normal')
        * generator (torch.Generator or None): RNG (default is None)
        * return_trace (bool): Whether to return every kept state (default is False)
        * thin (int): Keep one state every thin steps when return_trace (default is 1)
        * return_acceptance (bool): Whether to also return the mean acceptance rate
            (default is False)
        * per_chain (bool): Whether that acceptance is per chain rather than averaged over them
            (default is False)
        * n_adapt (int): Number of leading moves that adapt each chain's step size, see
            run_chains (default is 0)
        * target_acceptance (float): Acceptance rate the adaptation aims at, the optimal
            random-walk scaling's (default is 0.25)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim), or ((n_steps - n_adapt) // thin, n_chains,
            dim) if return_trace): The final states, or the whole thinned trace
        if return_acceptance:
            * acceptance (torch.Tensor of shape ()): Mean acceptance rate
    """
    assert proposal in ['normal', 'student_t2']
    draw = torch.randn if proposal == 'normal' else student_t2

    def move(state, gen):
        x = state['x']
        noise = draw(x.shape, generator=gen, dtype=x.dtype, device=x.device)
        y = x + state['step_size'] * noise
        accept, x, log_p = metropolis(state, log_prob(y), y, generator=gen)
        return {'x': x, 'log_p': log_p, 'step_size': state['step_size']}, accept

    state = {'x': x0.clone(), 'log_p': log_prob(x0),
             'step_size': per_chain_step_size(step_size, x0) if n_adapt > 0 else step_size}
    return run_chains(move, state, n_steps, generator=generator, return_trace=return_trace,
                      thin=thin, return_acceptance=return_acceptance, per_chain=per_chain,
                      n_adapt=n_adapt, target_acceptance=target_acceptance)


def mala(log_prob, score, x0, n_steps, step_size, generator=None, return_trace=False,
         thin=1, return_acceptance=False, per_chain=False, n_adapt=0, target_acceptance=0.57):
    """Runs the Metropolis-adjusted Langevin algorithm, one chain per row of x0

    Args:
        * log_prob (function): x (n_chains, dim) -> (n_chains,)
        * score (function): x (n_chains, dim) -> (n_chains, dim)
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting states
        * n_steps (int): Number of steps per chain
        * step_size (float or torch.Tensor of shape (n_chains, 1)): Step size h of the proposal
            x + h score(x) + sqrt(2 h) noise
        * generator (torch.Generator or None): RNG (default is None)
        * return_trace (bool): Whether to return every kept state (default is False)
        * thin (int): Keep one state every thin steps when return_trace (default is 1)
        * return_acceptance (bool): Whether to also return the mean acceptance rate
            (default is False)
        * per_chain (bool): Whether that acceptance is per chain rather than averaged over them
            (default is False)
        * n_adapt (int): Number of leading moves that adapt each chain's step size, see
            run_chains (default is 0)
        * target_acceptance (float): Acceptance rate the adaptation aims at, MALA's optimal
            scaling's (default is 0.57)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim), or ((n_steps - n_adapt) // thin, n_chains,
            dim) if return_trace): The final states, or the whole thinned trace
        if return_acceptance:
            * acceptance (torch.Tensor of shape ()): Mean acceptance rate
    """
    def move(state, gen):
        x, grad, step_size = state['x'], state['grad'], state['step_size']
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)
        y = x + step_size * grad + (2 * step_size) ** 0.5 * noise
        grad_y = score(y)

        # The Langevin proposal is asymmetric, so its ratio enters the acceptance. The sum keeps
        # its axis so that a per-chain step size divides row by row: dropping it first and then
        # dividing by an (n_chains, 1) step size broadcasts into (n_chains, n_chains) instead
        forward = -((y - x - step_size * grad) ** 2).sum(-1, keepdim=True) / (4 * step_size)
        backward = -((x - y - step_size * grad_y) ** 2).sum(-1, keepdim=True) / (4 * step_size)
        accept, x, log_p = metropolis(state, log_prob(y), y,
                                      log_ratio=(backward - forward).squeeze(-1), generator=gen)
        return {'x': x, 'log_p': log_p, 'grad': torch.where(accept[:, None], grad_y, grad),
                'step_size': step_size}, accept

    state = {'x': x0.clone(), 'log_p': log_prob(x0), 'grad': score(x0),
             'step_size': per_chain_step_size(step_size, x0) if n_adapt > 0 else step_size}
    return run_chains(move, state, n_steps, generator=generator, return_trace=return_trace,
                      thin=thin, return_acceptance=return_acceptance, per_chain=per_chain,
                      n_adapt=n_adapt, target_acceptance=target_acceptance)


def tempered_mala(log_prob, score, x0, factors, n_steps, n_frames, thin, generator=None,
                  step_size=1e-4):
    """Runs one adaptive MALA chain per row of x0 at its own temperature, keeping a few states

    Chain i samples exp(log_prob / factors[i]). Its step size starts at step_size * factors[i]
    and adapts towards 57% acceptance over the first n_steps moves, which are burn-in; it then
    keeps one state every thin moves, n_frames of them, from the frozen kernel.

    Args:
        * log_prob (function): x (n_chains, dim) -> (n_chains,)
        * score (function): x (n_chains, dim) -> (n_chains, dim)
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting states
        * factors (torch.Tensor of shape (n_chains,)): Temperature factor of every chain
        * n_steps (int): Adaptation moves per chain
        * n_frames (int): States kept per chain
        * thin (int): Moves between kept states
        * generator (torch.Generator or None): RNG (default is None)
        * step_size (float): Initial step size at factor 1 (default is 1e-4)

    Returns:
        * trace (torch.Tensor of shape (n_frames, n_chains, dim)): The kept states
    """
    return mala(lambda x: log_prob(x) / factors, lambda x: score(x) / factors[:, None], x0,
                n_steps + n_frames * thin, step_size * factors[:, None], generator=generator,
                return_trace=True, thin=thin, n_adapt=n_steps)


def per_chain_step_size(step_size, x0):
    """Returns step_size as a (n_chains, 1) tensor in x0's dtype, one step per chain to adapt"""
    step_size = torch.as_tensor(step_size, dtype=x0.dtype, device=x0.device)
    return step_size.expand(len(x0), 1).clone() if step_size.dim() < 2 else step_size.clone()


def temperature_ladder(n_chains, temperatures, dtype=torch.float64, device=None):
    """Returns one temperature factor per chain, geometric in the chain index

    Chain i of n runs at low * (high / low) ** (i / (n - 1)), so the ladder is fixed by the chain
    count alone: a chain restarted at the same index runs at the same temperature, and any prefix
    of an interleaved trace holds every temperature.

    Args:
        * n_chains (int): Number of chains
        * temperatures (tuple of two floats): Lowest and highest factor, (low, high)
        * dtype (torch.dtype): Output dtype (default is torch.float64)
        * device (torch.device or None): Output device (default is None)

    Returns:
        * factors (torch.Tensor of shape (n_chains,)): Temperature factors
    """
    low, high = temperatures
    position = torch.linspace(0.0, 1.0, n_chains, dtype=dtype, device=device)
    return low * (high / low) ** position


def langevin(force, x0, n_steps, step_size, mass, kT, friction, generator=None,
             return_trace=False, thin=1, velocities=None, return_velocities=False):
    """Runs underdamped Langevin dynamics with the BAOAB splitting, one trajectory per row of x0

    One step is a half force kick, a half position update, an exact Ornstein-Uhlenbeck velocity
    update, a second half position update and a second half kick, which is what OpenMM's
    LangevinMiddleIntegrator does. A batch of trajectories advances in one set of tensor
    operations, so a sampler on a GPU runs thousands of them for the cost of a few.

    Units are the caller's, as long as force / mass is an acceleration and kT / mass a squared
    velocity: for the peptides, positions in nm, velocities in nm/ps, masses in amu, kT in kJ/mol
    and step_size in ps.

    WARNING: this is dynamics, not a Metropolis chain, so a timestep too large biases the
    distribution rather than showing up as a rejection rate.

    NOTE: every random number is drawn on the generator's device (the CPU when generator is None)
    and moved to x0's, so a CPU generator gives the same draws on every device and a CUDA
    generator keeps the whole run on the GPU.

    Args:
        * force (function): x (n_chains, dim) -> (n_chains, dim), minus the gradient of the
            potential the dynamics samples
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting positions
        * n_steps (int): Number of integration steps
        * step_size (float): Timestep
        * mass (torch.Tensor of shape (dim,)): Mass of every coordinate
        * kT (float or torch.Tensor of shape (n_chains, 1)): Thermostat temperature, in energy
            units, one per trajectory when a tensor
        * friction (float): Friction coefficient, in inverse time units
        * generator (torch.Generator or None): RNG of the velocities and the noise
            (default is None)
        * return_trace (bool): Whether to return one position every thin steps instead of the
            last one (default is False)
        * thin (int): Steps between kept positions when return_trace (default is 1)
        * velocities (torch.Tensor of the shape of x0 or None): Starting velocities; None draws
            them from the Maxwell distribution at kT. Passing a previous call's velocities back in
            makes a sequence of calls one trajectory (default is None)
        * return_velocities (bool): Whether to also return the final velocities
            (default is False)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim), or (n_steps // thin, n_chains, dim) if
            return_trace): The final positions, or the thinned trajectory
        if return_velocities:
            * velocities (torch.Tensor of shape (n_chains, dim)): Final velocities
    """
    thermal = torch.sqrt(kT / mass)
    decay = math.exp(-friction * step_size)
    noise_scale = math.sqrt(1.0 - decay ** 2)
    draws = torch.device('cpu') if generator is None else generator.device

    def draw(shape):
        values = torch.randn(shape, generator=generator, dtype=x0.dtype, device=draws)
        return values.to(x0.device)

    x = x0.clone()
    velocity = thermal * draw(x.shape) if velocities is None else velocities.clone()
    kick = force(x)
    frames = []

    for step in range(n_steps):
        velocity = velocity + 0.5 * step_size * kick / mass
        x = x + 0.5 * step_size * velocity
        velocity = decay * velocity + noise_scale * thermal * draw(x.shape)
        x = x + 0.5 * step_size * velocity
        kick = force(x)
        velocity = velocity + 0.5 * step_size * kick / mass
        if return_trace and (step + 1) % thin == 0:
            frames.append(x.clone())

    out = torch.stack(frames) if return_trace else x
    if return_velocities:
        return out, velocity
    return out


def steepest_descent(force, x0, n_steps, max_step):
    """Moves every row down its force, the largest coordinate moving max_step at each step

    The step is normalised by the largest force component of its row, so a clash moves no faster
    than a relaxed structure, which makes it safe to start dynamics afterwards. With no line search
    the descent ends within max_step of a minimum rather than at it.

    Args:
        * force (function): x (n_chains, dim) -> (n_chains, dim), minus the gradient of the
            potential to descend
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting positions
        * n_steps (int): Number of steps
        * max_step (float): Displacement of the largest force component at each step

    Returns:
        * x (torch.Tensor of shape (n_chains, dim)): Final positions
    """
    x = x0.clone()
    for _ in range(n_steps):
        kick = force(x)
        x = x + max_step * kick / kick.abs().amax(-1, keepdim=True)
    return x
