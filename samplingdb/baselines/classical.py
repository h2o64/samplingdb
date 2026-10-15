# Classical MCMC kernels, vectorized over chains: the baseline floor and the train-split generator

# Libraries
import math
import torch


def student_t2(shape, generator=None, dtype=torch.float64, device=None):
    """Draws standard Student-t samples with 2 degrees of freedom by inverse CDF

    NOTE: df = 2 has a closed-form quantile, x = (2u - 1) / sqrt(2 u (1 - u)), so the draw
    respects a torch.Generator and stays bit-reproducible (torch's Gamma sampler does not).

    Args:
        * shape (tuple): Output shape
        * generator (torch.Generator or None): RNG (default is None)
        * dtype (torch.dtype): Output dtype (default is torch.float64)
        * device (torch.device or None): Output device (default is None)

    Returns:
        * x (torch.Tensor of the given shape): Student-t(2) samples
    """
    u = torch.rand(shape, generator=generator, dtype=dtype, device=device)
    return (2 * u - 1) / torch.sqrt(2 * u * (1 - u))


def run_chains(move, state, n_steps, generator=None, return_trace=False, thin=1,
               return_acceptance=False, per_chain=False):
    """Drives any of the kernels below, which differ only in what one move does

    Every kernel is a Markov move on a state dict plus the same bookkeeping -- draw, propose,
    accept, keep the thinned trace, count the acceptances -- so the bookkeeping is written once
    here and each kernel supplies only its move.

    Args:
        * move (function): (state, generator) -> (state, accepted), where state['x'] holds the
            positions and accepted is a per-chain bool tensor or None for an unadjusted kernel
        * state (dict): Initial state, whose 'x' entry is (n_chains, dim)
        * n_steps (int): Number of moves per chain
        * generator (torch.Generator or None): RNG (default is None)
        * return_trace (bool): Whether to return every kept state instead of the last one
            (default is False)
        * thin (int): Keep one state every thin steps when return_trace (default is 1)
        * return_acceptance (bool): Whether to also return the mean acceptance rate
            (default is False)
        * per_chain (bool): Whether the acceptance is returned per chain rather than averaged
            over them, which is what an adaptation with one step size per chain needs
            (default is False)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim), or (n_steps // thin, n_chains, dim) if
            return_trace): The final states, or the whole thinned trace
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
        if return_trace and (step + 1) % thin == 0:
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
                   return_trace=False, thin=1, return_acceptance=False, per_chain=False):
    """Runs random-walk Metropolis-Hastings, one chain per row of x0

    Args:
        * log_prob (function): x (n_chains, dim) -> (n_chains,), the unnormalised log density
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting states
        * n_steps (int): Number of proposals per chain
        * step_size (float): Proposal scale
        * proposal (str): 'normal' or 'student_t2' (default is 'normal')
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
    assert proposal in ['normal', 'student_t2']
    draw = torch.randn if proposal == 'normal' else student_t2

    def move(state, gen):
        x = state['x']
        noise = draw(x.shape, generator=gen, dtype=x.dtype, device=x.device)
        y = x + step_size * noise
        accept, x, log_p = metropolis(state, log_prob(y), y, generator=gen)
        return {'x': x, 'log_p': log_p}, accept

    state = {'x': x0.clone(), 'log_p': log_prob(x0)}
    return run_chains(move, state, n_steps, generator=generator, return_trace=return_trace,
                      thin=thin, return_acceptance=return_acceptance, per_chain=per_chain)


def ula(score, x0, n_steps, step_size, generator=None, return_trace=False, thin=1):
    """Runs the unadjusted Langevin algorithm, one chain per row of x0

    Args:
        * score (function): x (n_chains, dim) -> (n_chains, dim)
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting states
        * n_steps (int): Number of steps per chain
        * step_size (float): Step size, in the x = x + eps * grad + sqrt(2 eps) * noise scaling
        * generator (torch.Generator or None): RNG (default is None)
        * return_trace (bool): Whether to return every kept state (default is False)
        * thin (int): Keep one state every thin steps when return_trace (default is 1)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim), or (n_steps // thin, n_chains, dim) if
            return_trace): The final states, or the whole thinned trace
    """
    def move(state, gen):
        x = state['x']
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)
        return {'x': x + step_size * score(x) + (2 * step_size) ** 0.5 * noise}, None

    return run_chains(move, {'x': x0.clone()}, n_steps, generator=generator,
                      return_trace=return_trace, thin=thin)


def mala(log_prob, score, x0, n_steps, step_size, generator=None, return_trace=False,
         thin=1, return_acceptance=False, per_chain=False):
    """Runs the Metropolis-adjusted Langevin algorithm, one chain per row of x0

    Args:
        * log_prob (function): x (n_chains, dim) -> (n_chains,)
        * score (function): x (n_chains, dim) -> (n_chains, dim)
        * x0 (torch.Tensor of shape (n_chains, dim)): Starting states
        * n_steps (int): Number of steps per chain
        * step_size (float): Step size, as in ula
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
        x, grad = state['x'], state['grad']
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
        return {'x': x, 'log_p': log_p, 'grad': torch.where(accept[:, None], grad_y, grad)}, accept

    state = {'x': x0.clone(), 'log_p': log_prob(x0), 'grad': score(x0)}
    return run_chains(move, state, n_steps, generator=generator, return_trace=return_trace,
                      thin=thin, return_acceptance=return_acceptance, per_chain=per_chain)


def leapfrog(score, x, momentum, step_size, n_leapfrog, beta=1.0):
    """Integrates Hamilton's equations for beta * log_gamma

    Half a momentum kick at each end and a full one between, which is the standard leapfrog. beta
    may be a tensor broadcasting against x, which is what lets parallel tempering integrate every
    rung at its own inverse temperature in one call.

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
