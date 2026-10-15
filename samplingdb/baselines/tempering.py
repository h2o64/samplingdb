# Parallel tempering, the physics-literature reference algorithm for a multimodal energy

# Libraries
import math
import torch
from .classical import leapfrog


def geometric_ladder(n_rungs, beta_min, dtype=torch.float64, device=None):
    """Returns n_rungs inverse temperatures from 1 down to beta_min, geometrically spaced

    A geometric ladder keeps the KL between neighbours roughly constant for an energy whose scale
    is set by the temperature, which is why it is the default in the physics literature rather
    than a linear one.

    Args:
        * n_rungs (int): Number of rungs, the first being the target
        * beta_min (float): Inverse temperature of the hottest rung
        * dtype (torch.dtype): Working dtype (default is torch.float64)
        * device (torch.device or None): Device (default is None)

    Returns:
        * beta (torch.Tensor of shape (n_rungs,)): Decreasing inverse temperatures, beta[0] = 1
    """
    exponent = torch.linspace(0.0, 1.0, n_rungs, dtype=dtype, device=device)
    return torch.exp(exponent * math.log(beta_min))


def parallel_tempering(log_prob, score, x0, beta, n_sweeps, step_size, n_leapfrog=10,
                       n_local=1, adapt_sweeps=0, target_acceptance=0.65, generator=None,
                       n_keep=0, thin=1, return_diagnostics=False):
    """Runs parallel tempering with HMC local moves and deterministic even/odd swaps

    One sweep is n_local HMC moves on every rung, then one swap pass. The swap pass alternates
    between the even and the odd adjacent pairs rather than drawing pairs at random, which is the
    non-reversible scheme (Syed et al. 2022, which Cheng et al. 2026 also run on the Kob-Andersen
    glass): a configuration then walks up and down the ladder in a directed way instead of
    diffusing, and the round-trip rate is what limits the run rather than the swap rate alone.

    Tempering is in the DENSITY, log gamma_beta = beta * log_gamma. For every target whose
    log_gamma is -E / T that is exactly temperature tempering, so rung k sits at T / beta_k.

    WARNING: step sizes are adapted for the first adapt_sweeps sweeps and then FROZEN, and only
    states collected after the adaptation are returned. Adapting while collecting would break
    detailed balance and bias the reference set it is meant to produce.

    Args:
        * log_prob (function): x -> log_gamma(x), broadcasting over leading axes
        * score (function): x -> grad log_gamma(x), the analytic score
        * x0 (torch.Tensor of shape (n_rungs, n_chains, dim)): Starting states per rung
        * beta (torch.Tensor of shape (n_rungs,)): Inverse temperatures, beta[0] = 1
        * n_sweeps (int): Sweeps after the adaptation, i.e. the collection budget
        * step_size (float or torch.Tensor of shape (n_rungs,)): Initial HMC step size per rung
        * n_leapfrog (int): Leapfrog steps per HMC move (default is 10)
        * n_local (int): HMC moves per rung per sweep (default is 1)
        * adapt_sweeps (int): Warm-up sweeps during which the step sizes are tuned towards
            target_acceptance and nothing is collected (default is 0)
        * target_acceptance (float): Acceptance the adaptation aims for (default is 0.65)
        * generator (torch.Generator or None): RNG (default is None)
        * n_keep (int): Number of states to keep from the beta = 1 rung; 0 keeps only the final
            state (default is 0)
        * thin (int): Keep one state every thin sweeps (default is 1)
        * return_diagnostics (bool): Whether to also return the acceptance and round-trip
            diagnostics (default is False)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim), or (n_keep, n_chains, dim) if n_keep): The
            cold-rung states
        if return_diagnostics:
            * diagnostics (dict): 'hmc_acceptance' per rung, 'swap_acceptance' per adjacent pair,
                'round_trips' per chain, summed over the n_rungs configurations, and 'step_size'
                per rung
    """
    n_rungs, n_chains, dim = x0.shape
    x = x0.clone()

    # Targets take a flat batch of configurations, so the (rung, chain) axes are folded away at
    # the boundary rather than every density in the catalog having to accept a rank-3 input
    def stacked_log_prob(z):
        return log_prob(z.reshape(-1, dim)).reshape(z.shape[:-1])

    def stacked_score(z):
        return score(z.reshape(-1, dim)).reshape(z.shape)

    beta = beta.reshape(n_rungs, 1, 1).to(x)
    if not torch.is_tensor(step_size):
        step_size = torch.full((n_rungs,), float(step_size), dtype=x.dtype, device=x.device)
    step_size = step_size.reshape(n_rungs, 1, 1).to(x)

    # An index that walks the ladder with each configuration, so round trips are countable
    ladder = torch.arange(n_rungs, device=x.device)[:, None].expand(n_rungs, n_chains)
    rung_of = ladder.clone()
    been_hot = torch.zeros(n_rungs, n_chains, dtype=torch.bool, device=x.device)
    round_trips = torch.zeros(n_chains, dtype=x.dtype, device=x.device)

    hmc_tries = torch.zeros(n_rungs, dtype=x.dtype, device=x.device)
    hmc_accepts = torch.zeros(n_rungs, dtype=x.dtype, device=x.device)
    swap_tries = torch.zeros(n_rungs - 1, dtype=x.dtype, device=x.device)
    swap_accepts = torch.zeros(n_rungs - 1, dtype=x.dtype, device=x.device)
    kept = []

    for sweep in range(adapt_sweeps + n_sweeps):
        adapting = sweep < adapt_sweeps

        # 1. Local moves: n_local HMC moves per rung, at that rung's own step size
        for _ in range(n_local):
            momentum = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
            energy = -beta[..., 0] * stacked_log_prob(x) + 0.5 * (momentum ** 2).sum(-1)
            y, q = leapfrog(stacked_score, x, momentum, step_size, n_leapfrog, beta)
            proposed = -beta[..., 0] * stacked_log_prob(y) + 0.5 * (q ** 2).sum(-1)

            uniform = torch.rand(n_rungs, n_chains, generator=generator, dtype=x.dtype,
                                 device=x.device)
            accept = torch.log(uniform) < energy - proposed
            accept = accept & torch.isfinite(proposed)
            x = torch.where(accept[..., None], y, x)
            hmc_accepts += accept.to(x.dtype).sum(-1)
            hmc_tries += n_chains

        # 2. Adapt the step sizes towards the target acceptance, then freeze them
        if adapting:
            rate = (hmc_accepts / hmc_tries.clamp_min(1.0)).reshape(n_rungs, 1, 1)
            step_size = step_size * torch.exp(0.1 * (rate - target_acceptance))
            hmc_accepts, hmc_tries = torch.zeros_like(hmc_accepts), torch.zeros_like(hmc_tries)

        # 3. Swap pass: even pairs on even sweeps, odd pairs on odd ones
        lower = torch.arange(sweep % 2, n_rungs - 1, 2, device=x.device)
        if len(lower) > 0:
            upper = lower + 1
            log_gamma = stacked_log_prob(x)
            delta = (beta[lower, 0] - beta[upper, 0]) * (log_gamma[lower] - log_gamma[upper])
            uniform = torch.rand(len(lower), n_chains, generator=generator, dtype=x.dtype,
                                 device=x.device)
            accept = torch.log(uniform) < -delta

            # Exchange the CONFIGURATIONS of the accepted pairs, and their ladder indexes with them
            for position, (low, high) in enumerate(zip(lower.tolist(), upper.tolist())):
                mask = accept[position]
                if not bool(mask.any()):
                    continue
                keep_low, keep_index = x[low, mask].clone(), rung_of[low, mask].clone()
                x[low, mask], x[high, mask] = x[high, mask], keep_low
                rung_of[low, mask], rung_of[high, mask] = rung_of[high, mask], keep_index
            swap_accepts[lower] += accept.to(x.dtype).sum(-1)
            swap_tries[lower] += n_chains

        # 4. Count round trips: a configuration that has touched the hot end and comes back.
        # rung_of names the configuration sitting at each RUNG, so it has to be inverted before
        # the flag can travel with the configuration instead of staying behind on the rung it left
        rung_of_configuration = torch.empty_like(rung_of)
        rung_of_configuration.scatter_(0, rung_of, ladder)
        been_hot = been_hot | (rung_of_configuration == n_rungs - 1)
        home = (rung_of_configuration == 0) & been_hot
        round_trips += home.to(x.dtype).sum(0)
        been_hot = been_hot & ~home

        # 5. Collect from the cold rung, after the adaptation only
        if not adapting and n_keep > 0 and (sweep - adapt_sweeps) % thin == 0:
            kept.append(x[0].clone())

    out = torch.stack(kept[-n_keep:]) if n_keep > 0 else x[0]
    if return_diagnostics:
        diagnostics = {
            'hmc_acceptance': hmc_accepts / hmc_tries.clamp_min(1.0),
            'swap_acceptance': swap_accepts / swap_tries.clamp_min(1.0),
            'round_trips': round_trips,
            'step_size': step_size.reshape(n_rungs),
        }
        return out, diagnostics
    return out
