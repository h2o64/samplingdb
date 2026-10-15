# How many independent samples a correlated chain is actually worth

# Libraries
import torch


def autocorrelation(series, max_lag=None):
    """Estimates the autocorrelation function of a batch of chains, averaged over them

    WARNING: the chains must be the axis that is INDEPENDENT and the steps the axis that is
    correlated. Every generator in gen/ writes its output chain-interleaved, so a reference set
    read off disk has to be unflattened into (n_steps, n_chains, dim) before it means anything
    here -- reading the flat file as one series measures nothing, because consecutive rows are
    different chains.

    Args:
        * series (torch.Tensor of shape (n_chains, n_steps)): One scalar observable per chain
        * max_lag (int or None): Largest lag to return; None uses n_steps // 2 (default is None)

    Returns:
        * rho (torch.Tensor of shape (max_lag,)): Autocorrelation, rho[0] = 1
    """
    n_chains, n_steps = series.shape
    if max_lag is None:
        max_lag = n_steps // 2
    centred = series - series.mean()

    # Wiener-Khinchin, zero-padded to the next power of two so the circular wrap is discarded
    size = 1
    while size < 2 * n_steps:
        size *= 2
    spectrum = torch.fft.rfft(centred, n=size, dim=-1)
    correlation = torch.fft.irfft(spectrum * spectrum.conj(), n=size, dim=-1)[:, :n_steps]
    correlation = correlation / (n_steps - torch.arange(n_steps, device=series.device))
    rho = correlation.mean(0)
    return (rho / rho[0])[:max_lag]


def integrated_autocorrelation_time(series, c=5.0):
    """Estimates the integrated autocorrelation time of a batch of chains, in steps

    The sum of an empirical autocorrelation function does not converge -- its tail is noise that
    accumulates -- so it is truncated at the first window m with m >= c * tau(m), which is Sokal's
    standard automatic choice.

    WARNING: tau is only estimable from a series many times longer than itself, and the estimator
    reads HIGH below roughly 50 tau steps. A short chain therefore looks worse than it is, which
    is the opposite of the failure to worry about but still a wrong number.

    Args:
        * series (torch.Tensor of shape (n_chains, n_steps)): One scalar observable per chain
        * c (float): Window factor (default is 5.0)

    Returns:
        * tau (float): Integrated autocorrelation time, at least 1
    """
    rho = autocorrelation(series)
    tau = 2.0 * torch.cumsum(rho, dim=0) - 1.0
    window = torch.arange(len(tau), device=series.device, dtype=tau.dtype)
    reached = window >= c * tau
    m = int(torch.argmax(reached.to(torch.uint8))) if bool(reached.any()) else len(tau) - 1
    return float(max(float(tau[m]), 1.0))


def chain_mean_dispersion(observables):
    """Returns the autocorrelation time implied by the spread of the per-chain means

    For n independent draws per chain, a chain mean has variance Var(state) / n, so

        tau_implied = n * Var(chain mean) / Var(state)

    is 1 for an independent set and grows with the correlation. It is worth having next to
    integrated_autocorrelation_time because it estimates the same quantity a completely different
    way -- from the spread of the chain means rather than from the lag structure -- with no
    truncation rule and no Sokal window to choose. When the two agree, the number is not an
    artefact of that choice. On LJ-55 at 128 chains and 64 stored states each, the chain means have
    sd 1.611 against the 13.557 / 8 = 1.695 independence predicts, i.e. tau_implied 0.90, next to
    an ACF reading of 1.00.

    WARNING: it does NOT see further than the ACF does. Both are bounded by the chain length and
    both read LOW on a short series, by the same amount: on an AR(1) whose true tau is 39, the two
    give 21.8 / 20.6 at 32 steps, 36.5 / 32.8 at 100 and 45.3 / 40.6 at 400. Neither can tell a
    handful of stuck chains from a genuinely long correlation time either -- both report ~73 for one
    frozen chain in 64. What this buys is an independent estimate, not a longer reach.

    WARNING: Var(chain mean) is itself estimated from n_chains numbers, so this has a relative
    error of about sqrt(2 / n_chains) -- 12% at 128 chains and 45% at 10. It is a cross-check, not
    a replacement.

    Args:
        * observables (dict): Name -> torch.Tensor of shape (n_chains, n_steps)

    Returns:
        * dispersion (dict): 'tau' the worst implied autocorrelation time, 'name' the observable
            that attains it, and 'tau_per_observable'
    """
    taus = {}
    for name, series in observables.items():
        n_chains, n_steps = series.shape
        spread = float(series.mean(-1).var(unbiased=True))
        total = float(series.var(unbiased=True))
        taus[name] = max(n_steps * spread / total, 1e-6) if total > 0 else 1.0
    name = max(taus, key=taus.get)
    return {'tau': taus[name], 'name': name, 'tau_per_observable': taus}


def effective_sample_size(observables):
    """Returns what a stored set is worth, over the slowest observable it is scored on

    A reference set is only as good as its slowest coordinate, so this reports the WORST
    integrated autocorrelation time rather than an average: an ensemble that decorrelates in
    energy while its structure does not is the standard way a reference set is over-claimed, and
    the energy is usually the fast one.

    Args:
        * observables (dict): Name -> torch.Tensor of shape (n_chains, n_steps)

    Returns:
        * ess (dict): 'tau' the worst integrated autocorrelation time in stored steps, 'name' the
            observable that attains it, 'n_effective' the number of independent samples the set is
            worth, 'efficiency' that as a fraction of its length, and 'tau_per_observable'
    """
    taus = {name: integrated_autocorrelation_time(series) for name, series in observables.items()}
    name = max(taus, key=taus.get)
    n_chains, n_steps = next(iter(observables.values())).shape
    return {'tau': taus[name], 'name': name, 'n_effective': n_chains * n_steps / taus[name],
            'efficiency': 1.0 / taus[name], 'tau_per_observable': taus}
