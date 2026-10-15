# The chain diagnostics the generators gate on must read the closed-form autocorrelation times

# Libraries
import torch
from samplingdb.targets.base import generator
import diagnostics


def test_the_autocorrelation_time_matches_an_ar1_process():
    """AR(1) has tau = (1 + phi) / (1 - phi) in closed form, which pins the estimator"""
    gen = generator(0)
    white = torch.randn(64, 20000, generator=gen, dtype=torch.float64)
    assert abs(diagnostics.integrated_autocorrelation_time(white) - 1.0) < 0.05

    for phi in [0.5, 0.8]:
        noise = torch.randn(64, 20000, generator=gen, dtype=torch.float64)
        x = torch.zeros(64, 20000, dtype=torch.float64)
        for step in range(1, 20000):
            x[:, step] = phi * x[:, step - 1] + noise[:, step]
        exact = (1 + phi) / (1 - phi)
        assert abs(diagnostics.integrated_autocorrelation_time(x[:, 1000:]) - exact) < 0.1 * exact


def test_the_effective_sample_size_reports_the_slowest_observable():
    """A set is worth what its worst coordinate says, not what its average does"""
    gen = generator(0)
    fast = torch.randn(32, 4000, generator=gen, dtype=torch.float64)
    slow = torch.zeros(32, 4000, dtype=torch.float64)
    noise = torch.randn(32, 4000, generator=gen, dtype=torch.float64)
    for step in range(1, 4000):
        slow[:, step] = 0.8 * slow[:, step - 1] + noise[:, step]

    ess = diagnostics.effective_sample_size({'fast': fast, 'slow': slow})
    assert ess['name'] == 'slow'
    assert abs(ess['tau'] - 9.0) < 1.5
    assert abs(ess['n_effective'] - 32 * 4000 / ess['tau']) < 1.0
    assert abs(ess['efficiency'] - 1.0 / ess['tau']) < 1e-12
    assert ess['tau_per_observable']['fast'] < ess['tau_per_observable']['slow']


def test_chain_mean_dispersion_agrees_with_the_autocorrelation_function():
    """A second estimate of tau from the spread of the chain means, with no lag window to choose

    It is a cross-check on the truncation rule, not a longer reach: both estimators are bounded by
    the chain length and both read low on a short series by the same amount.
    """
    gen = generator(0)

    # Independent chains: both read 1
    independent = torch.randn(128, 400, generator=gen, dtype=torch.float64)
    assert abs(diagnostics.chain_mean_dispersion({'x': independent})['tau'] - 1.0) < 0.35
    assert abs(diagnostics.effective_sample_size({'x': independent})['tau'] - 1.0) < 0.35

    # AR(1) with rho = 0.9, so tau = (1 + rho) / (1 - rho) = 19
    rho = 0.9
    series = torch.zeros(128, 1600, dtype=torch.float64)
    for step in range(1, series.shape[1]):
        noise = torch.randn(128, generator=gen, dtype=torch.float64)
        series[:, step] = rho * series[:, step - 1] + (1 - rho ** 2) ** 0.5 * noise
    series = series[:, 800:]

    dispersion = diagnostics.chain_mean_dispersion({'x': series})['tau']
    autocorrelation_time = diagnostics.effective_sample_size({'x': series})['tau']
    assert 12.0 < dispersion < 26.0
    assert abs(dispersion - autocorrelation_time) < 0.35 * autocorrelation_time

    # One chain stuck at its own mean is loss of independence too, and both must see it
    stuck = torch.randn(64, 400, generator=gen, dtype=torch.float64)
    stuck[0] = 6.0
    assert diagnostics.chain_mean_dispersion({'x': stuck})['tau'] > 10.0
    assert diagnostics.effective_sample_size({'x': stuck})['tau'] > 10.0
