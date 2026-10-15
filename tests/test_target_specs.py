# Pin the constants and conventions of the synthetic and particle targets

# Libraries
import hashlib, math
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm, t as student_t
from samplingdb import DW4, LJ13, Funnel, GMM40, LennardJones, ManyWell, MoS
from samplingdb.data import load_asset
from samplingdb.targets.base import generator
from samplingdb.targets.particles import MultiDoubleWell


# Half of log(2 pi), the constant a Gaussian proposal density carries
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def test_funnel_uses_scale_not_variance():
    """The dominant coordinate carries scale 3, not variance 9"""
    target = Funnel()
    x = torch.zeros(3, target.dim, dtype=torch.float64)
    x[:, 0] = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64)

    # log_gamma matches scipy's N(0, 3) marginal times the N(0, exp(x_1)) conditional
    marginal = norm.logpdf(x[:, 0].numpy(), 0.0, 3.0)
    conditional = norm.logpdf(x[:, 1:].numpy(), 0.0, np.exp(0.5 * x[:, 0].numpy())[:, None])
    assert np.allclose(target.log_prob(x).numpy(), marginal + conditional.sum(-1))

    # Mis-reading the scale as the variance shifts log_gamma by log(3) at the mode
    shift = Funnel(scale=9.0).log_prob(x[:1]) - target.log_prob(x[:1])
    assert abs(float(shift) + math.log(3.0)) < 1e-12


def test_funnel_exact_samples_have_the_right_moments():
    """log Z = 0 by construction; x_1 has variance 9 and exp(x_1) averages exp(9/2)"""
    target = Funnel()
    x = target.test_samples(n=200000)
    assert torch.isfinite(target.log_prob(x)).all()
    assert abs(float(x[:, 0].var()) - 9.0) < 0.2
    assert abs(float(torch.exp(x[:, 0]).mean()) / math.exp(4.5) - 1.0) < 0.15

    # The analytic CDF the x_1 metrics score against agrees with scipy
    grid = torch.linspace(-10, 10, 41, dtype=torch.float64)
    assert np.allclose(target.x1_cdf(grid).numpy(), norm.cdf(grid.numpy(), 0.0, 3.0))


def test_gmm40_mean_matrix_hash():
    """The mean matrix, not the seed, is the specification"""
    means = load_asset('gmm40_means_dim2.npy')
    assert means.shape == (40, 2)
    digest = hashlib.sha256(means.astype(np.float64).tobytes()).hexdigest()
    assert digest.startswith('3e285d1fe875b184e8b4ad949764ec57')

    # The reproduced first rows, min, max and mean
    assert np.allclose(means[0], [35.813362, 38.286392])
    assert np.allclose(means[1], [-13.416681, -2.506523])
    assert np.allclose(means[2], [5.591097, -26.759758])
    assert np.allclose(means[3], [-15.184431, 15.158443])
    assert abs(means.min() + 37.9774) < 1e-4
    assert abs(means.max() - 38.8283) < 1e-4
    assert abs(means.mean() - 2.5251) < 1e-4


def test_gmm40_components_are_unit_isotropic_gaussians():
    """Each component is N(mu_k, I) with weight 1/40, and log Z = 0"""
    target = GMM40()
    x = target.test_samples(n=4096)
    expected = norm.logpdf(x.numpy()[:, None, :], target.means.numpy()[None], 1.0).sum(-1)
    assert np.allclose(target.component_log_prob(x).numpy(), expected)
    assert torch.allclose(target.weights, torch.full((40,), 1.0 / 40, dtype=torch.float64))
    assert target.log_Z == 0.0

    # Importance sampling from a broad Gaussian of prior_scale recovers Z = 1. The target no
    # longer carries a reference distribution, so the proposal is built here, which is exactly
    # what a sampler does with prior_scale
    scale = target.prior_scale
    y = scale * torch.randn(200000, target.dim, generator=generator(0), dtype=torch.float64)
    log_proposal = (-0.5 * (y / scale) ** 2 - math.log(scale) - HALF_LOG_2PI).sum(-1)
    ratio = torch.exp(target.log_prob(y) - log_proposal)
    assert abs(float(ratio.mean()) - 1.0) < 0.05


def test_gmm40_mode_descriptor():
    """Mode assignment is argmax_k pi_k(x), which is what makes coverage computable"""
    target = GMM40()
    x = target.test_samples(n=4096)
    labels = target.mode_labels(x)
    assert labels.min() >= 0 and labels.max() < 40
    assert float((x - target.means[labels]).norm(dim=-1).max()) < 8.0

    # The exact draw covers every mode with the right weight
    hist = target.compute_mode_weights(x)
    assert hist.shape == (1, target.n_modes)
    assert abs(float(hist.sum()) - 1.0) < 1e-12
    assert float((hist[0] - target.weights).abs().max()) < 0.01


def test_mos_components_are_student_t2():
    """df = 2 means an infinite variance, which is the point of the target"""
    target = MoS()
    assert target.df == 2.0
    x = target.test_samples(n=64)
    expected = student_t.logpdf(x.numpy()[:, None, :], 2.0, target.means.numpy()[None], 1.0).sum(-1)
    assert np.allclose(target.component_log_prob(x).numpy(), expected)

    # The exact sampler's tails are far heavier than a Gaussian's
    assert float(target.test_samples(n=200000).abs().max()) > 100.0


def test_manywell_log_Z():
    """The 1D quadrature and the resulting log Z, checked against their pinned values"""
    a, b, c = -0.5, -6.0, 1.0
    density = lambda x: math.exp(-(a * x + b * x ** 2 + c * x ** 4))
    z1, _ = quad(density, -20, 20, limit=500)
    assert abs(z1 - 11784.50927) < 1e-3

    # log Z_2D and its multiples
    assert abs(ManyWell(dim=2).log_Z_2d - 10.29348) < 1e-5
    assert abs(ManyWell(dim=16).log_Z - 82.348) < 1e-3
    assert abs(ManyWell(dim=32).log_Z - 164.696) < 1e-3
    assert abs(ManyWell(dim=64).log_Z - 329.391) < 1e-3


def test_manywell_closed_form_normaliser_agrees_with_the_quadrature():
    """Two independent computations of Z_x1: parabolic-cylinder series and 1D quadrature"""
    target = ManyWell(dim=2)
    assert abs(target.compute_Z_x1() - 11784.50927) < 1e-2
    assert abs(math.log(target.compute_Z_x1())
               - (target.log_Z_2d - 0.5 * math.log(2 * math.pi))) < 1e-6


def test_manywell_moments():
    """The per-coordinate moments, against the values pinned in DREEM's manywell.py"""
    target = ManyWell(dim=4)
    assert abs(float(target.mean[0]) - 1.18796098) < 1e-6
    assert abs(float(target.variance[0]) - 1.54855479) < 1e-6
    assert float(target.mean[1]) == 0.0 and float(target.variance[1]) == 1.0

    # The exact sampler reproduces them
    x = target.test_samples(n=200000)
    assert abs(float(x[:, 0::2].mean()) - 1.18796098) < 0.01
    assert abs(float(x[:, 0::2].var()) - 1.54855479) < 0.02


def test_manywell_wells_are_asymmetric():
    """The linear term tilts the potential: minima at -1.7108 / +1.7525, mass 15.57% / 84.43%"""
    target = ManyWell(dim=2)
    assert abs(float(target.mode_weights[0]) - 0.1557) < 1e-4
    assert abs(float(target.mode_weights[1]) - 0.8443) < 1e-4

    # The two minima, on a fine grid of the 1D energy
    grid = np.linspace(-3.0, 3.0, 600001)
    energy = -target.log_prob_1d(torch.as_tensor(grid)).numpy()
    assert abs(grid[grid < 0][energy[grid < 0].argmin()] + 1.7108) < 1e-3
    assert abs(grid[grid > 0][energy[grid > 0].argmin()] - 1.7525) < 1e-3
    assert target.shallow_well_bounds == (-1.75, -1.65)
    assert target.deep_well_bounds == (1.7, 1.8)

    # The exact sampler reproduces the mass split
    x = target.test_samples(n=200000)
    assert abs(float((x[:, 0] > 0).double().mean()) - 0.8443) < 5e-3


def test_manywell_per_well_weights_are_not_uniform():
    """Mode coverage is per well against the tilted weights, never a uniform-weight count"""
    target = ManyWell(dim=8)
    x = target.test_samples(n=20000)
    hist = target.compute_mode_weights(x)
    assert hist.shape == (4, 2)
    assert torch.allclose(hist.sum(-1), torch.ones(4, dtype=torch.float64))
    assert torch.allclose(hist, target.mode_weights.expand(4, 2), atol=0.02)

    # A perfectly-covered sample is 1 bit of entropy per well only if the wells were symmetric
    metrics, _ = target.compute_mode_metrics(x)
    assert metrics['emc'] < 0.7
    assert metrics['kl_weights'] < 1e-3
    assert metrics['perc_forgotten_modes'] == 0.0


def test_manywell_log_prob_factorises_on_interleaved_coordinates():
    """log_gamma is the sum of d/2 independent 2D wells, on (even, odd) coordinate pairs"""
    target, well = ManyWell(dim=8), ManyWell(dim=2)
    x = 1.5 * torch.randn(32, 8, dtype=torch.float64)
    wells = sum(well.log_prob(x[:, 2 * i:2 * i + 2]) for i in range(4))
    assert torch.allclose(target.log_prob(x), wells)


def test_dw4_conventions_agree_and_mis_transcription_is_improper():
    """The (quartic, quadratic, constant) and (linear, quadratic, quartic) triples agree"""
    distances = torch.linspace(2.5, 5.5, 7, dtype=torch.float64)
    target = DW4()
    kohler = 0.0 * (distances - 4) - 4 * (distances - 4) ** 2 + 0.9 * (distances - 4) ** 4
    assert torch.allclose(target.pair_energy(distances), kohler)

    # Pair energies verified independently at five distances
    checked = torch.tensor([4.0, 6.0, 10.0, 20.0, 50.0], dtype=torch.float64)
    expected = torch.tensor([0.0, -1.6, 1022.0, 57958.0, 4.02e6], dtype=torch.float64)
    assert torch.allclose(target.pair_energy(checked), expected, rtol=2e-3, atol=1e-9)

    # The mis-transcription loses the quartic term, so the density is not normalisable
    mis = 0.9 * (checked - 4) - 4 * (checked - 4) ** 2
    assert float(mis[-1]) < -8000.0


def test_dw4_energy_is_a_sum_over_unordered_pairs():
    """Two particles at distance d contribute exactly one pair energy, and DW-4 has 6 pairs"""
    pair = MultiDoubleWell(n_particles=2, spatial_dim=2)
    x = torch.tensor([[0.0, 0.0, 3.0, 0.0]], dtype=torch.float64)
    single = pair.pair_energy(torch.tensor([3.0], dtype=torch.float64))
    assert torch.allclose(pair.energy(x), single)
    assert DW4().distances(DW4().initial_configuration()[None]).shape == (1, 6)
    assert DW4().n_pairs == 6


def test_dw4_is_invariant():
    """Translation, rotation and permutation invariance of the energy"""
    target = DW4()
    x = torch.randn(8, target.dim, dtype=torch.float64)

    shifted = (x.reshape(8, 4, 2) + torch.tensor([1.3, -0.7], dtype=torch.float64)).reshape(8, 8)
    assert torch.allclose(target.energy(x), target.energy(shifted))

    angle = torch.tensor(0.4, dtype=torch.float64)
    rotation = torch.stack([
        torch.stack([torch.cos(angle), -torch.sin(angle)]),
        torch.stack([torch.sin(angle), torch.cos(angle)]),
    ])
    rotated = (x.reshape(8, 4, 2) @ rotation.T).reshape(8, 8)
    assert torch.allclose(target.energy(x), target.energy(rotated))

    permuted = x.reshape(8, 4, 2)[:, [2, 0, 3, 1]].reshape(8, 8)
    assert torch.allclose(target.energy(x), target.energy(permuted))


def test_lj_counts_every_pair_twice():
    """Two particles at d = r contribute -2, not -1: the reference implementation never halves

    The published formula writes the energy as eps/(2*tau) over ORDERED pairs, i.e. eps/tau over
    unordered ones. That is not what the released data follows. bgflow's LennardJonesPotential sums
    the full off-diagonal distance matrix with dem's energy_factor left at 1.0, so each pair is
    counted twice, and tests/test_particles_data.py pins that against the data two independent
    ways. This assertion is therefore a deliberate departure from the published formula, not a
    tolerance relaxation -- that formula disagrees with both the reference code and the data.
    """
    x = torch.tensor([[0.0, 0, 0, 1.0, 0, 0]], dtype=torch.float64)
    free = LennardJones(n_particles=2, spatial_dim=3, c_harm=0.0)
    assert free.pair_multiplicity == 2
    assert abs(float(free.pair_energy(torch.ones(1, dtype=torch.float64))) + 1.0) < 1e-9
    assert abs(float(free.energy(x)) + 2.0) < 1e-9

    # The centre-of-mass oscillator adds c_harm * sum_i ||x_i - mean||^2 = 0.5 * 2 * 0.25
    bound = LennardJones(n_particles=2, spatial_dim=3)
    assert bound.c_harm == 0.5
    assert abs(float(bound.energy(x)) - (-2.0 + 0.5 * 0.5)) < 1e-9

    # LJ-13 and LJ-55 carry the same pinned constants
    assert (LJ13().eps, LJ13().tau, LJ13().r, LJ13().c_harm) == (1.0, 1.0, 1.0, 0.5)
    assert (LJ13().name, LJ13().pair_multiplicity) == ('lj13', 2)

    # The pair minimum sits at d = r, where the derivative vanishes
    assert abs(float(free.pair_energy_grad(torch.ones(1, dtype=torch.float64)))) < 1e-12


def test_dw4_counts_every_pair_once():
    """The double well does NOT double count, which is the opposite of the LJ systems"""
    pair = MultiDoubleWell(n_particles=2, spatial_dim=2)
    assert pair.pair_multiplicity == 1
    x = torch.tensor([[0.0, 0.0, 3.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(pair.energy(x),
                          pair.pair_energy(torch.tensor([3.0], dtype=torch.float64)))


def test_particle_targets_are_mean_free_after_preprocess():
    """Metrics compare mean-free configurations, since the energy is translation-invariant"""
    for target in [DW4(), LJ13()]:
        x = torch.randn(4, target.dim, dtype=torch.float64)
        centred = target.preprocess(x).reshape(4, target.n_particles, target.spatial_dim)
        assert torch.allclose(centred.mean(1), torch.zeros_like(centred.mean(1)), atol=1e-12)


def test_energy_histogram_ranges_are_the_reference_implementations():
    """The per-system energy ranges are jarridrb/DEM's, not invented here"""
    from samplingdb import LJ55
    assert DW4().energy_histogram_range == (-26.0, 0.0)
    assert LJ13().energy_histogram_range == (-60.0, 0.0)
    assert LJ55().energy_histogram_range == (-380.0, -180.0)


@pytest.mark.parametrize('dim', [2, 50, 200])
def test_pinned_mean_matrices_exist(dim):
    """Every dim the corpus sweeps over ships a pinned mean matrix"""
    assert load_asset('gmm40_means_dim' + str(dim) + '.npy').shape == (40, dim)
    assert load_asset('mos_means_dim' + str(dim) + '.npy').shape == (10, dim)
