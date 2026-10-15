# Pin the constants and conventions of the synthetic and particle targets

# Libraries
import hashlib, math
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm, t as student_t
from samplingdb import DW4, LJ13, Funnel, GMM40, LennardJones, ManyModes, ManyWell, MoS, build
from samplingdb.targets.base import generator, spec_id
from samplingdb.targets.mixtures import uniform_means
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


# The sha256 of every mean matrix the corpus sweeps over, as float32 bytes
MEAN_HASHES = {
    ('gmm40', 2): '5a2a3a155b1b17b71691699cdca4f84d3d7291af9ebdae6e0bf9737aec4921b1',
    ('gmm40', 50): '599bf6b2fb64f53743888e97f81995173fcb6742e24e869965c603bbff089bf8',
    ('gmm40', 200): '43ba1b1e702eee42c3b2d95f8df1e6d1e37198119dfd6ec105274aebc2fdfe6c',
    ('mos', 2): 'a795de2725063b3278215a2a6ab52f6b7cf6605f6b5aaca34350dbeea07e7369',
    ('mos', 50): '9380d149f193fde0dff61ccf5da6f7d678691158e181756e947bdace9908a56a',
    ('mos', 200): 'd56c0472d87e1c836d1f3b654bc52c20de1ebe8ed3531a7b7bd46daae6ac0e92',
}


@pytest.mark.parametrize('name,dim', list(MEAN_HASHES))
def test_the_mean_matrices_are_pinned(name, dim):
    """The mean matrix, not the seed, is the specification: a change in torch's RNG fails here"""
    target = GMM40(dim=dim) if name == 'gmm40' else MoS(dim=dim)
    means = target.means.to(torch.float32)
    assert means.shape == ((40 if name == 'gmm40' else 10), dim)
    assert hashlib.sha256(means.numpy().tobytes()).hexdigest() == MEAN_HASHES[(name, dim)]

    # Drawn in float32, so the working dtype does not move a single location
    single = GMM40(dim=dim, dtype=torch.float32) if name == 'gmm40' \
        else MoS(dim=dim, dtype=torch.float32)
    assert torch.equal(single.means.double(), target.means)


# The spec_id of every Tier 0 target as catalogued, which a build in either dtype must carry. A
# change here is a change of density: it moves every cached split, so it has to be deliberate
SPEC_IDS = {
    'funnel': '556b9f8f', 'manywell': '30beb4d2', 'gmm40': 'a1f669cb', 'mos': '60299a87',
    'twomodes': '0eddf6c9', 'brackettwomodes': '334746c8', 'manymodes-k3': 'edf2fcb2',
    'eightgaussians': 'ae0110e3', 'rings': '5deb01ff', 'checkerboard': '9547623f',
}


@pytest.mark.parametrize('name', list(SPEC_IDS))
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_the_tier_0_spec_ids_are_pinned(name, dtype):
    """Each Tier 0 density hashes to its pinned spec_id, whatever the working dtype"""
    built = build(name, dtype=dtype)
    assert built.truth_tier == 0
    assert spec_id(built.spec()) == SPEC_IDS[name]


def test_gmm40_means_are_uniform_on_the_box():
    """The first rows, min, max and mean of the dim-2 draw, and the one formula all three share"""
    means = GMM40().means
    assert np.allclose(means[0], [-0.299473, 21.457745])
    assert np.allclose(means[1], [-32.921806, -29.437561])
    assert abs(float(means.min()) + 38.2139) < 1e-4
    assert abs(float(means.max()) - 39.7653) < 1e-4
    assert abs(float(means.mean()) + 0.4502) < 1e-4
    assert torch.equal(ManyModes().means, uniform_means(3, 2, -3, 3, 42).double())


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


def closed_form_Z_x1():
    """Computes the 1D quartic normaliser in closed form, as a check on the quadrature

    Uses -x^4 + 6 x^2 = 9 - (x^2 - 3)^2 and folds the odd 1/2 x term into a cosh, so
    Z_x1 = sum_m J_{2m} / (4^m (2m)!) with the even moments
    J_{2m} = exp(9/2) 2^{-(2m+1)/4} Gamma(m + 1/2) D_{-(m+1/2)}(-3 sqrt(2)), where D is the
    parabolic-cylinder function. The series reaches machine precision in a handful of terms.

    NOTE: this closed form holds for the pinned (a, b, c) = (-0.5, -6, 1) only.
    """
    from scipy.special import gamma, pbdv
    z = -3 * math.sqrt(2)
    total = 0.0
    for m in range(12):
        moment = (math.exp(4.5) * 2 ** (-(2 * m + 1) / 4) * gamma(m + 0.5)
                  * pbdv(-(m + 0.5), z)[0])
        total += moment / (4 ** m * math.factorial(2 * m))
    return float(total)


def test_manywell_closed_form_normaliser_agrees_with_the_quadrature():
    """Two independent computations of Z_x1: parabolic-cylinder series and 1D quadrature"""
    target = ManyWell(dim=2)
    assert (target.a, target.b, target.c) == (-0.5, -6.0, 1.0)
    assert abs(closed_form_Z_x1() - 11784.50927) < 1e-2
    assert abs(math.log(closed_form_Z_x1())
               - (target.log_Z_2d - 0.5 * math.log(2 * math.pi))) < 1e-6


def test_manywell_moments():
    """The per-coordinate moments, against values pinned by an independent trapezoid quadrature"""
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


def test_manywell_w1_x1_is_against_the_exact_marginal():
    """A shift of the quartic coordinates by c reads c, and the reference set plays no part"""
    target = ManyWell()
    x = target.sample((20000,), generator=generator(0))
    exact = target.compute_metrics(x, compute_standard_metrics=False)['w1_x1']
    assert exact < 0.005

    shifted = x.clone()
    shifted[:, 0::2] += 0.3
    moved = target.compute_metrics(shifted, compute_standard_metrics=False)['w1_x1']
    assert abs(moved - 0.3) < 0.005

    # Any reference set gives the same number, since none is read
    other = target.sample((100,), generator=generator(1))
    assert target.compute_metrics(x, ref_samples=other,
                                  compute_standard_metrics=False)['w1_x1'] == exact


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_the_lennard_jones_score_is_never_nan(dtype):
    """Particles closing in: the score grows large along their axis, and stays finite

    In float32 the radial factor f'(d) / d overflows below d ~ 3e-5. With two particles that
    used to make inf times a zero displacement NaN; with three on a line, the middle one's two
    infinite forces cancelled to NaN. The force is now f'(d) along the unit vector, and f'(d) is
    finite at every distance.
    """
    target = LJ13(dtype=dtype)
    x0 = target.initial_configuration()[None].clone()
    step = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    for separation in [1e-2, 1e-4, 3e-5, 1e-5, 1e-6, 0.0]:
        pair = x0.clone()
        pair[0, 3:6] = pair[0, 0:3] + separation * step
        line = pair.clone()
        line[0, 6:9] = line[0, 0:3] - separation * step
        for x in [pair, line]:
            assert torch.isfinite(target.score(x)).all(), separation
