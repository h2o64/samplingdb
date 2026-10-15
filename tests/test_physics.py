# The glass formers: pinned potentials, torus invariances, and the released reference data

# Libraries
import math
import pytest
import torch
from samplingdb import KobAndersen2D
from samplingdb.data import cache_dir
from samplingdb.targets import score_autograd
from samplingdb.targets.base import generator
from samplingdb.targets.physics import (
    ZENODO_RECORD, minimum_image, smoothing_coefficients, swap_monte_carlo,
)

# The released data is a download, so the tests that need it run only once it is cached
HAS_DATA = (cache_dir() / ('zenodo-' + ZENODO_RECORD) / 'ka44_T1.0_positions.pt').exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason='released configurations not in the cache')


def jittered_lattice(target, n=8, jitter=0.05):
    """Returns physically sensible configurations: a square lattice filling the box, plus noise"""
    side = int(math.ceil(math.sqrt(target.n_particles)))
    axis = torch.arange(side, dtype=target.dtype) * target.box / side
    grid = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), dim=-1).reshape(-1, 2)
    x = grid[:target.n_particles].reshape(1, -1).expand(n, target.dim).clone()
    return x + jitter * torch.randn(x.shape, generator=generator(0), dtype=target.dtype)


def test_smoothing_coefficients_match_the_pinned_triple():
    """The pinned (C0, C2, C4) are re-derived from W(rc) = W'(rc) = W''(rc) = 0"""
    derived = smoothing_coefficients(2.5)
    pinned = KobAndersen2D.PINNED_SMOOTHING[2.5]
    assert pinned == (0.04049023795, -0.00970155098, 0.00062012616)
    for got, expected in zip(derived, pinned):
        assert abs(got - expected) < 1e-8

    # And the released cutoff still gets the released triple, so the density is unchanged
    assert KobAndersen2D(temperature=1.0).smoothing == pinned


def test_a_different_cutoff_solves_for_its_own_smoothing():
    """(C0, C2, C4) belong to a cutoff, and carrying them to another one breaks the potential

    The triple is what makes W and its first two derivatives vanish AT r_cut. Reusing the 2.5 one
    at a 3.0 cutoff leaves W(rc) = +0.0081 and W'(rc) = +0.0460 -- a jump in the potential and a
    delta in the force, exactly at the truncation. Any cutoff other than the released one
    therefore solves for its own.
    """
    one = torch.ones(1, dtype=torch.float64)
    for cutoff in [2.0, 3.0]:
        target = KobAndersen2D(temperature=1.0, cutoff_factor=cutoff)
        assert target.smoothing == smoothing_coefficients(cutoff)
        r = torch.tensor([cutoff - 1e-9], dtype=torch.float64)
        assert abs(float(target.pair_potential(r, one, one))) < 1e-12
        assert abs(float(target.pair_potential_grad(r, one, one))) < 1e-12

    # The stale triple is what the discontinuity above measures
    stale = KobAndersen2D(temperature=1.0, cutoff_factor=3.0,
                          smoothing=KobAndersen2D.PINNED_SMOOTHING[2.5])
    r = torch.tensor([3.0 - 1e-9], dtype=torch.float64)
    assert abs(float(stale.pair_potential(r, one, one))) > 1e-3

    # The cutoff and the smoothing both version the density, so neither can collide in the cache
    assert (KobAndersen2D(temperature=1.0).train_cache_key(8)
            != KobAndersen2D(temperature=1.0, cutoff_factor=3.0).train_cache_key(8))


def test_ka_potential_and_its_first_two_derivatives_vanish_at_the_cutoff():
    """C2 continuity is what keeps the score continuous, so check all three by autograd

    NOTE: the pinned triple is quoted to 8 decimals, which leaves a residual of ~1e-9 at the
    cutoff rather than the ~1e-17 the exactly-solved coefficients give. That residual is what the
    released data was generated with, so it is the specification; the exact solution is checked
    separately below.
    """
    target = KobAndersen2D(temperature=1.0)
    one = torch.ones(1, dtype=torch.float64)
    r = torch.tensor([2.5 - 1e-7], dtype=torch.float64).requires_grad_(True)
    value = target.pair_potential(r, one, one)
    (first,) = torch.autograd.grad(value.sum(), r, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), r)
    assert abs(float(value.detach())) < 1e-8
    assert abs(float(first.detach())) < 1e-6
    assert abs(float(second)) < 1e-3

    # With the exactly-solved coefficients the same three quantities vanish to machine precision
    exact = KobAndersen2D(temperature=1.0, smoothing=smoothing_coefficients(2.5))
    r = torch.tensor([2.5 - 1e-9], dtype=torch.float64)
    assert abs(float(exact.pair_potential(r, one, one))) < 1e-15
    assert abs(float(exact.pair_potential_grad(r, one, one))) < 1e-14

    # And the potential is exactly zero beyond the cutoff
    beyond = torch.tensor([2.5 + 1e-9, 4.0], dtype=torch.float64)
    assert torch.equal(target.pair_potential(beyond, one.expand(2), one.expand(2)),
                       torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize('target_cls,kwargs', [
    (KobAndersen2D, {'temperature': 1.0}),
    (KobAndersen2D, {'temperature': 0.32}),
])
def test_analytic_score_matches_autograd_on_physical_configurations(target_cls, kwargs):
    """On a jittered lattice, not on overlapping particles where r^-13 swamps everything"""
    # float64: this asserts the analytic score EQUALS autograd, at 1e-8 on scores of order 1e3,
    # which is an algebraic identity rather than a numerical one. float32 lands at 1e-5, its own
    # working precision, and loosening the bound would stop the test seeing a wrong derivative
    target = target_cls(**{**kwargs, 'dtype': torch.float64})
    x = jittered_lattice(target)
    score, reference = target.score(x), score_autograd(target, x)
    assert torch.allclose(score, reference, rtol=1e-8, atol=1e-8)
    assert torch.isfinite(target.log_prob(x)).all()


def test_energy_is_invariant_on_the_torus():
    """Translation, periodic image, particle permutation within a species, and species relabeling"""
    target = KobAndersen2D(temperature=1.0)
    x = jittered_lattice(target, n=4)
    energy = target.energy(x)

    shifted = (target.positions(x) + torch.tensor([0.7, -1.3], dtype=x.dtype)).reshape(4, -1)
    assert torch.allclose(energy, target.energy(shifted))

    image = target.positions(x).clone()
    image[:, 3] = image[:, 3] + target.box
    assert torch.allclose(energy, target.energy(image.reshape(4, -1)))

    # Particles 0..19 all carry species 0, so permuting them changes nothing
    permuted = target.positions(x).clone()
    permuted[:, [0, 5]] = permuted[:, [5, 0]]
    assert torch.allclose(energy, target.energy(permuted.reshape(4, -1)))


def test_minimum_image_wraps_into_the_box():
    """Displacements come back inside (-L/2, L/2], whatever image they started in"""
    delta = torch.tensor([0.3, 5.0, -5.0, 12.7], dtype=torch.float64)
    wrapped = minimum_image(delta, 6.0)
    assert float(wrapped.abs().max()) <= 3.0
    assert torch.allclose(minimum_image(delta + 6.0, 6.0), wrapped)


def test_preprocess_is_the_sorted_distance_representation():
    """The universal panel compares an invariant representation, not raw coordinates"""
    target = KobAndersen2D(temperature=1.0)
    x = jittered_lattice(target, n=4)
    features = target.preprocess(x)
    assert features.shape == (4, target.n_pairs)
    assert torch.all(features[:, 1:] >= features[:, :-1])

    # Invariant under a translation and a particle permutation
    shifted = (target.positions(x) + 1.1).reshape(4, -1)
    assert torch.allclose(features, target.preprocess(shifted))
    order = list(range(target.n_particles))
    order[0], order[1] = order[1], order[0]
    permuted = target.positions(x)[:, order].reshape(4, -1)
    assert torch.allclose(features, target.preprocess(permuted))


def test_distance_edges_cover_every_reachable_distance():
    """The pinned range is box / sqrt(2), the largest minimum-image distance in a square box

    float64 for the 1e-12 identity: in float32 the last edge is off by ~1e-7 on a box of ~10, which
    is that dtype's own resolution and a thousandth of an RDF bin, so it changes no histogram -- but
    the point of the assertion is that the edge IS the geometric bound, not that it is close to it.
    """
    target = KobAndersen2D(temperature=1.0, dtype=torch.float64)
    edges = target.distance_edges()
    assert float(edges[0]) == 0.0
    assert abs(float(edges[-1]) - target.box / math.sqrt(2)) < 1e-12

    x = target.box * torch.rand(64, target.dim, generator=generator(0), dtype=torch.float64)
    assert float(target.distances(x).max()) <= float(edges[-1]) + 1e-9


def test_rdf_of_an_ideal_gas_is_one():
    """g(r) is normalised by the ideal-gas count, so uniform positions give g = 1"""
    target = KobAndersen2D(temperature=1.0)
    x = target.box * torch.rand(4096, target.dim, generator=generator(0), dtype=torch.float64)
    g, centres = target.rdf(x)
    middle = (centres > 0.5) & (centres < target.box / 2)
    assert abs(float(g[middle].mean()) - 1.0) < 0.05


def test_train_split_is_the_displacement_only_chain():
    """The literature-native biased split: no swap moves, far too few sweeps to equilibrate"""
    class ShortKobAndersen(KobAndersen2D):
        train_time_units = 5

    target = ShortKobAndersen(temperature=0.32)
    x = target.train_samples(n=8)
    assert x.shape == (8, target.dim)
    assert torch.equal(x, target.train_samples(n=8))
    assert torch.isfinite(target.energy(x)).all()

    # The recipe tag and the chain length ride on top of the density spec, so two budgets cannot
    # collide and a reader can see which recipe a cached file came from without opening it
    assert target.train_cache_key(8).endswith('_displacement-local-v1_t5')
    assert target.train_cache_key(8) != KobAndersen2D(temperature=0.32).train_cache_key(8)


def test_local_energy_sums_to_the_total_energy():
    """Every move scores itself from the rows it touches, so those rows must add up

    float64 for the 1e-10: the local rows summing to the total is exact algebra, and in float32 the
    two sides agree only to ~1e-7 relative on energies of order 1e3. A tolerance loose enough for
    float32 would no longer catch a row that double-counts a pair.
    """
    for target in [KobAndersen2D(temperature=1.0, dtype=torch.float64),
                   KobAndersen2D(temperature=0.32, dtype=torch.float64)]:
        x = jittered_lattice(target, n=8)
        positions = target.positions(x)
        total = 0.5 * sum(target.local_energy(positions, torch.full((8,), i))
                          for i in range(target.n_particles))
        assert torch.allclose(total, target.energy(x), rtol=1e-10, atol=1e-10)


def test_displacement_only_moves_the_particles():
    """p_swap = 0 must still displace: the partner write must not undo the displacement"""
    target = KobAndersen2D(temperature=1.0)
    x = target.positions(jittered_lattice(target, n=8))
    out, acceptance = swap_monte_carlo(target, x, n_time_units=2, step_size=0.065, p_swap=0.0,
                                       generator=generator(0), return_acceptance=True)
    assert not torch.allclose(out, x.reshape(8, -1))
    assert 0.1 < acceptance['displacement'] < 0.99
    assert acceptance['swap_fraction'] == 0.0


def test_swap_moves_exchange_two_particles_of_different_species():
    """The move set is ParticlesMC's: A<->C and B<->C only, never A<->B"""
    target = KobAndersen2D(temperature=1.0)
    assert target.swap_channels == ((0, 2), (1, 2))
    for first, second in target.swap_channels:
        assert target.species[target.species_indexes(first)].unique().tolist() == [first]
        assert target.species[target.species_indexes(second)].unique().tolist() == [second]

    # With p_swap = 1 every move is a swap, so the multiset of positions never changes
    x = target.positions(jittered_lattice(target, n=8))
    out, acceptance = swap_monte_carlo(target, x, n_time_units=2, step_size=0.065, p_swap=1.0,
                                       generator=generator(0), return_acceptance=True)
    assert acceptance['swap_fraction'] == 1.0
    assert torch.allclose(torch.sort(out.reshape(8, -1), dim=-1).values,
                          torch.sort(x.reshape(8, -1), dim=-1).values)


def test_swap_monte_carlo_relaxes_where_displacement_alone_cannot():
    """The paper's own reason for swap moves, on a budget small enough for a test

    At T = 0.32 the displacement-only chain is the train split and stays far above equilibrium,
    while the same budget with p_swap = 0.2 gets much closer. The long-run version of this check,
    against the released reference, is
    test_swap_monte_carlo_is_stationary_for_the_released_ensemble below.
    """
    target = KobAndersen2D(temperature=0.32)
    x = target.lattice_configuration(8, jitter=0.05, generator=generator(0))
    swapped = swap_monte_carlo(target, x, n_time_units=100, step_size=0.065, p_swap=0.2,
                               generator=generator(1))
    plain = swap_monte_carlo(target, x, n_time_units=100, step_size=0.065, p_swap=0.0,
                            generator=generator(1))
    assert float(target.energy(swapped).mean()) < float(target.energy(plain).mean())


@needs_data
def test_released_configurations_load_in_canonical_species_order():
    """The loader sorts each configuration by species and moves the positions with it"""
    target = KobAndersen2D(temperature=1.0)
    x = target.released_samples(n=64)
    assert x.shape == (64, target.dim)

    # Physically sensible: no overlaps, and every distance inside the box
    distances = target.distances(x)
    assert float(distances.min()) > 0.6
    assert float(distances.max()) <= target.box / math.sqrt(2) + 1e-9

    # The measured energy per particle of the released liquid
    assert abs(float(target.energy(x).mean()) / target.n_particles + 1.93) < 0.05


@needs_data
def test_released_configurations_are_stationary_for_our_law():
    """log_gamma = -E / T, not -(E/N) / T: only the former leaves the released data invariant

    This is the check behind PeriodicPairMixture.log_prob's warning. learndiffeq's KobAndersen
    defines U = E / n_particles and its experiments sample exp(-U / temp); under that law the
    released configurations are far from stationary.
    """
    target = KobAndersen2D(temperature=1.0)
    x = target.positions(target.released_samples(n=64))
    before = float(target.energy(x.reshape(64, -1)).mean()) / target.n_particles

    out = swap_monte_carlo(target, x, n_time_units=50, step_size=0.065, p_swap=0.0,
                           generator=generator(0))
    after = float(target.energy(out).mean()) / target.n_particles
    assert abs(after - before) < 0.1

    # The per-particle convention is exp(-E / (N T)), i.e. our law at N times the temperature,
    # and it heats the released configurations right out of their basin
    hot_target = KobAndersen2D(temperature=1.0)
    hot_target.temperature = float(hot_target.n_particles)
    hot = swap_monte_carlo(hot_target, x, n_time_units=50, step_size=0.065, p_swap=0.0,
                           generator=generator(0))
    assert float(target.energy(hot).mean()) / target.n_particles > before + 1.0

    # And a target whose log_prob is not -E / T cannot be sampled by this kernel at all
    class PerParticle(KobAndersen2D):
        def log_prob(self, x):
            return -self.energy(x) / (self.n_particles * self.temperature)

    with pytest.raises(AssertionError):
        swap_monte_carlo(PerParticle(temperature=1.0), x, n_time_units=1, p_swap=0.0,
                         generator=generator(0))


@needs_data
def test_released_temperatures_order_as_physics_requires():
    """Cooling from T = 1.0 to T = 0.32 lowers the energy and narrows its spread"""
    hot = KobAndersen2D(temperature=1.0).released_samples(n=512)
    cold = KobAndersen2D(temperature=0.32).released_samples(n=512)
    target = KobAndersen2D(temperature=1.0)
    e_hot = target.energy(hot) / target.n_particles
    e_cold = target.energy(cold) / target.n_particles
    assert float(e_cold.mean()) < float(e_hot.mean()) - 0.5
    assert float(e_cold.std()) < float(e_hot.std())


@needs_data
def test_structural_panel_against_the_released_data():
    """The panel: distance and energy TVDs with their dropped mass, g(r), and the raw histograms"""
    target = KobAndersen2D(temperature=1.0)
    reference = target.released_samples(n=1024)
    ret, hist = target.compute_metrics(reference[:512], ref_samples=reference[512:],
                                       compute_standard_metrics=False, return_hist=True)
    assert {'dist_total_var', 'dist_out_of_range', 'dist_w1', 'rdf_max_abs_err',
            'energy_total_var', 'energy_out_of_range', 'energy_w1'} <= set(ret)

    # Two halves of the same equilibrium set agree, and the box-pinned edges drop nothing
    assert ret['dist_total_var'] < 0.1
    assert ret['dist_out_of_range'] < 1e-9
    assert ret['rdf_max_abs_err'] < 0.5

    # The raw g(r) of a glass former has its first peak near the smallest pair diameter
    g, centres = hist['rdf']['g'], hist['rdf']['centres']
    assert float(centres[g.argmax()]) < 1.2
    assert float(g.max()) > 2.0


@needs_data
def test_swap_monte_carlo_is_stationary_for_the_released_ensemble():
    """A correct swap move leaves the equilibrium ensemble where it found it

    Recorded numbers from a 32-chain, 3000-time-unit run: the released reference sits at
    E/N = -2.6273 +- 0.052; swap Monte Carlo from the released configurations stays at -2.6357,
    swap Monte Carlo from a jittered lattice reaches -2.6284, and displacement-only from the same
    lattice is still stuck at -1.8950. This test is the short version of the first of those.
    """
    target = KobAndersen2D(temperature=0.32)
    released = target.released_samples(n=32)
    reference = float(target.energy(released).mean()) / target.n_particles
    assert abs(reference + 2.6273) < 0.05

    out = swap_monte_carlo(target, target.positions(released), n_time_units=100, step_size=0.065,
                           p_swap=0.2, generator=generator(0))
    assert abs(float(target.energy(out).mean()) / target.n_particles - reference) < 0.1


def has_generated(target):
    """Returns whether this library's own swap-Monte-Carlo reference exists yet"""
    return (cache_dir() / 'swapmc' / (target.name + '.npy')).exists()


def test_the_relaxation_time_is_measured_not_assumed():
    """tau_alpha comes from F_s(k, t) crossing 1/e, and swap moves are what make it short

    Measured from the released ensembles with 64 chains: at T = 0.32 the swap chain relaxes in 509
    time units and the displacement-only chain needs 2674, so swap moves are worth 5.3x here. That
    ratio is small because T = 0.32 is the ONSET regime of this model family (Jung et al. 2024
    put T_g near 0.15), not the deeply supercooled one -- and 509 rather than the released
    5 x 10^4 is what makes a 100 000-configuration reference set an hour's work instead of a week's.
    """
    target = KobAndersen2D(temperature=0.32)
    assert target.relaxation_times == {1.0: 40.0, 0.32: 509.0}
    assert abs(target.wavevector_peak - 7.30) < 0.01
    needs_data_or_skip()

    x0 = target.positions(target.released_samples(n=32))
    assert abs(target.self_intermediate_scattering(x0, x0) - 1.0) < 1e-12

    # After one tau_alpha of swap Monte Carlo it has decayed to roughly 1/e, and the
    # displacement-only chain of the same length has not
    tau = int(target.relaxation_times[0.32])
    swapped = target.positions(swap_monte_carlo(target, x0, n_time_units=tau, p_swap=0.2,
                                                generator=generator(0)))
    plain = target.positions(swap_monte_carlo(target, x0, n_time_units=tau, p_swap=0.0,
                                             generator=generator(0)))
    assert 0.2 < target.self_intermediate_scattering(x0, swapped) < 0.55
    assert target.self_intermediate_scattering(x0, plain) > \
        target.self_intermediate_scattering(x0, swapped)


def needs_data_or_skip():
    """Skips unless the released configurations are cached"""
    if not HAS_DATA:
        pytest.skip('released configurations not in the cache')


@needs_data
def test_our_generated_reference_matches_the_released_one():
    """Our swap-Monte-Carlo set must be indistinguishable from the published one, not merely close

    Recorded from 106 496 generated configurations against 10 000 released ones: E/N -2.6247 +-
    0.0525 against -2.6272 +- 0.0519, and every structural metric at the released set's own
    self-comparison floor -- distance TVD 0.0034 across the two against 0.0033 between two halves of
    theirs, g(r) error 0.038 against 0.048.
    """
    target = KobAndersen2D(temperature=0.32)
    if not has_generated(target):
        with pytest.raises(FileNotFoundError, match='gen/swap_mc/generate.py'):
            target.test_samples(n=8)
        pytest.skip('no generated reference for ' + target.name + ' yet')

    ours = target.test_samples(n=2000)
    released = target.released_samples(n=4000)
    assert len(target.test_samples()) >= 100000

    energy = target.energy(ours).mean() / target.n_particles
    assert abs(float(energy) + 2.6273) < 0.02

    floor = target.compute_metrics(released[0::2], ref_samples=released[1::2],
                                   compute_standard_metrics=False)
    cross = target.compute_metrics(ours, ref_samples=released[:2000],
                                   compute_standard_metrics=False)
    for name in ['dist_total_var', 'dist_w1', 'rdf_max_abs_err', 'energy_w1']:
        assert cross[name] < 3 * floor[name], name
