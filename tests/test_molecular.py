# The four SBG peptides: two pinned force-field families, the OpenMM bridge, and the torsion panel

# Libraries
import importlib.util
import math
import numpy as np
import pytest
import torch
from samplingdb.data import cache_dir, fetch_hf
from samplingdb.targets.base import generator
from samplingdb.targets.molecular import (
    FF14SB_OBC1, FF99SBILDN_OBC, PEPTIDE_SYSTEMS, SBG_CHECKSUMS, SBG_REPO, SBG_REVISION, center,
)
from samplingdb.targets.forcefield import GAS_CONSTANT, linlogcut

# openmm and mdtraj come with the [molecular] extra; the PDB and the trajectories are downloads
HAS_OPENMM = all(importlib.util.find_spec(name) is not None for name in ['openmm', 'mdtraj'])
needs_openmm = pytest.mark.skipif(not HAS_OPENMM, reason='the [molecular] extra is not installed')


def cached(path_in_repo):
    """Returns whether one file of the SBG dataset is already in the local cache"""
    if not HAS_OPENMM:
        return False
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(SBG_REPO, path_in_repo, repo_type='dataset',
                                   cache_dir=str(cache_dir()))
    return isinstance(found, str)


def has_pdb(sequence):
    """Returns whether the released PDB of one system is cached, which is all building needs"""
    name = sequence + '_' + str(int(PEPTIDE_SYSTEMS[sequence]['temperature'])) + 'K'
    return cached(name + '/' + name + '.pdb')


def has_split(sequence, split='test'):
    """Returns whether one released trajectory split is cached"""
    name = sequence + '_' + str(int(PEPTIDE_SYSTEMS[sequence]['temperature'])) + 'K'
    return cached(name + '/' + name + '_' + split + '.npy')


def reference_ensemble(target, n=None):
    """Returns whatever this tier's reference ensemble is, released or generated

    test_samples() reads the replica-exchange ensemble gen/md/generate.py writes, which is the
    tier's reference now that the released splits have been measured and found to carry between
    18 and 42 000 independent structures rather than the 20 000 to 980 000 they hold. Until that
    ensemble exists on a given machine, these tests fall back to the released split so that the
    force field, the metrics and the integrator stay covered -- what they check is not the
    ensemble's quality.
    """
    if target.reference_path().exists():
        return target.test_samples(n)
    return target.load_split('test', n=n)


def peptide(sequence, **kwargs):
    """Builds one peptide target, uncompiled and in float64

    compile=True is the library default and the right one for a real run, but it costs ~15 s per
    batch shape and this suite is a few hundred tiny batches, so every test here would pay for a
    kernel it uses once. gen/openmm_check/compare.py is what checks the compiled path.

    float64 although the library default is float32, because what this suite checks is EXACTNESS --
    the score against OpenMM's force, an exception excluded rather than subtracted back, the same
    answer at every batch size, TICA features invariant to a rotation. Those hold algebraically and
    are asserted near 1e-9, which float32 cannot resolve: chunking alone reorders the summation, so
    in single precision the answer depends on the batch size at the 1e-6 level. float32 is validated
    against float64 instead, at its own tolerance, by
    test_float32_agrees_with_float64_on_real_conformations below. That is the chain -- float64
    against OpenMM at 1e-5 kT, float32 against float64 at 2e-3 kT -- and loosening these tolerances
    to accommodate float32 would have retired the exactness checks instead of adding a precision
    one.
    """
    from samplingdb import Peptide

    return Peptide(sequence=sequence, **{'compile': False, 'dtype': torch.float64, **kwargs})


def test_the_registry_pins_two_force_field_families():
    """Four systems, two families -- and the name has to keep them apart"""
    assert set(PEPTIDE_SYSTEMS) == {'Ace-A-Nme', 'Ace-AAA-Nme', 'AAA', 'AAAAAA'}

    # The 300 K family: amber99sbildn + tip3p + amber99_obc, no nonbonded cutoff
    assert FF99SBILDN_OBC['force_field'] == ('amber99sbildn.xml', 'tip3p.xml', 'amber99_obc.xml')
    assert FF99SBILDN_OBC['nonbonded_method'] == 'NoCutoff'
    assert [s for s in PEPTIDE_SYSTEMS if PEPTIDE_SYSTEMS[s]['family'] is FF99SBILDN_OBC] == \
        ['Ace-A-Nme', 'Ace-AAA-Nme']

    # The 310 K family: amber14-all + implicit/obc1, 2 nm non-periodic cutoff
    assert FF14SB_OBC1['force_field'] == ('amber14-all.xml', 'implicit/obc1.xml')
    assert FF14SB_OBC1['nonbonded_method'] == 'CutoffNonPeriodic'
    assert FF14SB_OBC1['nonbonded_cutoff'] == 2.0

    # Every system's temperature matches its family, and the frictions are the released ones
    for sequence, system in PEPTIDE_SYSTEMS.items():
        expected = 300.0 if system['family'] is FF99SBILDN_OBC else 310.0
        assert system['temperature'] == expected, sequence
        assert system['friction'] == (1.0 if sequence == 'Ace-A-Nme' else 0.3), sequence


def test_linlogcut_is_the_identity_below_the_cut():
    """FAB's (energy_cut, energy_max) never touch a physical energy, and soften an unphysical one

    The clamp runs after the logarithm, so a finite energy stays below 1e8 + 710 and only an
    infinite one reaches max_val.
    """
    values = torch.tensor([-1e3, 0.0, 1e7, 1e9, 1e30, float('inf')], dtype=torch.float64)
    cut = linlogcut(values, high_val=1e8, max_val=1e20)
    assert torch.equal(cut[:3], values[:3])
    assert abs(float(cut[3]) - (1e8 + math.log1p(1e9 - 1e8))) < 1e-6
    assert abs(float(cut[4]) - (1e8 + math.log1p(1e30 - 1e8))) < 1e-6
    assert float(cut[5]) == 1e20
    assert bool((cut[1:] > cut[:-1]).all())

    # The largest finite energy in either precision lands far below max_val
    for dtype, bound in [(torch.float64, 710.0), (torch.float32, 89.0)]:
        largest = torch.tensor([torch.finfo(dtype).max], dtype=dtype)
        assert float(linlogcut(largest, high_val=1e8, max_val=1e20)) <= 1e8 + bound


def test_linlogcut_has_a_finite_gradient_on_both_branches():
    """Both branches of a torch.where are evaluated, so the dead one must not be a nan"""
    values = torch.tensor([0.0, 1e7, 1e8 + 5.0], dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(linlogcut(values, high_val=1e8, max_val=1e20).sum(), values)
    assert torch.isfinite(grad).all()
    assert torch.allclose(grad[:2], torch.ones(2, dtype=torch.float64))


def test_gas_constant_gives_the_right_kT():
    """kT is 2.494339 kJ/mol at 300 K and 2.577484 at 310 K, the two temperatures in play"""
    assert abs(GAS_CONSTANT * 300.0 - 2.494339) < 1e-6
    assert abs(GAS_CONSTANT * 310.0 - 2.577484) < 1e-6


def test_center_removes_the_centre_of_geometry():
    """The released splits disagree on centring, so the loader always centres"""
    x = torch.randn(4, 3 * 5, dtype=torch.float64) + 7.0
    centred = center(x, 5)
    assert torch.allclose(centred.reshape(4, 5, 3).mean(1), torch.zeros(4, 3, dtype=torch.float64),
                          atol=1e-12)

    # Centring is a translation, so it leaves every interatomic distance alone
    pair = lambda y: (y.reshape(4, 5, 3)[:, 0] - y.reshape(4, 5, 3)[:, 3]).norm(dim=-1)
    assert torch.allclose(pair(x), pair(centred))


@needs_openmm
@pytest.mark.parametrize('sequence', list(PEPTIDE_SYSTEMS))
def test_every_system_builds_with_the_atom_count_it_claims(sequence):
    """A force-field or topology mismatch would show up here as a wrong atom count"""
    if not has_pdb(sequence):
        pytest.skip('the released PDB of ' + sequence + ' is not cached')
    target = peptide(sequence)
    system = PEPTIDE_SYSTEMS[sequence]
    assert target.n_atoms == system['n_atoms']
    assert target.dim == 3 * system['n_atoms']
    assert target.system.getNumConstraints() == 0
    assert target.name == sequence.lower() + '-T' + str(int(system['temperature']))

    # Every backbone torsion mdtraj finds, phi first then psi, and no duplicates
    assert len(target.torsion_names) == len(target.torsion_indexes)
    assert target.torsion_names[0] == 'phi1'
    assert len(set(map(tuple, target.torsion_indexes.tolist()))) == len(target.torsion_indexes)


@needs_openmm
def test_the_score_is_deterministic_and_is_the_force():
    """The torch force field is deterministic, and its score is the gradient of its energy"""
    if not has_pdb('Ace-A-Nme'):
        pytest.skip('the released PDB is not cached')
    target = peptide('Ace-A-Nme')
    x = torch.as_tensor(target.reference_positions.reshape(1, -1), dtype=torch.float64)

    assert torch.equal(target.score(x), target.score(x))
    log_prob, score = target.log_prob_and_grad(x)
    assert torch.equal(log_prob, target.log_prob(x))
    assert torch.equal(score, target.score(x))

    # The score is the gradient of the very energy log_prob returns, so autograd agrees with it
    grad_x = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(target.log_prob(grad_x).sum(), grad_x)
    assert torch.allclose(grad, score, rtol=1e-9, atol=1e-9)

    # And it matches a central difference of its own energy
    for coordinate in [0, 17, 40, 65]:
        plus, minus = x.clone(), x.clone()
        plus[0, coordinate] += 1e-5
        minus[0, coordinate] -= 1e-5
        finite = float(target.log_prob(plus) - target.log_prob(minus)) / 2e-5
        assert abs(float(score[0, coordinate]) - finite) < 1e-2 * (1 + abs(finite))


@needs_openmm
@pytest.mark.parametrize('sequence', list(PEPTIDE_SYSTEMS))
def test_the_torch_force_field_reproduces_openmm(sequence):
    """The energy is a torch re-implementation, and this is what pins it to the force field

    OpenMM builds the system and owns every parameter; only the summation is reimplemented. Both
    tolerances are far below the ~1e-4 by which OpenMM's own CPU and CUDA platforms differ from
    Reference, so what is being checked here is the re-implementation and not a platform.
    """
    import openmm

    if not (has_pdb(sequence) and has_split(sequence)):
        pytest.skip('the released trajectory of ' + sequence + ' is not cached')
    target = peptide(sequence)
    x = reference_ensemble(target, n=8).to(torch.float64)
    log_prob, score = target.log_prob_and_grad(x)

    context = openmm.Context(target.system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName('Reference'))
    for index, frame in enumerate(target.positions(x).numpy()):
        context.setPositions(frame)
        state = context.getState(getEnergy=True, getForces=True)
        assert abs(state.getPotentialEnergy()._value / target.kT + float(log_prob[index])) < 1e-5
        force = state.getForces(asNumpy=True)._value.reshape(-1) / target.kT
        assert np.abs(force - score[index].numpy()).max() < 1e-4


@needs_openmm
def test_a_system_with_no_implicit_solvent_is_still_a_system():
    """The force field carries no Born term when the system has no GB force, rather than raising

    Every peptide in the catalog is solvated, so nothing here exercised the vacuum branch, and
    extract_parameters used to leave gb_surface and gb_constant unset for it -- a KeyError in
    TorchForceField's constructor. A vacuum peptide is a legitimate target (the ff96 variant of
    catalog's aldp entry mentions one), so the parameters carry defaults and the potential
    skips the term.
    """
    import openmm
    from openmm import app
    from samplingdb.targets.forcefield import TorchForceField

    if not has_pdb('Ace-A-Nme'):
        pytest.skip('the released PDB is not cached')
    name = 'Ace-A-Nme_300K/Ace-A-Nme_300K.pdb'
    pdb = app.PDBFile(str(fetch_hf(SBG_REPO, name, SBG_REVISION, SBG_CHECKSUMS[name])))
    system = app.ForceField('amber99sbildn.xml').createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)

    bridge = TorchForceField(system, 300.0, compile=False, dtype=torch.float64)
    assert bridge.gb_model is None
    x = torch.as_tensor(pdb.getPositions(asNumpy=True)._value.reshape(1, -1), dtype=torch.float64)
    x = x.repeat(4, 1) + 0.001 * torch.randn(4, x.shape[1], generator=generator(0),
                                             dtype=torch.float64)
    energy, force = bridge.evaluate(x)

    kT = GAS_CONSTANT * 300.0
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName('Reference'))
    for index, frame in enumerate(x.reshape(len(x), -1, 3).numpy()):
        context.setPositions(frame)
        state = context.getState(getEnergy=True, getForces=True)
        assert abs(state.getPotentialEnergy()._value / kT - float(energy[index])) < 1e-5
        reference = state.getForces(asNumpy=True)._value.reshape(-1) / kT
        assert np.abs(reference - force[index].numpy()).max() < 1e-4


@needs_openmm
def test_the_score_survives_being_differentiated_through():
    """A loss may be built on the score, which is what the retired OpenMM bridge could not support

    The bridge injected the forces through a custom autograd Function holding a saved tensor, so
    the score came back detached and a second derivative did not exist. Here it can be a real
    graph node, but only when asked: retaining the backward graph is not free.
    """
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme', differentiable_score=True)
    x = reference_ensemble(target, n=4).clone().requires_grad_(True)
    log_prob, score = target.log_prob_and_grad(x)
    assert log_prob.requires_grad and score.requires_grad

    # The distance matrix carries an exact zero on its diagonal, where torch.linalg.norm has a
    # nan second derivative; forcefield.norm is what keeps this finite
    (grad,) = torch.autograd.grad(score.pow(2).sum(), x)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().max()) > 0

    # Off by default: the energy is still differentiable, the score is not
    target.bridge.differentiable_score = False
    x = reference_ensemble(target, n=4).clone().requires_grad_(True)
    log_prob, score = target.log_prob_and_grad(x)
    assert log_prob.requires_grad and not score.requires_grad
    (grad,) = torch.autograd.grad(log_prob.sum(), x)
    assert torch.allclose(grad, score, rtol=1e-9, atol=1e-9)

    # And without requires_grad nothing is kept alive at all
    plain_log_prob, plain_score = target.log_prob_and_grad(reference_ensemble(target, n=4))
    assert not plain_log_prob.requires_grad and not plain_score.requires_grad


@needs_openmm
def test_chunking_bounds_the_memory_without_changing_the_answer():
    """A 10^5-frame reference set would allocate (n, n_atoms, n_atoms) all at once otherwise"""
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme')
    x = reference_ensemble(target, n=16)
    energy, score = target.bridge.evaluate(x)

    target.bridge.chunk_size = 3
    chunked_energy, chunked_score = target.bridge.evaluate(x)
    assert torch.allclose(energy, chunked_energy, rtol=0, atol=1e-9)
    assert torch.allclose(score, chunked_score, rtol=0, atol=1e-7)


@needs_openmm
@pytest.mark.parametrize('sequence', ['Ace-A-Nme', 'AAA'])
def test_an_exception_is_excluded_from_the_pair_list_not_subtracted_back(sequence):
    """A 1-2 pair driven into an overlap is where two algebraically identical formulas part

    An exception REPLACES the standard pair value, so summing the full triangle and adding
    (replacement - standard) is exact on paper, and it agrees to 1e-6 kJ/mol on any released
    frame -- which is why it survived a validation restricted to those. Below ~0.02 nm the
    standard Lennard-Jones term of the pair about to be replaced passes 1e18 kJ/mol; float64
    resolution there is coarser than the whole physical nonbonded energy, so the sum loses every
    real term before the subtraction can undo it. That cost 125 to 650 kJ/mol, sometimes
    DOWNWARD, which is a spurious low-energy region rather than an honest inf. Excluding the pair
    instead costs nothing and is what OpenMM does.
    """
    import openmm

    if not (has_pdb(sequence) and has_split(sequence)):
        pytest.skip('the released trajectory of ' + sequence + ' is not cached')
    target = peptide(sequence)
    field = target.bridge

    # Every exception is gone from the nonbonded list, and the Born term still carries them all
    triangle = field.n_atoms * (field.n_atoms - 1) // 2
    assert len(field.pair_rows) == triangle - len(field.exception_index)
    assert len(field.gb_rows) == triangle

    # Walk one bonded pair from its equilibrium length into a hard overlap
    a, b = (int(v) for v in field.bond_index[0])
    frame = reference_ensemble(target, n=1).reshape(1, field.n_atoms, 3).to(torch.float64)
    batch = frame.repeat(5, 1, 1)
    direction = frame[0, b] - frame[0, a]
    direction = direction / direction.norm()
    for index, separation in enumerate([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
        batch[index, b] = batch[index, a] + separation * direction

    energy = field.energy(batch.reshape(len(batch), -1))
    context = openmm.Context(target.system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName('Reference'))
    for index in range(len(batch)):
        context.setPositions(batch[index].numpy())
        reference = context.getState(getEnergy=True).getPotentialEnergy()._value / field.kT
        assert abs(reference - float(energy[index])) < 1e-4, index


@needs_openmm
@pytest.mark.parametrize('sequence', ['Ace-A-Nme', 'AAA'])
def test_the_born_term_follows_the_force_each_branch_is_read_from(sequence):
    """Ponder's inside-atom correction is in GBSAOBCForce and not in the Amber app's CustomGBForce

    Ace-A-Nme is the obc2 branch, read from the built-in force that carries the term, and AAA the
    obc1 branch, read from the expression that omits it. Without the term obc2 read up to 12.7
    kJ/mol below Reference inside the window while obc1 agreed to 1e-12, so asserting both pins
    each branch to its own source. The Born term is isolated in its own force group because the
    Lennard-Jones term at these separations would swamp it. float32 is run as well because the
    term divides by L_ij; it fires only where L_ij > or_i, so it stays bounded.
    """
    import copy
    import openmm

    if not (has_pdb(sequence) and has_split(sequence)):
        pytest.skip('the released trajectory of ' + sequence + ' is not cached')
    target = peptide(sequence)
    system = copy.deepcopy(target.system)
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(1 if force.__class__.__name__ in ['GBSAOBCForce', 'CustomGBForce']
                            else 0)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName('Reference'))

    # Put atom i on the pair with the widest window sr_j - or_i, at both sides of its threshold
    gap = target.bridge.gb_sr[None, :] - target.bridge.gb_or[:, None]
    gap.fill_diagonal_(-float('inf'))
    i, j = divmod(int(gap.argmax()), target.n_atoms)
    window = float(gap[i, j])
    frame = reference_ensemble(target, n=1).reshape(target.n_atoms, 3).to(torch.float64)
    batch = frame.repeat(4, 1, 1)
    batch[:, i] = frame[j]
    batch[:, i, 0] += window * torch.tensor([1.5, 0.95, 0.5, 0.01], dtype=torch.float64)

    # Score it in both precisions, against Reference
    expected = []
    for positions in batch.numpy():
        context.setPositions(positions)
        expected.append(context.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value)
    expected = torch.tensor(expected, dtype=torch.float64)
    for dtype, tolerance in [(torch.float64, 1e-5), (torch.float32, 1e-2)]:
        field = peptide(sequence, dtype=dtype).bridge
        positions = batch.to(dtype).requires_grad_(True)
        energy = field.generalized_born(positions)
        (gradient,) = torch.autograd.grad(energy.sum(), positions)
        assert (energy.double() - expected).abs().max() < tolerance, (dtype, energy - expected)
        assert torch.isfinite(gradient).all(), dtype


@needs_openmm
def test_the_angle_force_is_continuous_through_a_collinear_triple():
    """The angle is atan2 of a cross product, not acos of a cosine clamped off +-1

    The clamp saturates at pi - theta ~ 1.4e-6, and there it freezes theta and takes ~31% of the
    angle force with it in a single step. OpenMM's own HarmonicAngleForce clamps the same way, so
    this is the one place the module deliberately does NOT match Reference; it is the more
    physical of the two, and they still agree to 1e-4 kJ/mol/nm at every geometry short of a
    clash.

    NOTE: two other terms have to be got out of the way first, and neither is a defect. Straight-
    ening a triple walks the moved atom past its neighbours, so the nonbonded force on it reaches
    7e8 kT/nm; and a periodic torsion through a collinear triple is singular by construction,
    its phi undefined and its gradient going as 1/(pi - theta), in OpenMM exactly as here. So
    bonded() is called directly and the torsion stiffnesses are zeroed, leaving bonds -- smooth
    under this move, since it preserves every bond length -- and the angle being checked.
    """
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme')
    field = target.bridge
    a, b, c = (int(v) for v in field.angle_index[0])

    # Lay a-b-c on a straight line, then bend it away by shrinking angles
    frame = reference_ensemble(target, n=1).reshape(1, field.n_atoms, 3).to(torch.float64)
    batch = frame.repeat(3, 1, 1)
    axis = frame[0, a] - frame[0, b]
    axis = axis / axis.norm()
    away = torch.linalg.cross(axis, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    away = away / away.norm()
    length = (frame[0, c] - frame[0, b]).norm()
    for index, offset in enumerate([1e-4, 1e-6, 1e-8]):
        batch[index, c] = batch[index, b] + length * (-np.cos(offset) * axis
                                                      + np.sin(offset) * away)

    # The force on the moved atom must not step as the clamp's threshold is crossed
    field.torsion_k.zero_()
    batch.requires_grad_(True)
    (gradient,) = torch.autograd.grad(field.bonded(batch).sum(), batch)
    magnitude = gradient[:, c].norm(dim=-1)
    assert float((magnitude.max() - magnitude.min()) / magnitude.mean()) < 1e-2


@needs_openmm
def test_the_chunk_size_follows_the_compiled_flag():
    """torch.compile holds ~8x fewer work matrices, so a compiled potential earns a bigger chunk

    A fixed entry count sized the chunk for eager memory and then used an eighth of it on the
    compiled path, which is the library default.
    """
    from samplingdb.targets.forcefield import MATRICES_COMPILED_NO_CUTOFF, MATRICES_EAGER

    if not has_pdb('Ace-A-Nme'):
        pytest.skip('the released PDB is not cached')
    eager = peptide('Ace-A-Nme').bridge
    compiled = peptide('Ace-A-Nme', compile=True).bridge

    assert eager.bytes_per_configuration() == int(MATRICES_EAGER * eager.n_atoms ** 2 * 8)
    assert compiled.bytes_per_configuration() == \
        int(MATRICES_COMPILED_NO_CUTOFF * compiled.n_atoms ** 2 * 8)
    assert compiled.chunk_size > 3 * eager.chunk_size
    assert eager.suggested_chunk_size(budget=eager.bytes_per_configuration()) == 1


@needs_openmm
def test_the_input_dtype_does_not_set_the_working_precision():
    """A float32 input is CAST to the working dtype, not promoted from

    float64 buffers promote a float32 input, so the returned dtype looks right while the
    coordinates that went in have already lost their digits. Several released trajectories are
    stored as float32 on disk, so a caller's `samples * std` can silently be one. The cast fixes
    the arithmetic and cannot recover what the caller threw away -- which is the other half of
    why float64 is the default rather than a preference.
    """
    from samplingdb.targets.forcefield import TorchForceField

    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme')
    x = reference_ensemble(target, n=8).to(torch.float64)

    # The generated ensemble is stored in float32, which a cast to float32 reproduces exactly, so
    # give the coordinates digits that float32 cannot hold
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(0), dtype=x.dtype)
    x = x + 1e-6 * noise.to(x.device)

    # A float32 input still comes back at the module's precision, and its digits are gone
    assert target.bridge.energy(x.to(torch.float32)).dtype == torch.float64
    lost = (target.bridge.energy(x.to(torch.float32)) - target.bridge.energy(x)).abs().max()
    assert 0.0 < float(lost) < 1.0

    # And a float32 module is float32 whatever it is fed, rather than promoting to the input
    single = TorchForceField(target.system, target.temperature, compile=False,
                             dtype=torch.float32)
    assert single.energy(x).dtype == torch.float32


@needs_openmm
def test_preprocess_is_the_ordered_distance_matrix():
    """Atoms are labelled here, unlike Tier 1 particles, so the distances are NOT sorted"""
    if not has_pdb('Ace-A-Nme'):
        pytest.skip('the released PDB is not cached')
    target = peptide('Ace-A-Nme')
    x = torch.as_tensor(target.reference_positions.reshape(1, -1), dtype=torch.float64)
    features = target.preprocess(x)
    assert features.shape == (1, 22 * 21 // 2)

    rotation = torch.linalg.qr(torch.randn(3, 3, generator=generator(0), dtype=x.dtype)).Q
    rotated = (target.positions(x) @ rotation).reshape(1, -1)
    assert torch.allclose(features, target.preprocess(rotated), atol=1e-10)
    assert torch.allclose(target.torsions(x).abs(), target.torsions(rotated).abs(), atol=1e-8)


@needs_openmm
@pytest.mark.parametrize('sequence', list(PEPTIDE_SYSTEMS))
def test_the_released_splits_load_centred_and_physical(sequence):
    """Recorded mean energies in kT: -22.5, +12.6, -152.9, -84.2, -453.5 for the five systems"""
    if not (has_pdb(sequence) and has_split(sequence)):
        pytest.skip('the released trajectory of ' + sequence + ' is not cached')
    target = peptide(sequence)
    x = reference_ensemble(target, n=64)
    assert x.shape == (64, target.dim)

    # load_split centres, whichever split it read
    assert torch.allclose(target.positions(x).mean(1), torch.zeros(64, 3, dtype=x.dtype),
                          atol=1e-6)

    # Energies are finite and narrow: a topology mismatch would blow this up by orders of magnitude
    energy = target.energy(x).detach()
    assert torch.isfinite(energy).all()
    assert float(energy.std()) < 30.0
    expected = {'Ace-A-Nme': -22.5, 'Ace-AAA-Nme': 12.6, 'AAA': -152.9, 'AAAAAA': -84.2}[sequence]
    assert abs(float(energy.mean()) - expected) < 5.0


@needs_openmm
def test_the_released_ensemble_is_stationary_under_its_own_integrator():
    """The anchor that replaced FAB's -138.969687 kJ/mol constant

    Running the released integrator from released frames must leave the energy where it found it.
    That validates the force field, the solvent model, the nonbonded treatment, the temperature,
    the nm/kT units and the atom ordering of the arrays, all at once, and it does so per system
    rather than at one structure. Measured drift over 2 ps on 32 frames, in units of the ensemble
    standard deviation: Ace-A-Nme +0.16, Ace-AAA-Nme -0.10, AAA -0.01, AAAAAA -0.14. That also
    settles Ace-AAA-Nme's POSITIVE mean energy as genuine for its force field rather than a
    mismatch.

    NOTE: eight frames is too few for a mean drift, which is why this runs one frozen draw rather
    than a fresh one. Seed 2 on the Reference platform gives -0.267 and is bit-reproducible; a
    seed of ZERO is not, because OpenMM reads it as an instruction to pick a random one, and this
    test measured anywhere between 0.05 and 1.40 while it did. Reference rather than CPU for the
    same reason: it is the one platform that repeats itself exactly.
    """
    import openmm

    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme')
    x = reference_ensemble(target, n=8)
    before = target.energy(x).detach()

    integrator = openmm.LangevinMiddleIntegrator(target.temperature * openmm.unit.kelvin,
                                                target.friction / openmm.unit.picosecond,
                                                0.001 * openmm.unit.picosecond)
    integrator.setRandomNumberSeed(2)
    simulation = openmm.app.Simulation(target.topology, target.system, integrator,
                                       openmm.Platform.getPlatformByName('Reference'))
    frames = []
    for frame in target.positions(x).numpy():
        simulation.context.setPositions(frame)
        simulation.context.setVelocitiesToTemperature(target.temperature * openmm.unit.kelvin, 2)
        simulation.step(1000)
        state = simulation.context.getState(getPositions=True)
        frames.append(state.getPositions(asNumpy=True)._value.reshape(-1))
    after = target.energy(torch.as_tensor(np.stack(frames), dtype=torch.float64)).detach()

    drift = abs(float(after.mean() - before.mean())) / float(before.std())
    assert drift < 1.0


@needs_openmm
def test_the_md_train_split_is_non_equilibrium_but_physical():
    """100 ps at 300 K gets the basin POPULATIONS wrong, not the coverage"""
    if not has_pdb('Ace-A-Nme'):
        pytest.skip('the released PDB is not cached')
    target = peptide('Ace-A-Nme')
    train = target.train_samples(n=200)
    assert train.shape == (200, target.dim)
    assert torch.equal(train, target.train_samples(n=200))

    # Centred, physical, and visibly one trajectory rather than an i.i.d. sample
    assert torch.allclose(target.positions(train).mean(1), torch.zeros(200, 3, dtype=train.dtype),
                          atol=1e-6)
    assert torch.isfinite(target.energy(train)).all()
    step = (train[1:] - train[:-1]).pow(2).mean()
    spread = (train - train.mean(0)).pow(2).mean()
    assert float(step) < float(spread)


@needs_openmm
def test_the_torsion_panel_separates_the_train_split_from_the_reference():
    """Divergences run on a coarse grid so they beat their own self-comparison floor"""
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme')
    reference = reference_ensemble(target, n=8000)
    train = target.train_samples(n=200)

    floor = target.compute_metrics(reference[:4000], ref_samples=reference[4000:],
                                   compute_standard_metrics=False)
    biased = target.compute_metrics(train, ref_samples=reference[:4000],
                                    compute_standard_metrics=False)
    assert floor['js_ramachandran'] < 0.05
    assert biased['js_ramachandran'] > 2 * floor['js_ramachandran']
    assert 0.0 < floor['ramachandran_occupancy'] <= 1.0


@needs_openmm
def test_the_panel_returns_plot_grade_histograms():
    """Divergences use a coarse grid; the returned histograms keep the corpus's 200 x 200 one"""
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme')
    reference = reference_ensemble(target, n=2000)
    ret, hist = target.compute_metrics(reference[:1000], ref_samples=reference[1000:],
                                       compute_standard_metrics=False, return_hist=True)
    assert hist['torsions']['ramachandran']['hist'].shape == (200, 200)
    assert hist['torsions']['edges'].shape == (201,)
    assert abs(float(hist['torsions']['phi1']['ref_hist'].sum()) - 1.0) < 1e-12
    assert {'energy_total_var', 'energy_out_of_range', 'energy_w1'} <= set(ret)

    # The structural energy histogram and the upstream one both come back, under their own keys
    assert set(hist) == {'torsions', 'dist', 'energy', 'upstream'}
    assert set(hist['upstream']) == {'energy', 'distances', 'com_norm', 'ramachandran', 'tica'}

    # The structural energy edges span the reference alone, no published range existing for this
    # target, where the upstream ones span both sets
    assert target.energy_histogram_range is None
    ref_energy = target.energy(reference[1000:]).detach()
    energy = target.energy(reference[:1000]).detach()
    edges, upstream_edges = hist['energy']['edges'], hist['upstream']['energy']['edges']
    assert abs(float(edges[0]) - float(ref_energy.min())) < 1e-9
    assert abs(float(edges[-1]) - float(ref_energy.max())) < 1e-9
    assert abs(float(upstream_edges[0]) - float(torch.cat([energy, ref_energy]).min())) < 1e-9
    assert abs(float(upstream_edges[-1]) - float(torch.cat([energy, ref_energy]).max())) < 1e-9


@needs_openmm
@pytest.mark.parametrize('sequence', list(PEPTIDE_SYSTEMS))
def test_the_tica_features_are_laid_out_the_way_upstream_fits_them(sequence):
    """C/N/S distances, then sin and cos of phi, of psi and of omega, one family at a time

    The layout matters beyond bookkeeping: a projection is only meaningful for the ordering it was
    fitted on, and reordering the columns keeps the shape, so nothing downstream would complain.
    """
    if not has_pdb(sequence):
        pytest.skip('the released PDB is not cached')
    target = peptide(sequence)
    x = torch.as_tensor(target.reference_positions.reshape(1, -1), dtype=target.dtype)
    x = torch.cat([x, x + 0.01, x - 0.02])
    features = target.tica_features(x)

    n_distances = len(target.tica_pair_rows)
    n_angles = target.tica_torsions.shape[0]
    assert features.shape == (3, n_distances + 2 * n_angles)
    assert sum(target.tica_angle_splits) == n_angles

    # The distances are the upper triangle of the selected subset, in that order
    positions = target.positions(x)[:, target.tica_selection]
    expected = torch.linalg.norm(positions[:, target.tica_pair_rows]
                                 - positions[:, target.tica_pair_cols], dim=-1)
    assert torch.allclose(features[:, :n_distances], expected)

    # And each family contributes its sines before its cosines, so the two halves square to one
    start = n_distances
    for size in target.tica_angle_splits:
        block = features[:, start:start + 2 * size]
        assert torch.allclose(block[:, :size] ** 2 + block[:, size:] ** 2,
                              torch.ones_like(block[:, :size]))
        start += 2 * size

    # Everything is rotation and translation invariant, being distances and dihedrals
    theta = torch.tensor(0.7, dtype=target.dtype)
    rotation = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.0],
                             [torch.sin(theta), torch.cos(theta), 0.0],
                             [0.0, 0.0, 1.0]], dtype=target.dtype)
    moved = (target.positions(x) @ rotation.T + 3.0).reshape(len(x), -1)
    assert torch.allclose(target.tica_features(moved), features, atol=1e-9)


@needs_openmm
def test_the_ramachandran_pairs_belong_to_one_residue_each():
    """A phi and a psi only make a Ramachandran plot when they share a residue"""
    if not has_pdb('AAAAAA'):
        pytest.skip('the released PDB is not cached')
    import mdtraj

    target = peptide('AAAAAA')
    topology = mdtraj.Topology.from_openmm(target.topology)
    frame = mdtraj.Trajectory(np.zeros((1, target.n_atoms, 3)), topology)
    phi_indexes, psi_indexes = mdtraj.compute_phi(frame)[0], mdtraj.compute_psi(frame)[0]

    assert len(target.ramachandran_pairs) > 0
    for phi_column, psi_column in target.ramachandran_pairs.tolist():
        # The columns index torsions(), which is every phi and then every psi
        assert phi_column < target.n_phi <= psi_column

        # mdtraj's phi is C(i-1)-N(i)-CA(i)-C(i) and its psi is N(i)-CA(i)-C(i)-N(i+1), so the
        # pair shares the N, CA and C of the residue the plot is about
        phi_atoms = phi_indexes[phi_column]
        psi_atoms = psi_indexes[psi_column - target.n_phi]
        assert list(phi_atoms[1:]) == list(psi_atoms[:3])
        residues = {topology.atom(int(index)).residue.index for index in phi_atoms[1:]}
        assert len(residues) == 1


@needs_openmm
def test_the_upstream_panel_separates_the_train_split_from_the_reference():
    """transferable-samplers' six scalars, each against its own reference-versus-itself floor"""
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')
            and has_split('Ace-A-Nme', 'train')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme')
    reference = reference_ensemble(target, n=4000)
    train = target.train_samples(n=2000)

    names = ('energy_w1', 'energy_w2', 'torus_w2', 'tica_w2', 'tica_k_jsd', 'torus_k_jsd')
    floor = target.compute_upstream_metrics(reference[:2000], reference[2000:])
    biased = target.compute_upstream_metrics(train, reference[:2000])
    assert set(names) <= set(floor)
    for name in names:
        assert floor[name] >= 0.0
        assert biased[name] > floor[name], name

    # Transport is capped at max_ot_samples, and the panel says how many points it used
    assert floor['n_ot_samples'] == 2000
    assert target.compute_upstream_metrics(train, reference,
                                           max_ot_samples=256)['n_ot_samples'] == 256

    # Weights are honoured: putting all the mass on the first half matches passing that half
    weights = torch.zeros(2000, dtype=target.dtype)
    weights[:1000] = 1.0
    weighted = target.compute_upstream_metrics(train, reference[:2000], weights=weights)
    half = target.compute_upstream_metrics(train[:1000], reference[:2000])
    for name in ['energy_w1', 'energy_w2']:
        assert abs(weighted[name] - half[name]) < 1e-9, name

    # The transport rows are drawn by the weights wherever they sit. With the train rows first and
    # 1000 rows per side, the mass on the reference rows behind them reads 0.17, 0.20 and 0.008,
    # near the 1000-row floor, and the mass on the train rows 1.63, 0.45 and 0.021
    mixed = torch.cat([train[:1000], reference[2000:3000]])
    on_train = torch.cat([torch.ones(1000), torch.zeros(1000)]).to(target.dtype)
    on_reference, on_train = [
        target.compute_upstream_metrics(mixed, reference[:2000], weights=weights,
                                        max_ot_samples=1000, generator=generator(0))
        for weights in [1.0 - on_train, on_train]
    ]
    assert on_reference['n_ot_samples'] == 1000
    for name in ['torus_w2', 'tica_w2', 'tica_k_jsd']:
        assert on_reference[name] < 0.75 * on_train[name], name


@needs_openmm
def test_the_upstream_panel_returns_its_histograms():
    """The plots that go next to the scalars: energy, distances, com, Ramachandran and TICA"""
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    from samplingdb.metrics.molecular import N_BINS

    target = peptide('Ace-A-Nme')
    reference = reference_ensemble(target, n=2000)
    _, hist = target.compute_upstream_metrics(reference[:1000], reference[1000:],
                                              return_hist=True)
    assert set(hist) == {'energy', 'distances', 'com_norm', 'ramachandran', 'tica'}
    for name in ['energy', 'distances', 'com_norm']:
        assert hist[name]['hist'].shape == (N_BINS,)
        assert hist[name]['edges'].shape == (N_BINS + 1,)

        # Every histogram in the panel is a probability, and the shared edges span both sets, so
        # no mass is dropped on the way out
        for key in ['hist', 'ref_hist']:
            assert abs(float(hist[name][key].sum()) - 1.0) < 1e-9

    # One Ramachandran per residue, on angle edges, and the TICA plane as a joint histogram
    rama = hist['ramachandran']
    assert len(rama['hist']) == len(rama['ref_hist']) == len(target.ramachandran_pairs)
    assert len(rama['residues']) == len(target.ramachandran_pairs)
    assert rama['hist'][0].shape == (N_BINS, N_BINS)
    assert abs(float(rama['hist'][0].sum()) - 1.0) < 1e-9
    assert hist['tica']['hist'].shape == (N_BINS, N_BINS)
    assert abs(float(hist['tica']['hist'].sum()) - 1.0) < 1e-9


@needs_openmm
def test_the_tica_projection_is_cached_under_its_feature_layout():
    """A cached projection is only valid for the ordering it was fitted on, so the name says it"""
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    from samplingdb.targets.molecular import TICA_LAYOUT

    target = peptide('Ace-A-Nme')
    model = target.tica_model()
    path = cache_dir() / 'tica' / (target.name + '-' + TICA_LAYOUT + '-test.npz')
    assert path.exists()

    # It is a projection onto two components of the feature space, applied where the samples are
    x = reference_ensemble(target, n=16)
    features = target.tica_features(x)
    assert model.projection.shape == (features.shape[1], 2)
    assert model(features).shape == (16, 2)
    assert torch.allclose(model(features), (features - model.mean) @ model.projection)


@needs_openmm
def test_every_peptide_has_one_catalogued_name():
    """The catalog name is the lower-cased sequence and the temperature, and nothing else builds it"""
    from samplingdb import CATALOG, Peptide, build

    names = [name for name in CATALOG if CATALOG[name]['cls'] is Peptide]
    assert sorted(names) == sorted(sequence.lower() + '-T' + str(int(system['temperature']))
                                   for sequence, system in PEPTIDE_SYSTEMS.items())
    for name in names:
        sequence = CATALOG[name]['kwargs']['sequence']
        if has_pdb(sequence):
            built = build(name, compile=False)
            assert built.name == name
            assert built.family is PEPTIDE_SYSTEMS[sequence]['family']


@needs_openmm
def test_test_samples_is_the_generated_ensemble():
    """Once the replica-exchange ensemble exists it is the reference, never the released split"""
    if not has_pdb('Ace-A-Nme'):
        pytest.skip('the released PDB is not cached')
    target = peptide('Ace-A-Nme')
    if not target.reference_path().exists():
        pytest.skip('the replica-exchange ensemble is not cached')
    ours = target.test_samples(n=64)
    assert ours.shape == (64, target.dim)
    assert torch.equal(ours, reference_ensemble(target, n=64))
    if has_split('Ace-A-Nme'):
        assert not torch.equal(ours, target.load_split('test', n=64))


@needs_openmm
@pytest.mark.parametrize('sequence', ['Ace-A-Nme', 'AAA'])
def test_float32_agrees_with_float64_on_real_conformations(sequence):
    """The second link of the validation chain: float64 against OpenMM, float32 against float64

    The library works in float32 as of 2026-08-25. The claim it replaced -- that float32 carried
    0.15 to 0.8 kT and so moved an importance weight by a factor of two -- was measured when the
    nonbonded sum kept every pair and subtracted the excluded ones back out. A bonded pair sits well
    inside sigma, so its (sigma/r)^12 is enormous and adding it in destroys the mantissa of every
    physical term before the subtraction can undo it; float64 survives that and float32 does not.
    Exceptions are excluded from the pair list here, and nobody had re-measured since.

    The bound is 2e-3 kT, which is loose against the 1.2e-04 and 2.4e-04 actually measured and still
    forty times below the 0.08 kT bias of the 1 fs timestep the reference sets are generated with.
    """
    if not (has_pdb(sequence) and has_split(sequence)):
        pytest.skip('the released trajectory of ' + sequence + ' is not cached')
    exact = peptide(sequence)
    single = peptide(sequence, dtype=torch.float32)
    assert exact.dtype == torch.float64 and single.dtype == torch.float32

    x = exact.load_split('test', n=512)
    energy = exact.energy(x)
    error = (energy - single.energy(x.to(torch.float32)).to(torch.float64)).abs()
    assert torch.isfinite(error).all()
    assert float(error.max()) < 2e-3, float(error.max())

    # The force is what the dynamics integrates, so it is bounded against its own scale
    _, force = exact.bridge.evaluate(x)
    _, approximate = single.bridge.evaluate(x.to(torch.float32))
    difference = (force - approximate.to(torch.float64)).abs().max()
    assert float(difference / force.pow(2).mean().sqrt()) < 1e-3

    # Halving the width doubles the chunk the force field derives for itself
    assert single.bridge.chunk_size == 2 * exact.bridge.chunk_size


@needs_openmm
def test_a_coincident_pair_stays_finite_in_both_precisions():
    """(sigma/r)^12 leaves float32's range around r = 4e-4 nm, and the guard has to catch the FORCE

    Two ways to get this wrong, both measured. Clamping the sixth power does not work: it has
    already overflowed to inf, and clamp passes zero gradient above its ceiling, so the backward is
    0 * inf = NaN. And a bound taken from base^12 keeps the energy finite while the gradient still
    overflows, because the derivative carries a further 12 / r -- at r = 4e-4 nm the energy is
    6.3e34 and the gradient 1.1e39, past float32's 3.4e38. The bound comes from base^13.
    """
    from samplingdb.targets.forcefield import MIN_SQUARE
    sigma, epsilon = 0.3, 0.5

    for dtype in [torch.float64, torch.float32]:
        for separation in [1e-2, 1e-3, 4e-4, 1e-4, 1e-5, 0.0]:
            a = torch.zeros(1, 3, dtype=dtype, requires_grad=True)
            b = torch.full((1, 3), separation / 3 ** 0.5, dtype=dtype)
            r = torch.sqrt(((a - b) ** 2).sum(-1) + MIN_SQUARE)
            base = (sigma / r).clamp(max=0.25 * torch.finfo(dtype).max ** (1.0 / 13.0))
            ratio = base ** 6
            energy = (4 * epsilon * ratio * (ratio - 1)).sum()
            (gradient,) = torch.autograd.grad(energy, a)
            assert torch.isfinite(energy), (dtype, separation)
            assert torch.isfinite(gradient).all(), (dtype, separation)

    # And the guard never binds on anything physical: the closest approach is ~0.05 nm
    bound = 0.25 * torch.finfo(torch.float32).max ** (1.0 / 13.0)
    assert bound > 20 * (0.35 / 0.05)


@needs_openmm
@pytest.mark.parametrize('sequence', ['Ace-A-Nme', 'AAA'])
def test_a_mirror_image_sampler_is_scored_as_exact(sequence):
    """Inversion leaves every Amber term unchanged, so the panel maps mirror images back first

    Every torsion phase in both families is 0 or pi, so cos(n phi - phase) is even in phi and the
    energy is invariant under x -> -x although every torsion changes sign.
    """
    if not (has_pdb(sequence) and has_split(sequence)):
        pytest.skip('the released trajectory of ' + sequence + ' is not cached')
    target = peptide(sequence)
    x = reference_ensemble(target, n=256)
    assert len(target.chirality_centres) > 0
    assert torch.allclose(target.energy(-x), target.energy(x), atol=1e-8)

    # The reference has one handedness, and its mirror image the other at every centre
    signs = target.chirality_signs(x)
    assert bool((signs == signs[:1]).all())
    assert bool((target.chirality_signs(-x) == -signs).all())

    # Half the samples mirrored: the fix restores them exactly and reports the half it fixed
    mixed = torch.cat([x[:128], -x[128:]])
    fixed, rates = target.fix_chirality(mixed, x)
    assert torch.equal(fixed, x)
    assert abs(rates['correct_chirality_rate'] - 0.5) < 1e-12
    assert rates['uncorrectable_chirality_rate'] == 0.0

    # So the panel cannot tell the mirrored set from the reference it came from
    plain = target.compute_metrics(x[:128], ref_samples=x[128:], compute_standard_metrics=False)
    mirrored = target.compute_metrics(-x[:128], ref_samples=x[128:],
                                      compute_standard_metrics=False)
    for name, value in plain.items():
        if name != 'correct_chirality_rate':
            assert abs(mirrored[name] - value) < 1e-9, name
    assert mirrored['correct_chirality_rate'] == 0.0
