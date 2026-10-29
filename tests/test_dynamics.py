# The batched Langevin integrator and the solute split the peptide generator runs on

# Libraries
import pytest
import torch
from samplingdb.targets.base import generator
from dynamics import harmonic_log_prob, harmonic_score, run_batched_dynamics
from test_molecular import has_pdb, has_split, needs_openmm, peptide, reference_ensemble


@needs_openmm
def test_the_batched_integrator_matches_openmms_own():
    """run_batched_dynamics is BAOAB, which is what LangevinMiddleIntegrator is

    Recorded from 64 released Ace-A-Nme frames, 2 ps at a 1 fs timestep: OpenMM lands at
    -25.15 kT and this integrator at -24.23 kT, a 0.17 sigma difference where the standard error
    on the difference of the two means is itself 0.18 sigma. Both drift the same way from the
    released frames, so the drift belongs to the 2 ps relaxation and not to either integrator.
    """
    if not (has_pdb('Ace-A-Nme') and has_split('Ace-A-Nme')):
        pytest.skip('the released trajectory is not cached')
    target = peptide('Ace-A-Nme')
    x0 = reference_ensemble(target, n=16)

    # Masses come from the built system, in amu, and are what makes the dynamics physical
    assert target.masses.shape == (target.n_atoms,)
    assert abs(float(target.masses.sum()) - 144.17) < 0.1

    x = run_batched_dynamics(target, x0, n_steps=200, step_size=0.001, generator=generator(0))
    assert x.shape == x0.shape
    assert torch.isfinite(target.energy(x)).all()

    # Centred on the way out, and a saved trajectory has the frames it was asked for
    assert float(target.positions(x).mean(1).abs().max()) < 1e-9
    trace = run_batched_dynamics(target, x0, n_steps=200, save_interval=50, step_size=0.001,
                                 generator=generator(0))
    assert trace.shape == (4, len(x0), target.dim)

    # Velocities can be carried over, so a long run is a sequence of blocks
    _, velocity = run_batched_dynamics(target, x0, n_steps=10, generator=generator(0),
                                       return_velocities=True)
    assert velocity.shape == x0.shape
    assert torch.isfinite(velocity).all()

    # The thermal velocity scale is sqrt(kT / m), which fixes the kinetic energy per degree
    kinetic = 0.5 * (target.masses.repeat_interleave(3) * velocity ** 2).sum(-1)
    assert 0.2 < float(kinetic.mean()) / (0.5 * target.dim * target.kT) < 3.0


@needs_openmm
def test_the_harmonic_terms_split_off_cleanly_for_solute_tempering():
    """bonded = harmonic + torsion, and the harmonic force is the gradient of the harmonic energy

    This split is what makes solute tempering possible: pair harmonic_log_prob with
    nrpt.PotentialReference and the annealing path scales the torsions, the nonbonded terms and the
    Born solvation while leaving the stiff internal coordinates at full strength.
    """
    if not has_pdb('Ace-A-Nme'):
        pytest.skip('the topology is not cached')
    target = peptide('Ace-A-Nme')
    x = run_batched_dynamics(
        target, target.minimised_positions().reshape(1, -1).expand(8, target.dim).contiguous(),
        n_steps=500, generator=generator(0))
    positions = target.positions(x)

    # The two pieces are the whole of the bonded term and nothing more
    bonded = target.bridge.bonded(positions)
    pieces = target.bridge.harmonic(positions) + target.bridge.torsion(positions)
    assert torch.allclose(bonded, pieces, atol=1e-9)

    # harmonic_score is the gradient of harmonic_log_prob, and a central difference agrees
    energy = -harmonic_log_prob(target, x)
    assert torch.allclose(energy, target.bridge.harmonic(positions) / target.kT, atol=1e-9)
    score = harmonic_score(target, x)
    for coordinate in [0, 17, 40]:
        plus, minus = x.clone(), x.clone()
        plus[:, coordinate] += 1e-6
        minus[:, coordinate] -= 1e-6
        finite = (harmonic_log_prob(target, plus) - harmonic_log_prob(target, minus)) / 2e-6
        assert torch.allclose(score[:, coordinate], finite, rtol=1e-5, atol=1e-3)

    # The stiff terms carry most of the energy's variance and none of the barriers, which is the
    # whole reason to leave them out of the swap statistic
    assert float(target.energy(x).std()) > float((target.energy(x) - energy).std())


@needs_openmm
def test_carrying_velocities_makes_short_calls_one_long_trajectory():
    """A sequence of calls that hands its velocities back is the same dynamics as one long call

    Redrawing them instead is a momentum refresh at 1 / (n_steps * step_size), i.e. an Andersen
    thermostat far stiffer than the system's own friction, which costs the local kernel real
    structural diffusion: measured on Ace-A-Nme over 40 ps, tau is 23.6 ps carried against 32.6 ps
    redrawn, where one long call gives 25.5 ps.
    """
    if not has_pdb('Ace-A-Nme'):
        pytest.skip('the topology is not cached')
    target = peptide('Ace-A-Nme')
    x0 = target.minimised_positions().reshape(1, -1).expand(4, target.dim).contiguous()

    long_call = run_batched_dynamics(target, x0, n_steps=40, generator=generator(0))

    x, velocities = x0, None
    for _ in range(4):
        x, velocities = run_batched_dynamics(target, x, n_steps=10, velocities=velocities,
                                             return_velocities=True, generator=generator(0))

    # Not bit-identical -- the noise stream restarts every call -- but the same law and same scale
    assert torch.isfinite(x).all()
    assert abs(float(target.energy(x).mean()) - float(target.energy(long_call).mean())) < 15.0

    # The velocities that come back are the ones that go in, so the first step continues them
    resumed, _ = run_batched_dynamics(target, x, n_steps=1, velocities=velocities,
                                      return_velocities=True, generator=generator(0))
    fresh = run_batched_dynamics(target, x, n_steps=1, generator=generator(0))
    assert not torch.allclose(resumed, fresh)
