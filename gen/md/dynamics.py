# Molecular-dynamics pieces of the peptide generator: the batched integrator and the solute split

# Libraries
import math
import torch
from samplingdb.targets.molecular import center

# The harmonic term of each peptide, compiled once when the peptide's own potential is
COMPILED = {}


def masses(target):
    """Returns the particle masses of a peptide in amu, which only the integrator needs"""
    return torch.as_tensor([target.system.getParticleMass(index)._value
                            for index in range(target.n_atoms)], dtype=target.dtype,
                           device=target.device)


def harmonic_log_prob(target, x):
    """Evaluates the log-density of the bond and angle terms alone, in kT

    This is the beta = 0 end of a SOLUTE-TEMPERING path: pair it with nrpt.PotentialReference and
    the linear path becomes -(E_harmonic + beta * E_soft) / kT, which scales the torsions, the
    nonbonded terms and the Born solvation and leaves the stiff internal coordinates alone. See
    TorchForceField.harmonic for why that is the split worth making.

    Args:
        * target (Peptide): The peptide
        * x (torch.Tensor of shape (n, dim)): Positions, in nanometres

    Returns:
        * log_prob (torch.Tensor of shape (n,)): -E_harmonic / kT
    """
    if target not in COMPILED:
        COMPILED[target] = torch.compile(target.bridge.harmonic) \
            if target.bridge.compiled is not None else target.bridge.harmonic
    positions = target.positions(x).to(target.dtype)
    return torch.cat([-COMPILED[target](chunk) / target.bridge.kT
                      for chunk in positions.split(target.bridge.chunk_size)])


def harmonic_score(target, x):
    """Evaluates the gradient of harmonic_log_prob at x, in 1 / nm"""
    with torch.enable_grad():
        inputs = x.detach().requires_grad_(True)
        (gradient,) = torch.autograd.grad(harmonic_log_prob(target, inputs).sum(), inputs)
    return gradient


def minimised_positions(target, x0=None):
    """Returns the energy-minimised structure, which is where a trajectory has to start

    WARNING: target.reference_positions is the shipped pdb verbatim, and a pdb is not a minimum of
    this force field. On a ten-residue peptide it sat at E/kT = 9.0e5 against an ensemble at
    -453 +- 12, and run_batched_dynamics from there does NOT raise -- linlogcut turns the energy
    above energy_cut into a logarithm, so the molecule flies apart to |x| ~ 2.6e4 nm inside 200 fs
    while every number stays finite and the run writes a full file of it. This is the minimisation OpenMM's
    Simulation runs before its own dynamics.

    Args:
        * target (Peptide): The peptide
        * x0 (numpy.ndarray of shape (n_atoms, 3) or None): Starting positions in nanometres;
            None uses reference_positions (default is None)

    Returns:
        * x (torch.Tensor of shape (dim,)): The minimised structure, in nanometres
    """
    import openmm

    if x0 is None:
        x0 = target.reference_positions
    simulation = openmm.app.Simulation(target.topology, target.system,
                                       openmm.VerletIntegrator(0.001),
                                       openmm.Platform.getPlatformByName('Reference'))
    simulation.context.setPositions(x0)
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)._value.reshape(-1)
    return torch.as_tensor(positions, dtype=target.dtype, device=target.device)


def run_batched_dynamics(target, x0, n_steps, save_interval=None, step_size=0.001,
                         generator=None, return_velocities=False, score=None, velocities=None):
    """Runs Langevin dynamics on a whole batch of replicas at once, in torch

    The integrator is BAOAB -- position update, Ornstein-Uhlenbeck velocity update, position
    update, both halves of the force kick around them -- which is what OpenMM's
    LangevinMiddleIntegrator is, so this samples the same law at the same friction and timestep.
    The difference is that a batch of replicas advances in one set of
    tensor operations instead of one Context per replica, which is what makes generating a
    reference ensemble affordable: on Ace-A-Nme the force field does ~4.7e5 configurations per
    second compiled, so 1024 replicas advance at ~0.46 ps of their own time per second.

    Units are OpenMM's, in which a force in kJ/mol/nm divided by a mass in amu is directly an
    acceleration in nm/ps^2: positions nm, velocities nm/ps, masses amu, friction 1/ps,
    step_size ps. The force is kT times the score, the score being the gradient of -E/kT.

    WARNING: this is dynamics, not a Metropolis chain, so a timestep too large biases the
    distribution silently rather than showing up as a rejection rate. 1 fs is the released
    protocol's own timestep and is what the reference generator uses; check any change against
    the energy of the released frames.

    Args:
        * target (Peptide): The peptide, whose friction and masses are the protocol's
        * x0 (torch.Tensor of shape (n_replicas, dim)): Starting positions, in nanometres
        * n_steps (int): Number of integration steps
        * save_interval (int or None): Steps between kept frames; None keeps only the last
            state (default is None)
        * step_size (float): Timestep in picoseconds (default is 0.001, i.e. 1 fs)
        * generator (torch.Generator or None): RNG of the velocities and the noise
            (default is None)
        * return_velocities (bool): Whether to also return the final velocities, so a run can
            be continued exactly where it stopped (default is False)
        * velocities (torch.Tensor of the shape of x0, or None): Starting velocities; None
            draws them from the Maxwell distribution at this system's temperature. Passing
            the previous call's velocities back in is what makes a sequence of short calls
            the same dynamics as one long one -- redrawing them every call is a momentum
            refresh, i.e. an Andersen thermostat at 1 / (n_steps * step_size), which at 100 fs
            per call is 10/ps against this system's own 0.3/ps (default is None)
        * score (function or None): Score to integrate against; None uses this target's own.
            A TEMPERED score turns this into the local kernel of a replica-exchange run: the
            force becomes kT * beta * score, which is the gradient of the potential beta * E,
            and the thermostat stays at kT, so the stationary law is exp(-beta E / kT) --
            the same annealed density parallel tempering asks for. Scaling the potential
            rather than the thermostat is what lets one call serve every level
            (default is None)

    Returns:
        * x (torch.Tensor of shape (n_replicas, dim), or (n_frames, n_replicas, dim) when
            save_interval is given): The trajectory
        if return_velocities:
            * velocities (torch.Tensor of shape (n_replicas, dim)): Final velocities
    """
    score = target.score if score is None else score
    mass = masses(target).to(x0).repeat_interleave(3)
    thermal = torch.sqrt(target.kT / mass)
    friction = math.exp(-target.friction * step_size)
    noise_scale = math.sqrt(1.0 - friction ** 2)

    def draw(shape):
        values = torch.randn(shape, generator=generator, dtype=target.dtype)
        return values.to(x0.device)

    x = x0.clone()
    velocity = thermal * draw(x.shape) if velocities is None else velocities.clone()
    force = target.kT * score(x).detach()
    frames = []

    for step in range(n_steps):
        velocity = velocity + 0.5 * step_size * force / mass
        x = x + 0.5 * step_size * velocity
        velocity = friction * velocity + noise_scale * thermal * draw(x.shape)
        x = x + 0.5 * step_size * velocity
        force = target.kT * score(x).detach()
        velocity = velocity + 0.5 * step_size * force / mass
        if save_interval is not None and (step + 1) % save_interval == 0:
            frames.append(center(x, target.n_atoms))

    out = torch.stack(frames) if save_interval is not None else center(x, target.n_atoms)
    if return_velocities:
        return out, velocity
    return out
