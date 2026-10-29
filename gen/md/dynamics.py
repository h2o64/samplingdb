# Molecular-dynamics pieces of the peptide generator: the integrator wrapper and the solute split

# Libraries
import torch
from samplingdb.targets.kernels import langevin
from samplingdb.targets.molecular import center

# The harmonic term of each peptide, compiled once when the peptide's own potential is
COMPILED = {}


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


def run_batched_dynamics(target, x0, n_steps, save_interval=None, step_size=0.001,
                         generator=None, return_velocities=False, score=None, velocities=None):
    """Runs Langevin dynamics on a whole batch of replicas at once, in torch

    The integrator is samplingdb.targets.kernels.langevin, BAOAB -- position update,
    Ornstein-Uhlenbeck velocity update, position update, both halves of the force kick around them
    -- which is what OpenMM's LangevinMiddleIntegrator is, so this samples the same law at the same
    friction and timestep. The difference is that a batch of replicas advances in one set of
    tensor operations instead of one Context per replica, which is what makes generating a
    reference ensemble affordable. This wrapper adds the peptide's own masses, friction and
    temperature, and centres the frames.

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
    mass = target.masses.to(x0).repeat_interleave(3)
    force = lambda x: target.kT * score(x).detach()
    out = langevin(force, x0, n_steps, step_size, mass, target.kT, target.friction,
                   generator=generator, return_trace=save_interval is not None,
                   thin=save_interval or 1, velocities=velocities,
                   return_velocities=return_velocities)
    if return_velocities:
        out, velocity = out
    out = center(out.reshape(-1, target.dim), target.n_atoms).reshape(out.shape)
    if return_velocities:
        return out, velocity
    return out
