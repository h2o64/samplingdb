# The retired OpenMM bridge, kept as the reference the torch force field is checked against

# Libraries
import numpy as np
import torch
from samplingdb.targets.forcefield import GAS_CONSTANT, linlogcut


class _OpenMMEnergy(torch.autograd.Function):
    """Evaluates an OpenMM energy in the forward pass and its forces in the backward one

    NOTE: this is exactly why the bridge was retired. A custom Function that hands back a saved
    tensor has no double backward, so a loss built on the score could never be differentiated.
    """

    @staticmethod
    def forward(ctx, x, bridge):
        energy, force = bridge.evaluate(x)
        ctx.save_for_backward(force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        (force,) = ctx.saved_tensors
        return grad_output[:, None] * (-force), None


class OpenMMBridge:
    """Holds one OpenMM Context and evaluates batches of energies and forces through it

    Positions are in nanometres and energies in kT, matching every reference implementation in
    this corpus (transferable-samplers' OpenMMEnergy, bgflow's OpenMMBridge).

    WARNING: this was the slow part of the whole library, and is no longer what the targets use.
    A single Context evaluates one configuration at a time -- OpenMM has no batched API -- so a
    molecular target cost a Python loop per batch, with no two energies ever computed at the same
    time. It lives on here as the reference compare.py checks samplingdb's TorchForceField
    against, and as the thing whose throughput compare.py measures.
    """

    def __init__(self, system, temperature, platform_name='Reference', n_threads=1,
                 energy_cut=1e8, energy_max=1e20):
        """Builds the bridge

        Args:
            * system (openmm.System): The system to evaluate
            * temperature (float): Temperature in kelvin, which fixes kT
            * platform_name (str): OpenMM platform, 'Reference', 'CPU', 'CUDA' or 'OpenCL'
                (default is 'Reference')
            * n_threads (int): Threads for the CPU platform (default is 1)
            * energy_cut (float): linlogcut's high_val, in kT (default is 1e8)
            * energy_max (float): linlogcut's max_val, in kT (default is 1e20)
        """
        import openmm

        self.system = system
        self.temperature = temperature
        self.n_atoms = system.getNumParticles()
        self.kT = GAS_CONSTANT * temperature
        self.energy_cut = energy_cut
        self.energy_max = energy_max

        # One integrator and one context, kept for the lifetime of the bridge
        integrator = openmm.LangevinIntegrator(
            temperature * openmm.unit.kelvin, 1.0 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )
        platform = openmm.Platform.getPlatformByName(platform_name)
        properties = {'Threads': str(n_threads)} if platform_name == 'CPU' else {}
        self.context = openmm.Context(system, integrator, platform, properties)

    def evaluate(self, x):
        """Evaluates the energy and the force of every configuration in a batch

        Args:
            * x (torch.Tensor of shape (n, n_atoms * 3)): Positions, in nanometres

        Returns:
            * energy (torch.Tensor of shape (n,)): Energies, in kT, after linlogcut
            * force (torch.Tensor of shape (n, n_atoms * 3)): Forces, in kT / nm
        """
        positions = x.detach().cpu().numpy().astype(np.float64).reshape(-1, self.n_atoms, 3)
        energy = np.zeros(len(positions))
        force = np.zeros_like(positions)

        for index in range(len(positions)):
            self.context.setPositions(positions[index])
            state = self.context.getState(getEnergy=True, getForces=True)
            energy[index] = state.getPotentialEnergy()._value
            force[index] = state.getForces(asNumpy=True)._value

        # Convert to kT, soften the unphysical tail, and keep the forces of softened energies finite
        energy = torch.as_tensor(energy / self.kT, dtype=x.dtype, device=x.device)
        force = torch.as_tensor(force.reshape(len(positions), -1) / self.kT, dtype=x.dtype,
                                device=x.device)
        force = torch.where(torch.isfinite(force), force, torch.zeros_like(force))
        softened = linlogcut(energy, self.energy_cut, self.energy_max)
        scale = torch.where(energy >= self.energy_cut, 1.0 / (1 + energy - self.energy_cut),
                            torch.ones_like(energy))
        return softened, force * scale[:, None]

    def energy(self, x):
        """Evaluates the energy in kT, differentiably through the OpenMM forces"""
        return _OpenMMEnergy.apply(x, self)
