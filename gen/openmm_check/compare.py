# Check samplingdb's torch force field against OpenMM, term by term and platform by platform

# Libraries
import argparse, os, pprint, sys, time
import multiprocessing as mp
import numpy as np
import torch
from samplingdb.targets.molecular import PEPTIDE_SYSTEMS, Peptide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openmm_bridge import OpenMMBridge

# Every OpenMM platform worth timing, with the properties that define what it actually computes.
# 'mixed' is what boltzkit asks for and is NOT double precision; it is here to be honest about the
# speed such a choice buys.
PLATFORMS = [
    ('Reference', {}),
    ('CPU 1 thread', {'Threads': '1'}),
    ('CUDA double', {'Precision': 'double'}),
    ('CUDA mixed', {'Precision': 'mixed'}),
    ('OpenCL double', {'Precision': 'double'}),
]

# One Context per process, which is how boltzkit's ParallelEnergyEval parallelises
_worker = {}


def _init_worker(system, platform_name, properties):
    """Builds this worker's Context once"""
    import openmm

    _worker['context'] = openmm.Context(system, openmm.VerletIntegrator(0.001),
                                        openmm.Platform.getPlatformByName(platform_name),
                                        properties)


def _eval_worker(positions):
    """Evaluates one chunk of configurations in this worker"""
    context = _worker['context']
    energy = np.empty(len(positions))
    force = np.empty_like(positions)
    for index in range(len(positions)):
        context.setPositions(positions[index])
        state = context.getState(getEnergy=True, getForces=True)
        energy[index] = state.getPotentialEnergy()._value
        force[index] = state.getForces(asNumpy=True)._value
    return energy, force


def compare_terms(target, n_frames):
    """Compares the torch energy against OpenMM force group by force group

    Args:
        * target (Peptide): The target to check
        * n_frames (int): Number of reference-set frames to score

    Returns:
        * errors (dict): Maximum absolute deviation per term and of the total energy and force, in
            kJ/mol and kJ/mol/nm, and the rms of OpenMM's energies and force components
    """
    import openmm

    system = target.system
    for index in range(system.getNumForces()):
        system.getForce(index).setForceGroup(index)
    names = [system.getForce(i).__class__.__name__ for i in range(system.getNumForces())]
    groups = {
        'bonded': [names.index(n) for n in ['HarmonicBondForce', 'HarmonicAngleForce',
                                            'PeriodicTorsionForce']],
        'nonbonded': [names.index('NonbondedForce')],
        'solvent': [names.index('GBSAOBCForce' if 'GBSAOBCForce' in names else 'CustomGBForce')],
    }

    frames = target.test_samples(n=n_frames)
    positions = frames.numpy().reshape(len(frames), target.n_atoms, 3).astype(np.float64)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName('Reference'))

    reference = {key: [] for key in groups}
    reference_energy, reference_force = [], []
    for index in range(len(positions)):
        context.setPositions(positions[index])
        for key, indexes in groups.items():
            state = context.getState(getEnergy=True, groups=set(indexes))
            reference[key].append(state.getPotentialEnergy()._value)
        state = context.getState(getEnergy=True, getForces=True)
        reference_energy.append(state.getPotentialEnergy()._value)
        reference_force.append(state.getForces(asNumpy=True)._value.reshape(-1))

    # The same three groups out of the torch force field, in kJ/mol rather than kT
    field = target.bridge
    x = frames.clone().requires_grad_(True)
    coordinates = x.reshape(len(x), target.n_atoms, 3)
    mine = {'bonded': field.bonded(coordinates), 'nonbonded': field.nonbonded(coordinates),
            'solvent': field.generalized_born(coordinates)}
    energy, force = target.bridge.evaluate(x)

    errors = {key: float(np.abs(np.array(reference[key])
                                - mine[key].detach().numpy()).max()) for key in groups}
    errors['energy'] = float(np.abs(np.array(reference_energy) / field.kT
                                    - energy.detach().numpy()).max() * field.kT)
    errors['force'] = float(np.abs(np.array(reference_force) / field.kT
                                   - force.detach().numpy()).max() * field.kT)
    errors['energy_rms'] = float(np.sqrt((np.array(reference_energy) ** 2).mean()))
    errors['force_rms'] = float(np.sqrt((np.array(reference_force) ** 2).mean()))
    return errors


def time_openmm(system, positions, platform_name, properties, n_rep):
    """Returns seconds per configuration for OpenMM's sequential energy + force evaluation"""
    import openmm

    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName(platform_name), properties)
    context.setPositions(positions[0])
    for _ in range(10):
        state = context.getState(getEnergy=True, getForces=True)
        state.getPotentialEnergy()._value, state.getForces(asNumpy=True)._value

    t0 = time.perf_counter()
    for index in range(n_rep):
        context.setPositions(positions[index % len(positions)])
        state = context.getState(getEnergy=True, getForces=True)
        state.getPotentialEnergy()._value
        state.getForces(asNumpy=True)._value
    return (time.perf_counter() - t0) / n_rep


def time_pool(system, positions, platform_name, properties, n_workers, n_total):
    """Returns seconds per configuration for a boltzkit-style pool, pool construction excluded"""
    context = mp.get_context('spawn')
    pool = context.Pool(processes=n_workers, initializer=_init_worker,
                        initargs=(system, platform_name, properties))
    batch = np.repeat(positions, max(1, n_total // len(positions)), axis=0)[:n_total]
    chunks = np.array_split(batch, n_workers)
    pool.map(_eval_worker, [chunk[:2] for chunk in chunks])

    t0 = time.perf_counter()
    pool.map(_eval_worker, chunks)
    elapsed = time.perf_counter() - t0
    pool.close()
    pool.join()
    return elapsed / len(batch)


def time_torch(target, batch, device, compile_it, n_rep=20):
    """Returns seconds per configuration for one batched energy + score pass

    The whole batch goes in at once, which is the entire point: chunk_size is raised so the
    comparison measures the potential rather than the memory guard.
    """
    field = target.to(device).bridge
    field.chunk_size = len(batch)
    field.compiled = torch.compile(field.potential) if compile_it else None
    x = batch.to(device)

    for _ in range(5):
        field.evaluate(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        field.evaluate(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_rep / len(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', type=str, nargs='+', default=list(PEPTIDE_SYSTEMS))
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--n_workers', type=int, default=os.cpu_count())
    parser.add_argument('--n_rep', type=int, default=200)
    parser.add_argument('--skip_timing', action='store_true')
    args = parser.parse_args()

    config = vars(args)
    pprint.pprint(config)

    # 1. Agreement, which is the part that has to hold before any timing means anything. It is
    # checked in float64, against the Reference platform, which is double precision throughout:
    # the largest energy error within 1e-6 of the rms energy and the largest force error within
    # 1e-4 of the rms force component
    print()
    print('%-12s %6s %12s %12s %12s %12s %12s %12s %12s'
          % ('sequence', 'atoms', 'bonded', 'nonbonded', 'solvent', 'energy', 'force',
             'energy rel', 'force rel'))
    targets = {}
    for sequence in args.sequences:
        target = Peptide(sequence=sequence, dtype=torch.float64)
        targets[sequence] = target
        errors = compare_terms(target, args.n_frames)
        energy_error = errors['energy'] / errors['energy_rms']
        force_error = errors['force'] / errors['force_rms']
        print('%-12s %6d %12.2e %12.2e %12.2e %12.2e %12.2e %12.2e %12.2e'
              % (sequence, target.n_atoms, errors['bonded'], errors['nonbonded'],
                 errors['solvent'], errors['energy'], errors['force'], energy_error, force_error))
        assert energy_error < 1e-6, '%s: the energies differ by %.2e of their rms' % (
            sequence, energy_error)
        assert force_error < 1e-4, '%s: the forces differ by %.2e of their rms' % (
            sequence, force_error)

    # 2. That the score survives being differentiated through, which the OpenMM bridge never did
    print()
    for sequence in args.sequences:
        target = targets[sequence]
        target.bridge.differentiable_score = True
        x = target.test_samples(n=4).clone().requires_grad_(True)
        _, force = target.bridge.evaluate(x)
        (second,) = torch.autograd.grad(force.pow(2).sum(), x)
        target.bridge.differentiable_score = False
        bridge = OpenMMBridge(target.system, target.temperature)
        y = target.test_samples(n=4).clone().requires_grad_(True)
        openmm_force = bridge.evaluate(y)[1]
        print('%-12s double backward finite %-6s max %.3e, and the OpenMM bridge gives a force '
              'with requires_grad %s'
              % (sequence, bool(torch.isfinite(second).all()), float(second.abs().max()),
                 openmm_force.requires_grad))

    if args.skip_timing:
        return

    # 3. Throughput, every OpenMM platform against the batched torch path, which is timed at the
    # library's working dtype, float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = [label for label, _ in PLATFORMS] + ['CPU pool x%d' % args.n_workers,
                                                  'torch eager', 'torch compiled']
    rates = {}
    for sequence in args.sequences:
        target = Peptide(sequence=sequence)
        frames = target.test_samples(n=64)
        positions = frames.numpy().reshape(len(frames), target.n_atoms, 3).astype(np.float64)
        batch = frames.repeat(max(1, args.batch // len(frames)), 1)

        row = {}
        for label, properties in PLATFORMS:
            row[label] = 1 / time_openmm(target.system, positions, label.split()[0], properties,
                                         args.n_rep)
        row['CPU pool x%d' % args.n_workers] = 1 / time_pool(
            target.system, positions, 'CPU', {'Threads': '1'}, args.n_workers, 8192)
        row['torch eager'] = 1 / time_torch(target, batch, device, False)
        eager = target.bridge.evaluate(batch.to(device))
        row['torch compiled'] = 1 / time_torch(target, batch, device, True)
        compiled = target.bridge.evaluate(batch.to(device))

        # Compiling must return the same number, not merely a faster one
        assert torch.allclose(eager[0], compiled[0], rtol=0, atol=1e-7), sequence
        assert torch.allclose(eager[1], compiled[1], rtol=0, atol=1e-6), sequence
        rates[sequence] = row

    print()
    print('configurations per second, energy + forces, %d cores / %s'
          % (os.cpu_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))
    print('%-22s' % 'engine' + ''.join('%14s' % s for s in args.sequences))
    print('%-22s' % 'atoms' + ''.join('%14d' % targets[s].n_atoms for s in args.sequences))
    for label in labels:
        print('%-22s' % label + ''.join('%14.0f' % rates[s][label] for s in args.sequences))


if __name__ == '__main__':
    main()
