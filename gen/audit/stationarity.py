# Is a released split an equilibrium sample of the density this corpus builds for it

# Libraries
import argparse, pprint
import torch
from samplingdb import Peptide
from samplingdb.targets.molecular import PEPTIDE_SYSTEMS

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, default='Ace-A-Nme', choices=list(PEPTIDE_SYSTEMS))
parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
parser.add_argument('--n_replicas', type=int, default=1024)
parser.add_argument('--total_ps', type=float, default=100.0)
parser.add_argument('--interval_ps', type=float, default=5.0)
parser.add_argument('--step_size', type=float, default=0.001)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Build the target
device = torch.device(args.device)
target = Peptide(sequence=args.sequence).to(device)

# Start ON the released frames, evenly spaced through the file. If they are an equilibrium sample
# of this density then the energy does not move; if they are not, it relaxes, and which way it
# relaxes says which of the two is wrong. This is the test that settled the Lennard-Jones energy
# convention -- the released frames were stationary under one reading of the pair sum and drifted
# by 3.3 sigma under the other -- and it is the only one that compares a density to a file without
# needing either to be right first.
released = target.load_split(args.split).to(device).reshape(-1, target.dim)
step = max(1, len(released) // args.n_replicas)
x = released[::step][:args.n_replicas].contiguous()

def structure(x):
    """Returns the invariant observables the upstream panel is built from, averaged over frames"""
    distances = target.preprocess(x)
    ordered = torch.sort(distances, dim=-1).values
    torsions = target.torsions(target.positions(x))
    return {'mean pair distance': float(distances.mean()),
            'distance quantile 10%': float(ordered[:, int(0.1 * (ordered.shape[-1] - 1))].mean()),
            'distance quantile 90%': float(ordered[:, int(0.9 * (ordered.shape[-1] - 1))].mean()),
            'cos torsion 0': float(torch.cos(torsions[:, 0]).mean()),
            'cos torsion 1': float(torch.cos(torsions[:, 1]).mean())}


print('\n%s %s split: %d frames of %d, %.0f fs step, friction %.1f/ps'
      % (target.name, args.split, len(x), len(released), 1000 * args.step_size, target.friction))
print('%8s %12s %12s' % ('t, ps', 'E/kT', 'sd'))
energy = target.energy(x)
start = float(energy.mean())
before = structure(x)
print('%8.1f %12.3f %12.3f' % (0.0, start, energy.std()))

velocities = None
for k in range(int(args.total_ps / args.interval_ps)):
    x, velocities = target.run_batched_dynamics(
        x, n_steps=int(args.interval_ps / args.step_size), step_size=args.step_size,
        velocities=velocities, return_velocities=True)
    energy = target.energy(x)
    print('%8.1f %12.3f %12.3f' % ((k + 1) * args.interval_ps, energy.mean(), energy.std()),
          flush=True)

# The drift, in units of its own standard error, so the verdict does not depend on the replica count
final = float(energy.mean())
standard_error = float(energy.std()) / len(x) ** 0.5
print('\ndrift %+.3f kT over %.0f ps, %.1f standard errors of the final mean'
      % (final - start, args.total_ps, abs(final - start) / max(standard_error, 1e-12)))

# Whether the CONFORMATIONS moved too, or only the energy. A relaxation faster than the slowest
# torsion is a local one, and it can leave the conformational content of a split usable while
# putting its energy histogram somewhere the density does not agree with
after = structure(x)
print('\n%24s %12s %12s %10s' % ('observable', 'released', 'relaxed', 'change'))
for name in before:
    print('%24s %12.4f %12.4f %+10.4f' % (name, before[name], after[name],
                                          after[name] - before[name]))
