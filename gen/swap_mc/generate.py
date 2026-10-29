# Generate a ternary glass reference set with swap Monte Carlo

# Libraries
import argparse, pathlib, pprint, sys, time
import numpy as np
import torch
from samplingdb.data import cache_dir, sha256
from samplingdb.targets import spec_id
from samplingdb.targets.physics import KobAndersen2D, swap_monte_carlo
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from manifest import manifest

# Per temperature: time units between stored states, and the burn-in before the first one. One
# time unit is n_particles attempted moves. The spacing is deliberately finer than the
# decorrelation time: gen/merge_shards.py keeps every k-th stored state for the smallest k at which
# the merged set is 90% efficient, so the thin is measured on the set rather than predicted.
#
# The structural relaxation time read off trajectories, following each particle through the swaps
# that exchange positions, is ~2300 time units at T = 0.32 and ~265 at T = 1.0. A per-index
# self-intermediate scattering function reads it far shorter, because every accepted swap makes two
# indexes jump across the box.
SETTINGS = {
    0.32: {'spacing': 5000, 'n_stored': 44, 'burn': 20000},
    1.0: {'spacing': 400, 'n_stored': 20, 'burn': 4000},
}

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.32, choices=list(SETTINGS))
parser.add_argument('--n_chains', type=int, default=16384)
parser.add_argument('--spacing', type=int, default=None)
parser.add_argument('--n_stored', type=int, default=None)
parser.add_argument('--burn', type=int, default=None)
parser.add_argument('--start', type=str, default=None)
parser.add_argument('--start_stride', type=int, default=1)
parser.add_argument('--step_size', type=float, default=0.065)
parser.add_argument('--p_swap', type=float, default=0.2)
parser.add_argument('--shard', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
for key in ['spacing', 'n_stored', 'burn']:
    if getattr(args, key) is None:
        setattr(args, key, SETTINGS[args.temperature][key])

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Build the target and the generator on the run's device: the draws are made there, and on a GPU
# the move is compiled
device = torch.device(args.device)
target = KobAndersen2D(temperature=args.temperature).to(device)
folder = cache_dir() / 'swapmc'
folder.mkdir(parents=True, exist_ok=True)
gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * args.shard)
compile = device.type == 'cuda'

# Start from equilibrium configurations when given (any earlier set of this target), every
# start_stride-th row so neighbouring chains do not start from consecutive states of one old chain;
# otherwise from a jittered square lattice, in which case --burn has to cover the whole relaxation
if args.start is not None:
    rows = np.load(args.start, mmap_mode='r')[::args.start_stride][:args.n_chains]
    assert len(rows) == args.n_chains, 'the start file has %d rows at this stride' % len(rows)
    x = target.positions(torch.as_tensor(np.array(rows), dtype=target.dtype, device=device))
else:
    lattice = torch.Generator().manual_seed(args.seed + 1000 * args.shard)
    x = target.lattice_configuration(args.n_chains, jitter=0.05, generator=lattice).to(device)
print('%s: %d chains from %s, burn %d, then %d states every %d time units'
      % (target.name, args.n_chains, args.start or 'the lattice', args.burn, args.n_stored,
         args.spacing))

# Burn in, then store a state every spacing time units, scan-major so the merge can keep every k-th
start = time.time()
x, acceptance = swap_monte_carlo(target, x, n_time_units=args.burn, step_size=args.step_size,
                                 p_swap=args.p_swap, generator=gen, return_acceptance=True,
                                 compile=compile)
x = target.positions(x)
energy = target.energy(x.reshape(args.n_chains, -1)) / target.n_particles
print('burn-in done in %.0f s: E/N %+.4f +- %.4f, acceptance %s'
      % (time.time() - start, energy.mean(), energy.std(), acceptance), flush=True)
samples, energies = [], []
for step in range(args.n_stored):
    x = target.positions(swap_monte_carlo(target, x, n_time_units=args.spacing,
                                          step_size=args.step_size, p_swap=args.p_swap,
                                          generator=gen, compile=compile))
    samples.append(x.reshape(args.n_chains, -1).clone())
    energies.append(target.energy(samples[-1]))
    print('state %3d / %3d  E/N %+.5f  %.0f s' % (step + 1, args.n_stored,
          energies[-1].mean() / target.n_particles, time.time() - start), flush=True)
elapsed = time.time() - start

# The specific heat needs independent configurations rather than merely equilibrated ones (Jung et
# al. 2024), so it is recorded per half of the stored states as a plateau check next to the merge's
# autocorrelation gate
energy = torch.stack(energies)
halves = [float(part.var(unbiased=True)) / (target.n_particles * args.temperature ** 2)
          for part in [energy[:args.n_stored // 2], energy[args.n_stored // 2:]]]
print('c_V by half of the stored states: %.4f / %.4f' % tuple(halves))

# Save the shard and a manifest naming its real layout; gen/merge_shards.py --folder swapmc
# chooses the spacing and writes the reference set
path = folder / ('%s_shard%02d.npy' % (target.name, args.shard))
np.save(path, torch.stack(samples).reshape(-1, target.dim).cpu().numpy())
protocol = dict(config)
protocol.update({'start': None if args.start is None else pathlib.Path(args.start).name,
                 'sampler': 'swap_monte_carlo', 'n_keep': args.n_stored,
                 'acceptance': acceptance, 'specific_heat_halves': halves,
                 'start_sha256': sha256(args.start) if args.start is not None else None,
                 'wall_clock_seconds': round(elapsed, 1), 'torch': torch.__version__})
(folder / (path.stem + '_manifest.json')).write_text(manifest(
    target.name, spec_id(target.spec()), protocol, {path.name: sha256(path)}))
print('wrote %s  (%d states x %d chains) in %.0f s' % (path, args.n_stored, args.n_chains, elapsed))
