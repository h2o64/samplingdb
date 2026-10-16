# Generate a glass-former reference set with swap Monte Carlo

# Libraries
import argparse, pprint, time
import numpy as np
import torch
from samplingdb.data import cache_dir, manifest, sha256
from samplingdb.targets import spec_id
from samplingdb.targets.physics import KobAndersen2D, swap_monte_carlo

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_particles', type=int, default=44)
parser.add_argument('--temperature', type=float, default=0.32)
parser.add_argument('--n_chains', type=int, default=250)
parser.add_argument('--equilibrate_taus', type=float, default=100.0)
parser.add_argument('--spacing_taus', type=float, default=20.0)
parser.add_argument('--n_configurations', type=int, default=400)
parser.add_argument('--step_size', type=float, default=0.065)
parser.add_argument('--p_swap', type=float, default=0.2)
parser.add_argument('--shard', type=int, default=0)
parser.add_argument('--n_shards', type=int, default=1)
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--merge', action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Build the target, and read the run length off its MEASURED relaxation time, in units of which
# the defaults reproduce the released protocol: 20 tau_alpha is 1.0e4 time units at T = 0.32 and
# 100 tau_alpha is 5.1e4, which is what that protocol used.
#
# The spacing is NOT 2 tau_alpha, which an exponential F_s would make ample. Run from the
# released ensemble under this kernel, F_s decays far more slowly than exp(-t / tau_alpha) in the
# tail: 0.365 at 1 tau_alpha (so the pinned tau_alpha is right, 1/e = 0.368), then 0.217 at 2,
# 0.136 at 3.5, 0.089 at 5, 0.049 at 8, 0.021 at 12.5 and 0.006 at 18.5. An exponential would
# have reached 0.135 by 2. Storing every 2 tau_alpha therefore leaves 22% of the displacement
# correlation in the reference set; 20 leaves 0.5%.
torch.set_num_threads(args.threads)
target = KobAndersen2D(temperature=args.temperature, n_particles=args.n_particles)
folder = cache_dir() / 'swapmc'
folder.mkdir(parents=True, exist_ok=True)
tau = target.relaxation_times[args.temperature]
n_equilibrate = int(args.equilibrate_taus * tau)
n_between = int(args.spacing_taus * tau)
print('target', target.name, 'box %.6f' % target.box, 'swap channels', target.swap_channels)
print('tau_alpha %.0f time units (measured): equilibrate %d, store every %d, so %d chain-time-units'
      % (tau, n_equilibrate, n_between,
         args.n_chains * (n_equilibrate + args.n_configurations * n_between)))

# --merge interleaves the shards into the reference set the target reads: configuration by
# configuration, every chain of every shard, so reading the first n samples spans all the chains.
# A shard stopped early contributes only the configurations every shard holds
if args.merge:
    shards = sorted(folder.glob(target.name + '_shard*.npy'))
    assert shards, 'no shards to merge in ' + str(folder)
    blocks = [np.load(path) for path in shards]
    n_keep = min(len(block) for block in blocks)
    array = np.concatenate([block[:n_keep] for block in blocks], axis=1).reshape(-1, target.dim)
    path = folder / (target.reference_key() + '.npy')

    # The specific heat is the acceptance criterion, following Jung et al 2024: it needs
    # independent configurations rather than merely equilibrated ones, so it plateaus far later
    # than the mean energy and is what catches an ensemble stored faster than it decorrelates.
    # The test is that it has PLATEAUED, not that it clears a threshold -- its equilibrium value
    # is a property of the temperature, 1.1683 at T = 0.32 but 0.9013 at T = 1.0 on the released
    # configurations, so there is no one number to compare against
    merged = torch.as_tensor(array, dtype=target.dtype)
    cv = target.specific_heat(merged)
    halves = [target.specific_heat(merged[:len(merged) // 2]),
              target.specific_heat(merged[len(merged) // 2:])]
    print('c_V %.4f  (halves %.4f / %.4f)  E/N %+.5f' % (
        cv, halves[0], halves[1], target.energy(merged).mean() / target.n_particles))
    assert abs(halves[0] - halves[1]) < 0.05 * cv, (
        'c_V has not plateaued -- %.4f on the first half against %.4f on the second. Raise '
        '--n_configurations, or --equilibrate_taus if the drift is monotonic.' % tuple(halves))

    np.save(path, array)
    protocol = dict(config)
    protocol.update({'n_samples': len(array), 'n_shards_merged': len(shards), 'n_keep': n_keep,
                     'shards': [p.name for p in shards], 'tau_alpha': tau,
                     'specific_heat': cv, 'specific_heat_halves': halves,
                     'torch': torch.__version__})
    (folder / (target.reference_key() + '_manifest.json')).write_text(manifest(
        target.name, spec_id(target.spec()), protocol, {path.name: sha256(path)}))
    print('merged %d shards into %s  %s' % (len(shards), path, array.shape))
    raise SystemExit

# One shard runs its own chains, seeded from its index so no two shards repeat each other
gen = torch.Generator().manual_seed(args.seed + 1000 * args.shard)
x = target.box * torch.rand(args.n_chains, target.n_particles, target.spatial_dim,
                            generator=gen, dtype=target.dtype)
print('shard %d of %d: %d chains x %d configurations' % (
    args.shard, args.n_shards, args.n_chains, args.n_configurations))

# Equilibrate, watching the potential energy relax to a steady value. The uniform start overlaps
# particles, so the first blocks are spent unjamming rather than ordering
start = time.time()
for block in range(10):
    x = target.positions(swap_monte_carlo(target, x, n_time_units=n_equilibrate // 10,
                                          step_size=args.step_size, p_swap=args.p_swap,
                                          generator=gen))
    energy = target.energy(x.reshape(args.n_chains, -1)) / target.n_particles
    print('equilibration %3d%%  E/N %+.4f +- %.4f  %.0f s'
          % (10 * (block + 1), energy.mean(), energy.std(), time.time() - start), flush=True)

# Collect one configuration per chain every n_between time units, persisting as we go
path = folder / (target.name + '_shard%03d.npy' % args.shard)
samples, energies, scatterings = [], [], []
for step in range(args.n_configurations):
    previous = x
    x = target.positions(swap_monte_carlo(target, x, n_time_units=n_between,
                                          step_size=args.step_size, p_swap=args.p_swap,
                                          generator=gen))
    samples.append(x.reshape(args.n_chains, -1).clone())

    # F_s between CONSECUTIVE stored configurations is what says the spacing was long enough:
    # it is 1 for a configuration that has not moved and decays to ~0 once the chain has
    # forgotten where it was, so it measures the thinning rather than assuming it
    scattering = target.self_intermediate_scattering(previous, x)
    scatterings.append(scattering)
    energies.append(target.energy(samples[-1]))
    stacked = torch.cat(energies)
    print('configuration %5d / %5d  E/N %+.4f +- %.4f  F_s %+.4f  c_V %.4f  %.0f s'
          % (step + 1, args.n_configurations, energies[-1].mean() / target.n_particles,
             energies[-1].std() / target.n_particles, scattering,
             float(stacked.var(unbiased=True)) / (target.n_particles * args.temperature ** 2),
             time.time() - start), flush=True)

    # Keep the configuration and chain axes apart, so the merge can interleave every shard's chains
    array = torch.stack(samples).cpu().numpy()
    np.save(path, array)

print('wrote %s  %s in %.0f s' % (path, array.shape, time.time() - start))

# The spacing has to be long enough that a stored configuration has forgotten the previous one.
# F_s is the field's measure of exactly that and costs nothing here, so it is asserted rather
# than assumed -- at 2 tau_alpha it sits at 0.22, which is what this gate exists to reject
mean_scattering = sum(scatterings) / len(scatterings)
print('F_s between consecutive stored configurations: mean %.4f  max %.4f'
      % (mean_scattering, max(scatterings)))
assert mean_scattering < 0.05, (
    'F_s %.4f between consecutive stored configurations: they are not independent draws. '
    'Raise --spacing_taus (20 gives 0.006 at T = 0.32).' % mean_scattering)
