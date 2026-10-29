# Generate a reference set with non-reversible parallel tempering (Syed et al. 2022)

# Libraries
import argparse, json, pathlib, pprint, sys, time
import numpy as np
import torch
from samplingdb import DW4, KobAndersenBinary, LJ13, LJ55
from samplingdb.data import cache_dir, sha256
from samplingdb.targets import spec_id
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diagnostics import chain_mean_dispersion, effective_sample_size
from manifest import manifest
from nrpt import hmc_explorer, linear_schedule, non_reversible_parallel_tempering

# Per-target ladder and budget. beta_min puts the hot level in a liquid or a gas, where the
# particles mix. n_local and thin come from a calibration run (--calibrate, then
# gen/audit/decorrelation.py on a finished set). Where the observed round-trip rate is far below
# 1 / (2 + 2 Lambda), the local kernel is the bottleneck and n_local is raised. Thinning a
# fat-tailed autocorrelation does not divide tau by the thinning factor, so thin is found by
# iterating the audit.
#
# ka-binary-N64-T0.2 runs 10 levels geometric in T from T = 0.6, where its structure relaxes in a
# few thousand HMC moves, down to its T = 0.2 target; a ladder reaching higher spends its barrier
# on a liquid. The ladder is kept as given and only the step sizes are tuned. Independent states
# are ~3e5 scans apart, so the shards store densely and gen/merge_shards.py picks the spacing.
SETTINGS = {
    'dw4': {'target': lambda: DW4(), 'n_rungs': 12, 'beta_min': 0.1, 'step_size': 0.05,
            'n_local': 2, 'thin': 16},
    'lj13': {'target': lambda: LJ13(), 'n_rungs': 16, 'beta_min': 0.1, 'step_size': 0.01,
             'n_local': 32, 'thin': 2},
    # beta_min 0.5. Under the doubled pair sum beta = 1 is T* = 0.5, above LJ-55's melting point,
    # so the target is already a liquid and the hot end needs no melt. Below beta = 0.5 the
    # cluster is a dilute gas held by its own oscillator, which carries most of the barrier and no
    # state the target needs. A hot end at twice the target temperature keeps some insurance
    # against a mode the observables do not see.
    'lj55': {'target': lambda: LJ55(), 'n_rungs': 12, 'beta_min': 0.5, 'step_size': 0.005,
             'n_local': 8, 'thin': 8},
    'ka-binary-N64-T0.2': {'target': lambda: KobAndersenBinary(temperature=0.2, n_particles=64),
                           'n_rungs': 10, 'beta_min': 1.0 / 3.0, 'step_size': 0.005,
                           'n_local': 2, 'thin': 16384, 'melt_beta': 1.0 / 3.0,
                           'melt_scans': 2000, 'schedule': 'geometric', 'adapt_schedule': False},
}

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True, choices=list(SETTINGS))
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_rounds', type=int, default=10)
parser.add_argument('--thin', type=int, default=None)
parser.add_argument('--n_leapfrog', type=int, default=10)
parser.add_argument('--n_local', type=int, default=None)
parser.add_argument('--n_rungs', type=int, default=None)
parser.add_argument('--beta_min', type=float, default=None)
parser.add_argument('--step_size', type=float, default=None)
parser.add_argument('--min_round_trips', type=float, default=10.0)
parser.add_argument('--calibrate', action='store_true')
parser.add_argument('--freeze', action='store_true')
parser.add_argument('--n_scans', type=int, default=None)
parser.add_argument('--n_burn', type=int, default=None)
parser.add_argument('--hours', type=float, default=None)
parser.add_argument('--melt_beta', type=float, default=None)
parser.add_argument('--melt_scans', type=int, default=None)
parser.add_argument('--start', type=str, default=None)
parser.add_argument('--start_offset', type=int, default=0)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--shard', type=int, default=None)
parser.add_argument('--n_shards', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# A shard is an independent run of the same sampler under its own seed, written beside the others
# and merged by gen/merge_shards.py. Independent chains are independent, so the effective sample
# sizes add and the merge is exact
if args.shard is not None:
    args.seed = args.seed + args.shard

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Build the target and its ladder. The schedule is a starting point: unless adapt_schedule is
# False, the sampler re-places the levels between rounds from the barrier it measures
settings = SETTINGS[args.target]
for key in ['thin', 'n_local', 'n_rungs', 'beta_min', 'step_size', 'melt_beta', 'melt_scans']:
    if getattr(args, key) is None:
        setattr(args, key, settings.get(key))
config = vars(args)
device = torch.device(args.device)
target = settings['target']().to(device)

# Compile the energy and score chunks, so inductor fuses the elementwise ops on their all-pairs
# temporaries
target.energy_chunk = torch.compile(target.energy_chunk, dynamic=False)
target.score_chunk = torch.compile(target.score_chunk, dynamic=False)
if settings.get('schedule') == 'geometric':
    beta = args.beta_min ** torch.linspace(1.0, 0.0, args.n_rungs, dtype=target.dtype,
                                           device=device)
else:
    beta = linear_schedule(args.n_rungs, args.beta_min, dtype=target.dtype, device=device)
print('target %s  dim %d  rungs %d  beta %.4f .. 1.0' % (
    target.name, target.dim, len(beta), float(beta[0])))

# Set the seed and start every level from the target's own fixed starting configuration, so the
# run is reproducible and no released data is needed to begin. The lattice jitter is 0.02: the
# binary glass's lattice spacing of 1.0 is already inside the AA minimum, and a larger jitter puts
# pairs so close that HMC tuned on the equilibrium ensemble never accepts a move from them
torch.manual_seed(args.seed)
gen = torch.Generator(device='cpu').manual_seed(args.seed)
# --start replaces both with configurations of an earlier run, deep states of the same target,
# rows start_offset onwards so shards started from one file do not share a start. Half the shards
# of a production run start deep and half from the melt below, and gen/merge_shards.py checks that
# the two halves agree
if args.start is not None:
    rows = np.load(args.start, mmap_mode='r')[args.start_offset:args.start_offset + args.n_chains]
    assert len(rows) == args.n_chains, 'the start file has %d rows from offset %d' % (
        len(rows), args.start_offset)
    start = torch.as_tensor(np.array(rows), dtype=target.dtype)
elif hasattr(target, 'lattice_configuration'):
    start = target.lattice_configuration(args.n_chains, jitter=0.02, generator=gen)
    start = start.reshape(args.n_chains, -1)
else:
    start = target.initial_configuration().reshape(1, -1).expand(args.n_chains, target.dim).cpu()
    start = start + 0.05 * torch.randn(args.n_chains, target.dim, generator=gen,
                                       dtype=target.dtype)
x0 = start.to(device)[None].expand(len(beta), args.n_chains, target.dim).contiguous()

# Run: rounds of doubling length, the schedule and the step sizes re-tuned between them, and
# every kept state drawn in the final round where nothing is being adapted any more
# --freeze reuses a calibration's converged schedule and step sizes and adapts nothing, which
# replaces the discarded doubling rounds, as long as the collection round, with a burn-in of a few
# autocorrelation times. n_rounds becomes 1 and n_scans sets the length.
schedule, step_size, n_rounds = beta, args.step_size, args.n_rounds
if args.freeze:
    measured = json.loads((cache_dir() / 'plan' / (target.name + '_plan.json')).read_text())
    schedule = torch.tensor(measured['final_beta'], dtype=target.dtype, device=device)
    step_size = torch.tensor(measured['final_step_size'], dtype=target.dtype, device=device)
    n_rounds = 1
    x0 = x0[:len(schedule)] if len(x0) >= len(schedule) else x0
    args.n_rungs = len(schedule)
    args.n_local, args.n_leapfrog = measured['n_local'], measured['n_leapfrog']
    args.beta_min, args.step_size = measured['final_beta'][0], measured['final_step_size']
    print('frozen from the calibration: %d rungs, beta %.4f .. %.4f, %d scans after %d burn-in'
          % (len(schedule), schedule[0], schedule[-1], args.n_scans or 0, args.n_burn or 0))

# --resume continues a shard from the ladder state it saved last, with no melt and no burn-in, and
# appends to the states it already stored, so a set is extended without paying its burn-in twice.
# The seed moves on with every resumption, so the continuation does not replay the noise of the run
# it continues
folder = cache_dir() / 'pt'
stem = (target.reference_key() if args.shard is None
        else '%s_shard%02d' % (target.name, args.shard))
stored, previous = None, {}
if args.resume:
    assert args.freeze and args.shard is not None, '--resume continues a frozen shard'
    previous = json.loads((folder / (stem + '_manifest.json')).read_text())['protocol']
    assert previous['n_chains'] == args.n_chains, 'the shard ran %d chains' % previous['n_chains']
    x0 = torch.load(folder / (stem + '_state.pt')).to(dtype=target.dtype, device=device)
    stored = torch.as_tensor(np.load(folder / (stem + '.npy')), dtype=target.dtype,
                             device=device).reshape(-1, args.n_chains, target.dim)
    args.n_burn, args.melt_scans = 0, 0
    torch.manual_seed(args.seed + 7919 * previous['n_segments'])
    print('resuming %s: %d stored per chain over %d segments' % (
        stem, len(stored), previous['n_segments']))

# Melt the start at a level where the particles mix, then hand the melt to every level. The lattice
# puts the binary glass's B particles on consecutive sites, which the cold levels take far longer
# to undo than a melt does
start_time = time.time()
explorer = hmc_explorer(n_steps=args.n_local, n_leapfrog=args.n_leapfrog)
if args.melt_scans and args.start is None:
    level = int(torch.argmin((schedule - args.melt_beta).abs()))
    melt_beta = schedule[level]
    melt_step = step_size[level] if torch.is_tensor(step_size) else step_size
    melted = x0[0]
    for _ in range(args.melt_scans):
        melted, _ = explorer(lambda z: melt_beta * target.log_prob(z),
                             lambda z: melt_beta * target.score(z), melted, melt_step, None)
    x0 = melted[None].expand(len(schedule), args.n_chains, target.dim).contiguous()
    print('melted the start for %d scans at beta %.4f' % (args.melt_scans, float(melt_beta)))


def write(samples, diagnostics, extra):
    """Saves the stored states chain-interleaved, next to a manifest naming the protocol

    The ladder's last state goes beside them as <stem>_state.pt, which is what --resume continues
    from. The manifest records where the chains started: 'start_group' is 'deep' for a --start file,
    whose name and sha256 are kept, and 'melt' otherwise.

    Args:
        * samples (torch.Tensor of shape (n_keep, n_chains, dim)): States of the beta = 1 level
        * diagnostics (dict): What non_reversible_parallel_tempering returned last
        * extra (dict): Diagnostics to record beside the protocol

    Returns:
        * path (pathlib.Path): Where the states went
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (stem + '.npy')
    np.save(path, samples.reshape(-1, target.dim).cpu().numpy())
    torch.save(diagnostics['rung_state'].cpu(), folder / (stem + '_state.pt'))
    protocol = dict(config)
    protocol.update({
        'n_keep': len(samples),
        'sampler': 'non_reversible_parallel_tempering', 'n_rungs': args.n_rungs,
        'beta_min': args.beta_min, 'initial_step_size': args.step_size,
        'final_beta': [float(v) for v in diagnostics['beta']],
        'final_step_size': [float(v) for v in diagnostics['step_size']],
        'start': None if args.start is None else pathlib.Path(args.start).name,
        'start_sha256': None if args.start is None else sha256(args.start),
        'start_group': 'melt' if args.start is None else 'deep',
        'torch': torch.__version__,
    })
    protocol.update(extra)

    # A resumed shard keeps the start and the burn-in of the run it continues
    if previous:
        protocol.update({key: previous[key] for key in ['start', 'start_sha256', 'start_group',
                                                        'n_burn', 'seed']})
        protocol['n_segments'] = previous['n_segments'] + extra['n_segments']
        protocol['resumed'] = previous.get('resumed', 0) + 1
    (folder / (stem + '_manifest.json')).write_text(manifest(
        target.name, spec_id(target.spec()), protocol, {path.name: sha256(path)}))
    return path


samples, diagnostics = non_reversible_parallel_tempering(
    target.log_prob, target.score, x0, schedule, explorer=explorer,
    n_rounds=n_rounds, reference=None, step_size=step_size,
    adaptation_gain=0.0 if args.freeze else 1.0,
    adapt_schedule=settings.get('adapt_schedule', True), generator=None,
    thin=args.thin, n_scans=args.n_scans, n_burn=args.n_burn or 0, return_diagnostics=True,
)
if stored is not None:
    samples = torch.cat([stored, samples])

# --hours makes n_scans one SEGMENT of a run that lasts as long as the wall clock allows. Each
# segment continues from the last one's level-ordered state, so the segments are one chain, and the
# shard is rewritten after every one: a job then costs its hours whatever GPU it lands on, and a
# job stopped early keeps every segment it finished
n_segments, round_trips = 1, diagnostics['round_trips']
if args.hours is not None:
    assert args.freeze, '--hours continues a frozen ladder, so it needs --freeze'
    assert float(diagnostics['explorer_acceptance_per_chain'].min()) > 0.01, (
        'a chain is frozen after the burn-in, so nothing was written')
    write(samples, diagnostics, {'n_segments': n_segments})
    segment = (time.time() - start_time) * args.n_scans / (args.n_scans + (args.n_burn or 0))
    while time.time() - start_time + 1.5 * segment < 3600 * args.hours:
        begun = time.time()
        more, diagnostics = non_reversible_parallel_tempering(
            target.log_prob, target.score, diagnostics['rung_state'], schedule, explorer=explorer,
            n_rounds=1, reference=None, step_size=step_size, adaptation_gain=0.0, generator=None,
            thin=args.thin, n_scans=args.n_scans, return_diagnostics=True)
        samples = torch.cat([samples, more])
        round_trips = round_trips + diagnostics['round_trips']
        n_segments += 1
        segment = time.time() - begun
        write(samples, diagnostics, {'n_segments': n_segments})
        print('segment %d: %d stored per chain, %.0f s' % (n_segments, len(samples),
                                                          time.time() - start_time), flush=True)
elapsed = time.time() - start_time

# Report the diagnostics the run has to clear before its output is worth anything. The scans run
# are the discarded doubling rounds, the final round and its burn-in, which a frozen run reduces to
# the last two
n_scans = (args.n_scans or 2 ** n_rounds) * n_segments
total_scans = 2 ** n_rounds - 2 + n_scans + (args.n_burn or 0)
print('\n%.1f s for %d scans over %d rounds, collecting %d of the last %d'
      % (elapsed, total_scans, n_rounds, n_scans // args.thin, n_scans))
print('barrier Lambda   %.4f   (recommends %d rungs; this ladder has %d)'
      % (diagnostics['barrier'], diagnostics['recommended_n_rungs'], args.n_rungs))
print('beta            ', [round(float(v), 4) for v in diagnostics['beta']])
print('rejection       ', [round(float(v), 3) for v in diagnostics['rejection']])
print('explorer accept ', [round(float(v), 3) for v in diagnostics['explorer_acceptance']])
print('step size       ', [round(float(v), 5) for v in diagnostics['step_size']])
print('round trips per chain (final round): mean %.1f  min %.0f  max %.0f'
      % (float(round_trips.mean()), float(round_trips.min()), float(round_trips.max())))

# The round-trip rate against 1 / (2 + 2 Lambda), what an ideal local kernel reaches on this path.
# One machine's round trip walks 2 (n_rungs - 1) levels at best and takes n_rungs (2 + 2 Lambda)
# scans in practice, so one independent configuration costs about n_rungs^2 (2 + 2 Lambda) n_local
# energy evaluations, ~8 Lambda^3 n_local at n_rungs = 2 Lambda. The hot end of the path sets the
# cost, and gen/audit/barrier.py profiles it
print('one machine round trip: %.0f scans observed, %.0f ideal, %.0f ballistic floor'
      % (diagnostics['scans_per_round_trip'], diagnostics['ideal_scans_per_round_trip'],
         diagnostics['ballistic_scans_per_round_trip']))
print('round trip rate  %.4f   against the %.4f this barrier allows (%.0f%% of it)'
      % (diagnostics['round_trip_rate'], diagnostics['predicted_round_trip_rate'],
         100.0 * diagnostics['round_trip_rate'] / diagnostics['predicted_round_trip_rate']))
print('log Z(1) / Z(%.2f)  %.4f' % (args.beta_min, diagnostics['log_normaliser']))

# Assert that no level and no chain is frozen. With no reference at beta = 0 nothing refreshes a
# frozen chain, and since every level starts from the same configuration it accepts every swap and
# reaches the ballistic round-trip count while contributing one configuration. Both checks hold
# under --calibrate too, where a frozen chain would inflate the tau the production run is sized from
frozen = float(diagnostics['explorer_acceptance_per_chain'].min())
assert frozen > 0.01, (
    'some (rung, chain) pair accepts only %.4f of its local moves: that chain is frozen and holds '
    'the configuration every rung started from' % frozen
)
assert float(diagnostics['explorer_acceptance'].min()) > 0.01, (
    'rung %d never accepts a local move: its step size is past the kernel\'s stability limit, or '
    'it started on a configuration of effectively zero density'
    % int(diagnostics['explorer_acceptance'].argmin())
)

# Every chain must make several round trips between the two ends of the ladder, and no adjacent
# pair may reject almost every swap
assert args.calibrate or float(round_trips.min()) >= args.min_round_trips, (
    'only %.0f round trips on the worst chain: raise n_rounds, soften beta_min, or give the '
    'ladder the %d rungs its barrier asks for'
    % (float(round_trips.min()), diagnostics['recommended_n_rungs'])
)
assert args.calibrate or float(diagnostics['rejection'].max()) < 0.95, (
    'adjacent pair %d rejects %.3f of its swaps: the ladder has a gap the schedule adaptation '
    'could not close' % (int(diagnostics['rejection'].argmax()),
                         float(diagnostics['rejection'].max()))
)

# Measure the efficiency of the stored set on the observables the metrics are built from. The
# energy alone is not enough, since it decorrelates long before the structure does (Jung et al.
# 2024 for the glass formers). The distances are sorted because compare_histograms pools every pair
# into one histogram, while a distance at a fixed index tracks one labelled pair and decorrelates
# only when particles permute, a symmetry the metric quotients out.
n_keep = len(samples)
flat = samples.reshape(-1, target.dim)
energy = target.energy(flat) if hasattr(target, 'energy') else -target.log_prob(flat)
observables = {'energy': energy.reshape(n_keep, args.n_chains).T}
distances = target.distances(flat) if hasattr(target, 'distances') else target.preprocess(flat)
distances = distances.reshape(n_keep, args.n_chains, -1)
observables['mean pair distance'] = distances.mean(-1).T
ordered = torch.sort(distances, dim=-1).values
for fraction in [0.1, 0.5, 0.9]:
    column = int(fraction * (ordered.shape[-1] - 1))
    observables['distance quantile %.0f%%' % (100 * fraction)] = ordered[:, :, column].T

ess = effective_sample_size(observables)

# The spread of the per-chain means against what independence predicts. It needs no lag window,
# so it disagrees with the autocorrelation function where the ACF is weakest: a lag window that is
# too short reads tau low, and a few chains stuck at their own mean read tau high
dispersion = chain_mean_dispersion(observables)
print('\nworst tau %.2f stored states, on %s' % (ess['tau'], ess['name']))
print('between-chain check: tau %.2f on %s (1 is independent)'
      % (dispersion['tau'], dispersion['name']))
print('effective sample size %.0f of %d stored (%.0f%%)'
      % (ess['n_effective'], n_keep * args.n_chains, 100 * ess['efficiency']))

# A shard is not gated on its own efficiency, which is too noisy at a shard's chain count and
# would lose its states on a miss: gen/merge_shards.py gates the merged set on every chain
assert args.calibrate or args.shard is not None or ess['efficiency'] > 0.5, (
    'the stored set is worth %.0f%% of its length: raise --thin to about %d, or raise --n_local '
    'if the round-trip rate is far below what the barrier allows'
    % (100 * ess['efficiency'], int(args.thin * ess['tau']) + 1)
)
if ess['efficiency'] < 0.9:
    # First order only: thinning a series whose autocorrelation has a fat tail does not divide
    # tau by the thinning factor, so this under-shoots and the estimate wants iterating
    print('NOTE: thin %d leaves %.0f%% efficiency; about %d should reach 90%%, iterate from there'
          % (args.thin, 100 * ess['efficiency'], int(args.thin * ess['tau'] / 0.9) + 1))

# Report the ENERGY, not -log_prob: they differ by the temperature wherever one is folded in
print('\ncollected %s from the beta = 1 rung' % (tuple(samples.reshape(-1, target.dim).shape),))
head = samples.reshape(-1, target.dim)[:1000]
energy = target.energy(head) if hasattr(target, 'energy') else -target.log_prob(head)
per_particle = getattr(target, 'n_particles', getattr(target, 'n_atoms', 1))
print('energy: mean %.4f  std %.4f  (per particle %.4f)' % (
    float(energy.mean()), float(energy.std()), float(energy.mean()) / per_particle))

# A calibration run measures and writes no reference set. It skips the round-trip, swap and
# efficiency checks above, since its thin is what it measures, and leaves a plan file that --freeze
# reads back
# and the production shards are sized from. Its throughput is recorded too, as a property of the
# GPU it ran on.
if args.calibrate:
    folder = cache_dir() / 'plan'
    folder.mkdir(parents=True, exist_ok=True)
    measured = {
        'target': target.name, 'generator': 'pt',
        'barrier': diagnostics['barrier'],
        'recommended_n_rungs': diagnostics['recommended_n_rungs'],
        'n_rungs': args.n_rungs, 'beta_min': args.beta_min, 'thin': args.thin,
        'n_chains': args.n_chains, 'n_local': args.n_local, 'n_leapfrog': args.n_leapfrog,
        'final_beta': [float(v) for v in diagnostics['beta']],
        'final_step_size': [float(v) for v in diagnostics['step_size']],
        'tau_stored': ess['tau'], 'tau_scans': ess['tau'] * args.thin,
        'between_chain_tau': dispersion['tau'],
        'scans_per_round_trip': diagnostics['scans_per_round_trip'],
        'ideal_scans_per_round_trip': diagnostics['ideal_scans_per_round_trip'],
        'configurations_per_second':
            total_scans * args.n_rungs * args.n_chains * args.n_local * args.n_leapfrog / elapsed,
        'energy_mean': float(energy.mean()), 'energy_std': float(energy.std()),
        'wall_clock_seconds': round(elapsed, 1),
    }
    (folder / (target.name + '_plan.json')).write_text(json.dumps(measured, indent=2))
    print('\nwrote the calibration to %s' % (folder / (target.name + '_plan.json')))
    raise SystemExit(0)

# Save the draws next to a manifest naming the protocol
path = write(samples, diagnostics, {
    'n_segments': n_segments,
    'explorer_acceptance': [float(v) for v in diagnostics['explorer_acceptance']],
    'rejection': [float(v) for v in diagnostics['rejection']],
    'barrier': diagnostics['barrier'],
    'recommended_n_rungs': diagnostics['recommended_n_rungs'],
    'round_trip_rate': diagnostics['round_trip_rate'],
    'predicted_round_trip_rate': diagnostics['predicted_round_trip_rate'],
    'log_normaliser': diagnostics['log_normaliser'],
    'min_round_trips': float(round_trips.min()),
    'stored_tau': ess['tau'], 'between_chain_tau': dispersion['tau'],
    'n_effective': ess['n_effective'],
    'efficiency': ess['efficiency'],
    'wall_clock_seconds': round(elapsed, 1),
})
print('wrote %s' % path)
