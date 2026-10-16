# Generate a reference set with non-reversible parallel tempering (Syed et al. 2022)

# Libraries
import argparse, json, pprint, time
import numpy as np
import torch
from samplingdb import DW4, KobAndersenBinary, LJ13, LJ55
from samplingdb.baselines import (hmc_explorer, linear_schedule,
                                  non_reversible_parallel_tempering)
from samplingdb.data import cache_dir, manifest, sha256
from samplingdb.metrics import chain_mean_dispersion, effective_sample_size

# Per-target ladder and budget. beta_min is chosen so the hot rung is a liquid or a gas: the
# particle clusters melt well below their own energy scale, and the binary glass is a liquid at
# T = 2.0, which is beta = 0.1 of its T = 0.2 target.
#
# n_local and thin are MEASURED, not guessed, and the measurement is what the barrier makes
# possible. Diagnosed at 32 chains over 512 collection scans (gen/audit/decorrelation.py reads
# the same quantities off a finished set):
#
# | target        | Lambda | rungs it asks for | rate / the cap Lambda allows | tau, scans |
# | ---           | ---    | ---               | ---                          | ---        |
# | dw4           | 1.67   | 4                 | 85%                          | 4.6        |
# | lj13          | 6.79   | 14                | 4% at n_local 2, 47% at 32   | 5.7 -> 1.2 |
# | ka-binary-2.0 | 2.27   | 5                 | 62%                          | 3.2        |
# | ka-binary-0.5 | 5.80   | 12                | 29%                          | ~106       |
# | ka-binary-0.2 | 8.69   | 18                | 8% at n_local 2, 13% at 8    | ~88        |
# | lj55          | 13.81  | 28                | 0% at 24 rungs               | ~8.5       |
#
# dw4 is the one carried all the way through: 12 rungs, n_local 2, thin 16, 14 rounds gives
# 102 400 states whose worst tau is 1.06, i.e. 94% of them independent, in two minutes. The
# thinning was found by iterating the audit rather than predicted -- thin 5 leaves 65%, 8 leaves
# 81%, 16 leaves 94%, because the autocorrelation has a fat tail and thinning does not simply
# divide tau by the thinning factor.
#
# Two readings matter. Where the observed round-trip rate is far below 1 / (2 + 2 Lambda), the
# LOCAL kernel is the bottleneck and not the ladder, and buying more of it pays: lj13 goes from 4%
# to 47% of the cap for 16x the work per scan, and its stored autocorrelation falls from 5.7 scans
# to 1.2. Where it does not pay, that is worth knowing too -- ka-binary at T = 0.2 gains only 1.5x
# in rate for 4x the work, because a supercooled glass relaxes by structural rearrangement that
# more Hamiltonian trajectory does not accelerate.
#
# WARNING: the two ~ values are not trustworthy. An autocorrelation time is only estimable from a
# series many times longer than itself, and 512 stored scans against tau ~ 100 is six times its
# own length. They are recorded as a lower bound and a warning, not as a thin to rely on.
#
# ka-binary-N64-T0.2 is slower than the table says. Over 8192 collection scans at these settings
# (64 chains, 2026-10-09) its slowest observables were structural: B-B contacts tau 1257 scans,
# mean pair distance 1067, the orientation and size of an A crystallite 772 and 724, against 137
# for the energy -- and those are still lower bounds. At T = 0.2 the A particles partly crystallise
# (see KobAndersenBinary), and the crystallite only melts at the hot levels. Compiled, one
# chain-scan runs at ~33 600/s on an RTX A6000 from 256 chains up, and its calibration has to
# collect for well over a round trip (~3000 scans).
SETTINGS = {
    'dw4': {'target': lambda: DW4(), 'n_rungs': 12, 'beta_min': 0.1, 'step_size': 0.05,
            'n_local': 2, 'thin': 16},
    'lj13': {'target': lambda: LJ13(), 'n_rungs': 16, 'beta_min': 0.1, 'step_size': 0.01,
             'n_local': 32, 'thin': 2},
    # beta_min 0.5 rather than 0.1. The barrier profile (gen/audit/barrier.py) shows LJ-55's
    # barrier is not a bottleneck anywhere: lambda(beta) falls MONOTONICALLY from 90 at
    # beta = 0.02 to 7.5 at beta = 1, with no spike, so there is nothing to cross and 72% of
    # Lambda simply sits below beta = 0.5 -- where E is POSITIVE and the cluster is a dilute gas
    # held only by its own oscillator, which is not a state the target needs to visit.
    #
    # It does not need to visit a melt either, because the target is already molten. Under this
    # convention the pair sum is doubled, so beta = 1 is T* = 0.5 against a published melting
    # point of 0.294 for LJ-55, and independently of that number the corpus's own measurement
    # says the same thing: a LABELLED pair distance decorrelates in tau = 63 scans here, which it
    # can only do if particles exchange neighbours. Measured at 32 chains over 512 scans:
    #
    # | beta_min | rungs | Lambda | round trips / cap | min | tau | wall clock |
    # | ---      | ---   | ---    | ---               | --- | --- | ---        |
    # | 0.1      | 28    | 14.59  | 0%                | 0   | 1.13 | 595 s     |
    # | 0.3      | 16    | 8.65   | 8%                | 0   | 1.20 | 507 s     |
    # | 0.5      | 12    | 5.15   | 38%               | 11  | 1.20 | 319 s     |
    # | 0.7      | 8     | 2.67   | 56%               | 30  | 1.13 | 264 s     |
    #
    # The cold rung is the SAME ensemble in all four, E = -325.3 +- 14.0 against a released -323,
    # so this buys round trips and costs nothing. 0.5 rather than 0.7 because 0.7 is a hot end
    # only 1.4x the target temperature, which is barely tempering at all and leaves no insurance
    # against a mode the observables do not see; 0.5 is 2x the target and still clears the gate
    # with room.
    'lj55': {'target': lambda: LJ55(), 'n_rungs': 12, 'beta_min': 0.5, 'step_size': 0.005,
             'n_local': 8, 'thin': 8},
    'ka-binary-N64-T0.2': {'target': lambda: KobAndersenBinary(temperature=0.2, n_particles=64),
                           'n_rungs': 20, 'beta_min': 0.1, 'step_size': 0.005,
                           'n_local': 2, 'thin': 100},
    'ka-binary-N64-T0.5': {'target': lambda: KobAndersenBinary(temperature=0.5, n_particles=64),
                           'n_rungs': 12, 'beta_min': 0.2, 'step_size': 0.01,
                           'n_local': 2, 'thin': 110},
    'ka-binary-N64-T2.0': {'target': lambda: KobAndersenBinary(temperature=2.0, n_particles=64),
                           'n_rungs': 8, 'beta_min': 0.5, 'step_size': 0.02,
                           'n_local': 2, 'thin': 4},
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

# Build the target and its ladder. The schedule is only a STARTING point now -- the sampler
# re-places the rungs between rounds from the communication barrier it measures, so beta_min is
# the one thing left to choose and n_rungs no longer has to be guessed well
settings = SETTINGS[args.target]
for key in ['thin', 'n_local', 'n_rungs', 'beta_min', 'step_size']:
    if getattr(args, key) is None:
        setattr(args, key, settings[key])
config = vars(args)
device = torch.device(args.device)
target = settings['target']().to(device)

# Compile the energy and score chunks. Every target here evaluates on all-pairs temporaries, which
# eager torch reads and writes once per elementwise op, and inductor fuses the whole chain: on the
# binary glass that is 1300 against 33 600 chain-scans per second on an RTX A6000, with the score
# agreeing to 2e-6
target.energy_chunk = torch.compile(target.energy_chunk, dynamic=False)
target.score_chunk = torch.compile(target.score_chunk, dynamic=False)
beta = linear_schedule(args.n_rungs, args.beta_min, dtype=target.dtype, device=device)
print('target %s  dim %d  rungs %d  beta %.4f .. 1.0' % (
    target.name, target.dim, len(beta), float(beta[0])))

# Set the seed and start every rung from the target's own fixed starting configuration, so the run
# is reproducible and no released data is needed to begin
# The lattice jitter is 0.02 because the binary glass's lattice spacing is 1.0, already inside the
# AA minimum: at 0.05, 214 of 10 240 starts put a pair closer than 0.75, where the AA force is
# ~1e4 per unit of log density, and HMC tuned on the equilibrium ensemble never accepts a move from
# there. One calibration had two such chains (E/N 4.09 and 2.95, closest pairs 0.717 and
# 0.730) and read tau as its whole collection window. At 0.02 the closest pair over 10 240 starts
# is 0.853
torch.manual_seed(args.seed)
gen = torch.Generator(device='cpu').manual_seed(args.seed)
if hasattr(target, 'lattice_configuration'):
    start = target.lattice_configuration(args.n_chains, jitter=0.02, generator=gen)
    start = start.reshape(args.n_chains, -1)
else:
    start = target.initial_configuration().reshape(1, -1).expand(args.n_chains, target.dim).cpu()
    start = start + 0.05 * torch.randn(args.n_chains, target.dim, generator=gen,
                                       dtype=target.dtype)
x0 = start.to(device)[None].expand(len(beta), args.n_chains, target.dim).contiguous()

# Run: rounds of doubling length, the schedule and the step sizes re-tuned between them, and
# every kept state drawn in the final round where nothing is being adapted any more
# --freeze reuses a calibration's converged schedule and step sizes and adapts NOTHING, which
# turns the doubling's permanent 2x -- the discarded rounds sum to 2^n_rounds - 2 against a
# collection round of 2^n_rounds -- into a burn-in of a few autocorrelation times. Measured on dw4
# from the start this generator actually uses, a frozen run reproduces the calibration it came from:
# Lambda 1.6539 against 1.6515, explorer acceptance 0.644-0.649 against 0.644-0.652, round trips
# 0.1338/scan against 0.1394. n_rounds becomes 1 and n_scans sets the length.
schedule, step_size, n_rounds = beta, args.step_size, args.n_rounds
if args.freeze:
    measured = json.loads((cache_dir() / 'plan' / (target.name + '_plan.json')).read_text())
    schedule = torch.tensor(measured['final_beta'], dtype=target.dtype, device=device)
    step_size = torch.tensor(measured['final_step_size'], dtype=target.dtype, device=device)
    n_rounds = 1
    x0 = x0[:len(schedule)] if len(x0) >= len(schedule) else x0
    args.n_rungs = len(schedule)
    args.n_local, args.n_leapfrog = measured['n_local'], measured['n_leapfrog']
    print('frozen from the calibration: %d rungs, beta %.4f .. %.4f, %d scans after %d burn-in'
          % (len(schedule), schedule[0], schedule[-1], args.n_scans or 0, args.n_burn or 0))

start_time = time.time()
samples, diagnostics = non_reversible_parallel_tempering(
    target.log_prob, target.score, x0, schedule,
    explorer=hmc_explorer(n_steps=args.n_local, n_leapfrog=args.n_leapfrog),
    n_rounds=n_rounds, reference=None, step_size=step_size,
    adaptation_gain=0.0 if args.freeze else 1.0, generator=None,
    thin=args.thin, n_scans=args.n_scans, n_burn=args.n_burn or 0, return_diagnostics=True,
)
elapsed = time.time() - start_time

# Report the diagnostics the run has to clear before its output is worth anything. The scans run
# are the discarded doubling rounds, the final round and its burn-in, which a frozen run reduces to
# the last two
n_scans = args.n_scans or 2 ** n_rounds
total_scans = 2 ** n_rounds - 2 + n_scans + (args.n_burn or 0)
print('\n%.1f s for %d scans over %d rounds, collecting %d of the last %d'
      % (elapsed, total_scans, n_rounds, n_scans // args.thin, n_scans))
print('barrier Lambda   %.4f   (recommends %d rungs; this ladder has %d)'
      % (diagnostics['barrier'], diagnostics['recommended_n_rungs'], args.n_rungs))
print('beta            ', [round(float(v), 4) for v in diagnostics['beta']])
print('rejection       ', [round(float(v), 3) for v in diagnostics['rejection']])
print('explorer accept ', [round(float(v), 3) for v in diagnostics['explorer_acceptance']])
print('step size       ', [round(float(v), 5) for v in diagnostics['step_size']])
round_trips = diagnostics['round_trips']
print('round trips per chain (final round): mean %.1f  min %.0f  max %.0f'
      % (float(round_trips.mean()), float(round_trips.min()), float(round_trips.max())))

# The barrier turns the round-trip rate into a number with something to be compared against:
# 1 / (2 + 2 Lambda) is what an ideal local kernel would reach on this path, so the ratio is a
# direct measure of how far the local exploration falls short of refreshing a state per scan
# One machine's round trip is not free even on a perfect ladder: it walks 2 (n_rungs - 1) rungs
# at best and takes n_rungs (2 + 2 Lambda) scans in practice, so the cost of one INDEPENDENT
# configuration is n_rungs^2 (2 + 2 Lambda) n_local energy evaluations. At the optimal n_rungs =
# 2 Lambda that is ~8 Lambda^3 n_local, i.e. CUBIC in the barrier -- which is why the hot end of
# the path is worth choosing carefully and why gen/audit/barrier.py exists
print('one machine round trip: %.0f scans observed, %.0f ideal, %.0f ballistic floor'
      % (diagnostics['scans_per_round_trip'], diagnostics['ideal_scans_per_round_trip'],
         diagnostics['ballistic_scans_per_round_trip']))
print('round trip rate  %.4f   against the %.4f this barrier allows (%.0f%% of it)'
      % (diagnostics['round_trip_rate'], diagnostics['predicted_round_trip_rate'],
         100.0 * diagnostics['round_trip_rate'] / diagnostics['predicted_round_trip_rate']))
print('log Z(1) / Z(%.2f)  %.4f' % (args.beta_min, diagnostics['log_normaliser']))

# A rung whose local kernel never accepts is frozen, and with no reference at beta = 0 there is
# nothing to refresh it: it holds one configuration for the whole run and poisons every swap it
# is offered. The schedule adaptation cannot see this, so it is asserted separately
# The per-rung MEAN hides a single frozen chain among hundreds, and a frozen chain is not
# harmless: every rung starts from the same configuration, so a chain whose local kernel never
# accepts holds twelve identical states, accepts every swap it is offered, and runs its round-trip
# count to the ballistic maximum while contributing one configuration to the output
# Both checks hold under --calibrate too: a frozen chain's constant energy makes the between-chain
# spread read tau as the length of the run, and the production run would be sized from it
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

# A ladder nobody travels is a ladder that did nothing: every chain must have made the round trip
# from the cold rung to the hot one and back, several times over, or the cold samples are just a
# long local chain wearing a tempering costume
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

# What the thinning bought, measured rather than assumed, over the observables the metrics are
# actually built from. The ENERGY is not enough on its own and that is the whole point: on
# ka-binary-N64-T0.2 the energy reads tau = 8 scans while the pair distances read ~88, so an
# energy-only diagnostic calls a set fine when it carries a tenth of its nominal size. Jung et al
# 2024 make the same point for the glass formers -- the energy plateaus long before the ensemble
# does.
#
# The distances are SORTED, because compare_histograms pools every pair of every configuration
# into one histogram and so never sees which pair is which; a distance read at a fixed index
# instead tracks one labelled atom pair and decorrelates only when particles swap places, a
# symmetry the metric quotients out. On LJ-55 that is tau = 63 against 9.
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
# so it disagrees with the autocorrelation function exactly where the ACF is weakest: a lag window
# that is too short reads tau LOW, and a few chains stuck at their own mean read tau HIGH. On
# LJ-55 the two disagreed by a factor of sixty and this one was right.
dispersion = chain_mean_dispersion(observables)
print('\nworst tau %.2f stored states, on %s' % (ess['tau'], ess['name']))
print('between-chain check: tau %.2f on %s (1 is independent)'
      % (dispersion['tau'], dispersion['name']))
print('effective sample size %.0f of %d stored (%.0f%%)'
      % (ess['n_effective'], n_keep * args.n_chains, 100 * ess['efficiency']))
assert args.calibrate or ess['efficiency'] > 0.5, (
    'the stored set is worth %.0f%% of its length: raise --thin to about %d, or raise --n_local '
    'if the round-trip rate is far below what the barrier allows'
    % (100 * ess['efficiency'], int(args.thin * ess['tau']) + 1)
)
if ess['efficiency'] < 0.9:
    # First order only: thinning a series whose autocorrelation has a fat tail does not divide
    # tau by the thinning factor, so this under-shoots and the estimate wants iterating
    print('NOTE: thin %d leaves %.0f%% efficiency; about %d should reach 90%%, iterate from there'
          % (args.thin, 100 * ess['efficiency'], int(args.thin * ess['tau'] / 0.9) + 1))

# Flatten chain-interleaved, so reading the first n samples spans every chain
samples = samples.reshape(-1, target.dim).cpu().numpy()
# Report the ENERGY, not -log_prob: they differ by the temperature wherever one is folded in
print('\ncollected %s from the beta = 1 rung' % (samples.shape,))
head = torch.as_tensor(samples[:1000]).to(device)
energy = target.energy(head) if hasattr(target, 'energy') else -target.log_prob(head)
per_particle = getattr(target, 'n_particles', getattr(target, 'n_atoms', 1))
print('energy: mean %.4f  std %.4f  (per particle %.4f)' % (
    float(energy.mean()), float(energy.std()), float(energy.mean()) / per_particle))

# A calibration run measures and does not produce. It skips the acceptance checks above -- the
# whole point is that its thin is wrong -- writes no reference set, and leaves a plan file that
# --freeze reads back and the production shards are sized from. Its throughput is recorded too,
# because that is a property of the GPU it landed on and not of this machine.
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
folder = cache_dir() / 'pt'
folder.mkdir(parents=True, exist_ok=True)
stem = target.name if args.shard is None else '%s_shard%02d' % (target.name, args.shard)
path = folder / (stem + '.npy')
np.save(path, samples)
protocol = dict(config)
protocol.update({
    'n_keep': n_keep,
    'sampler': 'non_reversible_parallel_tempering', 'n_rungs': args.n_rungs,
    'beta_min': settings['beta_min'], 'initial_step_size': settings['step_size'],
    'final_beta': [float(v) for v in diagnostics['beta']],
    'final_step_size': [float(v) for v in diagnostics['step_size']],
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
    'torch': torch.__version__,
})
(folder / (stem + '_manifest.json')).write_text(manifest(
    target.name, protocol, {path.name: sha256(path)}))
print('wrote %s' % path)
