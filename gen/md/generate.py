# Generate a peptide reference set with replica-exchange molecular dynamics

# Libraries
import argparse, functools, json, pathlib, pprint, sys, time
import numpy as np
import openmm
import torch
from samplingdb import Peptide
from samplingdb.data import cache_dir, sha256
from samplingdb.targets import spec_id
from samplingdb.targets.molecular import PEPTIDE_SYSTEMS
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diagnostics import chain_mean_dispersion, effective_sample_size
from dynamics import harmonic_log_prob, harmonic_score, minimised_positions, run_batched_dynamics
from manifest import manifest
from nrpt import (PotentialReference, dynamics_explorer, linear_schedule,
                  non_reversible_parallel_tempering)

# Per system. n_rungs is ~2 Lambda under SOLUTE tempering, which is where the ladder length is
# set: scaling only the soft terms takes the bond and angle vibrations out of the swap statistic,
# and on these systems they are 1.55x to 1.70x of sd(E) all by themselves. Measured on Ace-A-Nme
# at beta_min 0.4, Lambda goes 2.7556 global -> 1.265 solute, so the ladder halves.
#
# thin is the MEASURED tau in scans. WARNING: tau_scans is not comparable across different thins --
# a fat tail means thinning does not divide tau by the thinning factor, and Ace-A-Nme reads 82 at
# thin 64 against 17 at thin 8. Compare only at equal thin, and let the audit iterate.
#
# beta_min 0.4 is NOT the cheapest hot end for the small peptides and is kept for insurance. Held at
# thin 32 on Ace-A-Nme, tau is flat across beta_min (49.0, 48.6, 45.8, 53.1 scans for 0.55, 0.4,
# 0.3, 0.2) while the round trip more than doubles -- so the ladder is not what decorrelates this
# peptide, the local MD is, and beta_min 0.55 is 1.5x cheaper. It also leaves a two-rung ladder,
# which carries no insurance against a discrete mode the observables do not see.
#
# n_rounds is set from the cost of one round trip, n_rungs * (2 + 2 Lambda) scans, so that the
# collection round carries several per replica rather than none -- the arithmetic that a bare
# "zero round trips" hides. n_replicas is then chosen to put n_rungs * n_replicas near where the
# force field's throughput saturates (~3000 configurations for the small systems, measured),
# because the total work for a given number of independent structures does not depend on how it
# is split between replicas and rounds but the throughput does.
SETTINGS = {
    'Ace-A-Nme': {'n_rungs': 6, 'beta_min': 0.4, 'n_replicas': 512, 'n_rounds': 12, 'thin': 96,
                  'md_steps': 100},
    'AAA': {'n_rungs': 6, 'beta_min': 0.4, 'n_replicas': 512, 'n_rounds': 11, 'thin': 96,
            'md_steps': 100},
    'Ace-AAA-Nme': {'n_rungs': 6, 'beta_min': 0.4, 'n_replicas': 512, 'n_rounds': 11, 'thin': 96,
                    'md_steps': 100},
    'AAAAAA': {'n_rungs': 6, 'beta_min': 0.4, 'n_replicas': 170, 'n_rounds': 11, 'thin': 96,
               'md_steps': 100},
}

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, default='Ace-A-Nme', choices=list(PEPTIDE_SYSTEMS))
parser.add_argument('--tempering', type=str, default='solute', choices=['solute', 'global'])
parser.add_argument('--n_rungs', type=int, default=None)
parser.add_argument('--n_replicas', type=int, default=None)
parser.add_argument('--n_rounds', type=int, default=None)
parser.add_argument('--beta_min', type=float, default=None)
parser.add_argument('--thin', type=int, default=None)
parser.add_argument('--md_steps', type=int, default=None)
parser.add_argument('--step_size', type=float, default=0.001)
parser.add_argument('--equilibrate_ps', type=float, default=50.0)
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
# sizes add and the merge is exact -- which is what makes a run too long for one allocation fit in
# several. The seed follows the shard so two shards never repeat a trajectory.
if args.shard is not None:
    args.seed = args.seed + args.shard

# Fill the unset options from the system's own settings
settings = SETTINGS[args.sequence]
for key, value in settings.items():
    if getattr(args, key) is None:
        setattr(args, key, value)

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Build the target
device = torch.device(args.device)
target = Peptide(sequence=args.sequence).to(device)
print('%s: %d atoms, %d rungs, %d replicas, %d fs per scan'
      % (target.name, target.n_atoms, args.n_rungs, args.n_replicas, args.md_steps))

# The path. SOLUTE tempering scales the torsions, the nonbonded terms and the Born solvation and
# leaves the bonds and angles at full strength, so the swap statistic is V = -E_soft / kT. Those
# stiff terms are the larger half of Var(E) and none of the conformational barriers, so tempering
# the whole potential spends a chunk of the ladder heating up bond vibrations for nothing. The
# conformational enhancement is unchanged: beta multiplies exactly the terms that make the
# barriers. Wang, Friesner and Berne 2011 for the idea, Kamiya and Sugita 2018 for doing it term
# by term.
reference = None if args.tempering == 'global' \
    else PotentialReference(functools.partial(harmonic_log_prob, target),
                            functools.partial(harmonic_score, target))
beta = linear_schedule(args.n_rungs, args.beta_min, dtype=target.dtype, device=device)

# --freeze reuses a calibration's converged schedule and adapts nothing, which replaces the
# doubling's permanent 2x with a burn-in of a few autocorrelation times. The step size was never
# adapted here anyway -- the move is unadjusted, so the timestep is a bias and not a rejection rate
# -- so the schedule is the whole of what carries over.
if args.freeze:
    measured = json.loads((cache_dir() / 'plan' / (target.name + '_plan.json')).read_text())
    beta = torch.tensor(measured['final_beta'], dtype=target.dtype, device=device)
    args.n_rungs = len(beta)
    args.md_steps = measured['md_steps']
print('%s tempering, beta %.3f .. 1.0 (%.0f K effective on the tempered terms)'
      % (args.tempering, args.beta_min, target.temperature / args.beta_min))

# Tempering is over the POTENTIAL, not the thermostat: the local kernel integrates the annealed
# potential at the target's own kT, which is the same annealed density and lets one BAOAB call
# serve every rung. It is also why a swap needs no velocity rescaling -- every rung is Maxwell at
# the same kT -- which is what makes carrying the velocities across scans correct.
#
# HMC is not usable here and that was measured: parallel_tempering's kernel has unit masses on
# Cartesian coordinates, so on a ten-residue peptide its step size adapted down to 1.1e-5 and the
# cold level's radius of gyration was 1.0270 +- 0.0006 at the first sweep and 1.0272 +- 0.0007 six
# hundred sweeps later. A BAOAB integrator with the real masses and a 1 fs step has no such problem.
torch.manual_seed(args.seed)
x0 = minimised_positions(target).to(device).reshape(1, -1)
x0 = x0.expand(args.n_rungs * args.n_replicas, target.dim).contiguous()
print('start E/kT %.3f' % target.energy(x0[:1]), flush=True)

# Equilibrate every replica at its OWN rung's annealed potential before the rounds begin. Without
# it the first rounds -- which are two, four and eight scans long -- estimate the barrier off the
# descent from the minimised structure rather than off the ensemble, and hand the run a schedule
# it then spends the rest of the run recovering from. It does not have to be long, because rounds
# 1 to n_rounds - 1 are themselves burn-in and are half the total work; it only has to be long
# enough that round 1's barrier is measured on an ensemble. Its cost is
# equilibrate_ps * n_rungs * n_replicas / step_size configurations.
beta_row = beta.repeat_interleave(args.n_replicas)[:, None]
if reference is None:
    def annealed_score(z):
        return beta_row * target.score(z)
else:
    def annealed_score(z):
        base = reference.score(z)
        return base + beta_row * (target.score(z) - base)

start = time.time()
x0 = run_batched_dynamics(target, x0, n_steps=int(args.equilibrate_ps / args.step_size),
                                 step_size=args.step_size, score=annealed_score)
x0 = x0.reshape(args.n_rungs, args.n_replicas, target.dim)
print('equilibrated %.0f ps per replica in %.0f s, E/kT %.3f'
      % (args.equilibrate_ps, time.time() - start,
         target.energy(x0[-1]).mean()), flush=True)

# Run: rounds of doubling length, the schedule re-tuned between them, every kept state drawn in
# the final round where nothing is being adapted any more. The move is UNADJUSTED, so the timestep
# is a bias and not a rejection rate and its acceptance is 1 by construction -- adaptation_gain
# must therefore be 0, or the adaptation reads that as room to grow and raises the timestep until
# the integrator is wrong.
dynamics = functools.partial(run_batched_dynamics, target, step_size=args.step_size)
start = time.time()
samples, diagnostics = non_reversible_parallel_tempering(
    target.log_prob, target.score, x0, beta,
    explorer=dynamics_explorer(dynamics, n_steps=args.md_steps),
    n_rounds=1 if args.freeze else args.n_rounds,
    reference=reference, step_size=args.step_size, adaptation_gain=0.0, generator=None,
    thin=args.thin, n_scans=args.n_scans, n_burn=args.n_burn or 0, return_diagnostics=True,
)
elapsed = time.time() - start

# Report the diagnostics the run has to clear before its output is worth anything. The scans run
# are the discarded doubling rounds, the final round and its burn-in, which a frozen run reduces to
# the last two
n_rounds = 1 if args.freeze else args.n_rounds
n_scans = args.n_scans or 2 ** n_rounds
total_scans = 2 ** n_rounds - 2 + n_scans + (args.n_burn or 0)
print('\n%.0f s for %d scans over %d rounds, %.1f ns of cold-rung time per replica'
      % (elapsed, total_scans, n_rounds, n_scans * args.md_steps * args.step_size / 1000.0))
print('barrier Lambda   %.4f   (recommends %d rungs; this ladder has %d)'
      % (diagnostics['barrier'], diagnostics['recommended_n_rungs'], args.n_rungs))
print('beta            ', [round(float(v), 4) for v in diagnostics['beta']])
print('rejection       ', [round(float(v), 3) for v in diagnostics['rejection']])
round_trips = diagnostics['round_trips']
print('round trips per replica (final round): mean %.1f  min %.0f'
      % (float(round_trips.mean()), float(round_trips.min())))

# A round trip is not free even when the ladder is perfect: one machine has to walk 2 (n_rungs - 1)
# rungs at best, and n_rungs (2 + 2 Lambda) scans in practice. Reporting the observed cost against
# both is what separates a badly configured run from a run that is merely too short, which a bare
# count of zero does not
print('one machine round trip: %.0f scans observed, %.0f ideal, %.0f ballistic floor'
      % (diagnostics['scans_per_round_trip'], diagnostics['ideal_scans_per_round_trip'],
         diagnostics['ballistic_scans_per_round_trip']))
print('round trip rate  %.4f   against the %.4f this barrier allows (%.0f%% of it)'
      % (diagnostics['round_trip_rate'], diagnostics['predicted_round_trip_rate'],
         100.0 * diagnostics['round_trip_rate'] / diagnostics['predicted_round_trip_rate']))

assert float(diagnostics['rejection'].max()) < 0.95, (
    'adjacent pair %d rejects %.3f of its swaps: the ladder has a gap'
    % (int(diagnostics['rejection'].argmax()), float(diagnostics['rejection'].max())))
assert args.calibrate or float(round_trips.min()) >= 1.0, (
    'a replica never crossed the ladder: one round trip costs about %.0f scans on this ladder and '
    'the collection round is %d, so raise n_rounds to %d or shorten the ladder'
    % (diagnostics['ideal_scans_per_round_trip'], n_scans,
       args.n_rounds + max(1, int(np.ceil(np.log2(
           diagnostics['ideal_scans_per_round_trip'] / n_scans))))))

# What the stored set is worth, on the observables the upstream panel is built from. The torsions
# are the slow ones -- a peptide's distances relax long before its backbone does
n_keep = len(samples)
flat = samples.reshape(-1, target.dim)
distances = target.preprocess(flat).reshape(n_keep, args.n_replicas, -1)
observables = {'mean pair distance': distances.mean(-1).T}
ordered = torch.sort(distances, dim=-1).values
for fraction in [0.1, 0.5, 0.9]:
    column = int(fraction * (ordered.shape[-1] - 1))
    observables['distance quantile %.0f%%' % (100 * fraction)] = ordered[:, :, column].T
torsions = target.torsions(target.positions(flat)).reshape(n_keep, args.n_replicas, -1)
for column in range(min(4, torsions.shape[-1])):
    observables['torsion %d' % column] = torch.cos(torsions[:, :, column]).T

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
      % (ess['n_effective'], n_keep * args.n_replicas, 100 * ess['efficiency']))
assert args.calibrate or ess['efficiency'] > 0.5, (
    'the stored set is worth %.0f%% of its length: raise --thin to about %d'
    % (100 * ess['efficiency'], int(args.thin * ess['tau']) + 1))

energy = target.energy(flat)
print('energy E/kT %.3f +- %.3f' % (energy.mean(), energy.std()))

# Half against half along the SCAN axis, which is the cheapest test that the collection round is
# stationary rather than still relaxing away from the minimised structure. It is worth having
# because the failure is invisible in every other number here: a run can equalise its rejections,
# make its round trips and report tau ~ 1 while its whole ensemble sits in the basin it started in.
# The threshold is on the standard error of the difference, using the measured tau, so it tightens
# as the run gets longer rather than being a fixed tolerance.
by_scan = energy.reshape(n_keep, args.n_replicas)
first, second = by_scan[:n_keep // 2], by_scan[n_keep // 2:]
drift = float(second.mean() - first.mean())
standard_error = float(energy.std()) * (2.0 * ess['tau'] / energy.numel()) ** 0.5
print('drift over the collection round %+.3f kT, %.1f standard errors'
      % (drift, abs(drift) / max(standard_error, 1e-12)))
assert args.calibrate or abs(drift) < 5.0 * standard_error, (
    'the energy moved %+.3f kT between the first and second half of the collection round, %.1f '
    'standard errors: the run is still equilibrating, so raise --equilibrate_ps or --n_rounds'
    % (drift, abs(drift) / max(standard_error, 1e-12)))

# A calibration run measures and does not produce. It skips the acceptance checks above -- the
# whole point is that its thin is wrong -- writes no reference set, and leaves a plan file that
# --freeze reads back and the production shards are sized from. Its throughput is recorded too,
# because that is a property of the GPU it landed on and not of this machine.
if args.calibrate:
    folder = cache_dir() / 'plan'
    folder.mkdir(parents=True, exist_ok=True)
    measured = {
        'target': target.name, 'generator': 'md', 'sequence': args.sequence,
        'barrier': diagnostics['barrier'],
        'recommended_n_rungs': diagnostics['recommended_n_rungs'],
        'n_rungs': args.n_rungs, 'beta_min': args.beta_min, 'thin': args.thin,
        'n_replicas': args.n_replicas, 'md_steps': args.md_steps,
        'final_beta': [float(v) for v in diagnostics['beta']],
        'tau_stored': ess['tau'], 'tau_scans': ess['tau'] * args.thin,
        'between_chain_tau': dispersion['tau'],
        'scans_per_round_trip': diagnostics['scans_per_round_trip'],
        'ideal_scans_per_round_trip': diagnostics['ideal_scans_per_round_trip'],
        'configurations_per_second':
            total_scans * args.n_rungs * args.n_replicas * args.md_steps / elapsed,
        'energy_mean': float(energy.mean()), 'energy_std': float(energy.std()),
        'energy_drift': drift,
        'wall_clock_seconds': round(elapsed, 1),
    }
    (folder / (target.name + '_plan.json')).write_text(json.dumps(measured, indent=2))
    print('\nwrote the calibration to %s' % (folder / (target.name + '_plan.json')))
    raise SystemExit(0)

# Save, interleaved by scan so reading the first n spans every replica
folder = cache_dir() / 'md'
folder.mkdir(parents=True, exist_ok=True)
stem = target.reference_key() if args.shard is None else '%s_shard%02d' % (target.name, args.shard)
path = folder / (stem + '.npy')
array = samples.reshape(-1, target.dim).cpu().numpy()
np.save(path, array)

# Write the manifest next to it
protocol = dict(config)
protocol.update({'n_samples': len(array), 'n_keep': n_keep,
                 'sampler': 'replica_exchange_dynamics',
                 'integrator': 'baoab-batched', 'n_chains': args.n_replicas,
                 'final_beta': [float(v) for v in diagnostics['beta']],
                 'rejection': [float(v) for v in diagnostics['rejection']],
                 'barrier': diagnostics['barrier'],
                 'recommended_n_rungs': diagnostics['recommended_n_rungs'],
                 'round_trip_rate': diagnostics['round_trip_rate'],
                 'predicted_round_trip_rate': diagnostics['predicted_round_trip_rate'],
                 'scans_per_round_trip': diagnostics['scans_per_round_trip'],
                 'ideal_scans_per_round_trip': diagnostics['ideal_scans_per_round_trip'],
                 'min_round_trips': float(round_trips.min()),
                 'stored_tau': ess['tau'], 'between_chain_tau': dispersion['tau'],
                 'n_effective': ess['n_effective'],
                 'efficiency': ess['efficiency'], 'energy_drift': drift,
                 'energy_mean': float(energy.mean()), 'energy_std': float(energy.std()),
                 'wall_clock_seconds': round(elapsed, 1),
                 'torch': torch.__version__, 'openmm': openmm.__version__})
(folder / (stem + '_manifest.json')).write_text(manifest(
    target.name, spec_id(target.spec()), protocol, {path.name: sha256(path)}))
print('wrote %s  %s in %.0f s' % (path, array.shape, elapsed))
