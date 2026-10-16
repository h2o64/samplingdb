# Merge the shards of a sharded generator run into one reference set, and audit the result

# Libraries
import argparse, json, pprint
import numpy as np
import torch
from samplingdb import Peptide, build
from samplingdb.data import cache_dir, manifest, sha256
from samplingdb.metrics import chain_mean_dispersion, effective_sample_size

# Where each generator writes, and the manifest key naming how many chains it interleaved
FOLDERS = {'pt': 'n_chains', 'md': 'n_replicas'}

# The parallel-tempering generator's names that are not catalog keys. The MD generator names a
# peptide by its sequence, which builds directly
CATALOG_KEYS = {'ka-binary-N64-T0.2': 'ka-binary-T0.2'}

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--folder', type=str, default='pt', choices=list(FOLDERS))
parser.add_argument('--keep_shards', action='store_true')
parser.add_argument('--min_samples', type=int, default=100000)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# A shard is an independent run of the same sampler under a different seed, so the shards are
# independent samples of the same density and their effective sizes ADD. That is the whole reason
# sharding is exact: nothing has to be stitched, only concatenated, and the audit is re-run on the
# result rather than inherited from the parts.
device = torch.device(args.device)
if args.folder == 'md':
    target = Peptide(sequence=args.target).to(device)
else:
    target = build(CATALOG_KEYS.get(args.target, args.target)).to(device)
folder = cache_dir() / args.folder
paths = sorted(folder.glob(target.name + '_shard*.npy'))
assert paths, 'no shards for %s in %s' % (target.name, folder)

# Keep each shard whole: shards that ran to a wall clock rather than a scan count stored different
# numbers of states, and the spacing the merge keeps is only chosen further down
blocks, protocols = [], []
for path in paths:
    protocol = json.loads((folder / (path.stem + '_manifest.json')).read_text())['protocol']
    protocols.append(protocol)
    n_chains = protocol[FOLDERS[args.folder]]
    array = np.load(path)
    blocks.append(torch.as_tensor(array.reshape(-1, n_chains, array.shape[-1]),
                                  dtype=target.dtype, device=device))
    print('%s  %s  %d chains' % (path.name, array.shape, n_chains))


def observe(block):
    """Evaluates every audited observable on one shard's stored states

    Args:
        * block (torch.Tensor of shape (n_keep, n_chains, dim)): One shard, scan-major

    Returns:
        * observables (dict): Name -> torch.Tensor of shape (n_keep, n_chains)
    """
    n_keep, n_chains = block.shape[:2]
    states = block.reshape(-1, block.shape[-1])
    energy = target.energy(states) if hasattr(target, 'energy') else -target.log_prob(states)
    observables = {'energy': energy.reshape(n_keep, n_chains)}
    distances = target.distances(states) if hasattr(target, 'distances') \
        else target.preprocess(states)
    distances = distances.reshape(n_keep, n_chains, -1)
    observables['mean pair distance'] = distances.mean(-1)
    ordered = torch.sort(distances, dim=-1).values
    for fraction in [0.1, 0.5, 0.9]:
        column = int(fraction * (ordered.shape[-1] - 1))
        observables['distance quantile %.0f%%' % (100 * fraction)] = ordered[:, :, column]
    if hasattr(target, 'torsions'):
        torsions = target.torsions(target.positions(states)).reshape(n_keep, n_chains, -1)
        for column in range(min(4, torsions.shape[-1])):
            observables['torsion %d' % column] = torch.cos(torsions[:, :, column])
    return observables


def keep(values, k, m):
    """Returns the first m of every k-th stored state, the first of them being the k-th"""
    return values[k - 1 + k * torch.arange(m, device=values.device)]


# Choose the spacing from the data: the densest one, every k-th stored state, at which the merged
# set is worth 90% of its length by BOTH estimates and has forgotten its start. Shards that ran to a
# wall clock stored more states on a faster GPU, so the gate reads the states every shard has, at
# the same scans, and the longer shards then add their later ones at the same spacing. The
# autocorrelation reads tau LOW when a few chains hold their own mean for the whole run, and the
# spread of the chain means catches exactly that, so the gate takes the worse of the two.
# When no spacing leaves two states per chain that pass, the last state of every chain is kept:
# the chains are independent by construction, so that set needs no autocorrelation to vouch for it
per_shard = [observe(block) for block in blocks]
lengths = [len(block) for block in blocks]
choice = None
for k in range(1, max(lengths) + 1):
    m = min(length // k for length in lengths)
    if m < 2:
        break
    observables = {name: torch.cat([keep(shard[name], k, m) for shard in per_shard], dim=1).T
                   for name in per_shard[0]}
    n_chains = observables['energy'].shape[0]

    # Whether the start is forgotten: each observable's first tenth of kept states against its
    # last half, chain by chain. Chains are independent, so the spread of their differences is an
    # honest standard error whatever the autocorrelation, and a transient shows up directly instead
    # of through the efficiency. AAAAAA's first production run drifted 52 standard errors on
    # cos phi5 here while its efficiency read 73%
    head = max(1, m // 10)
    drifts = {}
    for name, values in observables.items():
        difference = values[:, :head].mean(1) - values[:, m // 2:].mean(1)
        drifts[name] = float(difference.mean() / (difference.std() / n_chains ** 0.5))
    drift_name = max(drifts, key=lambda name: abs(drifts[name]))
    ess = effective_sample_size(observables)
    dispersion = chain_mean_dispersion(observables)
    efficiency = min(ess['efficiency'], 1.0 / max(dispersion['tau'], 1.0))
    print('every %3d stored: %4d per chain, efficiency %.0f%% (tau %.2f on %s, between-chain %.2f '
          'on %s), start drift %+.1f SE on %s'
          % (k, m, 100 * efficiency, ess['tau'], ess['name'], dispersion['tau'], dispersion['name'],
             drifts[drift_name], drift_name))
    if efficiency > 0.9 and abs(drifts[drift_name]) < 5.0:
        choice = k
        break
if choice is None:
    ess = dispersion = drifts = drift_name = efficiency = None
    print('no spacing passes with two states per chain: keeping the last state of every chain')

# Scan-major and chain-interleaved, so the first n rows span every chain: the states every shard
# has first, then the later ones of the longer shards
if choice is None:
    kept = [block[-1:] for block in blocks]
    choice, m = 0, 1
else:
    kept = [keep(block, choice, len(block) // choice) for block in blocks]
merged = torch.cat([shard[:m] for shard in kept], dim=1)
n_keep, n_chains = merged.shape[:2]
states = merged.reshape(-1, merged.shape[-1])
later = [shard[index] for index in range(m, max(len(shard) for shard in kept))
         for shard in kept if len(shard) > index]
flat = torch.cat([states] + later).cpu().numpy()
energy = target.energy(states) if hasattr(target, 'energy') else -target.log_prob(states)
print('\nmerged %d shards -> %s: %d states x %d chains every %d stored, then %d later states'
      % (len(paths), flat.shape, n_keep, n_chains, choice, len(flat) - len(states)))

# A fold, when the generator audited one: its atom pairs and which columns started folded are in
# every shard's manifest, so the gate gen/md/generate.py applied per shard is re-run on the whole
fold_pairs = protocols[0].get('fold_pairs')
if fold_pairs is not None:
    p = states.reshape(len(states), -1, 3)
    folding = torch.stack([torch.linalg.norm(p[:, i] - p[:, j], dim=-1) for i, j in fold_pairs],
                          dim=-1).reshape(n_keep, n_chains, -1)
    started_folded = torch.tensor(sum((protocol['started_folded'] for protocol in protocols), []),
                                  device=device)
    per_column = (folding[:, :, 0] < protocols[0]['compact']).to(energy.dtype).mean(0)
    from_folded, from_extended = per_column[started_folded], per_column[~started_folded]
    gap = float(from_folded.mean() - from_extended.mean())
    error = float((from_folded.var() / len(from_folded)
                   + from_extended.var() / len(from_extended)) ** 0.5)
    print('compact fraction %.3f from folded starts, %.3f from extended ones, %.1f standard errors'
          ' apart' % (from_folded.mean(), from_extended.mean(), abs(gap) / max(error, 1e-12)))
    assert abs(gap) <= 4.0 * error, (
        'the merged set still remembers its starts: %.3f compact from folded ones against %.3f from '
        'extended ones' % (from_folded.mean(), from_extended.mean()))

if ess is not None:
    print('worst tau %.2f stored states, on %s' % (ess['tau'], ess['name']))
    print('between-chain check: tau %.2f on %s (1 is independent)'
          % (dispersion['tau'], dispersion['name']))
    print('efficiency %.0f%%, start drift %+.1f standard errors on %s'
          % (100 * efficiency, drifts[drift_name], drift_name))
print('energy %.4f +- %.4f' % (energy.mean(), energy.std()))

assert len(flat) >= args.min_samples, (
    'the merged set is %d samples and %d were asked for: run more shards'
    % (len(flat), args.min_samples))

# One file and one manifest, with every shard's protocol kept so the run is reproducible
path = folder / (target.name + '.npy')
np.save(path, flat)
protocol = {'sampler': protocols[0]['sampler'], 'n_shards': len(paths), 'shards': protocols,
            'n_samples': len(flat), 'n_keep': n_keep, FOLDERS[args.folder]: n_chains,
            'every_stored': choice, 'last_state_only': ess is None,
            'energy_mean': float(energy.mean()), 'energy_std': float(energy.std()),
            'torch': torch.__version__}
if ess is not None:
    protocol.update({'stored_tau': ess['tau'], 'between_chain_tau': dispersion['tau'],
                     'n_effective': ess['n_effective'], 'efficiency': efficiency,
                     'start_drift': drifts[drift_name]})
if fold_pairs is not None:
    protocol['compact_by_start'] = [float(from_folded.mean()), float(from_extended.mean())]
(folder / (target.name + '_manifest.json')).write_text(manifest(
    target.name, protocol, {path.name: sha256(path)}))
print('wrote %s  %s' % (path, flat.shape))

if not args.keep_shards:
    for path in paths:
        path.unlink()
        (folder / (path.stem + '_manifest.json')).unlink()
    print('removed %d shard files; pass --keep_shards to keep them' % len(paths))
