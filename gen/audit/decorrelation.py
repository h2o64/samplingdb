# Measure how decorrelated a generated reference set actually is, along its own chain axis

# Libraries
import argparse, json, pathlib, pprint, sys
import numpy as np
import torch
from samplingdb import build
from samplingdb.data import cache_dir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diagnostics import effective_sample_size

# Where each generator writes, and the manifest key naming how many chains it interleaved
FOLDERS = {'pt': 'n_chains', 'swapmc': 'n_chains', 'md': 'n_replicas', 'stan': None}


def observables(x, target):
    """Returns the scalar time series a reference set has to be decorrelated in

    The energy is the global one, and the pair distances are what every coordinate-space metric in
    the library is built from, so a set decorrelated in these is decorrelated in what it is scored
    on. Distances come from the target's own representation where it has one -- PeriodicPairMixture
    applies the minimum image and Peptide.preprocess IS the distance map, and a hand-rolled
    triangle would get both wrong.

    WARNING: for anything with a particle structure the raw coordinates are deliberately NOT
    reported. A cluster's centre of mass and its orientation both diffuse freely, and neither is
    part of the density: an unpreprocessed radius of gyration reads tau = 429 sweeps on DW-4
    against 5.5 once centred, and a Cartesian coordinate of LJ-13 reads 21 against 11 for the pair
    distances it is actually scored on. Those numbers measure the symmetry, not the sampler.

    Args:
        * x (torch.Tensor of shape (n_frames, n_chains, dim)): The reference set, unflattened
        * target (Target): The target it belongs to

    Returns:
        * series (dict): Name -> numpy.ndarray of shape (n_chains, n_frames)
    """
    n_frames, n_chains, dim = x.shape
    flat = x.reshape(-1, dim)
    energy = target.energy(flat) if hasattr(target, 'energy') else -target.log_prob(flat)
    series = {'energy': energy.reshape(n_frames, n_chains).T}

    n_particles = getattr(target, 'n_particles', getattr(target, 'n_atoms', 1))
    if n_particles > 1:
        if hasattr(target, 'distances'):
            distances = target.distances(flat)
        else:
            distances = target.preprocess(flat)
        distances = distances.reshape(n_frames, n_chains, -1)
        series['mean pair distance'] = distances.mean(-1).T

        # SORTED, because compare_histograms pools every pair of every configuration into one
        # histogram: the metric never sees which pair is which. A distance read at a fixed INDEX
        # instead tracks one labelled atom pair, and decorrelates only when the particles permute
        # -- a symmetry the metric quotients out, so it reads slow for a reason that is not the
        # density. On LJ-55 a fixed-index distance reads tau = 65 sweeps where the
        # permutation-invariant mean over the same distances reads 8.5
        ordered = torch.sort(distances, dim=-1).values
        for fraction in [0.1, 0.3, 0.6, 0.9]:
            k = int(fraction * (ordered.shape[-1] - 1))
            series['distance quantile %.0f%%' % (100 * fraction)] = \
                ordered[:, :, k].T
    else:
        for k in range(min(4, dim)):
            series['coordinate %d' % k] = x[:, :, k].T
    return series


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--folder', type=str, default='pt', choices=list(FOLDERS))
parser.add_argument('--n_chains', type=int, default=None)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Read the reference set and the chain count the generator interleaved it by. The catalog name is
# not the file name -- every generator writes target.reference_key(), so the target has to be built
# first
target = build(args.target).to(torch.device(args.device))
folder = cache_dir() / args.folder
array = np.load(folder / (target.reference_key() + '.npy'))
n_chains = args.n_chains
if n_chains is None:
    # Stan's draws are randomly PERMUTED before saving, so they have no chain axis left and an
    # autocorrelation along the file would read 1 whatever the sampler did. Its manifest records
    # min_ess_bulk instead, which is the honest number and is already gated in gen/stan
    assert FOLDERS[args.folder] is not None, (
        'the ' + args.folder + ' sets are permuted before saving, so there is no ordering to '
        'measure. Read min_ess_bulk out of the manifest instead, or pass --n_chains to override.')
    manifest = json.loads((folder / (target.reference_key() + '_manifest.json')).read_text())
    n_chains = manifest['protocol'][FOLDERS[args.folder]]
    thin = manifest['protocol'].get('thin', 1)
    print('manifest: %d chains, thin %s' % (n_chains, thin))

# Unflatten the interleaving the generators write: index = frame * n_chains + chain
n_frames = len(array) // n_chains
x = torch.as_tensor(array[:n_frames * n_chains], dtype=target.dtype, device=target.device)
x = x.reshape(n_frames, n_chains, target.dim)
print('%s  %d frames x %d chains x %d dims' % (target.name, n_frames, n_chains, target.dim))

# The autocorrelation is along the FRAME axis, per chain: consecutive rows of the flat file are
# different chains and are independent, so reading the file as one series would say nothing
ess = effective_sample_size(observables(x, target))
print('\n%-24s %10s %12s %10s' % ('observable', 'tau', 'ess', 'ess / n'))
for name, tau in ess['tau_per_observable'].items():
    print('%-24s %10.2f %12.0f %10.3f' % (name, tau, len(array) / tau, 1.0 / tau))
worst, worst_name = ess['tau'], ess['name']

# A stored set is decorrelated when tau is 1 in STORED units: the thinning already paid for it
print('\nworst tau %.2f stored frames, on %s' % (worst, worst_name))
print('effective sample size %.0f of %d stored (%.1f%%)' % (
    len(array) / worst, len(array), 100.0 / worst))
if worst > 1.2:
    print('the spacing is too short by %.1fx: regenerate with thin (or spacing) that much larger'
          % worst)

# An autocorrelation time is only estimable from a series many times longer than itself. Below
# ~50 frames per chain the window lands on noise and reads HIGH, so a short file looks worse than
# it is -- a 20-frame dw4 run reads 1.51 where 4000 sweeps of the same protocol read 0.60
if n_frames < 50 * worst:
    print('WARNING: %d frames per chain is too few to estimate tau %.2f -- this reads high. '
          'Audit a full-length run.' % (n_frames, worst))
