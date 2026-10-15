# Score the biased MD split of a peptide against its released reference trajectory

# Libraries
import torch
from samplingdb import Peptide
from samplingdb.targets.molecular import PEPTIDE_SYSTEMS

# The five released systems, and the two force-field families they split into
print('%-12s %-38s %5s %6s %8s' % ('sequence', 'name', 'atoms', 'T', 'torsions'))
targets = {}
for sequence in PEPTIDE_SYSTEMS:
    # The first call downloads this system's PDB from the SBG dataset on Hugging Face.
    # compile=False because this script evaluates a handful of batches and would otherwise spend
    # most of its time compiling kernels it uses once; leave it on for a real run.
    targets[sequence] = Peptide(sequence=sequence, compile=False)
    target = targets[sequence]
    print('%-12s %-38s %5d %6.1f %8d' % (sequence, target.name, target.n_atoms,
                                         target.temperature, len(target.torsion_indexes)))

# Take alanine dipeptide, whose released trajectory is the cheapest to pull
target = targets['Ace-A-Nme']
reference = target.test_samples(n=8000)
train = target.train_samples(n=500)
print('\n%-16s %-24s %s' % ('split', 'E/kT', 'phi > 0'))
for name, samples in [('released test', reference), ('100 ps MD train', train)]:
    energy = target.energy(samples).detach()
    phi = target.torsions(samples)[:, 0]
    print('%-16s %8.2f +- %5.2f        %.4f' % (name, energy.mean(), energy.std(),
                                                float((phi > 0).double().mean())))

# Read every divergence next to the floor its own reference sets
floor = target.compute_metrics(reference[:4000], ref_samples=reference[4000:],
                               compute_standard_metrics=False)
biased, hist = target.compute_metrics(train, ref_samples=reference[:4000],
                                      compute_standard_metrics=False, return_hist=True)
print('\n%-24s %10s %10s' % ('metric', 'test/test', 'train/test'))
for name in ['kl_phi1', 'kl_psi1', 'w1_phi1', 'w1_psi1', 'js_ramachandran',
             'ramachandran_occupancy', 'energy_total_var', 'energy_w1']:
    print('%-24s %10.4f %10.4f' % (name, floor[name], biased[name]))

# The plot-grade Ramachandran histograms come back on the corpus's 200 x 200 grid
joint = hist['torsions']['ramachandran']
print('\nramachandran grid', tuple(joint['hist'].shape),
      'occupied cells: train %d, reference %d' % (int((joint['hist'] > 0).sum()),
                                                  int((joint['ref_hist'] > 0).sum())))

# The transferable-samplers panel, on whichever device the target is on
target = target.to('cuda' if torch.cuda.is_available() else 'cpu')
reference, train = reference.to(target.device), train.to(target.device)
floor = target.compute_upstream_metrics(reference[:4000], reference[4000:])
biased, upstream = target.compute_upstream_metrics(train, reference[:4000], return_hist=True)
print('\n%-24s %10s %10s   (on %s)' % ('upstream metric', 'test/test', 'train/test',
                                       target.device))
for name in ['energy_w1', 'energy_w2', 'torus_w2', 'tica_w2', 'tica_k_jsd', 'torus_k_jsd']:
    print('%-24s %10.4f %10.4f' % (name, floor[name], biased[name]))

# Their five figures come back as arrays, one Ramachandran per residue
print('\nhistograms: ' + ', '.join(
    '%s %s' % (name, tuple(block['hist'][0].shape if name == 'ramachandran'
                           else block['hist'].shape)) for name, block in upstream.items()))
