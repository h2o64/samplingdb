# Round-trip a baseline sampler through a target, its metric panel and its raw histograms

# Libraries
import pprint
import torch
from samplingdb import DW4, GMM40, KobAndersen2D, ManyWell, catalog
from samplingdb.baselines import hmc
from samplingdb.targets.base import generator

# The catalog is the one list of what this library ships
print(len(catalog()), 'targets;', len(catalog(data=False)), 'of them build with no download')

# Build the target
target = GMM40(dim=2)
print('target', type(target).__name__, 'dim', target.dim, 'name', target.name)

# Look at the two shipped splits
train = target.train_samples(n=2048)
test = target.test_samples(n=2048)
print('train modes visited', len(target.mode_labels(train).unique()), 'of', target.n_modes)
print('test  modes visited', len(target.mode_labels(test).unique()), 'of', target.n_modes)

# Run one baseline sampler, using the analytic score. A target carries no reference distribution
# of its own -- prior_scale is the hint it offers about what scale a Gaussian one would need
x0 = target.prior_scale * torch.randn(2048, target.dim, generator=generator(1),
                                      dtype=torch.float64)
x = hmc(target.log_prob, target.score, x0, n_steps=200, step_size=0.1, generator=generator(0))

# Score it: the universal panel plus the mixture panel, and the raw mode weights on the side
ret, hist = target.compute_metrics(x, generator=generator(0), return_hist=True)
pprint.pprint({name: round(float(value), 4) for name, value in ret.items()})
print('empirical mode weights', torch.round(hist[0], decimals=3).tolist())

# Same call shape on a target whose panel is per well, with the raw histogram it rests on
well = ManyWell(dim=8)
ret, hist = well.compute_metrics(well.train_samples(n=2048), compute_standard_metrics=False,
                                 return_hist=True)
print('\nManyWell train split:', {name: round(float(value), 4) for name, value in ret.items()})
print('per-well weights (shallow, deep)', torch.round(hist, decimals=3).tolist())

# And on a particle system, where the panel returns the distance and energy histograms
particles = DW4()
reference = particles.train_samples(n=4096)[2048:]
ret, hist = particles.compute_metrics(reference + 0.05, ref_samples=reference,
                                      compute_standard_metrics=False, return_hist=True)
print('\nDW4 histogram panel:', {name: round(float(value), 4) for name, value in ret.items()})
print('distance histogram over', len(hist['dist']['edges']) - 1, 'bins in',
      (round(float(hist['dist']['edges'][0]), 3), round(float(hist['dist']['edges'][-1]), 3)))
print('energy histogram over', len(hist['energy']['edges']) - 1, 'bins in',
      particles.energy_histogram_range)

# And on a glass former. test_samples() is this library's own swap-Monte-Carlo set, which
# gen/swap_mc/generate.py produces; released_samples() is the published Zenodo one, and is what
# this example scores against so it needs no generation run
glass = KobAndersen2D(temperature=1.0)
print('\nKobAndersen2D N', glass.n_particles, 'box %.4f' % glass.box, 'name', glass.name)

# The first call downloads ~4 MB from Zenodo 10.5281/zenodo.17966995 and caches it
reference = glass.released_samples(n=512)
ret, hist = glass.compute_metrics(reference[:256], ref_samples=reference[256:],
                                  compute_standard_metrics=False, return_hist=True)
print('  E/N %.4f' % (glass.energy(reference).mean() / glass.n_particles))
print(' ', {name: round(float(value), 4) for name, value in ret.items()})
g, centres = hist['rdf']['g'], hist['rdf']['centres']
print('  g(r) peaks at r = %.3f with g = %.2f' % (centres[g.argmax()], g.max()))
