# samplingdb

Unnormalised sampling targets for PyTorch. Each target has an analytic score, a cheap biased
**train** split, an accurate **test** split, and a `compute_metrics` panel that scores samples,
optionally weighted, against that test split.

## Install

```sh
pip install -e .                  # torch, numpy, scipy, pot
pip install -e ".[molecular]"     # adds openmm, mdtraj, huggingface_hub, deeptime: the peptides
```

Downloads, train splits and reference sets are cached in `~/.cache/samplingdb`. Set
`SAMPLINGDB_CACHE` to put the cache somewhere else.

## Quickstart

```python
import torch
from samplingdb import build, catalog
from samplingdb.baselines import hmc

print(catalog())                        # every target, under the name it is cited by
target = build('gmm40')

train = target.train_samples(n=2048)    # cheap and biased: one short chain per mode
test = target.test_samples(n=2048)      # exact draws

x0 = target.prior_scale * torch.randn(2048, target.dim, dtype=torch.float64)
x = hmc(target.log_prob, target.score, x0, n_steps=200, step_size=0.1)

metrics, hist = target.compute_metrics(x, return_hist=True)
# metrics: ks, w2_sliced, emc, kl_weights, tv_weights, perc_forgotten_modes
# hist: the empirical weight of each of the 40 modes
```

`examples/quickstart.py` runs the same round trip on a mixture, ManyWell, DW-4 and a glass former,
and `examples/peptides.py` on a peptide.

## Documentation

- [The Target interface](docs/interface.md): scores, batch sizes, devices, names, the catalog and
  the cache
- [Targets](docs/targets/README.md): every target, its splits and its metric panel
- [Metrics](docs/metrics.md): the standard panel, the target panels and their binning
- [Baselines](docs/baselines.md): the classical samplers, parallel tempering and swap Monte Carlo
- [Reference sets](docs/reference-sets.md): how the test splits are generated and checked

## Tests

```sh
pytest tests -q
```

Tests that need a download or a generated reference set skip without it, and the summary lists
why. Point `SAMPLINGDB_CACHE` at a populated cache to run all of them.
