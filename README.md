# samplingdb

[![Reference sets on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/h2o64/samplingdb)

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

## Reference sets

The test splits that are not exact draws (particle systems, Bayesian posteriors, peptides, glass
formers) are long MCMC runs made by this library and published on Hugging Face:
[**h2o64/samplingdb**](https://huggingface.co/datasets/h2o64/samplingdb). `test_samples()`
downloads a set the first time it needs it, checks its sha256 and caches it; no account is needed.
The dataset page has a viewer for browsing every set and its log density, and the card explains
each file.

```python
import samplingdb

x = samplingdb.build('lj55').test_samples(n=10000)    # downloads the set once (66 MB)
samplingdb.download_references()                      # or fetch all 17 ahead of time (3.1 GB)
```

See [Data and cache](docs/interface.md#data-and-cache) for offline machines, and
[Reference sets](docs/reference-sets.md) for how each set was generated and checked.

## Documentation

- [The Target interface](docs/interface.md): scores, batch sizes, devices, names, the catalog and
  the cache
- [Targets](docs/targets/README.md): every target, its splits and its metric panel
- [Metrics](docs/metrics.md): the standard panel, the target panels and their binning
- [Baselines](docs/baselines.md): the classical samplers, parallel tempering and swap Monte Carlo
- [Reference sets](docs/reference-sets.md): how the test splits are generated, checked and
  published

## Tests

```sh
pytest tests -q
```

Tests that need a download or a reference set skip without it, and the summary lists why. The
suite never downloads a reference set itself: run `samplingdb.download_references()` once, or point
`SAMPLINGDB_CACHE` at a populated cache, to run all of them.
