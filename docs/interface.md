# The Target interface

Every target is a `samplingdb.targets.Target`, a `torch.nn.Module` whose constants are registered
buffers.

| Member | What it gives |
| --- | --- |
| `log_prob(x)` | log of the unnormalised density, shape `(n,)` |
| `score(x)` | its gradient, shape `(n, dim)` |
| `log_prob_and_grad(x)` | both, sharing work where the target can |
| `train_samples(n=None)` | the cheap, biased split |
| `test_samples(n=None)` | the reference split |
| `compute_metrics(samples, weights=None, ...)` | the metric panel, see [Metrics](metrics.md) |
| `preprocess(x)` | the representation the metrics compare in |
| `sample(shape, generator=None)` | exact draws, on the synthetic targets only |
| `dim`, `name`, `dtype` | dimension, short identifier, working dtype |
| `truth_tier`, `has_ground_truth` | how the reference set exists, see [Targets](targets/README.md) |
| `prior_scale` | the scale a Gaussian reference distribution would need |
| `log_Z` | the log-normaliser, where it is known |

## Scores are analytic

Every target writes its score out by hand, and nothing in the library differentiates through
`log_prob`. A mixture's `log_prob_and_grad` computes both from one softmax, ManyWell and Funnel
use their closed forms, and the particle systems use the radial pair force with the diagonal
masked out. `targets.score_autograd` exists only so `tests/test_scores.py` can check every score against
autograd.

The peptides are the one exception: their score is autograd through the closed-form force-field
energy. It is exact to machine precision, and it keeps the score itself differentiable for a loss
built on it (`differentiable_score=True`, off by default because it retains a backward graph).

## No reference distribution

A target carries the density and nothing about how to sample it. An annealed sampler builds its own
interpolation, `beta * score(x) + (1 - beta) * grad log reference(x)`, and `prior_scale` is the
scale a Gaussian reference would need.

## Any batch size, any device

`log_prob` and `score` take any number of configurations. A density whose temporaries grow faster
than its input (an all-pairs tensor, a per-mode Gaussian, a per-datapoint logit) evaluates itself
in slices that stay under `Target.max_elements`, 2^27 elements or a gigabyte of float64. The
slicing is exact, and the public method calls `chunked` on a `*_chunk` method that holds the
arithmetic, so a subclass overrides the chunk and inherits the slicing. The peptides do the same
inside `TorchForceField` with `chunk_size`.

`target.to('cuda')` moves the density. The metric panels run on the device of the samples they are
given.

## Working dtype

The particle clusters, the glass formers and the peptides work in float32: on real configurations
their energies agree with float64 to about 1e-4 kT, and they run 2x to 11x faster. The synthetic
targets and the posteriors stay in float64, because their exactness checks (analytic score against
autograd) are not precision-tolerant. Every constructor takes `dtype`.

## Names, specifications and the cache

`name` is a short lowercase identifier that carries only what changes the density, with no version
tag: `lj13`, `ka-binary-N64-T0.2`, `lgcp-whitened`. `spec()` returns every value that defines the
density, `spec_id` hashes it, and the cached train split is keyed by name, `spec_id` and `n`, so a
changed constant is a cache miss rather than a stale array. The reference set is keyed the same
way, as `reference_key()`, `<name>_<spec_id>`. A new constructor argument that moves
`log_prob` must appear in that target's `spec()`.

## The catalog

```python
from samplingdb import CATALOG, build, catalog

catalog()                       # every catalogued name, in catalog order
catalog(data=False)             # the 15 that build with no download
build('lj13')                   # one target, by the name it is cited by
build('aldp', compile=False)    # extra arguments override the catalogued ones
```

`CATALOG` is the one list of what the library ships. A target that is only a parameterisation of
another (a peptide sequence, a glass temperature, the whitened LGCP) is an entry there rather than
a subclass.

## Data and cache

| Variable | Use |
| --- | --- |
| `SAMPLINGDB_CACHE` | where downloads, train splits and reference sets live (default `~/.cache/samplingdb`) |
| `SAMPLINGDB_DATA_DIR` | a directory of local copies, used by file name instead of downloading; checksums are still verified |

The reference set behind `test_samples()` downloads itself. The first call fetches the published
set from the Hugging Face dataset [`h2o64/samplingdb`](https://huggingface.co/datasets/h2o64/samplingdb),
checks its sha256, and saves it in the cache; every later call reads the cache. Nothing needs an
account or a token.

```python
from samplingdb import build

target = build('lj55')
x = target.test_samples(n=10000)    # Downloading pt/lj55_dc3f6b7e.npy ..., once
target.reference_path()             # where it now sits in the cache
```

The sets take 3.1 GB together, of which the two LGCP posteriors are 1.3 GB each; the others run
from 3 MB to 76 MB. To fetch them ahead of time, for instance on a login node before a job runs
where there is no internet access, call `download_references`. It skips whatever is already
cached:

```python
import samplingdb

samplingdb.download_references(['lj55', 'aldp'])    # by catalog name
samplingdb.download_references()                    # every published set
```

The cache is laid out like the dataset repo (`pt/`, `md/`, `stan/`, `swapmc/`), so a copy of the
repo is a ready cache, for a machine that never gets internet access:

```sh
hf download h2o64/samplingdb --repo-type dataset --revision v1 --local-dir "$SAMPLINGDB_CACHE"
```

The dataset page also has a viewer: every set is a subset there, named as in `build`, with each
sample and its `log_prob` on a row. The same Parquet copies load without samplingdb, through
`datasets.load_dataset('h2o64/samplingdb', 'lj55', split='test')`.

The library reads one tag of that repo, `REFERENCE_REVISION` in `samplingdb/data.py`, and holds the
sha256 of every file in `REFERENCE_CHECKSUMS`, so an upgrade never changes a set under a running
benchmark. A set generated locally with the scripts in `gen/` lands in the same place and is read
before anything is downloaded; see [Reference sets](reference-sets.md).

A target whose set is neither cached nor published, today the binary glass and chignolin, raises a
`FileNotFoundError` that says where it looked and names the command producing the set. Asking for
more rows than a set holds raises a `ValueError` naming the command for a larger one.

Third-party inputs are fetched from whoever published them. The GitHub sources are pinned to a
commit and a sha256 (the published Tier 1 splits from `jarridrb/DEM`, the Tier 2 datasets from
`DenisBless/variational_sampling_methods`), the released glass configurations come from Zenodo
record 17966995 with its md5 digests, and the peptide topologies and trajectories come from the
Hugging Face dataset `transferable-samplers/sequential-boltzmann-generators-data`.
