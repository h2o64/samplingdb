# AGENTS.md — samplingdb

Instructions for an agent working in this repository. Read this before editing anything here.

## What this is

`samplingdb` holds four things and nothing else: unnormalised targets, their reference sets, the
generators that make those sets, and the metrics that score samples against them. The package
(`samplingdb/`) is the targets, the metrics and the code that fetches the sets. The generators and
every sampler behind them live in `gen/`, which is never installed. A sampler that is not needed to
build a train split or a reference set does not belong in either.

It settles each target's conventions against a named reference implementation:

- **`jarridrb/DEM`** — the reference for the particle targets' binning conventions (see below), the
  publisher of the Tier 1 splits that settled their pair conventions (fetched by the tests from a
  pinned commit), and, via bgflow, the authority on those conventions: `MultiDoubleWellPotential`
  counts each pair once, `LennardJonesPotential` with `energy_factor=1.0` counts each twice. The
  published LJ formula disagrees with the latter and with the data, and the data decided.
- **`transferable-samplers/transferable-samplers`** (`data/single_peptide_datamodule.py`,
  `data/energy/openmm_energy.py`) — the implementation the peptide trajectories were generated
  against, and therefore the authority on their two force-field families, on the unit conventions
  (positions in nm, energies in kT), on the TICA fit and on the chirality fix. Its
  per-configuration OpenMM loop is NOT followed: `forcefield.py` evaluates the same systems batched
  in torch, and `gen/openmm_check/compare.py` holds the two to ~1e-5 kJ/mol/nm. `lollcat/fab-torch`'s
  ff96 + OBC1 ALDP variant is not used; see molecular.py's warnings.
- **`TheDisorderedOrganization/ParticlesMC`** — the field's reference implementation of the
  ternary glass, and the source for the swap Monte Carlo move pool (`p_swap = 0.2`, A<->C and B<->C
  channels, `sweepstep = N`) that `swap_monte_carlo` follows, for its `SmoothLennardJones`
  constants, and for the `compute_energy_particle` convention that `local_energy` mirrors. The
  configurations released with the model (Zenodo 17966995) settled its energy normalisation:
  `exp(-E / T)`, not `exp(-E / (N T))`; `tests/test_physics.py` fetches them to keep that check.

Current state: Tier 0 (Funnel, ManyWell, GMM-40, MoS, TwoModes, BracketTwoModes, ManyModes,
EightGaussians, Rings, Checkerboard) on exact draws, Tier 1 (DW-4, LJ-13, LJ-55) on our own
parallel-tempering references, Tier 2 (the seven Bayesian posteriors, six on Stan draws and Brownian
on an exact sampler, each with the metrics its own literature reports), Tier 3 (four SBG peptides
through one Peptide class and the batched torch force field, on our own replica-exchange MD
references), Tier 4 (both Kob-Andersen systems: the swap-friendly ternary on our own GPU swap Monte
Carlo, and the binary of Cheng et al. 2026 on our own non-reversible parallel tempering), the
metrics package, 629 tests.

Not yet built: the adjoint-sampling / GEOM drug-like molecules; phi^4 and the rest of Tier 4.

Dropped rather than deferred: the transferability family (many-peptides-md), DW-2, IPL, the ALDP
variants other than SBG's, SBG's chignolin (its slowest states relax over ~5 ns of MD per replica
under replica exchange, so no affordable run makes a near-independent reference set), and the
metallic glasses (each sits at its own glass transition, so no reference ensemble exists to score
a sample against). Removed from the interface -- do not add
back: the laplacian; every metric derived from importance weights alone or from a model's own
density (reverse/forward ESS, log_Z_error, NLL, bits-per-dimension); `samplingdb.baselines` and
every sampler in it (ULA and the fixed-schedule parallel tempering are deleted, the rest moved to
`gen/`); the loaders of third-party reference data (the DEM splits, the Zenodo glass
configurations), which live in the tests that need them; and every second name of a target.

## Environment

Use `venv/` at the repo root — do not create another one, and do not mix `pip` with environment
modules (ABI mismatch):

```sh
./venv/bin/pip install -e ".[dev,molecular]"
./venv/bin/python -m pytest tests -q  # 629 tests, ~5 min with the reference sets cached and a GPU
./venv/bin/python examples/quickstart.py
./venv/bin/pip install -e ".[docs]"   # matplotlib and PyMOL, only to redraw docs/figures
./venv/bin/python docs/figures/synthetic.py  # one script per page, plus hero.py for the banner
```

## Layout

```
samplingdb/
  metrics/
    ks.py            compute_sliced_ks, compute_projected_ks
    mmd.py           compute_mmd
    wasserstein.py   the one exact-OT wrapper, its Euclidean and torus costs, sliced, 1D, circular 1D
    histogram.py     edges, binning, weighted histograms, TVD/KL/JS, out-of-range mass
    molecular.py     what is molecular: torch k-means JSD, TICA fit and projection
    structure.py     the pair-distance and energy panel all four structured families report
    weights.py       normalize_weights, and subsample for the rows the quadratic metrics use
    standard.py      the target-agnostic panel, linear half always and quadratic half opt-in
  targets/
    base.py          Target, ExactTarget, spec_id, score_autograd (test-only)
    kernels.py       run_chains, metropolis, random_walk_mh, mala, student_t2 -- the train recipes'
                     kernels, vectorised over chains, and nothing else
    modes.py         ModeCoverage: the mode-weight panel every mode-descriptor target reports
    catalog.py       CATALOG, build, catalog, download_references -- one name per target
    mixtures.py      Mixture, MOG, GMM40, MoS, TwoModes, BracketTwoModes, ManyModes,
                     EightGaussians
    toys.py          Rings, Checkerboard
    synthetic.py     Funnel, ManyWell
    particles.py     ParticleTarget, MultiDoubleWell, DW4, LennardJones, LJ13, LJ55
    bayesian.py      BayesianTarget, LogisticRegression, Seeds, Brownian, LGCP
    physics.py       PeriodicPairMixture, KobAndersen2D, KobAndersenBinary, swap_monte_carlo
    molecular.py     Peptide, PEPTIDE_SYSTEMS, SBG_REVISION, SBG_CHECKSUMS -- one class, four
                     catalog entries
    forcefield.py    TorchForceField, extract_parameters, linlogcut -- openmm imported lazily
  data.py            cache dir, sha256, verify_once, fetch_url, fetch_hf (pinned revision and
                     sha256), cached_local, and the published reference sets: REFERENCE_REPO,
                     REFERENCE_REVISION, REFERENCE_CHECKSUMS, fetch_reference
gen/                 maintainer-only, NOT part of the package's import graph
  nrpt.py            non_reversible_parallel_tempering, its references, explorers and step-size
                     calibration, hmc and leapfrog -- the sampler behind the tempered sets
  diagnostics.py     integrated autocorrelation time, effective sample size, chain dispersion
  manifest.py        manifest, the JSON written next to every generated set and shard
  stan/              three .stan files and generate.py, six of the Tier 2 reference posteriors
  exact/             brownian.py, the exact sampler of the Brownian motion posterior
  swap_mc/           generate.py, the ternary glass's compiled GPU swap Monte Carlo driver
  parallel_tempering/ generate.py, the Tier 1 and binary-glass reference driver
  md/                generate.py, replica-exchange MD with solute tempering for the peptides, and
                     dynamics.py, the batched BAOAB integrator and the solute split
  merge_shards.py    checks every shard's spec_id and ladder gates, concatenates the shards of one
                     run and re-audits the merged set
  audit/             barrier.py, decorrelation.py, stationarity.py -- what a set is worth
  publish/           publish.py and the dataset card: checks the sets against the code, uploads
                     them to the Hugging Face dataset and tags the release
  openmm_check/      the retired OpenMMBridge, and compare.py, which asserts float64 agreement with
                     it and times against it
tests/               conftest.py puts gen/ and gen/md/ on sys.path for the tests of the samplers
examples/
docs/                the user documentation: interface, targets, metrics, reference sets
  figures/           common.py (themes, palette, writers, PyMOL), one script per page, hero.py for
                     the README banner, and images/, every figure in a light and a dark variant
venv/                the project's virtualenv, not packaged
```

## The catalog invariants

`tests/test_catalog.py` parametrises over every target and asserts what all of them owe, whatever
tier they are in: a unique `name` (short, path-safe, no version suffix) that is also its catalog
key, a train cache key, a tier and a working dtype, `log_prob_and_grad` agreeing with its parts,
`log_prob`, `score` and every feature taking **any batch size** (forced down to one configuration
per call and compared against the whole batch), a train cache key that moves with every
density-changing argument, a panel that accepts weights, a move to another device and dtype, and
ground truth that is either usable or raises an error naming the fix. Its evaluation points come
from `evaluation_points`, which uses a target's own fixed structure rather than a reference set, so
these invariants hold before any reference data has been generated. Add a target to `CATALOG` and
it is checked here.

## Rules specific to this library

1. **Scores are analytic, with one exception.** `Target.score` raises `NotImplementedError` in the
   base class; every subclass writes the derivative out, and adding an autograd fallback is a
   regression. The exception is the peptides: `TorchForceField` differentiates its closed-form
   energy with autograd, because the score has to be able to stay differentiable and hand-written
   first derivatives would mean hand-written second ones too (see its NOTE). `score_autograd`
   exists only for `tests/test_scores.py`, which checks every target's score against autograd —
   that file is the first one to keep green. A mixture's score shares its softmax with
   `log_prob_and_grad`; a particle system's comes from `pair_energy_grad` with the diagonal masked
   so the distance epsilon never reaches it.
2. **A target owns its metrics, and every metric compares samples against a reference.**
   `compute_metrics(samples, weights=None, ref_samples=None, compute_standard_metrics=True,
   skip_costly_metrics=True, generator=None, return_hist=False)`, calling `super()` first and then
   adding its own panel. Never add a parallel metrics entry point outside the target, and never add
   a metric computed from the weights alone or from a model's own density.
   A hierarchical posterior additionally names the parameters IT is judged on, through
   `compute_parameter_metrics` — log tau for Seeds, the two scales and the unobserved middle for
   Brownian, the total intensity and the field correlation for LGCP — because an aggregate over
   coordinates hides the one that is hard.
3. **Weights are a plain optional tensor, normalized inside.** Not a container, not a separate code
   path. One exception is tested: `compute_mmd` switches from the unbiased U-statistic to the
   weighted V-statistic only when weights are given.
4. **Non-scalar output is a documented return, not a print.** Anything computed from a histogram
   returns `{'edges', 'hist', 'ref_hist'}` under `return_hist=True`; anything computed from mode
   labels returns the empirical weights the same way.
5. **A density takes any batch size.** A sampler decides how many walkers it runs, so no
   `log_prob` or `score` may impose a batch size. Anything whose temporaries grow faster than the
   input -- an all-pairs tensor, a per-mode Gaussian, a per-datapoint logit -- splits itself through
   `Target.chunked(fn, x, cost=...)`, where `cost` is the elements one configuration needs and
   `Target.max_elements` (2^27, a gigabyte of float64) is the budget. The convention is a public
   `log_prob`/`score`/`energy` that only calls `chunked` on a `*_chunk` method holding the
   arithmetic, so a subclass overrides the chunk and inherits the splitting; the peptides do the
   same thing inside `TorchForceField` with `chunk_size`. On LJ-55's score at 10^5 points it is the
   difference between 2.9 GB and 18.3 GB, at the same speed.
6. **Binning conventions are inherited, never invented, and their blind spot is always reported.**
   The distance TVD is `jarridrb/DEM`'s protocol (200 bins over the reference's own range) and a
   test asserts we reproduce their number to 1e-12; the energy ranges are the ones that same
   implementation hardcodes per system. Because that protocol drops out-of-range mass without a
   word, `out_of_range` ships next to every TVD. If you add a histogram metric, add its dropped-mass
   diagnostic in the same commit. A metric with a known blind spot that has no diagnostic carries a
   WARNING in its docstring and in `docs/metrics.md`.
7. **When samplingdb generates its own ground truth, the generator proves the two densities are
   the same and gates on diagnostics.** `gen/stan/` declares its models in samplingdb's own
   unconstrained coordinates with hand-written Jacobians, asserts Stan's log density equals the
   torch one to 1e-9 BEFORE sampling, and refuses to write draws that fail R-hat, ESS or divergence
   checks. `gen/exact/brownian.py` asserts its factorised density equals `log_prob` up to one
   constant, to 1e-8, before drawing. Any future generator gets the same two gates. Draws are saved
   chain-interleaved, because chain-blocked output makes `test_samples(n=...)` return one chain.
8. **A pair convention, a normalisation or a temperature is settled against the released data, not
   read off a formula.** Three tiers have had one of these wrong at some point, and every time the
   deciding test was the same: run the protocol's own kernel from released frames and check the
   energy distribution is stationary. Ship that check, and ship the failing alternative next to it,
   so the number means something. A paper's written formula can be wrong about its own reference
   implementation -- when it is, correct it in a docstring and say so.
9. **`spec_id` is the version of the density.** Any change to a constant, convention or
   parameterisation is a new `spec_id`, never an in-place edit. Cache names embed it, so a spec
   change surfaces as a cache miss. Adding a `spec_id` means adding its assertion to
   `tests/test_target_specs.py`. It is computed rather than declared: `Target.spec()` returns the
   density-defining values and `train_cache_key` hashes them, so **a new constructor argument that
   moves `log_gamma` must appear in that target's `spec()`** — otherwise two densities share one
   cache entry and the second reads the first's split. `VARIANTS` in `tests/test_catalog.py` is
   where each such knob is pinned. Reference sets are keyed the same way, `reference_key()` =
   `<name>_<spec_id>`, in the cache and in the published dataset, so the working dtype must not
   move `spec_id`: a float64 build has to read the set the float32 default reads.
10. **Never weaken a test in `tests/test_target_specs.py` to make code pass.** Those assertions come
    from independent reimplementations and were verified by execution. If one fails the
    implementation is wrong, unless the tolerance reflects a rounding in the *published* constant
    (0.8443 is a 4-digit rounding of the true well mass) — say which.
11. **Every host-generated `train_samples()` states its sampler, step size, step count and fixed
    starting point in its docstring, and is reproducible from a fixed seed.** If a recipe's actual
    behaviour differs from what was predicted, fix the docstring to describe reality (the
    mixtures' does: on equal weights one chain per mode is stratified sampling, so GMM-40's mode
    weights come out right and only its mode-to-mode dynamics are missing). Its kernel comes from
    `targets/kernels.py`; a recipe that needs a new kernel adds it there. The recipe runs on the
    CPU whatever the target's device (`Target.train_samples` moves it), so a seeded split is one
    array on every machine.
12. **Do not fabricate ground truth.** A target with no reference set raises a `FileNotFoundError`
    from `test_samples()` that names where it looked and the `gen/` command that makes the set.
    Where an exact marginal exists (Funnel's x₁, ManyWell's pooled quartic), score against **that**,
    not against a Monte-Carlo reference — it removes the noise from the reference side entirely.
13. **A metric's grid must be justified against its own self-comparison floor.** Before shipping a
    histogram divergence, score two halves of the reference ensemble against each other and record
    that number in the docstring. The corpus's 200x200 Ramachandran grid fails this at realistic
    sample sizes, which is why the divergences use a coarser grid and the plot-grade one is only
    returned. Where the floor is close to the signal, ship an occupancy diagnostic next to it.
14. **When two reference implementations disagree on a convention, settle it against the released
    data and record the experiment.** The glass tier's energy normalisation was decided by a
    stationarity check, not by picking a side: the released configurations must be invariant under
    the documented Metropolis kernel applied to our own `log_prob`. That check lives in
    `tests/test_physics.py` and runs whenever the data is cached.
15. **A published reference set is a hypothesis, not an oracle: test it, and generate your own when
    it fails.** The Tier 1 splits every diffusion-sampler paper reports against are not stationary
    for the density they claim to come from -- DW-4's mean pair distance moves from 3.970 to 4.037
    under the target's own HMC kernel and stays there, while our parallel-tempering set does not move
    at all, and the two agree afterwards. Our own Stan Brownian set failed the same way against an
    exact sampler. Three rules follow. First, a generated reference must carry its protocol, its
    diagnostics and a sha256 in a manifest next to it, and must pass gates BEFORE it is written:
    R-hat/ESS/divergences for Stan, ten ladder round trips per chain and a 5% floor on every swap
    pair for parallel tempering, 90% efficiency and no start drift at the merge. Second, a generated
    reference has to beat the published one on the published one's own self-comparison floor before
    it replaces it. Third, our own sets are published to the Hugging Face dataset `REFERENCE_REPO`
    only once they pass their audit, by `gen/publish/publish.py`. A tag is never moved; a release is
    a new tag with new lines in `REFERENCE_CHECKSUMS`, and a set regenerated for an unchanged
    density keeps its name and changes its bytes under the new tag, which `verify_once` catches in
    a cache filled at an older one.
16. **Reproducing somebody's panel means running theirs and diffing the numbers.** The molecular
    panel is `transferable-samplers`' six scalars rewritten as batched torch, and it was accepted
    only after importing their originals and comparing: `energy_w1`/`energy_w2` and `tica_w2` agree
    to 1e-12, `torus_w2` and the TICA features to 1e-7 (mdtraj works in float32), and the two
    k-means divergences land inside scikit-learn's own seed spread, which the `kmeans` docstring
    quotes. Where an ordering is arbitrary but matters — the TICA feature columns are distances,
    then sin/cos of phi, of psi, of omega — match theirs and version it (`TICA_LAYOUT` names the
    cached projections), because a permutation keeps the shape and nothing downstream would
    complain. Same-shape-different-numbers is the failure mode to design against.
17. **Every third-party input is pinned to a revision and a sha256.** The SBG files that define the
    peptides (`SBG_REVISION`, `SBG_CHECKSUMS`), the Tier 2 datasets, and the reference sets
    (`REFERENCE_REVISION`, `REFERENCE_CHECKSUMS`). A file that does not match raises, naming the fix.
18. **Nothing under `samplingdb/` may import from `gen/`,** and `gen/` may not become a runtime
    dependency. That boundary keeps the samplers and CmdStan out of the install.

## Style

A distribution is a `torch.nn.Module` with its state in `register_buffer`, an analytic `log_prob` /
`score`, and its own `compute_metrics(samples, weights=None, ...)`. A file opens with a one-line `#`
module description, then `# Libraries` and the imports; no type hints in signatures — types live
in the docstring as `* name (type): Description` with tensor shapes spelled out; `Args:`/`Returns:`
blocks with `*` bullets and `(default is X)`; `NOTE:`/`WARNING:` in the docstring body, never as
code comments; comments are sparse section headers (`# Run the chains`, `# Return everything`);
explicit `if return_X: return a, b else: return a` at the end rather than building and trimming
tuples; no `dataclasses`, no `abc`, no logging framework, no defensive checks.

Where a metric departs from the implementation it was first written against, the docstring says so
and why — see `wasserstein_distance_1d`, which is exact where a linear interpolation of the CDFs
carries an O(1/n) bias.

Docstrings here carry one extra duty: **the provenance of every constant**. A pinned number without
a citation or a named reference implementation is not finished work.

`README.md` stays a short introduction, and the pages under `docs/` state what the library does and
the decision behind it, not how it got there. A measurement that settled a convention, or an
alternative that was rejected, belongs in the docstring or the test that keeps it settled.

Every image in the docs is drawn by a script under `docs/figures/` from the library itself
(`build`, both splits, `compute_metrics(..., return_hist=True)`), so a number on a figure is the
panel's own and a spec change is redrawn by rerunning the script. Each is written twice by
`common.save` or `common.save_gif`, `<name>_light` and `<name>_dark`, and embedded in a `<picture>`
whose dark source GitHub picks in dark mode. The colours carry one meaning everywhere: test split
`theme['test']`, train split `theme['train']`, any other sampler `theme['sample']`, and every density
`common.density_map` (GnBu, reversed on the dark page). A figure compares the train split with the
test split wherever the data allows, and its caption says what to read off it without naming a
colour.
