# AGENTS.md — samplingdb

Instructions for an agent working in this repository. Read this before editing anything here.

## What this is

`samplingdb` is written in the style and API shape of two existing codebases, `islips` and `DREEM`,
and settles each target's conventions against a named reference implementation:

- **`islips`** (`h2o64/islips_wip`) — the reference for how a distribution is
  written here: `torch.nn.Module`, `register_buffer` for state, analytic `log_prob` / `score` /
  `compute_metrics(samples, weights=None, ...)` owned by the distribution, and the
  metric implementations under `islips/metrics` (sliced KS, MMD, Wasserstein, KLD) which
  `samplingdb/metrics` ports so numbers stay comparable across the two libraries.
- **`DREEM`** — the reference for what clean means, and specifically for
  `dreem/distr/manywell.py`, which `ManyWell` follows: interleaved coordinates, the analytic score,
  the closed-form `Z_x1`, the pinned moments, and the per-well mode-weight panel
  (`emc`, `kl_weights`, `tv_weights`, `perc_forgotten_modes`).
- **`jarridrb/DEM`** — the reference for the particle targets' binning conventions (see below), the
  publisher of the Tier 1 reference splits (fetched from a pinned commit), and — via bgflow — the
  authority on their pair conventions: `MultiDoubleWellPotential` counts each pair once,
  `LennardJonesPotential` with `energy_factor=1.0` counts each twice. The published LJ formula
  disagrees with the latter and with the data; the data won.
- **`h2o64/learndiffeq_wip`** (`learndiffeq/particles/distributions/kob_andersen.py`) — the
  implementation the released glass data came from. Its potential constants are pinned in
  `physics.py`; its `grad_Wpp` and its `U = E / n_particles` convention are NOT followed, for
  reasons recorded in the docstrings and checked in `tests/test_physics.py`.
- **`transferable-samplers/transferable-samplers`** (`data/single_peptide_datamodule.py`,
  `data/energy/openmm_energy.py`) — the implementation the five peptide trajectories were generated
  against, and therefore the authority on their two force-field families and on the unit
  conventions (positions in nm, energies in kT). Its per-configuration OpenMM loop is NOT followed:
  `forcefield.py` evaluates the same systems batched in torch, and `gen/openmm_check/compare.py`
  is what holds the two to ~1e-5 kJ/mol/nm. `lollcat/fab-torch`'s
  ff96 + OBC1 ALDP variant is deliberately NOT used; see molecular.py's warnings.
- **`TheDisorderedOrganization/ParticlesMC`** — the field's reference implementation of the same
  model, and the source for the swap Monte Carlo move pool (`p_swap = 0.2`, A<->C and B<->C
  channels, `sweepstep = N`) that `swap_monte_carlo` follows, for its `SmoothLennardJones`
  constants (a third agreement), and for the `compute_energy_particle` convention that
  `local_energy` mirrors.

Current state: Tier 0 (Funnel, ManyWell, GMM-40, MoS, TwoModes, BracketTwoModes, ManyModes,
EightGaussians, Rings, Checkerboard), Tier 1 (DW-4, LJ-13, LJ-55) on our own parallel-tempering
references, Tier 2 (the seven Bayesian posteriors on Stan draws, each with the metrics its own
literature reports), Tier 3 (the five SBG peptides via one Peptide class and the batched torch force
field, plus a validated in-house MD generator), Tier 4 (both Kob-Andersen systems: the swap-friendly
ternary and the binary of Cheng et al. 2026), the metrics package, the classical baselines, parallel
tempering, 508 tests.

Not yet built: the learned-SDE baseline class (PIS/DDS/DIS/CMCD) and SMC/AIS; the
adjoint-sampling / GEOM drug-like molecules; phi^4 and the rest of Tier 4.

Dropped rather than deferred: the transferability family (many-peptides-md), DW-2, IPL, the ALDP
variants other than SBG's, and the metallic glasses (cuzr and crconi were built on a batched EAM
evaluator and removed on 2026-10-09, with the larger alloys of Cheng et al. 2026: each sits at its
own glass transition, so no reference ensemble exists to score a sample against). Removed from the
interface -- do not add back: the laplacian, and every metric derived from importance weights alone
or from a model's own density (reverse/forward ESS, log_Z_error, NLL, bits-per-dimension).

## Environment

Use `venv/` at the repo root — do not create another one, and do not mix `pip` with environment
modules (ABI mismatch):

```sh
./venv/bin/pip install -e ".[dev,molecular]"
./venv/bin/python -m pytest tests -q  # 508 tests, ~2 min with the reference sets cached
./venv/bin/python examples/quickstart.py
```

`jax` is deliberately **not** in this venv: it is needed only to regenerate `gen/assets/`, so that
one script runs from a separate environment. Keep it out of the package's dependencies.

## Layout

```
samplingdb/
  metrics/
    autocorrelation.py  integrated autocorrelation time, effective sample size, chain dispersion
    ks.py            compute_sliced_ks, compute_random_proj_cdf   (ported from islips)
    mmd.py           compute_mmd                                   (ported from islips)
    wasserstein.py   the one exact-OT wrapper, its Euclidean and torus costs, sliced, 1D
    histogram.py     edges, binning, weighted histograms, TVD/KL/JS, out-of-range mass
    molecular.py     what is genuinely molecular: torch k-means JSD, TICA fit and projection
    structure.py     the pair-distance and energy panel all four structured families report
    weights.py       normalize_weights, which is all a weighted metric needs
    standard.py      the target-agnostic panel, linear half always and quadratic half opt-in
  targets/
    base.py          Target, ExactTarget, spec_id, score_autograd (test-only)
    modes.py         ModeCoverage: the mode-weight panel every mode-descriptor target reports
    catalog.py       CATALOG, build, catalog -- the one list of what this library ships
    mixtures.py      Mixture, MOG, GMM40, MoS, TwoModes, BracketTwoModes, ManyModes,
                     EightGaussians
    toys.py          Rings, Checkerboard
    synthetic.py     Funnel, ManyWell
    particles.py     ParticleTarget, MultiDoubleWell, DW4, LennardJones, LJ13, LJ55
    bayesian.py      BayesianTarget, LogisticRegression, Seeds, Brownian, LGCP
    physics.py       PeriodicPairMixture, KobAndersen2D, KobAndersenBinary, swap_monte_carlo
    molecular.py     OpenMMTarget, Peptide, PEPTIDE_SYSTEMS -- one class, five catalog entries
    forcefield.py    TorchForceField, extract_parameters, linlogcut -- openmm imported lazily
    assets/          pinned mean matrices (.npy), shipped in the wheel
  baselines/
    classical.py     run_chains + metropolis drive random_walk_mh, ula, mala, hmc; student_t2,
                     leapfrog -- all vectorized over chains
    nrpt.py          non_reversible_parallel_tempering, its references, explorers and step-size
                     calibration -- the sampler behind every generated reference set
    tempering.py     parallel_tempering, geometric_ladder -- HMC rungs, even/odd swaps
  data.py            cache dir, checksums, fetch_url / fetch_hf / fetch_zenodo, cached_local
gen/                 maintainer-only, NOT part of the package's import graph
  assets/            the jax script behind the pinned mean matrices
  stan/              four .stan files and generate.py, the Tier 2 reference posteriors
  swap_mc/           generate.py, the ternary glass's swap Monte Carlo driver, shardable
  parallel_tempering/ generate.py, the Tier 1 and binary-glass reference driver
  md/                generate.py, replica-exchange MD with solute tempering for the peptides
  merge_shards.py    concatenates the shards of one run and re-audits the merged set
  audit/             barrier.py, decorrelation.py, stationarity.py -- what a set is worth
  openmm_check/      the retired OpenMMBridge, and compare.py which checks and times against it
tests/
examples/
docs/                the user documentation: interface, targets, metrics, baselines, reference sets
venv/                the project's virtualenv, not packaged
```

## The catalog invariants

`tests/test_catalog.py` parametrises over every target and asserts what all of them owe, whatever
tier they are in: a unique `name` (short, path-safe, no version suffix) and train cache key, a tier
and a working dtype, `log_prob_and_grad` agreeing with its parts, `log_prob`, `score` and every
feature taking **any batch size** (forced down to one configuration per call and compared against
the whole batch), a train cache key that moves with every density-changing argument, a panel that
accepts weights, a move to another device and dtype, and ground truth that is either usable or
raises an error naming the fix. Its evaluation points come from `evaluation_points`, which uses a
target's own fixed structure rather than a reference set, so these invariants hold before any
reference data has been generated. Add a target to `CATALOG` and it is checked here -- that is
where a missing piece gets caught, and it is how the DW-2 `KeyError` was found.

## Rules specific to this library

1. **Scores are analytic. There is no autograd fallback, and adding one is a regression.**
   `Target.score` raises `NotImplementedError` in the base class; every subclass writes the
   derivative out. `score_autograd` exists only for `tests/test_scores.py`, which checks every
   target's score against autograd — that file is the first one to keep green. A mixture's score
   shares its softmax with `log_prob_and_grad`; a particle system's comes from `pair_energy_grad`
   with the diagonal masked so the distance epsilon never reaches it.
2. **A target owns its metrics, and every metric compares samples against a reference.**
   `compute_metrics(samples, weights=None, ref_samples=None, compute_standard_metrics=True,
   skip_costly_metrics=True, generator=None, return_hist=False)`, calling `super()` first and then
   adding its own panel — the islips/DREEM shape. Never add a parallel metrics entry point outside
   the target, and never add a metric computed from the weights alone or from a model's own density:
   an ESS, a `log_Z_error`, an NLL and a bits-per-dimension were all removed on purpose.
   A hierarchical posterior additionally names the parameters IT is judged on, through
   `compute_parameter_metrics` — log tau for Seeds, the two scales and the unobserved middle for
   Brownian, the total intensity and the field correlation for LGCP — because an aggregate over
   coordinates hides the one that is hard.
3. **Weights are a plain optional tensor, normalized inside.** Not a container, not a separate code
   path. One exception is load-bearing and tested: `compute_mmd` switches from the unbiased
   U-statistic to the weighted V-statistic only when weights are actually given.
4. **Non-scalar output is a documented return, not a print.** Anything computed from a histogram
   returns `{'edges', 'hist', 'ref_hist'}` under `return_hist=True`; anything computed from mode
   labels returns the empirical weights the same way.
5. **A density takes any batch size.** A sampler decides how many walkers it runs, so no
   `log_prob` or `score` may impose a batch size. Anything whose temporaries grow faster than the
   input -- an all-pairs tensor, a per-mode Gaussian, a per-datapoint logit -- splits itself through
   `Target.chunked(fn, x, cost=...)`, where `cost` is the elements one configuration needs and
   `Target.max_elements` (2^27, a gigabyte of float64) is the budget. The convention is a public
   `log_prob`/`score`/`energy` that only calls `chunked` on a `*_chunk` method holding the real
   arithmetic, so a subclass overrides the chunk and inherits the splitting; the peptides do the
   same thing inside `TorchForceField` with `chunk_size`. It costs nothing and it is the difference
   between 2.9 GB and 18.3 GB on LJ-55's score at 10^5 points, at the same speed.
6. **Binning conventions are inherited, never invented, and their blind spot is always reported.**
   The distance TVD is `jarridrb/DEM`'s protocol (200 bins over the reference's own range) and a
   test asserts we reproduce their number to 1e-12; the energy ranges are the ones that same
   implementation hardcodes per system. Because that protocol silently drops out-of-range mass,
   `out_of_range` ships next to every TVD. If you add a histogram metric, add its dropped-mass
   diagnostic in the same commit.
7. **When samplingdb generates its own ground truth, the generator proves the two densities are
   the same and gates on diagnostics.** `gen/stan/` declares its models in samplingdb's own
   unconstrained coordinates with hand-written Jacobians, asserts Stan's log density equals the torch
   one to 1e-9 (they agree to 1e-14) BEFORE sampling, and refuses to write draws that fail R-hat,
   ESS or divergence checks. Any future generator gets the same two gates. Draws are saved
   chain-interleaved, because chain-blocked output makes `test_samples(n=...)` return one chain.
8. **A pair convention, a normalisation or a temperature is settled against the released data, not
   read off a formula.** Three tiers have now had one of these wrong at some point, and every time
   the deciding test was the same: run the protocol's own kernel from released frames and check the
   energy distribution is stationary. Ship that check, and ship the failing alternative next to it,
   so the number means something. A paper's written formula can be wrong about its own reference
   implementation -- when it is, correct it in a docstring and say so.
9. **`spec_id` is the version of the density.** Any change to a constant, convention or
   parameterisation is a new `spec_id`, never an in-place edit. Cache names embed it, so a spec
   change surfaces as a cache miss. Adding a `spec_id` means adding its assertion to
   `tests/test_target_specs.py`. It is computed rather than declared: `Target.spec()` returns the
   density-defining values and `train_cache_key` hashes them, so **a new constructor argument that
   moves `log_gamma` must appear in that target's `spec()`** — otherwise two densities share one
   cache entry and the second silently reads the first's split. `VARIANTS` in
   `tests/test_catalog.py` is where each such knob is pinned.
10. **Never weaken a test in `tests/test_target_specs.py` to make code pass.** Those assertions come
   from independent reimplementations and were verified by execution. If one fails the
   implementation is wrong, unless the tolerance genuinely reflects a rounding in the *published*
   constant (0.8443 is a 4-digit rounding of the true well mass) — say which.
11. **Every host-generated `train_samples()` states its sampler, step size, step count and fixed
   starting point in its docstring, and is reproducible from a fixed seed.** If a recipe's actual
   behaviour differs from what was predicted, fix the docstring to describe reality (GMM-40's
   does: 3 of 40 modes, not 1, because the pinned mean matrix clusters components 0, 7 and 37).
12. **Do not fabricate ground truth.** `has_ground_truth = False` targets get no `test_samples()`;
   reference data we do not hold raises a `FileNotFoundError` naming the path. Where an exact
   marginal exists (Funnel's x₁, ManyWell's pooled quartic), score against **that**, not against a
   Monte-Carlo reference — it removes the noise from the reference side entirely.
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
    at all, and the two agree afterwards. So `test_samples()` is ours and `load_split()` is theirs.
    Two rules follow. First, a generated reference must carry its protocol, its diagnostics and a
    sha256 in a manifest next to it, and must pass gates BEFORE it is written: R-hat/ESS/divergences
    for Stan, ten ladder round trips per chain and a 5% floor on every swap pair for parallel
    tempering. Second, a generated reference has to beat the published one on the published one's own
    self-comparison floor before it replaces it.
16. **Reproducing somebody's panel means running theirs and diffing the numbers.** The molecular
    panel is `transferable-samplers`' six scalars rewritten as batched torch, and it was accepted
    only after importing their originals and comparing: `energy_w1`/`energy_w2` and `tica_w2` agree
    to 1e-12, `torus_w2` and the TICA features to 1e-7 (mdtraj works in float32), and the two
    k-means divergences land inside scikit-learn's own seed spread, which the `kmeans` docstring
    quotes. Where an ordering is arbitrary but load-bearing — the TICA feature columns are
    distances, then sin/cos of phi, of psi, of omega — match theirs and version it (`TICA_LAYOUT`
    names the cached projections), because a permutation keeps the shape and nothing downstream
    would complain. Same-shape-different-numbers is the failure mode to design against.
17. **Nothing under `samplingdb/` may import from `gen/`,** and `gen/` may not become a runtime
    dependency. That boundary keeps Julia, CmdStan and jax out of the install.

## Style

`islips` and `DREEM` are the worked examples. Concretely: a one-line `#` module description, then
`# Libraries` and the imports; no type hints in signatures — types live in the docstring as
`* name (type): Description` with tensor shapes spelled out; `Args:`/`Returns:` blocks with `*`
bullets and `(default is X)`; `NOTE:`/`WARNING:` in the docstring body, never as code comments;
comments are sparse section headers (`# Run the chains`, `# Return everything`); explicit
`if return_X: return a, b else: return a` at the end rather than building and trimming tuples; no
`dataclasses`, no `abc`, no logging framework, no defensive checks.

Ported functions keep their upstream names, signatures and numerics so results stay comparable with
`islips`. Where a port had to change, the docstring says so and why — see `wasserstein.interp`,
whose flat branch fixes a division by zero on the repeated states every MCMC chain produces.

Docstrings here carry one extra duty: **the provenance of every constant**. A pinned number without
a citation or a named reference implementation is not finished work.

`README.md` stays a short introduction, and the pages under `docs/` state what the library does and
the decision behind it, not how it got there. A measurement that settled a convention, or an
alternative that was rejected, belongs in the docstring or the test that keeps it settled.
