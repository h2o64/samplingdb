# Metrics

Every metric compares samples against a reference ensemble. Importance weights are an optional
argument, normalised inside; no metric is computed from the weights alone or from a model's own
density, so there is no effective sample size, `log Z` error, NLL or bits per dimension.

## `compute_metrics`

```python
metrics = target.compute_metrics(samples, weights=None, ref_samples=None,
                                 compute_standard_metrics=True, skip_costly_metrics=True,
                                 generator=None, return_hist=False)
```

- `ref_samples` defaults to the target's test split at the size of `samples` (exact draws are seeded
  at 0, so a reported number is reproducible). Pass a larger or different reference explicitly.
- `generator` seeds the random projections of the sliced metrics.
- `return_hist=True` also returns what the scalars rest on: `{'edges', 'hist', 'ref_hist'}` for a
  histogram, the empirical mode weights for a mode panel, per-coordinate moments for a posterior.

Each target's panel is listed on its family page: [synthetic](targets/synthetic.md#metric-panel),
[particles](targets/particles.md#metric-panel), [posteriors](targets/posteriors.md#metric-panel),
[peptides](targets/peptides.md#metric-panel), [glass formers](targets/glasses.md#metric-panel).

## The standard panel

The synthetic targets, whose coordinates compare directly, report it before their own panel
(`standard_panel = True`). Particle, glass and peptide configurations are defined only up to
translation, permutation, rotation or periodic image, so those targets report an invariant panel
instead. `metrics.compute_standard_metrics` can still be called directly on any samples.

| Metric | Definition |
| --- | --- |
| `ks` | sliced Kolmogorov-Smirnov, 128 random projections, 256 bins |
| `w2_sliced` | sliced Wasserstein-2 |
| `mmd` | Gaussian-kernel MMD with the median-heuristic bandwidth: the unbiased U-statistic, or the weighted V-statistic when weights are given |
| `w2`, `w2_reg` | exact and entropic Wasserstein-2 |
| `n_costly_samples` | the number of samples per side the three metrics above were computed on |

`mmd`, `w2` and `w2_reg` are quadratic, so they are computed only with `skip_costly_metrics=False`,
on both sides cut to `max_costly_samples` (2000). The implementations are ported from `islips` so
numbers compare across the two libraries.

## Mode coverage

Every target with a discrete mode descriptor (the mixtures, `Rings`, `Checkerboard`, `ManyWell`
per well) reports four numbers on the empirical mode weights:

| Metric | Definition |
| --- | --- |
| `emc` | entropy of the empirical mode weights divided by `log(n_modes)` |
| `kl_weights` | `KL(true weights \|\| empirical weights)`, infinite when a mode gets no mass |
| `tv_weights` | `sum_k \|empirical_k - true_k\|` |
| `perc_forgotten_modes` | fraction of modes holding less than 5% of the smallest true weight |

## Binning

- **Particle distances** follow `jarridrb/DEM`: 200 equal-width bins over the reference distances'
  own range, the samples binned on the same edges, each side renormalised.
  `tests/test_metrics.py` reproduces DEM's value to 1e-12. Mass that falls outside the reference
  range is dropped by that protocol, so every total variation comes with its dropped mass,
  `*_out_of_range`; the `*_w1` metrics need no binning.
- **Particle energies** use 100 bins over the ranges DEM fixes per system: `(-26, 0)` for DW-4,
  `(-60, 0)` for LJ-13, `(-380, -180)` for LJ-55.
- **Glass distances** use edges pinned by the box, from 0 to `L / sqrt(2)`, so nothing is dropped.
  Glass energies are per particle.
- **Peptide torsion divergences** use 64 bins per marginal and a 32 x 32 Ramachandran joint, with
  `ramachandran_occupancy` reporting how much of that grid the reference fills. The 200 x 200 grid
  is returned for plotting only: at realistic sample sizes it is mostly empty, and two halves of the
  same ensemble would read as different.
- **Jensen-Shannon divergences** are the squared Jensen-Shannon distance in base 2,
  `scipy.spatial.distance.jensenshannon(p, q, base=2) ** 2`.

There is no generic sample-level TVD, KL or JSD: no binning rule is defensible in 1600 dimensions,
so histogram metrics exist only where the binning is fixed by the observable.

## Building blocks

`samplingdb.metrics` exposes the pieces the panels are built from:

| Function | Use |
| --- | --- |
| `compute_sliced_ks`, `compute_mmd`, `compute_wasserstein_distance`, `compute_sliced_wasserstein_distance` | the standard panel |
| `wasserstein_distance_1d` | weighted 1D Wasserstein, exact for powers 1 and 2 and safe with repeated values |
| `torus_wasserstein` | exact transport with the periodic per-coordinate distance |
| `weighted_histogram`, `histogram_2d`, `compare_histograms`, `reference_edges`, `shared_edges` | histograms on fixed or shared edges, with the out-of-range mass |
| `histogram_tvd`, `histogram_kl`, `jensen_shannon` | divergences between histograms |
| `compute_structure_metrics` | the distance and energy panel the structured targets share |
| `kmeans`, `kmeans_jsd`, `fit_tica`, `TicaModel`, `wrap_angles` | the molecular features |
| `integrated_autocorrelation_time`, `effective_sample_size`, `chain_mean_dispersion` | how many independent samples a chain holds |
| `normalize_weights` | `None` becomes uniform weights |
