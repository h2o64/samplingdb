# Synthetic targets (Tier 0)

Closed-form densities whose reference samples are exact draws. They work in float64, and a float32
build has the same `spec_id`: the constants that define each density are hashed in float64 whatever
the working dtype.

| Class | dim | `name` | Density |
| --- | --- | --- | --- |
| `Funnel` | 10 | `funnel` | Neal's funnel: `x_1 ~ N(0, 3^2)`, `x_{2:d} \| x_1 ~ N(0, exp(x_1) I)`. The 3 is a scale, not a variance |
| `GMM40` | 2, any | `gmm40` | 40 equally weighted unit-variance Gaussians with means uniform in [-40, 40]^d (Midgley et al. 2023, Blessing et al. 2024) |
| `MoS` | 2, any | `mos` | 10 equally weighted Student-t components with 2 degrees of freedom, so infinite variance, means uniform in [-10, 10]^d (Blessing et al. 2024) |
| `ManyWell` | 32, any even | `manywell` | `d/2` independent tilted 2D double wells, `E = a x_1 + b x_1^2 + c x_1^4 + x_2^2 / 2` with `(a, b, c) = (-0.5, -6, 1)` (Wu et al. 2020) |
| `TwoModes` | any | `twomodes` | `(2/3) N(-1, 0.05 I) + (1/3) N(+1, 0.05 I)` |
| `BracketTwoModes` | any | `brackettwomodes` | two modes at ±0.75, weighted 2/3 : 1/3, whose diagonal variances ramp from 0.01 to 0.2 in one and from 0.2 to 0.01 in the other |
| `ManyModes` | any | `manymodes-k<K>` | `K` Gaussians (3 by default) at seeded uniform locations in `[-K, K]^d`, variance 0.1, weights rising geometrically to 3x |
| `EightGaussians` | 2 | `eightgaussians` | eight equally weighted Gaussians on a circle of radius 2, scale 0.1 |
| `Rings` | 2 | `rings`, `rings-equal` | radius a Gaussian mixture at 1, 3, 5 with scale 0.1, angle uniform, weights proportional to the radius (equal with `equilibrated=True`) |
| `Checkerboard` | 2 | `checkerboard` | eight uniform 2 x 2 squares, every second one three times heavier |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/synthetic_splits_dark.png">
  <img alt="The eight 2D targets: test draws, train samples and mode weights of each" src="../figures/images/synthetic_splits_light.png">
</picture>

Each 2D target's log density in ten levels of one nat, with the marginal histograms of 2048 test
draws and 2048 train samples over the exact marginals, and below it the empirical mode weights of
both splits against the true ones. The train chains keep the allocation of their starting points:
`TwoModes` and
`BracketTwoModes` come out near 1/2 : 1/2 against 2/3 : 1/3, `Rings` at 1/3 per ring against
weights proportional to the radius, and `Checkerboard`'s light squares too heavy. On the equally
weighted mixtures the train split's weights are right, and on `gmm40` closer to the truth than 2048
exact draws (`tv_weights` 0.06 against 0.12), because one chain per mode is stratified sampling.

`Rings` and `Checkerboard` follow the layout of `h2o64/sde_sampler_lrds`. The parameters of `EightGaussians` are a choice:
no two papers use the same ones.

Details that matter when comparing with other implementations:

- The GMM-40, MoS and ManyModes means are drawn by a seeded PyTorch generator, in float32 whatever
  the working dtype, so a float64 build has the same locations. `tests/test_target_specs.py`
  asserts the sha256 of the GMM-40 and MoS matrices at 2, 50 and 200 dimensions: the matrix is the
  specification, not the seed. The papers drew theirs with JAX, whose generator changed across
  versions, so the locations differ from theirs while the construction is the same.
- ManyWell's coordinates are interleaved: even indexes carry the quartic, odd ones the Gaussian. The
  tilt puts 15.57% of each well's mass on the left and 84.43% on the right, and `log Z = 164.696`
  at `d = 32`.
- Rings' density diverges at the origin through the polar Jacobian `-log |x|`. It is integrable and
  no sampler visits it.
- Checkerboard's score is identically zero wherever the density is finite, and `log_prob` is `-inf`
  off the board. It is a coverage test for samplers that use `log_prob`.

## Metric panel

Every synthetic target reports the [standard panel](../metrics.md#the-standard-panel), then its own:

| Targets | Metrics | `return_hist=True` gives |
| --- | --- | --- |
| `GMM40`, `MoS`, `TwoModes`, `BracketTwoModes`, `ManyModes`, `EightGaussians` | `emc`, `kl_weights`, `tv_weights`, `perc_forgotten_modes` on the mean component responsibilities | the empirical mode weights, `(1, n_modes)` |
| `Rings`, `Checkerboard` | the same four, on the nearest ring and on which square | the same |
| `ManyWell` | the four per well, averaged over wells, plus `tvd_x1`, `x1_out_of_range`, `w1_x1` and `x1_mean_err` against the exact pooled quartic marginal | the per-well weights, `(n_wells, 2)` |
| `Funnel` | `ks_x1`, `tvd_x1`, `x1_out_of_range`, `x1_mean_err`, `x1_var_err` and `neck_var_rel_err` against the analytic `N(0, 9)` marginal of `x_1` | the `x_1` histogram and its exact counterpart |

The mixtures' empirical mode weights are mean responsibilities, whose expectation under exact draws
is the mixture weights even where components overlap; see [Mode coverage](../metrics.md#mode-coverage).
On `TwoModes` and `BracketTwoModes` the heavy mode's weight is `hist[0, 0]` of `return_hist=True`, and
`tv_weights` is twice its error. ManyWell's panel is per well because no finite sample can populate
its `2^(d/2)` joint modes, while the per-well marginals are estimable.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/synthetic_mode_coverage_dark.png">
  <img alt="GMM-40 under exact draws, Langevin dynamics and one Metropolis chain, with sorted mode weights and the four coverage metrics" src="../figures/images/synthetic_mode_coverage_light.png">
</picture>

Three sets of 2048 GMM-40 samples and the four coverage numbers. Each mode is a disc whose area is
its empirical weight, inside a ring whose area is the true weight 1/40. The exact draws set the
floor at this size, `tv_weights` 0.12. The unadjusted Langevin loop of the README reaches all 40
modes with the wrong weights (`tv_weights` 0.66, nothing forgotten). One random-walk Metropolis
chain of step 3 from the first mean, kept every 50 steps, never reaches the 18 empty rings:
`perc_forgotten_modes` is 0.45 and `kl_weights` infinite, while `emc` still reads 0.78.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/synthetic_funnel_dark.png">
  <img alt="Funnel: test and train samples in the first two coordinates, and the x1 histograms against the exact marginal" src="../figures/images/synthetic_funnel_light.png">
</picture>

The exact density of the funnel's first two coordinates, with the marginals of 2048 test draws and
2048 train samples over the exact ones; the `x_2` marginal has a logarithmic density axis. `ks_x1`
and `tvd_x1` compare `x_1` with its exact `N(0, 3^2)`. The train chains start at the origin and
stay between -3.6 and 4.0, so they miss the narrow neck and the wide end. `neck_var_rel_err` is the relative error on `E[exp(x_1)]`, the
variance of `x_2` to `x_10`; a few samples at large `x_1` carry it, so 2048 exact draws still read
-0.34.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/synthetic_manywell_dark.png">
  <img alt="ManyWell: one well of both splits, the pooled quartic marginal and the left-well mass of each of the 16 wells" src="../figures/images/synthetic_manywell_light.png">
</picture>

One of ManyWell's 16 identical double wells, with the marginals of 2048 test draws and 2048 train
samples pooled over the wells, and the mass on the left of each well, which is what the coverage
panel reads (0.1557 under the target). 99% of the train chains never accept a move from
`(1.75, 0)`, so both train marginals are spikes and no train sample reaches a left well:
`perc_forgotten_modes` reads 0.5, one state of two per well, and `emc` 0.

## Splits

| Split | How it is built |
| --- | --- |
| test | Exact draws, seeded at 0. ManyWell inverts the 1D quartic CDF per well, Rings draws a radius then an angle, Checkerboard picks a square by weight then a uniform point in it. |
| train, mixtures | 200 Metropolis steps of size `2.4 min_scale / sqrt(dim)`, the chains spread evenly over the components and each started at its own mean. Every mode is visited and the mode weights follow the allocation, not the mixture: `TwoModes` comes out 50/50 against a true 2/3 : 1/3. |
| train, `Rings` | one chain per ring, from a uniform point on it, 200 steps of size 0.1 |
| train, `Checkerboard` | one chain per square, from its centre, 200 steps of size 0.5. The squares meet at their corners, so the weights partly equilibrate: the heavy-to-light ratio comes out at 1.84 against 3. |
| train, `Funnel` | 200 steps of size 0.5 from the origin, too few to reach the wide end of the neck |
| train, `ManyWell` | 200 steps of size 0.3 from `(1.75, 0)` in every well, below the barrier, so exactly one of the `2^(d/2)` modes is visited |

On the equally weighted mixtures (`GMM40`, `MoS`, `EightGaussians`) one chain per mode is
stratified sampling, so their train split has the right mode weights and lacks only mode-to-mode
moves. The unequally weighted targets are the test of mode proportions.
