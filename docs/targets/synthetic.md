# Synthetic targets (Tier 0)

Closed-form densities whose reference samples are exact draws. They work in float64.

| Class | dim | `name` | Density |
| --- | --- | --- | --- |
| `Funnel` | 10 | `funnel` | Neal's funnel: `x_1 ~ N(0, 3^2)`, `x_{2:d} \| x_1 ~ N(0, exp(x_1) I)`. The 3 is a scale, not a variance |
| `GMM40` | 2, 50 or 200 | `gmm40` | 40 equally weighted unit-variance Gaussians with means spread over ±40 (Midgley et al. 2023, Blessing et al. 2024) |
| `MoS` | 2, 50 or 200 | `mos` | 10 equally weighted Student-t components with 2 degrees of freedom, so infinite variance (Blessing et al. 2024) |
| `ManyWell` | 32, any even | `manywell` | `d/2` independent tilted 2D double wells, `E = a x_1 + b x_1^2 + c x_1^4 + x_2^2 / 2` with `(a, b, c) = (-0.5, -6, 1)` (Wu et al. 2020) |
| `TwoModes` | any | `twomodes` | `(2/3) N(-1, 0.05 I) + (1/3) N(+1, 0.05 I)` |
| `BracketTwoModes` | any | `brackettwomodes` | two modes at ±0.75, weighted 2/3 : 1/3, whose diagonal variances ramp from 0.01 to 0.2 in one and from 0.2 to 0.01 in the other |
| `ManyModes` | any | `manymodes-k<K>` | `K` Gaussians (3 by default) at seeded uniform locations in `[-K, K]^d`, variance 0.1, weights rising geometrically to 3x |
| `EightGaussians` | 2 | `eightgaussians` | eight equally weighted Gaussians on a circle of radius 2, scale 0.1 |
| `Rings` | 2 | `rings`, `rings-equal` | radius a Gaussian mixture at 1, 3, 5 with scale 0.1, angle uniform, weights proportional to the radius (equal with `equilibrated=True`) |
| `Checkerboard` | 2 | `checkerboard` | eight uniform 2 x 2 squares, every second one three times heavier |

`TwoModes`, `BracketTwoModes` and `ManyModes` are ported from `islips`; `Rings` and `Checkerboard`
follow the layout of `h2o64/sde_sampler_lrds`. The parameters of `EightGaussians` are a choice:
no two papers use the same ones.

Details that matter when comparing with other implementations:

- The GMM-40 and MoS mean matrices are pinned as `.npy` assets in `samplingdb/targets/assets/`, and
  `tests/test_target_specs.py` asserts their sha256. The matrix is the specification, not the seed
  that drew it.
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
| `GMM40`, `MoS`, `ManyModes`, `EightGaussians` | `emc`, `kl_weights`, `tv_weights`, `perc_forgotten_modes` on the nearest component | the empirical mode weights |
| `Rings`, `Checkerboard` | the same four, on the nearest ring and on which square | the same |
| `TwoModes`, `BracketTwoModes` | the same four, plus `mode_weight` (the heavy mode, in percent) and `mode_weight_abs_err` | the same |
| `ManyWell` | the four per well, averaged over wells, plus `tvd_x1`, `x1_out_of_range`, `w1_x1` and `x1_mean_err` against the exact pooled quartic marginal | the per-well weights, `(n_wells, 2)` |
| `Funnel` | `ks_x1`, `tvd_x1`, `x1_out_of_range`, `x1_mean_err`, `x1_var_err` and `neck_var_rel_err` against the analytic `N(0, 9)` marginal of `x_1` | the `x_1` histogram and its exact counterpart |

ManyWell's panel is per well because no finite sample can populate its `2^(d/2)` joint modes, while
the per-well marginals are estimable.

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
