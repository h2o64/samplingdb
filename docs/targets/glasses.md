# Glass formers (Tier 4)

Two-dimensional Kob-Andersen mixtures on a periodic square box, where the question is whether a
sampler reaches equilibrium at all. `log_prob(x) = -E(x) / T`, with `E` the total pair energy over
unordered pairs. They work in float32.

| | `KobAndersen2D` | `KobAndersenBinary` |
| --- | --- | --- |
| composition | ternary 5/11 : 3/11 : 3/11 (Jung et al. 2023) | binary 80:20 (Cheng et al. 2026) |
| particles, density | 44 at `rho = 1.192075` | 64 (51 A, 13 B) at `rho = 1`, so the box side is 8 |
| dim | 88 | 128 |
| pair parameters | `eps` and `sigma` 3 x 3 matrices, pinned in the class | `(eps, sigma)` = (1, 1) for AA, (1.5, 0.8) for AB, (0.5, 0.88) for BB |
| cutoff at `2.5 sigma` | smoothed: the potential and its first two derivatives vanish | shifted: the potential vanishes, the force jumps |
| swap moves | effective, A<->C and B<->C | ineffective, which is the point of this system |
| temperatures | 0.32 and 1.0 | 2.0, 0.5 and 0.2; the catalog ships 0.2 |
| `name` | `ka-ternary-N44-T<T>` | `ka-binary-N64-T<T>` |

## Conventions

- **Total energy.** The density is `exp(-E / T)`, not `exp(-E / (N T))`. The released T = 1.0
  configurations are stationary under the released Metropolis kernel with this normalisation, and
  `tests/test_physics.py` keeps that check.
- **Smoothing coefficients.** The ternary potential is
  `4 eps [(sigma/r)^12 - (sigma/r)^6 + C0 + C2 (r/sigma)^2 + C4 (r/sigma)^4]` with
  `(C0, C2, C4) = (0.04049023795, -0.00970155098, 0.00062012616)` at the 2.5 cutoff, the values of
  `h2o64/learndiffeq_wip` and of ParticlesMC's `SmoothLennardJones`. Any other `cutoff_factor` solves
  for its own with `smoothing_coefficients()`.
- **Species order.** Species labels are fixed and canonically ordered. `released_samples()` sorts
  each released configuration into that order, moving the positions with their labels.
- **Invariant representation.** `preprocess()` returns the sorted minimum-image pair distances, which
  are invariant under translation, permutation and periodic image.
- **The binary mixture at T = 0.2 is partly crystalline.** Its equilibrium state is a hexagonal A
  crystallite spanning the box next to a B-rich amorphous band, the known behaviour of the 2D 80:20
  mixture (Bruening et al. 2009). The density is kept as Cheng et al. 2026 define it, so a sampler
  has to get the crystallite and the band right.

The ternary mixture's structural relaxation time `tau_alpha`, read where the self-intermediate
scattering function `F_s(k, t)` at the first peak of `S(k)` (`k = 7.30`) crosses `1/e` under swap
Monte Carlo, is 509 time units at T = 0.32 and 40 at T = 1.0 (`relaxation_times`). The targets also
expose `rdf`, `self_intermediate_scattering` and `specific_heat`.

## Metric panel

| Metric | What it compares |
| --- | --- |
| `dist_total_var`, `dist_w1`, `dist_out_of_range` | the minimum-image pair distances |
| `energy_total_var`, `energy_w1`, `energy_out_of_range` | the energies per particle |
| `rdf_max_abs_err` | the radial distribution functions g(r) |

The distance edges are pinned by the box, from 0 to `L / sqrt(2)`, which covers every reachable
minimum-image distance, so `dist_out_of_range` is 0. The energy edges come from the reference.
`return_hist=True` adds `'dist'`, `'energy'` and `'rdf'` (bin centres and both g(r)).

## Splits

| Split | How it is built |
| --- | --- |
| test, ternary | This library's swap Monte Carlo, `gen/swap_mc/generate.py`, read from `swapmc/<name>.npy` in the cache. See [Reference sets](../reference-sets.md#swap-monte-carlo-the-ternary-glass). |
| test, binary | This library's non-reversible parallel tempering, `gen/parallel_tempering/generate.py`, read from `pt/<name>.npy`. Cheng et al. 2026 published no configurations. |
| `released_samples()`, ternary | The released configurations, fetched from Zenodo record 17966995 (CC-BY-4.0) |
| train | Displacement-only single-particle Metropolis, `N(0, 0.065^2)` moves, for 2000 time units (one time unit is `N` attempted moves) from a jittered square lattice: the run that shows swap moves are needed, far short of equilibrium at the low temperatures. |
