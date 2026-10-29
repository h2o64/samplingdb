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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/glasses_configurations_dark.png">
  <img alt="One test and one train configuration of each glass former, and the binary's hexatic order" src="../figures/images/glasses_configurations_light.png">
</picture>

One configuration of each split for every catalogued system, each particle a disc of diameter
`sigma_ii`; on the ternary, the one closest to its split's median energy per particle. The train
split's basins are a hundred Monte Carlo runs from a lattice with the particles shuffled over its
sites, with swaps on the ternary, at temperatures on both sides of the target's; chains of
displacement moves started in them, most in the coldest, give its configurations. The species are
mixed in both splits, so the ternary columns look alike and the difference is in the panel below.
The last column shades the binary by its local hexatic order `|psi_6|`, on a configuration of each
split typical in the crystalline order of A. In both, A crystallises away from the B particles; a
median of 39% of the A particles have `|psi_6| > 0.8` in the train split, against 51% in the test
split.

## Conventions

- **Total energy.** The density is `exp(-E / T)`, not `exp(-E / (N T))`. The configurations
  released with the ternary model (Zenodo record 17966995) are stationary under the released
  Metropolis kernel with this normalisation, which `tests/test_physics.py` checks.
- **Smoothing coefficients.** The ternary potential is
  `4 eps [(sigma/r)^12 - (sigma/r)^6 + C0 + C2 (r/sigma)^2 + C4 (r/sigma)^4]` with
  `(C0, C2, C4) = (0.04049023795, -0.00970155098, 0.00062012616)` at the 2.5 cutoff, the values
  that solve W(rc) = W'(rc) = W''(rc) = 0 and those of ParticlesMC's `SmoothLennardJones`. Any other
  `cutoff_factor` solves for its own with `smoothing_coefficients()`.
- **Species order.** Species labels are fixed and canonically ordered: A, then B, then C.
- **Invariant representation.** `preprocess()` returns the sorted minimum-image pair distances, which
  are invariant under translation, permutation and periodic image.
- **The binary mixture at T = 0.2 is partly crystalline.** Its equilibrium state is a hexagonal A
  crystallite spanning the box next to a B-rich amorphous band, the known behaviour of the 2D 80:20
  mixture (Bruening et al. 2009). The density is kept as Cheng et al. 2026 define it, so a sampler
  has to get the crystallite and the band right.

Under swap Monte Carlo, the ternary mixture's structural relaxation time read off the particle
trajectories is about 2 300 time units at T = 0.32 and 265 at T = 1.0. The targets also expose
`rdf`, the radial distribution function the panel compares.

## Metric panel

| Metric | What it compares |
| --- | --- |
| `dist_total_var`, `dist_w1`, `dist_out_of_range` | the minimum-image pair distances |
| `energy_total_var`, `energy_w1`, `energy_out_of_range` | the energies per particle |
| `rdf_max_abs_err` | the radial distribution functions g(r) |

The distance edges are pinned by the box, from 0 to `L / sqrt(2)`, which covers every reachable
minimum-image distance, so `dist_out_of_range` is 0. The energy edges come from the reference.
`return_hist=True` adds `'dist'`, `'energy'` and `'rdf'` (bin centres and both g(r)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/glasses_structure_dark.png">
  <img alt="g(r) and per-particle energy histograms of the train and test splits for the three glass formers" src="../figures/images/glasses_structure_light.png">
</picture>

The panel on the 100 000-configuration train split against one half of the test split, next to
its floor, the other half scored against the same. The vertical line marks where
`|g(r) - g_ref(r)|` peaks, the value `rdf_max_abs_err` reports. The shaded span is the panel's
energy edges, from the reference; train energies outside it are dropped from `energy_total_var` and
counted by `energy_out_of_range`. On all three systems the train energies reach past the test ones
at both ends. On the ternary the two g(r) nearly coincide. On the binary the train split has fewer
A-B contacts, the first peak, and `rdf_max_abs_err` reads 0.59 against a floor of 0.01; 9.7% of
its energies fall outside the edges.

## Splits

| Split | How it is built |
| --- | --- |
| test, ternary | This library's swap Monte Carlo, `gen/swap_mc/generate.py`, published as `swapmc/<name>_<spec_id>.npy` and downloaded on first use. See [Reference sets](../reference-sets.md#swap-monte-carlo-the-ternary-glass). |
| test, binary | This library's non-reversible parallel tempering, `gen/parallel_tempering/generate.py`, published as `pt/<name>_<spec_id>.npy` and downloaded on first use: 30 720 configurations at T = 0.2, 91% independent (see [Reference sets](../reference-sets.md#parallel-tempering-particles-and-the-binary-glass)). Cheng et al. 2026 published no configurations. |
| train | Random displacements inside 100 basins. Each basin is a single-particle Monte Carlo chain run for 1500 time units (one time unit is `N` attempted moves), with `N(0, 0.065^2)` displacements and, on the ternary, swaps with probability 0.2. It starts from a lattice with every second row offset by half a spacing and the particles assigned to its sites at random, and runs at the target temperature times a factor running geometrically from 0.8 to 1.25 on the ternary and from 0.25 to 1.25 on the binary, whose cold chains reach its crystalline low end. Each of 1000 chains then starts from a basin, with shares from 1 to 10 that favour the coldest, and runs displacements only, at its own factor of the same range: 100 time units, then one configuration every 5. At T = 0.32 and on the binary the chains stay in their basins. Every test energy lies inside the train range. |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/glasses_swap_dark.gif">
  <img alt="Displacement-only Monte Carlo and the train recipe's schedule run side by side from the recipe's lattice start at T = 0.32" src="../figures/images/glasses_swap_light.gif">
</picture>

Two sets of 64 chains at T = 0.32 from the train recipe's start, the lattice with every second row
offset and the particles shuffled over its sites. One follows the recipe's schedule at the target
temperature: swaps and displacements until t = 1500, where the basins are made, then displacements
only until t = 2100, where the train split ends. The other takes displacements only. The boxes show
one chain of each, the curves the mean energy per particle against the central 90% of the test
split's energies. With swaps the mean enters that band at t = 530 and reads -2.60 when the split
ends; with displacements only it reads -2.46 after 10 000 time units, above the band's top at
-2.54. The recipe itself spreads its basins and chains over temperatures on both sides of the
target's.
