# Particle systems (Tier 1)

Clusters whose energy is invariant to translation, rotation and permutation of the particles. They
work in float32, and `preprocess()` removes the centre of mass before any metric sees them.

| Class | dim | `name` | Energy |
| --- | --- | --- | --- |
| `DW4` | 8 | `dw4` | 4 particles in 2D, `E = sum_{i<j} [a (d_ij - d_0)^4 + b (d_ij - d_0)^2 + c] / tau` with `(a, b, c, d_0, tau) = (0.9, -4, 0, 4, 1)` |
| `LJ13` | 39 | `lj13` | 13 particles in 3D, `E = (eps / tau) sum_{i != j} [(r / d_ij)^12 - 2 (r / d_ij)^6] + c_harm sum_i \|x_i - mean(x)\|^2` with `eps = tau = r = 1`, `c_harm = 0.5` |
| `LJ55` | 165 | `lj55` | the same with 55 particles |

Both potentials follow bgflow's `MultiDoubleWellPotential` and `LennardJonesPotential`, which
generated the published data (Köhler et al. 2020, Klein et al. 2023). The general classes are
`MultiDoubleWell` and `LennardJones`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/particles_configurations_dark.png">
  <img alt="DW-4 configurations of the test and train splits, and LJ-13 and LJ-55 renders of each split" src="../figures/images/particles_configurations_light.png">
</picture>

Top: six DW-4 configurations of each split, each turned onto its principal axes, with the six pair
distances drawn. Each pair sits near one of the two wells of the pair potential, at about 2.5 or 5.5.
The first train state is the starting lattice, a square of side `d_0`. Bottom: one configuration of
each split per cluster, at one scale, each particle shaded by its number of neighbours within
`1.4 r`, the first minimum of the pair distribution. The test configuration has the median energy
of the first 2048 test samples, and the train one is the last state of `train_samples(n=2048)`. At
this temperature both clusters are liquid droplets, and after 2048 steps the LJ-55 train chain is
still close to the cubic lattice it starts from.

## Conventions

- **Pair multiplicity.** DW-4 counts each unordered pair once; the Lennard-Jones clusters count
  each pair twice, summing over `i != j` (`pair_multiplicity = 2`). That is what bgflow computes with
  DEM's `energy_factor = 1.0`, and the released frames are stationary under a Metropolis chain at
  their own step size with this convention and drift away under the halved one.
  `tests/test_particles_data.py` checks both. The published formula, `eps / (2 tau)` over ordered
  pairs, is therefore not used.
- **Coefficient order.** DW-4's `(a, b, c)` are (quartic, quadratic, constant). Köhler et al. 2020
  and Havens et al. 2026 write a (linear, quadratic, quartic) triple; transcribing one into the
  other's slots drops the quartic term and makes the density improper.
- **Centre-of-mass oscillator.** The `c_harm = 0.5` term of the Lennard-Jones energy comes from
  `vgsatorras/en_flows`.
- **Finite score.** The score is finite at every configuration, particles on top of each other
  included: each pair force is `f'(d)` along the unit vector between the two particles, and the
  Lennard-Jones `r / d` is capped so that `f'(d)` stays representable in float32.

## Metric panel

| Metric | What it compares |
| --- | --- |
| `dist_total_var`, `dist_w1`, `dist_out_of_range` | the interatomic distances, pooled over pairs |
| `energy_total_var`, `energy_w1`, `energy_out_of_range` | the energies |

`return_hist=True` adds `'dist'` and `'energy'`, each with `'edges'`, `'hist'` and `'ref_hist'`.
The binning is DEM's: 200 bins over the reference distances' own range, and 100 energy bins over
`(-26, 0)` for DW-4, `(-60, 0)` for LJ-13 and `(-380, -180)` for LJ-55. See
[Metrics](../metrics.md#binning).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/particles_histograms_dark.png">
  <img alt="Pair-distance and energy histograms of the train and test splits of DW-4, LJ-13 and LJ-55, with the panel's numbers" src="../figures/images/particles_histograms_light.png">
</picture>

The panel of `train_samples(n=2048)` against the first 2048 test configurations, from
`compute_metrics(train, ref_samples=test, return_hist=True)`, drawn on bins half as wide as the
panel's, each bin the share of all the samples it holds. The shaded bands lie outside the edges, where the panel drops mass without
counting it in the total variation. `*_out_of_range` reports the dropped share of the scored samples,
and the reference is cut the same way: 0.15% of the LJ-13 test energies lie below the -60 edge. The
LJ-55 train chain still sits on its starting lattice, so its distances fall at the lattice spacings
and its energies do not overlap the reference at all: `energy_total_var` reads its maximum of 1
while `energy_out_of_range` reads 0.

## Splits

| Split | How it is built |
| --- | --- |
| test | This library's non-reversible parallel tempering, `gen/parallel_tempering/generate.py`, published as `pt/<name>_<spec_id>.npy` and downloaded on first use. See [Reference sets](../reference-sets.md#parallel-tempering-particles-and-the-binary-glass). |
| train | One Metropolis chain, burn-in included and nothing discarded, from a fixed lattice at the step size of Köhler et al. 2020 (0.5 for DW-4, 0.025 for Lennard-Jones), returned with the centre of mass removed like the test split. |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/images/particles_train_chain_dark.gif">
  <img alt="The LJ-55 train chain over 100 000 Metropolis steps, next to its energy and the test split's energies" src="../figures/images/particles_train_chain_light.gif">
</picture>

The LJ-55 train recipe run for the default 100 000 steps, shown at 100 evenly spaced states with
particles shaded by neighbour count as above. `train_samples(n)` keeps the first `n` states of this
one chain, so every train split is a prefix of it. The chain accepts 0.4% of its moves. Its energy
falls from -262.8 on the lattice to -304.9 at the last step, which is still above 92.6% of the
first 2048 test energies, and 63% of its states lie above the test split's 99th percentile.

The test split is this library's own because the published DW-4 split of Klein et al. 2023 (in
`jarridrb/DEM`) is not stationary under DW-4's own HMC kernel: its mean pair distance moves from
3.970 to 4.037 and stays there, while the parallel-tempering set moves by 0.002. The published
Lennard-Jones splits are close to ours (LJ-13 energy -43.13 ± 7.02 against -42.95 ± 6.91 here,
LJ-55 -325.15 ± 14.19 against -325.40 ± 13.76); ours are generated for consistency and size.
