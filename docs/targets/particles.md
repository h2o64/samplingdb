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

## Metric panel

| Metric | What it compares |
| --- | --- |
| `dist_total_var`, `dist_w1`, `dist_out_of_range` | the interatomic distances, pooled over pairs |
| `energy_total_var`, `energy_w1`, `energy_out_of_range` | the energies |

`return_hist=True` adds `'dist'` and `'energy'`, each with `'edges'`, `'hist'` and `'ref_hist'`.
The binning is DEM's: 200 bins over the reference distances' own range, and 100 energy bins over
`(-26, 0)` for DW-4, `(-60, 0)` for LJ-13 and `(-380, -180)` for LJ-55. See
[Metrics](../metrics.md#binning).

## Splits

| Split | How it is built |
| --- | --- |
| test | This library's non-reversible parallel tempering, `gen/parallel_tempering/generate.py`, published as `pt/<name>_<spec_id>.npy` and downloaded on first use. See [Reference sets](../reference-sets.md#parallel-tempering-particles-and-the-binary-glass). |
| `load_split('train' \| 'val' \| 'test')` | The published splits of Klein et al. 2023, fetched from `jarridrb/DEM` at a pinned commit and checksum: 100 000 train and 10 000 each for val and test, centred on load. Use these to compare with numbers reported against them. |
| train | One Metropolis chain, burn-in included and nothing discarded, from a fixed lattice at the step size of Köhler et al. 2020 (0.5 for DW-4, 0.025 for Lennard-Jones). |

The test split is this library's own because the published DW-4 split is not stationary under
DW-4's own HMC kernel: its mean pair distance moves from 3.970 to 4.037 and stays there, while the
parallel-tempering set does not move. The published Lennard-Jones splits agree with ours; they are
regenerated for consistency and size.
