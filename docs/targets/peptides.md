# Peptides (Tier 3)

Five peptides from the Sequential Boltzmann Generators dataset
(`transferable-samplers/sequential-boltzmann-generators-data` on Hugging Face, MIT). `Peptide(sequence)`
builds any of them. Coordinates are Cartesian positions in nanometres, energies are in kT, and the
working dtype is float32. They need the `[molecular]` extra.

| `build` name | Sequence | Atoms | dim | T | Force field | `name` |
| --- | --- | --- | --- | --- | --- | --- |
| `aldp` | `Ace-A-Nme` | 22 | 66 | 300 K | amber99sbildn + tip3p + amber99_obc, `NoCutoff` | `ace-a-nme-T300` |
| `ala4` | `Ace-AAA-Nme` | 42 | 126 | 300 K | same | `ace-aaa-nme-T300` |
| `trialanine` | `AAA` | 33 | 99 | 310 K | amber14-all + implicit/obc1, `CutoffNonPeriodic` at 2 nm | `aaa-T310` |
| `hexaalanine` | `AAAAAA` | 63 | 189 | 310 K | same | `aaaaaa-T310` |
| `chignolin` | `GYDPETGTWG` | 138 | 414 | 310 K | same | `gydpetgtwg-T310` |

The two force-field families at two temperatures make these five different densities, and a number
from one family does not compare with a number from the other. `AAA` and `AAAAAA` form the one
chain-length ladder at a fixed force field. `aldp` is the alanine dipeptide of Köhler et al. 2021,
not the ff96 + OBC1 variant that `lollcat/fab-torch` evaluates; `ala4` is the Ala4 of Nam et al.
2025; `AAA`, `AAAAAA` and chignolin come from Tan et al. 2025.

## The energy

OpenMM builds each system and owns every parameter: force-field files, atom typing and topology.
`TorchForceField` reads those parameters out of the built `openmm.System` and evaluates the potential
in batched torch. It covers the two families above and asserts on any force, constraint, switching
function or solvent setting it does not know. Agreement with OpenMM's `Reference` platform is
asserted per system in `tests/test_molecular.py` (about 1e-5 kJ/mol/nm on forces of order 4e3), and
`gen/openmm_check/compare.py` reports it force group by force group.

- `log_prob_and_grad` returns both from a single pass and is the call to use.
- `compile=True` (the default) runs the potential through `torch.compile`, about 2.2x faster after a
  one-off ~15 s per batch shape. Turn it off for short, interactive or shape-varying work. The eager
  path is used automatically whenever the graph has to survive the call.
- `differentiable_score=True` keeps the score differentiable for a loss built on it, at the cost of
  a retained backward graph. It is a plain attribute of `target.bridge` and can be flipped later.
- Energies above `energy_cut = 1e8` kT are softened logarithmically up to `energy_max = 1e20` kT
  (bgflow's `linlogcut`), so an atomic overlap gives a large finite number. Both constants are part
  of the specification.
- `harmonic_log_prob` and `harmonic_score` give the bond and angle terms alone, the beta = 0 end of a
  solute-tempering path.
- `minimised_positions()` minimises the released structure, which is where any trajectory should
  start. `run_dynamics` runs OpenMM's `LangevinMiddleIntegrator`; `run_batched_dynamics` runs the same
  BAOAB scheme on a whole batch in torch, on the same force field.

## Metric panel

| Metrics | What they compare |
| --- | --- |
| `kl_<torsion>`, `w1_<torsion>` | each backbone torsion marginal (`phi1`, ..., `psi1`, ..., `omega1`, ...) |
| `js_ramachandran`, `ramachandran_occupancy` | the first residue's phi/psi joint, on a 32 x 32 grid, and the fraction of that grid the reference fills |
| `dist_total_var`, `dist_w1`, `dist_out_of_range` | the interatomic distances |
| `energy_total_var`, `energy_out_of_range` | the energies, on edges derived from the reference |
| `energy_w1`, `energy_w2`, `torus_w2`, `tica_w2`, `tica_k_jsd`, `torus_k_jsd`, `n_ot_samples` | the panel of `transferable-samplers`, under its names |

The last row reproduces `transferable-samplers`' metrics in batched torch and was checked against
their implementation: `energy_w1`, `energy_w2` and `tica_w2` agree to 1e-12, `torus_w2` and the TICA
features to 1e-7, and the k-means divergences fall inside scikit-learn's own seed spread.
`compute_upstream_metrics` returns that panel alone. Its transport problems are cut to
`max_ot_samples` (2000) samples per side, reported back as `n_ot_samples`.

- `torus_w2` is exact transport on all phi and psi at once, with the per-coordinate distance
  `min(|d|, 2 pi - |d|)`. `tica_w2` is exact transport on the first two TICA components, and the two
  `*_k_jsd` are Jensen-Shannon divergences of 20 k-means cluster populations, on the TICA plane and on
  the sin/cos-wrapped dihedrals.
- TICA is deeptime's Koopman-reweighted TICA at lag 100, fitted once on 20 000 time-ordered frames of
  the released train split and cached per system. Its input features are ordered as upstream orders
  them: C, N and S upper-triangular distances, then sin and cos of phi, of psi and of omega.
  `TICA_LAYOUT` names that ordering in the cache file.
- A Ramachandran pair takes phi and psi of the same residue, matched through mdtraj's atom indexes.
  On the uncapped `AAA` and `AAAAAA`, mdtraj's phi and psi arrays are offset by one residue.
- The divergences use 64 bins per marginal and a 32 x 32 joint. `return_hist=True` returns the
  200 x 200 plot-grade Ramachandran histograms, every torsion marginal, and the upstream histograms:
  energy, pair distances, centre-of-mass norm, one Ramachandran per residue, and the TICA plane.

## Splits

| Split | How it is built |
| --- | --- |
| test | This library's replica-exchange molecular dynamics with solute tempering, `gen/md/generate.py`, read from `md/<name>.npy` in the cache: 100 000 structures, 10 000 for chignolin. See [Reference sets](../reference-sets.md#replica-exchange-md-peptides). |
| `load_split('train' \| 'val' \| 'test')` | The released SBG frames, centred on load. Use these to compare with numbers reported against them. |
| train | The pretraining protocol of Nam et al. 2025: 100 ps of Langevin dynamics at the system's temperature and friction, 1 fs timestep, from the minimised released structure. It is one trajectory, so `n` sets the stride between kept frames. |

The test split is this library's own because the released splits are far from independent: their
integrated autocorrelation times range from 4.3 stored frames on `AAA` to 1115 on chignolin, whose
20 000-frame test split is worth about 18 independent structures. Chignolin's set holds 10 000
structures, the size of the chignolin test set of Tan et al. 2025, because its fold is slow to
decorrelate.

About the released files: the train split is mean-centred and val and test are not, which
`load_split` corrects, and chignolin's val and test splits are the same file.
