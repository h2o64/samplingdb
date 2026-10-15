# Baselines

`samplingdb.baselines` holds the classical samplers and the parallel tempering behind the generated
reference sets. Every sampler runs one chain per row of its starting tensor, vectorised over chains,
and drives the target's analytic score.

## Classical samplers

| Function | Kernel |
| --- | --- |
| `random_walk_mh(log_prob, x0, n_steps, step_size, proposal='normal')` | random-walk Metropolis-Hastings, Gaussian or Student-t(2) proposals |
| `ula(score, x0, n_steps, step_size)` | the unadjusted Langevin algorithm |
| `mala(log_prob, score, x0, n_steps, step_size)` | Metropolis-adjusted Langevin |
| `hmc(log_prob, score, x0, n_steps, step_size, n_leapfrog=10)` | Hamiltonian Monte Carlo |

They share `generator`, `return_trace` and `thin` arguments, and are built on `run_chains`,
`metropolis` and `leapfrog`. `student_t2` draws the Student-t(2) noise.

```python
import torch
from samplingdb import build
from samplingdb.baselines import mala

target = build('seeds')
x0 = target.prior_scale * torch.randn(1024, target.dim, dtype=torch.float64)
x = mala(target.log_prob, target.score, x0, n_steps=1000, step_size=0.005)
```

## Non-reversible parallel tempering

`non_reversible_parallel_tempering(log_prob, score, x0, beta, explorer=None, n_rounds=10,
reference=None, ...)` is the scheme of Syed et al. 2022 (JRSS-B, Algorithms 1 to 4), the one
Pigeons.jl implements, and it generates every tempered reference set in this library.

- **Swaps are deterministic**, alternating between even and odd pairs of adjacent levels, so a
  configuration walks the ladder ballistically. A swap permutes an index array; the configurations
  never move.
- **The schedule adapts.** The run proceeds in rounds of doubling length, and between rounds the
  levels are re-placed at equal increments of the measured communication barrier. Samples come from
  the final round, where nothing adapts. The cumulative barrier is interpolated piecewise linearly,
  which is monotone by construction.
- **Step sizes adapt** towards `target_acceptance` (0.65) with a damped update between rounds, after
  `calibrate_step_size` brackets each level's step size by geometric search before round 1.
- **A frozen run** passes a converged schedule and step sizes back in with `n_rounds=1`,
  `adaptation_gain=0.0` and any `n_scans`, with `n_burn` scans discarded first. That is how a
  calibrated ladder is reused across shards.

The local kernel comes from an explorer factory: `hmc_explorer`, `mala_explorer`,
`random_walk_explorer` (for a target with no useful score, such as `Checkerboard`), or
`dynamics_explorer`, which wraps a molecular-dynamics integrator and carries velocities from one
scan to the next.

With `reference=None` the path is `beta * log_prob` and the hottest level sits at the schedule's
smallest `beta`. Otherwise the `beta = 0` end of the linear path is one of three references:

| Reference | Use |
| --- | --- |
| `GaussianReference` | the isotropic Gaussian of `prior_scale`; normalised, so the stepping stone gives `log Z` |
| `HarmonicReference` | particles tethered to a structure (the Einstein crystal of Frenkel and Ladd 1984) |
| `PotentialReference` | a sub-potential of the target, which tempers only the rest of it: solute tempering (Wang, Friesner and Berne 2011; Kamiya and Sugita 2018) |

`return_diagnostics=True` returns the tuned schedule and step sizes, the global barrier `Lambda`, the
rejection rate per pair of levels, the explorer acceptance per level and per chain, the round trips
per chain against the rate `1 / (2 + 2 Lambda)` the barrier allows, the recommended number of
levels `2 Lambda`, the cost of one round trip in scans, the stepping-stone `log_normaliser`, and the
final state.

`linear_schedule(n_rungs, beta_min)` builds a starting schedule and `update_schedule` re-places one
from a measured barrier.

## Fixed-ladder parallel tempering

`parallel_tempering(log_prob, score, x0, beta, n_sweeps, step_size)` keeps a fixed ladder, built
for instance by `geometric_ladder(n_rungs, beta_min)`, with HMC moves on every level and the same
deterministic even/odd swaps. It reports round trips per chain and swap acceptance per pair.

## Swap Monte Carlo

`targets.physics.swap_monte_carlo(target, x, n_time_units, step_size=0.065, p_swap=0.0)` runs the
move pool of ParticlesMC (`TheDisorderedOrganization/ParticlesMC`) on a glass former. With
probability `1 - p_swap` a move displaces one particle by an isotropic Gaussian of scale
`step_size`; otherwise it exchanges two particles of the species pair a swap channel names. The
ternary mixture's channels are A<->C and B<->C, never A<->B, and ParticlesMC's setup uses
`p_swap = 0.2`. One time unit is `n_particles` attempted moves. Since species labels are fixed here,
a swap exchanges the two particles' positions, which is the same move. `p_swap = 0` is the
displacement-only chain of the train splits. A seeded run gives the same chain on CPU and GPU.
