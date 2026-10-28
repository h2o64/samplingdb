# samplingdb documentation

- [The Target interface](interface.md): what every target provides, how it handles batches,
  devices and dtypes, how it is named and cached, and the catalog
- [Targets](targets/README.md): the two splits, the full catalog, and one page per family
  - [Synthetic targets](targets/synthetic.md): mixtures, toys, Funnel, ManyWell
  - [Particle systems](targets/particles.md): DW-4, LJ-13, LJ-55
  - [Bayesian posteriors](targets/posteriors.md): four logistic regressions, Seeds, Brownian, LGCP
  - [Peptides](targets/peptides.md): four systems from the Sequential Boltzmann Generators data
  - [Glass formers](targets/glasses.md): the ternary and binary Kob-Andersen mixtures
- [Metrics](metrics.md): `compute_metrics`, the standard panel, each family's panel, binning
- [Reference sets](reference-sets.md): the generators and samplers in `gen/`, their acceptance
  gates, the audits, and how the sets are published

Every figure in these pages is drawn by a script in [figures/](figures/) from the library itself;
`pip install -e ".[docs]"` adds what they need.
