# samplingdb: unnormalised sampling targets with train/test splits, and metrics for their samples

# Libraries
from . import data, metrics
from .targets import (
    CATALOG, DW4, LGCP, LJ13, LJ55, MOG, PEPTIDE_SYSTEMS, BayesianTarget,
    BracketTwoModes, Brownian, Checkerboard, EightGaussians, ExactTarget, Funnel, GMM40,
    KobAndersen2D, KobAndersenBinary, LennardJones, LogisticRegression, ManyModes, ManyWell,
    Mixture, ModeCoverage, MoS, MultiDoubleWell, ParticleTarget, Peptide,
    PeriodicPairMixture, Rings, Seeds, Target, TwoModes, build, catalog, download_references,
)

# The single source of the version: pyproject.toml reads this attribute rather than repeating it
__version__ = '0.3.0'
