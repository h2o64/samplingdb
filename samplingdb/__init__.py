# samplingdb: unnormalised sampling targets with train/test splits, and metrics for their samples

# Libraries
from . import baselines, data, metrics
from .targets import (
    CATALOG, DW4, LJ13, LJ55, MOG, BracketTwoModes, Checkerboard, EightGaussians, ExactTarget,
    Funnel, GMM40, LennardJones, ManyModes, ManyWell, Mixture, ModeCoverage, MoS, MultiDoubleWell,
    ParticleTarget, Rings, Target, TwoModes, build, catalog,
)

# The single source of the version: pyproject.toml reads this attribute rather than repeating it
__version__ = '0.3.0'
