# samplingdb: unnormalised sampling targets with train/test splits, and metrics for their samples

# Libraries
from . import baselines, data, metrics
from .targets import (
    CATALOG, MOG, BracketTwoModes, Checkerboard, EightGaussians, ExactTarget, Funnel, GMM40,
    ManyModes, ManyWell, Mixture, ModeCoverage, MoS, Rings, Target, TwoModes, build, catalog,
)

# The single source of the version: pyproject.toml reads this attribute rather than repeating it
__version__ = '0.3.0'
