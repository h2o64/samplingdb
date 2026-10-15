# samplingdb: unnormalised sampling targets with train/test splits, and metrics for their samples

# Libraries
from . import baselines, data, metrics
from .targets import (
    ExactTarget, ModeCoverage, Target,
)

# The single source of the version: pyproject.toml reads this attribute rather than repeating it
__version__ = '0.3.0'
