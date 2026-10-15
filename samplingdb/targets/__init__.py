# The target catalog

# Libraries
from .base import ExactTarget, Target, score_autograd, spec_id
from .modes import ModeCoverage
from .mixtures import (
    MOG, BracketTwoModes, EightGaussians, GMM40, ManyModes, Mixture, MoS, TwoModes,
    log_prob_gaussian,
)
from .synthetic import Funnel, ManyWell
from .toys import Checkerboard, Rings
from .catalog import CATALOG, build, catalog
