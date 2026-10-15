# The target catalog

# Libraries
from .base import ExactTarget, Target, score_autograd, spec_id
from .modes import ModeCoverage
from .mixtures import (
    MOG, BracketTwoModes, EightGaussians, GMM40, ManyModes, Mixture, MoS, TwoModes,
    log_prob_gaussian,
)
from .particles import (
    DW4, LJ13, LJ55, LennardJones, MultiDoubleWell, ParticleTarget, remove_mean,
)
from .synthetic import Funnel, ManyWell
from .toys import Checkerboard, Rings
from .catalog import CATALOG, build, catalog
