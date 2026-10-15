# The target catalog

# Libraries
from .base import ExactTarget, Target, score_autograd, spec_id
from .modes import ModeCoverage
from .bayesian import (
    LOGISTIC_DATASETS, BayesianTarget, Brownian, LGCP, LogisticRegression, Seeds,
)
from .mixtures import (
    MOG, BracketTwoModes, EightGaussians, GMM40, ManyModes, Mixture, MoS, TwoModes,
    log_prob_gaussian,
)
from .molecular import (
    OpenMMTarget, PEPTIDE_SYSTEMS, Peptide, center, dihedral,
)
from .physics import (
    KobAndersen2D, KobAndersenBinary, PeriodicPairMixture, swap_monte_carlo,
)
from .particles import (
    DW4, LJ13, LJ55, LennardJones, MultiDoubleWell, ParticleTarget, remove_mean,
)
from .synthetic import Funnel, ManyWell
from .toys import Checkerboard, Rings
from .catalog import CATALOG, PEPTIDE_NAMES, build, catalog
