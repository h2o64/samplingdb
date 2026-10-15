# The catalog: every target this library ships, under the name it is cited by

# Libraries
from .mixtures import BracketTwoModes, EightGaussians, GMM40, ManyModes, MoS, TwoModes
from .particles import DW4, LJ13, LJ55
from .synthetic import Funnel, ManyWell
from .toys import Checkerboard, Rings

# One entry per catalogued target: the short name it is cited by, the class, and the arguments
# that pin it. Everything that needs to enumerate the corpus reads this -- a sweep script, the
# uniqueness test, the documentation -- rather than keeping a list of its own, and a target that is
# only a parameterisation of another (a peptide sequence, a glass temperature) is an entry here
# rather than a subclass that binds one keyword.
#
# The 'data' flag says whether building the target needs a download: False for a closed-form
# density, True for anything reading a dataset, a released ensemble or a force-field file.
CATALOG = {
    # Tier 0: closed-form densities with exact draws
    'funnel': {'cls': Funnel, 'kwargs': {}, 'data': False},
    'manywell': {'cls': ManyWell, 'kwargs': {}, 'data': False},
    'gmm40': {'cls': GMM40, 'kwargs': {}, 'data': False},
    'mos': {'cls': MoS, 'kwargs': {}, 'data': False},
    'twomodes': {'cls': TwoModes, 'kwargs': {}, 'data': False},
    'brackettwomodes': {'cls': BracketTwoModes, 'kwargs': {}, 'data': False},
    'manymodes': {'cls': ManyModes, 'kwargs': {}, 'data': False},
    'eightgaussians': {'cls': EightGaussians, 'kwargs': {}, 'data': False},
    'rings': {'cls': Rings, 'kwargs': {}, 'data': False},
    'checkerboard': {'cls': Checkerboard, 'kwargs': {}, 'data': False},

    # Tier 1: many-particle systems with released reference splits
    'dw4': {'cls': DW4, 'kwargs': {}, 'data': False},
    'lj13': {'cls': LJ13, 'kwargs': {}, 'data': False},
    'lj55': {'cls': LJ55, 'kwargs': {}, 'data': False},
}


def build(name, **kwargs):
    """Builds one catalogued target by name

    Args:
        * name (str): A key of CATALOG
        * kwargs: Extra constructor arguments, overriding the catalogued ones

    Returns:
        * target (Target): The target
    """
    assert name in CATALOG, name + ' is not in the catalog: ' + ', '.join(sorted(CATALOG))
    entry = CATALOG[name]
    return entry['cls'](**{**entry['kwargs'], **kwargs})


def catalog(data=None):
    """Returns the catalogued names, optionally only those that do or do not need a download

    Args:
        * data (bool or None): True for the targets needing a download, False for the ones that
            build offline, None for all of them (default is None)

    Returns:
        * names (list of str): Catalog keys, in catalog order
    """
    return [name for name, entry in CATALOG.items() if data is None or entry['data'] == data]
