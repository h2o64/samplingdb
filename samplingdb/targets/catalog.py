# The catalog: every target this library ships, under its one name

# Libraries
from .. import data
from ..data import fetch_reference
from .bayesian import LOGISTIC_DATASETS, Brownian, LGCP, LogisticRegression, Seeds
from .mixtures import BracketTwoModes, EightGaussians, GMM40, ManyModes, MoS, TwoModes
from .molecular import PEPTIDE_SYSTEMS, Peptide
from .particles import DW4, LJ13, LJ55
from .physics import KobAndersen2D, KobAndersenBinary
from .synthetic import Funnel, ManyWell
from .toys import Checkerboard, Rings

# One entry per catalogued target, keyed by the target's own name: the name it is built by, cited
# by, cached under and published under. Everything that needs to enumerate the corpus reads this --
# a sweep script, the uniqueness test, the documentation -- rather than keeping a list of its own,
# and a target that is only a parameterisation of another (a peptide sequence, a glass temperature)
# is an entry here rather than a subclass that binds one keyword.
#
# The 'data' flag says whether building the target needs a download: False for a closed-form
# density or a pinned constant, True for anything reading a dataset or a force-field input.
CATALOG = {
    # Tier 0: closed-form densities with exact draws
    'funnel': {'cls': Funnel, 'kwargs': {}, 'data': False},
    'manywell': {'cls': ManyWell, 'kwargs': {}, 'data': False},
    'gmm40': {'cls': GMM40, 'kwargs': {}, 'data': False},
    'mos': {'cls': MoS, 'kwargs': {}, 'data': False},
    'twomodes': {'cls': TwoModes, 'kwargs': {}, 'data': False},
    'brackettwomodes': {'cls': BracketTwoModes, 'kwargs': {}, 'data': False},
    'manymodes-k3': {'cls': ManyModes, 'kwargs': {}, 'data': False},
    'eightgaussians': {'cls': EightGaussians, 'kwargs': {}, 'data': False},
    'rings': {'cls': Rings, 'kwargs': {}, 'data': False},
    'checkerboard': {'cls': Checkerboard, 'kwargs': {}, 'data': False},

    # Tier 1: many-particle systems
    'dw4': {'cls': DW4, 'kwargs': {}, 'data': False},
    'lj13': {'cls': LJ13, 'kwargs': {}, 'data': False},
    'lj55': {'cls': LJ55, 'kwargs': {}, 'data': False},

    # Tier 2: Bayesian posteriors, referenced against Stan or an exact sampler
    'seeds': {'cls': Seeds, 'kwargs': {}, 'data': False},
    'brownian': {'cls': Brownian, 'kwargs': {}, 'data': False},
    # Both LGCP geometries are catalogued: the direct one is what the literature reports, and the
    # whitened one is the geometry Stan samples efficiently
    'lgcp': {'cls': LGCP, 'kwargs': {}, 'data': True},
    'lgcp-whitened': {'cls': LGCP, 'kwargs': {'whitened': True}, 'data': True},

    # Tier 4: glass formers
    'ka-ternary-N44-T1.0': {'cls': KobAndersen2D, 'kwargs': {'temperature': 1.0}, 'data': False},
    'ka-ternary-N44-T0.32': {'cls': KobAndersen2D, 'kwargs': {'temperature': 0.32},
                             'data': False},
    'ka-binary-N64-T0.2': {'cls': KobAndersenBinary, 'kwargs': {'temperature': 0.2},
                           'data': False},
}

# Tier 2's four logistic regressions, one per UCI dataset
CATALOG.update({
    dataset.replace('_', '-'): {'cls': LogisticRegression, 'kwargs': {'dataset': dataset},
                                'data': True}
    for dataset in LOGISTIC_DATASETS
})

# Tier 3's four peptides, one per sequence of the released Sequential Boltzmann Generators data,
# named by sequence and temperature because they come in two force-field families. ace-a-nme-T300
# is the alanine dipeptide of Kohler et al. 2021 and ace-aaa-nme-T300 the Ala4 of Nam et al. 2025.
#
# WARNING: ace-a-nme-T300 is NOT the ff96 + OBC1 variant that fab-torch evaluates against the REMD
# ensemble of Midgley et al. 2023. Same molecule, same atom count, same temperature, different
# force field and therefore a different density. Numbers do not transfer between them.
#
# WARNING: aaa-T310 and aaaaaa-T310 are UNCAPPED, so mdtraj's phi and psi are offset by one
# residue; see Peptide.build_torsion_indexes. They are the catalog's one chain-length ladder at
# fixed specification.
CATALOG.update({
    sequence.lower() + '-T' + str(int(system['temperature'])):
        {'cls': Peptide, 'kwargs': {'sequence': sequence}, 'data': True}
    for sequence, system in PEPTIDE_SYSTEMS.items()
})


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


def download_references(names=None):
    """Downloads the published reference sets and train splits into the cache, ahead of first use

    test_samples() and train_samples() download a missing file by themselves, so this is only for
    fetching ahead of time: on a login node before a job runs on compute nodes with no internet
    access, for instance. A file already in the cache is left as it is, so calling this twice costs
    nothing.

        import samplingdb
        samplingdb.download_references(['dw4', 'lj55'])  # both splits of two targets, 150 MB
        samplingdb.download_references()                 # every published file, 6.3 GB

    Args:
        * names (list of str or None): Catalog keys; None takes every published file, without
            building a target, so it needs none of the optional dependencies (default is None)

    Returns:
        * paths (dict): File in the dataset repo -> its path on this machine

    Raises:
        * FileNotFoundError: For a named target whose reference set is not published yet
    """
    # Name the files, by building each named target for its spec_id
    if names is None:
        files = list(data.REFERENCE_CHECKSUMS)
    else:
        files = []
        for name in names:
            target = build(name)
            if target.reference_folder is not None:
                files.append(target.reference_folder + '/' + target.reference_key() + '.npy')
            train = 'train/' + target.train_key() + '.npy'
            if train in data.REFERENCE_CHECKSUMS:
                files.append(train)

    # Fetch them
    paths = {}
    for file in files:
        folder, key = file[:-len('.npy')].split('/')
        paths[file] = fetch_reference(folder, key, 'Its test_samples() docstring says how to '
                                                   'generate it.')
    return paths
