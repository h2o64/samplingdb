# The catalog: every target this library ships, under its one name

# Libraries
from ..data import REFERENCE_CHECKSUMS, fetch_reference
from .mixtures import BracketTwoModes, EightGaussians, GMM40, ManyModes, MoS, TwoModes
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


def download_references(names=None):
    """Downloads the published reference sets into the cache, ahead of their first use

    test_samples() downloads a missing set by itself, so this is only for fetching ahead of time:
    on a login node before a job runs on compute nodes with no internet access, for instance. A set
    already in the cache is left as it is, so calling this twice costs nothing.

        import samplingdb
        samplingdb.download_references(['dw4', 'lj55'])  # two sets, 70 MB
        samplingdb.download_references()                 # every published set, 3.1 GB

    Args:
        * names (list of str or None): Catalog keys; None takes every published set, without
            building a target, so it needs none of the optional dependencies (default is None)

    Returns:
        * paths (dict): File in the dataset repo -> its path on this machine

    Raises:
        * FileNotFoundError: For a named target whose set is not published yet
    """
    # Name the files, by building each named target for its spec_id
    if names is None:
        files = list(REFERENCE_CHECKSUMS)
    else:
        files = []
        for name in names:
            target = build(name)
            assert target.reference_folder is not None, (
                name + ' draws its reference set exactly, so there is nothing to download')
            files.append(target.reference_folder + '/' + target.reference_key() + '.npy')

    # Fetch them
    paths = {}
    for file in files:
        folder, key = file[:-len('.npy')].split('/')
        paths[file] = fetch_reference(folder, key, 'Its test_samples() docstring says how to '
                                                   'generate it.')
    return paths
