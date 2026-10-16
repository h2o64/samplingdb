# Cache-dir resolution, checksums and downloads for the reference datasets

# NOTE: every third-party file is fetched from the source that published it -- a pinned GitHub
# commit for Tier 1, a Hugging Face dataset for Tier 3, a Zenodo record for Tier 4 -- rather than
# mirrored into a repo of our own, so the redistribution question never has to be answered for
# them. The reference sets this library generates itself are the exception: they are published in
# REFERENCE_REPO and fetched from a pinned tag. SAMPLINGDB_DATA_DIR overrides any of them: if the
# file it names exists, it is used and nothing is downloaded.

# Libraries
import hashlib, json, os, pathlib, shutil, urllib.request
import numpy as np

# Where the assets shipped inside the wheel live
ASSETS = pathlib.Path(__file__).resolve().parent / 'targets' / 'assets'

# The Hugging Face dataset repo holding the generated reference sets, and the tag they are read at.
# A file is named <folder>/<name>_<spec_id>.npy, so a density change is a new file; a published
# file is never replaced, and a release adds files under a new tag, so every tag keeps serving what
# the code that pins it expects. gen/publish/publish.py uploads and tags a release
REFERENCE_REPO = 'h2o64/samplingdb'
REFERENCE_REVISION = 'v1'
REFERENCE_URL = ('https://huggingface.co/datasets/' + REFERENCE_REPO + '/resolve/'
                 + REFERENCE_REVISION + '/')

# The sha256 of every published reference set at REFERENCE_REVISION. The binary glass and chignolin
# are not in it: the first set failed its stationarity audit and the second has not been run
REFERENCE_CHECKSUMS = {
    'pt/dw4_6b739c6d.npy': '73bd27b31005afb9c86a4bb26e9f8f57f9332479d5966d9728dbe6cb9126a319',
    'pt/lj13_6a0a72c4.npy': '19f1d7b274e0dbf83584ae01873663885c6a0f1e43d79256d490861715faa26e',
    'pt/lj55_dc3f6b7e.npy': '2465eac78b416d6e3e9b10b96182bc18cab607c29cd1f9e381d1640c0aa3bfa5',
    'stan/seeds_f9a50f20.npy': '19c0c996178d02acf0c9e45460792f0f469da1e89a598ab54ed4119e3b0d501d',
    'stan/brownian_45794097.npy':
        'c55757ca7d4c0ce7680971c8ef1866a08bede69944fe9b4db360a170eb046ede',
    'stan/lgcp_cea6d9a6.npy': '3c1f27ec1b4be94eab7847e0b92f234a13dec96fa4bbfd22ceb689c1c063d001',
    'stan/lgcp-whitened_30e98363.npy':
        '10110da822a0cb78b97d5650d56f20bd52c76ff4545039fb432876745a4675ad',
    'stan/breast-cancer_543f5f94.npy':
        'ebde66482d5bc9104e90db35b19e303d5800d1b64a7b9412038f25065eaf4d5e',
    'stan/german-credit_3cfdf175.npy':
        'd185191886828640a84ef38b87d636db995d2e77bf92dfc0433f54b5ea53165e',
    'stan/ionosphere_d00109c3.npy':
        'bb8f283aacf0511bec321e9724037330ea26ce55879f76ef13d438a6599cecb8',
    'stan/sonar_52aa10ef.npy': '6780d7e487bd5929dc08cae21e6e80a5ad018ce010e78e28092f878761eb3e50',
    'md/aaa-T310_e0bfbee3.npy': '92f1c30a9bab0f943b5bb5f229bd0592b8a059426e792d49d4c64c6e629f21ea',
    'md/aaaaaa-T310_08d2d916.npy':
        '2a947acef2db083fdf19d3a85b024c7d520e722557aa6d2c744214fb24e2fa50',
    'md/ace-a-nme-T300_54968005.npy':
        '5722c09dde488b0f0182076c4b932821381c8f79d3a4a850dac257ee18bc96f5',
    'md/ace-aaa-nme-T300_cda31701.npy':
        '5eeb2b8d38d24fd9032423862d22ce97102901adb79ff49ec1a3e746c67addfb',
    'swapmc/ka-ternary-N44-T0.32_d50e8a3f.npy':
        '61c4dd3bc1511f8ad70a7db7104a1cee95f395ef5fd68bfa18ef7aafda05e3a2',
    'swapmc/ka-ternary-N44-T1.0_c0f9b3b8.npy':
        'ffe190b55e087563c8d9cba910627bb987e8c90275b72fec204739cc5289693e',
}


def cache_dir():
    """Returns the cache directory, creating it if needed

    NOTE: where home directories are quota-limited, point SAMPLINGDB_CACHE at bulk storage --
    the generated reference sets of the whole catalog run to a few gigabytes.

    Returns:
        * path (pathlib.Path): Cache directory
    """
    path = os.environ.get('SAMPLINGDB_CACHE')
    if path is None:
        path = pathlib.Path.home() / '.cache' / 'samplingdb'
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path):
    """Returns the sha256 hex digest of a file"""
    return _digest(path, hashlib.sha256())


def md5(path):
    """Returns the md5 hex digest of a file, the checksum Zenodo publishes"""
    return _digest(path, hashlib.md5())


def _digest(path, digest):
    """Streams a file through a hashlib object and returns its hex digest"""
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_asset(name):
    """Loads a small pinned array shipped next to the code (GMM-40 / MoS mean matrices)

    Args:
        * name (str): File name inside samplingdb/targets/assets

    Returns:
        * array (numpy.ndarray): The pinned array
    """
    path = ASSETS / name
    if not path.exists():
        raise FileNotFoundError(
            str(path) + ' is missing. Regenerate it with gen/assets/make_mixture_means.py.'
        )
    return np.load(path)


def local_path(name):
    """Returns the cache path of a host-generated array"""
    return cache_dir() / (name + '.npy')


def cached_local(name, build_fn):
    """Returns a host-generated array from the cache, building it on first call

    Args:
        * name (str): Cache key, which must embed the target's name
        * build_fn (function): () -> numpy.ndarray, run only on a cache miss

    Returns:
        * array (numpy.ndarray): The cached array
    """
    path = local_path(name)
    if path.exists():
        return np.load(path)
    array = build_fn()
    np.save(path, array)
    return array


def read_rows(path, n, command):
    """Reads the first n rows of a generated reference set

    Args:
        * path (pathlib.Path): The .npy file
        * n (int or None): Number of rows; None reads all of them
        * command (str): What generates a larger set, quoted when the file holds fewer than n

    Returns:
        * array (numpy.ndarray of shape (n, dim)): The rows, in memory

    Raises:
        * ValueError: When the file holds fewer than n rows
    """
    array = np.load(path, mmap_mode='r')
    if n is not None and n > len(array):
        raise ValueError(
            '%s holds %d samples and %d were asked for. Score fewer samples, pass ref_samples, or '
            'generate a larger set with: %s' % (path, len(array), n, command)
        )
    return np.array(array[:n])


def fetch_url(url, name, checksum=None, folder=None):
    """Downloads one file from a URL into the cache, once

    Args:
        * url (str): The URL to fetch
        * name (str): File name to cache it under
        * checksum (str or None): Expected sha256, verified when given (default is None)
        * folder (str or None): Sub-directory of the cache (default is None)

    Returns:
        * path (pathlib.Path): Path to the downloaded (or locally provided) file
    """
    directory = cache_dir() if folder is None else cache_dir() / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name

    # A local directory always wins, so an offline user can point at their own copy
    local_dir = os.environ.get('SAMPLINGDB_DATA_DIR')
    if local_dir is not None and (pathlib.Path(local_dir) / name).exists():
        path = pathlib.Path(local_dir) / name
    elif not path.exists():
        # A User-Agent is not optional: the NIST potentials repository answers 403 to urllib's
        # default one, and a silent 403 would be indistinguishable from a missing file. The body is
        # streamed to disk, since a reference set runs to a gigabyte, into a file of this process's
        # own, so two jobs sharing a cache never write into each other's download
        request = urllib.request.Request(url, headers={'User-Agent': 'samplingdb'})
        partial = path.with_suffix(path.suffix + '.partial' + str(os.getpid()))
        with urllib.request.urlopen(request) as response, open(partial, 'wb') as f:
            shutil.copyfileobj(response, f, 1 << 24)

        # A download that fails its checksum is deleted, so the next call fetches it again
        if checksum is not None and sha256(partial) != checksum:
            partial.unlink()
            raise ValueError('sha256 mismatch for the download of ' + url + ': expected '
                             + checksum + '. The file was deleted; try again.')
        partial.rename(path)
        return path

    # Verify the checksum when the caller pins one
    if checksum is not None:
        digest = sha256(path)
        if digest != checksum:
            raise ValueError(
                'sha256 mismatch for ' + str(path) + ': expected ' + checksum + ', got ' + digest
            )
    return path


def fetch_reference(folder, key, hint):
    """Returns the path of a generated reference set, downloading the published one on a cache miss

    A set in the cache is used as it is, whether it was downloaded or generated on this machine, so
    a freshly generated set is read before it is published. Otherwise the set is downloaded from
    REFERENCE_REPO at REFERENCE_REVISION and checked against REFERENCE_CHECKSUMS once, when it
    lands. The cache is laid out like the dataset repo, so on a machine with no internet access
    `hf download <REFERENCE_REPO> --repo-type dataset --local-dir $SAMPLINGDB_CACHE`, run anywhere
    and copied over, fills it.

    Args:
        * folder (str): Cache sub-directory of the generator that writes it: 'pt', 'md', 'stan' or
            'swapmc'
        * key (str): File stem, the target's name and spec_id, as Target.reference_key returns it
        * hint (str): How to generate the set, quoted when it is neither cached nor published

    Returns:
        * path (pathlib.Path): Path to the .npy file

    Raises:
        * FileNotFoundError: When the set is neither in the cache nor published at
            REFERENCE_REVISION
    """
    # A set in the cache is used as it is
    name = folder + '/' + key + '.npy'
    path = cache_dir() / name
    if path.exists():
        return path

    # Otherwise download the published set, once
    if name not in REFERENCE_CHECKSUMS:
        raise FileNotFoundError(
            'No reference set for ' + key + '. It is not in the cache (' + str(path) + '), and '
            'the dataset repo ' + REFERENCE_REPO + ' publishes none at revision '
            + REFERENCE_REVISION + '. ' + hint
        )
    print('Downloading ' + name + ' from huggingface.co/datasets/' + REFERENCE_REPO + ' into '
          + str(path) + ', once')
    return fetch_url(REFERENCE_URL + name, key + '.npy', checksum=REFERENCE_CHECKSUMS[name],
                     folder=folder)


def fetch_zenodo(record_id, filename, checksum=None):
    """Downloads one file of a Zenodo record into the cache, once

    Reference sets published under a DOI are fetched from the DOI itself rather than mirrored,
    -- slower once, correct always, and no redistribution question to answer.

    Args:
        * record_id (str): Zenodo record id, e.g. '17966995'
        * filename (str): File name inside the record
        * checksum (str or None): Expected md5, the digest Zenodo publishes (default is None)

    Returns:
        * path (pathlib.Path): Path to the downloaded file
    """
    folder = cache_dir() / ('zenodo-' + record_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename

    # A local directory always wins, so an offline user can point at their own copy
    local_dir = os.environ.get('SAMPLINGDB_DATA_DIR')
    if local_dir is not None and (pathlib.Path(local_dir) / filename).exists():
        path = pathlib.Path(local_dir) / filename
    elif not path.exists():
        url = 'https://zenodo.org/api/records/' + record_id + '/files/' + filename + '/content'
        partial = path.with_suffix(path.suffix + '.partial')
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)

    # Verify the checksum when the caller pins one
    if checksum is not None:
        digest = md5(path)
        if digest != checksum:
            raise ValueError(
                'md5 mismatch for ' + str(path) + ': expected ' + checksum + ', got ' + digest
            )
    return path


def manifest(name, spec, protocol, checksums):
    """Builds the manifest.json payload that travels with every generated reference set

    A reference ensemble is only as good as the protocol that made it, so every file the gen/
    scripts write gets a sibling manifest.json naming the target, the version of its density, the
    sampler settings and the sha256 of the array. That is what lets a number be traced back to the
    run that produced it, and what makes a regenerated set comparable to the one it replaces.

    Args:
        * name (str): The target's name
        * spec (str): The spec_id of the density the set was drawn from
        * protocol (dict): Generation protocol metadata (sampler, steps, seed, versions)
        * checksums (dict): File name -> sha256

    Returns:
        * payload (str): The manifest, as pretty-printed JSON
    """
    return json.dumps(
        {'target': name, 'spec_id': spec, 'protocol': protocol, 'checksums': checksums},
        indent=2,
        sort_keys=True,
    )


def fetch_hf(repo_id, path_in_repo, repo_type='dataset'):
    """Downloads one file from any Hugging Face repo into the cache, once

    NOTE: huggingface_hub is an optional dependency of the [molecular] extra, since the peptide
    tier is the only thing that fetches from the hub, so the import is guarded and lazy.

    Args:
        * repo_id (str): Repo id, e.g. 'transferable-samplers/sequential-boltzmann-generators-data'
        * path_in_repo (str): File path inside the repo
        * repo_type (str): 'dataset' or 'model' (default is 'dataset')

    Returns:
        * path (pathlib.Path): Path to the downloaded (or locally provided) file
    """
    local_dir = os.environ.get('SAMPLINGDB_DATA_DIR')
    if local_dir is not None and (pathlib.Path(local_dir) / path_in_repo).exists():
        return pathlib.Path(local_dir) / path_in_repo
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            'huggingface_hub is required to download reference data. Install it with '
            'pip install samplingdb[molecular], or set SAMPLINGDB_DATA_DIR to a local copy.'
        )
    return pathlib.Path(hf_hub_download(repo_id=repo_id, repo_type=repo_type,
                                        filename=path_in_repo, cache_dir=str(cache_dir())))
