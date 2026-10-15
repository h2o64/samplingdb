# Cache-dir resolution, checksums and downloads for the reference datasets

# NOTE: every reference set is fetched from the source that published it -- a pinned GitHub commit
# for Tier 1, a Hugging Face dataset for Tier 3, a Zenodo record for Tier 4 -- rather than mirrored
# into a repo of our own. Following that rule everywhere means the redistribution question never
# has to be answered at all. SAMPLINGDB_DATA_DIR overrides
# any of them: if the file it names exists, it is used and nothing is downloaded.

# Libraries
import hashlib, json, os, pathlib, urllib.request
import numpy as np

# Where the assets shipped inside the wheel live
ASSETS = pathlib.Path(__file__).resolve().parent / 'targets' / 'assets'


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
        # default one, and a silent 403 would be indistinguishable from a missing file
        request = urllib.request.Request(url, headers={'User-Agent': 'samplingdb'})
        partial = path.with_suffix(path.suffix + '.partial')
        with urllib.request.urlopen(request) as response:
            partial.write_bytes(response.read())
        partial.rename(path)

    # Verify the checksum when the caller pins one
    if checksum is not None:
        digest = sha256(path)
        if digest != checksum:
            raise ValueError(
                'sha256 mismatch for ' + str(path) + ': expected ' + checksum + ', got ' + digest
            )
    return path


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


def manifest(name, protocol, checksums):
    """Builds the manifest.json payload that travels with every generated reference set

    A reference ensemble is only as good as the protocol that made it, so every file the gen/
    scripts write gets a sibling manifest.json naming the target, the sampler settings and the
    sha256 of the array. That is what lets a number be traced back to the run that produced it,
    and what makes a regenerated set comparable to the one it replaces.

    Args:
        * name (str): The target's name
        * protocol (dict): Generation protocol metadata (sampler, steps, seed, versions)
        * checksums (dict): File name -> sha256

    Returns:
        * payload (str): The manifest, as pretty-printed JSON
    """
    return json.dumps(
        {'target': name, 'protocol': protocol, 'checksums': checksums},
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
