# Cache-dir resolution, checksums and downloads for the reference datasets

# NOTE: every third-party input is fetched from the source that published it -- a pinned GitHub
# commit for the Tier 2 datasets, a pinned Hugging Face revision for the Tier 3 structures --
# rather than mirrored into a repo of our own, so the redistribution question never has to be
# answered for them, and every one is checked against its sha256. The reference sets and the
# default train splits this library generates itself are published in REFERENCE_REPO and fetched
# from a pinned tag.
# SAMPLINGDB_DATA_DIR overrides any of them: if the file it names exists, it is used, checksum
# still verified, and nothing is downloaded.

# Libraries
import hashlib, os, pathlib, shutil, urllib.request
import numpy as np

# The Hugging Face dataset repo holding the generated reference sets and train splits, and the tag
# they are read at. A reference set is named <folder>/<name>_<spec_id>.npy and a train split
# train/<name>_<spec_id>_<recipe>.npy, so a density or recipe change is a new file. A set
# regenerated for the same density keeps its name and changes its bytes under a new tag: every tag
# keeps serving what the code that pins it expects, and verify_once refuses a cache filled at
# another one. gen/publish/publish.py uploads and tags a release
REFERENCE_REPO = 'h2o64/samplingdb'
REFERENCE_REVISION = 'v5'
REFERENCE_URL = ('https://huggingface.co/datasets/' + REFERENCE_REPO + '/resolve/'
                 + REFERENCE_REVISION + '/')

# The sha256 of every published file at REFERENCE_REVISION
REFERENCE_CHECKSUMS = {
    'pt/dw4_d9994289.npy': '73bd27b31005afb9c86a4bb26e9f8f57f9332479d5966d9728dbe6cb9126a319',
    'pt/lj13_bd3bc3a0.npy': '19f1d7b274e0dbf83584ae01873663885c6a0f1e43d79256d490861715faa26e',
    'pt/lj55_01c56181.npy': '2465eac78b416d6e3e9b10b96182bc18cab607c29cd1f9e381d1640c0aa3bfa5',
    'pt/ka-binary-N64-T0.2_e60ecba4.npy':
        '7f45f4ac67d6e2bbd789bd35efc3634d913dc1058665ad123c7987b35db53201',
    'stan/seeds_f9a50f20.npy': '19c0c996178d02acf0c9e45460792f0f469da1e89a598ab54ed4119e3b0d501d',
    'exact/brownian_45794097.npy':
        '6aa333f731c918643f23b360cbe43458453fdd4db120a067f37de6de938a6f47',
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
        '4bd3a688ebe1044d0441779f64947396c74dcbab6011f209f7efafc4e7d925fb',
    'swapmc/ka-ternary-N44-T1.0_c0f9b3b8.npy':
        '4aa1a2acb7e8f8dc6b7e5f6633c72b36ce5b53fb5c052b8ab1dfda5b313d7eec',
    'train/funnel_556b9f8f_uniform-neck.npy':
        '608cb9d343f7dc09b47f40b48e387485bd4ebfa681a488a947f400e798dcea20',
    'train/manywell_30beb4d2_uniform-sides.npy':
        'c0f2d79f009babc1e2427326c8c22ef645b0451d08e90b909c815468428e23c0',
    'train/gmm40_a1f669cb_skewed-modes.npy':
        '902e77305819a6df4b02e7a60732091456993890c5f38ca2e7f50dea4b714f5f',
    'train/mos_3d819321_skewed-modes.npy':
        'abaf41a6588c755a2f8c157859fce2c8b0a2e8d9214a0b47fa5f7290c0d6c804',
    'train/twomodes_0eddf6c9_skewed-modes.npy':
        'b7bc9333930b86d772fe28c5ba616a37cc7611f04c554d98f885186a2403f432',
    'train/brackettwomodes_334746c8_skewed-modes.npy':
        'f54c5639478d89911835a9554865b608b0063c79d1e7b829a6ee333e578ae739',
    'train/manymodes-k3_edf2fcb2_skewed-modes.npy':
        'd8b76e7d2fb77766e93ae76d25d57b91bac09dd0351839e9580f9a9644cfdb3a',
    'train/eightgaussians_ae0110e3_skewed-modes.npy':
        'c7b19e7d2425fab5198a94af5376e5121037d86e75683adca407c5a184e2d66e',
    'train/rings_5deb01ff_skewed-modes.npy':
        '8331ce81170c341f3fe20c302ab56e8028e80f3de40ffee35bc4d8f32c04d079',
    'train/checkerboard_9547623f_skewed-modes.npy':
        'd5333a6241bcc8d4990ee84429b30b0ce2a0d481a55e75e8be769a5707688ddf',
    'train/dw4_d9994289_hot-mala.npy':
        '786afde848598c49543742ca85160e266f17fd38e530b118c00f626be80b841c',
    'train/lj13_bd3bc3a0_split-mala.npy':
        '84a14fbd8d830188d4cebaae33a60a46b35059c453f937d92a80b3f81431dce5',
    'train/lj55_01c56181_split-mala.npy':
        '436f818cf5954b2e7cf313d1154b1be1e63fc84b17f7c771421550ed1ea790a5',
    'train/seeds_f9a50f20_tempered-mala.npy':
        '7dd420a38455ca59752f11ebf6597829b9f979652c9e0e0af44af447c0a4527a',
    'train/brownian_45794097_tempered-mala.npy':
        '80e41cc8bd5951ffb0a8dcbd58ba49aeee9520bd389987b9c3ff74700606c9da',
    'train/lgcp_cea6d9a6_tempered-mala.npy':
        'd9631f245d06a9edc50f406bcb0f8b3ebe4463686534bc5148d7ccbc1944866a',
    'train/lgcp-whitened_30e98363_tempered-mala.npy':
        '1c693f538dbcd2f12faa2b2087dfb7af65336efd4e5f28b13d027090a328cb7c',
    'train/ka-ternary-N44-T1.0_c0f9b3b8_hot-basin-mc.npy':
        '402224ce3130cb212b66f266f0d5c2f98a5fa9236882e348f6c9437d5900c140',
    'train/ka-ternary-N44-T0.32_d50e8a3f_hot-basin-mc.npy':
        '4679fb695558a2628c78f970ee3baedb95d330b70fcc48b20a547c692ee1ae2a',
    'train/ka-binary-N64-T0.2_e60ecba4_hot-basin-mc.npy':
        'f02dd6baff3f1b165b1330072956a387b96c9361bc1f12672738f94fb77e720e',
    'train/ionosphere_d00109c3_tempered-mala.npy':
        'dcd863520b14de5f143577919a3ac95c92f2eadc8f3f0bfcea1b0544e7ed1a87',
    'train/sonar_52aa10ef_tempered-mala.npy':
        '51ab9e72398d68bdf11a902edea126a449b24edb063e8649a2e866dc8974d5e1',
    'train/german-credit_3cfdf175_tempered-mala.npy':
        '160f60f93c8558ea4b737b1b83c534465de0c388d740eea6a60cef790c35a443',
    'train/breast-cancer_543f5f94_tempered-mala.npy':
        '663061fd6a04e88d85dfc0a37927bf0e120556d51c7dea3ff88d05338c280992',
    'train/ace-a-nme-T300_54968005_basin-md.npy':
        '2164f5cf241f73513d0c594efa90b9ff2770c55fbe3be8996872832b32a0cd39',
    'train/ace-aaa-nme-T300_cda31701_basin-md.npy':
        '1a7c6d4887f3ebc9e5240690f24c025f98f2c5c95c3a3b2e13c4b6783a40905f',
    'train/aaa-T310_e0bfbee3_basin-md.npy':
        '0fc238170e4df165182af2d45c3eee3bcdc8555671d570ed84378bff1c6e9b3c',
    'train/aaaaaa-T310_08d2d916_basin-md.npy':
        '13d720b153f42d3ef8e57be48d0ca200f7d62c2aa09a5ccd15d6b8cfc97d7228',
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


def _digest(path, digest):
    """Streams a file through a hashlib object and returns its hex digest"""
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_once(path, checksum, source):
    """Checks a cached file against its sha256 once, leaving a stamp so later calls are free

    A published file can change between tags while its path does not, so a cache filled at an
    older tag would otherwise be read as if it were the current one. The stamp holds the checksum
    and counts only while it is newer than the file, so a file rewritten in place is checked again.

    Args:
        * path (pathlib.Path): The cached file
        * checksum (str): Its expected sha256
        * source (str): Where the expected file is published, quoted in the error

    Raises:
        * ValueError: When the file is not the one expected, naming how to fix it
    """
    stamp = path.with_name(path.name + '.sha256')
    if stamp.exists() and stamp.read_text() == checksum \
            and stamp.stat().st_mtime >= path.stat().st_mtime:
        return
    digest = sha256(path)
    if digest != checksum:
        raise ValueError(
            str(path) + ' is not the file published at ' + source + ' (sha256 ' + digest
            + ', expected ' + checksum + '). Delete it and call again to download the published one.'
        )
    stamp.write_text(checksum)


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

    # Write under a name of this process's own and rename, so an interrupted write never leaves a
    # truncated file under the name later calls read
    partial = path.with_name(path.name + '.partial' + str(os.getpid()))
    try:
        with open(partial, 'wb') as f:
            np.save(f, array)
        partial.rename(path)
    finally:
        partial.unlink(missing_ok=True)
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
        # Some servers answer 403 to urllib's default User-Agent, which would be indistinguishable
        # from a missing file. The body is streamed to disk, since a reference set runs to a
        # gigabyte, into a file of this process's own, so two jobs sharing a cache never write into
        # each other's download
        request = urllib.request.Request(url, headers={'User-Agent': 'samplingdb'})
        partial = path.with_suffix(path.suffix + '.partial' + str(os.getpid()))
        try:
            with urllib.request.urlopen(request) as response, open(partial, 'wb') as f:
                shutil.copyfileobj(response, f, 1 << 24)

            # A download that fails its checksum or stops midway is deleted, so the next call
            # fetches it again
            if checksum is not None and sha256(partial) != checksum:
                raise ValueError('sha256 mismatch for the download of ' + url + ': expected '
                                 + checksum + '. The file was deleted; try again.')
            partial.rename(path)
        finally:
            partial.unlink(missing_ok=True)
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
    """Returns the path of a generated set, downloading the published one on a cache miss

    A set in the cache that is not published yet is used as it is, so a freshly generated set is
    read before it is published. A published one is checked against REFERENCE_CHECKSUMS once, and
    a cache miss downloads it from REFERENCE_REPO at REFERENCE_REVISION. The cache is laid out like
    the dataset repo, so on a machine with no internet access
    `hf download <REFERENCE_REPO> --repo-type dataset --local-dir $SAMPLINGDB_CACHE`, run anywhere
    and copied over, fills it.

    Args:
        * folder (str): Cache sub-directory of the generator that writes it: 'pt', 'md', 'stan',
            'swapmc', 'exact', or 'train' for a train split
        * key (str): File stem, as Target.reference_key or Target.train_key returns it
        * hint (str): How to generate the set, quoted when it is neither cached nor published

    Returns:
        * path (pathlib.Path): Path to the .npy file

    Raises:
        * FileNotFoundError: When the set is neither in the cache nor published at
            REFERENCE_REVISION
    """
    # A set in the cache is used as it is, once checked against the published one when it is one
    name = folder + '/' + key + '.npy'
    path = cache_dir() / name
    if path.exists():
        if name in REFERENCE_CHECKSUMS:
            verify_once(path, REFERENCE_CHECKSUMS[name],
                        REFERENCE_REPO + ' revision ' + REFERENCE_REVISION)
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
    path = fetch_url(REFERENCE_URL + name, key + '.npy', checksum=REFERENCE_CHECKSUMS[name],
                     folder=folder)
    path.with_name(path.name + '.sha256').write_text(REFERENCE_CHECKSUMS[name])
    return path


def fetch_hf(repo_id, path_in_repo, revision, checksum, repo_type='dataset'):
    """Downloads one file of a Hugging Face repo at a pinned revision into the cache, once

    The peptide structures fetched this way DEFINE their densities -- the PDB fixes the topology and
    the atom order the force field is built on -- so both the revision and the file's sha256 are
    pinned, and a file that does not match raises instead of being read.

    NOTE: huggingface_hub is an optional dependency of the [molecular] extra, since the peptide
    tier is the only thing that fetches from the hub, so the import is lazy.

    Args:
        * repo_id (str): Repo id, e.g. 'transferable-samplers/sequential-boltzmann-generators-data'
        * path_in_repo (str): File path inside the repo
        * revision (str): Commit to read the file at
        * checksum (str): Expected sha256 of the file
        * repo_type (str): 'dataset' or 'model' (default is 'dataset')

    Returns:
        * path (pathlib.Path): Path to the downloaded (or locally provided) file

    Raises:
        * ValueError: When the file does not match its checksum
    """
    local_dir = os.environ.get('SAMPLINGDB_DATA_DIR')
    if local_dir is not None and (pathlib.Path(local_dir) / path_in_repo).exists():
        path = pathlib.Path(local_dir) / path_in_repo
    else:
        from huggingface_hub import hf_hub_download
        path = pathlib.Path(hf_hub_download(repo_id=repo_id, repo_type=repo_type,
                                            filename=path_in_repo, revision=revision,
                                            cache_dir=str(cache_dir())))
    verify_once(path, checksum, repo_id + ' revision ' + revision)
    return path
