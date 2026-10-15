# Cache-dir resolution, the name-keyed file names, and the local data override

# Libraries
import os
import numpy as np
import pytest
import torch
from samplingdb import DW4, Seeds, data, download_references
from samplingdb.targets.base import spec_id
from manifest import manifest


def test_cache_dir_follows_the_environment(monkeypatch, tmp_path):
    """SAMPLINGDB_CACHE wins, and the directory is created on demand"""
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path / 'cache'))
    assert data.cache_dir() == tmp_path / 'cache'
    assert data.cache_dir().is_dir()


def test_local_cache_keys_embed_the_name_the_spec_and_the_size(monkeypatch, tmp_path):
    """A cache entry is keyed by the name, the density spec and n, so none can be confused

    The spec digest is what stops two parameterisations of one class -- which share a name -- from
    reading each other's split; see test_catalog.test_a_density_change_changes_the_train_cache_key.
    The cache is an empty one, so the split checked is the one this call wrote.
    """
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path))
    target = DW4()
    target.train_samples(n=8)
    key = target.train_cache_key(8)
    assert [path.name for path in tmp_path.glob('*.npy')] == [key + '.npy']
    assert key.startswith('train_dw4_' + spec_id(target.spec()) + '_') and '_n8' in key
    assert key != target.train_cache_key(16)


def test_local_data_dir_override(monkeypatch, tmp_path):
    """SAMPLINGDB_DATA_DIR serves a user's own copy of a file, with no download

    The override is by file name and wins whenever the named file exists, which is what makes an
    offline run possible. Two things it deliberately does NOT do: fall back silently to a wrong
    file, and bypass a pinned checksum. A target that pins one -- every Tier 1 split does -- will
    reject an override whose bytes differ, so pointing the variable at a copy of someone else's
    data fails loudly rather than scoring against it.
    """
    monkeypatch.setenv('SAMPLINGDB_DATA_DIR', str(tmp_path))
    reference = np.random.default_rng(0).normal(size=(16, 8))
    np.save(tmp_path / 'thing.npy', reference)
    path = tmp_path / 'thing.npy'

    assert data.fetch_url('unused://', 'thing.npy') == path
    assert data.fetch_url('unused://', 'thing.npy', checksum=data.sha256(path)) == path
    with pytest.raises(ValueError):
        data.fetch_url('unused://', 'thing.npy', checksum='0' * 64)


def test_every_fetch_helper_honours_the_override(monkeypatch, tmp_path):
    """One override for both sources, so an offline user sets one variable"""
    monkeypatch.setenv('SAMPLINGDB_DATA_DIR', str(tmp_path))
    (tmp_path / 'thing.npy').write_bytes(b'0')
    (tmp_path / 'folder').mkdir()
    (tmp_path / 'folder' / 'thing.npy').write_bytes(b'0')
    digest = data.sha256(tmp_path / 'thing.npy')
    assert data.fetch_url('unused://', 'thing.npy') == tmp_path / 'thing.npy'
    assert data.fetch_hf('unused/repo', 'folder/thing.npy', '0' * 40, digest) \
        == tmp_path / 'folder' / 'thing.npy'

    # A pinned file is checked even when it comes from the override
    with pytest.raises(ValueError, match='unused/repo'):
        data.fetch_hf('unused/repo', 'thing.npy', '0' * 40, '0' * 64)


def test_manifest_records_the_protocol():
    """Every generated reference set travels with its name, spec_id, protocol and checksums"""
    payload = manifest('dw4', 'd9994289', {'sampler': 'metropolis', 'step_size': 0.5},
                       {'test.npy': 'abc'})
    assert '"target": "dw4"' in payload
    assert '"spec_id": "d9994289"' in payload
    assert '"step_size": 0.5' in payload


def test_a_reference_set_refuses_more_rows_than_it_holds(monkeypatch, tmp_path):
    """Asking a generated set for more samples than it holds raises, naming what makes more

    Slicing past the end of an array returns the whole array, so without the check a sampler scored
    on 200 000 samples against a 100 000-row set would meet a reference half its size.
    """
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path))
    monkeypatch.setattr(data, 'REFERENCE_CHECKSUMS', {})
    target = DW4()
    (tmp_path / 'pt').mkdir()
    np.save(target.reference_path(), np.random.default_rng(0).normal(size=(16, target.dim)))
    assert len(target.test_samples()) == 16
    assert len(target.test_samples(n=16)) == 16
    with pytest.raises(ValueError, match='gen/parallel_tempering/generate.py --target dw4'):
        target.test_samples(n=17)


def publish(tmp_path, target, array):
    """Writes array where a dataset repo would serve target's reference set, returning its registry

    The repo is a local folder read through a file:// URL, which urllib fetches like any other, so
    the download path runs end to end without the network.
    """
    name = target.reference_folder + '/' + target.reference_key() + '.npy'
    (tmp_path / 'repo' / target.reference_folder).mkdir(parents=True)
    np.save(tmp_path / 'repo' / name, array)
    return (tmp_path / 'repo').as_uri() + '/', {name: data.sha256(tmp_path / 'repo' / name)}


def test_a_published_reference_set_is_downloaded_once_into_the_cache(monkeypatch, tmp_path):
    """A cache miss downloads the published set to the path the generators write, then reads it"""
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path / 'cache'))
    target = DW4()
    array = np.random.default_rng(0).normal(size=(16, target.dim)).astype(np.float32)
    url, checksums = publish(tmp_path, target, array)
    monkeypatch.setattr(data, 'REFERENCE_URL', url)
    monkeypatch.setattr(data, 'REFERENCE_CHECKSUMS', checksums)

    # The first call downloads, into the generators' own layout
    x = target.test_samples(n=16)
    assert target.reference_path() == tmp_path / 'cache' / 'pt' / (target.reference_key() + '.npy')
    assert target.reference_path().exists()
    assert np.array_equal(np.load(target.reference_path()), array)

    # Later calls read the cache, so the repo can go away
    (tmp_path / 'repo' / list(checksums)[0]).unlink()
    assert torch.equal(target.test_samples(n=16), x)

    # download_references returns the same file, by catalog name
    assert download_references(['dw4']) == {list(checksums)[0]: target.reference_path()}


def test_a_download_that_fails_its_checksum_is_not_kept(monkeypatch, tmp_path):
    """A corrupted download raises and leaves nothing behind for the next call to read"""
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path / 'cache'))
    target = DW4()
    url, checksums = publish(tmp_path, target, np.zeros((4, target.dim), dtype=np.float32))
    monkeypatch.setattr(data, 'REFERENCE_URL', url)
    monkeypatch.setattr(data, 'REFERENCE_CHECKSUMS', {name: '0' * 64 for name in checksums})
    with pytest.raises(ValueError, match='sha256 mismatch'):
        target.test_samples()
    assert list((tmp_path / 'cache' / 'pt').iterdir()) == []


def test_a_download_that_stops_midway_is_not_kept(monkeypatch, tmp_path):
    """An interrupted download raises and leaves no partial file in the cache"""
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path / 'cache'))
    target = DW4()
    url, checksums = publish(tmp_path, target, np.zeros((4, target.dim), dtype=np.float32))
    monkeypatch.setattr(data, 'REFERENCE_URL', url)
    monkeypatch.setattr(data, 'REFERENCE_CHECKSUMS', checksums)

    # The connection drops after the first bytes
    def interrupted(source, destination, length):
        destination.write(source.read(16))
        raise ConnectionResetError('the connection dropped')

    monkeypatch.setattr(data.shutil, 'copyfileobj', interrupted)
    with pytest.raises(ConnectionResetError):
        target.test_samples()
    assert list((tmp_path / 'cache' / 'pt').iterdir()) == []


def test_an_interrupted_cache_write_leaves_nothing_to_read(monkeypatch, tmp_path):
    """A host-generated array is written under another name and renamed, so a write that stops
    midway leaves no truncated file for the next call to read as the array"""
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path))
    array = np.arange(12.0).reshape(3, 4)
    save = np.save

    # The disk fills after the header
    def interrupted(file, values):
        file.write(b'\x93NUMPY')
        raise OSError('no space left on device')

    monkeypatch.setattr(data.np, 'save', interrupted)
    with pytest.raises(OSError):
        data.cached_local('train_thing', lambda: array)
    assert list(tmp_path.iterdir()) == []

    # The next call builds the array again and keeps it
    monkeypatch.setattr(data.np, 'save', save)
    assert np.array_equal(data.cached_local('train_thing', lambda: array), array)
    assert np.array_equal(data.cached_local('train_thing', lambda: None), array)
    assert [path.name for path in tmp_path.iterdir()] == ['train_thing.npy']


def test_an_unpublished_reference_set_names_the_command_that_generates_it(monkeypatch, tmp_path):
    """Neither cached nor published, test_samples says where it looked and how to make the set"""
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path))
    monkeypatch.setattr(data, 'REFERENCE_CHECKSUMS', {})
    target = Seeds()
    with pytest.raises(FileNotFoundError, match='gen/stan/generate.py') as error:
        target.test_samples()
    assert data.REFERENCE_REPO in str(error.value)
    assert str(target.reference_path()) in str(error.value)


def test_a_cached_set_from_another_release_is_refused(monkeypatch, tmp_path):
    """A published set keeps its path across releases, so the cached bytes are checked, once"""
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path))
    target = DW4()
    (tmp_path / 'pt').mkdir()
    np.save(target.reference_path(), np.zeros((16, target.dim), dtype=np.float32))
    name = 'pt/' + target.reference_key() + '.npy'

    # Another release's bytes raise, naming the release they should have come from
    monkeypatch.setattr(data, 'REFERENCE_CHECKSUMS', {name: '0' * 64})
    with pytest.raises(ValueError, match=data.REFERENCE_REVISION):
        target.test_samples()

    # The right bytes pass and leave a stamp, and a file rewritten after it is checked again
    monkeypatch.setattr(data, 'REFERENCE_CHECKSUMS', {name: data.sha256(target.reference_path())})
    assert len(target.test_samples()) == 16
    stamp = target.reference_path().with_name(target.reference_path().name + '.sha256')
    assert stamp.read_text() == data.REFERENCE_CHECKSUMS[name]
    np.save(target.reference_path(), np.ones((16, target.dim), dtype=np.float32))
    os.utime(stamp, (0, 0))
    with pytest.raises(ValueError):
        target.test_samples()
