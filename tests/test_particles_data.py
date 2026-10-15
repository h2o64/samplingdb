# The published Tier 1 splits, and the pair conventions they pin

# Libraries
import numpy as np
import pytest
import torch
from samplingdb import DW4, LJ13, LJ55
from samplingdb.data import cache_dir, fetch_url
from samplingdb.targets.base import generator
from samplingdb.targets.kernels import random_walk_mh
from nrpt import hmc

# The splits Klein et al. 2023 regenerated, which iDEM adopted wholesale and DEM's own configs label
# 'EFM', published in jarridrb/DEM (MIT) and pinned to a commit so the URL cannot change under us.
# They are not a reference set of this library: they are what settled the pair conventions.
DEM_COMMIT = '8e9531987b0bfe4e770d5009c10a56b8434c0f0a'
DEM_URL = 'https://raw.githubusercontent.com/jarridrb/DEM/' + DEM_COMMIT + '/data/'
DEM_CHECKSUMS = {
    'test_split_DW4.npy': 'ef4466793b3502b2fe405aad22d3ae43b34827971594c5bab28684771d828244',
    'train_split_DW4.npy': '03e4cd085b52bf021cebc7f5f6d05aec0cdccff890a476b36595c2e343b1c58a',
    'val_split_DW4.npy': '05683af2442aea8c358d35482a1c6877fc53796e46c9f5335fa92aa4f3670de6',
    'test_split_LJ13-1000.npy': 'c5064a56282e96441f1ff2f1babb13c241a99cd1077aee3667951d06055a785b',
    'train_split_LJ13-1000.npy': 'd174a52b1b1660abebfa3b17e7a74c8aa45e3d1b32ad2fbc1c87d10b7705ca44',
    'val_split_LJ13-1000.npy': '77a051cd50b16bec41e8fa056f6f77e21804d530a7461369f3a3ee985a576eb7',
    'test_split_LJ55-1000-part1.npy':
        '29d4a49dc4e7cd81943380e1662a15cc45c0382bc107c2bd1cb66b5111855a9a',
    'train_split_LJ55-1000-part1.npy':
        'bed805590e67b181450b3aab976d2fd1d879304352917d86cc76a97fa4c6a36a',
    'val_split_LJ55-1000-part1.npy':
        'cdb04155778ba5a84eb33d2b5ccbcd0fd4491ce6287ca7d2bf14089cdca7d295',
}
DEM_NAMES = {'dw4': 'DW4', 'lj13': 'LJ13-1000', 'lj55': 'LJ55-1000-part1'}

# The splits are a download, so the tests that need them run only once they are cached
TARGETS = [DW4, LJ13, LJ55]


def split_name(target, split='test'):
    """Returns the file name of one published split"""
    return split + '_split_' + DEM_NAMES[target.name] + '.npy'


def has_split(target, split='test'):
    """Returns whether one published split is already in the cache"""
    return (cache_dir() / 'dem-efm' / split_name(target, split)).exists()


def needs(target, split='test'):
    """Skips unless the given split is cached"""
    if not has_split(target, split):
        pytest.skip(DEM_NAMES[target.name] + ' ' + split + ' split is not cached')


def load_split(target, split, n=None):
    """Reads one published split, checksum-verified and mean-free

    The arrays are centred on load, which is what dem does to them before they reach any model.
    The DW-4 files arrive with a centre of geometry up to ~16 while the LJ ones arrive already
    mean-free; the energy is translation-invariant, so this only matters for coordinate metrics.
    """
    name = split_name(target, split)
    path = fetch_url(DEM_URL + name, name, checksum=DEM_CHECKSUMS[name], folder='dem-efm')
    array = np.load(path, mmap_mode='r')
    if n is not None:
        array = array[:n]
    return target.preprocess(torch.as_tensor(np.array(array), dtype=target.dtype))


def test_the_source_is_pinned_to_a_commit():
    """A moving branch would let the reference data change under a fixed name"""
    assert len(DEM_COMMIT) == 40
    assert DEM_URL == ('https://raw.githubusercontent.com/jarridrb/DEM/' + DEM_COMMIT + '/data/')
    assert len(DEM_CHECKSUMS) == 9
    for name, digest in DEM_CHECKSUMS.items():
        assert len(digest) == 64, name
        assert name.split('_')[0] in ['train', 'val', 'test'], name


@pytest.mark.parametrize('target_cls', TARGETS)
def test_every_split_loads_mean_free_with_the_right_shape(target_cls):
    """Released sizes are 100 000 train and 10 000 each for val and test"""
    target = target_cls()
    needs(target)
    for split, expected in [('test', 10000), ('val', 10000)]:
        if not has_split(target, split):
            continue
        x = load_split(target, split)
        assert x.shape == (expected, target.dim)
        centred = x.reshape(len(x), target.n_particles, target.spatial_dim).mean(1)
        assert float(centred.abs().max()) < 1e-5

    # n truncates without reading the whole file
    assert load_split(target, 'test', n=32).shape == (32, target.dim)


@pytest.mark.parametrize('target_cls', TARGETS)
def test_the_reference_energies_sit_inside_dems_own_histogram_range(target_cls):
    """The first of two independent checks on the pair convention

    dem hardcodes an energy range per system for its own plots of these very files. Under the
    correct convention the reference energies land inside it: DW-4 at -22.8 in (-26, 0), LJ-13 at
    -42.8 in (-60, 0), LJ-55 at -325 in (-380, -180). Under the halved Lennard-Jones convention
    LJ-55 would sit at -121, entirely outside its own range.
    """
    target = target_cls()
    needs(target)
    energy = target.energy(load_split(target, 'test', n=256))
    low, high = target.energy_histogram_range
    assert float(((energy >= low) & (energy <= high)).double().mean()) > 0.99

    expected = {'DW4': -22.8, 'LJ13': -42.8, 'LJ55': -325.0}[type(target).__name__]
    assert abs(float(energy.mean()) - expected) < 3.0


@pytest.mark.parametrize('target_cls', TARGETS)
def test_the_released_ensemble_is_stationary_under_our_energy(target_cls):
    """The second, and the one that actually settled the conventions

    A Metropolis chain at the released step size, started from released frames, must leave the
    energy where it found it. Measured drift in units of the ensemble standard deviation: DW-4
    -0.13, LJ-13 +0.01, LJ-55 -0.03. The rejected alternatives drift far more: doubling DW-4's
    energy gives -0.75 sigma (too cold) and halving the Lennard-Jones one gives +3.27 sigma on
    LJ-13 and +2.16 on LJ-55 (too hot).
    """
    target = target_cls()
    needs(target)
    x = load_split(target, 'test', n=64)
    before = target.energy(x)
    out = random_walk_mh(target.log_prob, x.clone(), n_steps=3000,
                         step_size=target.train_step_size, generator=generator(0))
    drift = abs(float(target.energy(out).mean() - before.mean())) / float(before.std())
    assert drift < 0.5


@pytest.mark.parametrize('target_cls', TARGETS)
def test_halving_or_doubling_the_pair_sum_breaks_stationarity(target_cls):
    """The rejected convention must be visibly wrong, or the check above proves nothing"""
    target = target_cls()
    needs(target)
    x = load_split(target, 'test', n=64)
    before = target.energy(x)

    # The wrong convention for this system: doubling DW-4, halving Lennard-Jones
    factor = 2.0 if target.pair_multiplicity == 1 else 0.5
    wrong = lambda y: -(factor * target.pair_multiplicity
                        * target.pair_energy(target.distances(y)).sum(-1)
                        + (target.c_harm * (target.preprocess(y) ** 2).sum(-1)
                           if hasattr(target, 'c_harm') else 0.0))
    out = random_walk_mh(wrong, x.clone(), n_steps=3000, step_size=target.train_step_size,
                         generator=generator(0))
    drift = abs(float(target.energy(out).mean() - before.mean())) / float(before.std())
    assert drift > 0.5


@pytest.mark.parametrize('target_cls', TARGETS)
def test_the_train_split_is_the_biased_one_and_the_panel_says_so(target_cls):
    """Our host-generated train split must score worse than the released one it is compared to"""
    target = target_cls()
    needs(target)
    reference = load_split(target, 'test', n=512)
    train = target.train_samples(n=512)

    floor = target.compute_metrics(reference[:256], ref_samples=reference[256:],
                                   compute_standard_metrics=False)
    biased = target.compute_metrics(train, ref_samples=reference[:256],
                                    compute_standard_metrics=False)
    assert biased['dist_total_var'] > 2 * floor['dist_total_var']
    assert biased['energy_w1'] > floor['energy_w1']


def test_the_checksum_is_verified():
    """A corrupted or substituted file must fail loudly rather than be scored against"""
    target = DW4()
    needs(target)
    name = split_name(target)
    assert fetch_url(DEM_URL + name, name, checksum=DEM_CHECKSUMS[name], folder='dem-efm').exists()
    with pytest.raises(ValueError):
        fetch_url(DEM_URL + name, name, checksum='0' * 64, folder='dem-efm')


def has_reference(target):
    """Returns whether this library's own parallel-tempering reference set exists yet"""
    return target.reference_path().exists()


@pytest.mark.parametrize('target_cls', TARGETS)
def test_test_samples_is_our_own_reference_not_the_published_split(target_cls):
    """The published split is not what the panel scores against"""
    target = target_cls()
    if not has_reference(target):
        pytest.skip('the parallel-tempering reference for ' + target.name + ' is not cached')

    ours = target.test_samples()
    assert ours.shape[1] == target.dim
    assert len(ours) >= 100000, 'the reference set is meant to be 100k samples'
    assert torch.isfinite(target.log_prob(ours[:256])).all()

    # It is a different object from the published split
    needs(target)
    published = load_split(target, 'test', n=len(ours))
    assert not torch.equal(ours[:len(published)], published)


def test_our_reference_is_stationary_where_the_published_one_drifts():
    """The measurement behind test_samples' docstring, on DW-4

    Under the target's own HMC kernel the published set moves and the set generated here does not.
    Recorded numbers from 10 000 configurations of each, 200 HMC steps at step size 0.18:
    ours 4.0410 -> 4.0393 (-0.002), published 3.9700 -> 4.0374 (+0.067), and after the kernel
    the two agree to 0.002. So the published split is a slightly under-relaxed ensemble of
    the same density rather than a different one, and this library's own reference is the
    stationary one.
    """
    target = DW4()
    if not has_reference(target):
        pytest.skip('no parallel-tempering reference for dw4 yet')
    needs(target)
    ours = target.test_samples(n=10000)
    published = load_split(target, 'test', n=10000)

    gen = generator(0)
    moved_ours = hmc(target.log_prob, target.score, ours, n_steps=200, step_size=0.18,
                     n_leapfrog=10, generator=gen)
    moved_published = hmc(target.log_prob, target.score, published, n_steps=200, step_size=0.18,
                          n_leapfrog=10, generator=gen)

    def mean_distance(x):
        return float(target.distances(x).mean())

    # Ours starts where the kernel leaves it; theirs has to travel to get there
    assert abs(mean_distance(moved_ours) - mean_distance(ours)) < 0.02
    assert mean_distance(moved_published) - mean_distance(published) > 0.04

    # And they end up in the same place, which is what makes this a bias and not a different target
    assert abs(mean_distance(moved_published) - mean_distance(moved_ours)) < 0.02
