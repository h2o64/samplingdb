# The train splits must cover what the test splits cover, get the rest wrong, and be easy to extend

# Libraries
import math
import numpy as np
import pytest
import torch
from samplingdb import build, metrics
from samplingdb.data import REFERENCE_CHECKSUMS, sha256
from samplingdb.targets.base import chain_layout, cycle, generator, interleave
from samplingdb.targets.kernels import temperature_ladder

# The targets with a mode descriptor, whose train split must visit every mode
MODE_TARGETS = ['gmm40', 'mos', 'twomodes', 'brackettwomodes', 'manymodes-k3', 'eightgaussians',
                'rings', 'checkerboard', 'manywell']

# The cheap targets with a generated reference set, whose train split must cover its energies
TEMPERED_TARGETS = ['dw4', 'lj13', 'seeds', 'brownian']


def covers(target, train, test, level=0.005):
    """Returns whether the train energies reach past the test's level and 1 - level quantiles"""
    energy, ref_energy = -target.log_prob(train).double(), -target.log_prob(test).double()
    low, high = torch.quantile(ref_energy, torch.tensor([level, 1.0 - level],
                                                        dtype=torch.float64, device=test.device))
    return float(energy.min()) <= float(low) and float(energy.max()) >= float(high)


def reference(target, n=20000):
    """Returns the target's test split, skipping when it is neither cached nor downloadable"""
    try:
        return target.test_samples(n=n)
    except (FileNotFoundError, OSError) as error:
        pytest.skip(target.name + ' has no reference set here: ' + str(error)[:60])


def test_train_samples_are_cached_reproducible_and_seeded():
    """A second call reads the cache, the recipe reproduces it, and another seed is another split"""
    target = build('gmm40')
    first = target.train_samples(n=512)
    assert first.shape == (512, target.dim)
    assert torch.equal(first, target.train_samples(n=512))
    assert torch.equal(first, target.generate_train_samples(512))

    # Another seed is another split, under another cache entry
    other = target.train_samples(n=512, seed=1)
    assert not torch.equal(first, other)
    assert target.train_cache_key(512) != target.train_cache_key(512, seed=1)

    # The key names the recipe and the device type, since neither is in spec()
    key = target.train_cache_key(512)
    assert target.train_recipe in key and '_cpu_' in key


def test_the_default_split_is_the_published_one(monkeypatch, tmp_path):
    """The default size at seed 0 reads train/<train_key>.npy once registered; any other runs"""
    target = build('twomodes')
    path = tmp_path / 'train' / (target.train_key() + '.npy')
    path.parent.mkdir()
    np.save(path, np.zeros((target.default_n, target.dim)))
    monkeypatch.setenv('SAMPLINGDB_CACHE', str(tmp_path))
    monkeypatch.setitem(REFERENCE_CHECKSUMS, 'train/' + path.name, sha256(path))
    assert float(target.train_samples().abs().max()) == 0.0
    assert float(target.train_samples(n=16).abs().max()) > 0.0


@pytest.mark.parametrize('name', MODE_TARGETS)
def test_a_mode_train_split_visits_every_mode_in_the_wrong_proportions(name):
    """Every mode of the test split is in the train split, at a share far from its weight"""
    target = build(name)
    train = target.train_samples(n=20000)
    test = target.test_samples(n=20000)

    # Every state of every group the test split visits, the train split visits too
    labels, ref_labels = target.mode_labels(train), target.mode_labels(test)
    if labels.dim() == 1:
        labels, ref_labels = labels[:, None], ref_labels[:, None]
    for group in range(labels.shape[1]):
        assert set(ref_labels[:, group].tolist()) <= set(labels[:, group].tolist())

    # The proportions are wrong by an order of magnitude more than an exact draw's noise
    train_tv = target.compute_metrics(train, compute_standard_metrics=False)['tv_weights']
    exact_tv = target.compute_metrics(test, compute_standard_metrics=False)['tv_weights']
    assert train_tv > max(0.2, 10 * exact_tv)

    # Inside each mode the shape is right, so the energies span the test's, tails included
    assert covers(target, train, test)


def test_the_skewed_allocation_starves_the_heaviest_mode():
    """The shares run geometrically from 1 to train_skew, against the ranking by weight"""
    target = build('twomodes')
    share = torch.bincount(target.train_modes(20000, generator=generator(0)), minlength=2)
    share = share.double() / 20000
    assert abs(float(share[0]) - 1 / 11) < 0.01
    assert abs(float(share[1]) - 10 / 11) < 0.01


def test_the_funnel_train_split_covers_neck_and_wide_end_with_the_wrong_marginal():
    """x_1 uniform over four scales either side: every energy covered, the marginal far off"""
    target = build('funnel')
    train = target.train_samples(n=20000)
    test = target.test_samples(n=20000)
    assert covers(target, train, test)
    low, high = torch.quantile(test[:, 0], torch.tensor([0.001, 0.999], dtype=torch.float64))
    assert float(train[:, 0].min()) < float(low) and float(train[:, 0].max()) > float(high)
    assert float(train[:, 0].std()) > 1.5 * float(test[:, 0].std())

    # The analytic x_1 panel is where the wrong shape shows up most sharply
    floor = target.compute_metrics(test, compute_standard_metrics=False)['ks_x1']
    assert target.compute_metrics(train, compute_standard_metrics=False)['ks_x1'] > 10 * floor


@pytest.mark.parametrize('name', TEMPERED_TARGETS)
def test_a_tempered_train_split_covers_the_test_energies(name):
    """Chains either side of the target temperature reach past both ends of the test energies"""
    target = build(name)
    test = reference(target)
    train = target.train_samples(n=20000)
    assert train.shape == (20000, target.dim)
    assert torch.isfinite(target.log_prob(train)).all()
    assert covers(target, train, test)


@pytest.mark.parametrize('name', ['dw4', 'lj13', 'lj55'])
def test_a_particle_train_split_is_biased_where_the_panel_reads(name):
    """The temperature ladder widens the energies or the distances, and a W1 reads it"""
    target = build(name)
    test = reference(target, n=8192)
    train = target.train_samples(n=4096)
    floor = target.compute_metrics(test[:4096], ref_samples=test[4096:],
                                   compute_standard_metrics=False)
    biased = target.compute_metrics(train, ref_samples=test[4096:], compute_standard_metrics=False)
    ratios = [biased[key] / floor[key] for key in ['energy_w1', 'dist_w1']]
    assert max(ratios) > 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
@pytest.mark.parametrize('name', ['ka-binary-N64-T0.2', 'ace-a-nme-T300'])
def test_an_expensive_train_split_covers_the_test_energies(name):
    """The hardest glass and the smallest peptide, at the recipe's own settings on a GPU"""
    try:
        target = build(name).to('cuda')
    except (FileNotFoundError, ImportError, OSError) as error:
        pytest.skip(name + ' cannot be built here: ' + str(error)[:60])
    test = reference(target)
    train = target.generate_train_samples(4000)
    assert covers(target, train, test)


def test_a_split_continues_from_given_starting_points():
    """x0 replaces the recipe's starts, chain i starting from x0[i % len(x0)]"""
    target = build('rings')
    x = target.train_samples(n=64)

    # With no steps the chains stay where they were put, and the starts cycle
    assert torch.equal(target.generate_train_samples(64, x0=x, n_steps=0), x)
    assert torch.equal(target.generate_train_samples(8, x0=x[:4], n_steps=0), cycle(x[:4], 8))

    # A continued split moves on from where the first one ended, and is still a train split
    more = target.generate_train_samples(64, x0=x, n_steps=50, seed=1)
    assert not torch.equal(more, x)
    assert torch.isfinite(target.log_prob(more)).all()


def test_a_tempered_split_is_laid_out_frame_by_frame():
    """n_chains chains keep ceil(n / n_chains) states each, and every prefix holds every chain"""
    assert chain_layout(100000, 1000) == (1000, 100)
    assert chain_layout(10, 1000) == (10, 1)
    assert chain_layout(1001, 1000) == (1000, 2)

    # Frame-major: the first n_chains rows are one state of every chain
    trace = torch.arange(3 * 4).reshape(3, 4, 1)
    assert interleave(trace, 6)[:, 0].tolist() == [0, 1, 2, 3, 4, 5]

    # The ladder is geometric in the chain index, from the lowest factor to the highest
    factors = temperature_ladder(5, (0.8, 1.25))
    assert abs(float(factors[0]) - 0.8) < 1e-12 and abs(float(factors[-1]) - 1.25) < 1e-12
    assert torch.allclose(factors[1:] / factors[:-1], torch.full((4,), (1.25 / 0.8) ** 0.25,
                                                                  dtype=torch.float64))

    # And a particle split from 8 chains of 2 states puts all 8 chains first
    target = build('dw4')
    x = target.generate_train_samples(16, n_chains=8, n_steps=20, thin=1)
    assert x.shape == (16, target.dim)
    centres = x.reshape(16, target.n_particles, target.spatial_dim).mean(1)
    assert float(centres.abs().max()) < 1e-5


def test_train_and_test_splits_disagree_on_the_universal_metrics():
    """The whole point of the pair: scoring train against test must look bad"""
    target = build('twomodes')
    train = target.train_samples(n=1024)
    test = target.test_samples(n=1024, seed=7)
    exact = target.test_samples(n=1024, seed=8)
    assert metrics.compute_wasserstein_distance(test, train) > \
        1.8 * metrics.compute_wasserstein_distance(test, exact)
    assert metrics.compute_mmd(test, train) > metrics.compute_mmd(test, exact)
    assert metrics.compute_sliced_ks(test, train, generator=generator(0)) > \
        metrics.compute_sliced_ks(test, exact, generator=generator(0))


def test_the_manywell_train_split_puts_half_of_every_well_on_the_light_side():
    """Each chain draws p uniformly and every well's side from Bernoulli(p)"""
    target = build('manywell', dim=8)
    x = target.train_samples(n=20000)
    left = (x[:, 0::2] < 0).double()
    assert torch.allclose(left.mean(0), torch.full((4,), 0.5, dtype=torch.float64), atol=0.02)

    # The number of left wells is uniform over 0..n_wells, not binomial
    count = torch.bincount(left.sum(-1).long(), minlength=5).double() / len(x)
    assert torch.allclose(count, torch.full((5,), 0.2, dtype=torch.float64), atol=0.02)

    # The chains sit in the wells they started in, at the minima's local scale
    near = ((x[:, 0::2].abs() - 1.73).abs() < 1.0).double().mean()
    assert float(near) > 0.998
    assert math.isclose(target.well_minima[0], -1.7108, abs_tol=1e-3)
