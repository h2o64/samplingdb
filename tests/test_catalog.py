# Invariants every target in the catalog must satisfy, whatever tier it belongs to

# Libraries
import inspect, re
import numpy as np
import pytest
import torch
from samplingdb import CATALOG, LJ55, Target, build, catalog
from samplingdb.targets.base import generator, spec_id

# A physically sensible evaluation scale per target, for the ones that live on a Gaussian blob
# rather than on a configuration. None means the target supplies its own starting structure.
SCALES = {
    'funnel': 1.0, 'manywell': 1.5, 'gmm40': 20.0, 'mos': 5.0, 'twomodes': 1.0,
    'brackettwomodes': 1.0, 'manymodes': 2.0, 'eightgaussians': 2.0, 'rings': 3.0,
    'checkerboard': None, 'dw4': 3.0, 'lj13': 1.0, 'lj55': 1.0, 'seeds': 0.3, 'brownian': 0.3,
    'lgcp': 0.3, 'lgcp-whitened': 0.3, 'ionosphere': 0.3, 'sonar': 0.3,
    'german-credit': 0.3, 'breast-cancer': 0.3,
}


def target(name, **kwargs):
    """Builds one catalogued target, skipping when whatever it needs is not cached"""
    try:
        return build(name, **kwargs)
    except (FileNotFoundError, ImportError, OSError) as error:
        pytest.skip(name + ' cannot be built here: ' + str(error)[:60])


def catalogued(data=None):
    """Returns every catalogued name as a test parameter, cheapest first"""
    return [pytest.param(name, id=name) for name in catalog(data=data)]


def evaluation_points(name, built, n):
    """Returns n physically sensible points, without needing any generated reference data

    A target with no evaluation scale lives on configurations rather than on a Gaussian blob, so
    its points come from its own fixed starting structure. That keeps these invariants independent
    of whether a reference ensemble has been generated yet.
    """
    scale = SCALES.get(name)
    if scale is not None:
        return scale * torch.randn(n, built.dim, generator=generator(0), dtype=built.dtype)
    if built.truth_tier == 0:
        return built.test_samples(n=n)
    if hasattr(built, 'lattice_configuration'):
        return built.lattice_configuration(n, jitter=0.05,
                                           generator=generator(0)).reshape(n, built.dim)
    if hasattr(built, 'reference_positions'):
        start = torch.as_tensor(built.reference_positions.reshape(1, -1), dtype=built.dtype)
    else:
        start = built.initial_configuration().reshape(1, -1)
    return start + 0.02 * torch.randn(n, built.dim, generator=generator(0), dtype=built.dtype)


def uncompiled(name):
    """Peptides compile their potential by default, which costs ~15 s per batch shape"""
    return {'compile': False} if CATALOG[name]['cls'].__name__ == 'Peptide' else {}


def test_the_catalog_is_the_one_list_of_targets():
    """Everything that enumerates the corpus reads CATALOG, so it has to be complete and clean"""
    assert len(CATALOG) >= 25
    for name, entry in CATALOG.items():
        assert re.fullmatch(r'[a-z0-9][A-Za-z0-9.-]*', name), name
        assert issubclass(entry['cls'], Target), name
        assert isinstance(entry['kwargs'], dict) and isinstance(entry['data'], bool), name

    # The two halves partition it, so a test can iterate the offline one without a network
    assert set(catalog(data=False)) | set(catalog(data=True)) == set(CATALOG)
    assert not set(catalog(data=False)) & set(catalog(data=True))

    # And an unknown name says so rather than failing obscurely
    with pytest.raises(AssertionError, match='not in the catalog'):
        build('no-such-target')


@pytest.mark.parametrize('name', catalogued())
def test_every_target_declares_its_metadata(name):
    """A short name, a tier and a working dtype are all a target declares about itself

    The dtype is NOT uniform across the corpus as of 2026-08-25, and the split is deliberate. The
    targets whose cost is an O(N^2) force field -- the peptides and the particle clusters -- work in
    float32, where they agree with float64 to ~1e-4 kT on real configurations against a timestep
    bias of 0.08 kT, and run 2x to 11x faster. Everything else stays float64: the toys, the
    posteriors and the mixtures have no cost pressure worth the trade, and they carry the corpus's
    EXACTNESS checks -- analytic score against autograd, the metric panel vanishing on identical
    samples -- which are not precision-tolerant. A blanket flip broke 47 of them.
    """
    built = target(name, **uncompiled(name))
    assert built.truth_tier in [0, 1, 2, 3, 4]
    assert built.dtype in [torch.float32, torch.float64]
    assert built.data_shape == (built.dim,)

    # The name is a cache-file component, so it stays short and path-safe. Parameter tags that
    # change the density are allowed in it (N44, T0.32); version suffixes are not
    assert built.name and isinstance(built.name, str)
    assert len(built.name) <= 24, built.name
    assert all(character.isalnum() or character in '-.' for character in built.name), built.name
    assert not re.search(r'-v\d+$', built.name), built.name


@pytest.mark.parametrize('name', catalogued())
def test_log_prob_and_grad_agrees_with_its_parts(name):
    """A sampler calling the paired method must get exactly what the separate calls give"""
    built = target(name, **uncompiled(name))
    x = evaluation_points(name, built, 4)
    log_prob, score = built.log_prob_and_grad(x)
    assert torch.allclose(log_prob, built.log_prob(x))
    assert torch.allclose(score, built.score(x))

    # And the score must not require grad-enabled inputs
    with torch.no_grad():
        assert torch.isfinite(built.score(x)).all()


def shrink_batches(built, chunk_size=1):
    """Forces a target to evaluate at most chunk_size configurations per call"""
    built.max_elements = chunk_size
    if hasattr(built, 'bridge'):
        built.bridge.chunk_size = chunk_size


@pytest.mark.parametrize('name', catalogued())
def test_log_prob_and_score_take_any_batch_size(name):
    """A sampler hands in as many walkers as it has, so chunking must be exact and invisible

    Every density here is evaluated pointwise, so splitting a batch can only change the answer
    through a bug -- a chunk boundary landing inside a reduction, a shape derived from len(x), a
    buffer sized once. Forcing one configuration per call and comparing against the whole batch
    catches all three, and an odd batch size makes sure the last chunk is a partial one.

    float64, although the peptides default to float32: this asserts EXACTNESS, and in single
    precision a chunk boundary legitimately reorders the summation, so the answer genuinely moves at
    the 1e-6 level. Loosening the tolerance to accommodate that would retire the check -- a chunk
    boundary landing inside a reduction would then hide inside the slack. float32 is validated
    against float64 separately, at its own tolerance.
    """
    built = target(name, **{**uncompiled(name), 'dtype': torch.float64})
    x = evaluation_points(name, built, 7)
    expected_log_prob, expected_score = built.log_prob(x), built.score(x)

    shrink_batches(built)
    assert torch.allclose(built.log_prob(x), expected_log_prob)
    assert torch.allclose(built.score(x), expected_score)
    log_prob, score = built.log_prob_and_grad(x)
    assert torch.allclose(log_prob, expected_log_prob)
    assert torch.allclose(score, expected_score)

    # A single point and a single point in a batch of one agree too
    assert torch.allclose(built.log_prob(x[3:4]), expected_log_prob[3:4])


@pytest.mark.parametrize('name', catalogued())
def test_every_feature_takes_any_batch_size(name):
    """The metric path is chunked like the density path, and just as exactly

    distances(), preprocess() and the reductions built on them see whole reference sets, which is
    where the temporaries are largest -- an all-pairs block is n_atoms^2 * 3 per configuration.
    """
    built = target(name, **uncompiled(name))
    x = evaluation_points(name, built, 7)
    features = {name: getattr(built, name)(x) for name in ['preprocess', 'distances']
                if hasattr(built, name)}

    shrink_batches(built, chunk_size=2)
    for feature, expected in features.items():
        assert torch.equal(getattr(built, feature)(x), expected), feature


@pytest.mark.parametrize('name', catalogued())
def test_ground_truth_is_either_usable_or_says_what_is_missing(name):
    """A reference set loads at the right shape, or raises naming the command that makes it"""
    built = target(name, **uncompiled(name))
    try:
        reference = built.test_samples(n=8)
    except NotImplementedError as error:
        assert built.has_ground_truth is False
        assert 'no reference ensemble' in str(error) or 'no ground-truth' in str(error)
        return
    except FileNotFoundError as error:
        assert 'generate' in str(error) or 'SAMPLINGDB_DATA_DIR' in str(error), \
            'a missing-data error must name the command or the variable that fixes it'
        return
    assert reference.shape == (8, built.dim)
    assert torch.isfinite(built.log_prob(reference)).all()


def test_no_two_targets_share_a_name_or_a_train_cache_key():
    """name versions the density and the cache key versions the split; both must be unique"""
    names, cache_keys = {}, {}
    for name in catalog():
        try:
            built = build(name, **uncompiled(name))
        except Exception:
            continue
        assert built.name not in names, built.name + ' is shared by ' + names.get(built.name, '?')
        names[built.name] = name
        key = built.train_cache_key(8)
        assert key not in cache_keys, key + ' is shared by ' + cache_keys.get(key, '?')
        cache_keys[key] = name
    assert len(names) >= 13


@pytest.mark.parametrize('name', catalogued())
def test_the_working_dtype_does_not_change_the_reference_set(name):
    """A target built in float64 reads the same reference set as the float32 default

    The dtype is how the density is evaluated, not which density it is, so it must not move
    spec_id. The glass targets once hashed their pair constants at the working precision, and a
    float64 build looked for a reference set nobody had generated.
    """
    if 'dtype' not in inspect.signature(CATALOG[name]['cls'].__init__).parameters:
        return
    built = target(name, **uncompiled(name))
    other = target(name, dtype=torch.float64, **uncompiled(name))
    assert spec_id(built.spec()) == spec_id(other.spec())


# Two settings of one class that are two different densities under one name. Every entry differs
# from the catalogued default in something log_gamma depends on, which is exactly what the name
# does not carry and what train_cache_key therefore has to.
VARIANTS = [
    ('gmm40', {'dim': 50}), ('gmm40', {'scale': 2.0}),
    ('mos', {'dim': 50}), ('mos', {'df': 3.0}),
    ('funnel', {'dim': 20}), ('funnel', {'scale': 1.0}),
    ('manywell', {'dim': 8}), ('manywell', {'b': -5.0}),
    ('twomodes', {'dim': 5}), ('twomodes', {'a': 3.0}),
    ('brackettwomodes', {'var_max': 0.5}), ('brackettwomodes', {'equilibrated': True}),
    ('manymodes', {'var': 0.3}), ('manymodes', {'mixture_weight_factor': 10.0}),
    ('eightgaussians', {'radius': 5.0}),
    ('checkerboard', {'width': 8}),
    ('rings', {'n_rings': 5}), ('rings', {'scale': 0.3}),
    ('ionosphere', {'sigma_w': 3.0}),
    ('lgcp', {'signal_variance': 1.0}), ('lgcp', {'beta': 0.1}),
    ('dw4', None), ('lj13', None),
]


@pytest.mark.parametrize('name,changed', VARIANTS,
                         ids=[name + '-' + '-'.join(changed or ['identity'])
                              for name, changed in VARIANTS])
def test_a_density_change_changes_the_train_cache_key(name, changed):
    """Two parameterisations of one class never read each other's cached train split

    The name alone does not version a density -- GMM-40 at dim 2 and at dim 50 are two targets
    under the name 'gmm40' -- so train_cache_key hashes spec() instead. Before that, dim-50 GMM-40
    read back the dim-2 split and returned an array of the wrong shape, silently, forever, because
    the cache is on disk.
    """
    built = target(name)
    if changed is None:
        # A class with no density knobs still has to key on its own spec, not on the name alone
        assert spec_id(built.spec()) in built.train_cache_key(8)
        return

    other = target(name, **changed)
    assert spec_id(built.spec()) != spec_id(other.spec())
    assert built.train_cache_key(8) != other.train_cache_key(8)

    # And the split each one builds is its own, at its own shape
    assert built.train_samples(n=4).shape == (4, built.dim)
    assert other.train_samples(n=4).shape == (4, other.dim)


# kl_weights is KL(true || empirical), which is genuinely +inf when a mode gets no mass at all --
# that is the metric working, not failing, and both DREEM and islips behave the same way. Verified
# it is not a weighting artefact: at n = 64 seven of GMM-40's modes are empty and both the weighted
# and unweighted panels read inf; by n = 256 none are empty and both are finite.
MAY_BE_INFINITE = ('kl_weights',)


@pytest.mark.parametrize('name', catalogued())
def test_every_panel_accepts_weights(name):
    """Uniform weights must give the unweighted number, on every metric of every target"""
    built = target(name, **uncompiled(name))
    x = evaluation_points(name, built, 64)
    uniform = torch.full((64,), 1.0 / 64, dtype=built.dtype)

    unweighted = built.compute_metrics(x, ref_samples=x, compute_standard_metrics=False)
    weighted = built.compute_metrics(x, weights=uniform, ref_samples=x,
                                     compute_standard_metrics=False)
    assert set(unweighted) == set(weighted)
    for metric, value in unweighted.items():
        if metric in MAY_BE_INFINITE and not np.isfinite(value):
            assert not np.isfinite(weighted[metric]), metric
            continue
        assert abs(weighted[metric] - value) < 1e-9 * (1 + abs(value)), metric


@pytest.mark.parametrize('name', catalogued(data=False))
def test_a_target_moves_to_a_device_and_a_dtype(name):
    """.to() has to move every buffer, and float32 has to work where float64 does

    _apply rebuilds the inner torch.distributions object, which is the piece a plain buffer move
    would leave pointing at the old tensors. There is no GPU in CI, so the device half is checked
    against the CPU round trip and skipped on the accelerator when there is none.
    """
    built = target(name, **uncompiled(name))
    x = evaluation_points(name, built, 4)
    expected = built.log_prob(x)

    moved = built.to('cpu')
    assert moved.device.type == 'cpu'
    assert torch.allclose(moved.log_prob(x), expected)

    if torch.cuda.is_available():
        on_gpu = built.to('cuda')
        assert on_gpu.device.type == 'cuda'
        assert torch.allclose(on_gpu.log_prob(x.cuda()).cpu(), expected)
        built.to('cpu')

    # float32 is a documented knob: the density still evaluates, to single precision
    single = target(name, dtype=torch.float32, **uncompiled(name))
    assert single.dtype == torch.float32
    single_log_prob = single.log_prob(x.to(torch.float32))
    assert single_log_prob.dtype == torch.float32
    assert torch.isfinite(single_log_prob).all()


@pytest.mark.parametrize('name', catalogued())
def test_the_standard_panel_is_the_synthetic_tier(name):
    """The raw-coordinate panel runs exactly where the coordinates are directly comparable"""
    built = target(name, **uncompiled(name))
    assert built.standard_panel == (built.truth_tier == 0), name


def test_chunking_bounds_the_temporaries_of_a_large_batch():
    """LJ-55's score holds n x 55 x 55 x 3 doubles, which is 7 GB at n = 10^5 unless it is split"""
    built = LJ55()
    x = built.initial_configuration()[None] + 0.1 * torch.randn(
        64, built.dim, generator=generator(0), dtype=torch.float64)
    expected = built.score(x)

    # The budget decides the chunk size, and the whole point is that it is respected
    built.max_elements = 8 * built.n_particles ** 2 * built.spatial_dim
    sizes = []
    inner = built.score_chunk
    built.score_chunk = lambda chunk: (sizes.append(len(chunk)), inner(chunk))[1]
    assert torch.allclose(built.score(x), expected)
    assert max(sizes) == 8
    assert max(sizes) * built.n_particles ** 2 * built.spatial_dim <= built.max_elements
