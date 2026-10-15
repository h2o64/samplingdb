# The host-generated train splits must be reproducible and biased in the documented way

# Libraries
import torch
from samplingdb import DW4, GMM40, LJ13, ManyModes, ManyWell, TwoModes, metrics
from samplingdb.targets import Funnel
from samplingdb.targets.base import generator


def test_train_samples_are_cached_and_reproducible():
    """A second call reads the cache, so the split is bit-identical within and across runs"""
    target = GMM40()
    first = target.train_samples(n=512)
    second = target.train_samples(n=512)
    assert first.shape == (512, target.dim)
    assert torch.equal(first, second)

    # Regenerating from the recipe gives the same numbers as the cache holds
    assert torch.equal(first, target.generate_train_samples(512))


def test_mixture_train_split_visits_every_mode_with_the_wrong_weights():
    """One chain per mode: every mode is found, with the right local shape and the wrong weight

    The chains are allocated by mode INDEX, so each mode gets the same share of the samples
    whatever its weight. On a mixture whose weights differ that is a large mode-proportion error;
    on one whose weights are equal it is not an error at all, which is why the two cases are
    asserted separately here (see the recipe's docstring).
    """
    target = TwoModes(dim=2)
    x = target.train_samples(n=2048)
    labels = target.mode_labels(x)
    assert len(labels.unique()) == target.n_modes
    assert float((x - target.means[labels]).std()) < 1.5

    # Nothing is forgotten, and the weights are two orders of magnitude further off than an
    # exact draw of the same size
    train = target.compute_metrics(x, compute_standard_metrics=False)
    exact = target.compute_metrics(target.test_samples(n=2048), compute_standard_metrics=False)
    assert train['perc_forgotten_modes'] == 0.0
    assert train['tv_weights'] > 10 * exact['tv_weights']

    # A uniform-weight mixture is the case where allocating chains per mode is CORRECT, so
    # GMM-40's train split has the right mode weights and is biased only by having no
    # mode-to-mode dynamics at all
    equal = GMM40()
    y = equal.train_samples(n=2048)
    assert len(equal.mode_labels(y).unique()) == equal.n_modes
    assert equal.compute_metrics(y, compute_standard_metrics=False)['tv_weights'] < \
        1.5 * equal.compute_metrics(equal.test_samples(n=2048),
                                    compute_standard_metrics=False)['tv_weights']


def test_manywell_train_split_stays_in_the_deep_well():
    """Chains stay in the deep well of every well, so they visit one of 2^(d/2) modes"""
    target = ManyWell(dim=8)
    x = target.train_samples(n=1024)
    assert float((x[:, 0::2] > 0).double().mean()) > 0.99
    assert len(torch.unique(target.mode_labels(x), dim=0)) <= 3

    # The exact split, by contrast, spreads over all 16 modes with the tilted weights
    exact = target.test_samples(n=4096)
    assert len(torch.unique(target.mode_labels(exact), dim=0)) == 16
    assert torch.allclose(target.compute_mode_weights(exact), target.mode_weights.expand(4, 2),
                          atol=0.02)


def test_funnel_train_split_undersamples_the_neck():
    """The unimodal-but-pathological case: right local shape, wrong tail calibration"""
    target = Funnel()
    train = target.train_samples(n=4096)
    test = target.test_samples(n=4096)
    assert float(train[:, 0].std()) < 0.5 * float(test[:, 0].std())
    assert float(train[:, 0].min()) > float(test[:, 0].min())

    # The analytic x_1 panel is where this shows up most sharply
    floor = target.compute_metrics(test, compute_standard_metrics=False)['ks_x1']
    assert floor < 0.03
    assert target.compute_metrics(train, compute_standard_metrics=False)['ks_x1'] > 5 * floor


def test_particle_train_split_is_one_chain_from_a_fixed_configuration():
    """The historical single-chain recipe: no burn-in discard, one fixed starting structure"""
    for target in [DW4(), LJ13()]:
        x = target.train_samples(n=256)
        assert x.shape == (256, target.dim)

        # The chain starts at the lattice configuration, so its first state is near it
        start = target.initial_configuration()
        assert float((x[0] - start).abs().max()) < 10 * target.train_step_size

        # It is a chain, not an i.i.d. sample: consecutive states are highly correlated
        assert float(((x[1:] - x[:-1]) ** 2).mean()) < float(((x - x.mean(0)) ** 2).mean())
        assert torch.isfinite(target.log_prob(x)).all()


def test_train_and_test_splits_disagree_on_the_universal_metrics():
    """The whole point of the pair: scoring train against test must look bad

    On a mixture with unequal weights, where one chain per mode is genuinely the wrong allocation.
    """
    target = TwoModes(dim=2)
    train = target.train_samples(n=1024)
    test = target.test_samples(n=1024, seed=7)
    exact = target.test_samples(n=1024, seed=8)
    assert metrics.compute_wasserstein_distance(test, train) > \
        1.8 * metrics.compute_wasserstein_distance(test, exact)
    assert metrics.compute_mmd(test, train) > metrics.compute_mmd(test, exact)
    assert metrics.compute_sliced_ks(test, train, generator=generator(0)) > \
        metrics.compute_sliced_ks(test, exact, generator=generator(0))


def test_two_modes_train_split_gets_the_mode_weights_wrong():
    """One chain per mode splits the mass 50/50 where the truth is 2/3 to 1/3"""
    target = TwoModes(dim=2)
    ret = target.compute_metrics(target.train_samples(n=2048), compute_standard_metrics=False)
    assert abs(ret['mode_weight'] - 50.0) < 5.0
    assert ret['mode_weight_abs_err'] > 10.0

    # The exact split gets them right, which is the contrast the pair exists for
    exact = target.compute_metrics(target.test_samples(n=2048), compute_standard_metrics=False)
    assert exact['mode_weight_abs_err'] < 3.0
