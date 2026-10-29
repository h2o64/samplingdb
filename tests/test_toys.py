# The 2D toys: the rings' polar density, the checkerboard's flat one, and eight Gaussians

# Libraries
import math
import torch
from samplingdb import Checkerboard, EightGaussians, Rings
from samplingdb.targets import score_autograd


def test_rings_density_is_the_polar_mixture():
    """log gamma = log sum_k w_k N(|x|; r_k, scale^2) - log(2 pi) - log|x|, Jacobian included"""
    target = Rings()
    assert target.dim == 2 and target.n_modes == 3
    assert torch.allclose(target.radii, torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64))

    # The weights are proportional to the radius, so the outer ring carries the most mass
    assert torch.allclose(target.mode_weights,
                          torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64) / 9.0)
    assert float(target.mode_weights[-1]) > float(target.mode_weights[0])

    # The density is rotationally invariant and matches the closed form at one point
    x = torch.tensor([[3.0, 0.0], [0.0, -3.0], [3.0 / math.sqrt(2), 3.0 / math.sqrt(2)]],
                     dtype=torch.float64)
    log_prob = target.log_prob(x)
    assert torch.allclose(log_prob, log_prob[0].expand(3))

    variance = target.scale ** 2
    component = -0.5 * (3.0 - target.radii) ** 2 / variance \
        - 0.5 * math.log(2 * math.pi * variance)
    expected = torch.logsumexp(component + torch.log(target.mode_weights), dim=0) \
        - math.log(2 * math.pi) - math.log(3.0)
    assert abs(float(log_prob[0]) - float(expected)) < 1e-12

    # Equal weights are a different target, and say so in the name
    assert Rings(equilibrated=True).name != target.name
    assert torch.allclose(Rings(equilibrated=True).mode_weights,
                          torch.full((3,), 1 / 3, dtype=torch.float64))


def test_rings_score_is_analytic_and_exact_samples_land_on_the_rings():
    """The radial derivative projected onto x/|x|, against autograd"""
    target = Rings()
    x = target.test_samples(n=512)
    assert torch.allclose(target.score(x), score_autograd(target, x), rtol=1e-9, atol=1e-9)

    # Exact draws sit on a ring, within a few widths
    radius = torch.linalg.norm(x, dim=-1)
    nearest = target.radii[target.mode_labels(x)]
    assert float((radius - nearest).abs().max()) < 6 * target.scale

    # And their ring populations match the true weights
    hist = target.compute_mode_weights(target.test_samples(n=20000))[0]
    assert torch.allclose(hist, target.mode_weights, atol=0.02)


def test_rings_train_split_covers_every_ring_with_the_wrong_weights():
    """Chains drawn on every ring by a skewed allocation: the widths are right, the weights not"""
    target = Rings()
    x = target.train_samples(n=2048)
    assert len(target.mode_labels(x).unique()) == target.n_modes

    train = target.compute_metrics(x, compute_standard_metrics=False)
    exact = target.compute_metrics(target.test_samples(n=2048), compute_standard_metrics=False)
    assert train['perc_forgotten_modes'] == 0.0
    assert train['tv_weights'] > 10 * exact['tv_weights']

    # The chains travelled around their rings rather than sitting at their start
    angle = torch.atan2(x[:, 1], x[:, 0])
    assert float(angle.max() - angle.min()) > 5.0


def test_checkerboard_is_flat_on_its_squares_and_minus_infinity_off_them():
    """A uniform mixture: two density values in a 3 : 1 ratio, and no support between squares"""
    target = Checkerboard()
    assert target.n_modes == 8
    x = target.test_samples(n=4000)
    log_prob = target.log_prob(x)
    assert torch.isfinite(log_prob).all()

    # Exactly two density levels, and their ratio is the 3 : 1 weight alternation
    levels = torch.unique(torch.round(log_prob, decimals=9))
    assert len(levels) == 2
    assert abs(float(torch.exp(levels[1] - levels[0])) - 3.0) < 1e-6

    # Off the board there is no density at all
    off = torch.tensor([[100.0, 100.0], [-1.0, 1.0]], dtype=torch.float64)
    assert float(target.log_prob(off[:1])) == float('-inf')

    # Every square is 2 x 2 and they tile without overlapping
    assert torch.allclose(target.high - target.low, torch.full((8, 2), 2.0, dtype=torch.float64))
    assert int(target.inside(x).sum(dim=-1).max()) == 1


def test_checkerboard_score_is_exactly_zero():
    """The one target in the catalog a score-based sampler learns nothing from"""
    target = Checkerboard()
    x = target.test_samples(n=256)
    assert torch.equal(target.score(x), torch.zeros_like(x))

    # log_prob carries no gradient at all, so autograd cannot even build a graph for it
    y = x.clone().requires_grad_(True)
    assert target.log_prob(y).grad_fn is None

    # A score-based sampler gets no signal at all from this target, which is the point of it
    assert float(target.score(x).abs().max()) == 0.0


def test_checkerboard_train_split_starves_the_heavy_squares():
    """The skewed allocation gives the heavy squares the smallest shares, so the alternation flips

    Twenty steps of size 0.5 from a uniform point of each square leak little across the corners
    where the squares meet, so the shares stay those of the allocation.
    """
    target = Checkerboard()
    x = target.train_samples(n=2048)
    assert torch.isfinite(target.log_prob(x)).all()
    assert len(target.mode_labels(x).unique()) == target.n_modes

    # The heavy squares, three times the light ones under the target, get less than them here
    hist = target.compute_mode_weights(x)[0]
    heavy = target.mode_weights > target.mode_weights.mean()
    assert float(hist[heavy].mean() / hist[~heavy].mean()) < 1.0

    train = target.compute_metrics(x, compute_standard_metrics=False)
    exact = target.compute_metrics(target.test_samples(n=2048), compute_standard_metrics=False)
    assert train['tv_weights'] > 5 * exact['tv_weights']


def test_eight_gaussians_sit_on_a_circle_with_equal_weights():
    """Eight modes, equally weighted, on a circle whose radius is a documented choice"""
    target = EightGaussians()
    assert target.n_modes == 8
    assert torch.allclose(target.weights, torch.full((8,), 1 / 8, dtype=torch.float64))
    assert torch.allclose(torch.linalg.norm(target.means, dim=-1),
                          torch.full((8,), 2.0, dtype=torch.float64))

    # Consecutive means are equally spaced around the circle
    gaps = torch.linalg.norm(target.means - target.means.roll(1, dims=0), dim=-1)
    assert torch.allclose(gaps, gaps[0].expand(8))

    # Exact draws visit every mode, and the analytic score is exact
    x = target.test_samples(n=4000)
    assert len(target.mode_labels(x).unique()) == 8
    assert torch.allclose(target.score(x), score_autograd(target, x), rtol=1e-9, atol=1e-9)

    # Equal weights get the skewed allocation along the index order, so the train split visits
    # every mode with the wrong weights rather than stratifying them
    train = target.compute_metrics(target.train_samples(n=2048), compute_standard_metrics=False)
    assert train['perc_forgotten_modes'] == 0.0
    assert train['tv_weights'] > 0.4
