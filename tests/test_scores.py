# Every analytic score must match autograd, on every target

# Libraries
import pytest
import torch
from samplingdb import (
    DW4, LJ13, LJ55, BracketTwoModes, EightGaussians, Funnel, GMM40, ManyModes, ManyWell, MoS,
    Rings, TwoModes,
)
from samplingdb.targets import score_autograd
from samplingdb.targets.base import generator

# One instance of every target, with the scale of a sensible evaluation point
TARGETS = [
    (Funnel, 1.0),
    (GMM40, 20.0),
    (MoS, 5.0),
    (ManyWell, 1.5),
    (TwoModes, 1.0),
    (BracketTwoModes, 1.0),
    (ManyModes, 2.0),
    (EightGaussians, 2.0),
    (Rings, 3.0),
    (DW4, 3.0),
    (LJ13, 1.0),
    (LJ55, 1.0),
]


@pytest.mark.parametrize('target_cls,scale', TARGETS)
def test_analytic_score_matches_autograd(target_cls, scale):
    """The hand-written score is the reference; autograd only checks it"""
    target = target_cls()
    x = scale * torch.randn(16, target.dim, generator=generator(0), dtype=torch.float64)
    score = target.score(x)
    assert score.shape == x.shape
    assert torch.allclose(score, score_autograd(target, x), rtol=1e-9, atol=1e-9)

    # log_prob_and_grad returns the same two quantities
    log_prob, grad = target.log_prob_and_grad(x)
    assert torch.allclose(log_prob, target.log_prob(x))
    assert torch.allclose(grad, score)


@pytest.mark.parametrize('target_cls,scale', TARGETS)
def test_no_target_falls_back_on_autograd(target_cls, scale):
    """score() must not need grad-enabled inputs, which is what an autograd fallback would"""
    target = target_cls()
    x = scale * torch.randn(4, target.dim, generator=generator(0), dtype=torch.float64)
    with torch.no_grad():
        score = target.score(x)
    assert torch.isfinite(score).all()
    assert not score.requires_grad


def test_score_is_finite_at_coincident_particles():
    """The epsilon inside every distance keeps the particle scores finite at d_ij = 0"""
    for target in [DW4(), LJ13()]:
        x = torch.zeros(1, target.dim, dtype=torch.float64)
        assert torch.isfinite(target.score(x)).all()


def test_mixture_responsibilities_are_shared_between_log_prob_and_score():
    """log_prob_and_grad computes the softmax once and reuses it"""
    target = GMM40()
    x = 20.0 * torch.randn(32, target.dim, generator=generator(0), dtype=torch.float64)
    probs, log_prob = target.responsibilities(x, return_log_prob=True)
    assert torch.allclose(probs.sum(-1), torch.ones(len(x), dtype=torch.float64))
    assert torch.allclose(log_prob, target.log_prob(x))
    assert torch.allclose(torch.sum(probs[..., None] * target.component_score(x), dim=1),
                          target.score(x))
