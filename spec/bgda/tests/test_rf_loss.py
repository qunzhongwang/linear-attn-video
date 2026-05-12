"""Stage A2 rectified-flow loss tests."""
import torch

from spec.bgda.training.stage_a2_diffusion_loss import (
    sample_rf_timesteps, make_rf_input, rf_velocity_target,
)


def test_rf_t_zero_is_clean_latent():
    x0 = torch.randn(2, 3, 4, 5)
    eps = torch.randn(2, 3, 4, 5)
    t = torch.zeros(2)
    x_t = make_rf_input(x0, eps, t.view(2, 1, 1, 1))
    # broadcast t scalar through (1, 1, 1, 1) does (1-0)*x0 + 0*eps = x0
    # The make_rf_input expects t as 0-d for per-sample, with broadcasting; we
    # pass t already broadcast for simplicity.
    assert torch.allclose(x_t[0], x0[0]), "at t=0, x_t should equal x0"


def test_rf_t_one_is_pure_noise():
    x0 = torch.randn(2, 3, 4, 5)
    eps = torch.randn(2, 3, 4, 5)
    t = torch.ones(2).view(2, 1, 1, 1)
    x_t = make_rf_input(x0, eps, t)
    assert torch.allclose(x_t[0], eps[0]), "at t=1, x_t should equal eps"


def test_rf_velocity_target():
    x0 = torch.randn(3, 4, 5)
    eps = torch.randn(3, 4, 5)
    v = rf_velocity_target(x0, eps)
    assert torch.equal(v, eps - x0)


def test_sample_rf_timesteps_in_range():
    t = sample_rf_timesteps(100, t_low=0.1, t_high=0.9, device=torch.device("cpu"))
    assert t.shape == (100,)
    assert (t >= 0.1).all() and (t <= 0.9).all()


def test_make_rf_input_with_per_sample_t():
    """When t is a per-sample scalar, broadcasting should produce different
    interpolation per sample."""
    x0 = torch.zeros(3, 2, 2, 2)
    eps = torch.ones(3, 2, 2, 2)
    # Per-sample t: [0.0, 0.5, 1.0]
    # We test the non-list helper: x_t = (1-t)*x0 + t*eps.  Since x0=0 and
    # eps=1, x_t should be t broadcast.
    for i, ti in enumerate([0.0, 0.5, 1.0]):
        t = torch.full((1, 1, 1, 1), ti)
        out = make_rf_input(x0[i:i+1], eps[i:i+1], t)
        assert torch.allclose(out, torch.full_like(out, ti))
