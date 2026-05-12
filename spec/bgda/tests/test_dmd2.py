"""Stage C DMD2 building-block tests using tiny mock score models."""
import torch
import torch.nn as nn

from spec.bgda.training.stage_c_dmd2 import (
    compute_score_residual, dmd2_generator_loss, dmd2_fake_score_loss,
)


class _MockScoreModel(nn.Module):
    """A tiny model that mimics WanModel's signature: takes a list of latents
    and returns a list of velocity predictions."""
    def __init__(self, dim=8, n=2):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self._n = n

    def forward(self, x_list, t=None, context=None, seq_len=None):
        out = []
        for x in x_list:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            y = self.linear(flat).reshape(shape)
            out.append(y)
        return out


def test_compute_score_residual_no_grad():
    real = _MockScoreModel(dim=8)
    fake = _MockScoreModel(dim=8)
    x = [torch.randn(2, 4, 8) for _ in range(2)]
    t = torch.zeros(2, dtype=torch.long)
    ctx = [torch.randn(4, 8) for _ in range(2)]
    res = compute_score_residual(real, fake, x, t, ctx, seq_len=8)
    assert len(res) == 2
    for r in res:
        assert r.shape == x[0].shape
        # Residual is purely tensor data — should not have requires_grad
        # since both score models were called under no_grad.
        assert r.requires_grad is False


def test_dmd2_generator_loss_finite_and_grads_flow():
    gen = _MockScoreModel(dim=8)
    real = _MockScoreModel(dim=8)
    fake = _MockScoreModel(dim=8)
    x = [torch.randn(1, 4, 8)]
    t = torch.zeros(1, dtype=torch.long)
    ctx = [torch.randn(4, 8)]
    loss = dmd2_generator_loss(gen, real, fake, x, t, ctx, seq_len=8)
    assert torch.isfinite(loss)
    loss.backward()
    # Generator params should have grad; real/fake should not (no_grad inside).
    assert gen.linear.weight.grad is not None
    assert real.linear.weight.grad is None
    assert fake.linear.weight.grad is None


def test_dmd2_fake_score_loss_only_fake_grads():
    gen = _MockScoreModel(dim=8)
    fake = _MockScoreModel(dim=8)
    x = [torch.randn(1, 4, 8)]
    t = torch.zeros(1, dtype=torch.long)
    ctx = [torch.randn(4, 8)]
    loss = dmd2_fake_score_loss(fake, gen, x, t, ctx, seq_len=8)
    assert torch.isfinite(loss)
    loss.backward()
    # Only fake should have grad
    assert fake.linear.weight.grad is not None
    # Generator was called under no_grad, so no grad accumulated on G
    assert gen.linear.weight.grad is None
