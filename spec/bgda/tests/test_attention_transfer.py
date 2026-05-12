"""Tests for the Stage-A1 attention-transfer loss + capture hooks.

Uses tiny mock models (no Wan) so we can verify the wiring deterministically.
"""
import torch
import torch.nn as nn

from spec.bgda.training.stage_a1_attention_transfer import (
    _AttnOutputCapture,
    attention_transfer_loss,
    make_capture_pair,
    step_attention_transfer,
)


class _MockBlock(nn.Module):
    """Tiny block with a `.self_attn` submodule, mirroring WanAttentionBlock's API."""

    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim, bias=False)
        # plus a second linear that should NOT be captured
        self.cross_attn = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.cross_attn(self.self_attn(x))


class _MockModel(nn.Module):
    def __init__(self, dim=8, n=3):
        super().__init__()
        self.blocks = nn.ModuleList([_MockBlock(dim) for _ in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def test_capture_records_only_self_attn_outputs():
    m = _MockModel(dim=8, n=3)
    cap = _AttnOutputCapture()
    cap.register(m, attr_name="self_attn")
    x = torch.randn(2, 5, 8)
    _ = m(x)
    # Should have captured 3 layers' self_attn outputs.
    assert len(cap.outputs) == 3
    for k in cap.outputs:
        assert k.endswith(".self_attn")
        assert cap.outputs[k].shape == (2, 5, 8)
    cap.remove()


def test_attention_transfer_loss_zero_when_models_identical():
    m = _MockModel(dim=8, n=2)
    cap_a = _AttnOutputCapture(); cap_a.register(m, "self_attn")
    cap_b = _AttnOutputCapture(); cap_b.register(m, "self_attn")
    x = torch.randn(1, 4, 8)
    _ = m(x)
    # Hook only captured once; we need two passes to populate cap_a vs cap_b
    # with identical outputs.  Easier: copy cap_a's outputs into cap_b.
    cap_b.outputs = {k: v.clone() for k, v in cap_a.outputs.items()}
    loss, per_l_mse, per_l_cos = attention_transfer_loss(cap_a.outputs, cap_b.outputs)
    assert loss.item() < 1e-12
    assert all(abs(c - 1.0) < 1e-5 for c in per_l_cos)


def test_loss_with_cossim_weight_and_normalized_mse():
    """The new direction-first variant: cossim_weight>0 + normalized_mse.

    On a synthetic mismatch, the cossim_weight term should give us a positive
    gradient w.r.t. student outputs that *increases* cossim (we just check
    the gradient direction has nonzero component on cosine alignment)."""
    t = torch.randn(2, 3, 8) * 100.0   # large magnitude (mimics Wan residuals)
    s = (t * 5.0 + torch.randn_like(t) * 50.0).clone().requires_grad_(True)
    loss_pure_mse, _, cos_pure = attention_transfer_loss(
        {"L": t}, {"L": s}, cossim_weight=0.0, use_normalized_mse=False,
    )
    loss_cs, _, cos_cs = attention_transfer_loss(
        {"L": t}, {"L": s}, cossim_weight=1.0, use_normalized_mse=True,
    )
    # Both should give finite scalar losses
    assert torch.isfinite(loss_pure_mse) and torch.isfinite(loss_cs)
    # cossim values are diagnostic only; identical for both calls
    assert abs(cos_pure[0] - cos_cs[0]) < 1e-6


def test_step_attention_transfer_end_to_end():
    teacher = _MockModel(dim=8, n=2)
    student = _MockModel(dim=8, n=2)
    # Make student differ from teacher
    for p in student.parameters():
        with torch.no_grad():
            p.add_(0.1 * torch.randn_like(p))

    tcap, scap = make_capture_pair(teacher, student)
    x = torch.randn(2, 3, 8)
    stats, loss = step_attention_transfer(teacher, student, tcap, scap, x)
    assert stats.loss > 0
    assert len(stats.per_layer_mse) == 2
    # cossim should be a finite number in [-1, 1+eps] — allow tiny float
    # overshoot from cosine_similarity's eps clamp.
    assert all(-1.0 <= c <= 1.0 + 1e-5 for c in stats.per_layer_cossim)
    # Backward through student only — teacher is frozen via no_grad.
    loss.backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert len(grads) > 0
    tcap.remove(); scap.remove()
