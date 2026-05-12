"""Math correctness tests for BGDA reference impl.

These tests anchor the reference math against:
  1. A hand-rolled per-token gated-delta recurrence (single keys/values).
  2. Vanilla bidirectional linear attention (the use_recurrence=False target).
"""
import torch
import pytest

from spec.bgda.reference import (
    bgda_block_attention_reference,
    gdn_state_update,
    vanilla_linear_attention_reference,
)


def _make_random(B=1, F=6, HW=4, H=2, d=8, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    q = torch.nn.functional.normalize(rand(B, F, HW, H, d), dim=-1)
    k = torch.nn.functional.normalize(rand(B, F, HW, H, d), dim=-1)
    v = rand(B, F, HW, H, d)
    alpha = torch.sigmoid(rand(B, F, H))      # in (0, 1)
    beta = torch.sigmoid(rand(B, F, HW, H))   # in (0, 1)
    return q, k, v, alpha, beta


def test_use_recurrence_false_equals_vanilla_linear():
    """Stage A target: use_recurrence=False == full-sequence bidirectional linear attn."""
    q, k, v, alpha, beta = _make_random()
    out_bgda = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=3, use_recurrence=False
    )
    out_van = vanilla_linear_attention_reference(q, k, v)
    assert torch.allclose(out_bgda, out_van, atol=1e-10), \
        f"max diff = {(out_bgda - out_van).abs().max().item()}"


def test_recurrence_against_handrolled_per_token():
    """For W_latent=1 (one frame per block), within-block read is just the
    current frame's `q · K^T V`; across-block recurrence is exactly the
    canonical Yang+ICLR'25 GDN recurrence over all frames in order.

    We replay it by hand and compare to the reference."""
    B, Fnum, HW, H, d = 1, 5, 3, 2, 4
    q, k, v, alpha, beta = _make_random(B, Fnum, HW, H, d)
    out, S_final = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=1, use_recurrence=True, return_state=True,
    )
    # Hand recurrence: one block = one frame.
    S = torch.zeros(B, H, d, d, dtype=q.dtype)
    out_manual = torch.zeros_like(out)
    for t in range(Fnum):
        kv = torch.einsum("bnhd,bnhe->bhde", k[:, t], v[:, t])
        out_manual[:, t] = torch.einsum("bnhd,bhde->bnhe", q[:, t], S + kv)
        # alpha layout is [B, F, H] (no W axis), so alpha[:, t] is the
        # correct [B, H] slice for gdn_state_update.
        S = gdn_state_update(S, k[:, t], v[:, t], alpha[:, t], beta[:, t])
    assert torch.allclose(out, out_manual, atol=1e-10), \
        f"max diff = {(out - out_manual).abs().max().item()}"
    assert torch.allclose(S, S_final, atol=1e-10)


def test_block_size_divides_F():
    q, k, v, alpha, beta = _make_random(B=1, F=7, HW=2, H=1, d=4)
    with pytest.raises(AssertionError):
        bgda_block_attention_reference(q, k, v, alpha, beta, W_latent=3)


def test_dtype_preservation():
    q, k, v, alpha, beta = _make_random(dtype=torch.float32)
    out = bgda_block_attention_reference(q, k, v, alpha, beta, W_latent=2)
    assert out.dtype == torch.float32
