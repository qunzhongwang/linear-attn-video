"""Gradient-flow tests for BGDA."""
import torch

from spec.bgda.reference import bgda_block_attention_reference


def _make_param(B=1, F=4, HW=3, H=2, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    q = torch.nn.functional.normalize(rand(B, F, HW, H, d), dim=-1).requires_grad_(True)
    k = torch.nn.functional.normalize(rand(B, F, HW, H, d), dim=-1).requires_grad_(True)
    v = rand(B, F, HW, H, d).requires_grad_(True)
    alpha = torch.sigmoid(rand(B, F, H)).requires_grad_(True)
    beta = torch.sigmoid(rand(B, F, HW, H)).requires_grad_(True)
    return q, k, v, alpha, beta


def test_grads_flow_through_all_inputs():
    q, k, v, alpha, beta = _make_param()
    out = bgda_block_attention_reference(q, k, v, alpha, beta, W_latent=2, use_recurrence=True)
    loss = out.sum()
    loss.backward()
    for name, t in [("q", q), ("k", k), ("v", v), ("alpha", alpha), ("beta", beta)]:
        assert t.grad is not None, f"no grad for {name}"
        assert torch.isfinite(t.grad).all(), f"grad of {name} has NaN/Inf"
        assert (t.grad.abs() > 0).any(), f"grad of {name} is all zero"


def test_grads_finite_with_tiny_block_extreme_gates():
    """Stress: extreme gate values shouldn't NaN out."""
    q, k, v, _, _ = _make_param()
    B, F_, HW, H, _ = q.shape
    alpha = torch.full((B, F_, H), 0.999, dtype=torch.float64, requires_grad=True)
    beta = torch.full((B, F_, HW, H), 0.999, dtype=torch.float64, requires_grad=True)
    out = bgda_block_attention_reference(q, k, v, alpha, beta, W_latent=2, use_recurrence=True)
    loss = (out ** 2).sum()
    loss.backward()
    for t in (q, k, v, alpha, beta):
        assert torch.isfinite(t.grad).all()
