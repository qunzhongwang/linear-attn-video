"""Backward parity: chunked.backward ≡ reference.backward, plus gradcheck on reference."""
import pytest
import torch

from spec.gdsa.reference import gdsa_reference
from spec.gdsa.chunked import gdsa_chunked
from spec.gdsa.tests.conftest import make_qkv, make_gates


def _grads_for(out_r, inputs):
    """Sum the output, backprop, return grads (cloned) for each leaf in `inputs`."""
    g = []
    out_r.sum().backward(retain_graph=False)
    for x in inputs:
        g.append(x.grad.detach().clone() if x.grad is not None else None)
        if x.grad is not None:
            x.grad = None
    return g


def test_backward_parity_fp64(tiny_shape, device):
    Q, K, V = make_qkv(tiny_shape, dtype=torch.float64, device=device)
    alpha, beta = make_gates(tiny_shape, dtype=torch.float64, device=device)

    Q_r, K_r, V_r, alpha_r, beta_r = (t.detach().clone().requires_grad_(True) for t in (Q, K, V, alpha, beta))
    Q_c, K_c, V_c, alpha_c, beta_c = (t.detach().clone().requires_grad_(True) for t in (Q, K, V, alpha, beta))

    out_r, _, _ = gdsa_reference(Q_r, K_r, V_r, alpha_r, beta_r, chunk_sizes=tiny_shape["chunk_sizes"])
    out_c, _, _ = gdsa_chunked(Q_c, K_c, V_c, alpha_c, beta_c, chunk_sizes=tiny_shape["chunk_sizes"])

    g_r = _grads_for(out_r, [Q_r, K_r, V_r, alpha_r, beta_r])
    g_c = _grads_for(out_c, [Q_c, K_c, V_c, alpha_c, beta_c])

    for name, gr, gc in zip(["Q", "K", "V", "alpha", "beta"], g_r, g_c):
        assert torch.allclose(gr, gc, atol=1e-9, rtol=0), \
            f"grad mismatch on {name}: max abs diff = {(gr - gc).abs().max().item()}"


@pytest.mark.slow
def test_gradcheck_reference_tiny():
    """torch.autograd.gradcheck on a *very* small instance — confirms our analytic forward is
    differentiable and torch's autograd is consistent (catches in-place / non-diff bugs)."""
    B, H, T, D = 1, 1, 4, 4
    chunk_sizes = [2, 2]
    g = torch.Generator().manual_seed(7)
    Q = torch.randn(B, H, T, D, generator=g, dtype=torch.float64).abs().requires_grad_(True)
    K = torch.randn(B, H, T, D, generator=g, dtype=torch.float64).abs().requires_grad_(True)
    V = torch.randn(B, H, T, D, generator=g, dtype=torch.float64).requires_grad_(True)
    alpha = torch.sigmoid(torch.randn(B, H, len(chunk_sizes), generator=g, dtype=torch.float64)).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, H, len(chunk_sizes), generator=g, dtype=torch.float64)).requires_grad_(True)

    def fn(Q, K, V, alpha, beta):
        out, _, _ = gdsa_reference(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes)
        return out

    assert torch.autograd.gradcheck(fn, (Q, K, V, alpha, beta), eps=1e-6, atol=1e-4, rtol=1e-3)
