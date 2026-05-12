"""gdsa_chunked ≡ gdsa_reference for random α, β."""
import pytest
import torch

from spec.gdsa.reference import gdsa_reference
from spec.gdsa.chunked import gdsa_chunked
from spec.gdsa.tests.conftest import make_qkv, make_gates


def test_gdsa_parity_fp64(tiny_shape, device):
    Q, K, V = make_qkv(tiny_shape, dtype=torch.float64, device=device)
    alpha, beta = make_gates(tiny_shape, dtype=torch.float64, device=device)

    out_r, S_r, Z_r = gdsa_reference(Q, K, V, alpha, beta, chunk_sizes=tiny_shape["chunk_sizes"])
    out_c, S_c, Z_c = gdsa_chunked(Q, K, V, alpha, beta, chunk_sizes=tiny_shape["chunk_sizes"])

    assert torch.allclose(out_r, out_c, atol=1e-10, rtol=0), \
        f"max abs diff = {(out_r - out_c).abs().max().item()}"
    assert torch.allclose(S_r, S_c, atol=1e-10, rtol=0)
    assert torch.allclose(Z_r, Z_c, atol=1e-10, rtol=0)


def test_gdsa_alpha1_beta0_freezes_state(tiny_shape, device):
    """α=1, β=0  ⇒  S stays at zero forever (no writes), so the output is 0/eps."""
    Q, K, V = make_qkv(tiny_shape, dtype=torch.float64, device=device)
    alpha, beta = make_gates(tiny_shape, dtype=torch.float64, device=device, alpha_val=1.0, beta_val=0.0)
    out, S, Z = gdsa_chunked(Q, K, V, alpha, beta, chunk_sizes=tiny_shape["chunk_sizes"])
    assert torch.allclose(S, torch.zeros_like(S), atol=0)
    assert torch.allclose(Z, torch.zeros_like(Z), atol=0)


def test_gdsa_alpha1_beta_close_to_1_gradually_writes(tiny_shape, device):
    """α=1, β=1, single chunk: S after sweep equals the delta-rule result with no decay,
    full write strength. Sanity: nonzero output."""
    shape = dict(tiny_shape, chunk_sizes=[tiny_shape["T"]])
    Q, K, V = make_qkv(shape, dtype=torch.float64, device=device)
    alpha, beta = make_gates(shape, dtype=torch.float64, device=device, alpha_val=1.0, beta_val=1.0)
    out, S, Z = gdsa_chunked(Q, K, V, alpha, beta, chunk_sizes=shape["chunk_sizes"])
    assert S.abs().sum().item() > 0
    assert Z.abs().sum().item() > 0
    assert out.abs().sum().item() > 0
