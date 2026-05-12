"""vanilla_chunked ≡ vanilla_reference, byte-for-byte."""
import pytest
import torch

from spec.gdsa.reference import vanilla_reference
from spec.gdsa.chunked import vanilla_chunked
from spec.gdsa.tests.conftest import make_qkv


def test_vanilla_parity_fp64(tiny_shape, device):
    Q, K, V = make_qkv(tiny_shape, dtype=torch.float64, device=device)
    out_r, S_r, Z_r = vanilla_reference(Q, K, V, chunk_sizes=tiny_shape["chunk_sizes"])
    out_c, S_c, Z_c = vanilla_chunked(Q, K, V, chunk_sizes=tiny_shape["chunk_sizes"])
    assert torch.allclose(out_r, out_c, atol=1e-12, rtol=0)
    assert torch.allclose(S_r, S_c, atol=1e-12, rtol=0)
    assert torch.allclose(Z_r, Z_c, atol=1e-12, rtol=0)


def test_vanilla_matches_sana_kvcache(tiny_shape, device):
    """The cumulative state (post all chunks) equals K^T V over the full sequence — the property
    SANA's CachedCausalAttention exploits for constant-memory Stage-3 inference."""
    Q, K, V = make_qkv(tiny_shape, dtype=torch.float64, device=device)
    _, S, Z = vanilla_reference(Q, K, V, chunk_sizes=tiny_shape["chunk_sizes"])
    expected_S = K.transpose(-1, -2) @ V             # (B, H, D, D)
    expected_Z = K.sum(dim=-2, keepdim=True)         # (B, H, 1, D)
    assert torch.allclose(S, expected_S, atol=1e-12, rtol=0)
    assert torch.allclose(Z, expected_Z, atol=1e-12, rtol=0)
