"""Constant memory: state size is invariant to N (number of chunks).

This is the key property GDSA must preserve from vanilla SANA — the entire SANA-Video selling
point is constant memory at long sequence length. We assert state shapes don't grow.
"""
import torch

from spec.gdsa.reference import gdsa_reference
from spec.gdsa.tests.conftest import make_qkv, make_gates


def _final_state_shapes(n_chunks: int, n_per_chunk: int = 4, B=1, H=2, D=8):
    chunk_sizes = [n_per_chunk] * n_chunks
    T = sum(chunk_sizes)
    g = torch.Generator().manual_seed(0)
    Q = torch.randn(B, H, T, D, generator=g, dtype=torch.float64).abs()
    K = torch.randn(B, H, T, D, generator=g, dtype=torch.float64).abs()
    V = torch.randn(B, H, T, D, generator=g, dtype=torch.float64)
    alpha = torch.sigmoid(torch.randn(B, H, n_chunks, generator=g, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(B, H, n_chunks, generator=g, dtype=torch.float64))

    _, S, Z = gdsa_reference(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes)
    return S.shape, Z.shape


def test_state_shape_invariant_in_N():
    s1_S, s1_Z = _final_state_shapes(n_chunks=1)
    s8_S, s8_Z = _final_state_shapes(n_chunks=8)
    s32_S, s32_Z = _final_state_shapes(n_chunks=32)

    assert s1_S == s8_S == s32_S, f"S shape changed with N: {s1_S} vs {s8_S} vs {s32_S}"
    assert s1_Z == s8_Z == s32_Z, f"Z shape changed with N: {s1_Z} vs {s8_Z} vs {s32_Z}"
