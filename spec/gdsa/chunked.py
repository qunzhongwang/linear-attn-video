"""Chunked / vectorized PyTorch implementations.

Avoids Python-level token loops by fusing the token sweep inside a chunk into matrix ops.
Still pure PyTorch — no Triton, no custom CUDA. Profile first, then decide on fusion.

Same tensor convention as `reference.py`: (B, H, T, D), state (B, H, D, D), normalizer (B, H, 1, D).
"""
from __future__ import annotations

import torch


def vanilla_chunked(
    Q: torch.Tensor,          # (B, H, T, D)
    K: torch.Tensor,          # (B, H, T, D)
    V: torch.Tensor,          # (B, H, T, D)
    chunk_sizes: list[int],
    Q_for_value: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla SANA cumulative linear attention. The chunkwise batched matmul does the same job
    as the inner token loop in `vanilla_reference` since the update is additive."""
    B, H, T, D = Q.shape
    assert sum(chunk_sizes) == T

    Qv = Q if Q_for_value is None else Q_for_value
    out = torch.zeros_like(V)

    S = torch.zeros(B, H, D, D, device=Q.device, dtype=Q.dtype)
    Z = torch.zeros(B, H, 1, D, device=Q.device, dtype=Q.dtype)

    pos = 0
    for n_t in chunk_sizes:
        Kt = K[:, :, pos : pos + n_t, :]
        Vt = V[:, :, pos : pos + n_t, :]
        Qt = Q[:, :, pos : pos + n_t, :]
        Qvt = Qv[:, :, pos : pos + n_t, :]

        # Whole-chunk additive update: K^T V is exactly the per-token sum.
        S = S + Kt.transpose(-1, -2) @ Vt
        Z = Z + Kt.sum(dim=-2, keepdim=True)

        num = Qvt @ S
        den = (Qt @ Z.transpose(-1, -2)).clamp_min(eps)
        out[:, :, pos : pos + n_t, :] = num / den

        pos += n_t

    return out, S, Z


def _gdsa_within_chunk(
    S: torch.Tensor,         # (B, H, D, D) initial state
    Z: torch.Tensor,         # (B, H, 1, D)
    Kt: torch.Tensor,        # (B, H, n, D)
    Vt: torch.Tensor,        # (B, H, n, D)
    beta: torch.Tensor,      # (B, H, 1, 1) scalar gate broadcast
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the gated-delta recurrence to one chunk, token by token.

    This is the vectorization frontier. Token sweep is a tight loop; each iteration is
    O(D^2) matmuls, total O(n D^2). Same asymptotic as the loop in `reference.py` but with
    a tighter inner body (no Python-level slicing of Kt[:, :, i:i+1]).

    For chunk sizes that matter in SANA-Video Stage 3 (n ≈ 21 latent tokens per
    block-time × spatial = up to ~500 tokens for 480p; bigger blocks then). Chunk-internal
    parallelism via DPLR / chunk-recurrent kernels (Yang et al. 2024) is the natural next step;
    deferred until profiling shows it matters.
    """
    n = Kt.size(-2)
    for i in range(n):
        k = Kt[:, :, i, :].unsqueeze(-2)        # (B, H, 1, D)  — row vector
        v = Vt[:, :, i, :].unsqueeze(-2)        # (B, H, 1, D)
        # State layout: S[..., d_key, d_value]. Erase = (I - β k kᵀ) S = S - β k (kᵀ S).
        kS = k @ S                              # (B, H, 1, D)
        S = S - beta * (k.transpose(-1, -2) @ kS)
        # Write: S ← S + β k vᵀ (key on axis-0, value on axis-1)
        S = S + beta * (k.transpose(-1, -2) @ v)
        # Z follows the same rule along the key axis only
        Zk = Z @ k.transpose(-1, -2)            # (B, H, 1, 1)
        Z = Z - beta * (Zk @ k)
        Z = Z + beta * k

    return S, Z


def gdsa_chunked(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    alpha: torch.Tensor,                       # (B, H, n_chunks)
    beta: torch.Tensor,                        # (B, H, n_chunks)
    chunk_sizes: list[int],
    Q_for_value: torch.Tensor | None = None,
    eps: float = 1e-5,
    qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gated Delta chunked PyTorch. Same outputs as `gdsa_reference`, vectorized where free."""
    B, H, T, D = Q.shape
    assert sum(chunk_sizes) == T
    assert alpha.shape == (B, H, len(chunk_sizes))
    assert beta.shape == alpha.shape

    Qv = Q if Q_for_value is None else Q_for_value

    if qk_l2norm:
        Q_n = torch.nn.functional.normalize(Q, p=2, dim=-1, eps=eps)
        K_n = torch.nn.functional.normalize(K, p=2, dim=-1, eps=eps)
        Qv_n = torch.nn.functional.normalize(Qv, p=2, dim=-1, eps=eps)
    else:
        Q_n, K_n, Qv_n = Q, K, Qv

    out = torch.zeros_like(V)

    S = torch.zeros(B, H, D, D, device=Q.device, dtype=Q.dtype)
    Z = torch.zeros(B, H, 1, D, device=Q.device, dtype=Q.dtype)

    pos = 0
    for t, n_t in enumerate(chunk_sizes):
        a_t = alpha[:, :, t].view(B, H, 1, 1)
        b_t = beta[:, :, t].view(B, H, 1, 1)

        Kt = K_n[:, :, pos : pos + n_t, :]
        Vt = V[:, :, pos : pos + n_t, :]
        Qt = Q_n[:, :, pos : pos + n_t, :]
        Qvt = Qv_n[:, :, pos : pos + n_t, :]

        S, Z = _gdsa_within_chunk(S, Z, Kt, Vt, b_t)
        S = a_t * S
        Z = a_t * Z

        num = Qvt @ S
        den = (Qt @ Z.transpose(-1, -2)).clamp_min(eps)
        out[:, :, pos : pos + n_t, :] = num / den

        pos += n_t

    return out, S, Z
