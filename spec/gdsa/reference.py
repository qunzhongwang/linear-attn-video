"""Reference (slow, naive, correct) implementations of the GDSA-1 and vanilla recurrences.

Used as ground truth for testing the chunked / fused implementations.

Tensor convention (matches SANA `CachedCausalAttention`):
    Q, K, V : (B, H, T, D)        head-dim last  (we transpose from SANA's (B,H,D,T) layout
                                                  in the wrapper module; inside this file we use
                                                  the standard (B,H,T,D) for clarity)
    S       : (B, H, D, D)        accumulator
    Z       : (B, H, 1, D)        normalizer (sum over T of K)

Chunking convention:
    a "chunk" = a list of contiguous tokens that share scalar gates (alpha_t, beta_t).
    Within a chunk we apply the recurrence token-by-token (reference impl); across chunks the
    state S is threaded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def vanilla_reference(
    Q: torch.Tensor,  # (B, H, T, D), kernel-mapped (e.g. ReLU) queries
    K: torch.Tensor,  # (B, H, T, D), kernel-mapped (e.g. ReLU) keys
    V: torch.Tensor,  # (B, H, T, D)
    chunk_sizes: list[int] | None = None,  # per-chunk token counts; sum = T
    Q_for_value: torch.Tensor | None = None,  # if provided (e.g. RoPE-rotated Q), used in the (S @ q) path; otherwise Q
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla SANA cumulative linear attention with block-causal chunking.

    For each chunk t with kernel-mapped K_t, V_t:
        S_t = S_{t-1} + K_tᵀ V_t      (D x D)
        Z_t = Z_{t-1} + sum_i K_t[i]  (D,)
    Output for the chunk:
        O_t = (S_t @ Q_t_for_value) / (Z_t @ Q_t)

    Returns (output, final_S, final_Z) where output has shape (B, H, T, D).
    """
    B, H, T, D = Q.shape
    if chunk_sizes is None:
        chunk_sizes = [T]
    assert sum(chunk_sizes) == T, f"chunk_sizes={chunk_sizes} must sum to T={T}"

    Qv = Q if Q_for_value is None else Q_for_value
    out = torch.zeros_like(V)  # (B, H, T, D)

    S = torch.zeros(B, H, D, D, device=Q.device, dtype=Q.dtype)
    Z = torch.zeros(B, H, 1, D, device=Q.device, dtype=Q.dtype)

    pos = 0
    for n_t in chunk_sizes:
        Kt = K[:, :, pos : pos + n_t, :]   # (B, H, n_t, D)
        Vt = V[:, :, pos : pos + n_t, :]
        Qt = Q[:, :, pos : pos + n_t, :]
        Qvt = Qv[:, :, pos : pos + n_t, :]

        # Cumulative additive update (vanilla SANA)
        S = S + Kt.transpose(-1, -2) @ Vt          # (B, H, D, D) += (B, H, D, n_t) @ (B, H, n_t, D)
        Z = Z + Kt.sum(dim=-2, keepdim=True)       # (B, H, 1, D)

        # Output for the chunk: (B, H, n_t, D) = (B, H, n_t, D) @ S_T (B, H, D, D)
        num = Qvt @ S                              # (B, H, n_t, D)
        den = (Qt @ Z.transpose(-1, -2)).clamp_min(eps)  # (B, H, n_t, 1)
        out[:, :, pos : pos + n_t, :] = num / den

        pos += n_t

    return out, S, Z


def gdsa_reference(
    Q: torch.Tensor,                     # (B, H, T, D)
    K: torch.Tensor,                     # (B, H, T, D)
    V: torch.Tensor,                     # (B, H, T, D)
    alpha: torch.Tensor,                 # (B, H, n_chunks) decay gate per chunk in (0,1)
    beta: torch.Tensor,                  # (B, H, n_chunks) write gate per chunk in (0,1)
    chunk_sizes: list[int],
    Q_for_value: torch.Tensor | None = None,
    eps: float = 1e-5,
    qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gated Delta recurrence (Mechanism #1) — token-level inside each chunk, scalar α, β per chunk.

    For each chunk t:
        S ← S_{t-1}
        for i in 0..n_t-1:
            k = K_t[i]; v = V_t[i]
            S ← S (I - β_t k kᵀ) + β_t v kᵀ
        S_t ← α_t * S

    The Z normalizer follows the same gated-delta recurrence so the (Z q) denominator stays meaningful:
        Z ← Z_{t-1}
        for i in 0..n_t-1:
            k = K_t[i]
            Z ← Z (I - β_t k kᵀ) + β_t k       (treating Z as a single row)
        Z_t ← α_t * Z

    Returns (output, final_S, final_Z).
    """
    B, H, T, D = Q.shape
    assert sum(chunk_sizes) == T
    assert alpha.shape == (B, H, len(chunk_sizes)), f"alpha shape {alpha.shape} vs expected {(B, H, len(chunk_sizes))}"
    assert beta.shape == alpha.shape

    Qv = Q if Q_for_value is None else Q_for_value

    if qk_l2norm:
        # Gated DeltaNet: L2-normalize q and k along the head-dim. Stabilizes ‖I - β k k^T‖.
        Q_n = torch.nn.functional.normalize(Q, p=2, dim=-1, eps=eps)
        K_n = torch.nn.functional.normalize(K, p=2, dim=-1, eps=eps)
        Qv_n = torch.nn.functional.normalize(Qv, p=2, dim=-1, eps=eps)
    else:
        Q_n, K_n, Qv_n = Q, K, Qv

    out = torch.zeros_like(V)

    S = torch.zeros(B, H, D, D, device=Q.device, dtype=Q.dtype)
    Z = torch.zeros(B, H, 1, D, device=Q.device, dtype=Q.dtype)

    eye = torch.eye(D, device=Q.device, dtype=Q.dtype).expand(B, H, D, D)

    pos = 0
    for t, n_t in enumerate(chunk_sizes):
        a_t = alpha[:, :, t].view(B, H, 1, 1)  # broadcast to (B,H,D,D)
        b_t = beta[:, :, t].view(B, H, 1, 1)

        Kt = K_n[:, :, pos : pos + n_t, :]
        Vt = V[:, :, pos : pos + n_t, :]
        Qt = Q_n[:, :, pos : pos + n_t, :]
        Qvt = Qv_n[:, :, pos : pos + n_t, :]

        # Snapshot pre-update state for the chunk's output
        S_in = S
        Z_in = Z

        # Token-by-token update inside the chunk.
        # State layout: S[..., d_key, d_value] (matches vanilla S = Kᵀ V).
        # Erase: S ← (I - β k kᵀ) S = S - β k (kᵀ S)
        # Write: S ← S + β k vᵀ
        for i in range(n_t):
            k = Kt[:, :, i : i + 1, :]   # (B, H, 1, D)  — row vector along last axis
            v = Vt[:, :, i : i + 1, :]
            # erase: project out the rank-1 component along k from rows of S (the key axis)
            kS = k @ S                              # (B, H, 1, D) — (kᵀ S) view
            S = S - b_t * (k.transpose(-1, -2) @ kS)  # (B, H, D, D)
            # write: outer product k ⊗ v
            S = S + b_t * (k.transpose(-1, -2) @ v)
            # Z is a (1, D) running sum along the key axis (no value axis).
            Zk = Z @ k.transpose(-1, -2)            # (B, H, 1, 1)
            Z = Z - b_t * (Zk @ k)
            Z = Z + b_t * k

        # Apply chunk decay
        S = a_t * S
        Z = a_t * Z

        # Output for this chunk uses post-update S, Z
        num = Qvt @ S
        den = (Qt @ Z.transpose(-1, -2)).clamp_min(eps)
        out[:, :, pos : pos + n_t, :] = num / den

        pos += n_t

    return out, S, Z
