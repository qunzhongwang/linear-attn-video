"""Wrapper around `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (Gated DeltaNet, ICLR'25).

Strategy: call FLA **per chunk**, chaining `initial_state` between chunks. Within a chunk we
pass `g=0` (no decay) and `beta=β_t` constant. After each chunk we post-multiply S by α_t to
match the reference recurrence:

    for token i in chunk t:  S ← S (I - β k_i k_iᵀ) + β v_i k_iᵀ
    S_t ← α_t · S

This avoids the "where do I put log α?" trap of the per-token-decay approach.
"""
from __future__ import annotations

import math
import torch

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    HAS_FLA = True
except Exception:  # pragma: no cover
    chunk_gated_delta_rule = None
    HAS_FLA = False


def gdsa_fla(
    Q: torch.Tensor,                     # (B, H, T, D)
    K: torch.Tensor,                     # (B, H, T, D)
    V: torch.Tensor,                     # (B, H, T, D)
    alpha: torch.Tensor,                 # (B, H, n_chunks)
    beta: torch.Tensor,                  # (B, H, n_chunks)
    chunk_sizes: list[int],
    Q_for_value: torch.Tensor | None = None,
    eps: float = 1e-5,
    qk_l2norm: bool = True,
):
    """FLA-backed GDSA. Per-chunk call + α post-multiply.

    Note: when `qk_l2norm=True`, we normalize on our side so the recurrence's
    `(I - β k k^T)` operator has bounded spectral norm (Gated DeltaNet stability convention).
    The Z-normalizer denominator is computed on the slow path.
    """
    if not HAS_FLA:
        raise RuntimeError("flash-linear-attention not installed")

    B, H, T, D = Q.shape
    assert sum(chunk_sizes) == T

    Qv = Q if Q_for_value is None else Q_for_value

    if qk_l2norm:
        Q_n = torch.nn.functional.normalize(Q, p=2, dim=-1, eps=eps)
        K_n = torch.nn.functional.normalize(K, p=2, dim=-1, eps=eps)
        Qv_n = torch.nn.functional.normalize(Qv, p=2, dim=-1, eps=eps)
    else:
        Q_n, K_n, Qv_n = Q, K, Qv

    out_S = torch.zeros_like(V)            # (B, H, T, D), un-normalized output φ(q)·S

    # FLA layout: (B, T, H, D) and per-token (B, T, H) gates. State: (B, H, K, V).
    # initial_state shape: (N=B, HV=H, K=D, V=D).
    state = torch.zeros(B, H, D, D, device=Q.device, dtype=Q.dtype)

    pos = 0
    for t, n_t in enumerate(chunk_sizes):
        q_chunk = Qv_n[:, :, pos : pos + n_t, :].transpose(1, 2).contiguous()  # (B, n_t, H, D)
        k_chunk = K_n[:, :, pos : pos + n_t, :].transpose(1, 2).contiguous()
        v_chunk = V[:, :, pos : pos + n_t, :].transpose(1, 2).contiguous()
        g_chunk = torch.zeros(B, n_t, H, device=Q.device, dtype=Q.dtype)     # no decay within chunk
        b_chunk = beta[:, :, t].unsqueeze(1).expand(-1, n_t, -1).contiguous()  # (B, n_t, H)

        # Use FLA to compute the post-chunk state (S after all writes in this chunk). Discard the
        # per-token output: our block-causal semantics make all tokens of a chunk see the
        # POST-CHUNK state, not their per-token-causal state.
        _, state = chunk_gated_delta_rule(
            q=q_chunk, k=k_chunk, v=v_chunk,
            g=g_chunk, beta=b_chunk,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
        )

        # Apply chunk decay AFTER the writes — matches reference recurrence exactly.
        a_t = alpha[:, :, t].view(B, H, 1, 1).to(state.dtype)
        state = state * a_t

        # Now compute the chunk's S-path output using the post-chunk (and post-decay) state.
        # state shape from FLA: (B, H, K, V) where K=V=D in our setup.
        # FLA always returns fp32 state; cast back to input dtype for the output matmul.
        Qvt_n = Qv_n[:, :, pos : pos + n_t, :]   # (B, H, n_t, D)
        out_S[:, :, pos : pos + n_t, :] = Qvt_n @ state.to(Qvt_n.dtype)

        pos += n_t

    # Z (denominator) computed on the slow path — same per-chunk gated-delta recurrence with v=k.
    # Use the L2-normalized K and Q to match the S-path semantics.
    Z = torch.zeros(B, H, 1, D, device=Q.device, dtype=Q.dtype)
    out = torch.zeros_like(out_S)
    pos = 0
    for t, n_t in enumerate(chunk_sizes):
        b_t = beta[:, :, t].view(B, H, 1, 1)
        a_t = alpha[:, :, t].view(B, H, 1, 1)
        Kt = K_n[:, :, pos : pos + n_t, :]
        Qt = Q_n[:, :, pos : pos + n_t, :]
        for i in range(n_t):
            k = Kt[:, :, i : i + 1, :]
            Zk = Z @ k.transpose(-1, -2)
            Z = Z - b_t * (Zk @ k) + b_t * k
        Z = a_t * Z

        den = (Qt @ Z.transpose(-1, -2)).clamp_min(eps)
        out[:, :, pos : pos + n_t, :] = out_S[:, :, pos : pos + n_t, :] / den
        pos += n_t

    return out, state, Z
