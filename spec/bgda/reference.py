"""BGDA reference implementation — naive PyTorch, ground truth for tests.

Layout convention (single canonical convention; the plan v1 §4.2 sample code
mixes `[d_k, d_v]` and `[d_v, d_k]` between read and update — we fix this here
once and reuse one layout everywhere):

  - State `S`: shape `[B, H, d, d]` where axis 2 is the *key dim* and axis 3 is
    the *value dim*. Both equal `head_dim` in practice (we keep the names for
    clarity).
  - Read:  `out = q @ S`, contracting q's last axis (key dim) with S axis 2.
  - Write (per token):  contribution `S += β · k ⊗ v` ⇒ `S[..., k, v] += β k v`.
  - Erasure (per frame): `S ← α · (I − Σ_n β_n k_n k_n^T) @ S` (left-mul,
    eats the key axis).

This is the standard outer-product-state linear-attention layout.

All time units are LATENT frames (post Wan-VAE 4× temporal compression). Plan v1
docstring "W=6 frames (~0.4 s @ 16 fps)" is the latent-frame count; the
corresponding pixel time is `W * 4 / 16` ≈ 1.5 s @16fps, not 0.4 s.

Math (per layer, per head, per block of `W_latent` consecutive latent frames):

    Within-block read (bidirectional, parallel over the block):
        S_read_t = S_carry_{t-1} + Σ_{n in block_t} (k̂_n β_n) ⊗ v_n      [eq.1]
        out_n    = q̂_n · S_read_t                                         [eq.2]

    Note plan §2.6 writes the within-block sum as `K̂^T V` (no β-weighting),
    which is the unweighted sum of in-block contributions. We follow §2.6: the
    *read* aggregates raw (k,v) pairs (β is for the across-block recurrence).

        S_read_t = S_carry_{t-1} + K̂_t^T V_t                              [eq.1, plan §2.6]

    Across-block recurrence (frame-sequential within a block, W_latent steps):
        for f in 1..W_latent:
            E_f = K̂_f^T diag(β_f) K̂_f                       [B, H, d, d]
            W_f = K̂_f^T diag(β_f) V_f                        [B, H, d, d]
            S = α_f · (S − E_f @ S) + W_f                    [B, H, d, d]
        S_carry_t = S

L2 normalization is applied to Q and K per head before RoPE (rotation preserves
L2 norm, so RoPE-after-L2 is safe). V is NOT L2-normed (per plan §2.2).

There is no `Σφ(K)` denominator (no `Z`) — see plan §2.6 and §1.1.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize along `dim`, with eps for safety."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


def _apply_beta_budget(beta: torch.Tensor, mode: Optional[str]) -> torch.Tensor:
    """Rescale per-frame β to bound the erasure operator norm.

    Modes:
      None      : pass-through (canonical GDN math).
      'hw'      : β / HW          — bounds operator norm linearly in HW.
      'sqrt_hw' : β / sqrt(HW)    — softer cap, closer to attention's 1/√d.

    Phase-0 microbench shows β=0.5 stress with HW=1560 blows up; using
    mode='hw' contains it.
    """
    if mode is None:
        return beta
    HW = beta.shape[-2] if beta.dim() == 3 else beta.shape[1]
    if mode == "hw":
        return beta / float(HW)
    if mode == "sqrt_hw":
        return beta / (float(HW) ** 0.5)
    raise ValueError(f"unknown beta_normalize mode: {mode!r}")


def gdn_state_update(
    S: torch.Tensor,
    k_frame: torch.Tensor,
    v_frame: torch.Tensor,
    alpha_frame: torch.Tensor,
    beta_frame: torch.Tensor,
    beta_normalize: Optional[str] = None,
) -> torch.Tensor:
    """One frame of the gated-delta state update.

    Performs:
        E = K^T diag(β) K                # erasure operator
        W = K^T diag(β) V                # write
        S_new = α · (S − E @ S) + W

    Shapes:
      S            : [B, H, d, d]      key on axis 2, value on axis 3
      k_frame      : [B, HW, H, d]     L2-normalized
      v_frame      : [B, HW, H, d]
      alpha_frame  : [B, H]            scalar per (batch, head), per frame
      beta_frame   : [B, HW, H]        scalar per (batch, token, head)

    Returns:
      S_new        : [B, H, d, d]
    """
    beta_frame = _apply_beta_budget(beta_frame, beta_normalize)
    # K^T diag(β) K : sum_n β_n k_n k_n^T
    k_beta = k_frame * beta_frame.unsqueeze(-1)             # [B, HW, H, d]
    E = torch.einsum("bnhd,bnhe->bhde", k_beta, k_frame)    # [B, H, d, d]
    # K^T diag(β) V : sum_n β_n k_n v_n^T
    W = torch.einsum("bnhd,bnhe->bhde", k_beta, v_frame)    # [B, H, d, d]
    # E @ S : contract E's last axis with S's first non-batch/head axis
    ES = torch.einsum("bhde,bhef->bhdf", E, S)              # [B, H, d, d]
    # α · (S − E @ S) + W
    a = alpha_frame.unsqueeze(-1).unsqueeze(-1)             # [B, H, 1, 1]
    return a * (S - ES) + W


def bgda_block_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    W_latent: int,
    use_recurrence: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    return_state: bool = False,
    beta_normalize: Optional[str] = None,
):
    """Naive reference impl of one BGDA layer over a sequence of blocks.

    Inputs are already QKV-projected, RMS/L2-normed, RoPE-applied if applicable.
    The function is layer-agnostic: it just runs the BGDA math.

    Shapes:
      q : [B, F, HW, H, d]    queries, L2-normed
      k : [B, F, HW, H, d]    keys, L2-normed
      v : [B, F, HW, H, d]    values, NOT L2-normed
      alpha : [B, F, H]       per-(frame, head) gate in (0, 1)
      beta  : [B, F, HW, H]   per-token, per-head gate in (0, 1)

    `F` (number of latent frames) must be divisible by `W_latent`.

    Returns:
      out  : [B, F, HW, H, d]
      (optional) S_final : [B, H, d, d]
    """
    B, Fnum, HW, H, d = q.shape
    assert Fnum % W_latent == 0, (
        f"F={Fnum} not divisible by W_latent={W_latent}; "
        "either pad to nearest multiple or pick a divisor block size"
    )
    N = Fnum // W_latent

    # Reshape to [B, N, W_latent, HW, H, d] so blocks are contiguous on axis 1.
    q = q.view(B, N, W_latent, HW, H, d)
    k = k.view(B, N, W_latent, HW, H, d)
    v = v.view(B, N, W_latent, HW, H, d)
    alpha = alpha.view(B, N, W_latent, H)
    beta = beta.view(B, N, W_latent, HW, H)

    if initial_state is None:
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
    else:
        S = initial_state

    if not use_recurrence:
        # Stage-A target: fully bidirectional linear attention over the WHOLE
        # sequence — every token sees every other token through one shared
        # global state `S_full = Σ_t K_t^T V_t`. Block structure is irrelevant
        # in this mode; gates α/β are unused.
        k_flat = k.reshape(B, N * W_latent * HW, H, d)
        v_flat = v.reshape(B, N * W_latent * HW, H, d)
        q_flat = q.reshape(B, N * W_latent * HW, H, d)
        S_full = S + torch.einsum("bnhd,bnhe->bhde", k_flat, v_flat)
        out = torch.einsum("bnhd,bhde->bnhe", q_flat, S_full)
        out = out.view(B, Fnum, HW, H, d)
        if return_state:
            return out, S_full
        return out

    outputs = []
    for t in range(N):
        # Within-block: flatten W * HW tokens of block t.
        q_blk = q[:, t].reshape(B, W_latent * HW, H, d)
        k_blk = k[:, t].reshape(B, W_latent * HW, H, d)
        v_blk = v[:, t].reshape(B, W_latent * HW, H, d)

        # Within-block sum: K^T V (NOT β-weighted; β is for across-block recurrence)
        kv_blk = torch.einsum("bnhd,bnhe->bhde", k_blk, v_blk)   # [B, H, d, d]

        # Read: q @ (S_carry + KV_block).  All tokens in block see the same
        # combined state — within-block bidirectional, no causal mask inside.
        combined = S + kv_blk                                     # [B, H, d, d]
        out_blk = torch.einsum("bnhd,bhde->bnhe", q_blk, combined)  # [B, n_b, H, d]
        out_blk = out_blk.view(B, W_latent, HW, H, d)
        outputs.append(out_blk)

        # Across-block update (frame-sequential, W_latent steps per block)
        for f in range(W_latent):
            S = gdn_state_update(
                S,
                k_frame=k[:, t, f],
                v_frame=v[:, t, f],
                alpha_frame=alpha[:, t, f],
                beta_frame=beta[:, t, f],
                beta_normalize=beta_normalize,
            )

    out = torch.stack(outputs, dim=1).reshape(B, Fnum, HW, H, d)
    if return_state:
        return out, S
    return out


def vanilla_linear_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Vanilla bidirectional linear attention (no recurrence, no gates, no
    denominator). For init-recovery test: at α=1, β=0, BGDA collapses to this.

    out = q @ (sum_n k_n ⊗ v_n)
    """
    # Sum over all n=F*HW tokens, per head.
    B, Fnum, HW, H, d = q.shape
    q_flat = q.reshape(B, Fnum * HW, H, d)
    k_flat = k.reshape(B, Fnum * HW, H, d)
    v_flat = v.reshape(B, Fnum * HW, H, d)
    S = torch.einsum("bnhd,bnhe->bhde", k_flat, v_flat)
    out = torch.einsum("bnhd,bhde->bnhe", q_flat, S)
    return out.view(B, Fnum, HW, H, d)
