"""nn.Module wrapper for Gated Delta Split Attention (Mechanism #1).

Drop-in replacement for `CachedCausalAttention` in `diffusion/model/nets/sana_blocks.py`. Same
__init__ signature (proxied via the base class), same forward signature (`save_kv_cache`,
`kv_cache`), but the internal recurrence is gated-delta and the cache stores the post-update
**full** state matrix (not the per-chunk contribution).

Init recovers vanilla SANA at step 0:
    - alpha_proj.bias and beta_proj.bias initialized so σ(b_α)≈1, σ(b_β)≈0 (same as proposal §4)
    - vanilla_residual=True routes the additive linear-attention output through a residual
      until the gates open. We zero-init the GDSA branch's contribution.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chunked import gdsa_chunked, vanilla_chunked

try:
    from .fla_chunked import gdsa_fla, HAS_FLA
except Exception:
    HAS_FLA = False
    gdsa_fla = None


class GDSAAttention(nn.Module):
    """Gated-Delta variant of SANA's CachedCausalAttention.

    Args:
        in_dim, out_dim, heads, dim: same as LiteLA
        gate_init_alpha, gate_init_beta: target σ(bias) values for α / β at init.
            Default: α≈1.0 (full retain), β≈0.0 (no write) — pairs with `vanilla_residual=True`
            to make the module identical-output to vanilla SANA at init.
        vanilla_residual: if True, output = vanilla_linear_attn + lambda * delta(gdsa - vanilla),
            with `lambda` initialized to 0 so the module starts as vanilla SANA exactly.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int | None = None,
        dim: int = 32,
        eps: float = 1e-5,
        use_bias: bool = False,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
        gate_init_alpha: float = 1.0,   # σ(bias) target
        gate_init_beta: float = 0.0,
        vanilla_residual: bool = True,
        backend: str = "auto",          # "fla" | "chunked" | "auto" (= fla if available else chunked)
    ):
        super().__init__()
        heads = heads or out_dim // dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        self.vanilla_residual = vanilla_residual
        if backend == "auto":
            backend = "fla" if HAS_FLA else "chunked"
        if backend == "fla" and not HAS_FLA:
            raise RuntimeError("backend='fla' requested but flash-linear-attention is not available")
        self.backend = backend

        self.qkv = nn.Linear(in_dim, 3 * out_dim, bias=use_bias)
        self.proj = nn.Linear(out_dim, out_dim, bias=use_bias)

        # qk_norm: same RMSNorm-style as SANA. We import locally to avoid pulling in `mmcv`.
        if qk_norm:
            from diffusion.model.nets.sana_blocks import RMSNorm  # local; avoid module-load cycles
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.kernel_func = nn.ReLU(inplace=False)

        # Per-chunk gates: produced from a pooled-block-feature MLP. We keep a 2-layer MLP per
        # the proposal §4 ("a 2-layer MLP over pooled block features").
        # Input is the pooled (mean-over-tokens) k feature, dim = in_dim. Output 2 scalars per head.
        gate_hidden = max(in_dim // 4, 64)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, gate_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2 * heads, bias=True),
        )
        # Init: bias of the second layer such that σ(bias) = (alpha_init, beta_init).
        with torch.no_grad():
            self.gate_mlp[-1].weight.zero_()
            b = torch.empty(2 * heads)
            b[0::2] = self._inv_sigmoid(gate_init_alpha)
            b[1::2] = self._inv_sigmoid(gate_init_beta)
            self.gate_mlp[-1].bias.copy_(b)

        # Residual mixing scalar (vanilla branch + λ * delta). λ starts at 0 → vanilla-only output.
        if vanilla_residual:
            self.delta_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("delta_scale", None)

    @staticmethod
    def _inv_sigmoid(p: float) -> float:
        # logit. Clamp to avoid ±inf at exact 0/1.
        p = float(min(max(p, 1e-6), 1 - 1e-6))
        return torch.logit(torch.tensor(p)).item()

    def _gates(self, K_per_chunk: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-chunk α, β.

        K_per_chunk: list of (B, H, n_t, D) tensors (length = n_chunks). Returns (α, β) of shape
        (B, H, n_chunks), each ∈ (0,1).
        """
        # Pool over tokens within each chunk → (B, n_chunks, in_dim).
        # We use the *original* (pre-kernel) k features for the gate MLP. The wrapper passes them in.
        # Each tensor: (B, H, n_t, D); concat heads back to channel: → (B, n_t, H*D = in_dim).
        feats = []
        for kt in K_per_chunk:
            # kt: (B, H, n_t, D) → mean over n_t → (B, H, D) → flatten → (B, H*D)
            feats.append(kt.mean(dim=-2).flatten(-2, -1))
        feats = torch.stack(feats, dim=1)        # (B, n_chunks, H*D)
        g = self.gate_mlp(feats)                 # (B, n_chunks, 2*H)
        g = torch.sigmoid(g)
        # Split α, β; reshape to (B, H, n_chunks)
        alpha = g[..., 0::2].permute(0, 2, 1).contiguous()
        beta = g[..., 1::2].permute(0, 2, 1).contiguous()
        return alpha, beta

    def forward(
        self,
        x: torch.Tensor,                          # (B, N, C)
        chunk_sizes: list[int] | None = None,    # required for Stage-3 block-causal mode
        mask=None,
        HW=None,
        rotary_emb=None,
        block_mask=None,
        save_kv_cache: bool = False,
        kv_cache: Optional[list] = None,         # list[ S, Z, token_minus_1 ] per the SANA convention
        **kwargs,
    ):
        B, N, C = x.shape
        H, D = self.heads, self.dim

        if chunk_sizes is None:
            chunk_sizes = [N]
        assert sum(chunk_sizes) == N, f"chunk_sizes={chunk_sizes} must sum to N={N}"

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        # SANA uses `q_norm`/`k_norm` then transposes to (B, C, N). We keep (B, N, C) shape and
        # reshape directly to (B, H, N, D) for the recurrence math.
        q = self.q_norm(q).reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()  # (B, H, N, D)
        k = self.k_norm(k).reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()

        # Optionally apply RoPE to q, k for the (S @ q) value path. We follow SANA: RoPE is
        # applied AFTER ReLU. The `Q_for_value` argument lets us pass the rotated q to the
        # output computation while keeping the un-rotated q in the denominator (for stability).
        # Splitting handled inline:
        q_kf = self.kernel_func(q)
        k_kf = self.kernel_func(k)

        if rotary_emb is not None:
            # Match CachedCausalAttention's INLINE complex-tensor form. `rotary_emb` is a
            # complex tensor `freqs` of shape (..., D/2), and we apply it via complex multiply.
            def _rope(hidden, freqs):
                # hidden: (B, H, N, D). View as complex with D/2 pairs.
                x_c = torch.view_as_complex(
                    hidden.to(torch.float64).unflatten(3, (-1, 2))
                )                                          # (B, H, N, D/2)
                x_rot = torch.view_as_real(x_c * freqs)    # (B, H, N, D/2, 2)
                return x_rot.flatten(3, 4).type_as(hidden)
            q_for_value = _rope(q_kf, rotary_emb)
            k_kf = _rope(k_kf, rotary_emb)
        else:
            q_for_value = q_kf

        # Split into chunks (logical view, not data copy) per chunk_sizes; the inner kernel
        # iterates internally.
        out_vanilla, S_v, Z_v = vanilla_chunked(q_kf, k_kf, v, chunk_sizes, Q_for_value=q_for_value, eps=self.eps)

        if self.vanilla_residual and self.delta_scale.abs().item() == 0.0 and not self.training:
            # Hot path at init / inference before training: skip the GDSA branch entirely.
            out = out_vanilla
            S, Z = S_v, Z_v
        else:
            # Pool the *un-mapped* k for gate MLP (pre-ReLU keys carry sign information).
            k_chunks = []
            pos = 0
            for n_t in chunk_sizes:
                k_chunks.append(k[:, :, pos : pos + n_t, :])
                pos += n_t
            alpha, beta = self._gates(k_chunks)
            gdsa_impl = gdsa_fla if (self.backend == "fla" and HAS_FLA) else gdsa_chunked
            out_gdsa, S_g, Z_g = gdsa_impl(
                q_kf, k_kf, v, alpha, beta, chunk_sizes, Q_for_value=q_for_value, eps=self.eps,
            )
            if self.vanilla_residual:
                lam = self.delta_scale
                out = out_vanilla + lam * (out_gdsa - out_vanilla)
                # State for the *cache* is the GDSA state when λ has any weight; we always thread
                # the GDSA state since that's what makes the recurrence non-additive across chunks.
                S, Z = S_g, Z_g
            else:
                out = out_gdsa
                S, Z = S_g, Z_g

        out = out.to(dtype)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, H, N, D) → (B, N, C)
        out = self.proj(out)

        if kv_cache is not None:
            if save_kv_cache:
                kv_cache[0] = S.detach().clone()
                kv_cache[1] = Z.detach().clone()
            return out, kv_cache
        return out
