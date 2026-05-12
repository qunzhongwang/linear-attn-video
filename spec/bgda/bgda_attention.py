"""BGDA nn.Module wrapper, signature-compatible with Wan2.1 `WanSelfAttention`.

The forward signature mirrors `WanSelfAttention.forward(x, seq_lens, grid_sizes,
freqs)` so the module can be dropped into `WanAttentionBlock` in place of
`self.self_attn`. We add three constructor arguments: `W_latent`,
`use_recurrence`, `conv_kernel` (causal short conv along the temporal axis,
plan §2.2 GDN-style precondition).

Drop-in usage:

    from spec.bgda.bgda_attention import BGDABlockAttention

    block.self_attn = BGDABlockAttention(
        dim=1536, num_heads=12, qk_norm=True, eps=1e-6,
        W_latent=3, use_recurrence=False,   # Stage A
    )

Notes:
- Wan2.1's `WanSelfAttention` uses **RMSNorm** on Q/K (qk_norm=True). We keep
  that and additionally L2-normalize after the RMSNorm (so the L2 path is
  applied on top, not instead). RMS is for representational stability of the
  Q/K head; L2 is for the geometry of the (I − β k k^T) erasure (plan §2.2).
- 3D RoPE is applied via Wan's `rope_apply` (key-frequency complex tensor passed
  as `freqs`). RoPE is applied AFTER L2 — rotation preserves L2 norm.
- Causal short conv is along the **temporal** axis only (kernel 3×1×1), grouped
  per channel. Implemented as a 1D conv after `[B, T, HW, dim]` reshape.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference import bgda_block_attention_reference


def _try_import_wan():
    """Look up Wan's RMSNorm + rope_apply lazily.  We avoid importing them at
    module load because tests sometimes monkey-patch `wan.modules.model` after
    `bgda_attention` is first imported (e.g. integration tests that stub the
    Wan package).  Returns `(WanRMSNorm, rope_apply)` or `(None, None)`."""
    try:
        from wan.modules.model import WanRMSNorm, rope_apply  # type: ignore
        return WanRMSNorm, rope_apply
    except Exception:
        return None, None


class BGDABlockAttention(nn.Module):
    """Block Gated Delta Attention — drop-in for Wan2.1 WanSelfAttention.

    Constructor mirrors WanSelfAttention plus BGDA-specific knobs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),  # unused, kept for signature parity with Wan
        qk_norm: bool = True,
        eps: float = 1e-6,
        # BGDA-specific
        W_latent: int = 3,
        use_recurrence: bool = False,
        conv_kernel: int = 3,
        alpha_init_bias: float = 5.0,
        beta_init_bias: float = -5.0,
        beta_normalize: Optional[str] = None,  # None | 'hw' | 'sqrt_hw'
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.W_latent = W_latent
        self.use_recurrence = use_recurrence
        self.conv_kernel = conv_kernel
        self.beta_normalize = beta_normalize

        # Reuse names q / k / v / o so a Wan-pretrained checkpoint loads cleanly.
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        if qk_norm:
            WanRMSNorm, _ = _try_import_wan()
            assert WanRMSNorm is not None, (
                "WanRMSNorm needed for qk_norm=True; install Wan2.1 or pass qk_norm=False"
            )
            self.norm_q = WanRMSNorm(dim, eps=eps)
            self.norm_k = WanRMSNorm(dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        # Causal short conv along the temporal axis only.  Implemented as a
        # depthwise Conv1d acting along T after we reshape `[B, T, HW, dim]`
        # to `[B*HW, dim, T]`.  Causal trim removes the right-padded outputs.
        if conv_kernel > 1:
            self.conv_q = nn.Conv1d(
                dim, dim, kernel_size=conv_kernel, padding=conv_kernel - 1, groups=dim
            )
            self.conv_k = nn.Conv1d(
                dim, dim, kernel_size=conv_kernel, padding=conv_kernel - 1, groups=dim
            )
            # init to identity (delta function) — disables the conv at start so
            # Wan-pretrained behavior is preserved.
            with torch.no_grad():
                self.conv_q.weight.zero_()
                self.conv_k.weight.zero_()
                self.conv_q.weight[:, :, 0] = 1.0
                self.conv_k.weight[:, :, 0] = 1.0
                if self.conv_q.bias is not None:
                    self.conv_q.bias.zero_()
                    self.conv_k.bias.zero_()
        else:
            self.conv_q = None
            self.conv_k = None

        # α: per-(frame, head) gate.  Project from spatial-mean-pooled token.
        self.W_alpha = nn.Linear(dim, num_heads, bias=True)
        nn.init.zeros_(self.W_alpha.weight)
        nn.init.constant_(self.W_alpha.bias, alpha_init_bias)

        # β: per-(token, head) gate.
        self.W_beta = nn.Linear(dim, num_heads, bias=True)
        nn.init.zeros_(self.W_beta.weight)
        nn.init.constant_(self.W_beta.bias, beta_init_bias)

    @staticmethod
    def _split_seq_to_grid(x: torch.Tensor, grid_sizes: torch.Tensor):
        """Split flat [B, L, ...] into [B, F, HW, ...] using the per-sample (F, H, W)."""
        # WanModel feeds the same grid for every sample in a batch in T2V; we
        # check that and reshape directly.  Mixed grid sizes within a batch
        # would need padding logic — flag if encountered.
        gs = grid_sizes.tolist()
        f0, h0, w0 = gs[0]
        for f, h, w in gs:
            assert (f, h, w) == (f0, h0, w0), (
                "BGDA requires uniform grid_sizes within a batch; got mixed sizes."
            )
        F_, HW = f0, h0 * w0
        B = x.shape[0]
        return x.view(B, F_, HW, *x.shape[2:]), F_, HW

    def _maybe_conv(self, x: torch.Tensor, conv: Optional[nn.Conv1d]) -> torch.Tensor:
        """Apply causal short conv along temporal axis to [B, F, HW, dim]."""
        if conv is None:
            return x
        B, F_, HW, D = x.shape
        # [B, F, HW, D] → [B*HW, D, F]
        x_t = x.permute(0, 2, 3, 1).reshape(B * HW, D, F_)
        y = conv(x_t)              # right-padded; trim
        y = y[..., :F_]            # causal trim
        # back to [B, F, HW, D]
        y = y.reshape(B, HW, D, F_).permute(0, 3, 1, 2).contiguous()
        return y

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Args:
          x          : [B, L, dim]     L = F * H * W
          seq_lens   : [B]              (used by Wan flash_attention only — unused here)
          grid_sizes : [B, 3]          (F, H, W) per sample
          freqs      : RoPE complex tensor as in Wan's `rope_apply`.
        Returns:
          y          : [B, L, dim]
        """
        B, L, D = x.shape
        H, dh = self.num_heads, self.head_dim

        # Project Q/K/V (RMS-norm Q/K matches Wan defaults).
        q = self.norm_q(self.q(x))                  # [B, L, dim]
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # Reshape to [B, F, HW, dim] so we can apply temporal conv and gate
        # projections per frame/token.
        x_grid, F_, HW = self._split_seq_to_grid(x, grid_sizes)
        q_grid = q.view(B, F_, HW, D)
        k_grid = k.view(B, F_, HW, D)
        v_grid = v.view(B, F_, HW, D)

        # Causal short conv on Q, K (V untouched, plan §2.2).
        q_grid = self._maybe_conv(q_grid, self.conv_q)
        k_grid = self._maybe_conv(k_grid, self.conv_k)

        # Reshape into multi-head layout [B, F, HW, H, d].
        q_grid = q_grid.view(B, F_, HW, H, dh)
        k_grid = k_grid.view(B, F_, HW, H, dh)
        v_grid = v_grid.view(B, F_, HW, H, dh)

        # L2 normalize Q, K per head (V untouched).
        q_grid = F.normalize(q_grid, p=2, dim=-1, eps=self.eps)
        k_grid = F.normalize(k_grid, p=2, dim=-1, eps=self.eps)

        # Apply 3D RoPE via Wan's `rope_apply` if freqs is provided.  Wan's
        # rope_apply expects [B, L, H, d] flattened over (F, H, W) — we flatten
        # back to [B, L, H, d] and reuse it.  RoPE preserves L2 norm.
        if freqs is not None:
            _, rope_apply_fn = _try_import_wan()
            if rope_apply_fn is not None:
                q_flat = q_grid.view(B, L, H, dh)
                k_flat = k_grid.view(B, L, H, dh)
                q_flat = rope_apply_fn(q_flat, grid_sizes, freqs)
                k_flat = rope_apply_fn(k_flat, grid_sizes, freqs)
                q_grid = q_flat.view(B, F_, HW, H, dh)
                k_grid = k_flat.view(B, F_, HW, H, dh)

        # Gate projections.  α from spatial-mean per frame; β from per-token.
        x_pooled_per_frame = x_grid.mean(dim=2)               # [B, F, D]
        alpha = torch.sigmoid(self.W_alpha(x_pooled_per_frame))  # [B, F, H]
        beta = torch.sigmoid(self.W_beta(x_grid))             # [B, F, HW, H]

        # Pad F if it doesn't divide W_latent evenly.  Pad with zeros (kvs)
        # which are ignored when use_recurrence=False, and leave a no-op state
        # update when β=0 so it's still safe with recurrence on.  We pad on the
        # right and trim outputs afterwards.
        pad_F = (-F_) % self.W_latent
        if pad_F > 0:
            zeros_t = torch.zeros(B, pad_F, HW, H, dh, device=q_grid.device, dtype=q_grid.dtype)
            zeros_a = torch.zeros(B, pad_F, H, device=alpha.device, dtype=alpha.dtype)
            zeros_b = torch.zeros(B, pad_F, HW, H, device=beta.device, dtype=beta.dtype)
            q_grid = torch.cat([q_grid, zeros_t], dim=1)
            k_grid = torch.cat([k_grid, zeros_t], dim=1)
            v_grid = torch.cat([v_grid, zeros_t], dim=1)
            alpha = torch.cat([alpha, zeros_a], dim=1)
            beta = torch.cat([beta, zeros_b], dim=1)

        out = bgda_block_attention_reference(
            q_grid, k_grid, v_grid, alpha, beta,
            W_latent=self.W_latent,
            use_recurrence=self.use_recurrence,
            beta_normalize=self.beta_normalize,
        )                                                       # [B, F+pad, HW, H, d]
        if pad_F > 0:
            out = out[:, :F_]                                    # trim padding

        # Merge head + flatten back to [B, L, dim].
        out = out.reshape(B, L, H * dh)
        return self.o(out)
