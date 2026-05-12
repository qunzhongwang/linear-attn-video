"""Smoke test for `BGDABlockAttention` — Wan-signature drop-in.

Doesn't depend on Wan being importable (we pass `qk_norm=False` to skip
WanRMSNorm and `freqs=None` to skip rope_apply).  This just sanity-checks
shape / forward / backward for the wrapper module.
"""
import torch

from spec.bgda.bgda_attention import BGDABlockAttention


def test_module_forward_shape_no_qk_norm_no_rope():
    B, F_, H_, W_ = 1, 4, 8, 8
    L = F_ * H_ * W_
    dim = 64
    num_heads = 4
    layer = BGDABlockAttention(
        dim=dim, num_heads=num_heads, qk_norm=False, eps=1e-6,
        W_latent=2, use_recurrence=False, conv_kernel=3,
    )
    x = torch.randn(B, L, dim, dtype=torch.float32)
    grid_sizes = torch.tensor([[F_, H_, W_]])
    seq_lens = torch.tensor([L])
    y = layer(x, seq_lens, grid_sizes, freqs=None)
    assert y.shape == (B, L, dim), f"got {y.shape}"
    loss = y.sum()
    loss.backward()
    # Check at least one trainable param got a non-zero grad.
    any_grad = any((p.grad is not None and p.grad.abs().sum() > 0) for p in layer.parameters())
    assert any_grad


def test_module_use_recurrence_grad():
    B, F_, H_, W_ = 1, 6, 4, 4
    L = F_ * H_ * W_
    dim = 32
    num_heads = 4
    layer = BGDABlockAttention(
        dim=dim, num_heads=num_heads, qk_norm=False, eps=1e-6,
        W_latent=3, use_recurrence=True, conv_kernel=1,  # disable conv for purity
    )
    x = torch.randn(B, L, dim, dtype=torch.float32)
    grid_sizes = torch.tensor([[F_, H_, W_]])
    seq_lens = torch.tensor([L])
    y = layer(x, seq_lens, grid_sizes, freqs=None)
    assert y.shape == (B, L, dim)
    (y ** 2).mean().backward()
    # gates should receive gradient when recurrence is on
    assert layer.W_alpha.weight.grad is not None
    assert layer.W_beta.weight.grad is not None


def test_module_handles_F_not_divisible_by_W_latent():
    # F=5, W_latent=2 → padded to 6 internally, output trimmed back to F=5.
    B, F_, H_, W_ = 1, 5, 4, 4
    L = F_ * H_ * W_
    dim = 32
    layer = BGDABlockAttention(
        dim=dim, num_heads=4, qk_norm=False,
        W_latent=2, use_recurrence=True, conv_kernel=1,
    )
    x = torch.randn(B, L, dim)
    grid_sizes = torch.tensor([[F_, H_, W_]])
    y = layer(x, torch.tensor([L]), grid_sizes, freqs=None)
    assert y.shape == (B, L, dim)
