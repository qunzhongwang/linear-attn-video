"""GDSAAttention with delta_scale=0 (default init) must produce vanilla-SANA output exactly.

Pure-PyTorch test; doesn't import the real SANA blocks (which depend on mmcv shim, OK on this env).
"""
import pytest
import torch

from spec.gdsa.gdsa_attention import GDSAAttention


def test_init_recovery_no_rope(tiny_shape, device):
    B, T, C = tiny_shape["B"], tiny_shape["T"], tiny_shape["H"] * tiny_shape["D"]
    torch.manual_seed(0)
    x = torch.randn(B, T, C, dtype=torch.float64, device=device)

    m = GDSAAttention(
        in_dim=C, out_dim=C, heads=tiny_shape["H"], dim=tiny_shape["D"],
        qk_norm=False, gate_init_alpha=1.0, gate_init_beta=0.0, vanilla_residual=True,
    ).to(device=device, dtype=torch.float64)
    m.eval()

    # delta_scale starts at 0 → output is the vanilla branch only.
    assert float(m.delta_scale.item()) == 0.0
    out = m(x, chunk_sizes=tiny_shape["chunk_sizes"])

    # Reproduce the vanilla branch directly via vanilla_chunked + the same proj/qkv.
    qkv = m.qkv(x).reshape(B, T, 3, C)
    q, k, v = qkv.unbind(2)
    H, D = tiny_shape["H"], tiny_shape["D"]
    q_kf = torch.relu(q.reshape(B, T, H, D).permute(0, 2, 1, 3).contiguous())
    k_kf = torch.relu(k.reshape(B, T, H, D).permute(0, 2, 1, 3).contiguous())
    v_h = v.reshape(B, T, H, D).permute(0, 2, 1, 3).contiguous()
    from spec.gdsa.chunked import vanilla_chunked
    expected, _, _ = vanilla_chunked(q_kf, k_kf, v_h, chunk_sizes=tiny_shape["chunk_sizes"])
    expected = m.proj(expected.permute(0, 2, 1, 3).reshape(B, T, C))

    diff = (out - expected).abs().max().item()
    assert diff < 1e-12, f"GDSA at init didn't reproduce vanilla SANA, diff={diff}"


def test_kv_cache_round_trip(tiny_shape, device):
    """kv_cache=[None,None,None] + save_kv_cache=True returns S/Z written to slots [0],[1]."""
    B, T, C = tiny_shape["B"], tiny_shape["T"], tiny_shape["H"] * tiny_shape["D"]
    torch.manual_seed(0)
    x = torch.randn(B, T, C, dtype=torch.float64, device=device)
    m = GDSAAttention(
        in_dim=C, out_dim=C, heads=tiny_shape["H"], dim=tiny_shape["D"],
        qk_norm=False, gate_init_alpha=1.0, gate_init_beta=0.0, vanilla_residual=True,
    ).to(device=device, dtype=torch.float64)
    m.eval()

    cache = [None, None, None]
    out, cache_out = m(x, chunk_sizes=tiny_shape["chunk_sizes"], save_kv_cache=True, kv_cache=cache)
    assert cache_out[0] is not None and cache_out[0].shape[-1] == cache_out[0].shape[-2] == tiny_shape["D"]
    assert cache_out[1] is not None and cache_out[1].shape == (B, tiny_shape["H"], 1, tiny_shape["D"])
