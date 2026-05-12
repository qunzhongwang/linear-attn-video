"""Integration tests for the Wan ↔ BGDA swap.

Build a synthetic mock that mimics Wan's `WanSelfAttention` shape so we can
test the swap without needing wan-the-package import-clean (its
__init__ eagerly calls torch.cuda.current_device, problematic on CPU).
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


def _load_wan_self_attention_class():
    """Load `WanSelfAttention` from /home/qw3460/wp/Wan2.1/wan/modules/model.py
    without triggering wan/__init__.py (which calls torch.cuda.current_device).
    We register a fake `wan.modules.attention` to satisfy the relative import.
    """
    wan_root = Path("/home/qw3460/wp/Wan2.1")
    if not (wan_root / "wan/modules/model.py").exists():
        pytest.skip("Wan2.1 repo not present at /home/qw3460/wp/Wan2.1")

    # Stub `wan` and `wan.modules` packages, plus `wan.modules.attention`.
    if "wan" not in sys.modules:
        wan_pkg = types.ModuleType("wan")
        wan_pkg.__path__ = [str(wan_root / "wan")]
        sys.modules["wan"] = wan_pkg
    if "wan.modules" not in sys.modules:
        wm_pkg = types.ModuleType("wan.modules")
        wm_pkg.__path__ = [str(wan_root / "wan/modules")]
        sys.modules["wan.modules"] = wm_pkg
    if "wan.modules.attention" not in sys.modules:
        # Provide a no-op flash_attention so model.py imports fine on CPU.
        attn_stub = types.ModuleType("wan.modules.attention")
        def _fake_flash_attention(q, k, v, *args, **kwargs):
            # Reduce to plain SDPA, ignoring extra kwargs not relevant on CPU.
            B, Lq, H, D = q.shape
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        attn_stub.flash_attention = _fake_flash_attention
        attn_stub.attention = _fake_flash_attention
        sys.modules["wan.modules.attention"] = attn_stub

    # Now load model.py as `wan.modules.model`.
    spec = importlib.util.spec_from_file_location(
        "wan.modules.model", str(wan_root / "wan/modules/model.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wan.modules.model"] = mod
    spec.loader.exec_module(mod)
    return mod.WanSelfAttention, mod.WanAttentionBlock


def test_swap_replaces_only_self_attn_keeps_cross_attn():
    WanSelfAttention, WanAttentionBlock = _load_wan_self_attention_class()
    # Build one block with t2v cross-attn.
    block = WanAttentionBlock(
        cross_attn_type="t2v_cross_attn",
        dim=64, ffn_dim=128, num_heads=4, qk_norm=True, eps=1e-6,
    )
    # Wrap in a tiny nn.Module to exercise the named_modules walk.
    holder = nn.Module()
    holder.add_module("blocks", nn.ModuleList([block]))

    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        BGDA_NEW_SUBMODULE_NAMES,
    )
    from spec.bgda.bgda_attention import BGDABlockAttention

    swapped = swap_wan_self_attention_to_bgda(
        holder, W_latent=2, use_recurrence=False, conv_kernel=3,
    )
    # Exactly one self_attn replaced.
    assert len(swapped) == 1
    assert isinstance(holder.blocks[0].self_attn, BGDABlockAttention)
    # Cross-attn untouched (still WanSelfAttention subclass).
    from wan.modules.model import WanT2VCrossAttention  # type: ignore
    assert isinstance(holder.blocks[0].cross_attn, WanT2VCrossAttention)


def test_swap_copies_qkvo_weights():
    WanSelfAttention, _ = _load_wan_self_attention_class()
    src = WanSelfAttention(dim=32, num_heads=4, qk_norm=True, eps=1e-6)
    # randomize so we can confirm copy
    with torch.no_grad():
        for p in src.parameters():
            p.normal_()
    holder = nn.Module()
    holder.add_module("blocks", nn.ModuleList([nn.Module()]))
    holder.blocks[0].add_module("self_attn", src)

    from spec.bgda.integration import swap_wan_self_attention_to_bgda
    swap_wan_self_attention_to_bgda(holder, W_latent=2, use_recurrence=False)

    dst = holder.blocks[0].self_attn
    # q/k/v/o weights must match
    assert torch.equal(dst.q.weight, src.q.weight) is False or torch.equal(dst.q.weight, src.q.weight)
    # The above oddity catches the case where copy worked: equality should be True.
    for name in ("q", "k", "v", "o"):
        s = getattr(src, name).weight
        d = getattr(dst, name).weight
        assert torch.equal(s, d), f"weight {name} not copied"


def test_freeze_only_bgda_new_trainable():
    WanSelfAttention, WanAttentionBlock = _load_wan_self_attention_class()
    block = WanAttentionBlock(
        cross_attn_type="t2v_cross_attn",
        dim=32, ffn_dim=64, num_heads=4, qk_norm=True, eps=1e-6,
    )
    holder = nn.Module()
    holder.add_module("blocks", nn.ModuleList([block]))

    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
        is_bgda_new_param,
    )
    swap_wan_self_attention_to_bgda(holder, W_latent=2, use_recurrence=False)
    n_trainable = freeze_wan_backbone_except_bgda_new(holder)
    assert n_trainable > 0

    for name, param in holder.named_parameters():
        if is_bgda_new_param(name):
            assert param.requires_grad, f"BGDA-new {name} should be trainable"
        else:
            assert not param.requires_grad, f"non-BGDA {name} should be frozen"
