"""Integration helpers for swapping Wan2.1's `WanSelfAttention` modules with
`BGDABlockAttention` in a loaded `WanModel`, copying compatible weights, and
freezing the backbone for Stage A1 (attention transfer).

Usage (Stage A):

    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
    )

    cfg = WAN_CONFIGS['t2v-1.3B']
    model = WanModel.from_pretrained(<path-to-Wan2.1-T2V-1.3B-folder>)
    swap_wan_self_attention_to_bgda(
        model,
        W_latent=3,
        use_recurrence=False,        # Stage A: bidirectional, no recurrence
        conv_kernel=3,
    )
    freeze_wan_backbone_except_bgda_new(model)

    # Now `model` has BGDABlockAttention in every WanAttentionBlock.
    # All non-BGDA parameters are `requires_grad=False`.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .bgda_attention import BGDABlockAttention


# Names of submodules we add that are NEW relative to WanSelfAttention.  These
# are the trainable parameters in Stage A1 along with any LoRA adapters
# attached via `attach_lora_to_qkvo`.
BGDA_NEW_SUBMODULE_NAMES = ("conv_q", "conv_k", "W_alpha", "W_beta")
BGDA_LORA_SUBMODULE_NAMES = ("lora_A", "lora_B")


def _copy_weights_from_wan(src: nn.Module, dst: BGDABlockAttention) -> None:
    """Copy q/k/v/o + qk_norm weights from a `WanSelfAttention` into a
    `BGDABlockAttention`.  Both modules use the same names (`q`, `k`, `v`, `o`,
    `norm_q`, `norm_k`) so the copy is straightforward.

    Raises if shapes mismatch (a tripwire — e.g. if Wan ever changes its API).
    """
    name_pairs = [("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")]
    if dst.qk_norm:
        name_pairs += [("norm_q", "norm_q"), ("norm_k", "norm_k")]
    for s_name, d_name in name_pairs:
        s_mod = getattr(src, s_name)
        d_mod = getattr(dst, d_name)
        with torch.no_grad():
            for (key_s, p_s), (key_d, p_d) in zip(
                s_mod.state_dict().items(), d_mod.state_dict().items()
            ):
                if p_s.shape != p_d.shape:
                    raise RuntimeError(
                        f"shape mismatch copying {s_name}.{key_s} -> {d_name}.{key_d}: "
                        f"{tuple(p_s.shape)} vs {tuple(p_d.shape)}"
                    )
                p_d.copy_(p_s)


def swap_wan_self_attention_to_bgda(
    wan_model: nn.Module,
    *,
    W_latent: int = 3,
    use_recurrence: bool = False,
    conv_kernel: int = 3,
    alpha_init_bias: float = 5.0,
    beta_init_bias: float = -5.0,
    beta_normalize: Optional[str] = None,
    only_layers: Optional[List[int]] = None,
    lora_rank: int = 0,
    lora_alpha: Optional[float] = None,
    lora_targets: tuple = ("q", "k", "v", "o"),
) -> List[str]:
    """Walk `wan_model` and replace every `WanSelfAttention` submodule with a
    new `BGDABlockAttention` of matching dim/num_heads/qk_norm/eps. Returns the
    list of dotted module paths that were swapped.

    `only_layers` lets us swap a subset (e.g. for predicate-B hybrid attention
    with 6/30 softmax layers preserved).
    """
    # Late-import to avoid forcing `wan` package import (which calls
    # torch.cuda.current_device) on test machines.
    try:
        from wan.modules.model import WanSelfAttention
    except Exception as e:
        raise RuntimeError(
            "Wan2.1 not importable — install Wan or add /home/qw3460/wp/Wan2.1 "
            f"to PYTHONPATH. underlying error: {e!r}"
        )

    swapped: List[str] = []
    # First pass: collect (parent_module, attr_name, child) for self-attn modules.
    targets = []
    for path, mod in wan_model.named_modules():
        # find the parent's `self_attn` attribute that is a WanSelfAttention.
        for attr_name, child in mod.named_children():
            if isinstance(child, WanSelfAttention):
                # Distinguish self-attn from cross-attn — both subclass
                # WanSelfAttention but are stored under different attribute
                # names in WanAttentionBlock (`self_attn` vs `cross_attn`).
                if attr_name != "self_attn":
                    continue
                full_path = f"{path}.{attr_name}" if path else attr_name
                targets.append((full_path, mod, attr_name, child))

    if only_layers is not None:
        keep_idx = set(only_layers)
        # Wan layers are addressed as `blocks.<idx>.self_attn` — extract idx.
        filtered = []
        for full_path, parent, attr, child in targets:
            try:
                idx = int(full_path.split(".")[1])
            except (IndexError, ValueError):
                continue
            if idx in keep_idx:
                filtered.append((full_path, parent, attr, child))
        targets = filtered

    for full_path, parent, attr, child in targets:
        new_attn = BGDABlockAttention(
            dim=child.dim,
            num_heads=child.num_heads,
            window_size=child.window_size,
            qk_norm=child.qk_norm,
            eps=child.eps,
            W_latent=W_latent,
            use_recurrence=use_recurrence,
            conv_kernel=conv_kernel,
            alpha_init_bias=alpha_init_bias,
            beta_init_bias=beta_init_bias,
            beta_normalize=beta_normalize,
        ).to(next(child.parameters()).device, dtype=next(child.parameters()).dtype)
        _copy_weights_from_wan(child, new_attn)
        if lora_rank > 0:
            from .lora import attach_lora_to_qkvo
            attach_lora_to_qkvo(
                new_attn, rank=lora_rank, alpha=lora_alpha, targets=lora_targets
            )
        setattr(parent, attr, new_attn)
        swapped.append(full_path)

    return swapped


def is_bgda_new_param(name: str) -> bool:
    """Predicate: True iff `name` is a parameter that did NOT exist in Wan
    before BGDA swap.  Includes the BGDA-new submodules AND any LoRA adapter
    parameters attached via `attach_lora_to_qkvo`."""
    parts = name.split(".")
    if any(p in BGDA_NEW_SUBMODULE_NAMES for p in parts):
        return True
    if any(p in BGDA_LORA_SUBMODULE_NAMES for p in parts):
        return True
    return False


def freeze_wan_backbone_except_bgda_new(model: nn.Module) -> int:
    """Set `requires_grad=False` on every parameter that is NOT a BGDA-new
    parameter.  Returns count of trainable params remaining."""
    count_trainable = 0
    for name, param in model.named_parameters():
        if is_bgda_new_param(name):
            param.requires_grad_(True)
            count_trainable += param.numel()
        else:
            param.requires_grad_(False)
    return count_trainable


class _CheckpointedBlock(nn.Module):
    """Wrap a `WanAttentionBlock` so its forward is run under
    `torch.utils.checkpoint.checkpoint`, trading compute for activation memory.

    `use_reentrant=False` is the modern path that doesn't require any state
    quirks. Forward args/kwargs are passed through unchanged.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        from torch.utils.checkpoint import checkpoint
        if self.training:
            # Checkpointing only meaningful in train mode (eval doesn't need
            # backward graph).  But still apply uniformly to be safe.
            def fn(*a):
                return self.block(*a, **kwargs)
            return checkpoint(fn, *args, use_reentrant=False)
        return self.block(*args, **kwargs)


def enable_gradient_checkpointing(wan_model: nn.Module) -> int:
    """Wrap every WanModel block in `_CheckpointedBlock` so backward through
    the student fits in memory.  Returns the count of blocks wrapped."""
    if not hasattr(wan_model, "blocks"):
        raise RuntimeError("model has no .blocks attribute — not a WanModel?")
    n = 0
    new_blocks = nn.ModuleList()
    for block in wan_model.blocks:
        if isinstance(block, _CheckpointedBlock):
            new_blocks.append(block)
            continue
        new_blocks.append(_CheckpointedBlock(block))
        n += 1
    wan_model.blocks = new_blocks
    return n


def bgda_new_param_groups(model: nn.Module, lr: float, weight_decay: float = 0.0):
    """Return torch.optim parameter groups for Stage A1: only BGDA-new params."""
    return [{
        "params": [p for n, p in model.named_parameters()
                   if is_bgda_new_param(n) and p.requires_grad],
        "lr": lr,
        "weight_decay": weight_decay,
    }]
