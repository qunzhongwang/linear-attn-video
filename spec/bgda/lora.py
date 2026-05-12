"""LoRA adapter for `nn.Linear` modules in BGDA Stage A.

Plan v1 §3.1 sub-A2 prescribes "LoRA adapters on `{W_Q, W_K, W_V, W_O}` of
every attention layer (rank=64)". Stage-A1 capacity smoke (JID 7721177,
JID 7721359) showed that without these adapters, the only trainable params
(conv_q + conv_k = ~4.6 K/layer) are insufficient to reshape Q/K and bring
linear-attention output close to softmax-attention output.

We add LoRA earlier — usable in both sub-A1 (frozen base + LoRA + new BGDA
modules) and sub-A2 (LoRA + RF diffusion loss).

LoRA math:  Linear(x) = base(x) + B(A(x)) * (alpha / rank)
  - `base`  : original `nn.Linear(in, out)`, frozen.
  - `A`     : `nn.Linear(in, rank, bias=False)`, init Kaiming.
  - `B`     : `nn.Linear(rank, out, bias=False)`, init zeros (so the residual
              starts as the identity).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in wrapper around an existing `nn.Linear` adding a low-rank
    residual `B @ A`. The base linear's weights are frozen; A,B are trainable.
    """

    def __init__(self, base: nn.Linear, rank: int = 64, alpha: float = 64.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        in_f, out_f = base.in_features, base.out_features
        # Trainable adapter
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        # Provide read-only access for code that introspects `.weight` (e.g.
        # the integration test's weight-equality check).  Returns the BASE
        # weight; the LoRA delta is held in lora_A/lora_B.
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.alpha / self.rank
        return self.base(x) + self.lora_B(self.lora_A(x)) * scale


def attach_lora_to_qkvo(
    bgda_attn: nn.Module,
    *,
    rank: int = 64,
    alpha: float | None = None,
    targets: tuple = ("q", "k", "v", "o"),
) -> int:
    """Replace `bgda_attn.{q,k,v,o}` (each an `nn.Linear`) with `LoRALinear`
    wrappers.  Returns the count of LoRA-trainable params added.

    Default `alpha = rank` so `scale = alpha / rank = 1.0`.  Override for
    different scaling.
    """
    if alpha is None:
        alpha = float(rank)
    n_trainable = 0
    for name in targets:
        if not hasattr(bgda_attn, name):
            continue
        base = getattr(bgda_attn, name)
        if not isinstance(base, nn.Linear):
            continue
        # If already wrapped, skip.
        if isinstance(base, LoRALinear):
            continue
        wrapped = LoRALinear(base, rank=rank, alpha=alpha)
        # Place on same device + dtype as base
        wrapped.lora_A = wrapped.lora_A.to(base.weight.device, dtype=base.weight.dtype)
        wrapped.lora_B = wrapped.lora_B.to(base.weight.device, dtype=base.weight.dtype)
        setattr(bgda_attn, name, wrapped)
        n_trainable += sum(p.numel() for p in wrapped.lora_A.parameters())
        n_trainable += sum(p.numel() for p in wrapped.lora_B.parameters())
    return n_trainable
