"""Stage A1 — attention-transfer training (LoLCATs-style).

Plan v1 §3.1 sub-A1.

Procedure:
  1. Build a `WanModel` from pretrained weights (the *teacher*).
  2. Deep-copy and replace `WanSelfAttention` modules with
     `BGDABlockAttention(use_recurrence=False)` in the *student*.
  3. Freeze student's backbone; only NEW BGDA modules (conv_q, conv_k,
     W_alpha, W_beta) are trainable.
  4. For each training batch of latent video:
       - Run teacher's `WanModel` and capture per-layer self-attn outputs (hooks).
       - Run student's `WanModel` and capture per-layer self-attn outputs.
       - Loss = mean over layers of MSE(student_self_attn_out, teacher_self_attn_out).
       - Backward + step.
  5. Per-layer cosine similarity is logged; go/no-go threshold ≥0.95 averaged.

This file contains the LOSS LOGIC and HOOK INSTRUMENTATION; the dataloader
and outer training loop live in `train_video_scripts/` once data is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionTransferStats:
    """Per-step stats from one attention-transfer forward+backward."""

    loss: float
    per_layer_mse: List[float]
    per_layer_cossim: List[float]


class _AttnOutputCapture:
    """Forward-hook helper that captures the OUTPUT of a self-attn submodule
    on each forward call.  We register one hook per layer.
    """

    def __init__(self):
        self.outputs: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, model: nn.Module, attr_name: str = "self_attn") -> None:
        """Walk model, attach a hook on every submodule whose final attribute
        is `attr_name`."""
        def make_hook(path: str):
            def hook(_module, _inputs, output):
                self.outputs[path] = output
            return hook

        for path, mod in model.named_modules():
            for child_name, child in mod.named_children():
                if child_name == attr_name:
                    full = f"{path}.{child_name}" if path else child_name
                    self._handles.append(child.register_forward_hook(make_hook(full)))

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def attention_transfer_loss(
    teacher_outputs: Dict[str, torch.Tensor],
    student_outputs: Dict[str, torch.Tensor],
    *,
    reduction: str = "mean_over_layers",
    cossim_weight: float = 0.0,
    use_normalized_mse: bool = False,
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Per-layer attention-transfer loss between teacher and student self-attn
    outputs.

    Loss components per layer:
      - `MSE_raw`             : `mean((student − teacher)^2)`
      - `MSE_normalized`      : `mean(((student − teacher) / (‖teacher‖ + ε))^2)`
      - `1 − cossim`          : `1 − mean(cos(student, teacher))` per (batch, token)

    The Stage-A1 smoke (JID 7721177, scripts/bgda_stage_a1_train_smoke.py)
    found that pure raw MSE on Wan attention outputs is dominated by
    *magnitude* error: optimizer drops loss 47 % in 5 steps but cossim
    regresses (4 layers: 0.27 → 0.20; 30 layers: 0.07 → 0.03). Direction
    stays mis-aligned because magnitude error is so much larger.

    Mitigations exposed via flags:
      - `cossim_weight > 0`   adds `(1 - cossim)` term.  λ ≈ 1.0 typically
                              balances against MSE once magnitudes are large.
      - `use_normalized_mse`  divides residual by `‖teacher‖` per token so
                              MSE no longer scales with hidden magnitude.

    For Stage A1 we recommend `cossim_weight=1.0, use_normalized_mse=True`
    (LoLCATs-style direction-first transfer), then anneal `cossim_weight`
    down later.

    Returns:
      loss              : scalar tensor, sum over layers per `reduction`
      per_layer_mse     : python floats, `MSE_raw` per layer (for logging)
      per_layer_cossim  : python floats, mean cosine sim per layer
    """
    # The student may have its blocks wrapped in `_CheckpointedBlock`, which
    # injects an extra `.block` segment in the dotted module path
    # (e.g. `blocks.0.self_attn` → `blocks.0.block.self_attn`). Normalize by
    # stripping any `.block.` segment so teacher↔student paths align.
    def _norm(k: str) -> str:
        return k.replace(".block.self_attn", ".self_attn") \
                .replace(".block.cross_attn", ".cross_attn")

    student_outputs = {_norm(k): v for k, v in student_outputs.items()}
    teacher_outputs = {_norm(k): v for k, v in teacher_outputs.items()}

    keys = sorted(set(teacher_outputs) & set(student_outputs))
    if not keys:
        raise RuntimeError(
            "attention_transfer_loss: no overlapping self_attn paths; "
            f"teacher had {sorted(teacher_outputs)[:3]}..., "
            f"student had {sorted(student_outputs)[:3]}..."
        )

    per_layer_loss_t: List[torch.Tensor] = []
    per_layer_mse_log: List[torch.Tensor] = []
    per_layer_cossim: List[float] = []
    for k in keys:
        t = teacher_outputs[k].float()
        s = student_outputs[k].float()
        diff = s - t

        mse_raw = diff.pow(2).mean()
        per_layer_mse_log.append(mse_raw)

        if use_normalized_mse:
            t_norm = t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            mse_term = (diff / t_norm).pow(2).mean()
        else:
            mse_term = mse_raw

        # cosine sim per (batch, token); reduce to mean.  Differentiable.
        cos = F.cosine_similarity(
            s.flatten(0, -2), t.flatten(0, -2), dim=-1
        )
        cossim_mean_t = cos.mean()
        per_layer_cossim.append(float(cossim_mean_t.item()))

        layer_loss = mse_term + cossim_weight * (1.0 - cossim_mean_t)
        per_layer_loss_t.append(layer_loss)

    loss_t = torch.stack(per_layer_loss_t)
    if reduction == "mean_over_layers":
        loss = loss_t.mean()
    elif reduction == "sum":
        loss = loss_t.sum()
    else:
        raise ValueError(f"unknown reduction={reduction!r}")
    return loss, [float(m.item()) for m in per_layer_mse_log], per_layer_cossim


def step_attention_transfer(
    teacher: nn.Module,
    student: nn.Module,
    teacher_capture: _AttnOutputCapture,
    student_capture: _AttnOutputCapture,
    *args,
    **kwargs,
) -> AttentionTransferStats:
    """One forward step that returns a loss + diagnostics.  No optimizer call —
    leave that to the outer loop so callers can mix in other losses (e.g. the
    diffusion loss in sub-A2)."""
    teacher_capture.clear()
    student_capture.clear()
    with torch.no_grad():
        _ = teacher(*args, **kwargs)
    _ = student(*args, **kwargs)
    loss, per_l_mse, per_l_cos = attention_transfer_loss(
        teacher_capture.outputs, student_capture.outputs
    )
    return AttentionTransferStats(
        loss=float(loss.item()),
        per_layer_mse=per_l_mse,
        per_layer_cossim=per_l_cos,
    ), loss


def make_capture_pair(teacher: nn.Module, student: nn.Module) -> Tuple[_AttnOutputCapture, _AttnOutputCapture]:
    """Convenience: register hooks on both models' self-attn submodules."""
    tcap = _AttnOutputCapture()
    scap = _AttnOutputCapture()
    tcap.register(teacher, attr_name="self_attn")
    scap.register(student, attr_name="self_attn")
    return tcap, scap
