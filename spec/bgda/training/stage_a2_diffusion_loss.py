"""Stage A2 — rectified-flow diffusion loss for the BGDA student.

Plan v1 §3.1 sub-A2: with LoRA on QKVO + new BGDA modules trainable,
train end-to-end on the standard rectified-flow objective:

    x_t = (1 − t) · x_0 + t · ε                  (linear interpolation noise)
    v_target = ε − x_0                            (RF velocity target)
    loss = mean_t [ ‖u_θ(x_t, t, c) − v_target‖² ]

Backbone is frozen; only `bgda_new` + `lora_*` params train.
This module exposes the loss; the outer training loop wires data + opt + step.

Usage (smoke):

    from spec.bgda.training.stage_a2_diffusion_loss import rf_loss_step
    loss, stats = rf_loss_step(student, x0, t_low=0.0, t_high=1.0, context=ctx, seq_len=L)
    loss.backward(); opt.step()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class RFLossStats:
    loss: float
    timestep_min: float
    timestep_max: float
    pred_norm: float
    target_norm: float
    # per-sample diagnostics (for bucketing in the training loop)
    t_per_sample: list = None
    loss_per_sample: list = None
    pred_target_cos: float = 0.0


def sample_rf_timesteps(B: int, *, t_low: float = 0.0, t_high: float = 1.0,
                         device: torch.device, dtype: torch.dtype = torch.float32):
    """Uniform `t` in [t_low, t_high] per sample."""
    return torch.rand(B, device=device, dtype=dtype) * (t_high - t_low) + t_low


def make_rf_input(x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear-interpolation noising: x_t = (1-t) x0 + t eps.

    `t` is a per-sample scalar; we broadcast across all non-batch axes.
    Both x0 and eps have layout `[C, F, H, W]`; we receive them as a list
    in Wan's per-sample list convention.
    """
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * eps


def rf_velocity_target(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Rectified-flow velocity target: v = eps - x0."""
    return eps - x0


def rf_loss_step(
    student,
    x0_list: List[torch.Tensor],
    *,
    t_low: float = 0.0,
    t_high: float = 1.0,
    context_list: List[torch.Tensor],
    seq_len: int,
    eps_list: Optional[List[torch.Tensor]] = None,
    seed: Optional[int] = None,
):
    """Single rectified-flow training step.  Returns (loss_tensor, stats).

    Parameters mirror Wan's per-sample list convention:
      x0_list      : list of `[C_in, F_lat, H_lat, W_lat]` clean latents.
      context_list : list of `[L_text, C_text]` T5 hidden states (per sample).
      seq_len      : Wan's largest patched-sequence length across the batch.
      eps_list     : optional fixed noise (for deterministic smoke); else
                     drawn from N(0, I) on the same device/dtype as x0.
    """
    device = x0_list[0].device
    dtype = x0_list[0].dtype
    B = len(x0_list)

    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    if eps_list is None:
        eps_list = [torch.randn(*x.shape, generator=g, dtype=dtype, device=device)
                    for x in x0_list]
    assert len(eps_list) == B

    t_per_sample = sample_rf_timesteps(B, t_low=t_low, t_high=t_high, device=device,
                                        dtype=torch.float32)

    # Build x_t and target per sample.
    x_t_list, v_target_list = [], []
    for x0, eps, t_i in zip(x0_list, eps_list, t_per_sample):
        x_t_list.append(make_rf_input(x0.float(), eps.float(), t_i).to(dtype))
        v_target_list.append(rf_velocity_target(x0.float(), eps.float()).to(dtype))

    # Wan's WanModel.forward expects t as long timesteps in [0, 1000]
    # (sinusoidal_embedding_1d takes integer-ish positions).  We multiply
    # the [0,1] RF timestep by 1000 and round.
    t_long = (t_per_sample * 1000.0).round().clamp(0, 999).long()

    pred_list = student(x_t_list, t=t_long, context=context_list, seq_len=seq_len)

    # Per-sample MSE in float32 for numerical stability.
    losses, cos_sims = [], []
    pred_norms, tgt_norms = [], []
    for pred, tgt in zip(pred_list, v_target_list):
        pf = pred.float()
        tf = tgt.float()
        losses.append((pf - tf).pow(2).mean())
        pn = float(pf.norm().item())
        tn = float(tf.norm().item())
        pred_norms.append(pn)
        tgt_norms.append(tn)
        # cosine sim of flattened vectors
        denom = max(pn * tn, 1e-12)
        cos_sims.append(float((pf.flatten() @ tf.flatten()).item()) / denom)
    loss = torch.stack(losses).mean()

    stats = RFLossStats(
        loss=float(loss.item()),
        timestep_min=float(t_per_sample.min().item()),
        timestep_max=float(t_per_sample.max().item()),
        pred_norm=sum(pred_norms) / len(pred_norms),
        target_norm=sum(tgt_norms) / len(tgt_norms),
        t_per_sample=t_per_sample.detach().float().cpu().tolist(),
        loss_per_sample=[float(l.item()) for l in losses],
        pred_target_cos=sum(cos_sims) / len(cos_sims),
    )
    return loss, stats
