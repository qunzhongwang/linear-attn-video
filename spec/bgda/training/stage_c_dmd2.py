"""Stage C — DMD2 distillation from Wan-14B teacher to BGDA student.

Plan v1 §3.3:
  - Generator G_θ          : BGDA-1.3B-Stage-B student, 4-step sampler.
  - Real-score s_real     : Wan-14B (frozen).  Predicts velocity v(x) given
                            (x_t, t, c).  No gradient needed through it.
  - Fake-score s_fake     : trainable copy of BGDA-1.3B (separate from G_θ).
                            Updated to track G_θ's distribution.
  - Self-forcing rollout  : during DMD training, history blocks are generated
                            by G_θ itself, so train-test alignment is exact.

The DMD2 distribution-matching gradient (Yin+ NeurIPS'24):

    grad_{x_t} D_KL(q‖p) ∝ ∇_{x_t} ( s_fake(x_t) − s_real(x_t) ) · v_pred

where v_pred is the prediction of G_θ.  We propagate the residual
( s_fake(x_t) − s_real(x_t) ) back through G_θ.

To keep this implementation auditable and unit-testable we expose two
discrete pieces:
  (1) `compute_score_residual(s_real, s_fake, x_t, t, c)`  — pure forward.
  (2) `dmd2_step(generator, score_real, score_fake, x_t, t, c, opt_g, opt_f)` —
      one full step including (a) generator distribution-matching grad,
      (b) fake-score regression update toward G's outputs.

Self-forcing wrapper (block-AR replay) lives in `self_forcing.py`.

The actual dataloader / VBench-prompt-driven inference loop is in
`scripts/bgda_stage_c_train_smoke.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class DMDStepStats:
    g_loss: float          # generator distribution-matching surrogate
    f_loss: float          # fake-score regression toward G's samples
    pred_norm: float       # mean ‖G(x_t)‖
    real_score_norm: float # mean ‖s_real(x_t)‖
    fake_score_norm: float # mean ‖s_fake(x_t)‖


def compute_score_residual(
    score_real,
    score_fake,
    x_t_list: List[torch.Tensor],
    t_long: torch.Tensor,
    context_list: List[torch.Tensor],
    seq_len: int,
    *,
    autocast_dtype: torch.dtype = torch.bfloat16,
):
    """Run both score models and return the per-sample residuals
    `s_fake(x_t) − s_real(x_t)`.  Real-score is frozen `no_grad`; fake-score
    is also `no_grad` here (we only use it as a target later in the step)."""
    autocast = torch.amp.autocast("cuda", dtype=autocast_dtype)
    with autocast, torch.no_grad():
        v_real = score_real(x_t_list, t=t_long, context=context_list, seq_len=seq_len)
    with autocast, torch.no_grad():
        v_fake = score_fake(x_t_list, t=t_long, context=context_list, seq_len=seq_len)
    return [(vf.float() - vr.float()) for vr, vf in zip(v_real, v_fake)]


def dmd2_generator_loss(
    generator,
    score_real,
    score_fake,
    x_t_list: List[torch.Tensor],
    t_long: torch.Tensor,
    context_list: List[torch.Tensor],
    seq_len: int,
    *,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generator's distribution-matching surrogate loss.

    The DMD2 recipe replaces the analytical KL gradient by a 'fake target'
    constructed as `target = stop_grad(G(x_t)) + (s_fake − s_real)`.  Then
    the surrogate loss is `MSE( G(x_t), target ) = -[ G · (s_fake − s_real) ]`
    plus constants — its gradient matches the desired distribution-matching
    direction.

    Reference: Yin et al. (DMD2, NeurIPS'24), Eq. 3.

    Returns the surrogate scalar loss (call .backward() on it).
    """
    autocast = torch.amp.autocast("cuda", dtype=autocast_dtype)
    # Forward G with grad
    with autocast:
        g_pred_list = generator(x_t_list, t=t_long, context=context_list, seq_len=seq_len)
    # Compute residual — no grad.
    residual_list = compute_score_residual(
        score_real, score_fake, x_t_list, t_long, context_list, seq_len,
        autocast_dtype=autocast_dtype,
    )
    # Surrogate target = stop_grad(G_pred) + residual
    target_list = [g.detach().float() + r for g, r in zip(g_pred_list, residual_list)]
    losses = []
    for g, t in zip(g_pred_list, target_list):
        losses.append((g.float() - t).pow(2).mean())
    return torch.stack(losses).mean()


def dmd2_fake_score_loss(
    score_fake,
    generator,
    x_t_list: List[torch.Tensor],
    t_long: torch.Tensor,
    context_list: List[torch.Tensor],
    seq_len: int,
    *,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Update fake-score to track G_θ's distribution.

    Standard score-matching: minimize MSE between s_fake(x_t) and the RF
    velocity of G's noisy samples.  Generator stays in `no_grad` here so
    only fake-score gets updated.
    """
    autocast = torch.amp.autocast("cuda", dtype=autocast_dtype)
    with autocast, torch.no_grad():
        g_pred_list = generator(x_t_list, t=t_long, context=context_list, seq_len=seq_len)
    with autocast:
        f_pred_list = score_fake(x_t_list, t=t_long, context=context_list, seq_len=seq_len)
    losses = []
    for f, g in zip(f_pred_list, g_pred_list):
        losses.append((f.float() - g.float().detach()).pow(2).mean())
    return torch.stack(losses).mean()


def dmd2_step(
    generator,
    score_real,
    score_fake,
    x_t_list: List[torch.Tensor],
    t_long: torch.Tensor,
    context_list: List[torch.Tensor],
    seq_len: int,
    opt_g,
    opt_f,
    *,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> DMDStepStats:
    """One DMD2 training step: generator step then fake-score step."""
    # Generator update (uses real-score and CURRENT fake-score, both frozen).
    g_loss = dmd2_generator_loss(
        generator, score_real, score_fake, x_t_list, t_long, context_list, seq_len,
        autocast_dtype=autocast_dtype,
    )
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    # Fake-score update on the SAME x_t (post-generator step would use new
    # generator outputs; either way is fine — DMD2 paper updates fake-score
    # against pre-step generator).
    f_loss = dmd2_fake_score_loss(
        score_fake, generator, x_t_list, t_long, context_list, seq_len,
        autocast_dtype=autocast_dtype,
    )
    opt_f.zero_grad()
    f_loss.backward()
    opt_f.step()

    return DMDStepStats(
        g_loss=float(g_loss.item()),
        f_loss=float(f_loss.item()),
        pred_norm=0.0,            # filled by caller if it tracks it
        real_score_norm=0.0,
        fake_score_norm=0.0,
    )
