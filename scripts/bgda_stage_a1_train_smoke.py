"""Stage A1 attention-transfer training SMOKE on synthetic latents.

Purpose: prove that with a frozen-backbone student (only BGDA-new modules
trainable), running the attention-transfer MSE loss against a frozen Wan
teacher on synthetic data DECREASES per-layer attention MSE and raises mean
cossim from the ~0.07 baseline (see bgda_wan_smoke 7720153 for the untrained
30-layer numbers).

This is NOT real training — it uses synthetic Gaussian latents and a fixed
mini-batch repeated for K steps. We just want to confirm the loop is
plumbed correctly end-to-end (gradients flow, loss decreases) before
plugging in real OpenVid-1M latents.

Usage (from a SLURM gpu-test session or pli):

    python scripts/bgda_stage_a1_train_smoke.py \
      --n_layers 4 --n_steps 50 --lr 1e-3 --W_latent 3
"""
from __future__ import annotations

import argparse
import copy
import sys
import time

WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

import torch


def _build_synthetic_inputs(in_dim, text_dim, B=1, F_latent=20, H=30, W=52,
                             device="cuda", dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype, device=device)
    x = [rand(in_dim, F_latent, H, W) for _ in range(B)]
    t = torch.zeros(B, dtype=torch.long, device=device)
    context = [rand(64, text_dim) for _ in range(B)]
    p_h, p_w = 2, 2
    seq_len = F_latent * (H // p_h) * (W // p_w)
    return x, t, context, seq_len


def _load_real_latents(latents_dir, in_dim, text_dim, device, dtype, n_clips=1, seed=0):
    """Load `n_clips` real Wan-VAE latents from `latents_dir/<idx>.pt` files.

    Returns same (x, t, context, seq_len) tuple structure as the synthetic
    helper; context is still random (we don't have T5 hidden states cached
    for these clips, but cross-attention sees the same context for both
    teacher and student so the attention-transfer signal is unaffected).
    """
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"no latent .pt files under {latents_dir}")
    paths = paths[:n_clips]
    g = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype, device=device)

    xs = []
    F_, H_, W_ = None, None, None
    for p in paths:
        d = torch.load(p, map_location=device, weights_only=False)
        z = d["z"].to(device=device, dtype=dtype)        # [16, T, H, W]
        if z.shape[0] != in_dim:
            raise RuntimeError(
                f"latent {p} has C={z.shape[0]}; expected in_dim={in_dim}"
            )
        if F_ is None:
            F_, H_, W_ = z.shape[1], z.shape[2], z.shape[3]
        xs.append(z)
    B = len(xs)
    t = torch.zeros(B, dtype=torch.long, device=device)
    context = [rand(64, text_dim) for _ in range(B)]
    p_h, p_w = 2, 2
    seq_len = F_ * (H_ // p_h) * (W_ // p_w)
    return xs, t, context, seq_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--W_latent", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora_rank", type=int, default=64,
                   help="LoRA rank for {W_Q,W_K,W_V,W_O}; 0 disables.")
    p.add_argument("--grad_checkpointing", action="store_true",
                   help="Wrap student blocks in torch.utils.checkpoint to "
                        "trade compute for activation memory (needed for full "
                        "30-layer Wan-1.3B at 480P on H100/H200).")
    p.add_argument("--latents_dir", type=str, default=None,
                   help="If set, load real Wan-VAE latents from this dir "
                        "instead of using synthetic Gaussian inputs.")
    p.add_argument("--n_clips", type=int, default=1)
    p.add_argument("--cossim_weight", type=float, default=1.0,
                   help="Weight on (1 - cossim) loss term. 0 = pure MSE (was found "
                        "to regress cossim on synthetic data).")
    p.add_argument("--normalized_mse", action="store_true", default=True,
                   help="Divide residual by ‖teacher‖ per token before MSE "
                        "(removes magnitude-error dominance).")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"[a1] cuda: {torch.cuda.get_device_name(0)}")

    print("[a1] importing wan…")
    from wan.modules.model import WanModel  # noqa: E402

    print(f"[a1] loading WanModel from {args.ckpt}")
    teacher = WanModel.from_pretrained(args.ckpt)
    teacher.eval()
    teacher.to(device, dtype=torch.bfloat16)
    if args.n_layers and args.n_layers < len(teacher.blocks):
        teacher.blocks = teacher.blocks[: args.n_layers]
        print(f"[a1] truncated to first {args.n_layers} blocks for fast smoke")
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    print("[a1] building student via deep-copy + swap")
    student = copy.deepcopy(teacher)
    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
        bgda_new_param_groups,
    )
    swapped = swap_wan_self_attention_to_bgda(
        student, W_latent=args.W_latent, use_recurrence=False, conv_kernel=3,
        lora_rank=args.lora_rank,
    )
    print(f"[a1] swapped {len(swapped)} self_attn submodules")
    n_trainable = freeze_wan_backbone_except_bgda_new(student)
    print(f"[a1] trainable params: {n_trainable:,}")

    if args.grad_checkpointing:
        from spec.bgda.integration import enable_gradient_checkpointing
        n_ck = enable_gradient_checkpointing(student)
        print(f"[a1] enabled gradient checkpointing on {n_ck} blocks")

    student.train()  # student trainable modules in train mode

    from spec.bgda.training.stage_a1_attention_transfer import (
        make_capture_pair, attention_transfer_loss,
    )
    tcap, scap = make_capture_pair(teacher, student)

    in_dim = getattr(student, "in_dim", 16)
    text_dim = getattr(student, "text_dim", 4096)
    if args.latents_dir is not None:
        print(f"[a1] loading real latents from {args.latents_dir}")
        x, t, context, seq_len = _load_real_latents(
            args.latents_dir, in_dim, text_dim, device, torch.bfloat16,
            n_clips=args.n_clips, seed=args.seed,
        )
        print(f"[a1] loaded {len(x)} clips with shape={tuple(x[0].shape)}, seq_len={seq_len}")
    else:
        x, t, context, seq_len = _build_synthetic_inputs(
            in_dim, text_dim, B=1, F_latent=20, H=30, W=52,
            device=device, dtype=torch.bfloat16, seed=args.seed,
        )
        print(f"[a1] using synthetic Gaussian inputs (seq_len={seq_len})")

    # Optimizer over only BGDA-new params.
    groups = bgda_new_param_groups(student, lr=args.lr, weight_decay=0.0)
    opt = torch.optim.AdamW(groups[0]["params"], lr=args.lr, weight_decay=0.0)

    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    # Run teacher ONCE under no_grad to cache its self_attn outputs (synthetic
    # input is fixed across steps, so the teacher's outputs are constant).
    print("[a1] caching teacher outputs (synthetic input is fixed)")
    tcap.clear()
    with autocast, torch.no_grad():
        _ = teacher(x, t=t, context=context, seq_len=seq_len)
    teacher_outputs = {k: v.detach().clone() for k, v in tcap.outputs.items()}
    tcap.remove()  # don't need teacher hooks during student training

    losses = []
    cossim_means = []
    t0 = time.perf_counter()
    for step in range(args.n_steps):
        scap.clear()
        with autocast:
            _ = student(x, t=t, context=context, seq_len=seq_len)
        loss, per_l_mse, per_l_cos = attention_transfer_loss(
            teacher_outputs, scap.outputs,
            cossim_weight=args.cossim_weight,
            use_normalized_mse=args.normalized_mse,
        )
        opt.zero_grad()
        loss.backward()
        # Sanity: at least one trainable BGDA-new param has nonzero grad.
        if step == 0:
            grad_present = any(
                (p.grad is not None and p.grad.abs().sum().item() > 0)
                for p in opt.param_groups[0]["params"]
            )
            assert grad_present, "no gradient flowed into any BGDA-new param!"
        opt.step()
        losses.append(float(loss.item()))
        cossim_means.append(sum(per_l_cos) / len(per_l_cos))
        if step % max(1, args.n_steps // 10) == 0 or step == args.n_steps - 1:
            print(
                f"[a1] step {step:3d}: loss={loss.item():.4e}  "
                f"mean_cossim={cossim_means[-1]:+.4f}  "
                f"min_cossim={min(per_l_cos):+.4f}  "
                f"max_cossim={max(per_l_cos):+.4f}"
            )
    elapsed = time.perf_counter() - t0
    print(f"[a1] {args.n_steps} steps in {elapsed:.1f}s ({elapsed / args.n_steps * 1000:.1f} ms/step)")

    # Verdict
    loss_drop = losses[0] - losses[-1]
    cos_gain = cossim_means[-1] - cossim_means[0]
    print(f"[a1] loss   start={losses[0]:.3e}   end={losses[-1]:.3e}   drop={loss_drop:.3e}")
    print(f"[a1] cossim start={cossim_means[0]:+.4f} end={cossim_means[-1]:+.4f} gain={cos_gain:+.4f}")
    if loss_drop > 0 and cos_gain > 0:
        print("[a1] VERDICT: ✓ loss decreased AND mean cossim improved")
    elif loss_drop > 0:
        print("[a1] VERDICT: △ loss decreased but cossim didn't improve (may need longer run)")
    else:
        print("[a1] VERDICT: ✗ training did NOT decrease loss — investigate")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[a1] peak GPU memory: {peak:.0f} MiB")


if __name__ == "__main__":
    main()
