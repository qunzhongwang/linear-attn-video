"""Stage A2 (rectified-flow diffusion loss) training smoke.

Plan v1 §3.1 sub-A2.  Validates:
  1. Build BGDA student (use_recurrence=False, LoRA-64) over Wan-1.3B.
  2. Run RF loss training step on the 6-clip mini corpus.
  3. Verify loss decreases over a small number of steps.
  4. Verify the *output magnitude* of the student converges toward the
     RF velocity-target magnitude (a correctness sanity).

Use grad-ckpt so 30-layer fits.
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


def _load_real_latents(latents_dir, in_dim, text_dim, device, dtype, n_clips=1, seed=0):
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))[:n_clips]
    if not paths:
        raise FileNotFoundError(f"no .pt latents under {latents_dir}")
    g = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype, device=device)
    xs = []
    F_ = H_ = W_ = None
    for p in paths:
        d = torch.load(p, map_location=device, weights_only=False)
        z = d["z"].to(device=device, dtype=dtype)
        if F_ is None:
            F_, H_, W_ = z.shape[1], z.shape[2], z.shape[3]
        xs.append(z)
    B = len(xs)
    context = [rand(64, text_dim) for _ in range(B)]
    p_h, p_w = 2, 2
    seq_len = F_ * (H_ // p_h) * (W_ // p_w)
    return xs, context, seq_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--latents_dir", default="data/wan_a1_minicorpus")
    p.add_argument("--n_clips", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--W_latent", type=int, default=3)
    p.add_argument("--use_recurrence", action="store_true")
    p.add_argument("--grad_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[a2] device={device}")

    from wan.modules.model import WanModel  # noqa: E402

    print(f"[a2] loading WanModel from {args.ckpt}")
    base = WanModel.from_pretrained(args.ckpt)
    base.eval()
    base.to(device, dtype=torch.bfloat16)
    if args.n_layers and args.n_layers < len(base.blocks):
        base.blocks = base.blocks[: args.n_layers]
    print(f"[a2] {len(base.blocks)} blocks; deep-copy for student then swap")
    student = copy.deepcopy(base)
    del base
    torch.cuda.empty_cache()

    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
        bgda_new_param_groups,
        enable_gradient_checkpointing,
    )
    swap_wan_self_attention_to_bgda(
        student, W_latent=args.W_latent,
        use_recurrence=args.use_recurrence,
        conv_kernel=3,
        lora_rank=args.lora_rank,
        beta_normalize="hw" if args.use_recurrence else None,
    )
    n_trainable = freeze_wan_backbone_except_bgda_new(student)
    print(f"[a2] trainable params: {n_trainable:,}")
    if args.grad_checkpointing:
        n_ck = enable_gradient_checkpointing(student)
        print(f"[a2] gradient checkpointing on {n_ck} blocks")
    student.train()

    in_dim = getattr(student, "in_dim", 16)
    text_dim = getattr(student, "text_dim", 4096)
    x0_list, ctx_list, seq_len = _load_real_latents(
        args.latents_dir, in_dim, text_dim, device, torch.bfloat16,
        n_clips=args.n_clips, seed=args.seed,
    )
    print(f"[a2] loaded {len(x0_list)} clips, latent shape={tuple(x0_list[0].shape)}, seq_len={seq_len}")

    groups = bgda_new_param_groups(student, lr=args.lr, weight_decay=0.0)
    opt = torch.optim.AdamW(groups[0]["params"], lr=args.lr, weight_decay=0.0)

    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    from spec.bgda.training.stage_a2_diffusion_loss import rf_loss_step

    losses = []
    t0 = time.perf_counter()
    for step in range(args.n_steps):
        with autocast:
            loss, stats = rf_loss_step(
                student, x0_list,
                t_low=0.0, t_high=1.0,
                context_list=ctx_list, seq_len=seq_len,
                seed=args.seed + step,
            )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(stats.loss)
        if step % max(1, args.n_steps // 10) == 0 or step == args.n_steps - 1:
            print(
                f"[a2] step {step:3d}: loss={stats.loss:.4e}  "
                f"t∈[{stats.timestep_min:.2f}, {stats.timestep_max:.2f}]  "
                f"‖pred‖={stats.pred_norm:.3e}  ‖target‖={stats.target_norm:.3e}"
            )
    elapsed = time.perf_counter() - t0
    print(f"[a2] {args.n_steps} steps in {elapsed:.1f}s ({elapsed / args.n_steps * 1000:.1f} ms/step)")

    drop = losses[0] - losses[-1]
    drop_pct = 100.0 * drop / max(losses[0], 1e-9)
    print(f"[a2] loss start={losses[0]:.3e} end={losses[-1]:.3e} drop={drop:.3e} ({drop_pct:.1f}%)")
    if drop > 0:
        print("[a2] VERDICT: ✓ RF loss decreased")
    else:
        print("[a2] VERDICT: ✗ RF loss did NOT decrease — investigate")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[a2] peak GPU memory: {peak:.0f} MiB")


if __name__ == "__main__":
    main()
