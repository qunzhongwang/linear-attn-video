"""Stage C — DMD2 distillation training smoke.

Loads:
  - Wan-14B as `score_real` (teacher, frozen no-grad).
  - BGDA Stage-B-style student as `generator` (trainable: BGDA-new + LoRA).
  - Deep-copy of generator as `score_fake` (trainable: same set).

Runs `n_steps` of `dmd2_step` on the real-latent mini corpus and reports the
two losses + memory.

Wan-14B has 28 layers (vs 1.3B's 30 — slightly smaller).
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
    p.add_argument("--ckpt_1_3B", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--ckpt_14B",
                   default="/home/qw3460/qw3460-per/.cache/huggingface/Wan-AI/Wan2.1-T2V-14B")
    p.add_argument("--latents_dir", default="data/wan_a1_minicorpus")
    p.add_argument("--n_clips", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_layers_student", type=int, default=4,
                   help="Truncate student blocks for fast smoke. <=0 for full 30.")
    p.add_argument("--n_layers_teacher", type=int, default=4,
                   help="Truncate teacher blocks for fast smoke. <=0 for full 28.")
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--W_latent", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[c] device={device}")

    from wan.modules.model import WanModel  # noqa: E402

    print(f"[c] loading TEACHER Wan-14B from {args.ckpt_14B}")
    t0 = time.perf_counter()
    teacher = WanModel.from_pretrained(args.ckpt_14B)
    teacher.eval()
    teacher.to(device, dtype=torch.bfloat16)
    if args.n_layers_teacher > 0 and args.n_layers_teacher < len(teacher.blocks):
        teacher.blocks = teacher.blocks[: args.n_layers_teacher]
    for q in teacher.parameters():
        q.requires_grad_(False)
    print(f"[c] TEACHER {len(teacher.blocks)} blocks loaded in {time.perf_counter()-t0:.1f}s")

    print(f"[c] loading STUDENT BASE Wan-1.3B from {args.ckpt_1_3B}")
    t0 = time.perf_counter()
    base = WanModel.from_pretrained(args.ckpt_1_3B)
    base.eval()
    base.to(device, dtype=torch.bfloat16)
    if args.n_layers_student > 0 and args.n_layers_student < len(base.blocks):
        base.blocks = base.blocks[: args.n_layers_student]
    print(f"[c] STUDENT BASE {len(base.blocks)} blocks loaded in {time.perf_counter()-t0:.1f}s")

    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
        bgda_new_param_groups,
        enable_gradient_checkpointing,
    )

    print("[c] building generator (G_θ) and fake-score (s_fake) from base")
    generator = copy.deepcopy(base)
    score_fake = copy.deepcopy(base)
    del base
    torch.cuda.empty_cache()

    for student in (generator, score_fake):
        swap_wan_self_attention_to_bgda(
            student, W_latent=args.W_latent, use_recurrence=True,
            conv_kernel=3, lora_rank=args.lora_rank, beta_normalize="hw",
        )
        freeze_wan_backbone_except_bgda_new(student)
        enable_gradient_checkpointing(student)
        student.train()

    n_g = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    n_f = sum(p.numel() for p in score_fake.parameters() if p.requires_grad)
    print(f"[c] generator trainable: {n_g:,}   fake-score trainable: {n_f:,}")

    student_in_dim = getattr(generator, "in_dim", 16)
    text_dim = getattr(generator, "text_dim", 4096)
    x0_list, ctx_list, seq_len = _load_real_latents(
        args.latents_dir, student_in_dim, text_dim, device, torch.bfloat16,
        n_clips=args.n_clips, seed=args.seed,
    )
    print(f"[c] loaded {len(x0_list)} clips, latent shape={tuple(x0_list[0].shape)}")

    opt_g = torch.optim.AdamW(
        bgda_new_param_groups(generator, lr=args.lr)[0]["params"], lr=args.lr,
    )
    opt_f = torch.optim.AdamW(
        bgda_new_param_groups(score_fake, lr=args.lr)[0]["params"], lr=args.lr,
    )

    from spec.bgda.training.stage_c_dmd2 import dmd2_step

    g_losses, f_losses = [], []
    t0 = time.perf_counter()
    for step in range(args.n_steps):
        # Sample t for x_t the way RF does it. Using mid timesteps for now.
        eps_list = [torch.randn_like(x) for x in x0_list]
        t_per = torch.rand(len(x0_list), device=device) * 1.0
        x_t = []
        for x0, eps, ti in zip(x0_list, eps_list, t_per):
            xt = (1.0 - ti) * x0.float() + ti * eps.float()
            x_t.append(xt.to(torch.bfloat16))
        t_long = (t_per * 1000.0).round().clamp(0, 999).long()

        stats = dmd2_step(
            generator, teacher, score_fake,
            x_t, t_long, ctx_list, seq_len,
            opt_g, opt_f,
        )
        g_losses.append(stats.g_loss)
        f_losses.append(stats.f_loss)
        if step % max(1, args.n_steps // 5) == 0 or step == args.n_steps - 1:
            print(
                f"[c] step {step:3d}: g_loss={stats.g_loss:.4e}  "
                f"f_loss={stats.f_loss:.4e}"
            )
    elapsed = time.perf_counter() - t0
    print(f"[c] {args.n_steps} steps in {elapsed:.1f}s ({elapsed / args.n_steps * 1000:.1f} ms/step)")
    print(f"[c] g_loss start={g_losses[0]:.3e} end={g_losses[-1]:.3e}")
    print(f"[c] f_loss start={f_losses[0]:.3e} end={f_losses[-1]:.3e}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[c] peak GPU memory: {peak:.0f} MiB")
    print("[c] DONE")


if __name__ == "__main__":
    main()
