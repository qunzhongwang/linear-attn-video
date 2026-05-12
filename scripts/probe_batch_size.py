"""Probe maximum per-GPU batch size for BGDA Stage A2 (LoRA or Full-FT).

Tries batch sizes in sequence; for each, runs `--probe_iters` forward+backward
passes with grad-checkpointing, reports peak memory + per-iter time.  Bails
when OOM hits.

Usage:
    python scripts/probe_batch_size.py --mode lora --batch_sizes 1 2 3 4
    python scripts/probe_batch_size.py --mode full --batch_sizes 1 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

import torch


def build_student(ckpt_dir: str, mode: str, lora_rank: int,
                  W_latent: int, use_recurrence: bool, device):
    from wan.modules.model import WanModel
    teacher = WanModel.from_pretrained(ckpt_dir)
    teacher.eval()
    teacher.to(device, dtype=torch.bfloat16)
    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
        enable_gradient_checkpointing,
    )
    swap_wan_self_attention_to_bgda(
        teacher, W_latent=W_latent,
        use_recurrence=use_recurrence, conv_kernel=3,
        lora_rank=lora_rank,
        beta_normalize="hw" if use_recurrence else None,
    )
    if mode == "lora":
        n_trainable = freeze_wan_backbone_except_bgda_new(teacher)
    elif mode == "full":
        for p in teacher.parameters():
            p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    enable_gradient_checkpointing(teacher)
    teacher.train()
    return teacher, n_trainable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--latents_dir", default="data/wan_latents_openvid")
    p.add_argument("--mode", choices=["lora", "full"], default="lora")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--W_latent", type=int, default=3)
    p.add_argument("--use_recurrence", action="store_true")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--probe_iters", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    args.use_recurrence = True  # match training config

    print(f"[probe] mode={args.mode}  W_latent={args.W_latent}  "
          f"use_recurrence={args.use_recurrence}  device={device}")
    nvname = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[probe] GPU: {nvname}  total_mem={total_mem:.1f} GiB")

    # Load latents
    import glob, random
    paths = sorted(glob.glob(os.path.join(args.latents_dir, "*.pt")))
    print(f"[probe] {len(paths)} latents in {args.latents_dir}")

    student, n_trainable = build_student(
        args.ckpt_dir, args.mode, args.lora_rank,
        args.W_latent, args.use_recurrence, device,
    )
    print(f"[probe] trainable params = {n_trainable:,}")

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    from spec.bgda.training.stage_a2_diffusion_loss import rf_loss_step
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    results = []
    for bs in args.batch_sizes:
        print(f"\n[probe] === trying batch_size={bs} ===")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            for it in range(args.probe_iters):
                # Sample bs (z, ctx) pairs
                x0_list = []
                ctx_list = []
                for j in range(bs):
                    d = torch.load(paths[(it * bs + j) % len(paths)],
                                   map_location=device, weights_only=False)
                    if "ctx" not in d:
                        # Skip non-encoded
                        continue
                    x0_list.append(d["z"].to(device, dtype=torch.bfloat16))
                    ctx_list.append(d["ctx"].to(device, dtype=torch.bfloat16))
                if len(x0_list) == 0:
                    raise RuntimeError("no .pt with ctx")
                # If bs > available, replicate
                while len(x0_list) < bs:
                    x0_list.append(x0_list[0])
                    ctx_list.append(ctx_list[0])
                F_, H_, W_ = x0_list[0].shape[1], x0_list[0].shape[2], x0_list[0].shape[3]
                seq_len = F_ * (H_ // 2) * (W_ // 2)

                opt.zero_grad()
                with autocast:
                    loss, _ = rf_loss_step(
                        student, x0_list, t_low=0.0, t_high=1.0,
                        context_list=ctx_list, seq_len=seq_len, seed=42 + it,
                    )
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            iters_time = (t1 - t0) / args.probe_iters
            print(f"[probe] batch_size={bs}: OK  peak={peak:.2f} GiB  "
                  f"({peak/total_mem*100:.1f}% of total)  iter_time={iters_time:.2f}s")
            results.append((bs, "OK", peak, iters_time))
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower() or "OOM" in msg or "CUDA" in msg:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"[probe] batch_size={bs}: OOM  peak_before_OOM={peak:.2f} GiB")
                results.append((bs, "OOM", peak, None))
                break
            else:
                print(f"[probe] batch_size={bs}: ERROR {type(e).__name__}: {msg[:200]}")
                results.append((bs, "ERR", None, None))
                break

    print("\n[probe] === SUMMARY ===")
    print(f"  {'bs':>3}  {'state':<5}  {'peak GiB':>9}  {'iter sec':>9}")
    for bs, state, peak, t in results:
        peak_str = f"{peak:.2f}" if peak is not None else "—"
        t_str = f"{t:.2f}" if t is not None else "—"
        print(f"  {bs:>3}  {state:<5}  {peak_str:>9}  {t_str:>9}")


if __name__ == "__main__":
    main()
