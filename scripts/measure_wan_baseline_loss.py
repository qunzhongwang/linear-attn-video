"""Measure vanilla Wan2.1-1.3B RF loss on OpenVid latents (no BGDA swap).

Purpose: provide a baseline RF-loss number for the same data + objective we
trained BGDA against, so we can interpret the BGDA plateau (~1.20 EMA).

Reads the same OpenVid latents the training jobs used, samples random timesteps
exactly like rf_loss_step, runs Wan-1.3B forward in eval/no_grad mode, and
prints the mean loss across N batches.  Single-GPU; designed for gpu-test.

Usage:
    python scripts/measure_wan_baseline_loss.py \
        --latents_dir data/wan_latents_openvid \
        --n_batches 500 --seed 0

Optional comparison flags:
    --shifted_t          # match Wan inference: t ~ shifted_sigma * 1000
    --shift 8.0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

import numpy as np
import torch


def shifted_sigma(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply Wan's logit-shift: σ' = shift·σ / (1 + (shift-1)·σ)."""
    return shift * t / (1 + (shift - 1) * t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_1_3B", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--latents_dir", default="data/wan_latents_openvid")
    p.add_argument("--n_batches", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shifted_t", action="store_true",
                   help="Use Wan's inference sigma-shift distribution instead of uniform.")
    p.add_argument("--shift", type=float, default=8.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[base] loading Wan-1.3B from {args.ckpt_1_3B}")
    from wan.modules.model import WanModel
    model = WanModel.from_pretrained(args.ckpt_1_3B)
    model.eval()
    model.to(device, dtype=torch.bfloat16)
    print(f"[base] model loaded — params={sum(p.numel() for p in model.parameters()):,}")

    paths = sorted(glob.glob(os.path.join(args.latents_dir, "*.pt")))
    print(f"[base] {len(paths)} latent files in {args.latents_dir}")
    assert len(paths) > args.n_batches, "not enough latents"

    g = torch.Generator(device=device).manual_seed(args.seed)

    losses = []
    n_used = 0
    n_skip_no_ctx = 0
    t_buckets = {f"[{a:.2f},{a+0.1:.2f})": [] for a in np.arange(0, 1.0, 0.1)}
    t0 = time.perf_counter()

    with torch.no_grad():
        i = 0
        while n_used < args.n_batches and i < len(paths):
            try:
                d = torch.load(paths[i], map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[base] SKIP corrupt {paths[i]}: {e}")
                i += 1
                continue
            i += 1
            if "ctx" not in d:
                n_skip_no_ctx += 1
                continue
            x0 = d["z"].to(device, dtype=torch.bfloat16)             # [C, F, H, W]
            ctx = d["ctx"].to(device, dtype=torch.bfloat16)         # [L_text, 4096]
            eps = torch.randn(*x0.shape, generator=g, dtype=torch.bfloat16, device=device)

            # Timestep sample
            t = torch.rand(1, device=device, generator=g, dtype=torch.float32)
            if args.shifted_t:
                sigma = shifted_sigma(t, args.shift)
            else:
                sigma = t  # uniform t = sigma

            # Build x_t and target in float32, cast back to bf16
            x0f = x0.float(); epsf = eps.float()
            sigma_b = sigma.view(1, 1, 1, 1).float()
            x_t = ((1 - sigma_b) * x0f + sigma_b * epsf).to(torch.bfloat16)
            v_target = (epsf - x0f).to(torch.bfloat16)

            t_long = (sigma * 1000.0).round().clamp(0, 999).long()

            F_, H_, W_ = x0.shape[1], x0.shape[2], x0.shape[3]
            seq_len = F_ * (H_ // 2) * (W_ // 2)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model([x_t], t=t_long, context=[ctx], seq_len=seq_len)[0]
            n_used += 1

            l = (pred.float() - v_target.float()).pow(2).mean().item()
            losses.append(l)

            # Bucket by t for distribution-of-loss-vs-t
            t_val = float(sigma.item())
            for key, vs in t_buckets.items():
                lo, hi = float(key[1:5]), float(key[6:10])
                if lo <= t_val < hi:
                    vs.append(l); break

            if n_used % 50 == 0:
                avg = np.mean(losses[-50:])
                rate = n_used / max(1e-6, (time.perf_counter() - t0))
                print(f"[base] {n_used}/{args.n_batches}  last-50-mean={avg:.4f}  "
                      f"rate={rate:.2f} it/s  skip_no_ctx={n_skip_no_ctx}", flush=True)

    losses = np.array(losses)
    print(f"\n[base] === FINAL ===  n={len(losses)}  mean={losses.mean():.4f}  "
          f"std={losses.std():.4f}  median={np.median(losses):.4f}  "
          f"5%={np.percentile(losses, 5):.4f}  95%={np.percentile(losses, 95):.4f}")
    print(f"[base] skipped {n_skip_no_ctx} files (no ctx yet)")
    print(f"[base] mode: {'shifted t (shift=' + str(args.shift) + ')' if args.shifted_t else 'uniform t'}")
    print(f"[base] loss-vs-t bucket means:")
    for key, vs in t_buckets.items():
        if vs:
            print(f"  t∈{key}: n={len(vs):4d}  mean={np.mean(vs):.4f}")


if __name__ == "__main__":
    main()
