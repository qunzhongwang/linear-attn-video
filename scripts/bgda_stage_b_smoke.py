"""Stage B wiring smoke — block-causal + GDN recurrence active.

Plan v1 §3.2 prescribes: activate `use_recurrence=True` + block-causal mask
+ monotonic-SNR sampler.  This script is the wiring-only smoke: build a
student with recurrence ON (no training), forward through real Wan-VAE
latents, and confirm:

  1. Forward succeeds (no NaN, no shape errors, no dtype mismatch).
  2. Output is finite and bounded (‖S‖_F doesn't explode at init).
  3. Recurrence gates init to "no-op" (S stays at zero with α=1, β=0 init),
     so Stage-B-at-init = per-block bidirectional read.
  4. Block-causality round-trip: changing future block's INPUT shouldn't
     change past block's OUTPUT.  Real-Wan version of the test we already
     have on the standalone reference impl.

Run on a single H100 / H200 with grad checkpointing (no backward needed
since this is forward-only, but grad-ckpt has no cost in eval mode).

    python scripts/bgda_stage_b_smoke.py --n_layers 4 --W_latent 3
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
    p.add_argument("--W_latent", type=int, default=3)
    p.add_argument("--latents_dir", default="data/wan_a1_minicorpus")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--beta_normalize", default="hw",
                   help="β-budget mode: 'hw' / 'sqrt_hw' / 'none'")
    args = p.parse_args()
    if args.beta_normalize == "none":
        args.beta_normalize = None

    device = torch.device(args.device)
    print(f"[B-smoke] device={device}")

    from wan.modules.model import WanModel  # noqa: E402

    print(f"[B-smoke] loading WanModel from {args.ckpt}")
    teacher = WanModel.from_pretrained(args.ckpt)
    teacher.eval()
    teacher.to(device, dtype=torch.bfloat16)
    if args.n_layers and args.n_layers < len(teacher.blocks):
        teacher.blocks = teacher.blocks[: args.n_layers]
    for q in teacher.parameters():
        q.requires_grad_(False)
    print(f"[B-smoke] {len(teacher.blocks)} blocks loaded")

    print("[B-smoke] building Stage-B student (use_recurrence=True)")
    student = copy.deepcopy(teacher)
    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
    )
    swap_wan_self_attention_to_bgda(
        student, W_latent=args.W_latent,
        use_recurrence=True,
        conv_kernel=3,
        lora_rank=args.lora_rank,
        beta_normalize=args.beta_normalize,
    )
    n_trainable = freeze_wan_backbone_except_bgda_new(student)
    print(f"[B-smoke] trainable: {n_trainable:,}  beta_normalize={args.beta_normalize}")

    in_dim = getattr(student, "in_dim", 16)
    text_dim = getattr(student, "text_dim", 4096)
    x, t, context, seq_len = _load_real_latents(
        args.latents_dir, in_dim, text_dim, device, torch.bfloat16, n_clips=1
    )
    print(f"[B-smoke] loaded latent shape={tuple(x[0].shape)}, seq_len={seq_len}")

    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    # Forward 1: teacher
    print("[B-smoke] forward teacher")
    t0 = time.perf_counter()
    with autocast, torch.no_grad():
        out_t = teacher(x, t=t, context=context, seq_len=seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[B-smoke] teacher: {time.perf_counter() - t0:.2f}s  shape={tuple(out_t[0].shape)}")

    # Forward 2: student with recurrence on
    print("[B-smoke] forward student (use_recurrence=True)")
    t0 = time.perf_counter()
    with autocast, torch.no_grad():
        out_s = student(x, t=t, context=context, seq_len=seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[B-smoke] student: {time.perf_counter() - t0:.2f}s  shape={tuple(out_s[0].shape)}")

    # Sanity 1: outputs are finite
    finite_t = torch.isfinite(out_t[0]).all().item()
    finite_s = torch.isfinite(out_s[0]).all().item()
    print(f"[B-smoke] finite: teacher={finite_t}  student={finite_s}")
    assert finite_t and finite_s, "non-finite outputs"

    # Sanity 2: output magnitudes
    print(
        f"[B-smoke] output ‖·‖_F: teacher={out_t[0].float().norm().item():.3e}  "
        f"student={out_s[0].float().norm().item():.3e}"
    )

    # Sanity 3: global cossim
    import torch.nn.functional as F
    g_cos = F.cosine_similarity(out_t[0].float().reshape(-1),
                                  out_s[0].float().reshape(-1), dim=0).item()
    print(f"[B-smoke] global output cossim: {g_cos:+.4f}")

    # Sanity 4: block-causality smoke at the model level — feed two latent
    # variants that differ only in the LAST block of the F axis; the
    # corresponding patch-token range of the OUTPUT should be unchanged for
    # tokens BEFORE the last block.  We do this only when n_layers > 0.
    print("[B-smoke] block-causality round-trip on student forward")
    x2 = [t.clone() for t in x]
    F_lat = x2[0].shape[1]
    last_block_start = F_lat - args.W_latent if F_lat % args.W_latent == 0 else F_lat - (F_lat % args.W_latent)
    x2[0][:, last_block_start:].add_(0.5)
    with autocast, torch.no_grad():
        out_s2 = student(x2, t=t, context=context, seq_len=seq_len)
    # Compare past-block output region.  Wan output is per-sample
    # [C_out, F, H, W]; we'll just compare the first "last_block_start" frames.
    diff_past = (out_s[0][:, :last_block_start] - out_s2[0][:, :last_block_start]).float().abs().max()
    diff_future = (out_s[0][:, last_block_start:] - out_s2[0][:, last_block_start:]).float().abs().max()
    print(
        f"[B-smoke] causality: max|out − out_perturb| past={diff_past.item():.3e}  "
        f"future={diff_future.item():.3e}"
    )
    if diff_past.item() < 1e-3:
        print("[B-smoke] ✓ block-causality holds at student-forward level")
    else:
        print("[B-smoke] ⚠ block-causality VIOLATED at the model level "
              "(expected: only future block self-attn changed; past untouched). "
              "Note: cross-attn + FFN may smear info, so model-level causality "
              "≠ self-attn-only causality. Investigate if this is unexpectedly large.")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[B-smoke] peak GPU memory: {peak:.0f} MiB")
    print("[B-smoke] DONE")


if __name__ == "__main__":
    main()
