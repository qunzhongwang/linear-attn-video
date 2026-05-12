"""Wan ↔ BGDA integration smoke (Stage A pre-training sanity).

Loads the local Wan-1.3B T2V checkpoint, builds a SECOND copy with every
`WanSelfAttention` swapped for `BGDABlockAttention(use_recurrence=False)`,
and runs both on a small synthetic latent batch. Reports:

  - Per-layer attention-transfer MSE and cossim (teacher softmax vs student
    linear). Expectation: at init these are NOT close — sub-A1 training is
    what drives them up.
  - Whole-network forward cossim on the final velocity prediction.
  - Memory footprint (peak GPU MiB) of student.

Run on a single GPU with ≥24 GiB VRAM (DiT + T5 + VAE = ~17 GiB; with offload
40 GiB is comfortable).  Expected runtime: ~60 s end-to-end.

    sbatch scripts/bgda_wan_integration_smoke.sbatch
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

import torch
import torch.nn.functional as F


def _build_synthetic_inputs(model, B=1, F_latent=20, H=30, W=52, device="cuda", dtype=torch.bfloat16):
    """Manually produce inputs that `WanModel.forward` accepts.

    `WanModel.forward(x, t, context, seq_len, ...)` expects:
      x        : list[Tensor] of shape [C_in, F, H, W] (per-sample latents).
      t        : timestep [B] long
      context  : list[Tensor] of shape [L_text, C_text] (per-sample T5 hidden).
      seq_len  : the LARGEST patched sequence length across the batch.

    For our integration smoke we just want forward correctness — feed random
    latents with the right shapes.
    """
    cfg = model.config if hasattr(model, "config") else None
    in_dim = getattr(model, "in_dim", 16)            # Wan-1.3B in_dim = 16
    text_dim = getattr(model, "text_dim", 4096)
    g = torch.Generator(device=device).manual_seed(0)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype, device=device)

    # x:  list of tensors, one per sample.
    x = [rand(in_dim, F_latent, H, W) for _ in range(B)]
    t = torch.zeros(B, dtype=torch.long, device=device)
    context = [rand(64, text_dim) for _ in range(B)]
    # seq_len after patching (1, 2, 2): F * (H/2) * (W/2)
    p_t, p_h, p_w = (1, 2, 2)
    seq_len = F_latent * (H // p_h) * (W // p_w)
    return x, t, context, seq_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_layers", type=int, default=4,
                   help="When >0, only load/swap this many blocks for a fast smoke.")
    p.add_argument("--W_latent", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"[smoke] cuda: {torch.cuda.get_device_name(0)}")

    print("[smoke] importing wan…")
    from wan.modules.model import WanModel  # noqa: E402

    print(f"[smoke] loading WanModel from {args.ckpt}")
    teacher = WanModel.from_pretrained(args.ckpt)
    teacher.eval()
    teacher.to(device, dtype=torch.bfloat16)

    if args.n_layers and args.n_layers < len(teacher.blocks):
        teacher.blocks = teacher.blocks[: args.n_layers]
        print(f"[smoke] truncated to first {args.n_layers} blocks for fast smoke")
    n_blocks = len(teacher.blocks)

    # Build student via deep-copy (so teacher weights are untouched), swap in
    # BGDABlockAttention with use_recurrence=False.
    print("[smoke] building student via deep-copy + swap")
    student = copy.deepcopy(teacher)
    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
        is_bgda_new_param,
    )
    swapped = swap_wan_self_attention_to_bgda(
        student,
        W_latent=args.W_latent,
        use_recurrence=False,
        conv_kernel=3,
    )
    print(f"[smoke] swapped {len(swapped)} self_attn submodules")

    n_trainable = freeze_wan_backbone_except_bgda_new(student)
    n_total = sum(p.numel() for p in student.parameters())
    print(
        f"[smoke] student trainable {n_trainable:,} / total {n_total:,} = "
        f"{100.0 * n_trainable / n_total:.3f} %"
    )

    # Quick sanity: every BGDA-new param should be on the same device as backbone.
    for n, p in student.named_parameters():
        if is_bgda_new_param(n):
            assert p.device.type == device.type, f"{n} on {p.device}, expected {device}"
            assert p.requires_grad

    # Forward: compare per-layer self_attn outputs on synthetic inputs.
    from spec.bgda.training.stage_a1_attention_transfer import (
        make_capture_pair, attention_transfer_loss
    )
    tcap, scap = make_capture_pair(teacher, student)

    x, t, context, seq_len = _build_synthetic_inputs(
        teacher, B=1, F_latent=20, H=30, W=52, device=device, dtype=torch.bfloat16
    )

    # Wan's WanModel.forward expects an OUTER bf16 autocast — see
    # text2video.py:204 (`with amp.autocast(dtype=self.param_dtype)`). Without
    # it the internal float() upcasts fight bf16 weights and we get an
    # fp32-vs-bf16 dtype mismatch.
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    print("[smoke] forward teacher")
    t0 = time.perf_counter()
    with autocast_ctx, torch.no_grad():
        out_t = teacher(x, t=t, context=context, seq_len=seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[smoke] teacher forward took {time.perf_counter() - t0:.2f}s")

    print("[smoke] forward student")
    t0 = time.perf_counter()
    with autocast_ctx, torch.no_grad():
        out_s = student(x, t=t, context=context, seq_len=seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[smoke] student forward took {time.perf_counter() - t0:.2f}s")

    # Output cosine sim
    out_t_flat = out_t[0].float().reshape(-1)
    out_s_flat = out_s[0].float().reshape(-1)
    cos_global = F.cosine_similarity(out_t_flat, out_s_flat, dim=0).item()
    print(f"[smoke] global output cossim (teacher vs student): {cos_global:.4f}")

    # Per-layer attention-transfer loss (we don't backward, just inspect)
    loss, per_l_mse, per_l_cos = attention_transfer_loss(tcap.outputs, scap.outputs)
    print(f"[smoke] attention-transfer loss = {loss.item():.4f}  (n_layers={len(per_l_mse)})")
    for i, (m, c) in enumerate(zip(per_l_mse, per_l_cos)):
        print(f"[smoke]   layer {i:2d}: mse={m:.4e}  cossim={c:+.4f}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[smoke] peak GPU memory: {peak:.0f} MiB")

    tcap.remove(); scap.remove()
    print("[smoke] DONE")


if __name__ == "__main__":
    main()
