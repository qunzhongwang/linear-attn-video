"""Generate videos with the trained BGDA-1.3B student."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path: sys.path.insert(0, WAN_REPO)
WS = "/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video"
if WS not in sys.path: sys.path.insert(0, WS)
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--out_dir", default="outputs/bgda_eval/videos")
    p.add_argument("--size", default="832*480")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--W_latent", type=int, default=3)
    p.add_argument("--use_recurrence", action="store_true", default=True)
    p.add_argument("--no_recurrence", dest="use_recurrence", action="store_false")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[infer] importing wan + building T2V pipeline (base ckpt {args.base_ckpt})")
    from wan.configs import WAN_CONFIGS
    import wan
    cfg = WAN_CONFIGS["t2v-1.3B"]
    pipe = wan.WanT2V(config=cfg, checkpoint_dir=args.base_ckpt, device_id=0,
                       rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False)
    print(f"[infer] swapping self_attn -> BGDABlockAttention")
    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
    )
    swap_wan_self_attention_to_bgda(
        pipe.model, W_latent=args.W_latent,
        use_recurrence=args.use_recurrence, conv_kernel=3,
        lora_rank=args.lora_rank,
        beta_normalize="hw" if args.use_recurrence else None,
    )
    freeze_wan_backbone_except_bgda_new(pipe.model)
    pipe.model.eval()
    pipe.model.to(torch.device("cuda:0"), dtype=torch.bfloat16)
    print(f"[infer] loading student ckpt {args.student_ckpt}")
    ck = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    state = ck.get("student", ck)
    incompat = pipe.model.load_state_dict(state, strict=False)
    print(f"[infer] missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")
    sw, sh = args.size.split("*") if "*" in args.size else args.size.split("x")
    size_tuple = (int(sw), int(sh))
    for i, prompt in enumerate(args.prompts):
        print(f"\n[infer] [{i+1}/{len(args.prompts)}] {prompt!r}")
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            video = pipe.generate(
                input_prompt=prompt, size=size_tuple, frame_num=args.frame_num,
                shift=5.0, sample_solver="unipc", sampling_steps=args.sample_steps,
                guide_scale=args.guide_scale, n_prompt="", seed=args.seed + i,
                offload_model=True,
            )
        elapsed = time.perf_counter() - t0
        print(f"[infer] sampled in {elapsed:.1f}s")
        from wan.utils.utils import cache_video
        out_path = Path(args.out_dir) / f"bgda_{i:03d}_{args.sample_steps}step.mp4"
        cache_video(tensor=video[None], save_file=str(out_path), fps=cfg.sample_fps,
                    nrow=1, normalize=True, value_range=(-1, 1))
        print(f"[infer] saved -> {out_path}")
    print("\n[infer] DONE")

if __name__ == "__main__":
    main()
