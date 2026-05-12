"""Encode a small set of MP4 videos to Wan-VAE latents for the Stage A1
mini-corpus.

We re-use the SANA-Video / Wan-1.3B demo MP4s already in
`outputs/sana_video_demo/videos/` and the Wan baseline output. These have
real spatial/temporal coherence (unlike the Gaussian noise inputs in the
synthetic A1 smoke), so attention-transfer cossim should follow loss
properly when training on these latents.

Output: `[16, 21, 60, 104]` bf16 latent tensors saved as `<idx>.pt` under
`data/wan_a1_minicorpus/`.

Run on a GPU with ≥10 GiB VRAM (Wan-VAE only — no DiT, no T5):

    python scripts/encode_videos_to_latents.py \
      --videos_glob "outputs/sana_video_demo/videos/*/base_480p/*.mp4" \
      --out_dir data/wan_a1_minicorpus \
      --max_videos 10
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


def _read_video_frames(path: Path, n_frames: int = 81, target_h: int = 480,
                       target_w: int = 832) -> torch.Tensor:
    """Read up to `n_frames` and resize to (target_h, target_w).

    Returns: [3, n_frames, target_h, target_w] float in [-1, 1] (Wan convention).
    """
    arr = iio.imread(str(path), plugin="FFMPEG")  # [T, H, W, 3] uint8
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"unexpected video shape {arr.shape} from {path}")
    T = arr.shape[0]
    # Pad or truncate along time axis
    if T < n_frames:
        # Repeat-pad to n_frames
        rep = (n_frames + T - 1) // T
        arr = np.tile(arr, (rep, 1, 1, 1))[:n_frames]
    else:
        # Take first n_frames (uniform sampling could also work; first-N is fine for smoke)
        arr = arr[:n_frames]

    # [T, H, W, 3] uint8 → [3, T, H, W] float in [0, 1]
    x = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0
    # Resize spatial via interpolation per frame (bilinear).
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    # [-1, 1] (Wan convention; matches generate.py decode → cache_video)
    x = x * 2.0 - 1.0
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_glob",
                   default="outputs/sana_video_demo/videos/*/base_480p/*.mp4")
    p.add_argument("--out_dir", default="data/wan_a1_minicorpus")
    p.add_argument("--max_videos", type=int, default=10)
    p.add_argument("--n_frames", type=int, default=81)
    p.add_argument("--target_h", type=int, default=480)
    p.add_argument("--target_w", type=int, default=832)
    p.add_argument(
        "--wan_vae_ckpt",
        default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(args.videos_glob))[: args.max_videos]
    print(f"[encode] found {len(paths)} videos, processing {len(paths)}")

    print(f"[encode] loading Wan-VAE from {args.wan_vae_ckpt}")
    from wan.modules.vae import WanVAE
    vae = WanVAE(z_dim=16, vae_pth=args.wan_vae_ckpt,
                 dtype=torch.bfloat16, device=args.device)

    saved = []
    for i, path in enumerate(paths):
        try:
            x = _read_video_frames(Path(path), args.n_frames,
                                    args.target_h, args.target_w)
        except Exception as e:
            print(f"[encode] SKIP {path}: {e}")
            continue
        x = x.to(args.device, dtype=torch.bfloat16)
        with torch.no_grad():
            latents = vae.encode([x])
        z = latents[0]   # [16, T_latent, H/8, W/8]
        out_path = out_dir / f"{i:04d}.pt"
        torch.save({
            "z": z.cpu().to(torch.bfloat16),
            "src": path,
            "n_frames_in": args.n_frames,
            "shape": tuple(z.shape),
        }, out_path)
        saved.append(out_path)
        print(f"[encode] {out_path}  z.shape={tuple(z.shape)}  (from {Path(path).name})")

    print(f"[encode] DONE: saved {len(saved)} latents under {out_dir}")


if __name__ == "__main__":
    main()
