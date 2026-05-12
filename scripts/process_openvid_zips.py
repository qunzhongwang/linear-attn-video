"""Extract OpenVidHD zip(s), encode each clip with Wan-VAE, save latents.

Each `OpenVidHD_part_N.zip` is ~47 GB and contains ~5 K MP4 clips at 1080P.
We process one zip at a time:

  1. Stream-extract the MP4s into a temp dir (no full extraction at once).
  2. For each MP4: read 81 frames, downsample to 480 × 832, run Wan-VAE encode,
     save `[16, 21, 60, 104]` bf16 latent to `--out_dir/<global_id>.pt`.
  3. Delete the temp MP4s after each batch so disk doesn't blow up.
  4. Optionally delete the source zip after a clean pass (`--delete_zip`).

Usage (single-GPU pli/gpu-test):

    python scripts/process_openvid_zips.py \
      --zip_glob '/home/qw3460/huggingface/nkp37/OpenVid-1M/OpenVidHD/OpenVidHD_part_1.zip' \
      --out_dir data/wan_latents_openvid \
      --max_clips_per_zip 1000

`max_clips_per_zip` is the safety knob to cap encoding time per zip; set it
to 0 or unset for "all clips".
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


def _read_video_to_tensor(path: Path, n_frames: int = 81, target_h: int = 480,
                            target_w: int = 832) -> torch.Tensor:
    arr = iio.imread(str(path), plugin="FFMPEG")  # [T, H, W, 3] uint8
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"unexpected video shape {arr.shape}")
    T = arr.shape[0]
    if T < n_frames:
        rep = (n_frames + T - 1) // T
        arr = np.tile(arr, (rep, 1, 1, 1))[:n_frames]
    else:
        arr = arr[:n_frames]
    x = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0  # [-1, 1] (Wan convention)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip_glob", required=True,
                   help="Glob pattern matching zip files to process.")
    p.add_argument("--out_dir", default="data/wan_latents_openvid")
    p.add_argument(
        "--wan_vae_ckpt",
        default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    )
    p.add_argument("--max_clips_per_zip", type=int, default=0,
                   help="Cap clips per zip; 0 = all.")
    p.add_argument("--delete_zip", action="store_true",
                   help="Delete the zip after successful encoding.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_frames", type=int, default=81)
    p.add_argument("--target_h", type=int, default=480)
    p.add_argument("--target_w", type=int, default=832)
    p.add_argument("--start_id", type=int, default=0,
                   help="Starting global clip index (resume safety).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(glob.glob(args.zip_glob))
    print(f"[proc] processing {len(zips)} zip(s)")
    if not zips:
        print("[proc] nothing to do")
        return

    print(f"[proc] loading Wan-VAE from {args.wan_vae_ckpt}")
    from wan.modules.vae import WanVAE
    vae = WanVAE(z_dim=16, vae_pth=args.wan_vae_ckpt,
                 dtype=torch.bfloat16, device=args.device)

    global_id = args.start_id
    total_encoded = 0
    for zip_path in zips:
        print(f"\n[proc] === {zip_path}  ({Path(zip_path).stat().st_size / 1024**3:.1f} GiB)")
        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory(dir="/scratch/gpfs/ZHUANGL/qw3460/tmp") as tmpdir:
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    members = [m for m in zf.namelist() if m.lower().endswith(".mp4")]
                    print(f"[proc]   {len(members)} mp4 members in zip")
                    if args.max_clips_per_zip > 0:
                        members = members[: args.max_clips_per_zip]
                    for i, m in enumerate(members):
                        try:
                            zf.extract(m, tmpdir)
                            mp4_path = Path(tmpdir) / m
                            x = _read_video_to_tensor(mp4_path, args.n_frames,
                                                       args.target_h, args.target_w)
                        except Exception as e:
                            print(f"[proc]   SKIP {m}: {e}")
                            continue
                        try:
                            x = x.to(args.device, dtype=torch.bfloat16)
                            with torch.no_grad():
                                z = vae.encode([x])[0]
                            out_path = out_dir / f"{global_id:08d}.pt"
                            torch.save({
                                "z": z.cpu().to(torch.bfloat16),
                                "src": str(Path(zip_path).name) + "::" + m,
                                "shape": tuple(z.shape),
                            }, out_path)
                            global_id += 1
                            total_encoded += 1
                            if total_encoded % 50 == 0:
                                print(f"[proc]   encoded {total_encoded} so far  "
                                      f"(latest {out_path.name}, z={tuple(z.shape)})")
                        except Exception as e:
                            print(f"[proc]   ENCODE-FAIL {m}: {e}")
                            traceback.print_exc()
                        finally:
                            # Delete the extracted MP4 immediately to keep tmp small
                            try:
                                mp4_path.unlink(missing_ok=True)
                            except Exception:
                                pass
            except zipfile.BadZipFile as e:
                print(f"[proc] BAD ZIP {zip_path}: {e}")
                continue

        elapsed = time.perf_counter() - t0
        print(f"[proc] zip done in {elapsed/60:.1f}min  total_encoded={total_encoded}")
        if args.delete_zip:
            try:
                Path(zip_path).unlink()
                print(f"[proc] deleted zip {zip_path}")
            except Exception as e:
                print(f"[proc] could not delete zip: {e}")

    print(f"\n[proc] DONE — total encoded {total_encoded}, last id={global_id - 1}")


if __name__ == "__main__":
    main()
