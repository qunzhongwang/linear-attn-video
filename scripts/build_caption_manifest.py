"""Build (id, src, caption) manifest by matching .pt latents against OpenVidHD CSV.

Reads each `.pt` file's `src` field (e.g. 'OpenVidHD_part_1.zip::magictime_X.mp4'),
extracts the MP4 basename, looks it up in OpenVidHD.csv, and emits a JSONL
manifest at `data/openvid_meta/caption_manifest.jsonl`.

Resume-safe: just rewrites the manifest each run.

Usage:
    python scripts/build_caption_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_dir", default="data/wan_latents_openvid")
    p.add_argument("--csv", default="data/openvid_meta/OpenVidHD.csv")
    p.add_argument("--out", default="data/openvid_meta/caption_manifest.jsonl")
    args = p.parse_args()

    print(f"[manifest] reading {args.csv}")
    name_to_caption = {}
    csv.field_size_limit(1 << 30)
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            name_to_caption[row["video"]] = row["caption"]
    print(f"[manifest] loaded {len(name_to_caption):,} captions from CSV")

    paths = sorted(glob.glob(os.path.join(args.latents_dir, "*.pt")))
    print(f"[manifest] {len(paths):,} latents in {args.latents_dir}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_miss = 0, 0
    with open(args.out, "w") as out:
        for p_ in paths:
            try:
                d = torch.load(p_, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  SKIP unreadable {p_}: {e}")
                continue
            src = d.get("src", "")
            # 'OpenVidHD_part_1.zip::path/to/clip.mp4' → take everything after last '/'
            mp4 = src.split("::", 1)[-1].split("/")[-1]
            if mp4 in name_to_caption:
                rec = {
                    "id": os.path.basename(p_).replace(".pt", ""),
                    "path": p_,
                    "src_mp4": mp4,
                    "caption": name_to_caption[mp4],
                }
                out.write(json.dumps(rec) + "\n")
                n_ok += 1
            else:
                n_miss += 1
                if n_miss < 5:
                    print(f"  MISS {p_} src={mp4!r}")

    print(f"\n[manifest] DONE  ok={n_ok}  miss={n_miss}  →  {args.out}")


if __name__ == "__main__":
    main()
