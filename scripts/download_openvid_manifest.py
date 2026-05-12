"""Manifest-only download for OpenVid-1M.

Pulls only `*.json`/`*.csv` from `nkp37/OpenVid-1M` (~600 MB, not the 7.5 TB
videos) so we can plan filtering before committing to a bulk shard pull.

⚠️  STORAGE ADVISORY — DO NOT enable bulk-video patterns without first checking
`checkquota` and coordinating with the user. The ZHUANGL fileset is currently
~99% full at the project level (May 2026); OpenVid-1M videos alone are 7.5 TB.

Run on a login or vis node (compute nodes lack internet):

    conda activate wan21_14b
    python scripts/download_openvid_manifest.py --target_dir ~/wp/huggingface/datasets/nkp37/OpenVid-1M
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Match the routine convention from ~/wp/routines/subroutine_download_upload
os.environ.setdefault("HF_HOME", "/home/qw3460/qw3460-per/.cache/huggingface")
from huggingface_hub import snapshot_download  # noqa: E402


DEFAULT_PATTERNS = ["*.json", "*.csv", "README*", "metadata/*"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="nkp37/OpenVid-1M")
    p.add_argument(
        "--target_dir",
        default="/home/qw3460/wp/huggingface/datasets/nkp37/OpenVid-1M",
    )
    p.add_argument(
        "--allow_patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="HF Hub `allow_patterns` glob(s).  Default = manifests only.",
    )
    args = p.parse_args()
    Path(args.target_dir).mkdir(parents=True, exist_ok=True)

    print(f"[download] repo={args.repo_id}  patterns={args.allow_patterns}")
    print(f"[download] target={args.target_dir}")
    out = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.target_dir,
        allow_patterns=args.allow_patterns,
        local_dir_use_symlinks=False,
    )
    print(f"[download] done: {out}")


if __name__ == "__main__":
    main()
