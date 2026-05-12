"""Pre-encode UMT5-xxl text contexts for each .pt latent and inline them.

Reads `data/openvid_meta/caption_manifest.jsonl`, runs Wan's T5EncoderModel
(umt5-xxl, bf16) on each caption, and writes the resulting context tensor
back into the same `.pt` file under a new "ctx" key.  Overwriting an existing
file reuses its inode (verified earlier) — important under the ZHUANGL inode
cap.

Resume-safe: skips files that already have "ctx".

Usage:
    python scripts/precompute_t5_ctx.py \
        --manifest data/openvid_meta/caption_manifest.jsonl \
        --batch_size 8 --text_len 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/openvid_meta/caption_manifest.jsonl")
    p.add_argument("--ckpt_dir", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--text_len", type=int, default=256,
                   help="Max tokens; Wan default is 512 but we cap to 256 to save storage.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1, help="-1 = all")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"[t5] loading manifest {args.manifest}")
    with open(args.manifest) as f:
        records = [json.loads(line) for line in f]
    if args.end < 0:
        args.end = len(records)
    records = records[args.start:args.end]
    print(f"[t5] {len(records):,} records in slice [{args.start}:{args.end}]")

    print(f"[t5] loading UMT5-xxl from {args.ckpt_dir}")
    from wan.modules.t5 import T5EncoderModel
    t5 = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device=args.device,
        checkpoint_path=os.path.join(args.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.ckpt_dir, "google/umt5-xxl"),
    )
    print(f"[t5] T5 ready, text_len={args.text_len}")

    t0 = time.perf_counter()
    n_done = 0
    n_skip = 0
    n_fail = 0
    bs = args.batch_size

    for batch_start in range(0, len(records), bs):
        batch = records[batch_start:batch_start + bs]

        # Pre-load all batch files; skip ones that already have ctx.
        loaded = []
        for rec in batch:
            path = rec["path"]
            try:
                d = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  SKIP unreadable {path}: {e}")
                n_fail += 1
                continue
            if "ctx" in d:
                n_skip += 1
                continue
            loaded.append((path, d, rec["caption"]))

        if not loaded:
            continue

        captions = [c for _, _, c in loaded]
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ctx_list = t5(captions, args.device)
        except Exception as e:
            print(f"  ENCODE-FAIL batch starting {batch_start}: {e}")
            n_fail += len(loaded)
            continue

        for (path, d, _caption), ctx in zip(loaded, ctx_list):
            d["ctx"] = ctx.detach().to("cpu", dtype=torch.bfloat16)
            try:
                torch.save(d, path)
                n_done += 1
            except Exception as e:
                print(f"  SAVE-FAIL {path}: {e}")
                n_fail += 1

        if (batch_start // bs + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = n_done / max(1e-6, elapsed)
            todo = len(records) - n_done - n_skip - n_fail
            eta = todo / max(1e-6, rate)
            print(f"[t5] {n_done}/{len(records)} done  (skip={n_skip} fail={n_fail})  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True)

    print(f"\n[t5] === DONE === processed {n_done}  skipped {n_skip}  failed {n_fail}")
    print(f"[t5] elapsed: {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
