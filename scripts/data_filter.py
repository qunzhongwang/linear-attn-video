"""Per-clip filtering pipeline (plan v1 §4.5.6).

Filters input video files by:
  - Scene-cut detection      (PySceneDetect)        — keep clips with single scene
  - Motion bounds            (UniMatch optical flow) — drop static (<3) and chaotic (>30)
  - Aesthetic score          (DOVER)                — drop bottom 30 %
  - Saturation outliers      (HSV S-channel mean)   — drop top 5 %
  - Temporal length          (FFprobe)              — keep 5.0 ± 0.5 s for A–C, 30+ s for D

Outputs a CSV with one row per surviving clip:
  clip_path, duration_s, motion_score, aesthetic_score, saturation_p95, scene_cuts

This script is a SCAFFOLD with concrete CLI / I/O semantics.  Each filter
function is left as a stub-with-TODO so we don't carry false dependencies on
heavyweight packages until they are actually being invoked on real data.

Usage:

    python scripts/data_filter.py \
      --input_dir  /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/data/openvid/raw \
      --output_csv /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/data/openvid/filtered.csv \
      --motion_min 3 --motion_max 30 \
      --aesthetic_threshold 0.40 \
      --saturation_max 200 \
      --target_duration 5.0 --duration_tol 0.5 \
      --num_workers 32
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("data_filter")


def ffprobe_duration(path: Path) -> Optional[float]:
    """Return clip duration in seconds via ffprobe."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip())
    except Exception:
        return None


def scene_cut_count(path: Path) -> Optional[int]:
    """Count scene cuts via PySceneDetect.  Default content-detector threshold = 27.0.

    TODO(plan §4.5.6): pin scenedetect>=0.6 in pyproject when activating this path.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        return None
    video = open_video(str(path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27.0))
    sm.detect_scenes(video)
    return len(sm.get_scene_list())


def motion_score(path: Path) -> Optional[float]:
    """UniMatch-based optical-flow magnitude score (plan §4.5.6).

    TODO(plan §4.5.6): wire UniMatch.  For now returns None to indicate the
    filter is not yet applied — caller treats None as "skip this filter".
    """
    return None


def aesthetic_score(path: Path) -> Optional[float]:
    """DOVER overall-quality score (0..1).

    TODO(plan §4.5.6): wire DOVER inference.  Until then returns None.
    """
    return None


def saturation_p95(path: Path) -> Optional[float]:
    """Mean of HSV S-channel across sampled frames; we use it as a saturation
    proxy.  Returns None if OpenCV missing.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    sample_idx = list(range(0, n, max(1, n // 8)))[:8]
    sats = []
    for i in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sats.append(float(hsv[..., 1].mean()))
    cap.release()
    if not sats:
        return None
    return float(np.percentile(sats, 95))


def evaluate_clip(path: Path, args: argparse.Namespace) -> Optional[dict]:
    """Run all filters; return row dict if clip survives, else None."""
    dur = ffprobe_duration(path)
    if dur is None:
        log.debug("skip (no duration): %s", path)
        return None
    if args.target_duration is not None:
        if abs(dur - args.target_duration) > args.duration_tol:
            return None
    if args.min_duration is not None and dur < args.min_duration:
        return None

    cuts = scene_cut_count(path)
    if cuts is not None and cuts > args.max_scene_cuts:
        return None

    mscore = motion_score(path)
    if mscore is not None:
        if mscore < args.motion_min or mscore > args.motion_max:
            return None

    a = aesthetic_score(path)
    if a is not None and a < args.aesthetic_threshold:
        return None

    sat = saturation_p95(path)
    if sat is not None and sat > args.saturation_max:
        return None

    return {
        "clip_path": str(path),
        "duration_s": dur,
        "motion_score": mscore,
        "aesthetic_score": a,
        "saturation_p95": sat,
        "scene_cuts": cuts,
    }


def list_videos(root: Path):
    exts = {".mp4", ".mkv", ".webm", ".mov"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_csv", required=True, type=Path)
    p.add_argument("--target_duration", type=float, default=5.0,
                   help="Target clip duration (s); set to None or 0 to skip.")
    p.add_argument("--duration_tol", type=float, default=0.5,
                   help="Allowed deviation from target_duration.")
    p.add_argument("--min_duration", type=float, default=None,
                   help="Min duration; use for Stage D long-form (e.g. 30s).")
    p.add_argument("--max_scene_cuts", type=int, default=1,
                   help="Drop clips with > N scene cuts (0 = single scene).")
    p.add_argument("--motion_min", type=float, default=3.0)
    p.add_argument("--motion_max", type=float, default=30.0)
    p.add_argument("--aesthetic_threshold", type=float, default=0.40)
    p.add_argument("--saturation_max", type=float, default=200.0)
    p.add_argument("--num_workers", type=int, default=32)
    args = p.parse_args()
    if args.target_duration in (None, 0, 0.0):
        args.target_duration = None

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_path", "duration_s", "motion_score", "aesthetic_score",
        "saturation_p95", "scene_cuts",
    ]
    files = list(list_videos(args.input_dir))
    log.info("found %d candidate videos under %s", len(files), args.input_dir)

    kept = 0
    with open(args.output_csv, "w", newline="") as fh, \
            ProcessPoolExecutor(max_workers=args.num_workers) as exe:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        futs = {exe.submit(evaluate_clip, f, args): f for f in files}
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                continue
            w.writerow(row)
            kept += 1
            if kept % 1000 == 0:
                log.info("kept %d so far", kept)
    log.info("done: %d / %d retained → %s", kept, len(files), args.output_csv)


if __name__ == "__main__":
    main()
