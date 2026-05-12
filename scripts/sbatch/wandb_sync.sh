#!/usr/bin/env bash
# Sync all offline wandb runs under our two output dirs.  Run from login node.
# Safe to re-run during training; incremental.
set -e
source /home/qw3460/miniconda3/etc/profile.d/conda.sh
conda activate wan21_14b
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
for d in output/stage_a_v3_lora/wandb output/stage_a_v3_full/wandb; do
  if [ -d "$d" ]; then
    for run in "$d"/offline-run-*; do
      [ -d "$run" ] || continue
      echo "=== syncing $run ==="
      wandb sync "$run" || echo "  (sync failed, will retry next time)"
    done
  fi
done
