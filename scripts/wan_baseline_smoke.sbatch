#!/bin/bash
#SBATCH --job-name=wan-1.3b-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --qos=gpu-test
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/wan_baseline_smoke_%j.log

# Wan2.1-1.3B T2V smoke: generate 1 video with the baseline model.  Establishes
# that the locally-cached checkpoint at
#   /home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B
# loads + samples end-to-end on Della.  This is the "Wan-1.3B at home" baseline
# we will compare BGDA Stage A linearization against (plan v1 Phase 1).

set -e
cd /home/qw3460/wp/Wan2.1
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b

CKPT=/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B
OUT_DIR=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/wan_baseline_smoke
mkdir -p "$OUT_DIR"

echo "=== nvidia-smi ==="; nvidia-smi
echo "=== Wan2.1-1.3B T2V smoke (480P, 25 steps, 1 prompt) ==="
python generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --frame_num 81 \
  --sample_steps 25 \
  --ckpt_dir "$CKPT" \
  --prompt "A cinematic shot of a robot dancing on the edge of a cliff at sunset" \
  --save_file "$OUT_DIR/robot_cliff_480p_25steps.mp4" \
  --offload_model True

echo "=== DONE ==="
ls -lh "$OUT_DIR"/
