#!/bin/bash
#SBATCH --job-name=bgda-peri
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/periodic_eval_%j.log

# Periodic mid-training eval. Triggered by train_stage_a.py every --eval_interval
# steps via subprocess.run(['sbatch', ...]) with these env vars:
#   BGDA_EVAL_ITER  — training iter number (used in output dir)
#   BGDA_EVAL_CKPT  — path to frozen ckpt snapshot (latest_eval.pt)
#   BGDA_EVAL_OUT   — output dir for the 2 sampled videos
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}

ITER="${BGDA_EVAL_ITER:?BGDA_EVAL_ITER not set}"
CKPT="${BGDA_EVAL_CKPT:?BGDA_EVAL_CKPT not set}"
OUT="${BGDA_EVAL_OUT:?BGDA_EVAL_OUT not set}"

echo "=== periodic eval @ iter $ITER ==="
hostname; nvidia-smi | head -13
echo "ckpt: $CKPT"
ls -la "$CKPT" || { echo "FATAL: ckpt missing"; exit 1; }
mkdir -p "$OUT"

python /home/qw3460/bgda_eval_scripts/bgda_inference.py \
  --base_ckpt /home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B \
  --student_ckpt "$CKPT" \
  --out_dir "$OUT" \
  --size 832*480 --frame_num 81 --sample_steps 25 --guide_scale 5.0 --seed 42 \
  --prompts \
    "A cinematic shot of a robot dancing on the edge of a cliff at sunset" \
    "A cat playing a piano in a sunlit living room"

echo "=== periodic eval iter $ITER DONE ==="
ls -la "$OUT"
