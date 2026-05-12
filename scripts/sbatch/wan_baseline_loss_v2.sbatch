#!/bin/bash
#SBATCH --job-name=wan-base-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/wan_baseline_v2_%j.log

# Re-measure Wan-1.3B baseline RF loss WITH proper T5 contexts (post-fix).
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}

echo "=== node info ==="; hostname; nvidia-smi | head -13
echo ""
echo "=== A) uniform t (matches BGDA training) — WITH real T5 ctx ==="
python scripts/measure_wan_baseline_loss.py \
  --latents_dir data/wan_latents_openvid \
  --n_batches 300 --seed 0
echo ""
echo "=== B) shifted t (Wan inference, shift=8) — WITH real T5 ctx ==="
python scripts/measure_wan_baseline_loss.py \
  --latents_dir data/wan_latents_openvid \
  --n_batches 300 --seed 0 \
  --shifted_t --shift 8.0
echo "=== DONE ==="
