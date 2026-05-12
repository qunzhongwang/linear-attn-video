#!/bin/bash
#SBATCH --job-name=bgda-a1-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/bgda_a1_smoke_%j.log

set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}

echo "=== nvidia-smi ==="; nvidia-smi
echo "=== Stage A1 train smoke (4 layers, 50 steps, synthetic) ==="
python scripts/bgda_stage_a1_train_smoke.py --n_layers 4 --n_steps 50 --lr 1e-3 --W_latent 3
echo
echo "=== Stage A1 train smoke (30 layers, 30 steps, synthetic) ==="
python scripts/bgda_stage_a1_train_smoke.py --n_layers 30 --n_steps 30 --lr 1e-3 --W_latent 3
echo "=== DONE ==="
