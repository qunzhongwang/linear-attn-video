#!/bin/bash
#SBATCH --job-name=bgda-wan-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/bgda_wan_smoke_%j.log

set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}

echo "=== nvidia-smi ==="; nvidia-smi
echo "=== pytest (full BGDA suite quick) ==="
python -m pytest spec/bgda/tests/ -q --no-header 2>&1 | tail -5
echo "=== Wan↔BGDA integration smoke (4 blocks, fast) ==="
python scripts/bgda_wan_integration_smoke.py --n_layers 4 --W_latent 3
echo "=== Wan↔BGDA integration smoke (full 30 blocks) ==="
python scripts/bgda_wan_integration_smoke.py --n_layers 30 --W_latent 3
echo "=== DONE ==="
