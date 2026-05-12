#!/bin/bash
#SBATCH --job-name=bgda-probehi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/probe_bs_hi_%j.log

# Ceiling chase on A100-80GB.
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export DISABLE_XFORMERS=1

echo "=== node info ==="; hostname; nvidia-smi | head -13
echo ""
echo "=== A) LoRA mode — high range bs 5..8 ==="
python scripts/probe_batch_size.py --mode lora --batch_sizes 5 6 7 8 --probe_iters 3 --use_recurrence || true
echo ""
echo "=== B) Full-FT mode — bs 4..6 (only if A100 allows) ==="
python scripts/probe_batch_size.py --mode full --batch_sizes 4 5 6 --probe_iters 3 --use_recurrence || true
echo "=== DONE ==="
