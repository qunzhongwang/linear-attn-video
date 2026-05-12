#!/bin/bash
#SBATCH --job-name=bgda-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/probe_bs_%j.log

# Probe max per-GPU batch size for LoRA + Full-FT on A100-80GB.
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export DISABLE_XFORMERS=1

echo "=== node info ==="; hostname; nvidia-smi | head -13
echo ""
echo "=== A) LoRA mode — bs 1..4 ==="
python scripts/probe_batch_size.py --mode lora --batch_sizes 1 2 3 4 --probe_iters 3 --use_recurrence
echo ""
echo "=== B) Full-FT mode — bs 1..3 ==="
python scripts/probe_batch_size.py --mode full --batch_sizes 1 2 3 --probe_iters 3 --use_recurrence
echo "=== DONE ==="
