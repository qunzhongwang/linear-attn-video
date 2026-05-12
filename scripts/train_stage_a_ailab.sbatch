#!/bin/bash
#SBATCH --job-name=bgda-A2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:4
#SBATCH --partition=ailab
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/train_stage_a_%j.log

set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export DISABLE_XFORMERS=1
export NCCL_DEBUG=WARN

mkdir -p output/stage_a_ddp outputs

echo "=== nvidia-smi ==="; nvidia-smi
echo "=== Stage A (LoRA + RF diffusion loss) — 4×H200 DDP, 48h walltime ==="
echo "WORLD will be 4 (--nproc_per_node=4)"
echo "data: ${1:-data/wan_a1_minicorpus}"

torchrun --nproc_per_node=4 --master_port=$((RANDOM % 10000 + 30000)) \
  scripts/train_stage_a.py \
  --latents_dir "${1:-data/wan_a1_minicorpus}" \
  --output_dir output/stage_a_ddp \
  --max_iters 200000 \
  --batch_size 1 \
  --lr 1e-4 \
  --warmup 500 \
  --lora_rank 64 \
  --grad_checkpointing \
  --use_recurrence \
  --log_interval 20 \
  --ckpt_interval 1000 \
  --max_walltime_seconds 169200

echo "=== DONE ==="
