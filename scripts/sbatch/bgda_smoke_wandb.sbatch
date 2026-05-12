#!/bin/bash
#SBATCH --job-name=bgda-smkw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/bgda_smoke_wandb_%j.log

# Fast smoke: verify --wandb integration end-to-end before the 4 long jobs allocate.
# Same code path as the real run, just 30 iters with grad_accum=4.
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export DISABLE_XFORMERS=1

echo "=== node info ==="; hostname; nvidia-smi | head -13
mkdir -p output/stage_a_smoke_wandb outputs

torchrun --nproc_per_node=1 --master_port=$((RANDOM % 10000 + 30000)) \
  scripts/train_stage_a.py \
  --latents_dir data/wan_latents_openvid \
  --output_dir output/stage_a_smoke_wandb_v2 \
  --max_iters 15 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-4 \
  --warmup 3 \
  --lora_rank 64 \
  --grad_checkpointing \
  --use_recurrence \
  --log_interval 2 \
  --ckpt_interval 100 \
  --eval_interval 0 \
  --wandb --wandb_project bgda-stage-a --wandb_group "smoke-$(date +%m%d)" \
  --wandb_run_name "smoke_wandb_${SLURM_JOB_ID:-local}" \
  --max_walltime_seconds 1500

echo "=== DONE ==="
ls -la output/stage_a_smoke_wandb/wandb/