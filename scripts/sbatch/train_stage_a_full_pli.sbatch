#!/bin/bash
#SBATCH --job-name=bgda-Afullp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/train_stage_a_full_%j.log

# Redundant pli copy of the full-FT job.  See train_stage_a_full_ailab.sbatch.
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export DISABLE_XFORMERS=1
export NCCL_DEBUG=WARN

mkdir -p output/stage_a_ddp_full outputs

echo "=== node info ==="; hostname; nvidia-smi | head -25

if [ -n "${BGDA_TRAIN_SIBLING:-}" ]; then
  echo "=== cancelling sibling JID $BGDA_TRAIN_SIBLING ==="
  scancel "$BGDA_TRAIN_SIBLING" || true
fi

torchrun --nproc_per_node=4 --master_port=$((RANDOM % 10000 + 30000)) \
  scripts/train_stage_a.py \
  --latents_dir data/wan_latents_openvid \
  --output_dir output/stage_a_v3_full \
  --max_iters 565 \
  --batch_size 4 \
  --grad_accum 16 \
  --lr 4e-5 \
  --warmup 20 \
  --lora_rank 64 \
  --grad_checkpointing \
  --use_recurrence \
  --full_ft \
  --log_interval 5 \
  --ckpt_interval 100 \
  --eval_interval 200 \
  --wandb --wandb_project bgda-stage-a --wandb_group "full-$(date +%m%d)" \
  --wandb_run_name "full_pli_${SLURM_JOB_ID:-local}" \
  --max_walltime_seconds 169200

echo "=== DONE ==="
