#!/bin/bash
#SBATCH --job-name=bgda-A2v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:4
#SBATCH --partition=ailab
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/train_stage_a_v3_%j.log

# Stage A v3 — resumes from output/stage_a_ddp/latest.pt (iter 7000, EMA 1.27).
# Reads latents directly from ZHUANGL; the corrupt 00031096.pt has been
# removed and the dataloader skips on torch.load failure as belt-and-suspenders.
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export DISABLE_XFORMERS=1
export NCCL_DEBUG=WARN

mkdir -p output/stage_a_ddp outputs

echo "=== node info ==="; hostname; nvidia-smi | head -25
echo "=== latents: $(ls data/wan_latents_openvid | wc -l) files ==="

# Auto-cancel the redundant sibling once this one starts running.
if [ -n "${BGDA_TRAIN_SIBLING:-}" ]; then
  echo "=== cancelling sibling JID $BGDA_TRAIN_SIBLING ==="
  scancel "$BGDA_TRAIN_SIBLING" || true
fi

torchrun --nproc_per_node=4 --master_port=$((RANDOM % 10000 + 30000)) \
  scripts/train_stage_a.py \
  --latents_dir data/wan_latents_openvid \
  --output_dir output/stage_a_v3_lora \
  --max_iters 565 \
  --batch_size 8 \
  --grad_accum 8 \
  --lr 2e-4 \
  --warmup 20 \
  --lora_rank 64 \
  --grad_checkpointing \
  --use_recurrence \
  --log_interval 5 \
  --ckpt_interval 100 \
  --eval_interval 200 \
  --wandb --wandb_project bgda-stage-a --wandb_group "lora-$(date +%m%d)" \
  --wandb_run_name "lora_ailab_${SLURM_JOB_ID:-local}" \
  --max_walltime_seconds 169200

echo "=== DONE ==="
