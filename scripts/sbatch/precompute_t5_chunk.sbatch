#!/bin/bash
#SBATCH --job-name=bgda-t5c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/precompute_t5_chunk_%j.log

# Parameterize via env: BGDA_T5_START, BGDA_T5_END.
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}

S="${BGDA_T5_START:?BGDA_T5_START not set}"
E="${BGDA_T5_END:?BGDA_T5_END not set}"
echo "=== T5 chunk [start=$S, end=$E) ==="
hostname; nvidia-smi | head -13

python scripts/precompute_t5_ctx.py \
  --manifest data/openvid_meta/caption_manifest.jsonl \
  --batch_size 8 --text_len 256 \
  --start "$S" --end "$E"

echo "=== chunk DONE [$S..$E) ==="
