#!/bin/bash
#SBATCH --job-name=encode-vae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/encode_vae_%j.log

set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}

echo "=== nvidia-smi ==="; nvidia-smi
echo "=== encode 10 demo videos to Wan-VAE latents ==="
python scripts/encode_videos_to_latents.py \
  --videos_glob 'outputs/sana_video_demo/videos/*/base_480p/*.mp4' \
  --out_dir data/wan_a1_minicorpus \
  --max_videos 10
echo "=== DONE ==="
ls -la data/wan_a1_minicorpus/
