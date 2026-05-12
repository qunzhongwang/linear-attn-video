#!/bin/bash
#SBATCH --job-name=bgda-eval-now
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/bgda_eval_now_%j.log

# Sample from the current ckpt (iter 7000, EMA 1.27) — no dependency, runs ASAP
# on gpu-test (≤61min, weight 8000).  Faster than waiting for retraining.
set -e
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate wan21_14b
export PYTHONPATH=/home/qw3460/wp/Wan2.1:/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video:${PYTHONPATH:-}
mkdir -p outputs/bgda_eval/videos outputs/bgda_eval/baseline_videos

CKPT=output/stage_a_ddp/latest.pt
echo "=== node info ==="; hostname; nvidia-smi | head -15
echo "=== ckpt: $CKPT ($(stat -c %s $CKPT) bytes) ==="
ls -la "$CKPT" || { echo "FATAL: ckpt missing"; exit 1; }

PROMPTS=(
  "A cinematic shot of a robot dancing on the edge of a cliff at sunset"
  "A bipedal humanoid robot in a brightly lit research laboratory"
  "A cat playing a piano in a sunlit living room"
  "A drone view sweeping over rugged sea cliffs at sunrise"
  "Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage"
  "A futuristic city skyline at dusk with flying cars"
  "A close-up of a glass of red wine slowly tilting on a wooden table"
  "A stack of dominoes lined up on a polished marble floor"
)

echo "=== 1) BGDA student inference (8 prompts) ==="
python /home/qw3460/bgda_eval_scripts/bgda_inference.py \
  --base_ckpt /home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B \
  --student_ckpt "$CKPT" \
  --out_dir outputs/bgda_eval/videos \
  --size 832*480 --frame_num 81 --sample_steps 25 --guide_scale 5.0 --seed 42 \
  --prompts "${PROMPTS[@]}"

echo "=== 2) Wan-1.3B baseline (same prompts) ==="
WAN_BASE=/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B
cd /home/qw3460/wp/Wan2.1
i=0
for P in "${PROMPTS[@]}"; do
  python generate.py --task t2v-1.3B --size 832*480 --frame_num 81 --sample_steps 25 \
    --ckpt_dir "$WAN_BASE" --prompt "$P" \
    --save_file /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video/outputs/bgda_eval/baseline_videos/wan_$(printf "%03d" $i).mp4
  i=$((i + 1))
done
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video

echo "=== 3) summary ==="
echo "BGDA student videos:"; ls -la outputs/bgda_eval/videos/
echo "Wan-1.3B baseline videos:"; ls -la outputs/bgda_eval/baseline_videos/
echo "=== DONE ==="
