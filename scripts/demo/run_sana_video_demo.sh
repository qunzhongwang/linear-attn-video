#!/usr/bin/env bash
# Run SANA-Video / LongSANA demo across categories and model variants.
# Outputs land in a clean tree under outputs/sana_video_demo/videos/<category>/<variant>/.
#
# Usage:
#   bash scripts/demo/run_sana_video_demo.sh \
#       --variants base_480p,longsana_self_forcing \
#       --categories general,robotic,world_sim \
#       [--np 1] [--max-prompts 0]   # 0 = all
#
# Variants (each maps to a config + local checkpoint):
#   base_480p              - SANA-Video 2B 480p (DPM-solver, 50 steps, 81 frames)
#   longlive_480p_long     - SANA-Video 2B 480p LongLive (longlive_flow_euler, 161 frames)
#   longlive_480p_1min     - SANA-Video 2B 480p LongLive, ~1 minute (961 frames)
#   longsana_self_forcing  - LongSANA self-forcing distilled (.pt, few-step)
#   longsana_ode           - LongSANA ODE distilled (.pt, few-step)
#   base_720p              - SANA-Video 2B 720p with LTX-2 VAE (1280x704, 81 frames)

set -euo pipefail

REPO=/scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
HF_LOCAL=/home/qw3460/wp/huggingface/Efficient-Large-Model

# ----- args -----
VARIANTS="base_480p,longsana_self_forcing"
CATEGORIES="general,robotic,world_sim"
NP=1
MAX_PROMPTS=0   # 0 = all
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --variants=*)    VARIANTS="${1#*=}"; shift ;;
        --variants)      VARIANTS="$2"; shift 2 ;;
        --categories=*)  CATEGORIES="${1#*=}"; shift ;;
        --categories)    CATEGORIES="$2"; shift 2 ;;
        --np=*)          NP="${1#*=}"; shift ;;
        --np)            NP="$2"; shift 2 ;;
        --max-prompts=*) MAX_PROMPTS="${1#*=}"; shift ;;
        --max-prompts)   MAX_PROMPTS="$2"; shift 2 ;;
        --seed=*)        SEED="${1#*=}"; shift ;;
        --seed)          SEED="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

cd "$REPO"
export DISABLE_XFORMERS=1
export TOKENIZERS_PARALLELISM=false

OUT_BASE="$REPO/outputs/sana_video_demo"
RAW="$OUT_BASE/_raw"
mkdir -p "$RAW" "$OUT_BASE/videos" "$OUT_BASE/logs"

# Map variant -> config / model_path / vae_pretrained / extra args
variant_config() {
    case "$1" in
        base_480p)
            echo "configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml"
            ;;
        longlive_480p_long|longlive_480p_1min)
            echo "configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml"
            ;;
        longsana_self_forcing|longsana_ode)
            echo "configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml"
            ;;
        base_720p)
            echo "configs/sana_video_config/Sana_2000M_720px_ltx2vae_AdamW_fsdp.yaml"
            ;;
        *) echo "unknown variant $1" >&2; return 1 ;;
    esac
}

variant_model_path() {
    case "$1" in
        base_480p)
            echo "$HF_LOCAL/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth"
            ;;
        longlive_480p_long|longlive_480p_1min)
            echo "$HF_LOCAL/SANA-Video_2B_480p_LongLive/checkpoints/SANA_Video_2B_480p_LongLive.pth"
            ;;
        longsana_self_forcing)
            echo "$HF_LOCAL/LongSANA_2B_480p_self_forcing/checkpoints/LongSANA_2B_480p_self_forcing.pt"
            ;;
        longsana_ode)
            echo "$HF_LOCAL/LongSANA_2B_480p_ode/checkpoints/LongSANA_2B_480p_ode.pt"
            ;;
        base_720p)
            echo "$HF_LOCAL/SANA-Video_2B_720p/checkpoints/SANA_Video_2B_720p.pth"
            ;;
    esac
}

variant_vae() {
    case "$1" in
        base_720p)
            # 720p uses LTX-2 VAE via diffusers from_pretrained(repo_id, subfolder="vae")
            echo "Lightricks/LTX-2"
            ;;
        *)
            # All 480p variants use Wan2.1 VAE; the file is bundled in 480p / LongLive folders.
            echo "$HF_LOCAL/SANA-Video_2B_480p/vae/Wan2.1_VAE.pth"
            ;;
    esac
}

# Per-variant generation params (cfg, num_frames, extra flags as space-separated string)
variant_params() {
    case "$1" in
        base_480p)
            # 5s @ 16fps = 81 frames, 50 steps, cfg=6, motion=30
            echo "--cfg_scale=6 --motion_score=30 --flow_shift=8 --num_frames=81"
            ;;
        longlive_480p_long)
            # ~10s long form. cfg must be 1.0 for longlive_flow_euler
            echo "--cfg_scale=1.0 --num_frames=161"
            ;;
        longlive_480p_1min)
            # ~60s @ 16fps. num_frames must satisfy (N-1) % vae_temporal_stride == 0; 961 frames = 60s
            echo "--cfg_scale=1.0 --num_frames=961"
            ;;
        longsana_self_forcing|longsana_ode)
            # distilled few-step generator
            echo "--cfg_scale=1.0 --num_frames=161"
            ;;
        base_720p)
            # 5s @ 16fps; 720p with LTX-2 VAE (32x spatial, 8x temporal compression)
            echo "--cfg_scale=6 --motion_score=30 --flow_shift=8 --num_frames=81"
            ;;
    esac
}

run_one() {
    local category="$1" variant="$2"
    local cfg ckpt vae params
    cfg=$(variant_config "$variant")
    ckpt=$(variant_model_path "$variant")
    vae=$(variant_vae "$variant")
    params=$(variant_params "$variant")

    if [[ ! -f "$ckpt" ]]; then
        echo "[skip] missing ckpt: $ckpt" >&2; return
    fi
    # vae may be a local file path OR an HF repo_id (e.g. "Lightricks/LTX-2"); only file-check the former.
    if [[ "$vae" == /* ]] && [[ ! -f "$vae" ]]; then
        echo "[skip] missing vae:  $vae" >&2; return
    fi

    local prompt_file="$OUT_BASE/prompts/${category}.txt"
    if [[ ! -f "$prompt_file" ]]; then
        echo "[skip] missing prompts: $prompt_file" >&2; return
    fi

    local end_idx=30000
    if [[ "$MAX_PROMPTS" -gt 0 ]]; then
        end_idx="$MAX_PROMPTS"
    fi

    local work_dir="$RAW/$variant/$category"
    local clean_dir="$OUT_BASE/videos/$category/$variant"
    mkdir -p "$work_dir" "$clean_dir"
    local log_file="$OUT_BASE/logs/${variant}_${category}.log"

    echo "============================================================"
    echo "[run] variant=$variant  category=$category"
    echo "  cfg     = $cfg"
    echo "  ckpt    = $ckpt"
    echo "  vae     = $vae"
    echo "  prompts = $prompt_file"
    echo "  out     = $clean_dir"
    echo "  log     = $log_file"
    echo "============================================================"

    accelerate launch \
        --num_processes="$NP" \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --main_process_port=$((RANDOM % 10000 + 30000)) \
        inference_video_scripts/inference_sana_video.py \
            --config="$cfg" \
            --model_path="$ckpt" \
            --vae.vae_pretrained="$vae" \
            --txt_file="$prompt_file" \
            --work_dir="$work_dir" \
            --dataset="$category" \
            --seed="$SEED" \
            --end_index="$end_idx" \
            --debug=false \
            $params 2>&1 | tee "$log_file"

    # Move generated mp4s into clean tree.
    # Generated files live under $work_dir/vis/<auto_suffix>/*.mp4
    if compgen -G "$work_dir/vis/*/*.mp4" > /dev/null; then
        for f in "$work_dir"/vis/*/*.mp4; do
            cp -f "$f" "$clean_dir/"
        done
        echo "[ok] $(ls -1 "$clean_dir" | wc -l) video(s) -> $clean_dir"
    else
        echo "[warn] no mp4 found under $work_dir/vis/" >&2
    fi
}

IFS=',' read -r -a VARIANT_ARR <<< "$VARIANTS"
IFS=',' read -r -a CATEGORY_ARR <<< "$CATEGORIES"

for v in "${VARIANT_ARR[@]}"; do
    for c in "${CATEGORY_ARR[@]}"; do
        run_one "$c" "$v" || echo "[err] $v/$c failed (continuing)"
    done
done

echo
echo "DONE."
echo "Tree:"
find "$OUT_BASE/videos" -type f -name '*.mp4' | sort
