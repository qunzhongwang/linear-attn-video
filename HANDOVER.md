# BGDA Stage A2 — Handover

Status as of **2026-05-17**. This document is the single-page summary for anyone
(or future-you) picking up the **Block Gated Delta Attention (BGDA)** work that
sits on top of NVIDIA's SANA codebase + Wan2.1-1.3B.

For Claude-Code instructions, see `CLAUDE.md` and `.claude/skills/bgda-project/SKILL.md`.

---

## 1. What this project is

We are **linearizing the self-attention of Wan2.1-1.3B**:

- Replace each layer's `WanSelfAttention` (full softmax attention over all
  21 latent frames) with **block-causal linear attention + Gated DeltaNet (GDN)
  state recurrence**, with block width `W = 3` latent frames.
- Cross-attention (text conditioning via UMT5-xxl) and FFN are untouched.
- Two training modes:
  - **LoRA** mode — backbone frozen, rank-64 LoRA on Q/K/V/O of the new attention.
  - **Full-FT** mode — entire 1.3B backbone trainable + LoRA on the new attn.

Goal: long-form video generation with bounded state cost (essential for the
Stage D 60-second target). Plan v2 §3 has the architectural rationale.

## 2. The 4-stage pipeline

| Stage | Method | Library entry | Driver | Status |
|---|---|---|---|---|
| **A1** | Attention-transfer (MSE on Wan attn outputs, no diffusion) | `spec/bgda/training/stage_a1_attention_transfer.py` | (skipped) | Skipped — capacity-limited, A2 subsumes |
| **A2** | Rectified-flow diffusion loss + LoRA (or Full-FT) | `spec/bgda/training/stage_a2_diffusion_loss.py` | `scripts/train_stage_a.py` | **← we are here**, plateau at loss ≈ 1.27 |
| **B**  | Block-autoregressive refinement + tightened GDN | (scaffolded) | (pending) | Not started |
| **C**  | DMD2 distillation from Wan-14B teacher | `spec/bgda/training/stage_c_dmd2.py` | (pending) | Stub only |
| **D**  | Long-form 60 s memory variants (SingleS / FixedSlotBank / AdaptiveSplit) | `spec/bgda/state_memory.py` | (pending) | Scaffolded |

## 3. Current Stage A2 state

### What works
- End-to-end DDP training on 4× GPU, BF16, with grad-checkpointing.
- Real UMT5-xxl text context flowing through cross-attention (verified by
  `swap_wan_self_attention_to_bgda` only touching `self_attn` modules,
  `spec/bgda/integration.py:110`).
- Inode-cap-safe ckpt overwrites; corrupt-`.pt` skip in the dataloader.
- Full wandb instrumentation: `train/loss`, `train/loss_ema`, `train/grad_norm`,
  `train/param_norm`, `train/pred_target_cos`, 10 per-t-bucket means
  (`loss_t/t_0.X_0.Y`), throughput.
- Periodic eval auto-fires every `--eval_interval` steps via SLURM sibling job
  that samples 2 short videos.

### What's broken / unexplained

**The loss plateaus at ≈ 1.27 with real text conditioning** — the same number
we got with the **random-Gaussian-as-text-context bug** before. Vanilla Wan-1.3B
with the same data + real ctx gives **0.07–0.15** on the same RF objective.

Recent pli runs (both completed 565 iters, ~26-28 h):

| Run | Mode | iter 0 → 50 → 100 → 400 → 560 |
|---|---|---|
| **LoRA pli 8084844** | rank-64, lr 2e-4 | 3.52 → 1.55 → 1.63 → 1.30 → **1.27 ema** |
| **Full-FT pli 8084846** | unfrozen, lr 4e-5 | similar shape → **1.26 ema** |

Per-t-bucket means at iter 560 are nearly **uniform** across [0, 1] (1.20–1.34) —
the student isn't learning the velocity field well at any noise level.

Wandb runs (synced):
- LoRA: https://wandb.ai/Princeton-Vison-Mix/bgda-stage-a/runs/dfe1od71
- Full-FT: https://wandb.ai/Princeton-Vison-Mix/bgda-stage-a/runs/wipye1hp

### Hypotheses for the plateau (in order of suspicion)

1. **BGDA forward bug** — math error in the linear-attn or GDN path. The math
   in `spec/bgda/reference.py` was unit-tested in isolation, but the integration
   wrapper `BGDABlockAttention` may have drifted. **Highest-leverage debug
   target.**
2. **W = 3 too restrictive** — 21-frame full self-attention can't be
   approximated by 7 non-overlapping 3-frame blocks with linear recurrence.
3. **Capacity floor** — rank-64 LoRA on Q/K/V/O can't shift the projections
   far enough to match the new attention's basis.
4. **t-distribution mismatch** — training samples uniform t ∈ [0, 1] while
   Wan was trained with a logit-shifted σ (shift = 8); high-t batches are
   overweighted in our regime.

### Pre-fix contamination

Any checkpoint under `output/stage_a_ddp/*.pt` (without `_v3` in the path) was
trained with random Gaussian as the model's text context — those are unusable
for any downstream stage. The current run output dirs are
`output/stage_a_v3_lora/` and `output/stage_a_v3_full/`. Always seed Stage B/C/D
from Wan-1.3B base, not from those legacy ckpts.

---

## 4. Repo layout (BGDA-relevant)

```
spec/bgda/
  reference.py                 # canonical block-causal linear attention + GDN math (well-tested)
  bgda_attention.py            # BGDABlockAttention — the Wan drop-in module
  integration.py               # swap_wan_self_attention_to_bgda(), freeze helpers, grad-ckpt
  lora.py                      # LoRALinear (rank, alpha, B init=0)
  state_memory.py              # Stage D variants: SingleS / FixedSlotBank / AdaptiveSplit
  training/
    stage_a1_attention_transfer.py   # Stage A1 (deferred)
    stage_a2_diffusion_loss.py       # rf_loss_step — the current objective
    stage_c_dmd2.py                  # DMD2 stub
  tests/                       # 39+ unit tests (math correctness, grad flow, S-bounded, β-budget)

scripts/
  train_stage_a.py             # MAIN: rank-aware DDP training (LoRA or --full_ft)
  measure_wan_baseline_loss.py # Vanilla Wan loss on same data + objective (sanity baseline)
  probe_batch_size.py          # Find batch-size ceiling per GPU
  build_caption_manifest.py    # OpenVidHD CSV → caption_manifest.jsonl
  precompute_t5_ctx.py         # UMT5-xxl encode → inline `ctx` into each .pt latent
  process_openvid_zips.py      # Zips → MP4 → Wan-VAE encoded latents
  bgda_inference.py            # Student inference via Wan pipeline hot-swap
  sbatch/                      # Cluster job scripts (currently SLURM/Della-flavoured)
```

Top-level untouched-by-us SANA dirs: `diffusion/`, `configs/`, `train_scripts/`,
`train_video_scripts/`, `inference_video_scripts/`, `app/`, `tools/`, `tests/`.

## 5. Data pipeline

Current training corpus: **31,096 OpenVidHD clips**, each encoded as

```python
# data/wan_latents_openvid/<id>.pt
{"z":   tensor[16, 21, 60, 104] bf16,    # Wan-VAE latent, already z-scored
 "ctx": tensor[~115, 4096]    bf16,      # UMT5-xxl text context, variable seq_len
 "src": "OpenVidHD_part_N.zip::clip.mp4",
 "shape": (16, 21, 60, 104)}
```

To rebuild from scratch on a new cluster (needs internet on at least one node):

```bash
# 1. Download OpenVidHD video zips (~47 GB each, yields ~26k 1080p MP4s per zip)
python ~/wp/routines/subroutine_download_upload/download_asset.py \
  -u "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD/OpenVidHD_part_1.zip" \
  -o <local>/OpenVid-1M/OpenVidHD/

# 2. Captions CSV (~286 MB)
python ~/wp/routines/subroutine_download_upload/download_asset.py \
  -u "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv" \
  -o data/openvid_meta/

# 3. MP4 → Wan-VAE latents (single-GPU job)
python scripts/process_openvid_zips.py \
  --zip_glob '<local>/OpenVid-1M/OpenVidHD/OpenVidHD_part_*.zip' \
  --out_dir data/wan_latents_openvid \
  --max_clips_per_zip 5000

# 4. Caption manifest (no GPU, ~1 min)
python scripts/build_caption_manifest.py
#   → data/openvid_meta/caption_manifest.jsonl

# 5. UMT5-xxl ctx pre-encoding (4-way split for speed, single GPU per chunk)
T=$(wc -l < data/openvid_meta/caption_manifest.jsonl)
for i in 0 1 2 3; do
  S=$((i*T/4)); E=$(((i+1)*T/4))
  BGDA_T5_START=$S BGDA_T5_END=$E python scripts/precompute_t5_ctx.py
done
```

The `.pt` files are **overwritten in place** by the T5 step (inode-safe);
the dataloader `require_ctx=True` refuses any file without `ctx`.

## 6. How to train

`scripts/train_stage_a.py` accepts a uniform CLI; the cluster-specific
sbatches in `scripts/sbatch/` are just wrappers. Core call:

```bash
torchrun --nproc_per_node=$NGPU --master_port=$RANDOM \
  scripts/train_stage_a.py \
  --latents_dir   data/wan_latents_openvid \
  --output_dir    output/stage_a_v3_lora \
  --max_iters     565 \
  --batch_size    4 \
  --grad_accum    16 \
  --lr            2e-4 \
  --warmup        20 \
  --lora_rank     64 \
  --grad_checkpointing \
  --use_recurrence \
  --log_interval  5 \
  --ckpt_interval 100 \
  --eval_interval 200 \
  --wandb --wandb_project bgda-stage-a \
  --wandb_group   "lora-$(date +%m%d)" \
  --wandb_run_name "lora_${SLURM_JOB_ID:-local}" \
  --max_walltime_seconds 169200
```

Add `--full_ft` for full fine-tune (use `--lr 4e-5` instead of `2e-4`).

### Effective batch sizes used in the completed runs

| Hardware | Mode | bs × accum × world = eff. |
|---|---|---|
| H200 140 GB × 4 | LoRA | 8 × 8 × 4 = 256 |
| H100 80 GB × 4  | LoRA | 4 × 16 × 4 = 256 |
| H200 140 GB × 4 | Full-FT | 6 × 10 × 4 = 240 |
| H100 80 GB × 4  | Full-FT | 4 × 16 × 4 = 256 |

Re-probe ceilings on new hardware via `scripts/probe_batch_size.py` —
runs bs = 1..N and reports peak memory + step time per mode.

### Outputs

- **Checkpoints** → `output/<dir>/latest.pt` (torch.save in-place, no new inodes).
- **Frozen ckpt for periodic eval** → `output/<dir>/latest_eval.pt`
  (snapshot at each eval interval to avoid race with the next save).
- **Periodic eval samples** → `output/<dir>/periodic_eval/iter_<N>/*.mp4`.
- **SLURM logs** → `outputs/train_stage_a_*.log` (gitignored).
- **wandb** → `output/<dir>/wandb/offline-run-*` (gitignored).

## 7. How to monitor

Training writes wandb **in offline mode** (compute nodes have unreliable
internet on Della; same is likely on other clusters). Sync from a node that
*does* have internet:

```bash
bash scripts/sbatch/wandb_sync.sh   # incremental; safe to re-run
```

Entity defaults to `Princeton-Vison-Mix` from `~/.netrc`. Project
`bgda-stage-a`. Run groups: `lora-MMDD` or `full-MMDD`.

## 8. How to evaluate

- **Vanilla-Wan baseline RF loss** (the must-do sanity check):
  `scripts/measure_wan_baseline_loss.py`. Same data + same RF objective + real
  ctx, no BGDA swap. Expect mean ≈ 0.07–0.15. Anything above 0.2 means the
  data/objective is broken.
- **Student video samples**: `scripts/bgda_inference.py` hot-swaps the BGDA
  student into a Wan pipeline. Run 8 prompts + Wan-1.3B baseline on the same
  prompts for visual A/B.
- **Periodic samples during training**: the training script auto-submits
  `bgda_periodic_eval.sbatch` (a 1-GPU, ≤45 min job) every `--eval_interval`
  steps.

## 9. Sanity-check pipeline (do this before ANY > 1 GPU-hour run)

1. **Run `measure_wan_baseline_loss.py`** on the current data with real ctx.
   If mean is much above 0.15 → the data or objective is broken; fix before
   training.
2. **Verify a sample `.pt` has `ctx`**:
   ```bash
   python -c "import torch; d=torch.load('data/wan_latents_openvid/00000000.pt', weights_only=False); \
     print(list(d.keys())); print('ctx' in d, d.get('ctx').shape if 'ctx' in d else 'MISSING')"
   ```
3. **Run a 30-iter wandb smoke** (`bgda_smoke_wandb.sbatch` on Della, or the
   moral equivalent on your new cluster) and inspect:
   - Initial `loss` and `grad_norm` are sane (not 1e9 — that means warmup
     missing).
   - `loss` descends in the first 50 iters.
   - All 10 `loss_t/t_*_*` buckets populate.

> The most-wasted 54 GPU-hours of this project came from skipping step 1.

## 10. Open issues / next steps

1. **Diagnose the 1.27 plateau.** First step: layer-by-layer parity check
   between vanilla Wan and BGDA-swapped-with-LoRA-B=0 on identical
   (x, t, ctx). If BGDA isn't a near-identity at init (it should be — LoRA
   B = 0 means delta is zero, but the linear-attn operator itself is *not*
   an identity replacement for softmax-attn), measure the per-layer output
   distribution shift.
2. **Try wider W** (e.g. W = 7 = full sequence). If loss drops sharply,
   block-width is the bottleneck.
3. **Try sigma-shifted t sampling** matching Wan's inference distribution.
4. **Stage B scaffolding** is in place but unwired — block-AR + tighter GDN.
   Probably not worth wiring until Stage A2 actually converges.

## 11. Pointers

- `CLAUDE.md` — the same content with Della-cluster-specific runtime notes for
  Claude Code.
- `.claude/skills/bgda-project/SKILL.md` — cluster-agnostic skill that gives
  Claude Code the project context on any cluster.
- `spec/bgda/README.md` — module-level docs for the BGDA library.
- `spec/bgda/docs/` — design notes.
- `spec/bgda/tests/` — unit tests for the math layer.

---

*Last updated: 2026-05-17. Author: BGDA Stage A2 team.*
