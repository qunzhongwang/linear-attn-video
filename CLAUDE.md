# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is the NVlabs **SANA** codebase — a multi-family efficient-diffusion training/inference framework. A single repo houses several model lines that share the same `diffusion/` core:

- **SANA / SANA-1.5** — text-to-image DiT with Linear Attention + DC-AE.
- **SANA-Sprint** — sCM-based one/few-step distillation.
- **SANA-Video** — Block Causal Linear Attention DiT for video (uses Wan VAE).
- **LongSANA** — minute-length self-forcing video generation, streaming/chunk inference.
- **Sol-RL** — NVFP4 rollout / BF16 RL post-training (works on SANA, FLUX.1, SD3.5-L).

Everything else (configs, scripts, app, tests) is organized by family.

## Environment setup

```bash
./environment_setup.sh sana          # creates conda env "sana" (py3.10, cu12.8), pip install -e ., flash-attn, xformers
# or, if you have an env activated already:
SKIP_ENV_SETUP=true ./environment_setup.sh
```

`pyproject.toml` pins exact versions for `xformers==0.0.32.post2`, `transformers==4.57.0`, `triton==3.4.0`, `torchvision==0.23.0`, `peft==0.17.0`, `mmcv==1.7.2`, `accelerate==1.0.1`, `huggingface-hub==0.36.0`, `diffusers>=0.37.0`. Don't bump them casually. Installs a `sana` package providing two console scripts: `sana-run` and `sana-upload` (see `sana/cli/`).

## Lint / format

`pre-commit` is the only style gate. Black is configured at `line-length=120`, isort uses the black profile (also 120). Run before committing:

```bash
pre-commit run --all-files
```

CI's `pre-commit` job runs on every PR (`.github/workflows/ci.yaml`) and is required.

## Tests

There is **no pytest suite** — tests are bash scripts under `tests/bash/` that exercise full training/inference pipelines on tiny HF toy datasets. Each test auto-downloads its data via `hf download`.

```bash
bash tests/bash/entry.sh                                # everything
bash tests/bash/inference/test_inference.sh             # inference-only
bash tests/bash/training/test_training_fsdp.sh          # one family at a time
bash tests/bash/training/test_training_video.sh
bash tests/bash/training/test_training_longsana.sh
bash tests/bash/training/test_training_sol_rl.sh
bash tests/bash/training/test_training_vae.sh
```

The CI workflow runs these via `sana-run` on a self-hosted SLURM runner, gated by the PR label `run-tests`. Locally just run the bash scripts directly. `sana-run` itself requires `SANA_SLURM_ACCOUNT` and `SANA_SLURM_PARTITION` env vars and is for the upstream SLURM cluster — on Della you should run `torchrun`/`accelerate launch` directly (the shell scripts below already do).

## Training entry points

All training shells share a uniform CLI: positional YAML config + `--np=N` (GPU count), and any extra args are forwarded as **pyrallis** dotted overrides (e.g. `--train.lr=1e-4`, `--train.train_batch_size=2`, `--data.image_size=512`).

| Family | Shell | Python |
|---|---|---|
| SANA / SANA-1.5 image | `train_scripts/train.sh` | `train_scripts/train.py` |
| SANA-Sprint distillation | `train_scripts/train_scm_ladd.sh` | `train_scripts/train_scm_ladd.py` |
| LoRA / DreamBooth | `train_scripts/train_lora.sh` | `train_scripts/train_dreambooth_lora_sana.py` |
| SANA-Video joint I+V | `train_video_scripts/train_video_ivjoint.sh` | `train_video_scripts/train_video_ivjoint.py` |
| SANA-Video chunked | `train_video_scripts/train_video_ivjoint_chunk.sh` | `train_video_scripts/train_video_ivjoint_chunk.py` |
| LongSANA | `train_video_scripts/train_longsana.sh` | `train_video_scripts/train_longsana.py` |
| Sol-RL (SANA / FLUX.1 / SD3) | `train_scripts/sol_rl/run_*_single_node_8gpu.sh` | `train_scripts/sol_rl/train_*.py` |

Internally the shells build `torchrun --nproc_per_node=$np --master_port=$RANDOM ...` and pass `--config_path=<yaml>`. Common flags they hard-set: `--name=tmp`, `--resume_from=latest`, `--report_to=tensorboard`, `--debug=true`, `TRITON_PRINT_AUTOTUNING=1`. Video scripts also set `DISABLE_XFORMERS=1` and `DEBUG_MODE=1`.

Single-test/dev run example (mirrors what CI does):
```bash
bash train_video_scripts/train_video_ivjoint.sh \
  configs/sana_video_config/Sana_2000M_256px_AdamW_fsdp.yaml \
  --np=2 --train.num_epochs=1 --train.log_interval=1 --train.train_batch_size=1
```

## Inference entry points

- `scripts/inference.py` — main image inference + metric harness (FID/GenEval/DPG/ImageReward via `scripts/inference_*.py` and `scripts/bash_run_inference_metric*.sh`).
- `scripts/inference_sana_sprint*.py` — Sprint variants.
- `inference_video_scripts/inference_sana_video.{py,sh}` — video, launched via `accelerate launch --num_processes=$np --mixed_precision=bf16`.
- `app/app_sana*.py` — Gradio demos (4-bit, ControlNet, inpaint, multithread, Sprint, video refiner). Each app pairs with a pipeline class in `app/sana_*_pipeline.py` you can import directly.

## Big-picture architecture

The repo separates **library code** (`diffusion/`) from **runners** (`train_scripts/`, `inference_video_scripts/`, `scripts/`, `app/`). Everything is wired together by:

1. **A pyrallis dataclass config** loaded from a YAML in `configs/`. The schemas live in `diffusion/utils/config.py` (`SanaConfig`, `SanaVideoConfig`, plus `model_init_config` / `model_video_init_config` helpers). All YAMLs derive their structure from `configs/sana_base.yaml` (sections: `data`, `image_data`, `model`, `text_encoder`, `vae`, `scheduler`, `train`). The same dotted-key overrides work on the CLI.
2. **A model registry** in `diffusion/model/builder.py`. `build_model(config)` looks up a registered network by the `model.model` string key (e.g. `SanaMS_600M_P1_D28`, `SanaMSVideo_2000M_P2_D20`). Networks register themselves in `diffusion/model/nets/` — to add a new architecture you write the module there with `@MODELS.register_module()` and reference its name in the YAML.
3. **Builder helpers** for the rest of the pipeline: `get_tokenizer_and_text_encoder` (Gemma-2-2b-it by default), `get_vae` (DC-AE / DC-AE-Lite for image; WanVAE / LTX-VAE for video — selected by `vae.vae_type`), `vae_encode` / `vae_decode`. These are imported the same way in every training script.
4. **Schedulers** are re-exported from the top-level `diffusion` package (`diffusion/__init__.py`): `DPMS`, `FlowEuler`, `LTXFlowEuler`, `Scheduler`, `LongLiveFlowEuler`, `SASolverSampler`, `SCMScheduler`, `TrigFlowScheduler`. Pick via `scheduler.vis_sampler` / sCM configs.

### `diffusion/` layout

- `model/nets/` — all DiTs. Image: `sana.py`, `sana_multi_scale.py`, `sana_U_shape*.py`, `sana_multi_scale_adaln.py`, `sana_ladd.py`. Video: `sana_multi_scale_video.py`. ControlNet: `sana_multi_scale_controlnet.py`. Reusable blocks: `basic_modules.py`, `basic_modules_linear.py`, `ladd_blocks.py`, `fastlinear/` (Triton kernels).
- `model/dc_ae/efficientvit/` — DC-AE encoder/decoder.
- `model/wan/`, `model/wan2_2/` — Wan2.x model + VAE (used as the video VAE and as a reference T2V model in tests).
- `model/qwen/qwen_vl.py` — Qwen2.5-VL caption helper.
- `model/{respace,gaussian_diffusion,dpm_solver,sa_solver,edm_sample,timestep_sampler,model_growth_utils,norms,act}.py` — diffusion math + DiT utilities.
- `data/` — `builder.py` (dataset/dataloader factory), `datasets/` (`SanaWebDatasetMS`, `SanaZipDataset`, motion-score filtering for video), `wids/` (tar-shard sampler from WebDataset, with `DistributedRangedSampler` for resume), `transforms.py`.
- `scheduler/` — see above; one file per sampler/scheduler.
- `longsana/` — `pipeline/` (interactive long-chunk pipelines), `model/` (DMD distillation, ODE regression, streaming), `trainer/` (`longsana_trainer`, `self_forcing_trainer`, `ode`).
- `post_training/` — Sol-RL: `rewards.py`, `ema.py`, `stat_tracking.py`, `prompt_dataset.py`, `diffusers_patch/`, `dataset/`.
- `utils/` — `config.py` (pyrallis), `checkpoint.py` (load/save with EMA + FSDP), `dist_utils.py`, `data_sampler.py` (`AspectRatioBatchSampler`, `AspectRatioBatchSamplerVideo`), `lr_scheduler.py`, `optimizer.py`, `logger.py` (`LogBuffer`, `get_root_logger`), `git.py` (snapshot for run reproducibility), `misc.py`, `import_utils.py`.
- `guiders/` — guidance variants (CFG / PAG / etc.).

### `configs/` layout

Grouped by family and image size. Each subdir holds yamls that override `configs/sana_base.yaml`:

- `sana_config/{512ms,1024ms,2048ms,4096ms}/` — original SANA.
- `sana1-5_config/` — SANA-1.5 (FSDP, model growth, multi-scale).
- `sana_sprint_config/` — Sprint distillation.
- `sana_video_config/` — SANA-Video (incl. `longsana/` for LongSANA).
- `sana_controlnet_config/` — ControlNet.
- `sol_rl/` — Sol-RL, mixed yaml + `.py` configs (one per base model).
- `sana_app_config/` — gradio app presets.

### FSDP

FSDP is enabled per training entry (e.g. `train_video_ivjoint.py:set_fsdp_env`) by setting env vars before Accelerate launches: `FSDP_TRANSFORMER_CLS_TO_WRAP=<BlockClass>`, sharding strategy, prefetch, etc. **When you add a new top-level transformer block class, update that env var or FSDP will wrap nothing useful.**

## Tools and metrics

- `tools/convert_scripts/` — checkpoint format converters (pth ↔ diffusers, model_growth, video).
- `tools/metrics/` and `tools/scoring/` — FID, GenEval, DPG, ImageReward, HPS, CLIP score harnesses. Driven by `scripts/bash_run_inference_metric*.sh` and `scripts/inference_*.py` per metric.
- `tools/controlnet/` — preprocessors (HED etc.).
- `tools/inference_scaling/` — SANA-1.5 inference-time scaling utilities.
- `tools/download.py` (and `sana/tools/download.py`) — HF model fetcher used by `find_model` in inference scripts.

## Conventions to follow when editing

- New networks → add to `diffusion/model/nets/` with `@MODELS.register_module()`, reference by name in YAML's `model.model`. Don't import network modules directly from runners.
- New configs inherit from `configs/sana_base.yaml`; only override what you need. CLI overrides use pyrallis dotted keys.
- Tests = bash scripts. If you add a new training family, add a `tests/bash/training/test_training_<family>.sh` and wire it into `tests/bash/entry.sh` and `.github/workflows/ci.yaml`.
- Do not introduce a new style tool; use pre-commit (black 120 / isort black-profile / mdformat / yamlfmt / autoflake / pyupgrade --py37-plus). Pre-existing files have been auto-formatted.
- Per-family training shells must keep the YAML-positional + `--np=N` + pyrallis-passthrough convention so callers and CI keep working.

---

# BGDA Stage A2 — handover state (2026-05-12)

Below is a fast-handover summary of the **Block Gated Delta Attention (BGDA)** work
that lives in this repo on top of Wan2.1-1.3B. The library code is under
`spec/bgda/`, the training/eval/probe scripts under `scripts/`, and the SLURM
sbatches under `scripts/sbatch/`.

## What BGDA is

Linearization of Wan2.1-1.3B's self-attention with Gated DeltaNet (GDN) state
recurrence over W=3-latent-frame blocks. Backbone weights are frozen (LoRA mode)
or unfrozen (Full-FT mode); a LoRA rank-64 adapter sits on Q/K/V/O. Block-causal
attention within each W-block, GDN recurrence across blocks. Plan v2 §3.

## 4-stage pipeline

| Stage | Method | File | Status |
|---|---|---|---|
| A1 | Attention-transfer (MSE on Wan attn outputs) | `spec/bgda/training/stage_a1_attention_transfer.py` | Skipped — capacity-limited |
| **A2** | **Rectified-flow diffusion loss + LoRA** | `spec/bgda/training/stage_a2_diffusion_loss.py` | **← we are here** |
| B | Block-AR + GDN refinement | scaffolded | pending |
| C | DMD2 distillation from Wan-14B teacher | `spec/bgda/training/stage_c_dmd2.py` | pending |
| D | Long-form 60 s state memory variants | `spec/bgda/state_memory.py` | scaffolded |

## Current Stage A2 status

**Critical bugs found this session (2026-05-11) and fixed:**
1. **Random Gaussian as text context** — `train_stage_a.py` was feeding `torch.randn` as the model's T5 context. Vanilla-Wan baseline with random ctx = 0.99 RF loss; with **real UMT5-xxl ctx = 0.07-0.10** (~14× lower). The model had been training as an unconditional denoiser. Fix: pre-encode each .pt latent's caption via UMT5-xxl and inline a `ctx` key in the .pt. Training now refuses to use a .pt that lacks `ctx`.
2. **Corrupt latent under inode/quota pressure** — one of 31k OpenVidHD latents (`00031096.pt`) was truncated by an EDQUOT mid-write. Dataset now skips on `torch.load` failure (`require_ctx=True`).

**Known imperfect / open:**
- **Sigma-shift mismatch.** Training samples uniform t ∈ [0, 1]; Wan inference uses logit-shifted σ (shift=8). Not addressed.
- **Pre-fix ckpts contaminated** — anything at `output/stage_a_ddp/*.pt` was trained against random ctx and is unusable for downstream stages. Always seed fresh from Wan-1.3B base.
- **Stage B/C/D are scaffolded but unwired.** No real training has been done yet.

**Current jobs (queued 2026-05-12):** JID 8084844-8084847, effective batch 256.

## Repo layout (BGDA-relevant)

```
spec/bgda/
  reference.py                # canonical block-causal linear attention + GDN math
  bgda_attention.py           # BGDABlockAttention (Wan drop-in)
  integration.py              # swap_wan_self_attention_to_bgda(), freeze helpers, grad-ckpt wrapper
  lora.py                     # LoRALinear (rank, alpha, B init=0)
  state_memory.py             # SingleS / FixedSlotBank / AdaptiveSplit (Stage D variants)
  training/
    stage_a1_attention_transfer.py    # Stage A1 (deferred)
    stage_a2_diffusion_loss.py        # rf_loss_step (current)
    stage_c_dmd2.py                   # DMD2 stub
  tests/                      # 39+ unit tests (math correctness, grad flow, S-bounded, β-budget)

scripts/
  build_caption_manifest.py   # OpenVidHD CSV → caption_manifest.jsonl
  precompute_t5_ctx.py        # UMT5-xxl encode → inline ctx in .pt
  process_openvid_zips.py     # zips → MP4 → Wan-VAE encode → .pt latents
  train_stage_a.py            # MAIN: rank-aware DDP training (LoRA or Full-FT)
  measure_wan_baseline_loss.py # vanilla Wan loss measurement (sanity check)
  probe_batch_size.py         # ceiling probe per partition
  bgda_inference.py           # student inference via Wan pipeline hot-swap
  sbatch/                     # snapshot of the live sbatches (see below)
```

## How to fetch data (one-time, ~3 h on a vis node)

The current training corpus is **31,096 OpenVidHD clips encoded as `.pt` Wan-VAE
latents with inlined UMT5-xxl text contexts**. Everything is under
`data/wan_latents_openvid/<id>.pt`. Each file contains:
```python
{"z":   tensor[16, 21, 60, 104] bf16,        # Wan-VAE latent (already z-scored)
 "ctx": tensor[~115, 4096] bf16,              # UMT5-xxl text context, variable seq_len
 "src": "OpenVidHD_part_N.zip::clip.mp4",     # source filename
 "shape": (16, 21, 60, 104)}
```

To rebuild from scratch:

```bash
# 1. Download OpenVidHD video zips (FROM A VIS NODE; ~47 GB per zip).
#    Each zip yields ~26k MP4 clips at 1080p.
python ~/wp/routines/subroutine_download_upload/download_asset.py \
  -u "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD/OpenVidHD_part_1.zip" \
  -o ~/huggingface/nkp37/OpenVid-1M/OpenVidHD/

# 2. Download the captions CSV (small, ~286 MB).
python ~/wp/routines/subroutine_download_upload/download_asset.py \
  -u "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv" \
  -o data/openvid_meta/

# 3. Encode MP4s → Wan-VAE latents (gpu-test or gpu-short single GPU).
python scripts/process_openvid_zips.py \
  --zip_glob '~/huggingface/nkp37/OpenVid-1M/OpenVidHD/OpenVidHD_part_*.zip' \
  --out_dir data/wan_latents_openvid \
  --max_clips_per_zip 5000

# 4. Build caption manifest (1 min, no GPU).
python scripts/build_caption_manifest.py
#   → data/openvid_meta/caption_manifest.jsonl

# 5. Pre-encode UMT5-xxl T5 ctx into .pt files (gpu-test 4-way split, ~5 min total).
#    Split manifest into 4 chunks and submit 4 parallel jobs:
for i in 0 1 2 3; do
  T=$(wc -l < data/openvid_meta/caption_manifest.jsonl)
  S=$((i * T / 4)); E=$(((i+1) * T / 4))
  sbatch --export=ALL,BGDA_T5_START=$S,BGDA_T5_END=$E \
    scripts/sbatch/precompute_t5_chunk.sbatch
done
```

## Output & log conventions

The training scripts write into `output/` and `outputs/` (both gitignored):

- **Checkpoints** → `output/stage_a_v3_lora/latest.pt` and
  `output/stage_a_v3_full/latest.pt`. `torch.save` overwrites in place
  (inode-cap-safe — ZHUANGL is at the 175M inode limit; never rely on creating
  many new files there).
- **Frozen ckpt snapshot for periodic eval** → `output/<dir>/latest_eval.pt`
  (snapshotted at each `--eval_interval` to avoid race with the next ckpt save).
- **Periodic eval video samples** → `output/<dir>/periodic_eval/iter_<N>/*.mp4`.
- **SLURM stdout/stderr** → `outputs/train_stage_a_v3_<JID>.log` and friends.

**Important: ZHUANGL fileset is at the inode cap.** Any *new* file write may fail
with EDQUOT. Existing-file overwrites work (inode reused). The dataloader is
`try/except`-wrapped to skip corrupt `.pt` files. If you submit training and the
job dies in <60 s with `RaisedSignal:53`, slurm couldn't create the log file —
delete dead files via `find outputs -type f -mtime +7 -delete` etc.

## How to train

All 4 long-run sbatches live at `scripts/sbatch/`:

| File | Mode | Partition | bs × accum × world = effective |
|---|---|---|---|
| `train_stage_a_v3_ailab.sbatch` | LoRA | ailab H200 | 8 × 8 × 4 = 256 |
| `train_stage_a_v3_pli.sbatch` | LoRA | pli H100 | 4 × 16 × 4 = 256 |
| `train_stage_a_full_ailab.sbatch` | Full-FT | ailab H200 | 6 × 10 × 4 = 240 |
| `train_stage_a_full_pli.sbatch` | Full-FT | pli H100 | 4 × 16 × 4 = 256 |

Submit redundant on both partitions with sibling-cancel:
```bash
PLI=$(sbatch --parsable scripts/sbatch/train_stage_a_v3_pli.sbatch)
sbatch --export=ALL,BGDA_TRAIN_SIBLING=$PLI scripts/sbatch/train_stage_a_v3_ailab.sbatch
# (ailab job cancels the pli sibling when it starts running, via $BGDA_TRAIN_SIBLING)
```

All training jobs auto-submit `bgda_periodic_eval.sbatch` (gpu-test, 1 GPU, ≤45m)
every `--eval_interval` steps to sample 2 videos from the current ckpt.

## How to monitor + sync wandb

Training logs to wandb **in offline mode** (Della compute nodes have unreliable
internet). Each run writes to `output/<dir>/wandb/offline-run-<TS>-<id>/`. Sync
from a login node:

```bash
bash scripts/sbatch/wandb_sync.sh   # incremental; safe to re-run any time
```

Entity is auto-detected from `~/.netrc` → `Princeton-Vison-Mix`. Project:
`bgda-stage-a`. Run group: `lora-MMDD` or `full-MMDD`.

Logged metrics include `train/loss`, `train/loss_ema`, `train/grad_norm`,
`train/param_norm`, `train/pred_target_cos`, per-t-bucket means
(`loss_t/t_0.X_0.Y` for 10 buckets in [0,1]), and throughput
(`train/iters_per_sec`, `train/samples_per_sec`).

See [reference_wandb_della.md](/home/qw3460/.claude/projects/-scratch-gpfs-ZHUANGL-qw3460-workspace-linear-video/memory/reference_wandb_della.md)
for the full wandb-on-Della setup notes.

## How to evaluate

- **Vanilla-Wan baseline loss** (sanity check): `scripts/sbatch/wan_baseline_loss_v2.sbatch`. Runs `scripts/measure_wan_baseline_loss.py` on the same data + objective with real T5 ctx. Expected mean = 0.07–0.15.
- **Student video samples**: `scripts/sbatch/bgda_eval_gpu_test.sbatch` (8 prompts + Wan-1.3B baseline on same prompts). Runs `scripts/bgda_inference.py` which hot-swaps the BGDA student into a Wan pipeline.
- **Periodic samples during training**: auto-fires every `--eval_interval` steps.

## Batch-size sweet spots (from `scripts/sbatch/probe_batch_size.sbatch`)

Empirically measured 2026-05-12 on A100-80GB (apply with margin to H100 same / H200 ~1.8× capacity):

| Mode | Partition | "Safe" bs (≤70% mem) | Ceiling (OOM) |
|---|---|---|---|
| LoRA | A100/H100-80GB | 4 (55%) | 7 |
| LoRA | H200-140GB | 8 (60%) | ~12 |
| Full-FT | A100/H100-80GB | 4 (69%) | 6 |
| Full-FT | H200-140GB | 6 (~56%) | ~10 |

Re-run the probe sbatch if you change the architecture (different W_latent,
lora_rank, or grad_checkpointing setting changes the activation footprint).

## Sanity-check pipeline (do this before any > 1 GPU-hour run!)

1. **Run `scripts/measure_wan_baseline_loss.py`** with real T5 ctx. If the
   mean is much above 0.15 (uniform t), the data or pipeline is broken.
2. **Verify .pt latents have ctx**: `python -c "import torch; d=torch.load('data/wan_latents_openvid/00000000.pt'); print(list(d.keys()))"` — should include `"ctx"`.
3. **Run a 30-iter smoke** via `scripts/sbatch/bgda_smoke_wandb.sbatch` and check the offline wandb run for the full diagnostic set.

The most-wasted 54 GPU-hours of this project came from skipping step 1 and
training with random Gaussian as text context.

