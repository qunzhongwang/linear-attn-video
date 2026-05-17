---
name: bgda-project
description: Use whenever working in this repo (Block Gated Delta Attention linearization of Wan2.1-1.3B on top of SANA). Captures the 4-stage pipeline, where library vs runner code lives, training/eval entry points, data layout, sanity-check rules before long runs, and the lessons learned (text-conditioning bug, inode-safe writes, baseline-first discipline). Cluster-agnostic — assumes only "you have one or more GPU nodes and can submit jobs."
---

# BGDA project skill

You are operating inside the **Block Gated Delta Attention (BGDA)** project,
a fork of NVIDIA's SANA codebase extended with a Wan2.1-1.3B linearization.
This skill gives you the project context. It is **cluster-agnostic** — it does
not assume Della, SLURM partitions, or specific paths under `/scratch` or
`/home`. For cluster-specific details (Della partitions, QOS, storage), see
`CLAUDE.md` at the repo root and the global `~/.claude/CLAUDE.md`.

For the human-readable handover summary, see `HANDOVER.md` at the repo root —
read it first whenever you start work here.

## 1. What the project is doing

**Goal.** Replace Wan2.1-1.3B's self-attention with **block-causal linear
attention + Gated DeltaNet recurrence** (block width `W = 3` latent frames),
keeping cross-attention (text via UMT5-xxl) and FFN untouched. Backbone is
either frozen with rank-64 LoRA on Q/K/V/O of the new attention, or fully
fine-tuned.

**Pipeline (4 stages, A → B → C → D):**

| Stage | Method | Library file | Driver | State |
|---|---|---|---|---|
| **A1** | MSE on Wan attention outputs (no diffusion) | `spec/bgda/training/stage_a1_attention_transfer.py` | (skipped) | Deferred — A2 subsumes |
| **A2** | Rectified-flow diffusion loss + LoRA/Full-FT | `spec/bgda/training/stage_a2_diffusion_loss.py` | `scripts/train_stage_a.py` | **Current.** Loss plateau ≈ 1.27 unexplained |
| **B**  | Block-AR + tighter GDN | (scaffolded) | (pending) | Not started |
| **C**  | DMD2 distillation from Wan-14B | `spec/bgda/training/stage_c_dmd2.py` | (pending) | Stub only |
| **D**  | 60 s long-form memory variants | `spec/bgda/state_memory.py` | (pending) | Scaffolded |

When the user references "Stage A / Stage A2 / current run / current plateau",
they mean the rectified-flow training driven by `scripts/train_stage_a.py`.

## 2. Where to find things

```
spec/bgda/              # Library code — the BGDA mechanism
  reference.py          # Canonical block-causal linear attn + GDN math (well unit-tested)
  bgda_attention.py     # BGDABlockAttention — Wan drop-in module
  integration.py        # swap_wan_self_attention_to_bgda(), freeze helpers, grad-ckpt
  lora.py               # LoRALinear (rank, alpha, B init = 0)
  state_memory.py       # Stage D variants
  training/             # Stage-specific loss functions / training utilities
  tests/                # 39+ unit tests on the math layer

scripts/                # Runners (one-shot Python entry points)
  train_stage_a.py              # Main DDP training, both LoRA and --full_ft modes
  measure_wan_baseline_loss.py  # Vanilla Wan loss on same data — the sanity baseline
  probe_batch_size.py           # Find safe bs per GPU
  build_caption_manifest.py     # OpenVidHD captions → manifest jsonl
  precompute_t5_ctx.py          # UMT5-xxl encode → inline ctx into each .pt latent
  process_openvid_zips.py       # Video zips → MP4 → Wan-VAE encoded latents
  bgda_inference.py             # Student inference (hot-swap into Wan pipeline)
  sbatch/                       # Cluster job submission wrappers (currently SLURM)

data/wan_latents_openvid/<id>.pt   # The training corpus, each .pt has:
  z:   tensor[16, 21, 60, 104]   # Wan-VAE latent (already z-scored)
  ctx: tensor[~115, 4096]        # UMT5-xxl context (variable seq_len)
  src: original filename
  shape: tuple

output/stage_a_v3_lora/   # LoRA-mode run outputs (latest.pt, eval samples, wandb offline)
output/stage_a_v3_full/   # Full-FT run outputs
outputs/                  # Job logs (gitignored)
```

Top-level SANA dirs (`diffusion/`, `configs/`, `train_scripts/`,
`inference_video_scripts/`, `app/`, `tools/`, `tests/`) are mostly inherited
from upstream SANA and rarely touched by BGDA work.

## 3. Hard rules (learned from real losses)

### Rule 1 — Sanity-check inputs before any > 1 GPU-hour run

The most expensive bug in this project was training the BGDA student with
**random Gaussian as the text context** instead of real UMT5-xxl embeddings.
54 GPU-hours wasted. Vanilla Wan with random ctx → loss ≈ 0.99; with real
ctx → loss ≈ 0.10. The student was effectively trained unconditionally
without anyone noticing.

**Before any long run** (i.e. anything submitted to a non-test queue):

1. **Verify the model's inputs are real, not random.** Print shape, dtype,
   mean, std of every tensor that goes into the forward pass. Compare to
   what you expect.
2. **Run `scripts/measure_wan_baseline_loss.py`** on the *same data* and
   *same objective* with the *vanilla* model. If the baseline loss is far
   from expectation (here: should be 0.07–0.15), the pipeline is broken;
   fix before scaling up.
3. **Run a short smoke run with wandb on** and confirm the diagnostic curves
   (loss trajectory, per-t-bucket means, grad_norm) populate sensibly.

If you skip these and the user pays for a 24-hour failed run, that is on you.

### Rule 2 — Surface loose ends proactively

After any non-trivial change (data pipeline edit, training script change,
new sbatch), enumerate three buckets for the user:

- ✅ **Fixed / verified** — what you actually proved works.
- ⚠️ **Imperfect / suspected** — known gaps, deferred work, things only
  partially tested.
- 🔒 **Sanity-check before irreversible action** — anything that should be
  confirmed before launching a long run, force-pushing, or merging.

The user has explicitly asked for this and stored it as feedback memory.
Saying "done" without these is a regression.

### Rule 3 — Storage and inode hygiene

Big filesets in academic clusters often have **inode quotas separate from
byte quotas**. This project has hit the inode cap multiple times. Rules:

- **Overwrite `latest.pt` in place** instead of writing `iter_<N>.pt` per
  checkpoint — `torch.save` reuses the inode.
- **Periodic-eval snapshots** copy `latest.pt` → `latest_eval.pt` (still
  in-place per-name overwrite).
- **Don't create many small files** in shared scratch (no per-iter logs,
  no per-batch sample dumps).
- **Wrap `torch.load` in try/except** in the dataloader — a `.pt` truncated
  by an EDQUOT mid-write will silently appear in the listing and crash
  training mid-run.

### Rule 4 — Wandb runs offline on compute, sync from a node with internet

Most cluster compute nodes have flaky or no internet. Standard pattern:

- Training scripts default to `--wandb_mode=offline`. Runs land at
  `output/<dir>/wandb/offline-run-*`.
- From a node with internet (login / head / vis node), run
  `bash scripts/sbatch/wandb_sync.sh`. Idempotent and incremental.

Entity, project, group naming conventions (in `train_stage_a.py` defaults):

- entity: auto-detect from `~/.netrc` (currently `Princeton-Vison-Mix`)
- project: `bgda-stage-a`
- group: `lora-MMDD` or `full-MMDD`
- run_name: `<mode>_<partition_or_host>_<job_id>`

### Rule 5 — Pre-fix checkpoints are contaminated

Anything under `output/stage_a_ddp/*.pt` (no `_v3` suffix) was trained
against random ctx and is unusable for downstream stages. Always seed Stage
B/C/D from Wan-1.3B base, **never** from those legacy checkpoints. The
current run output dirs are `output/stage_a_v3_lora/` and
`output/stage_a_v3_full/`.

## 4. Training — the uniform CLI

`scripts/train_stage_a.py` is launched under `torchrun --nproc_per_node=N`.
The cluster sbatch files in `scripts/sbatch/` are thin wrappers that set
environment variables and pick partition/QOS; the Python CLI is the same on
every cluster.

```bash
torchrun --nproc_per_node=$NGPU --master_port=$RANDOM \
  scripts/train_stage_a.py \
  --latents_dir   data/wan_latents_openvid \
  --output_dir    output/stage_a_v3_lora \
  --max_iters     565 \
  --batch_size    $BS \
  --grad_accum    $ACCUM \
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
  --wandb_run_name "lora_${HOSTNAME:-local}" \
  --max_walltime_seconds 169200
```

For Full-FT: add `--full_ft`, drop `--lr` to `4e-5`. For new hardware:
**always run `scripts/probe_batch_size.py` first** to find a safe `--batch_size`,
then pick `--grad_accum` to hit the target effective batch size (256 is
what the converged setup used).

## 5. Decision rules / opinions

- **Use a short test queue first** — never go straight to a 24 h queue for a
  smoke run. The cluster docs (e.g. Della's `gpu-test` ≤ 61 min) define what
  "short" is.
- **Don't bump pinned dependency versions casually.** `pyproject.toml` pins
  exact versions of xformers, transformers, triton, etc. Changes break SANA
  upstream paths.
- **Prefer editing existing files** over creating new modules.
- **No new style tooling.** This repo uses pre-commit (black 120 / isort
  black-profile / autoflake / pyupgrade --py37-plus); run
  `pre-commit run --all-files` before committing.
- **Tests are bash scripts** under `tests/bash/`, not pytest. CI gates on the
  `run-tests` PR label.

## 6. The current open question (Stage A2 plateau)

Loss plateaus at ≈ 1.27 with real text conditioning, on both LoRA and Full-FT
modes, after 560 iter × eff-bs 256 ≈ 144k samples seen. Per-t-bucket means
are nearly uniform (1.20–1.34), suggesting the student isn't learning the
velocity field at any noise level.

Candidate causes, in order of suspicion:

1. BGDA forward bug — math drift between `reference.py` and the integration
   wrapper. **Test:** layer-by-layer parity vanilla-Wan vs
   BGDA-swapped-with-LoRA-B=0 on identical (x, t, ctx).
2. W = 3 is too restrictive vs Wan's 21-frame full self-attention.
3. Capacity floor — rank-64 LoRA can't shift the projections enough.
4. Training samples uniform t but Wan inference uses logit-shifted σ (shift=8).

If the user asks about the plateau, recommend the parity check first; it
has the highest information-per-GPU-hour ratio.

## 7. Suggested first actions on a fresh cluster

1. `cat HANDOVER.md` — read the current snapshot.
2. `ls data/wan_latents_openvid 2> /dev/null | wc -l` — does the corpus exist?
   If 0, run the data pipeline (see §5 of HANDOVER.md).
3. `python scripts/probe_batch_size.py --mode lora --batch_sizes 1 2 4 8`
   to find a safe `--batch_size` for the new hardware.
4. Write or adapt the cluster's sbatch (or equivalent) by copying one of
   `scripts/sbatch/train_stage_a_v3_*.sbatch` and editing partition,
   account, walltime, and module/conda activation.
5. Submit a short smoke first (e.g. `--max_iters 30` on a test queue) and
   verify the wandb diagnostic curves before launching the long run.

## 8. Don'ts

- ❌ Don't train with random / placeholder text context. Verify `ctx` is real.
- ❌ Don't write many small files into shared scratch.
- ❌ Don't trust a pre-fix `output/stage_a_ddp/*.pt` checkpoint for any
  downstream work.
- ❌ Don't bump pinned dependency versions casually.
- ❌ Don't claim "done" without listing fixed / imperfect / sanity-check
  buckets.
- ❌ Don't introduce new lint/format tools — pre-commit is the only gate.
- ❌ Don't skip the baseline measurement before long runs.
