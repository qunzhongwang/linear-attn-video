"""Stage A long-running training: 4-GPU DDP, RF diffusion loss + LoRA.

Plan v1 §5.2 — "Phase 1 — Stage A linearization (2 wk, 4×A100)".  We run the
Stage A2 sub-stage (LoRA + rectified-flow diffusion loss); sub-A1
attention-transfer was found to be capacity-limited on synthetic + 6-clip
mini-corpus (see spec/bgda/README.md "Stage A1 capacity finding") so we go
straight to A2 per the predicate-A fallback.

Launch (in sbatch / interactive):

    torchrun --nproc_per_node=4 --master_port=$RANDOM \
        scripts/train_stage_a.py \
        --latents_dir data/wan_a1_minicorpus \
        --output_dir output/stage_a_ddp \
        --max_iters 200000 --batch_size 1 --lr 1e-4 \
        --lora_rank 64 --grad_checkpointing --use_recurrence \
        --log_interval 20 --ckpt_interval 1000

Designed for safe long-running:
  - Resume from `output_dir/latest.pt` if it exists.
  - Periodic ckpt + EMA-of-loss for monitoring.
  - NaN/Inf check on every step → raise + log instead of silent corruption.
  - Cosine-warmup LR schedule.
  - Single-clip-per-GPU batches.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

WAN_REPO = "/home/qw3460/wp/Wan2.1"
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def maybe_print(*args, **kwargs):
    if is_main():
        print(*args, **kwargs, flush=True)


def setup_dist():
    if "RANK" not in os.environ:
        # Single-GPU fallback for local testing
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local)
    return rank, world, local


class LatentDataset:
    """Loads `*.pt` Wan-VAE latents on the fly.  Re-globs the directory every
    `reglob_every` accesses so new latents added by a parallel encoding job
    become available without restarting training.
    """

    def __init__(self, latents_dir: str, reglob_every: int = 200):
        self.latents_dir = latents_dir
        self.reglob_every = max(1, int(reglob_every))
        self._access_count = 0
        self._reglob()
        if not self.paths:
            raise FileNotFoundError(f"no .pt latents under {latents_dir}")

    def _reglob(self):
        self.paths = sorted(glob.glob(os.path.join(self.latents_dir, "*.pt")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, device=None, dtype=torch.bfloat16, require_ctx: bool = True):
        """Returns (z, ctx) tuple.  ctx is the pre-encoded UMT5-xxl text context
        (shape [L, 4096]).  If require_ctx=True (default), files lacking 'ctx'
        are skipped — used during training to force real text conditioning.
        """
        self._access_count += 1
        if self._access_count % self.reglob_every == 0:
            self._reglob()
        n = len(self.paths)
        last_err = None
        for tries in range(min(64, n)):
            p = self.paths[(idx + tries) % n]
            try:
                d = torch.load(p, map_location=device or "cpu", weights_only=False)
            except Exception as e:
                last_err = e
                print(f"[dataset] SKIP corrupt latent {p}: {type(e).__name__}: {str(e)[:100]}",
                      flush=True)
                continue
            if require_ctx and "ctx" not in d:
                # Not yet T5-encoded.  Skip.
                continue
            z = d["z"].to(device=device, dtype=dtype) if device is not None else d["z"]
            ctx = d.get("ctx", None)
            if ctx is not None and device is not None:
                ctx = ctx.to(device=device, dtype=dtype)
            return z, ctx
        raise RuntimeError(f"no valid latent with ctx found after {tries + 1} tries; last error: {last_err}")


def cosine_lr(step: int, *, warmup: int, total: int, base: float, min_ratio: float = 0.1):
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_student(args, device):
    from wan.modules.model import WanModel
    teacher = WanModel.from_pretrained(args.ckpt_1_3B)
    teacher.eval()
    teacher.to(device, dtype=torch.bfloat16)
    if args.n_layers and args.n_layers < len(teacher.blocks):
        teacher.blocks = teacher.blocks[: args.n_layers]
    # We don't need teacher for A2 (RF loss has no teacher); we just need its
    # weights as the student's frozen backbone.  So we do swap on the
    # teacher itself rather than deep-copy (saves 1.3 B × 2 bytes per GPU).
    from spec.bgda.integration import (
        swap_wan_self_attention_to_bgda,
        freeze_wan_backbone_except_bgda_new,
        bgda_new_param_groups,
        enable_gradient_checkpointing,
    )
    swap_wan_self_attention_to_bgda(
        teacher, W_latent=args.W_latent,
        use_recurrence=args.use_recurrence, conv_kernel=3,
        lora_rank=args.lora_rank,
        beta_normalize="hw" if args.use_recurrence else None,
    )
    if args.full_ft:
        # Full fine-tune: every param trainable, including the original Wan backbone.
        for p in teacher.parameters():
            p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    else:
        n_trainable = freeze_wan_backbone_except_bgda_new(teacher)
    if args.grad_checkpointing:
        enable_gradient_checkpointing(teacher)
    teacher.train()
    return teacher, n_trainable, bgda_new_param_groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_1_3B", default="/home/qw3460/wp/huggingface/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--latents_dir", default="data/wan_a1_minicorpus")
    p.add_argument("--output_dir", default="output/stage_a_ddp")
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--W_latent", type=int, default=3)
    p.add_argument("--use_recurrence", action="store_true")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--full_ft", action="store_true",
                   help="Unfreeze the Wan backbone; train all 1.3B params + BGDA-new.")
    p.add_argument("--init_from", default=None,
                   help="If set and <output_dir>/latest.pt does not exist, seed the model "
                        "(student state_dict only, NOT opt state) from this path.")

    p.add_argument("--max_iters", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=1, help="per-GPU; total batch = bs * world * grad_accum")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient-accumulation steps per optimizer step.  Effective "
                        "batch = batch_size * world * grad_accum.  DDP all-reduce is "
                        "deferred via no_sync() on intermediate micro-batches.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--eval_interval", type=int, default=0,
                   help="If >0, every N iters submit a gpu-test sbatch sampling 2 videos "
                        "from a frozen snapshot of latest.pt. 0 disables.")
    p.add_argument("--periodic_eval_sbatch",
                   default="/home/qw3460/bgda_eval_scripts/bgda_periodic_eval.sbatch",
                   help="Path to sbatch script fired for each periodic eval.")
    p.add_argument("--max_walltime_seconds", type=int, default=3600 * 47,
                   help="Stop training before walltime (default 47h to leave a buffer).")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    p.add_argument("--wandb_project", default="bgda-stage-a")
    p.add_argument("--wandb_entity", default=None,
                   help="None → use default from ~/.netrc (Princeton-Vison-Mix).")
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default="offline",
                   help="'offline' (default; sync later from login) or 'online'.")
    args = p.parse_args()

    rank, world, local = setup_dist()
    device = torch.device(f"cuda:{local}")
    out = Path(args.output_dir)
    if is_main():
        out.mkdir(parents=True, exist_ok=True)

    maybe_print(f"[train] world={world} rank={rank} device={device}")
    maybe_print(f"[train] args: {vars(args)}")

    # ── W&B (rank 0 only, offline-mode safe) ──
    wandb_run = None
    if args.wandb and is_main():
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        os.environ.setdefault("WANDB_DIR", str(out))
        import wandb as _wb
        wandb_run = _wb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            config=vars(args),
            mode=args.wandb_mode,
            resume="allow",
            dir=str(out),
        )
        maybe_print(f"[wandb] mode={args.wandb_mode} dir={out}/wandb url={wandb_run.url}")

    student, n_trainable, mk_groups = build_student(args, device)
    maybe_print(f"[train] trainable params per replica: {n_trainable:,}")

    if world > 1:
        student = DDP(student, device_ids=[local], find_unused_parameters=True)
        params = [p for p in student.parameters() if p.requires_grad]
    else:
        params = mk_groups(student, lr=args.lr)[0]["params"]

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Resume?
    start_iter = 0
    latest = out / "latest.pt"
    seed_path = latest if latest.exists() else (Path(args.init_from) if args.init_from else None)
    if seed_path is not None and seed_path.exists():
        ckpt = torch.load(seed_path, map_location=device, weights_only=False)
        target = student.module if isinstance(student, DDP) else student
        target.load_state_dict(ckpt["student"], strict=False)
        if seed_path == latest:
            try:
                opt.load_state_dict(ckpt["opt"])
                start_iter = ckpt["iter"]
                maybe_print(f"[train] resumed from {latest} at iter={start_iter}")
            except Exception as e:
                # E.g. param-group count differs (LoRA-only ckpt → full FT run).
                start_iter = ckpt["iter"]
                maybe_print(f"[train] kept iter={start_iter} but skipped opt load ({type(e).__name__}: {e})")
        else:
            # Seeding from external init: keep weights, fresh opt + iter=0.
            maybe_print(f"[train] seeded weights from {seed_path}; opt fresh, iter=0")

    dataset = LatentDataset(args.latents_dir)
    maybe_print(f"[train] {len(dataset)} latents under {args.latents_dir}")

    student_in_dim = 16
    text_dim = 4096
    g = torch.Generator(device=device).manual_seed(args.seed + rank)

    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    from spec.bgda.training.stage_a2_diffusion_loss import rf_loss_step

    walltime_start = time.perf_counter()
    losses_ema = None
    student.train()
    ACCUM = max(1, int(args.grad_accum))
    from contextlib import nullcontext
    from collections import deque

    # Ring buffer of (t, loss) pairs for per-t-bucket logging.
    diag_buf = deque(maxlen=2048)
    last_log_time = walltime_start
    last_log_iter = start_iter

    def _bucketed_t_losses(buf):
        """Return list of (bucket_label, n, mean) for 10 buckets in [0, 1]."""
        out = []
        for k in range(10):
            lo, hi = k / 10, (k + 1) / 10
            vs = [l for t, l in buf if lo <= t < hi]
            if vs:
                out.append((f"t_{lo:.1f}_{hi:.1f}", len(vs), sum(vs) / len(vs)))
        return out

    for it in range(start_iter, args.max_iters):
        # LR schedule (one update per optimizer step).
        lr = cosine_lr(it, warmup=args.warmup, total=args.max_iters, base=args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        accum_losses = []
        accum_pred_norm = 0.0
        accum_target_norm = 0.0
        accum_cos = 0.0
        non_finite = False
        for mb in range(ACCUM):
            # Sample one (z, ctx) per replica (round-robin), spreading the
            # accum window across the dataset.
            data_idx = ((it * ACCUM + mb) * world + rank) % len(dataset)
            x0, ctx = dataset.__getitem__(data_idx, device=device, dtype=torch.bfloat16,
                                          require_ctx=True)
            x0_list = [x0 for _ in range(args.batch_size)]
            ctx_list = [ctx for _ in range(args.batch_size)]
            F_, H_, W_ = x0.shape[1], x0.shape[2], x0.shape[3]
            seq_len = F_ * (H_ // 2) * (W_ // 2)

            sync_ctx = (
                student.no_sync()
                if (isinstance(student, DDP) and mb < ACCUM - 1)
                else nullcontext()
            )
            with sync_ctx, autocast:
                loss, st = rf_loss_step(
                    student.module if isinstance(student, DDP) else student,
                    x0_list, t_low=0.0, t_high=1.0,
                    context_list=ctx_list, seq_len=seq_len,
                    seed=args.seed + it * ACCUM + mb,
                )
            (loss / ACCUM).backward()
            if not torch.isfinite(loss):
                non_finite = True
                break
            accum_losses.append(float(loss.item()))
            accum_pred_norm += st.pred_norm
            accum_target_norm += st.target_norm
            accum_cos += st.pred_target_cos
            for t_i, l_i in zip(st.t_per_sample, st.loss_per_sample):
                diag_buf.append((float(t_i), float(l_i)))

        if non_finite:
            maybe_print(f"[train] step {it}: loss is non-finite; aborting")
            break

        # Gradient norm across all trainable params (before opt.step).
        with torch.no_grad():
            sq_sum = torch.zeros(1, device=device, dtype=torch.float32)
            for p in params:
                if p.grad is not None:
                    sq_sum += p.grad.detach().float().pow(2).sum()
            grad_norm = float(sq_sum.sqrt().item())

        opt.step()

        l = sum(accum_losses) / max(1, len(accum_losses))
        losses_ema = l if losses_ema is None else 0.97 * losses_ema + 0.03 * l

        if it % args.log_interval == 0 and is_main():
            elapsed = time.perf_counter() - walltime_start
            maybe_print(
                f"[train] iter {it:6d}  loss={l:.4e}  ema={losses_ema:.4e}  "
                f"lr={lr:.2e}  grad_norm={grad_norm:.3f}  elapsed={elapsed/60:.1f}min"
            )
            if wandb_run is not None:
                # Throughput since last log
                now = time.perf_counter()
                d_iters = max(1, it - last_log_iter)
                iters_per_sec = d_iters / max(1e-9, (now - last_log_time))
                samples_per_sec = iters_per_sec * world * args.batch_size * ACCUM
                last_log_time, last_log_iter = now, it

                # Parameter norm (rank 0 only — local approximation; for DDP each
                # rank has the same params after broadcast, so this is exact)
                with torch.no_grad():
                    pn_sq = sum(p.detach().float().pow(2).sum().item() for p in params)
                param_norm = pn_sq ** 0.5

                logs = {
                    "train/loss": l,
                    "train/loss_ema": losses_ema,
                    "train/lr": lr,
                    "train/elapsed_min": elapsed / 60,
                    "train/effective_batch": world * args.batch_size * ACCUM,
                    "train/loss_min_in_accum": min(accum_losses) if accum_losses else float("nan"),
                    "train/loss_max_in_accum": max(accum_losses) if accum_losses else float("nan"),
                    "train/grad_norm": grad_norm,
                    "train/param_norm": param_norm,
                    "train/pred_norm": accum_pred_norm / max(1, len(accum_losses)),
                    "train/target_norm": accum_target_norm / max(1, len(accum_losses)),
                    "train/pred_target_cos": accum_cos / max(1, len(accum_losses)),
                    "train/iters_per_sec": iters_per_sec,
                    "train/samples_per_sec": samples_per_sec,
                }
                # Per-t-bucket means over recent ring buffer
                for label, n, mean in _bucketed_t_losses(diag_buf):
                    logs[f"loss_t/{label}"] = mean
                    logs[f"loss_t_n/{label}"] = n
                logs["loss_t/buffer_size"] = len(diag_buf)
                wandb_run.log(logs, step=it)

        if it > 0 and it % args.ckpt_interval == 0 and is_main():
            target = student.module if isinstance(student, DDP) else student
            ck = {"student": target.state_dict(), "opt": opt.state_dict(),
                  "iter": it, "args": vars(args)}
            torch.save(ck, latest)
            maybe_print(f"[train] saved ckpt at iter {it} → {latest}")

            if args.eval_interval > 0 and it % args.eval_interval == 0:
                # Freeze a snapshot for eval (avoids race with later ckpt overwrites).
                eval_snap = out / "latest_eval.pt"
                torch.save(ck, eval_snap)
                # Fire-and-forget gpu-test eval job.
                import subprocess
                env = os.environ.copy()
                env["BGDA_EVAL_ITER"] = str(it)
                env["BGDA_EVAL_CKPT"] = str(eval_snap)
                env["BGDA_EVAL_OUT"] = str(out / "periodic_eval" / f"iter_{it:08d}")
                try:
                    r = subprocess.run(
                        ["sbatch", "--export=ALL,BGDA_EVAL_ITER,BGDA_EVAL_CKPT,BGDA_EVAL_OUT",
                         args.periodic_eval_sbatch],
                        env=env, check=False, capture_output=True, text=True, timeout=30,
                    )
                    maybe_print(f"[train] periodic eval submit: rc={r.returncode} "
                                f"stdout={r.stdout.strip()} stderr={r.stderr.strip()}")
                except Exception as e:
                    maybe_print(f"[train] periodic eval sbatch failed: {e}")

        if (time.perf_counter() - walltime_start) > args.max_walltime_seconds:
            maybe_print(f"[train] hit walltime budget ({args.max_walltime_seconds/3600:.1f}h); stopping")
            break

    # Final ckpt
    if is_main():
        target = student.module if isinstance(student, DDP) else student
        ck = {"student": target.state_dict(), "opt": opt.state_dict(),
              "iter": it + 1, "args": vars(args)}
        torch.save(ck, latest)
        maybe_print(f"[train] FINAL ckpt → {latest}")
        if wandb_run is not None:
            wandb_run.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
