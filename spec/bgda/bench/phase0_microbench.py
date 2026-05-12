"""Phase 0 microbench — plan v1 §5.1 deliverables.

Run on a single GPU (MIG or full A100/H100). Outputs a CSV with per-block
latency, S-norm growth, RoPE numerical-sanity checks, and single-layer
gradient-flow correctness.

Usage:
    cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
    python -m spec.bgda.bench.phase0_microbench --out spec/bgda/bench/phase0_results.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import Tuple

import torch
import torch.nn.functional as F

from spec.bgda.reference import bgda_block_attention_reference, gdn_state_update


# Wan2.1-1.3B reference shape:  dim=1536, num_heads=12, head_dim=128,
# num_layers=30, vae_stride=(4,8,8), patch=(1,2,2).
# 480P @ 5s @ 16fps:
#   pixel: 480 × 832 × 80      latent: 60 × 104 × 20  → patched: 30 × 52 × 20
#   = 31,200 tokens / sample.  HW = 30 * 52 = 1560.   F_latent = 20.
WAN_1_3B = dict(B=1, F=20, HW=30 * 52, H=12, d=128)


def _bench_one(
    cfg: dict, W_latent: int, dtype: torch.dtype, device: torch.device, n_iter: int = 5,
    alpha_value: float = 0.99, beta_value: float = 0.0067,
) -> Tuple[float, float, torch.Tensor]:
    """Time one forward pass and return (avg_seconds, S_norm_final, S_final).

    Default beta_value=0.0067 matches sigmoid(-5) — the BGDA init.  Use
    beta_value=0.5 for a stress test (early/mid-training behavior).
    """
    B, Fnum, HW, H, d = cfg["B"], cfg["F"], cfg["HW"], cfg["H"], cfg["d"]
    g = torch.Generator(device=device).manual_seed(0)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype, device=device)
    q = F.normalize(rand(B, Fnum, HW, H, d), dim=-1)
    k = F.normalize(rand(B, Fnum, HW, H, d), dim=-1)
    v = rand(B, Fnum, HW, H, d) * 0.1                                  # values modest
    alpha = torch.full((B, Fnum, H), alpha_value, dtype=dtype, device=device)
    beta = torch.full((B, Fnum, HW, H), beta_value, dtype=dtype, device=device)

    # Warmup
    for _ in range(2):
        _ = bgda_block_attention_reference(q, k, v, alpha, beta, W_latent=W_latent)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out, S_final = bgda_block_attention_reference(
            q, k, v, alpha, beta, W_latent=W_latent, use_recurrence=True, return_state=True
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), S_final.norm().item(), S_final


def _bench_recurrence_only(cfg: dict, dtype, device, n_iter=5) -> float:
    """Time JUST the W_latent state-update steps (cost-isolation per plan §5.1)."""
    B, _, HW, H, d = cfg["B"], cfg["F"], cfg["HW"], cfg["H"], cfg["d"]
    g = torch.Generator(device=device).manual_seed(0)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype, device=device)
    S = torch.zeros(B, H, d, d, device=device, dtype=dtype)
    k = F.normalize(rand(B, HW, H, d), dim=-1)
    v = rand(B, HW, H, d) * 0.1
    alpha = torch.full((B, H), 0.99, dtype=dtype, device=device)
    beta = torch.full((B, HW, H), 0.5, dtype=dtype, device=device)
    for _ in range(2):
        _ = gdn_state_update(S, k, v, alpha, beta)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        S = gdn_state_update(S, k, v, alpha, beta)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def _wan_rope_params(max_seq_len, dim, theta=10000):
    """Inlined copy of `wan.modules.model.rope_params` so we can call it
    without importing the wan package (which eagerly calls torch.cuda.current_device)."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _wan_rope_apply(x, grid_sizes, freqs):
    """Inlined copy of `wan.modules.model.rope_apply`."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def _check_l2_after_rope(device, dtype) -> Tuple[float, float]:
    """Verify L2 norm preservation under Wan's 3D RoPE.  Returns (mean_norm, max_dev)."""
    rope_apply = _wan_rope_apply
    rope_params = _wan_rope_params
    F_, H_, W_ = 20, 30, 52
    L = F_ * H_ * W_
    n_heads, d = 12, 128
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(1, L, n_heads, d, generator=g, dtype=torch.float32, device=device)
    x = F.normalize(x, dim=-1)
    grid_sizes = torch.tensor([[F_, H_, W_]], device=device)
    # Match Wan's freqs construction recipe (see WanModel.__init__):
    # freqs splits the head_dim into [d - 4*(d//6), 2*(d//6), 2*(d//6)] for
    # (temporal, height, width), each giving half-as-many columns of polar
    # (since rope_params(L, dim) returns [L, dim/2]).
    freqs = torch.cat(
        [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ],
        dim=1,
    ).to(device)
    out = rope_apply(x, grid_sizes, freqs)
    norms = out.norm(dim=-1)
    return norms.mean().item(), (norms - 1.0).abs().max().item()


def _check_grad_flow(device, dtype) -> bool:
    """Tiny single-layer model: confirm grads flow and loss decreases on toy data."""
    cfg = dict(B=1, F=4, HW=16, H=2, d=8)
    g = torch.Generator(device=device).manual_seed(0)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype, device=device, requires_grad=False)
    q = F.normalize(rand(cfg["B"], cfg["F"], cfg["HW"], cfg["H"], cfg["d"]), dim=-1)
    k = F.normalize(rand(cfg["B"], cfg["F"], cfg["HW"], cfg["H"], cfg["d"]), dim=-1)
    v = rand(cfg["B"], cfg["F"], cfg["HW"], cfg["H"], cfg["d"])
    alpha_logits = torch.zeros(cfg["B"], cfg["F"], cfg["H"], device=device, dtype=dtype, requires_grad=True)
    beta_logits = torch.zeros(cfg["B"], cfg["F"], cfg["HW"], cfg["H"], device=device, dtype=dtype, requires_grad=True)
    target = torch.randn_like(v) * 0.1
    opt = torch.optim.Adam([alpha_logits, beta_logits], lr=1e-2)
    losses = []
    for _ in range(50):
        a = torch.sigmoid(alpha_logits)
        b = torch.sigmoid(beta_logits)
        out = bgda_block_attention_reference(q, k, v, a, b, W_latent=2, use_recurrence=True)
        loss = ((out - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses[-1] < losses[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="spec/bgda/bench/phase0_results.csv")
    p.add_argument("--small", action="store_true",
                   help="use small shapes (for MIG / CPU smoke)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[microbench] device={device}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"[microbench] gpu={torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cfg = WAN_1_3B if not args.small else dict(B=1, F=8, HW=32 * 32, H=12, d=128)

    rows = []
    # 1) Recurrence cost vs total forward, varying W_latent + gate regime
    print("\n--- (A) Recurrence cost / S-norm over W_latent and β regime ---")
    for W_latent in (1, 2, 5, 10):
        if cfg["F"] % W_latent != 0:
            continue
        N = cfg["F"] // W_latent
        for beta_v, label in [(0.0067, "init"), (0.05, "early"), (0.5, "stress")]:
            avg_full, S_norm, _ = _bench_one(cfg, W_latent, dtype, device, beta_value=beta_v)
            rec_per_frame = _bench_recurrence_only(cfg, dtype, device)
            rec_total = rec_per_frame * cfg["F"]
            pct = 100.0 * rec_total / max(avg_full, 1e-9)
            print(
                f"  W_latent={W_latent:2d}  N={N:2d}  β={beta_v:.4f} ({label:6s})"
                f"  forward={avg_full*1e3:8.2f}ms  recurr_total={rec_total*1e3:6.2f}ms ({pct:5.2f}%)"
                f"  ‖S‖_F={S_norm:.3g}"
            )
            rows.append({
                "test": "recurrence_cost",
                "W_latent": W_latent,
                "N_blocks": N,
                "beta_value": beta_v,
                "regime": label,
                "forward_ms": avg_full * 1e3,
                "recurrence_total_ms": rec_total * 1e3,
                "recurrence_pct": pct,
                "S_norm_final": S_norm,
                "dtype": str(dtype),
            })

    # 2) Memory: state size sanity
    state_bytes = 1 * cfg["H"] * cfg["d"] * cfg["d"] * (2 if dtype == torch.bfloat16 else 4)
    print(f"  state bytes per layer = {state_bytes:,}  ({state_bytes/1024:.1f} KiB)")
    rows.append({
        "test": "state_bytes_per_layer",
        "value": state_bytes,
        "value_kib": state_bytes / 1024,
        "dtype": str(dtype),
    })

    # 3) RoPE + L2 norm preservation (only if Wan available)
    mean_norm, max_dev = _check_l2_after_rope(device, dtype)
    print(f"  rope L2 preservation: mean‖q_after_rope‖={mean_norm:.6f}, max|‖q‖−1|={max_dev:.2e}")
    rows.append({
        "test": "rope_l2_preservation",
        "mean_norm": mean_norm,
        "max_deviation": max_dev,
    })

    # 4) Single-layer gradient flow / loss decrease
    decreased = _check_grad_flow(device, dtype if dtype != torch.bfloat16 else torch.float32)
    print(f"  single-layer toy training: loss decreased = {decreased}")
    rows.append({"test": "loss_decrease_toy_training", "decreased": int(decreased)})

    # write CSV
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[microbench] wrote {args.out}")


if __name__ == "__main__":
    main()
