# GDSA — Gated Delta Split Attention spec

Reference + chunked + tested implementation of Mechanism #1 (Gated Delta) from
`linear-video/gdsa_iclr2027.pdf`. Mechanisms #2 (split) and #3 (merge+aggregate) are intentionally
out of scope for v1; only the gated-delta recurrence is implemented and tested here.

## Layout

```
spec/gdsa/
├── reference.py            # Naive PyTorch recurrence; correct, slow. Ground truth for tests.
├── chunked.py              # Vectorized PyTorch chunked implementation. Fast on GPU. No custom kernels.
├── gdsa_attention.py       # nn.Module wrapper that mirrors `CachedCausalAttention` so we can drop it in.
├── tests/                  # pytest suite — see "Tests" below
└── README.md
```

## Tensor layout (matches SANA's `CachedCausalAttention`)

Per chunk:
- `Q, K, V`: `(B, h, h_d, n)` (head-dim, then sequence length last)
- We transpose to `(B, h, n, h_d)` internally for the recurrence math.

Per-chunk state (post-update full S, NOT a per-chunk contribution):
- `S`:    `(B, h, h_d, h_d)`   — accumulator analogous to `vk` in vanilla, but updated via gated delta.
- `Z`:    `(B, h, 1,  h_d)`    — normalizer analogous to `k_sum`.

For GDSA, **the cache stores the post-update S, not the contribution**. Vanilla SANA could store
contributions because Σ-additive across chunks; gated delta is non-additive (each chunk's update
depends on the previous S), so the state itself must be threaded.

## Recurrence (Mechanism #1 only)

Per chunk t with kernel-mapped keys/values `(K_t, V_t)`, scalar gates `α_t, β_t ∈ (0,1)`:

```
# token-by-token within a chunk (reference impl)
S ← S_{t-1}
for i in 0..n_t-1:
    k_i = K_t[i]                  # (h_d,)
    v_i = V_t[i]                  # (h_d,)
    S ← S * (I - β_t * k_i k_iᵀ) + β_t * v_i k_iᵀ
S_t = α_t * S
```

The `I - β k k^T` is a Householder-like rank-1 erasure of the association at key `k`. With
α=1, β=0 the recurrence reduces to `S_t = S_{t-1}` (NOT vanilla SANA — see next section).

## Vanilla parity at init

The proposal's "init α≈1, β≈0 recovers vanilla SANA" is not exact for the gated-delta-only update
because vanilla SANA is `S_t = S_{t-1} + φ(K_t)ᵀ V_t` (additive write), whereas GDSA with β≈0 does
`S_t ≈ α S_{t-1}` (no write at all).

We provide a clean parity by toggling a flag `mode='vanilla'` in `chunked.py` that swaps the
gated-delta recurrence for the additive vanilla one. Tests assert the chunked impl in vanilla mode
matches the existing `CachedCausalAttention` math byte-for-byte at α=1, β=0.

For the *learned* GDSA module, we initialize gate MLPs so that the **output** matches vanilla SANA
at step 0 by zero-initializing both the gate-projection bias and the output-projection delta path,
and routing through the residual additive path until the gates "open." Detailed init in
`gdsa_attention.py`.

## Tests (run on 1×A100 ≤1h or MIG)

| Test file | What it asserts |
|---|---|
| `test_vanilla_parity.py` | `chunked(mode='vanilla') ≡ CachedCausalAttention` byte-for-byte (atol 1e-5 in fp32, 1e-3 bf16). |
| `test_recurrence_parity.py` | `chunked(mode='gdsa') ≡ reference(mode='gdsa')` for random α, β. |
| `test_backward_parity.py` | `gradcheck(reference)` passes; `chunked.backward ≡ reference.backward` for random inputs. |
| `test_causality.py` | Changing chunk t's K, V doesn't perturb chunks <t. |
| `test_constant_memory.py` | `S` and `Z` sizes are invariant in N (constant-memory property preserved). |
| `test_init_recovery.py` | A `GDSAAttention` module with all gate weights → 0 produces output close to vanilla `CachedCausalAttention` (after equivalent fwd). |

```bash
cd /scratch/gpfs/ZHUANGL/qw3460/workspace/linear-video
PYTHONPATH=tools/_compat pytest spec/gdsa/tests -q
```

## Triton / fusion decision

**Pure-PyTorch within-chunk loop is unusably slow** — confirmed by `bench.py` on H100:

| shape (B, H, D, n/chunk, chunks) | vanilla_chunked | gdsa_chunked | ratio |
|---|---|---|---|
| 1, 20, 112,  256, 4 |  0.7 ms |  200.7 ms |  **305×** slower |
| 1, 20, 112,  512, 4 |  0.7 ms |  400.1 ms |  **609×** slower |
| 1, 20, 112, 1024, 8 |  1.3 ms | 1611.2 ms | **1210×** slower |
| 1, 20, 112, 2048, 8 |  1.3 ms | 3208.0 ms | **2388×** slower |

Verdict: **GDSA-1 training requires a fused/chunkwise kernel before Stage-4**. The Python
within-chunk loop in `chunked.py` is fine for parity testing but cannot be used in real training.

We use `flash-linear-attention` (FLA, ICLR'25 reference impl from the Gated DeltaNet authors).
Module: `spec/gdsa/fla_chunked.py`. API:

```python
from spec.gdsa.fla_chunked import gdsa_fla
out, S, Z = gdsa_fla(Q, K, V, alpha, beta, chunk_sizes)   # same signature as gdsa_chunked
```

FLA computes only the un-normalized `φ(q) S` path; we run our slow chunked impl just for `Z`
(it's `(B,H,1,D)` and contributes ~zero compute). Per-chunk scalar gates are mapped to
per-token gates: `g = log α` on the LAST token of each chunk (zero elsewhere); `β` is
broadcast across the chunk's tokens.

A FLA-vs-reference parity test lives at `tests/test_fla_parity.py` (skipped without CUDA).
Tolerance is 5e-3 in fp32 to allow for chunked-kernel reduction-order drift.

**Status: parity passes on H100 in fp32. Microbench (after debugging):**

| shape | vanilla | pyloop | fla | fla speedup |
|---|---|---|---|---|
| n_per_chunk=256, chunks=4   |  0.66 ms |  202.6 ms |  89.5 ms | 2.3× |
| n_per_chunk=512, chunks=4   |  0.66 ms |  402.7 ms | 174.4 ms | 2.3× |
| n_per_chunk=1024, chunks=8  |  1.38 ms | 1612.8 ms | 689.5 ms | 2.3× |
| n_per_chunk=2048, chunks=8  |  1.34 ms | 3235.1 ms | 1383.4 ms | 2.3× |

Modest at small chunk sizes due to per-call FLA launch overhead. SANA-Video Stage-3 chunks
are O(100K) tokens, so the gap should narrow significantly. **Custom Triton fusion** stays on
the table if real-data FLA latency is still a bottleneck — TBD after Stage-4 first profiling.

Custom Triton kernel — only if FLA's kernel turns out to be a bottleneck of its own
(e.g. SANA-Video's specific `(B, H, T, D)` shapes don't hit FLA's optimal block sizes).
TBD after Stage-4 first profiling.
