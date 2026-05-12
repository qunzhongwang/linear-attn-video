"""Micro-benchmark: vanilla_chunked vs gdsa_chunked at SANA-Video Stage-3 shapes.

Run on 1×A100 80G:
    PYTHONPATH=tools/_compat python spec/gdsa/bench.py
"""
import time
import torch

from spec.gdsa.chunked import vanilla_chunked, gdsa_chunked


def bench(fn, *args, warmup=3, repeats=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000  # ms


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # SANA-Video 480p 5s shapes (per Sana_2000M_480px config):
    # heads=20, head_dim=112, batch=1, n_blocks_per_layer ≈ 21 latent frames * 60 * 104 / blocks
    # For Stage-3: 21 frames per block, latent (60, 104) at 480p, single block at a time.
    # Total tokens per block (single layer) ≈ 21 * 60 * 104 / something — we'll use a simple ladder.

    print(f"{'name':<50}{'time(ms)':>12}")
    print("-" * 64)
    for B, H, D, chunk_size, n_chunks in [
        (1, 20, 112, 256, 4),
        (1, 20, 112, 512, 4),
        (1, 20, 112, 1024, 8),
        (1, 20, 112, 2048, 8),
    ]:
        T = chunk_size * n_chunks
        chunk_sizes = [chunk_size] * n_chunks
        Q = torch.randn(B, H, T, D, device=device, dtype=dtype).abs()
        K = torch.randn(B, H, T, D, device=device, dtype=dtype).abs()
        V = torch.randn(B, H, T, D, device=device, dtype=dtype)
        alpha = torch.sigmoid(torch.randn(B, H, n_chunks, device=device, dtype=dtype))
        beta = torch.sigmoid(torch.randn(B, H, n_chunks, device=device, dtype=dtype))

        t_v = bench(vanilla_chunked, Q, K, V, chunk_sizes)
        t_g = bench(gdsa_chunked, Q, K, V, alpha, beta, chunk_sizes)

        cfg = f"B={B} H={H} D={D} n_per_chunk={chunk_size} n_chunks={n_chunks}"
        print(f"{cfg:<50}vanilla={t_v:>5.1f}  gdsa={t_g:>5.1f}  ratio={t_g/t_v:>4.2f}x")


if __name__ == "__main__":
    main()
