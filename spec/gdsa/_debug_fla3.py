"""Side-by-side: gdsa_chunked vs gdsa_fla with fixed reference layout.

Goal: pinpoint the residual ~2.0 difference in the parity test.
"""
import torch
import torch.nn.functional as F


def main():
    from spec.gdsa.chunked import gdsa_chunked
    from spec.gdsa.fla_chunked import gdsa_fla

    device = "cuda"
    dtype = torch.float32

    # Tiny case: 1 batch, 1 head, 4 tokens in 1 chunk
    B, H, T, D = 1, 1, 4, 8
    chunk_sizes = [T]
    torch.manual_seed(7)
    Q = torch.randn(B, H, T, D, device=device, dtype=dtype).abs()
    K = torch.randn(B, H, T, D, device=device, dtype=dtype).abs()
    V = torch.randn(B, H, T, D, device=device, dtype=dtype)
    alpha = torch.full((B, H, 1), 1.0, device=device, dtype=dtype)
    beta = torch.full((B, H, 1), 0.5, device=device, dtype=dtype)

    print("=== qk_l2norm=True ===")
    out_c, S_c, Z_c = gdsa_chunked(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes)
    out_f, S_f, Z_f = gdsa_fla(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes)
    print(f"S diff max: {(S_c - S_f).abs().max().item():.3e}")
    print(f"Z diff max: {(Z_c - Z_f).abs().max().item():.3e}")
    print(f"out diff max: {(out_c - out_f).abs().max().item():.3e}")
    print("S_c sample:\n", S_c[0, 0, :3, :3])
    print("S_f sample:\n", S_f[0, 0, :3, :3])

    print("\n=== qk_l2norm=False ===")
    out_c, S_c, Z_c = gdsa_chunked(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes, qk_l2norm=False)
    out_f, S_f, Z_f = gdsa_fla(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes, qk_l2norm=False)
    print(f"S diff max: {(S_c - S_f).abs().max().item():.3e}")
    print(f"Z diff max: {(Z_c - Z_f).abs().max().item():.3e}")
    print(f"out diff max: {(out_c - out_f).abs().max().item():.3e}")
    print("S_c sample:\n", S_c[0, 0, :3, :3])
    print("S_f sample:\n", S_f[0, 0, :3, :3])

    print("\n=== bigger: B=2, H=4, T=24, D=32, chunks=[6,6,6,6] ===")
    B, H, T, D = 2, 4, 24, 32
    chunk_sizes = [6, 6, 6, 6]
    Q = torch.randn(B, H, T, D, device=device, dtype=dtype).abs()
    K = torch.randn(B, H, T, D, device=device, dtype=dtype).abs()
    V = torch.randn(B, H, T, D, device=device, dtype=dtype)
    alpha = torch.sigmoid(torch.randn(B, H, len(chunk_sizes), device=device, dtype=dtype))
    beta = torch.sigmoid(torch.randn(B, H, len(chunk_sizes), device=device, dtype=dtype))

    out_c, S_c, Z_c = gdsa_chunked(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes)
    out_f, S_f, Z_f = gdsa_fla(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes)
    print(f"l2norm=True:  S {(S_c - S_f).abs().max():.3e}, Z {(Z_c - Z_f).abs().max():.3e}, out {(out_c - out_f).abs().max():.3e}")

    out_c, S_c, Z_c = gdsa_chunked(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes, qk_l2norm=False)
    out_f, S_f, Z_f = gdsa_fla(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes, qk_l2norm=False)
    print(f"l2norm=False: S {(S_c - S_f).abs().max():.3e}, Z {(Z_c - Z_f).abs().max():.3e}, out {(out_c - out_f).abs().max():.3e}")


if __name__ == "__main__":
    main()
