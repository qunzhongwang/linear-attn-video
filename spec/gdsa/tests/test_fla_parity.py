"""FLA-backed gdsa_fla ≡ reference gdsa_chunked, on GPU only (FLA's Triton kernels).

Skipped when CUDA isn't available or `fla` import failed.
"""
import pytest
import torch

from spec.gdsa.tests.conftest import make_qkv, make_gates


def _has_fla():
    try:
        from spec.gdsa.fla_chunked import HAS_FLA
        return HAS_FLA and torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_fla(), reason="needs CUDA + flash-linear-attention")
def test_fla_parity_fp32():
    from spec.gdsa.chunked import gdsa_chunked
    from spec.gdsa.fla_chunked import gdsa_fla

    shape = dict(B=2, H=4, T=24, D=32, chunk_sizes=[6, 6, 6, 6])
    device = torch.device("cuda")
    Q, K, V = make_qkv(shape, dtype=torch.float32, device=device)
    alpha, beta = make_gates(shape, dtype=torch.float32, device=device)

    out_r, S_r, Z_r = gdsa_chunked(Q, K, V, alpha, beta, chunk_sizes=shape["chunk_sizes"])
    out_f, S_f, Z_f = gdsa_fla(Q, K, V, alpha, beta, chunk_sizes=shape["chunk_sizes"])

    diff_out = (out_r - out_f).abs().max().item()
    diff_S = (S_r - S_f).abs().max().item()

    # FLA fp32 should be very close. Tolerance accounts for chunked-kernel reduction order.
    assert diff_out < 5e-3, f"FLA out diff {diff_out} too large"
    assert diff_S < 5e-3, f"FLA S diff {diff_S} too large"
    # Z is computed on the slow path in fla_chunked, so should match exactly.
    assert torch.allclose(Z_r, Z_f, atol=1e-6)
