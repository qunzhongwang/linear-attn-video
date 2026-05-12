"""Block-causality: perturbing chunk t can't change output of chunks <t."""
import pytest
import torch

from spec.gdsa.reference import gdsa_reference, vanilla_reference
from spec.gdsa.tests.conftest import make_qkv, make_gates


def _causal_check(fn, Q, K, V, chunk_sizes, perturb_chunk_idx, **kwargs):
    out_a, _, _ = fn(Q, K, V, chunk_sizes=chunk_sizes, **kwargs)

    # Perturb K, V only inside chunk `perturb_chunk_idx`. Output of all earlier chunks must match.
    pos = sum(chunk_sizes[:perturb_chunk_idx])
    end = pos + chunk_sizes[perturb_chunk_idx]
    K2 = K.clone(); V2 = V.clone()
    K2[:, :, pos:end, :] += 1.0
    V2[:, :, pos:end, :] += 1.0
    out_b, _, _ = fn(Q, K2, V2, chunk_sizes=chunk_sizes, **kwargs)

    pre = sum(chunk_sizes[:perturb_chunk_idx])
    assert torch.allclose(out_a[:, :, :pre, :], out_b[:, :, :pre, :], atol=1e-12, rtol=0), \
        "earlier chunks were perturbed by changing a later chunk — causality violated"


def test_causality_vanilla(tiny_shape, device):
    Q, K, V = make_qkv(tiny_shape, dtype=torch.float64, device=device)
    _causal_check(vanilla_reference, Q, K, V, tiny_shape["chunk_sizes"], perturb_chunk_idx=2)


def test_causality_gdsa(tiny_shape, device):
    Q, K, V = make_qkv(tiny_shape, dtype=torch.float64, device=device)
    alpha, beta = make_gates(tiny_shape, dtype=torch.float64, device=device)
    _causal_check(
        gdsa_reference, Q, K, V, tiny_shape["chunk_sizes"], perturb_chunk_idx=2,
        alpha=alpha, beta=beta,
    )
