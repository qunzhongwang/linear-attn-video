"""Tiny shapes that fit on CPU. GPU-specific tests opt-in via @pytest.mark.cuda."""
import pytest
import torch


@pytest.fixture(scope="session")
def tiny_shape():
    """(B, H, T, D, chunk_sizes). T must == sum(chunk_sizes)."""
    return dict(B=2, H=2, T=12, D=8, chunk_sizes=[3, 4, 5])


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def rng():
    g = torch.Generator()
    g.manual_seed(0)
    return g


def make_qkv(shape: dict, dtype=torch.float64, device=None):
    """ReLU-mapped (non-negative) Q, K, and unconstrained V — matches SANA's φ(x)=ReLU(x)."""
    B, H, T, D = shape["B"], shape["H"], shape["T"], shape["D"]
    g = torch.Generator(device="cpu").manual_seed(42)
    Q = torch.randn(B, H, T, D, generator=g, dtype=torch.float64).abs().to(dtype=dtype, device=device)
    K = torch.randn(B, H, T, D, generator=g, dtype=torch.float64).abs().to(dtype=dtype, device=device)
    V = torch.randn(B, H, T, D, generator=g, dtype=torch.float64).to(dtype=dtype, device=device)
    return Q, K, V


def make_gates(shape: dict, dtype=torch.float64, device=None, alpha_val=None, beta_val=None):
    """Per-chunk α, β. Pass scalars to pin them; otherwise random in (0,1)."""
    B, H = shape["B"], shape["H"]
    n = len(shape["chunk_sizes"])
    g = torch.Generator(device="cpu").manual_seed(123)
    if alpha_val is not None:
        alpha = torch.full((B, H, n), float(alpha_val), dtype=dtype, device=device)
    else:
        alpha = torch.sigmoid(torch.randn(B, H, n, generator=g, dtype=torch.float64)).to(dtype=dtype, device=device)
    if beta_val is not None:
        beta = torch.full((B, H, n), float(beta_val), dtype=dtype, device=device)
    else:
        beta = torch.sigmoid(torch.randn(B, H, n, generator=g, dtype=torch.float64)).to(dtype=dtype, device=device)
    return alpha, beta
