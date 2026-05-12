"""β-budget normalization tests.

Phase-0 microbench showed that β=0.5 with HW=1560 explodes ‖S‖_F to 1e15.
Verify the `beta_normalize='hw'` knob keeps it bounded for the same input."""
import torch

from spec.bgda.reference import bgda_block_attention_reference


def _stress_inputs(HW=400, F_=8, H=2, d=8, dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    q = torch.nn.functional.normalize(rand(1, F_, HW, H, d), dim=-1)
    k = torch.nn.functional.normalize(rand(1, F_, HW, H, d), dim=-1)
    v = rand(1, F_, HW, H, d) * 0.1
    alpha = torch.full((1, F_, H), 0.99, dtype=dtype)
    beta = torch.full((1, F_, HW, H), 0.5, dtype=dtype)
    return q, k, v, alpha, beta


def test_no_normalize_diverges_at_high_beta():
    """Sanity: confirm the regime we're protecting against actually exists."""
    q, k, v, alpha, beta = _stress_inputs(HW=400)
    _, S = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=2, use_recurrence=True,
        beta_normalize=None, return_state=True,
    )
    # With HW=400 and β=0.5, even at modest scale the state should grow large.
    assert S.norm() > 1e3, f"expected divergence, got ‖S‖_F={S.norm().item():.3g}"


def test_hw_normalize_stays_bounded():
    """beta_normalize='hw' divides β by HW → erasure operator stays well-conditioned."""
    q, k, v, alpha, beta = _stress_inputs(HW=400)
    _, S = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=2, use_recurrence=True,
        beta_normalize="hw", return_state=True,
    )
    assert torch.isfinite(S).all()
    assert S.norm() < 100.0, f"expected bounded, got ‖S‖_F={S.norm().item():.3g}"


def test_sqrt_hw_normalize_intermediate():
    """beta_normalize='sqrt_hw' is in between."""
    q, k, v, alpha, beta = _stress_inputs(HW=400)
    _, S_full = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=2, use_recurrence=True,
        beta_normalize=None, return_state=True,
    )
    _, S_sqrt = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=2, use_recurrence=True,
        beta_normalize="sqrt_hw", return_state=True,
    )
    _, S_hw = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=2, use_recurrence=True,
        beta_normalize="hw", return_state=True,
    )
    n_full, n_sqrt, n_hw = S_full.norm().item(), S_sqrt.norm().item(), S_hw.norm().item()
    # ordering: hw <= sqrt_hw <= full (more normalization → smaller state)
    assert n_hw < n_sqrt < n_full, (
        f"unexpected ordering: hw={n_hw:.3g}, sqrt={n_sqrt:.3g}, full={n_full:.3g}"
    )
