"""Init-recovery and block-causality tests for BGDA."""
import torch

from spec.bgda.reference import bgda_block_attention_reference


def _make(B=1, F=6, HW=4, H=2, d=8, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    q = torch.nn.functional.normalize(rand(B, F, HW, H, d), dim=-1)
    k = torch.nn.functional.normalize(rand(B, F, HW, H, d), dim=-1)
    v = rand(B, F, HW, H, d)
    return q, k, v


def test_init_alpha1_beta0_freezes_state():
    """At α=1, β=0 with use_recurrence=True, S stays at initial value (zero).
    Each block reads q · (0 + K_block^T V_block) — per-block bidirectional only,
    no cross-block carry."""
    q, k, v = _make()
    B, F_, HW, H, d = q.shape
    alpha = torch.ones(B, F_, H, dtype=q.dtype)
    beta = torch.zeros(B, F_, HW, H, dtype=q.dtype)
    out, S_final = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=2, use_recurrence=True, return_state=True,
    )
    # State should be exactly zero (well, after no updates).
    assert torch.allclose(S_final, torch.zeros_like(S_final), atol=1e-12)
    # Each block's output should match a per-block bidirectional read.
    for t_blk in range(F_ // 2):
        f0, f1 = t_blk * 2, (t_blk + 1) * 2
        q_b = q[:, f0:f1].reshape(B, 2 * HW, H, d)
        k_b = k[:, f0:f1].reshape(B, 2 * HW, H, d)
        v_b = v[:, f0:f1].reshape(B, 2 * HW, H, d)
        S_b = torch.einsum("bnhd,bnhe->bhde", k_b, v_b)
        out_b = torch.einsum("bnhd,bhde->bnhe", q_b, S_b).view(B, 2, HW, H, d)
        assert torch.allclose(out[:, f0:f1], out_b, atol=1e-10)


def test_block_causality_changing_future_does_not_change_past():
    """When recurrence is on, output of block t must depend ONLY on blocks ≤ t.

    Verify by perturbing v in a future block and checking past-block outputs
    are byte-equal."""
    q, k, v = _make()
    B, F_, HW, H, d = q.shape
    alpha = torch.full((B, F_, H), 0.95, dtype=q.dtype)
    beta = torch.full((B, F_, HW, H), 0.5, dtype=q.dtype)

    out1 = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=2, use_recurrence=True
    )
    # Perturb v in the LAST block only.
    v2 = v.clone()
    v2[:, F_ - 2:] += 1.0
    out2 = bgda_block_attention_reference(
        q, k, v2, alpha, beta, W_latent=2, use_recurrence=True
    )
    # All blocks except the last must be byte-equal.
    last_block_start = F_ - 2
    assert torch.equal(out1[:, :last_block_start], out2[:, :last_block_start])
    # Last block must differ.
    assert not torch.allclose(out1[:, last_block_start:], out2[:, last_block_start:])


def test_state_bounded_random_inputs():
    """Plan §4.4: ‖S‖_F should stay bounded over many blocks of random L2-normed
    keys.  Use realistic α=0.99, β=0.5."""
    q, k, v = _make(B=1, F=24, HW=8, H=4, d=16, seed=42)  # 24 frames → 12 blocks of W=2
    B, F_, HW, H, d = q.shape
    alpha = torch.full((B, F_, H), 0.99, dtype=q.dtype)
    beta = torch.full((B, F_, HW, H), 0.5, dtype=q.dtype)
    _, S_final = bgda_block_attention_reference(
        q, k, v, alpha, beta, W_latent=2, use_recurrence=True, return_state=True,
    )
    norm = S_final.norm()
    assert torch.isfinite(norm).all()
    # Plan §4.4 says flag if ‖S‖_F > 100; for this small head_dim it should be
    # comfortably below.
    assert norm.item() < 100.0, f"‖S‖_F = {norm.item():.2f} too large"
