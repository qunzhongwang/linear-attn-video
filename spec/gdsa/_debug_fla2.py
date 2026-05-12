"""Find the FLA convention by comparing single-token cases against hand-computed references."""
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def manual_step(S, k, v, beta, alpha=1.0):
    """One delta step: S = α (S (I - β k k^T) + β v k^T). k,v shape (D,)."""
    D = k.shape[-1]
    Sk = S @ k.unsqueeze(-1)            # (D, 1)
    S = S - beta * (Sk @ k.unsqueeze(0)) + beta * (v.unsqueeze(-1) @ k.unsqueeze(0))
    return alpha * S


def test_single_token():
    """Single token, β=0.5, α=1, no L2 norm. Reference vs FLA."""
    torch.manual_seed(0)
    B, T, H, D = 1, 1, 1, 4
    device = "cuda"

    q = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    g = torch.zeros(B, T, H, device=device, dtype=torch.float32)        # no decay
    beta_t = torch.tensor([[[0.5]]], device=device, dtype=torch.float32)  # (B, T, H)

    # FLA call — no L2 norm on either side
    o_fla, S_fla = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta_t, scale=1.0,
        initial_state=torch.zeros(B, H, D, D, device=device, dtype=torch.float32),
        output_final_state=True,
    )

    # Manual: single token, S_0=0 so S_1 = β v k^T
    S_manual = manual_step(
        torch.zeros(D, D, device=device, dtype=torch.float32),
        k[0, 0, 0], v[0, 0, 0], 0.5, 1.0,
    )
    o_manual = q[0, 0, 0] @ S_manual

    print("=== single token, β=0.5, no decay ===")
    print("S_fla[0,0]:\n", S_fla[0, 0])
    print("S_manual:\n", S_manual)
    print(f"S diff max: {(S_fla[0, 0] - S_manual).abs().max().item():.3e}")
    print(f"o diff max: {(o_fla[0, 0, 0] - o_manual).abs().max().item():.3e}")


def test_with_l2norm():
    """Same but with L2-normalized q, k. Reference uses our L2-norm convention."""
    torch.manual_seed(0)
    B, T, H, D = 1, 4, 1, 4
    device = "cuda"
    q = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    g = torch.zeros(B, T, H, device=device, dtype=torch.float32)
    beta_t = 0.5 * torch.ones(B, T, H, device=device, dtype=torch.float32)

    qn = F.normalize(q, p=2, dim=-1)
    kn = F.normalize(k, p=2, dim=-1)

    # FLA path: pass NORMALIZED q, k
    o_fla, S_fla = chunk_gated_delta_rule(
        q=qn, k=kn, v=v, g=g, beta=beta_t, scale=1.0,
        initial_state=torch.zeros(B, H, D, D, device=device, dtype=torch.float32),
        output_final_state=True,
    )

    # Manual path
    S = torch.zeros(D, D, device=device, dtype=torch.float32)
    for t in range(T):
        S = manual_step(S, kn[0, t, 0], v[0, t, 0], 0.5, 1.0)
    o_manual = qn[0, T-1, 0] @ S

    print("\n=== 4 tokens, L2-normed q,k, β=0.5, no decay ===")
    print(f"S diff max: {(S_fla[0, 0] - S).abs().max().item():.3e}")
    print(f"o[T-1] diff max: {(o_fla[0, T-1, 0] - o_manual).abs().max().item():.3e}")
    print("S_fla[0,0] norm:", S_fla[0, 0].norm().item())
    print("S_manual norm: ", S.norm().item())


def test_with_internal_l2norm():
    """Let FLA do the L2 norm internally. Compare to manual that L2-norms identically."""
    torch.manual_seed(0)
    B, T, H, D = 1, 4, 1, 4
    device = "cuda"
    q = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    g = torch.zeros(B, T, H, device=device, dtype=torch.float32)
    beta_t = 0.5 * torch.ones(B, T, H, device=device, dtype=torch.float32)

    o_fla, S_fla = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta_t, scale=1.0,
        initial_state=torch.zeros(B, H, D, D, device=device, dtype=torch.float32),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    qn = F.normalize(q, p=2, dim=-1)
    kn = F.normalize(k, p=2, dim=-1)
    S = torch.zeros(D, D, device=device, dtype=torch.float32)
    for t in range(T):
        S = manual_step(S, kn[0, t, 0], v[0, t, 0], 0.5, 1.0)
    o_manual = qn[0, T-1, 0] @ S

    print("\n=== 4 tokens, FLA-internal L2 norm (use_qk_l2norm_in_kernel=True) ===")
    print(f"S diff max: {(S_fla[0, 0] - S).abs().max().item():.3e}")
    print(f"o[T-1] diff max: {(o_fla[0, T-1, 0] - o_manual).abs().max().item():.3e}")


if __name__ == "__main__":
    test_single_token()
    test_with_l2norm()
    test_with_internal_l2norm()
