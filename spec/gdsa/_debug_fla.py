"""Debug FLA recurrence vs reference at multiple shapes."""
import torch
from spec.gdsa.reference import gdsa_reference
from spec.gdsa.tests.conftest import make_qkv, make_gates


def try_chunk(B, H, T, D, chunk_sizes, dtype=torch.float32, device="cuda", impl="chunk"):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    fn = chunk_gated_delta_rule if impl == "chunk" else fused_recurrent_gated_delta_rule

    shape = dict(B=B, H=H, T=T, D=D, chunk_sizes=chunk_sizes)
    Q, K, V = make_qkv(shape, dtype=dtype, device=device)
    alpha, beta = make_gates(shape, dtype=dtype, device=device)

    # Reference (per-chunk; α applied at chunk end; output uses post-update S, Z)
    out_r, S_r, Z_r = gdsa_reference(Q, K, V, alpha, beta, chunk_sizes=chunk_sizes)
    out_r_unnorm = out_r * (Q @ Z_r.transpose(-1, -2)).clamp_min(1e-5)  # remove Z normalization for fair comparison
    # Actually simpler: re-derive (B,H,T,D) of φ(q)·S step by step (without ÷ Z)
    out_S_ref = torch.zeros_like(V)
    S_ref = torch.zeros(B, H, D, D, device=device, dtype=dtype)
    pos = 0
    for t, n_t in enumerate(chunk_sizes):
        a_t = alpha[:, :, t].view(B, H, 1, 1)
        b_t = beta[:, :, t].view(B, H, 1, 1)
        Kt = K[:, :, pos : pos + n_t, :]
        Vt = V[:, :, pos : pos + n_t, :]
        Qt = Q[:, :, pos : pos + n_t, :]
        for i in range(n_t):
            k = Kt[:, :, i:i+1, :]; v = Vt[:, :, i:i+1, :]
            Sk = S_ref @ k.transpose(-1, -2)
            S_ref = S_ref - b_t * (Sk @ k) + b_t * (v.transpose(-1, -2) @ k)
        S_ref = a_t * S_ref
        out_S_ref[:, :, pos:pos+n_t, :] = (a_t * Qt) @ S_ref / a_t  # post-decay query path
        # Actually our reference applies α then queries with post-decay S, so output = Qt @ S_ref (already decayed)
        out_S_ref[:, :, pos:pos+n_t, :] = Qt @ S_ref
        pos += n_t

    # FLA per-chunk call
    state = torch.zeros(B, H, D, D, device=device, dtype=dtype)
    out_S_fla = torch.zeros_like(V)
    pos = 0
    for t, n_t in enumerate(chunk_sizes):
        q = Q[:, :, pos:pos+n_t, :].transpose(1, 2).contiguous()
        k = K[:, :, pos:pos+n_t, :].transpose(1, 2).contiguous()
        v = V[:, :, pos:pos+n_t, :].transpose(1, 2).contiguous()
        g = torch.zeros(B, n_t, H, device=device, dtype=dtype)
        b = beta[:, :, t].unsqueeze(1).expand(-1, n_t, -1).contiguous()
        if impl == "chunk":
            o, state = fn(q=q, k=k, v=v, g=g, beta=b, scale=1.0, initial_state=state, output_final_state=True)
        else:
            o, state = fn(q=q, k=k, v=v, g=g, beta=b, scale=1.0, initial_state=state, output_final_state=True)
        out_S_fla[:, :, pos:pos+n_t, :] = o.transpose(1, 2).contiguous()
        # post-decay
        a_t = alpha[:, :, t].view(B, H, 1, 1)
        state = state * a_t
        out_S_fla[:, :, pos:pos+n_t, :] *= a_t
        pos += n_t

    diff = (out_S_ref - out_S_fla).abs().max().item()
    print(f"impl={impl} B={B} H={H} D={D} chunks={chunk_sizes} max_diff={diff:.3e} ref_max={out_S_ref.abs().max().item():.3e} fla_max={out_S_fla.abs().max().item():.3e}")
    return diff


if __name__ == "__main__":
    # Hypothesis: FLA's chunk kernel has a min chunk size; try increasingly large chunks.
    for impl in ("fused", "chunk"):
        print(f"\n=== impl={impl} ===")
        for cs in [[64], [128], [64, 64], [16, 16], [8, 8], [4, 4]]:
            T = sum(cs)
            try:
                try_chunk(1, 4, T, 32, cs, impl=impl)
            except Exception as e:
                print(f"impl={impl} chunks={cs} -> EXCEPTION: {type(e).__name__}: {e}")
