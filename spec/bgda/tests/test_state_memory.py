"""Tests for Stage D memory-expansion variants."""
import torch

from spec.bgda.state_memory import SingleS, FixedSlotBank, AdaptiveSplit


def _block(B=1, W=2, HW=4, H=2, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    k = torch.nn.functional.normalize(rand(B, W, HW, H, d), dim=-1)
    v = rand(B, W, HW, H, d) * 0.1
    a = torch.full((B, W, H), 0.99, dtype=torch.float64)
    b = torch.full((B, W, HW, H), 0.05, dtype=torch.float64)
    return k, v, a, b


def test_single_s_basic_roundtrip():
    var = SingleS()
    bank = var.init_state(1, 2, 8, torch.float64, torch.device("cpu"))
    assert bank.K == 1
    k, v, a, b = _block()
    bank = var.update(bank, k, v, a, b, t_block=0)
    assert bank.K == 1
    q = torch.randn(1, 2 * 4, 2, 8, dtype=torch.float64)
    out = var.read(bank, q)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_fixed_slot_bank_spawns_then_evicts():
    var = FixedSlotBank(K_max=3, spawn_every_blocks=2)
    bank = var.init_state(1, 2, 8, torch.float64, torch.device("cpu"))
    assert bank.K == 1
    k, v, a, b = _block()
    # block 0 — single slot, no spawn.
    bank = var.update(bank, k, v, a, b, t_block=0)
    assert bank.K == 1
    # block 2 — spawn second slot.
    bank = var.update(bank, k, v, a, b, t_block=2)
    assert bank.K == 2
    # block 4 — spawn third slot.
    bank = var.update(bank, k, v, a, b, t_block=4)
    assert bank.K == 3
    # block 6 — at K_max=3, evict-oldest-via-merge then spawn.
    bank = var.update(bank, k, v, a, b, t_block=6)
    assert bank.K == 3   # still 3 (oldest merged into next-oldest, then 1 new)


def test_fixed_slot_read_sums_across_slots():
    var = FixedSlotBank(K_max=4, spawn_every_blocks=1)
    bank = var.init_state(1, 2, 8, torch.float64, torch.device("cpu"))
    k, v, a, b = _block()
    bank = var.update(bank, k, v, a, b, t_block=0)
    bank = var.update(bank, k, v, a, b, t_block=1)   # spawns slot 1
    bank = var.update(bank, k, v, a, b, t_block=2)   # spawns slot 2
    assert bank.K == 3
    # Make slot states distinguishable.
    bank.slots[0] = bank.slots[0] + 0.1
    bank.slots[1] = bank.slots[1] - 0.2
    q = torch.randn(1, 2 * 4, 2, 8, dtype=torch.float64)
    summed = var.read(bank, q)
    # Expected: q @ (S_0 + S_1 + S_2) per token.
    S_total = sum(bank.slots)
    expected = torch.einsum("bnhd,bhde->bnhe", q, S_total)
    assert torch.allclose(summed, expected, atol=1e-12)


def test_adaptive_split_spawns_on_rho():
    var = AdaptiveSplit(K_max=3, rho_split=0.0001, cos_merge=2.0)  # rho threshold tiny → always splits; merge disabled
    bank = var.init_state(1, 2, 8, torch.float64, torch.device("cpu"))
    k, v, a, b = _block()
    bank = var.update(bank, k, v, a, b, t_block=0)
    # First update: prev was zero, so rho = ‖ΔS‖/‖0‖ → effectively very large
    # → triggers spawn.  Bank now has 2 slots.
    assert bank.K == 2
    bank = var.update(bank, k, v, a, b, t_block=1)
    # Hits K_max if K_max small enough; otherwise keeps spawning.
    assert bank.K <= 3


def test_adaptive_split_merges_high_cos():
    var = AdaptiveSplit(K_max=4, rho_split=1e9, cos_merge=0.5)  # never split; merge anything > 0.5
    bank = var.init_state(1, 2, 8, torch.float64, torch.device("cpu"))
    # Inject two near-identical slots.
    s = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    bank.slots = [s.clone(), s.clone() + 1e-3 * torch.randn_like(s)]
    bank.meta = [{}, {}]
    k, v, a, b = _block()
    bank = var.update(bank, k, v, a, b, t_block=0)
    # The two near-identical slots should have merged → K reduced.
    assert bank.K < 2 or bank.K == 1
