"""Stage D state-memory variants (plan v1 §3.4).

Three memory-expansion options to ablate when extending BGDA from 5 s to 60 s:

  - D-1 SingleS         : keep just one `S ∈ R^{H×d×d}` per layer, rely on
                          α-decay to forget old contributions naturally.
                          Simplest; baseline.

  - D-2 FixedSlotBank   : K_max=4 slots per layer.  Spawn a new slot every
                          15 s (configurable); when full, evict oldest with a
                          merge into the next-oldest.  Bounded memory.

  - D-3 AdaptiveSplit   : v1's GDSA split/merge mechanism.  Spawn when
                          `ρ_t = ‖ΔS‖ / ‖αS_{t-1}‖` exceeds a threshold;
                          merge when two slots have cosine sim > τ_merge.
                          Caps K_max=4.

All three implement the same interface so the training loop can switch
between them via a config flag.

Each variant exposes:
  - `init_state(B, H, d, dtype, device) -> StateContainer`
  - `read(state_container, q_block) -> out_block`         (used by within-block read)
  - `update(state_container, k_block, v_block, alpha, beta, beta_normalize)`
       — runs frame-sequential GDN updates and may spawn/evict slots.
  - `to_legacy_S(state_container) -> Tensor`              (for compatibility)

We keep the recurrence math identical to `gdn_state_update` from
`reference.py`; the variants only change how MULTIPLE states are managed.

Note: these are not yet integrated with `BGDABlockAttention` — that will
require activating `use_recurrence=True` in a different mode where the read
sums over multiple slots rather than reading from a single S.  Wiring lives
in a follow-up PR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .reference import gdn_state_update


# ----------------------------------------------------------------------------
# Common state container
# ----------------------------------------------------------------------------


@dataclass
class StateBank:
    """A bank of `K` slots, each holding one `S` of shape `[B, H, d, d]`.

    `slots[k]` is the S of slot k; `meta[k]` carries arbitrary per-slot
    metadata (creation_time, last_update_time, age, etc.).
    """

    slots: List[torch.Tensor] = field(default_factory=list)
    meta: List[dict] = field(default_factory=list)

    @property
    def K(self) -> int:
        return len(self.slots)


def _zero_S(B: int, H: int, d: int, dtype, device) -> torch.Tensor:
    return torch.zeros(B, H, d, d, dtype=dtype, device=device)


# ----------------------------------------------------------------------------
# D-1  SingleS
# ----------------------------------------------------------------------------


class SingleS:
    """Plan-v1 §3.4 variant D-1: just one slot, α-decay does the work."""

    def init_state(self, B: int, H: int, d: int, dtype, device) -> StateBank:
        return StateBank(
            slots=[_zero_S(B, H, d, dtype, device)],
            meta=[{"created_at": 0}],
        )

    def update(self, bank: StateBank, k_block: torch.Tensor, v_block: torch.Tensor,
               alpha: torch.Tensor, beta: torch.Tensor, *, t_block: int = 0,
               beta_normalize: Optional[str] = None) -> StateBank:
        """`k_block, v_block: [B, W, HW, H, d]; alpha: [B, W, H]; beta: [B, W, HW, H]`."""
        S = bank.slots[0]
        W = k_block.shape[1]
        for f in range(W):
            S = gdn_state_update(
                S, k_block[:, f], v_block[:, f], alpha[:, f], beta[:, f],
                beta_normalize=beta_normalize,
            )
        bank.slots[0] = S
        return bank

    def read(self, bank: StateBank, q_block: torch.Tensor) -> torch.Tensor:
        """`q_block: [B, n_b, H, d]` → `[B, n_b, H, d]`.  Reads the single slot."""
        S = bank.slots[0]
        return torch.einsum("bnhd,bhde->bnhe", q_block, S)


# ----------------------------------------------------------------------------
# D-2  FixedSlotBank
# ----------------------------------------------------------------------------


class FixedSlotBank:
    """Variant D-2: K_max slots, spawn one every `spawn_every` blocks, evict
    oldest with merge when full.

    Read sums across all active slots (each slot contributes its own
    `q · S_k` to the output).  This means the read scales linearly with K.
    """

    def __init__(self, K_max: int = 4, spawn_every_blocks: int = 15):
        self.K_max = K_max
        self.spawn_every_blocks = spawn_every_blocks

    def init_state(self, B: int, H: int, d: int, dtype, device) -> StateBank:
        return StateBank(
            slots=[_zero_S(B, H, d, dtype, device)],
            meta=[{"created_at": 0}],
        )

    def _maybe_spawn(self, bank: StateBank, t_block: int, B, H, d, dtype, device):
        if t_block > 0 and t_block % self.spawn_every_blocks == 0:
            if bank.K >= self.K_max:
                # Evict oldest by merging into next-oldest (sum-merge).
                oldest_S = bank.slots.pop(0)
                bank.meta.pop(0)
                bank.slots[0] = bank.slots[0] + oldest_S
            bank.slots.append(_zero_S(B, H, d, dtype, device))
            bank.meta.append({"created_at": t_block})

    def update(self, bank: StateBank, k_block: torch.Tensor, v_block: torch.Tensor,
               alpha: torch.Tensor, beta: torch.Tensor, *, t_block: int = 0,
               beta_normalize: Optional[str] = None) -> StateBank:
        B, W, HW, H, d = k_block.shape
        self._maybe_spawn(bank, t_block, B, H, d, k_block.dtype, k_block.device)

        # Update the *most recent* slot — older slots are frozen for the
        # remainder of training.  This matches plan §3.4 D-2's intended
        # behavior: each slot represents a temporal context window.
        S = bank.slots[-1]
        for f in range(W):
            S = gdn_state_update(
                S, k_block[:, f], v_block[:, f], alpha[:, f], beta[:, f],
                beta_normalize=beta_normalize,
            )
        bank.slots[-1] = S
        return bank

    def read(self, bank: StateBank, q_block: torch.Tensor) -> torch.Tensor:
        out = None
        for S in bank.slots:
            o = torch.einsum("bnhd,bhde->bnhe", q_block, S)
            out = o if out is None else (out + o)
        return out


# ----------------------------------------------------------------------------
# D-3  AdaptiveSplit  (the v1 GDSA mechanism, demoted to ablation knob)
# ----------------------------------------------------------------------------


class AdaptiveSplit:
    """Variant D-3: split when `ρ_t = ‖ΔS‖ / ‖αS_{t-1}‖` exceeds threshold;
    merge when two slots have cosine sim > τ_merge. K_max cap.

    Implements the v1 GDSA split/merge mechanism, demoted to a plan-§3.4
    ablation knob (it was Mechanism #2 + #3 in the prior GDSA proposal).
    """

    def __init__(
        self,
        K_max: int = 4,
        rho_split: float = 1.0,
        cos_merge: float = 0.95,
    ):
        self.K_max = K_max
        self.rho_split = rho_split
        self.cos_merge = cos_merge

    def init_state(self, B: int, H: int, d: int, dtype, device) -> StateBank:
        return StateBank(
            slots=[_zero_S(B, H, d, dtype, device)],
            meta=[{"created_at": 0}],
        )

    @staticmethod
    def _slot_cossim(s1: torch.Tensor, s2: torch.Tensor) -> float:
        """Cosine sim between two `[B, H, d, d]` slots (flattened)."""
        a = s1.float().reshape(-1)
        b = s2.float().reshape(-1)
        denom = (a.norm() * b.norm()).clamp(min=1e-12)
        return float((a @ b / denom).item())

    def update(self, bank: StateBank, k_block: torch.Tensor, v_block: torch.Tensor,
               alpha: torch.Tensor, beta: torch.Tensor, *, t_block: int = 0,
               beta_normalize: Optional[str] = None) -> StateBank:
        B, W, HW, H, d = k_block.shape
        # Update most-recent slot like D-2.
        S_old = bank.slots[-1].clone()
        S = bank.slots[-1]
        for f in range(W):
            S = gdn_state_update(
                S, k_block[:, f], v_block[:, f], alpha[:, f], beta[:, f],
                beta_normalize=beta_normalize,
            )
        bank.slots[-1] = S

        # ρ-split decision: only if α applied (α decays the carry; ΔS is
        # the post-update vs the α-decayed pre-update).  We approximate ρ
        # using ‖S - S_old‖ / ‖S_old‖.
        delta_norm = (S - S_old).float().norm().item()
        prev_norm = max(S_old.float().norm().item(), 1e-9)
        rho = delta_norm / prev_norm
        if rho > self.rho_split and bank.K < self.K_max:
            # Spawn a fresh slot for the next block (current one is "saturated")
            bank.slots.append(_zero_S(B, H, d, k_block.dtype, k_block.device))
            bank.meta.append({"created_at": t_block, "spawn_rho": rho})

        # cosine-merge: collapse pairs that became redundant.
        if bank.K > 1:
            for i in range(bank.K - 1):
                for j in range(i + 1, bank.K):
                    if self._slot_cossim(bank.slots[i], bank.slots[j]) > self.cos_merge:
                        bank.slots[i] = bank.slots[i] + bank.slots[j]
                        del bank.slots[j]
                        del bank.meta[j]
                        return bank
        return bank

    def read(self, bank: StateBank, q_block: torch.Tensor) -> torch.Tensor:
        out = None
        for S in bank.slots:
            o = torch.einsum("bnhd,bhde->bnhe", q_block, S)
            out = o if out is None else (out + o)
        return out
