"""LoRA adapter tests."""
import torch
import torch.nn as nn

from spec.bgda.lora import LoRALinear, attach_lora_to_qkvo


def test_lora_init_acts_as_identity():
    """At init (B=0), LoRALinear should produce exactly base(x)."""
    base = nn.Linear(8, 16)
    base.weight.data.normal_()
    if base.bias is not None:
        base.bias.data.normal_()
    lora = LoRALinear(base, rank=4, alpha=4.0)
    x = torch.randn(2, 5, 8)
    y_base = base(x)
    y_lora = lora(x)
    assert torch.allclose(y_base, y_lora, atol=1e-6), \
        f"max diff = {(y_base - y_lora).abs().max().item():.3e}"


def test_lora_base_frozen_adapters_trainable():
    base = nn.Linear(8, 16)
    lora = LoRALinear(base, rank=4)
    base_params = list(lora.base.parameters())
    adapter_params = list(lora.lora_A.parameters()) + list(lora.lora_B.parameters())
    for p in base_params:
        assert not p.requires_grad
    for p in adapter_params:
        assert p.requires_grad


def test_lora_grads_flow_through_adapters():
    """At init B=0, so the chain rule gives dL/dA = B^T(dL/dy)x^T = 0 — that's
    correct LoRA math.  After ONE step of B updates, A's grad becomes nonzero.

    We test:
      - base params get NO grad (frozen);
      - lora_B always has nonzero grad (output path is BAx so dL/dB ∝ Ax≠0);
      - lora_A grad is zero at init (B=0) but becomes nonzero after one step.
    """
    base = nn.Linear(8, 16)
    lora = LoRALinear(base, rank=4)
    x = torch.randn(1, 3, 8)
    out = lora(x)
    loss = (out ** 2).sum()
    loss.backward()

    for p in lora.base.parameters():
        assert p.grad is None, "base params should be frozen (no grad)"
    assert lora.lora_B.weight.grad is not None
    assert lora.lora_B.weight.grad.abs().sum() > 0, "lora_B grad must be nonzero at init"
    # lora_A grad is exactly zero at init because B=0 makes the upstream zero.
    # Check after one optimizer step that it becomes nonzero.
    assert lora.lora_A.weight.grad is not None
    init_A_grad_zero = lora.lora_A.weight.grad.abs().sum().item() == 0
    # Simulate 1 step that nudges B off zero
    with torch.no_grad():
        lora.lora_B.weight.add_(0.01 * torch.randn_like(lora.lora_B.weight))
    lora.zero_grad()
    out2 = lora(x)
    (out2 ** 2).sum().backward()
    assert lora.lora_A.weight.grad.abs().sum() > 0, "lora_A grad must be nonzero after B≠0"
    assert init_A_grad_zero, "first-step lora_A grad expected to be zero (B=0)"


def test_attach_lora_to_qkvo_replaces_linears():
    """Verify attach_lora_to_qkvo swaps q,k,v,o for LoRALinear and exposes
    same .weight / .bias for downstream code."""
    class _MockAttn(nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.o = nn.Linear(dim, dim)
            self.unrelated = nn.Linear(dim, dim)
    m = _MockAttn(dim=8)
    n_added = attach_lora_to_qkvo(m, rank=4)
    assert isinstance(m.q, LoRALinear)
    assert isinstance(m.k, LoRALinear)
    assert isinstance(m.v, LoRALinear)
    assert isinstance(m.o, LoRALinear)
    assert isinstance(m.unrelated, nn.Linear) and not isinstance(m.unrelated, LoRALinear)
    # 4 layers × (2 lora linears × 4*8 + 8*4) = 4 * (32+32) = 256
    assert n_added == 4 * (4 * 8 + 8 * 4)
    # weight property still accessible
    assert m.q.weight.shape == (8, 8)
