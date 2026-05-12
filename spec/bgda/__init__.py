"""BGDA — Block Gated Delta Attention.

Reference + (later) optimized implementations for the BGDA plan v1
(`bgda_plan_v1.txt`). The submodules are:

- `reference.py`    — naive PyTorch reference, eager loops, ground truth.
- `bgda_attention.py` — nn.Module wrapper compatible with Wan2.1's `WanSelfAttention`
                        signature so it can be dropped into `WanModel`.
- `tests/`          — unit tests (math correctness, init recovery, gradients,
                       block-causality, S-norm boundedness).
- `bench/`          — Phase 0 microbench scripts.

See `README.md` for the math and design choices.
"""
from .reference import (
    bgda_block_attention_reference,
    gdn_state_update,
)
