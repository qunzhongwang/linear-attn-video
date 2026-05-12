"""GDSA — Gated Delta Split Attention. v1: Mechanism #1 (gated delta) only."""

from .reference import gdsa_reference, vanilla_reference
from .chunked import gdsa_chunked, vanilla_chunked

__all__ = ["gdsa_reference", "vanilla_reference", "gdsa_chunked", "vanilla_chunked"]
