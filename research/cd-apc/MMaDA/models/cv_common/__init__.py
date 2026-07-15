"""Shared utilities for CV-DCD v4 (cfg_rerank + defer_only).

Provides pure functions extracted from mmada_decode.py so both v4 directions
can reuse the same image-ablation, paired-forward, and log-probability logic
without pulling on the full DCD stack.

The originals in mmada_decode.py remain as thin wrappers around these, so
v3.2 code paths are unchanged.
"""
from . import image_drop, log_prob, paired_forward, types

__all__ = ["image_drop", "log_prob", "paired_forward", "types"]
