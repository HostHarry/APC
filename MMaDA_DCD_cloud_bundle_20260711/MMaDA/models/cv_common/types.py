"""Shared dataclasses for CV-DCD v4.

These are consumed by both directions:
- ``cfg_rerank`` uses ``StepRecord`` to record per-step commit info during
  generation, then reuses it to compute unconditional log-likelihood.
- ``defer_only`` does not use these directly today but they live here so any
  future extension can share the schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class StepRecord:
    """Per-step commit info recorded during DCD decoding.

    Populated by ``cfg_rerank.generator`` during candidate generation; the
    ``logp_uncond`` slot is later filled by ``cfg_rerank.uncond_ll`` when
    replaying the same commit trajectory under an image-dropped input.
    """

    step_idx: int
    committed_positions: List[int]
    committed_tokens: List[int]
    logp_cond: torch.Tensor  # shape [n_committed], float64
    logp_uncond: Optional[torch.Tensor] = None  # filled by uncond_ll.py


@dataclass
class Candidate:
    """One decoded sequence + its scoring metadata.

    ``seq`` is the full token tensor produced by DCD (shape ``[1, L]``); we
    keep it 2D to match the caller's convention. Cast to CPU before storing
    if you plan to accumulate many candidates.
    """

    seq: torch.Tensor
    step_records: List[StepRecord] = field(default_factory=list)
    seed: int = 0
    total_logp_cond: float = 0.0
    total_logp_uncond: float = 0.0
    cfg_score: float = 0.0
