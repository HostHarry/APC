"""Log-probability utilities shared across CV-DCD variants.

- ``logp_of_tokens``: gather the log-softmax of a target token at each
  position. Equivalent to v3.2's ``_logp_of_x0``.
- ``stepwise_logp_from_records`` / ``aggregate_sequence_ll``: helpers used
  by ``cfg_rerank`` to accumulate per-step commit log-probabilities into a
  single scalar for CFG scoring.

All computations upcast to float64 for numerical stability -- the aggregate
sums otherwise underflow to 0 for long sequences on bfloat16.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from .types import StepRecord


def logp_of_tokens(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Return ``log p(tokens[b, l] | logits[b, l])`` for each position.

    Shapes:
        logits : (B, L, V)
        tokens : (B, L) long
        output : (B, L) float64

    NaN-safe: caller controls ``tokens`` (must be valid indices).

    Memory-efficient: uses ``logsumexp`` on the original dtype instead of
    materializing a ``(B, L, V)`` float64 log-softmax tensor. For MMaDA
    (V=134656, L=2048) this saves ~2 GB of peak GPU memory per call.
    """
    logit_at_token = torch.gather(
        logits, dim=-1, index=tokens.unsqueeze(-1)
    ).squeeze(-1).to(torch.float64)
    lse = torch.logsumexp(logits, dim=-1).to(torch.float64)
    return logit_at_token - lse


def stepwise_logp_from_records(step_records: List["StepRecord"]) -> torch.Tensor:
    """Concatenate ``step.logp_cond`` across all steps into ``[n_committed]``.

    Commit order is preserved. Returned tensor lives on the first record's
    device. Empty input returns an empty tensor.
    """
    if not step_records:
        return torch.empty(0, dtype=torch.float64)
    chunks = [rec.logp_cond.to(torch.float64).reshape(-1) for rec in step_records]
    return torch.cat(chunks, dim=0)


def aggregate_sequence_ll(
    step_records: List["StepRecord"],
    reduction: str = "sum",
) -> float:
    """Reduce stepwise log-p to a scalar for CFG rerank scoring.

    Reductions
    ----------
    ``sum``                : plain sum (default; length-biased).
    ``mean``               : arithmetic mean over committed tokens.
    ``length_normalized``  : sum / n^0.7 (partial length correction).
    """
    logps = stepwise_logp_from_records(step_records)
    if logps.numel() == 0:
        return 0.0
    total = float(logps.sum().item())
    n = int(logps.numel())
    if reduction == "sum":
        return total
    if reduction == "mean":
        return total / n
    if reduction == "length_normalized":
        return total / (n ** 0.7)
    raise ValueError(
        f"unknown reduction={reduction!r}; expected one of "
        f"'sum' | 'mean' | 'length_normalized'"
    )
