from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class SparseHistory:
    """Top-V EMA distribution plus a single omitted-mass bucket."""

    token_ids: torch.LongTensor
    probs: torch.FloatTensor
    other_prob: torch.FloatTensor
    last_observed_context_version: int
    last_top1_token: int
    consecutive_top1_matches: int


@dataclass(frozen=True)
class HistoryObservation:
    stability: torch.FloatTensor
    next_history: SparseHistory
    updated: bool
    consecutive_top1_matches: int


def _normalized_parts(
    token_ids: torch.LongTensor,
    probs: torch.Tensor,
    other_prob: torch.Tensor,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    token_ids = token_ids.reshape(-1).long()
    probs = probs.reshape(-1).float().clamp_min(0.0)
    other_prob = other_prob.reshape(()).float().clamp_min(0.0)
    if token_ids.numel() != probs.numel():
        raise ValueError("Sparse token_ids and probs must have equal lengths")
    if token_ids.numel() == 0:
        raise ValueError("A sparse distribution must retain at least one token")
    if torch.unique(token_ids).numel() != token_ids.numel():
        raise ValueError("Sparse token_ids must be unique")
    total = probs.sum() + other_prob
    if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
        raise FloatingPointError("Sparse distribution has invalid total mass")
    return token_ids, probs / total, other_prob / total


def sparse_distribution_from_dense(
    distribution: torch.Tensor, top_v_tokens: int
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compress a normalized dense distribution into top-V plus OTHER."""

    if distribution.ndim != 1:
        raise ValueError("Dense distribution must be one-dimensional")
    if top_v_tokens < 2:
        raise ValueError("top_v_tokens must be at least 2")
    dense = distribution.float().clamp_min(0.0)
    total = dense.sum()
    if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
        raise FloatingPointError("Dense distribution has invalid total mass")
    dense = dense / total
    top_v = min(int(top_v_tokens), int(dense.numel()))
    probs, token_ids = torch.topk(
        dense, k=top_v, largest=True, sorted=True
    )
    other_prob = (1.0 - probs.sum()).clamp(0.0, 1.0)
    return _normalized_parts(token_ids, probs, other_prob)


def _aligned_sparse_vectors(
    first_ids: torch.LongTensor,
    first_probs: torch.Tensor,
    first_other: torch.Tensor,
    second_ids: torch.LongTensor,
    second_probs: torch.Tensor,
    second_other: torch.Tensor,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    first_ids, first_probs, first_other = _normalized_parts(
        first_ids, first_probs, first_other
    )
    second_ids, second_probs, second_other = _normalized_parts(
        second_ids, second_probs, second_other
    )
    if first_ids.device != second_ids.device:
        raise ValueError("Sparse distributions must be on the same device")

    support = torch.unique(
        torch.cat((first_ids, second_ids)), sorted=True
    )
    first = torch.zeros(
        support.numel() + 1, dtype=torch.float32, device=support.device
    )
    second = torch.zeros_like(first)
    first[torch.searchsorted(support, first_ids)] = first_probs
    second[torch.searchsorted(support, second_ids)] = second_probs
    first[-1] = first_other
    second[-1] = second_other
    return support, first, second


def sparse_jsd(
    first_ids: torch.LongTensor,
    first_probs: torch.Tensor,
    first_other: torch.Tensor,
    second_ids: torch.LongTensor,
    second_probs: torch.Tensor,
    second_other: torch.Tensor,
) -> torch.FloatTensor:
    """JSD on the union of retained tokens and one shared OTHER bucket."""

    _, first, second = _aligned_sparse_vectors(
        first_ids,
        first_probs,
        first_other,
        second_ids,
        second_probs,
        second_other,
    )
    mixture = 0.5 * (first + second)

    def contribution(values: torch.Tensor) -> torch.Tensor:
        return torch.where(
            values > 0.0,
            values * (torch.log(values) - torch.log(mixture)),
            torch.zeros_like(values),
        ).sum()

    divergence = 0.5 * (contribution(first) + contribution(second))
    if not bool(torch.isfinite(divergence)):
        raise FloatingPointError("Sparse history JSD produced NaN or Inf")
    return divergence.clamp(0.0, math.log(2.0))


def _ema_history(
    history: SparseHistory,
    current_ids: torch.LongTensor,
    current_probs: torch.Tensor,
    current_other: torch.Tensor,
    *,
    context_version: int,
    ema_decay: float,
    top_v_tokens: int,
) -> SparseHistory:
    support, current, previous = _aligned_sparse_vectors(
        current_ids,
        current_probs,
        current_other,
        history.token_ids,
        history.probs,
        history.other_prob,
    )
    merged = float(ema_decay) * previous + (
        1.0 - float(ema_decay)
    ) * current
    explicit = merged[:-1]
    order = torch.argsort(explicit, descending=True, stable=True)
    keep_count = min(int(top_v_tokens), int(order.numel()))
    keep = order[:keep_count]
    dropped = order[keep_count:]
    kept_ids = support[keep]
    kept_probs = explicit[keep]
    other_prob = merged[-1] + explicit[dropped].sum()
    kept_ids, kept_probs, other_prob = _normalized_parts(
        kept_ids, kept_probs, other_prob
    )
    return SparseHistory(
        token_ids=kept_ids,
        probs=kept_probs,
        other_prob=other_prob,
        last_observed_context_version=int(context_version),
        last_top1_token=int(current_ids[0].item()),
        consecutive_top1_matches=(
            history.consecutive_top1_matches + 1
            if int(current_ids[0].item()) == history.last_top1_token
            else 0
        ),
    )


def observe_sparse_history(
    history: Optional[SparseHistory],
    current_ids: torch.LongTensor,
    current_probs: torch.Tensor,
    current_other: torch.Tensor,
    *,
    context_version: int,
    ema_decay: float,
    top_v_tokens: int,
) -> HistoryObservation:
    """Score against pre-update history and plan at most one versioned update."""

    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if top_v_tokens < 2:
        raise ValueError("top_v_tokens must be at least 2")
    current_ids, current_probs, current_other = _normalized_parts(
        current_ids, current_probs, current_other
    )

    if history is None:
        next_history = SparseHistory(
            token_ids=current_ids[:top_v_tokens].clone(),
            probs=current_probs[:top_v_tokens].clone(),
            other_prob=(
                current_other + current_probs[top_v_tokens:].sum()
            ).clone(),
            last_observed_context_version=int(context_version),
            last_top1_token=int(current_ids[0].item()),
            consecutive_top1_matches=0,
        )
        return HistoryObservation(
            stability=torch.ones(
                (), dtype=torch.float32, device=current_probs.device
            ),
            next_history=next_history,
            updated=True,
            consecutive_top1_matches=0,
        )

    if history.last_observed_context_version > context_version:
        raise ValueError(
            "History context version is newer than the current snapshot"
        )
    divergence = sparse_jsd(
        current_ids,
        current_probs,
        current_other,
        history.token_ids,
        history.probs,
        history.other_prob,
    )
    stability = (1.0 - divergence / math.log(2.0)).clamp(0.0, 1.0)
    if history.last_observed_context_version == context_version:
        return HistoryObservation(
            stability=stability,
            next_history=history,
            updated=False,
            consecutive_top1_matches=history.consecutive_top1_matches,
        )

    next_history = _ema_history(
        history,
        current_ids,
        current_probs,
        current_other,
        context_version=context_version,
        ema_decay=ema_decay,
        top_v_tokens=top_v_tokens,
    )
    return HistoryObservation(
        stability=stability,
        next_history=next_history,
        updated=True,
        consecutive_top1_matches=next_history.consecutive_top1_matches,
    )


def history_adjusted_reliability(
    contrast_confidence: torch.Tensor,
    visual_relevance: torch.Tensor,
    history_stability: torch.Tensor,
    *,
    penalty_scale: float = 1.0,
) -> torch.FloatTensor:
    """R = C_contrast * [1 - clamp(gamma * rho * (1 - T), 0, 1)]."""

    if not (
        contrast_confidence.shape
        == visual_relevance.shape
        == history_stability.shape
    ):
        raise ValueError("Reliability inputs must have identical shapes")
    if not math.isfinite(float(penalty_scale)) or penalty_scale < 0.0:
        raise ValueError("penalty_scale must be finite and non-negative")
    penalty = (
        float(penalty_scale)
        * visual_relevance.float()
        * (1.0 - history_stability.float())
    ).clamp(0.0, 1.0)
    reliability = contrast_confidence.float() * (
        1.0 - penalty
    )
    if not bool(torch.isfinite(reliability).all()):
        raise FloatingPointError("History reliability produced NaN or Inf")
    return reliability.clamp(0.0, 1.0)
