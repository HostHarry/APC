from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .vocabulary import mask_invalid_logits


@dataclass(frozen=True)
class ContrastStats:
    raw_token: torch.LongTensor
    contrast_token: torch.LongTensor
    raw_confidence: torch.FloatTensor
    base_confidence: torch.FloatTensor
    contrast_confidence: torch.FloatTensor
    apc_mass: torch.FloatTensor
    visual_relevance: torch.FloatTensor
    absolute_visual_gain: torch.FloatTensor
    relative_visual_advantage: torch.FloatTensor
    contrast_top_ids: Optional[torch.LongTensor] = None
    contrast_top_probs: Optional[torch.FloatTensor] = None
    contrast_other_prob: Optional[torch.FloatTensor] = None


def _gather(values: torch.Tensor, token_ids: torch.LongTensor) -> torch.Tensor:
    return values.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def compute_contrast_stats(
    visual_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    valid_text_vocab: torch.BoolTensor,
    *,
    alpha: float,
    beta: float,
    history_top_v_tokens: Optional[int] = None,
) -> ContrastStats:
    """Compute CD-APC candidates and the independent confidence gates."""

    if visual_logits.shape != ablated_logits.shape:
        raise ValueError(
            "Visual and ablated logits must have the same shape, got "
            f"{tuple(visual_logits.shape)} and {tuple(ablated_logits.shape)}"
        )
    if visual_logits.ndim < 2:
        raise ValueError("Expected logits with position and vocabulary axes")
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    visual = mask_invalid_logits(visual_logits.float(), valid_text_vocab)
    ablated = mask_invalid_logits(ablated_logits.float(), valid_text_vocab)
    legal_visual = visual[..., valid_text_vocab]
    legal_ablated = ablated[..., valid_text_vocab]
    if not bool(
        (
            torch.isfinite(legal_visual).all(dim=-1)
            & torch.isfinite(legal_ablated).all(dim=-1)
        ).all()
    ):
        raise FloatingPointError(
            "Visual or ablated branch contains non-finite legal-text logits"
        )

    log_p_visual = torch.log_softmax(visual, dim=-1)
    log_p_ablated = torch.log_softmax(ablated, dim=-1)
    p_visual = log_p_visual.exp()
    p_ablated = log_p_ablated.exp()

    raw_token = visual.argmax(dim=-1)
    raw_confidence = _gather(p_visual, raw_token)

    apc = visual >= (
        visual.amax(dim=-1, keepdim=True) + math.log(float(beta))
    )
    apc &= valid_text_vocab
    if not bool(apc.any(dim=-1).all()):
        raise RuntimeError("APC unexpectedly removed every candidate")
    if not bool(_gather(apc, raw_token).all()):
        raise RuntimeError("Normal-visual top-1 must always remain in APC")

    apc_mass = (p_visual * apc).sum(dim=-1)
    contrast_score = (
        (1.0 + float(alpha)) * log_p_visual
        - float(alpha) * log_p_ablated
    ).masked_fill(~apc, -torch.inf)
    log_q = torch.log_softmax(contrast_score, dim=-1)
    q = log_q.exp()
    contrast_token = contrast_score.argmax(dim=-1)

    base_confidence = _gather(p_visual, contrast_token)
    contrast_confidence = apc_mass * _gather(q, contrast_token)

    log_mixture = torch.logaddexp(
        log_p_visual, log_p_ablated
    ) - math.log(2.0)
    valid = valid_text_vocab.view(
        *((1,) * (visual_logits.ndim - 1)), -1
    )
    visual_log_ratio = torch.where(
        valid, log_p_visual - log_mixture, torch.zeros_like(log_p_visual)
    )
    ablated_log_ratio = torch.where(
        valid, log_p_ablated - log_mixture, torch.zeros_like(log_p_ablated)
    )
    jsd = 0.5 * (
        (p_visual * visual_log_ratio).sum(dim=-1)
        + (p_ablated * ablated_log_ratio).sum(dim=-1)
    )
    visual_relevance = (jsd / math.log(2.0)).clamp(0.0, 1.0)

    gain_all = log_p_visual - log_p_ablated
    absolute_visual_gain = _gather(gain_all, contrast_token)
    raw_visual_gain = _gather(gain_all, raw_token)
    relative_visual_advantage = absolute_visual_gain - raw_visual_gain

    contrast_top_ids = None
    contrast_top_probs = None
    contrast_other_prob = None
    if history_top_v_tokens is not None:
        if history_top_v_tokens < 2:
            raise ValueError("history_top_v_tokens must be at least 2")
        top_v = min(int(history_top_v_tokens), int(q.shape[-1]))
        contrast_top_probs, contrast_top_ids = torch.topk(
            q, k=top_v, dim=-1, largest=True, sorted=True
        )
        contrast_other_prob = (
            1.0 - contrast_top_probs.sum(dim=-1)
        ).clamp(0.0, 1.0)

    scalar_outputs = (
        raw_confidence,
        base_confidence,
        contrast_confidence,
        apc_mass,
        visual_relevance,
        absolute_visual_gain,
        relative_visual_advantage,
    )
    if not all(bool(torch.isfinite(value).all()) for value in scalar_outputs):
        raise FloatingPointError("CD-APC produced NaN or Inf metrics")

    return ContrastStats(
        raw_token=raw_token,
        contrast_token=contrast_token,
        raw_confidence=raw_confidence,
        base_confidence=base_confidence,
        contrast_confidence=contrast_confidence,
        apc_mass=apc_mass,
        visual_relevance=visual_relevance,
        absolute_visual_gain=absolute_visual_gain,
        relative_visual_advantage=relative_visual_advantage,
        contrast_top_ids=contrast_top_ids,
        contrast_top_probs=contrast_top_probs,
        contrast_other_prob=contrast_other_prob,
    )
