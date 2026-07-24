"""Stability-Weighted Decoding (SWD) for diffusion LM remasking.

Wu & Huang, arXiv:2604.17068. Multiplicative stability modulator:
``S <- S * exp(-lambda * KL(p_prev || p_curr))`` (Algorithm 1 KL direction).

Performance note: only the response span is cached/scored. Image/prompt
positions never compete in low-confidence remasking TopK, so full-sequence
KL over |V|~134k was pure overhead (~5 min/sample). Restricting to the
response window (typically <=128 tokens) restores near-baseline speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class SwdState:
    """Caches previous-step predictive distributions on the response span.

    ``prev_probs`` has shape ``[B, response_len, V]`` (float32) or None.
    """

    prev_probs: Optional[torch.Tensor] = None
    response_start: Optional[int] = None
    response_end: Optional[int] = None


def kl_prev_vs_curr(
    p_prev: torch.Tensor,
    p_curr: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Per-position KL(p_prev || p_curr); returns shape ``p_curr.shape[:-1]``."""

    if p_prev.shape != p_curr.shape:
        raise ValueError(
            "SWD KL requires matching distribution shapes, got "
            f"{tuple(p_prev.shape)} vs {tuple(p_curr.shape)}"
        )
    p_prev = p_prev.clamp_min(eps)
    p_curr = p_curr.clamp_min(eps)
    p_prev = p_prev / p_prev.sum(dim=-1, keepdim=True)
    p_curr = p_curr / p_curr.sum(dim=-1, keepdim=True)
    return (p_prev * (p_prev.log() - p_curr.log())).sum(dim=-1)


def stability_weight(
    p_curr: torch.Tensor,
    p_prev: torch.Tensor,
    lambda_: float,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """``exp(-lambda * D_KL(p_prev || p_curr))`` per sequence position."""

    divergence = kl_prev_vs_curr(p_prev, p_curr, eps=eps)
    return torch.exp(-float(lambda_) * divergence)


def uniform_prior_like(probs: torch.Tensor) -> torch.Tensor:
    vocab = int(probs.shape[-1])
    return torch.full_like(probs, 1.0 / float(vocab))


def response_span_token_probs(
    logits: torch.Tensor,
    x0: torch.Tensor,
    *,
    response_start: int,
    response_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax + gather only on ``[response_start, response_end)``.

    Returns ``(x0_p, p_resp)`` where ``x0_p`` is ``[B, L]`` (``-inf`` outside
    the response) and ``p_resp`` is ``[B, resp_len, V]`` float32. Avoids the
    original remasking path's full-sequence ``float64`` softmax which materializes
    ~``L * V * 8`` bytes per step (multi-GB for MMaDA image+text contexts).
    """

    rs, re = int(response_start), int(response_end)
    if re <= rs:
        raise ValueError(f"Invalid response span [{rs}, {re})")
    if x0.shape != logits.shape[:-1]:
        raise ValueError(
            "x0 must match logits leading dims, got "
            f"{tuple(x0.shape)} vs {tuple(logits.shape[:-1])}"
        )

    p_resp = torch.softmax(logits[:, rs:re, :].float(), dim=-1)
    gathered = torch.gather(
        p_resp, dim=-1, index=x0[:, rs:re].unsqueeze(-1)
    ).squeeze(-1)
    x0_p = torch.full(
        x0.shape, float("-inf"), device=logits.device, dtype=torch.float32
    )
    x0_p[:, rs:re] = gathered
    return x0_p, p_resp


def apply_swd_to_confidence(
    confidence: torch.Tensor,
    logits: torch.Tensor,
    state: SwdState,
    *,
    lambda_: float,
    response_start: int,
    response_end: int,
    mask_index: Optional[torch.Tensor] = None,
    p_curr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multiply response-span confidence by SWD weights derived from logits.

    Softmax / KL / history are restricted to
    ``[response_start, response_end)``. Prompt/image positions are untouched.
    Pass ``p_curr`` (response-span softmax) to avoid a second softmax when the
    caller already computed it for remasking confidence.
    """

    if response_end <= response_start:
        raise ValueError(
            f"Invalid response span [{response_start}, {response_end})"
        )
    if confidence.shape != logits.shape[:-1]:
        raise ValueError(
            "confidence must match logits leading dims, got "
            f"{tuple(confidence.shape)} vs {tuple(logits.shape[:-1])}"
        )

    rs, re = int(response_start), int(response_end)
    if state.response_start is None:
        state.response_start = rs
        state.response_end = re
    elif (state.response_start, state.response_end) != (rs, re):
        raise RuntimeError(
            "SWD response span changed mid-decode: "
            f"{(state.response_start, state.response_end)} -> {(rs, re)}"
        )

    if p_curr is None:
        p_curr = torch.softmax(logits[:, rs:re, :].float(), dim=-1)
    else:
        expected = (logits.shape[0], re - rs, logits.shape[-1])
        if tuple(p_curr.shape) != expected:
            raise ValueError(
                "p_curr must be response-span probs "
                f"{expected}, got {tuple(p_curr.shape)}"
            )
        p_curr = p_curr.float()

    if state.prev_probs is None:
        p_prev = uniform_prior_like(p_curr)
    else:
        if state.prev_probs.shape != p_curr.shape:
            raise RuntimeError(
                "SWD history shape mismatch: "
                f"{tuple(state.prev_probs.shape)} vs {tuple(p_curr.shape)}"
            )
        p_prev = state.prev_probs

    weight = stability_weight(p_curr, p_prev, lambda_)  # [B, resp_len]
    resp_conf = confidence[:, rs:re]
    finite = torch.isfinite(resp_conf)
    modulated_resp = torch.where(
        finite, resp_conf * weight.to(resp_conf.dtype), resp_conf
    )
    if mask_index is not None:
        resp_mask = mask_index[:, rs:re]
        modulated_resp = torch.where(resp_mask, modulated_resp, resp_conf)

    out = confidence.clone()
    out[:, rs:re] = modulated_resp
    state.prev_probs = p_curr.detach()
    return out
