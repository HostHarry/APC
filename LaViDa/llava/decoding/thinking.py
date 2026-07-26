"""Training-free Thinking Diffusion and stability-weighted decoding helpers.

The score modulators in this module operate on the ordinary low-confidence
remasking path:

* PSP penalizes late response positions early in the reverse process.
* VRG amplifies the difference between visual and visual-access-ablated logits.
* SWD downweights positions whose predictive distribution changes between
  adjacent denoising steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch


@dataclass
class ThinkingDecodeConfig:
    """Optional score modulators for LaViDa's original decoder."""

    psp_enabled: bool = False
    psp_gamma: float = 0.5
    vrg_enabled: bool = False
    vrg_scale: float = 0.5
    swd_enabled: bool = False
    swd_lambda: float = 5.0
    force_math_sdpa: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.psp_enabled or self.vrg_enabled or self.swd_enabled)

    def validate(self) -> None:
        if not 0.0 <= float(self.psp_gamma) <= 1.0:
            raise ValueError(
                f"psp_gamma must be in [0, 1], got {self.psp_gamma}"
            )
        if float(self.vrg_scale) < 0.0:
            raise ValueError(
                f"vrg_scale must be non-negative, got {self.vrg_scale}"
            )
        if float(self.swd_lambda) < 0.0:
            raise ValueError(
                f"swd_lambda must be non-negative, got {self.swd_lambda}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def thinking_config_from_dict(values: Mapping[str, Any]) -> ThinkingDecodeConfig:
    known = {field for field in ThinkingDecodeConfig.__dataclass_fields__}
    config = ThinkingDecodeConfig(
        **{key: value for key, value in values.items() if key in known}
    )
    config.validate()
    return config


def resolve_thinking_config(
    config: Optional[Union[ThinkingDecodeConfig, Mapping[str, Any]]],
) -> ThinkingDecodeConfig:
    if config is None:
        return ThinkingDecodeConfig()
    if isinstance(config, ThinkingDecodeConfig):
        config.validate()
        return config
    if isinstance(config, Mapping):
        return thinking_config_from_dict(config)
    raise TypeError(
        "thinking config must be ThinkingDecodeConfig, a mapping, or None"
    )


def apply_vrg(
    logits_visual: torch.Tensor,
    logits_ablated: torch.Tensor,
    scale: float = 0.5,
) -> torch.Tensor:
    """Apply Visual Reasoning Guidance.

    ``L_vrg = L_ablated + (scale + 1) * (L_visual - L_ablated)``.
    A zero scale exactly recovers the visual logits.
    """

    if logits_visual.shape != logits_ablated.shape:
        raise ValueError(
            "VRG requires matching logits shapes, got "
            f"{tuple(logits_visual.shape)} and {tuple(logits_ablated.shape)}"
        )
    return logits_ablated + (float(scale) + 1.0) * (
        logits_visual - logits_ablated
    )


def visual_guided_logits(
    model,
    input_embeddings: torch.Tensor,
    visual_mask: torch.Tensor,
    *,
    scale: float,
    attention_mask: Optional[torch.Tensor] = None,
    force_math_sdpa: bool = True,
    past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> torch.Tensor:
    """Run paired visual/ablated LaViDa forwards and apply VRG."""

    from .lavida_adapter import (
        _math_sdpa_context,
        _normalize_attention_mask,
        build_paired_attention_bias_from_mask,
    )

    if input_embeddings.ndim != 3:
        raise ValueError(
            "input_embeddings must have shape [B, S, H], got "
            f"{tuple(input_embeddings.shape)}"
        )
    if input_embeddings.shape[0] != 1:
        raise ValueError("LaViDa VRG currently supports batch_size=1")
    if visual_mask.ndim != 1:
        raise ValueError("visual_mask must be one-dimensional")

    query_len = int(input_embeddings.shape[1])
    past_len = (
        0
        if past_key_values is None
        else int(past_key_values[0][0].shape[-2])
    )
    seq_len = past_len + query_len
    if visual_mask.numel() > seq_len:
        raise ValueError(
            f"visual_mask length {visual_mask.numel()} exceeds {seq_len}"
        )
    full_visual_mask = visual_mask.to(
        device=input_embeddings.device, dtype=torch.bool
    )
    if full_visual_mask.numel() < seq_len:
        full_visual_mask = torch.cat(
            [
                full_visual_mask,
                torch.zeros(
                    seq_len - full_visual_mask.numel(),
                    device=input_embeddings.device,
                    dtype=torch.bool,
                ),
            ]
        )
    if not bool(full_visual_mask.any()):
        raise ValueError("VRG requires at least one visual prompt position")

    branch_bias = build_paired_attention_bias_from_mask(full_visual_mask)
    if past_key_values is not None:
        # Match LaViDa's cached prefix path, which becomes causal over the
        # response tokens once past keys are supplied.
        causal = torch.zeros_like(branch_bias)
        upper_triangle = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=input_embeddings.device,
            ),
            diagonal=1,
        )
        causal.masked_fill_(
            upper_triangle[None, None],
            torch.finfo(causal.dtype).min,
        )
        branch_bias = torch.minimum(branch_bias, causal)
        normalized_mask = None
    else:
        normalized_mask = _normalize_attention_mask(
            attention_mask,
            seq_len=seq_len,
            device=input_embeddings.device,
        )
    pair_mask = (
        None if normalized_mask is None else normalized_mask.repeat(2, 1)
    )
    pair_embeddings = input_embeddings.repeat(2, 1, 1)
    with torch.inference_mode(), _math_sdpa_context(force_math_sdpa):
        output = model(
            None,
            input_embeddings=pair_embeddings,
            attention_mask=pair_mask,
            attention_bias=branch_bias,
            past_key_values=past_key_values,
            use_cache=False,
        )
    if output.logits.shape[:2] != (2, query_len):
        raise RuntimeError(
            "Unexpected paired VRG logits shape "
            f"{tuple(output.logits.shape)} for query length {query_len}"
        )
    logits_visual, logits_ablated = output.logits.chunk(2, dim=0)
    return apply_vrg(logits_visual, logits_ablated, scale)


def visual_guided_prefill(
    model,
    prompt_embeddings: torch.Tensor,
    visual_mask: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    force_math_sdpa: bool = True,
):
    """Build visual and visual-ablated prompt KV caches in one forward."""

    from .lavida_adapter import (
        _math_sdpa_context,
        _normalize_attention_mask,
        build_paired_attention_bias_from_mask,
    )

    if prompt_embeddings.ndim != 3 or prompt_embeddings.shape[0] != 1:
        raise ValueError("VRG prefill expects prompt embeddings with batch_size=1")
    prompt_len = int(prompt_embeddings.shape[1])
    if visual_mask.ndim != 1 or int(visual_mask.numel()) != prompt_len:
        raise ValueError(
            "visual_mask must be one-dimensional and match the prompt length"
        )
    visual_mask = visual_mask.to(
        device=prompt_embeddings.device, dtype=torch.bool
    )
    if not bool(visual_mask.any()):
        raise ValueError("VRG requires at least one visual prompt position")

    branch_bias = build_paired_attention_bias_from_mask(visual_mask)
    normalized_mask = _normalize_attention_mask(
        attention_mask,
        seq_len=prompt_len,
        device=prompt_embeddings.device,
    )
    pair_mask = (
        None if normalized_mask is None else normalized_mask.repeat(2, 1)
    )
    with torch.inference_mode(), _math_sdpa_context(force_math_sdpa):
        output = model(
            None,
            input_embeddings=prompt_embeddings.repeat(2, 1, 1),
            attention_mask=pair_mask,
            attention_bias=branch_bias,
            use_cache=True,
        )
    cache = getattr(output, "attn_key_values", None)
    if cache is None:
        raise RuntimeError("LaViDa VRG prefill did not return a KV cache")
    return cache


def apply_psp(
    confidence: torch.Tensor,
    *,
    step_index: int,
    num_steps: int,
    response_start: int,
    response_end: int,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Apply Position & Step Penalty to full-sequence confidence scores."""

    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if not 0 <= step_index < num_steps:
        raise ValueError(
            f"step_index must be in [0, {num_steps}), got {step_index}"
        )
    if response_end <= response_start:
        raise ValueError(
            "response_end must exceed response_start, got "
            f"[{response_start}, {response_end})"
        )

    tau = float(step_index + 1) / float(num_steps)
    span = max(1, int(response_end) - int(response_start) - 1)
    positions = torch.arange(
        confidence.shape[-1],
        device=confidence.device,
        dtype=torch.float32,
    )
    relative_position = (
        (positions - float(response_start)) / float(span)
    ).clamp(0.0, 1.0)
    in_response = (positions >= float(response_start)) & (
        positions < float(response_end)
    )
    relative_position = torch.where(
        in_response, relative_position, torch.zeros_like(relative_position)
    )
    factor = 1.0 - float(gamma) * (1.0 - tau) * relative_position
    factor = factor.to(dtype=confidence.dtype)
    while factor.ndim < confidence.ndim:
        factor = factor.unsqueeze(0)
    return torch.where(
        torch.isfinite(confidence), confidence * factor, confidence
    )


@dataclass
class SwdState:
    """Previous response-span distributions for SWD."""

    prev_probs: Optional[torch.Tensor] = None
    response_start: Optional[int] = None
    response_end: Optional[int] = None


def kl_prev_vs_curr(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute per-position ``KL(previous || current)``."""

    if previous.shape != current.shape:
        raise ValueError(
            "SWD KL requires matching shapes, got "
            f"{tuple(previous.shape)} and {tuple(current.shape)}"
        )
    previous = previous.float().clamp_min(eps)
    current = current.float().clamp_min(eps)
    previous = previous / previous.sum(dim=-1, keepdim=True)
    current = current / current.sum(dim=-1, keepdim=True)
    return (previous * (previous.log() - current.log())).sum(dim=-1)


def stability_weight(
    current: torch.Tensor,
    previous: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Return ``exp(-lambda * KL(previous || current))`` per position."""

    return torch.exp(-float(lambda_) * kl_prev_vs_curr(previous, current))


def response_span_token_probs(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    *,
    response_start: int,
    response_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute selected-token confidence and distributions on response only."""

    start, end = int(response_start), int(response_end)
    if end <= start:
        raise ValueError(f"Invalid response span [{start}, {end})")
    if predicted_tokens.shape != logits.shape[:-1]:
        raise ValueError(
            "predicted_tokens must match logits leading dimensions, got "
            f"{tuple(predicted_tokens.shape)} and {tuple(logits.shape[:-1])}"
        )

    probabilities = torch.softmax(logits[:, start:end, :].float(), dim=-1)
    selected = torch.gather(
        probabilities,
        dim=-1,
        index=predicted_tokens[:, start:end].unsqueeze(-1),
    ).squeeze(-1)
    confidence = torch.full(
        predicted_tokens.shape,
        float("-inf"),
        device=logits.device,
        dtype=torch.float32,
    )
    confidence[:, start:end] = selected
    return confidence, probabilities


def apply_swd(
    confidence: torch.Tensor,
    logits: torch.Tensor,
    state: SwdState,
    *,
    lambda_: float,
    response_start: int,
    response_end: int,
    mask_index: Optional[torch.Tensor] = None,
    current_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply SWD weights and update the previous-step state."""

    start, end = int(response_start), int(response_end)
    if end <= start:
        raise ValueError(f"Invalid response span [{start}, {end})")
    if confidence.shape != logits.shape[:-1]:
        raise ValueError(
            "confidence must match logits leading dimensions, got "
            f"{tuple(confidence.shape)} and {tuple(logits.shape[:-1])}"
        )

    if state.response_start is None:
        state.response_start = start
        state.response_end = end
    elif (state.response_start, state.response_end) != (start, end):
        raise RuntimeError(
            "SWD response span changed during decoding: "
            f"{(state.response_start, state.response_end)} -> {(start, end)}"
        )

    if current_probs is None:
        current_probs = torch.softmax(
            logits[:, start:end, :].float(), dim=-1
        )
    else:
        expected = (logits.shape[0], end - start, logits.shape[-1])
        if tuple(current_probs.shape) != expected:
            raise ValueError(
                f"current_probs must have shape {expected}, "
                f"got {tuple(current_probs.shape)}"
            )
        current_probs = current_probs.float()

    if state.prev_probs is None:
        previous = torch.full_like(
            current_probs, 1.0 / float(current_probs.shape[-1])
        )
    else:
        if state.prev_probs.shape != current_probs.shape:
            raise RuntimeError(
                "SWD history shape mismatch: "
                f"{tuple(state.prev_probs.shape)} and "
                f"{tuple(current_probs.shape)}"
            )
        previous = state.prev_probs

    weight = stability_weight(current_probs, previous, lambda_)
    response_confidence = confidence[:, start:end]
    modulated = torch.where(
        torch.isfinite(response_confidence),
        response_confidence * weight.to(response_confidence.dtype),
        response_confidence,
    )
    if mask_index is not None:
        modulated = torch.where(
            mask_index[:, start:end], modulated, response_confidence
        )

    output = confidence.clone()
    output[:, start:end] = modulated
    state.prev_probs = current_probs.detach()
    return output
