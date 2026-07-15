from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from .config import VCHDDecodeConfig
from .contrast import compute_contrast_stats
from .history import (
    SparseHistory,
    history_adjusted_reliability,
    observe_sparse_history,
)
from .mmada_adapter import MMaDAVisualAccessAdapter
from .selector import (
    MAX_WINDOW_TOP1_FALLBACK,
    THRESHOLD_COMMIT,
    select_ccaw_positions,
    select_fixed_window_positions,
)
from .vocabulary import build_valid_text_vocab
from .window import CCAWState, compute_window_pressure, update_ccaw_state


DecodeOutput = Union[torch.LongTensor, Tuple[torch.LongTensor, Dict[str, Any]]]


def _resolve_text_vocab_size(
    model, config: VCHDDecodeConfig, vocab_size: int
) -> int:
    if config.text_vocab_size is not None:
        return int(config.text_vocab_size)
    model_config = getattr(model, "config", None)
    configured = getattr(model_config, "llm_vocab_size", None)
    if configured is None:
        return vocab_size
    return int(configured)


def _truncate_full_sequence_at_eos(
    tokens: torch.LongTensor,
    *,
    decode_start: int,
    decode_end: int,
    eos_token_id: Optional[int],
) -> torch.LongTensor:
    if eos_token_id is None:
        return tokens
    response = tokens[0, decode_start:decode_end]
    eos_positions = torch.nonzero(
        response == int(eos_token_id), as_tuple=True
    )[0]
    if eos_positions.numel() == 0:
        return tokens
    end = decode_start + int(eos_positions[0].item()) + 1
    return tokens[:, :end]


@torch.no_grad()
def visual_contrast_decode(
    model,
    tokens: torch.LongTensor,
    *,
    decode_start: int,
    decode_end: int,
    image_span: Tuple[int, int],
    config: VCHDDecodeConfig,
    attention_mask: Optional[torch.Tensor] = None,
) -> DecodeOutput:
    """Run the phase 0--2 backend-matched, dual-gate VCHD decoder."""

    config.validate()
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("The phase 0--2 VCHD decoder supports batch_size=1 only")
    if not 0 <= decode_start < decode_end <= tokens.shape[1]:
        raise ValueError(
            f"Invalid decode range [{decode_start}, {decode_end}) for "
            f"sequence length {tokens.shape[1]}"
        )

    state = tokens.clone()
    adapter = MMaDAVisualAccessAdapter(
        model,
        decode_start=decode_start,
        decode_end=decode_end,
        image_span=image_span,
        attention_mask=attention_mask,
        force_math_sdpa=config.force_math_sdpa,
        mask_id=config.mask_id,
        cache_type=config.cache_type,
        cache_refresh_interval=config.cache_refresh_interval,
    )

    valid_text_vocab: Optional[torch.BoolTensor] = None
    model_evaluations = 0
    threshold_commits = 0
    fallback_commits = 0
    trace = []
    response_length = decode_end - decode_start
    context_version = 0
    history: Dict[int, SparseHistory] = {}
    history_observations = 0
    history_stability_sum = 0.0
    ccaw_state = CCAWState(mask_capacity=int(config.mask_capacity))
    ccaw_search_expansions = 0
    ccaw_pressure_sum = 0.0
    ccaw_pressure_count = 0
    ccaw_min_capacity = int(config.mask_capacity)
    ccaw_max_capacity = int(config.mask_capacity)
    cache_pressure_refresh_requests = 0

    while True:
        response = state[0, decode_start:decode_end]
        mask = response == int(config.mask_id)
        if not bool(mask.any()):
            break
        if context_version >= response_length:
            raise RuntimeError(
                "VCHD exceeded the one-commit-per-context termination bound"
            )

        paired = adapter.paired_forward(
            state,
            context_version=context_version,
        )
        model_evaluations += 1
        if valid_text_vocab is None:
            vocab_size = int(paired.visual.shape[-1])
            valid_text_vocab = build_valid_text_vocab(
                vocab_size,
                text_vocab_size=_resolve_text_vocab_size(
                    model, config, vocab_size
                ),
                forbidden_token_ids=config.forbidden_token_ids,
                eos_token_id=config.eos_token_id,
                device=paired.visual.device,
            )

        stats = compute_contrast_stats(
            paired.visual,
            paired.ablated,
            valid_text_vocab,
            alpha=config.alpha,
            beta=config.beta,
            history_top_v_tokens=(
                config.history_top_v_tokens
                if config.history_enabled
                else None
            ),
        )
        history_stability = torch.ones_like(stats.contrast_confidence)
        contrast_reliability = stats.contrast_confidence
        history_updates: Dict[int, SparseHistory] = {}
        snapshot_history_observations = 0
        snapshot_history_stability_sum = 0.0
        if config.history_enabled:
            if (
                stats.contrast_top_ids is None
                or stats.contrast_top_probs is None
                or stats.contrast_other_prob is None
            ):
                raise RuntimeError(
                    "History-enabled decoding requires sparse contrast outputs"
                )
            for position_tensor in torch.nonzero(mask, as_tuple=True)[0]:
                position = int(position_tensor.item())
                observation = observe_sparse_history(
                    history.get(position),
                    stats.contrast_top_ids[position],
                    stats.contrast_top_probs[position],
                    stats.contrast_other_prob[position],
                    context_version=context_version,
                    ema_decay=config.history_ema_decay,
                    top_v_tokens=config.history_top_v_tokens,
                )
                history_stability[position] = observation.stability
                history_updates[position] = observation.next_history
                snapshot_history_observations += int(observation.updated)
                snapshot_history_stability_sum += float(
                    observation.stability.item()
                )
            contrast_reliability = history_adjusted_reliability(
                stats.contrast_confidence,
                stats.visual_relevance,
                history_stability,
            )

        if config.ccaw_enabled:
            selection = select_ccaw_positions(
                stats,
                mask,
                config,
                contrast_reliability=contrast_reliability,
                current_mask_capacity=ccaw_state.mask_capacity,
            )
        else:
            selection = select_fixed_window_positions(
                stats,
                mask,
                config,
                contrast_reliability=contrast_reliability,
            )
        selected = selection.positions
        if selected.numel() == 0:
            raise RuntimeError("Selector returned no token and would deadlock")

        use_raw = (
            selection.reason == MAX_WINDOW_TOP1_FALLBACK
            and config.fallback_to_raw
        )
        selected_tokens = (
            stats.raw_token[selected]
            if use_raw
            else stats.contrast_token[selected]
        )
        compute_pressure = config.ccaw_enabled or (
            config.cache_type == "dual"
            and config.cache_refresh_on_pressure
        )
        pressure = (
            compute_window_pressure(
                stats,
                history_stability,
                contrast_reliability,
                selection.window,
                config,
            )
            if compute_pressure
            else None
        )
        if (
            pressure is not None
            and config.cache_type == "dual"
            and config.cache_refresh_on_pressure
            and paired.cache_event == "partial_refresh"
            and pressure.combined >= float(config.cache_pressure_threshold)
        ):
            # Re-evaluate the unchanged context with exact full-sequence branch
            # caches before accepting a high-conflict approximate decision.
            adapter.request_full_refresh("conflict_pressure")
            cache_pressure_refresh_requests += 1
            continue

        # Atomic commit: gather the complete decision before modifying state.
        absolute_positions = selected + int(decode_start)
        if not bool((state[0, absolute_positions] == config.mask_id).all()):
            raise RuntimeError("Selector attempted to overwrite a committed token")
        state[0, absolute_positions] = selected_tokens.to(state.dtype)

        if selection.reason == THRESHOLD_COMMIT:
            threshold_commits += int(selected.numel())
        else:
            fallback_commits += int(selected.numel())
        ccaw_search_expansions += int(selection.search_expansions)
        history_observations += snapshot_history_observations
        history_stability_sum += snapshot_history_stability_sum

        if config.collect_trace:
            pressure_values = (
                None
                if pressure is None
                else {
                    "candidate_conflict": pressure.candidate_conflict,
                    "history_instability": pressure.history_instability,
                    "qualification_deficit": pressure.qualification_deficit,
                    "combined": pressure.combined,
                }
            )
            trace.append(
                {
                    "model_evaluation": model_evaluations,
                    "context_version": context_version,
                    "cache_event": paired.cache_event,
                    "cache_refresh_reason": paired.cache_refresh_reason,
                    "cache_query_start": paired.query_start,
                    "cache_query_tokens": paired.query_tokens,
                    "cache_model_forward_calls": paired.model_forward_calls,
                    "window_left": selection.window.left,
                    "window_right": selection.window.right,
                    "window_mask_capacity": selection.window.mask_capacity,
                    "window_active_masks": int(
                        selection.window.active_positions.numel()
                    ),
                    "persistent_mask_capacity": ccaw_state.mask_capacity,
                    "search_expansions": selection.search_expansions,
                    "selected_positions": selected.detach().cpu().tolist(),
                    "selected_tokens": selected_tokens.detach().cpu().tolist(),
                    "commit_reason": selection.reason,
                    "base_confidence": stats.base_confidence[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "contrast_confidence": stats.contrast_confidence[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "history_stability": history_stability[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "commit_reliability": contrast_reliability[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "visual_relevance": stats.visual_relevance[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "raw_changed": (
                        stats.raw_token[selected] != stats.contrast_token[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "window_pressure": pressure_values,
                }
            )

        selected_set = set(selected.detach().cpu().tolist())
        if config.history_enabled:
            history = {
                position: update
                for position, update in history_updates.items()
                if position not in selected_set
            }
        context_version += 1
        if config.ccaw_enabled and pressure is not None:
            ccaw_pressure_sum += pressure.combined
            ccaw_pressure_count += 1
            update_ccaw_state(ccaw_state, pressure, config)
            ccaw_min_capacity = min(
                ccaw_min_capacity, ccaw_state.mask_capacity
            )
            ccaw_max_capacity = max(
                ccaw_max_capacity, ccaw_state.mask_capacity
            )

    output = (
        _truncate_full_sequence_at_eos(
            state,
            decode_start=decode_start,
            decode_end=decode_end,
            eos_token_id=config.eos_token_id,
        )
        if config.truncate_at_eos
        else state
    )
    report: Dict[str, Any] = {
        "decoder": "vchd_fixed_dual_gate",
        "model_evaluations": model_evaluations,
        "threshold_commits": threshold_commits,
        "fallback_commits": fallback_commits,
        "output_tokens": int(output.shape[1] - decode_start),
        "alpha": float(config.alpha),
        "beta": float(config.beta),
        "tau_base": float(config.tau_base),
        "tau_contrast": float(config.tau_contrast),
        "fallback_to_raw": bool(config.fallback_to_raw),
        "context_versions": context_version,
        "history_enabled": bool(config.history_enabled),
        "history_top_v_tokens": int(config.history_top_v_tokens),
        "history_ema_decay": float(config.history_ema_decay),
        "history_observations": history_observations,
        "mean_history_stability": (
            history_stability_sum / history_observations
            if history_observations
            else 1.0
        ),
        "ccaw_enabled": bool(config.ccaw_enabled),
        "ccaw_search_expansions": ccaw_search_expansions,
        "ccaw_final_mask_capacity": ccaw_state.mask_capacity,
        "ccaw_min_mask_capacity": ccaw_min_capacity,
        "ccaw_max_mask_capacity_seen": ccaw_max_capacity,
        "ccaw_mean_pressure": (
            ccaw_pressure_sum / ccaw_pressure_count
            if ccaw_pressure_count
            else 0.0
        ),
        "cache_refresh_interval": int(config.cache_refresh_interval),
        "cache_refresh_on_pressure": bool(
            config.cache_refresh_on_pressure
        ),
        "cache_pressure_threshold": float(
            config.cache_pressure_threshold
        ),
        "cache_pressure_refresh_requests": (
            cache_pressure_refresh_requests
        ),
    }
    report.update(adapter.cache_report())
    if config.collect_trace:
        report["trace"] = trace
    return (output, report) if config.return_report else output
