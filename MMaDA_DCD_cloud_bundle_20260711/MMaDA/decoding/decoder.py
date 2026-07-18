from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from .config import (
    EOSTokenId,
    VCHDDecodeConfig,
    normalize_eos_token_ids,
)
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
    build_fixed_window,
    select_ccaw_positions,
    select_fixed_window_positions,
)
from .vocabulary import build_valid_text_vocab
from .window import (
    CCAWState,
    compute_window_pressure,
    pressure_adaptive_commit_budget,
    scope_next_hard_block,
    update_ccaw_state,
    update_inverse_ccaw_state,
)


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
    eos_token_id: EOSTokenId,
) -> torch.LongTensor:
    eos_token_ids = normalize_eos_token_ids(eos_token_id)
    if not eos_token_ids:
        return tokens
    response = tokens[0, decode_start:decode_end]
    is_eos = torch.zeros_like(response, dtype=torch.bool)
    for eos_token_id_value in eos_token_ids:
        is_eos |= response == eos_token_id_value
    eos_positions = torch.nonzero(is_eos, as_tuple=True)[0]
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
    history_anchor_forced_deferrals = 0
    initial_ccaw_capacity = (
        int(config.ccaw_block_size)
        if config.ccaw_enabled and config.ccaw_mode == "hard_block"
        else int(config.mask_capacity)
    )
    ccaw_state = CCAWState(mask_capacity=initial_ccaw_capacity)
    hard_block_start = 0
    ccaw_search_expansions = 0
    ccaw_pressure_sum = 0.0
    ccaw_pressure_count = 0
    ccaw_min_capacity = initial_ccaw_capacity
    ccaw_max_capacity = initial_ccaw_capacity
    cache_pressure_refresh_requests = 0
    scored_mask_positions = 0
    selected_positions_scored = 0
    contrast_token_changes = 0
    selected_contrast_token_changes = 0
    base_confidence_sum = 0.0
    contrast_confidence_sum = 0.0
    contrast_reliability_sum = 0.0
    apc_mass_sum = 0.0
    visual_relevance_sum = 0.0
    threshold_commit_events = 0
    fallback_commit_events = 0
    ccaw_candidate_conflict_sum = 0.0
    ccaw_history_instability_sum = 0.0
    ccaw_qualification_deficit_sum = 0.0
    ccaw_pressure_sum_qualified = 0.0
    ccaw_pressure_sum_fallback = 0.0
    ccaw_qualification_deficit_sum_qualified = 0.0
    ccaw_qualification_deficit_sum_fallback = 0.0
    ccaw_qualified_commit_count = 0
    ccaw_fallback_commit_count = 0
    ccaw_commit_budget_sum = 0
    ccaw_commit_budget_count = 0
    ccaw_min_commit_budget = int(config.max_commit_per_iteration)
    ccaw_max_commit_budget = int(config.max_commit_per_iteration)

    while True:
        response = state[0, decode_start:decode_end]
        mask = response == int(config.mask_id)
        if not bool(mask.any()):
            break
        if context_version >= response_length:
            raise RuntimeError(
                "VCHD exceeded the one-commit-per-context termination bound"
            )
        selection_mask = mask
        active_block_start: Optional[int] = None
        active_block_end: Optional[int] = None
        if config.ccaw_enabled and config.ccaw_mode == "hard_block":
            (
                selection_mask,
                hard_block_start,
                active_block_end,
            ) = scope_next_hard_block(
                mask,
                block_start=hard_block_start,
                block_size=config.ccaw_block_size,
            )
            active_block_start = hard_block_start

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
        history_consistency = torch.zeros_like(
            stats.contrast_confidence,
            dtype=torch.long,
        )
        contrast_reliability = stats.contrast_confidence
        history_anchor_position = int(
            torch.nonzero(selection_mask, as_tuple=True)[0][0].item()
        )
        history_anchor_forced = False
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
                history_consistency[position] = (
                    observation.consecutive_top1_matches
                )
                history_updates[position] = observation.next_history
                snapshot_history_observations += int(observation.updated)
                snapshot_history_stability_sum += float(
                    observation.stability.item()
                )
            contrast_reliability = history_adjusted_reliability(
                stats.contrast_confidence,
                stats.visual_relevance,
                history_stability,
                penalty_scale=config.history_penalty_scale,
            )
            if (
                config.history_anchor_min_consistent
                and int(history_consistency[history_anchor_position].item())
                < int(config.history_anchor_min_consistent)
            ):
                contrast_reliability = contrast_reliability.clone()
                contrast_reliability[history_anchor_position] = 0.0
                history_anchor_forced = True

        pressure = None
        commit_budget = int(config.max_commit_per_iteration)
        if config.ccaw_enabled and config.ccaw_mode == "hard_block":
            pressure_window = build_fixed_window(
                selection_mask,
                mask_capacity=config.ccaw_block_size,
                max_physical_span=config.ccaw_block_size,
            )
            pressure = compute_window_pressure(
                stats,
                history_stability,
                contrast_reliability,
                pressure_window,
                config,
            )
            commit_budget = pressure_adaptive_commit_budget(
                pressure,
                config,
            )
            selection = select_fixed_window_positions(
                stats,
                selection_mask,
                config,
                contrast_reliability=contrast_reliability,
                mask_capacity=config.ccaw_block_size,
                max_commit_per_iteration=commit_budget,
            )
            ccaw_state.mask_capacity = int(config.ccaw_block_size)
        elif config.ccaw_enabled and config.ccaw_mode == "inverse_window":
            pressure_window = build_fixed_window(
                mask,
                mask_capacity=config.ccaw_max_mask_capacity,
                max_physical_span=config.max_physical_span,
            )
            pressure = compute_window_pressure(
                stats,
                history_stability,
                contrast_reliability,
                pressure_window,
                config,
            )
            update_inverse_ccaw_state(ccaw_state, pressure, config)
            selection = select_fixed_window_positions(
                stats,
                mask,
                config,
                contrast_reliability=contrast_reliability,
                mask_capacity=ccaw_state.mask_capacity,
            )
        elif config.ccaw_enabled:
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
        anchor_qualified = bool(
            stats.base_confidence[history_anchor_position]
            >= float(config.tau_base)
            and contrast_reliability[history_anchor_position]
            >= float(config.tau_contrast)
        )

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
        if pressure is None and compute_pressure:
            pressure = compute_window_pressure(
                stats,
                history_stability,
                contrast_reliability,
                selection.window,
                config,
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

        # Aggregate only the accepted pre-commit snapshot. Approximate cache
        # snapshots retried above must not count as decoder decisions.
        masked_positions = torch.nonzero(mask, as_tuple=True)[0]
        scored_mask_positions += int(masked_positions.numel())
        selected_positions_scored += int(selected.numel())
        contrast_token_changes += int(
            (
                stats.raw_token[masked_positions]
                != stats.contrast_token[masked_positions]
            ).sum().item()
        )
        selected_contrast_token_changes += int(
            (
                stats.raw_token[selected] != stats.contrast_token[selected]
            ).sum().item()
        )
        base_confidence_sum += float(
            stats.base_confidence[masked_positions].sum().item()
        )
        contrast_confidence_sum += float(
            stats.contrast_confidence[masked_positions].sum().item()
        )
        contrast_reliability_sum += float(
            contrast_reliability[masked_positions].sum().item()
        )
        apc_mass_sum += float(stats.apc_mass[masked_positions].sum().item())
        visual_relevance_sum += float(
            stats.visual_relevance[masked_positions].sum().item()
        )

        # Atomic commit: gather the complete decision before modifying state.
        absolute_positions = selected + int(decode_start)
        if not bool((state[0, absolute_positions] == config.mask_id).all()):
            raise RuntimeError("Selector attempted to overwrite a committed token")
        state[0, absolute_positions] = selected_tokens.to(state.dtype)

        if selection.reason == THRESHOLD_COMMIT:
            threshold_commits += int(selected.numel())
            threshold_commit_events += 1
        else:
            fallback_commits += int(selected.numel())
            fallback_commit_events += 1
        ccaw_search_expansions += int(selection.search_expansions)
        history_observations += snapshot_history_observations
        history_stability_sum += snapshot_history_stability_sum
        history_anchor_forced_deferrals += int(
            history_anchor_forced
            and not bool((selected == history_anchor_position).any())
        )

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
                    "ccaw_mode": config.ccaw_mode,
                    "hard_block_left": active_block_start,
                    "hard_block_right": active_block_end,
                    "window_left": selection.window.left,
                    "window_right": selection.window.right,
                    "window_mask_capacity": selection.window.mask_capacity,
                    "window_active_masks": int(
                        selection.window.active_positions.numel()
                    ),
                    "qualified_count": selection.qualified_count,
                    "qualified_budget": min(
                        int(config.ccaw_qualified_budget),
                        int(selection.window.active_positions.numel()),
                    ),
                    "history_anchor_position": history_anchor_position,
                    "history_anchor_qualified": anchor_qualified,
                    "history_anchor_consistent_observations": int(
                        history_consistency[history_anchor_position].item()
                    ),
                    "history_anchor_forced_deferral": history_anchor_forced,
                    "commit_budget": commit_budget,
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
            ccaw_candidate_conflict_sum += pressure.candidate_conflict
            ccaw_history_instability_sum += pressure.history_instability
            ccaw_qualification_deficit_sum += pressure.qualification_deficit
            ccaw_pressure_count += 1
            # Split-by-reason accumulators. Fallback steps (no qualified
            # candidate) systematically produce pressure~=1/3 because
            # qualification_deficit=1 and the other two terms are ~0, so
            # aggregating them into the overall mean masks the true
            # per-step signal. Keep the total mean, but also expose the
            # qualified/fallback strata separately.
            if selection.reason == THRESHOLD_COMMIT:
                ccaw_pressure_sum_qualified += pressure.combined
                ccaw_qualification_deficit_sum_qualified += (
                    pressure.qualification_deficit
                )
                ccaw_qualified_commit_count += 1
            else:
                ccaw_pressure_sum_fallback += pressure.combined
                ccaw_qualification_deficit_sum_fallback += (
                    pressure.qualification_deficit
                )
                ccaw_fallback_commit_count += 1
            ccaw_commit_budget_sum += commit_budget
            ccaw_commit_budget_count += 1
            ccaw_min_commit_budget = min(
                ccaw_min_commit_budget,
                commit_budget,
            )
            ccaw_max_commit_budget = max(
                ccaw_max_commit_budget,
                commit_budget,
            )
            if config.ccaw_mode == "legacy":
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
        "threshold_commit_events": threshold_commit_events,
        "fallback_commit_events": fallback_commit_events,
        "output_tokens": int(output.shape[1] - decode_start),
        "alpha": float(config.alpha),
        "beta": float(config.beta),
        "tau_base": float(config.tau_base),
        "tau_contrast": float(config.tau_contrast),
        "fallback_to_raw": bool(config.fallback_to_raw),
        "fallback_mask_capacity": int(config.fallback_mask_capacity),
        "fallback_policy": config.fallback_policy,
        "context_versions": context_version,
        "scored_mask_positions": scored_mask_positions,
        "contrast_token_changes": contrast_token_changes,
        "contrast_token_change_rate": (
            contrast_token_changes / scored_mask_positions
            if scored_mask_positions
            else 0.0
        ),
        "selected_contrast_token_change_rate": (
            selected_contrast_token_changes / selected_positions_scored
            if selected_positions_scored
            else 0.0
        ),
        "mean_base_confidence": (
            base_confidence_sum / scored_mask_positions
            if scored_mask_positions
            else 0.0
        ),
        "mean_contrast_confidence": (
            contrast_confidence_sum / scored_mask_positions
            if scored_mask_positions
            else 0.0
        ),
        "mean_contrast_reliability": (
            contrast_reliability_sum / scored_mask_positions
            if scored_mask_positions
            else 0.0
        ),
        "mean_history_reliability_penalty": (
            (contrast_confidence_sum - contrast_reliability_sum)
            / scored_mask_positions
            if scored_mask_positions
            else 0.0
        ),
        "mean_apc_mass": (
            apc_mass_sum / scored_mask_positions
            if scored_mask_positions
            else 0.0
        ),
        "mean_visual_relevance": (
            visual_relevance_sum / scored_mask_positions
            if scored_mask_positions
            else 0.0
        ),
        "history_enabled": bool(config.history_enabled),
        "history_top_v_tokens": int(config.history_top_v_tokens),
        "history_ema_decay": float(config.history_ema_decay),
        "history_penalty_scale": float(config.history_penalty_scale),
        "history_anchor_min_consistent": int(
            config.history_anchor_min_consistent
        ),
        "history_anchor_forced_deferrals": history_anchor_forced_deferrals,
        "history_observations": history_observations,
        "mean_history_stability": (
            history_stability_sum / history_observations
            if history_observations
            else 1.0
        ),
        "ccaw_enabled": bool(config.ccaw_enabled),
        "ccaw_mode": config.ccaw_mode,
        "ccaw_pressure_scale": float(config.ccaw_pressure_scale),
        "ccaw_block_size": int(config.ccaw_block_size),
        "ccaw_min_commit_per_iteration": int(
            config.ccaw_min_commit_per_iteration
        ),
        "ccaw_qualified_budget": int(config.ccaw_qualified_budget),
        "ccaw_search_expansions": ccaw_search_expansions,
        "ccaw_final_mask_capacity": ccaw_state.mask_capacity,
        "ccaw_min_mask_capacity": ccaw_min_capacity,
        "ccaw_max_mask_capacity_seen": ccaw_max_capacity,
        "ccaw_mean_pressure": (
            ccaw_pressure_sum / ccaw_pressure_count
            if ccaw_pressure_count
            else 0.0
        ),
        "ccaw_mean_candidate_conflict": (
            ccaw_candidate_conflict_sum / ccaw_pressure_count
            if ccaw_pressure_count
            else 0.0
        ),
        "ccaw_mean_history_instability": (
            ccaw_history_instability_sum / ccaw_pressure_count
            if ccaw_pressure_count
            else 0.0
        ),
        "ccaw_mean_qualification_deficit": (
            ccaw_qualification_deficit_sum / ccaw_pressure_count
            if ccaw_pressure_count
            else 0.0
        ),
        "ccaw_pressure_filter": config.ccaw_pressure_filter,
        "ccaw_qualified_commit_count": ccaw_qualified_commit_count,
        "ccaw_fallback_commit_count": ccaw_fallback_commit_count,
        "ccaw_mean_pressure_qualified": (
            ccaw_pressure_sum_qualified / ccaw_qualified_commit_count
            if ccaw_qualified_commit_count
            else 0.0
        ),
        "ccaw_mean_pressure_fallback": (
            ccaw_pressure_sum_fallback / ccaw_fallback_commit_count
            if ccaw_fallback_commit_count
            else 0.0
        ),
        "ccaw_mean_qualification_deficit_qualified": (
            ccaw_qualification_deficit_sum_qualified
            / ccaw_qualified_commit_count
            if ccaw_qualified_commit_count
            else 0.0
        ),
        "ccaw_mean_qualification_deficit_fallback": (
            ccaw_qualification_deficit_sum_fallback
            / ccaw_fallback_commit_count
            if ccaw_fallback_commit_count
            else 0.0
        ),
        "ccaw_mean_commit_budget": (
            ccaw_commit_budget_sum / ccaw_commit_budget_count
            if ccaw_commit_budget_count
            else float(config.max_commit_per_iteration)
        ),
        "ccaw_min_commit_budget_seen": (
            ccaw_min_commit_budget
            if ccaw_commit_budget_count
            else int(config.max_commit_per_iteration)
        ),
        "ccaw_max_commit_budget_seen": (
            ccaw_max_commit_budget
            if ccaw_commit_budget_count
            else int(config.max_commit_per_iteration)
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
