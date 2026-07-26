from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from .config import EOSTokenId, VCHDDecodeConfig, normalize_eos_token_ids
from .contrast import compute_contrast_stats
from .history import (
    SparseHistory,
    history_adjusted_reliability,
    observe_sparse_history,
)
from .lavida_adapter import TokenVisualAccessAdapter
from .selector import (
    MAX_WINDOW_TOP1_FALLBACK,
    THRESHOLD_COMMIT,
    select_ccaw_positions,
    select_fixed_window_positions,
)
from .selector_cd_apc import (
    position_gate_mask,
    select_ccaw_cd_apc_positions,
    select_cd_apc_triple_gate_positions,
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
        configured = getattr(model_config, "vocab_size", None)
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
    eos_ids = normalize_eos_token_ids(eos_token_id)
    if not eos_ids:
        return tokens
    response = tokens[0, decode_start:decode_end]
    hits = torch.zeros_like(response, dtype=torch.bool)
    for eos_id in eos_ids:
        hits |= response == int(eos_id)
    eos_positions = torch.nonzero(hits, as_tuple=True)[0]
    if eos_positions.numel() == 0:
        return tokens
    end = decode_start + int(eos_positions[0].item()) + 1
    return tokens[:, :end]


def _select_positions(
    stats,
    mask,
    config: VCHDDecodeConfig,
    *,
    contrast_reliability,
    current_mask_capacity: int,
    max_commit_per_iteration: Optional[int] = None,
):
    use_triple = bool(config.enable_g_gate)
    if config.ccaw_enabled:
        if use_triple:
            return select_ccaw_cd_apc_positions(
                stats,
                mask,
                config,
                contrast_reliability=contrast_reliability,
                current_mask_capacity=current_mask_capacity,
                max_commit_per_iteration=max_commit_per_iteration,
            )
        return select_ccaw_positions(
            stats,
            mask,
            config,
            contrast_reliability=contrast_reliability,
            current_mask_capacity=current_mask_capacity,
            max_commit_per_iteration=max_commit_per_iteration,
        )
    if use_triple:
        return select_cd_apc_triple_gate_positions(
            stats,
            mask,
            config,
            contrast_reliability=contrast_reliability,
            max_commit_per_iteration=max_commit_per_iteration,
        )
    return select_fixed_window_positions(
        stats,
        mask,
        config,
        contrast_reliability=contrast_reliability,
        max_commit_per_iteration=max_commit_per_iteration,
    )


@torch.no_grad()
def visual_contrast_decode(
    model,
    tokens: torch.LongTensor,
    *,
    decode_start: int,
    decode_end: int,
    image_span: Optional[Tuple[int, int]] = None,
    visual_mask: Optional[torch.BoolTensor] = None,
    config: VCHDDecodeConfig,
    attention_mask: Optional[torch.Tensor] = None,
    adapter=None,
) -> DecodeOutput:
    """Run backend-matched VCHD / CD-APC decoding with optional CCAW."""

    config.validate()
    if config.cache_type != "none":
        raise ValueError(
            "LaViDa VCHD currently supports cache_type='none' only; "
            f"got {config.cache_type!r}"
        )
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("The VCHD decoder supports batch_size=1 only")
    if not 0 <= decode_start < decode_end <= tokens.shape[1]:
        raise ValueError(
            f"Invalid decode range [{decode_start}, {decode_end}) for "
            f"sequence length {tokens.shape[1]}"
        )

    state = tokens.clone()
    if adapter is None:
        adapter = TokenVisualAccessAdapter(
            model,
            decode_start=decode_start,
            decode_end=decode_end,
            image_span=image_span,
            visual_mask=visual_mask,
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
    threshold_commit_events = 0
    fallback_commit_events = 0
    trace = []
    response_length = decode_end - decode_start
    context_version = 0
    history: Dict[int, SparseHistory] = {}
    history_observations = 0
    history_stability_sum = 0.0
    history_anchor_forced_deferrals = 0
    initial_ccaw_capacity = int(config.mask_capacity)
    ccaw_state = CCAWState(mask_capacity=initial_ccaw_capacity)
    hard_block_start = 0
    ccaw_search_expansions = 0
    ccaw_pressure_sum = 0.0
    ccaw_pressure_count = 0
    ccaw_min_capacity = initial_ccaw_capacity
    ccaw_max_capacity = initial_ccaw_capacity
    scored_mask_positions = 0
    selected_positions_scored = 0
    contrast_token_changes = 0
    selected_contrast_token_changes = 0
    base_confidence_sum = 0.0
    contrast_confidence_sum = 0.0
    contrast_reliability_sum = 0.0
    reliability_penalty_sum = 0.0
    apc_mass_sum = 0.0
    visual_relevance_sum = 0.0
    visual_gain_sum = 0.0
    g_gate_ur_pass_positions = 0
    g_gate_blocked_positions = 0

    while True:
        response = state[0, decode_start:decode_end]
        mask = response == int(config.mask_id)
        if not bool(mask.any()):
            break
        if context_version >= response_length:
            raise RuntimeError(
                "VCHD exceeded the one-commit-per-context termination bound"
            )

        hard_block_left = 0
        hard_block_right = int(mask.numel())
        selection_mask = mask
        commit_budget = int(config.max_commit_per_iteration)
        if config.ccaw_enabled and config.ccaw_mode == "hard_block":
            selection_mask, hard_block_left, hard_block_right = scope_next_hard_block(
                mask,
                block_start=hard_block_start,
                block_size=int(config.ccaw_block_size),
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
        history_anchor_matches = {
            position: 0 for position in range(int(mask.numel()))
        }
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
                history_anchor_matches[position] = int(
                    observation.consecutive_top1_matches
                )
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

        if config.ccaw_enabled and config.ccaw_mode == "inverse_window":
            # Inverse mode maps low pressure to a larger capacity before search.
            minimum = int(config.mask_capacity)
            maximum = int(config.ccaw_max_mask_capacity)
            current_pressure = min(
                1.0,
                max(
                    0.0,
                    float(ccaw_state.pressure_ema)
                    * float(config.ccaw_pressure_scale),
                ),
            )
            ccaw_state.mask_capacity = maximum - round(
                (maximum - minimum) * current_pressure
            )

        if config.ccaw_enabled and config.ccaw_mode == "hard_block":
            # Estimate pressure on the current hard block with a provisional window.
            provisional = select_fixed_window_positions(
                stats,
                selection_mask,
                config,
                contrast_reliability=contrast_reliability,
                mask_capacity=ccaw_state.mask_capacity,
            )
            provisional_pressure = compute_window_pressure(
                stats,
                history_stability,
                contrast_reliability,
                provisional.window,
                config,
            )
            commit_budget = pressure_adaptive_commit_budget(
                provisional_pressure, config
            )

        selection = _select_positions(
            stats,
            selection_mask,
            config,
            contrast_reliability=contrast_reliability,
            current_mask_capacity=ccaw_state.mask_capacity,
            max_commit_per_iteration=commit_budget,
        )

        # History-anchor: defer committing the left-most mask until stable.
        history_anchor_forced_deferral = False
        history_anchor_position = None
        history_anchor_qualified = True
        if (
            config.history_enabled
            and int(config.history_anchor_min_consistent) > 0
            and selection.positions.numel() > 0
        ):
            anchor = int(torch.nonzero(mask, as_tuple=True)[0][0].item())
            history_anchor_position = anchor
            matches = int(history_anchor_matches.get(anchor, 0))
            history_anchor_qualified = (
                matches >= int(config.history_anchor_min_consistent)
            )
            if not history_anchor_qualified and int(anchor) in set(
                selection.positions.detach().cpu().tolist()
            ):
                kept = selection.positions[selection.positions != anchor]
                if kept.numel() == 0:
                    # Force a non-anchor progress token inside the current
                    # selection window (respects CCAW hard-block scoping).
                    others = selection.window.active_positions
                    others = others[others != anchor]
                    if others.numel() > 0:
                        kept = others[:1]
                        selection = type(selection)(
                            positions=kept,
                            reason=MAX_WINDOW_TOP1_FALLBACK,
                            qualified_count=selection.qualified_count,
                            window=selection.window,
                            search_expansions=selection.search_expansions,
                        )
                        history_anchor_forced_deferral = True
                        history_anchor_forced_deferrals += 1
                else:
                    selection = type(selection)(
                        positions=kept,
                        reason=selection.reason,
                        qualified_count=selection.qualified_count,
                        window=selection.window,
                        search_expansions=selection.search_expansions,
                    )
                    history_anchor_forced_deferral = True
                    history_anchor_forced_deferrals += 1

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
        pressure = compute_window_pressure(
            stats,
            history_stability,
            contrast_reliability,
            selection.window,
            config,
        )

        absolute_positions = selected + int(decode_start)
        if not bool((state[0, absolute_positions] == config.mask_id).all()):
            raise RuntimeError("Selector attempted to overwrite a committed token")
        state[0, absolute_positions] = selected_tokens.to(state.dtype)

        mask_count = int(mask.sum().item())
        scored_mask_positions += mask_count
        selected_positions_scored += int(selected.numel())
        changed = stats.contrast_token != stats.raw_token
        contrast_token_changes += int(changed[mask].sum().item())
        selected_contrast_token_changes += int(changed[selected].sum().item())
        base_confidence_sum += float(stats.base_confidence[selected].sum().item())
        contrast_confidence_sum += float(
            stats.contrast_confidence[selected].sum().item()
        )
        contrast_reliability_sum += float(
            contrast_reliability[selected].sum().item()
        )
        reliability_penalty_sum += float(
            (
                stats.contrast_confidence[selected]
                - contrast_reliability[selected]
            )
            .clamp_min(0.0)
            .sum()
            .item()
        )
        apc_mass_sum += float(stats.apc_mass[selected].sum().item())
        visual_relevance_sum += float(
            stats.visual_relevance[selected].sum().item()
        )
        visual_gain_sum += float(
            stats.absolute_visual_gain[selected].sum().item()
        )
        if config.enable_g_gate:
            active = selection.window.active_positions
            base = stats.base_confidence[active]
            reliability = contrast_reliability[active]
            # Match position_gate_mask(): disabled u/r gates pass all positions.
            u_ok = (
                base >= float(config.tau_base)
                if config.enable_u_gate
                else torch.ones_like(base, dtype=torch.bool)
            )
            r_ok = (
                reliability >= float(config.tau_contrast)
                if config.enable_r_gate
                else torch.ones_like(reliability, dtype=torch.bool)
            )
            ur_pass = u_ok & r_ok
            if bool(ur_pass.any()):
                ur_pos = active[ur_pass]
                g_ok = position_gate_mask(
                    stats,
                    ur_pos,
                    config,
                    contrast_reliability=contrast_reliability,
                )
                g_gate_ur_pass_positions += int(ur_pos.numel())
                g_gate_blocked_positions += int((~g_ok).sum().item())

        if selection.reason == THRESHOLD_COMMIT:
            threshold_commits += int(selected.numel())
            threshold_commit_events += 1
        else:
            fallback_commits += int(selected.numel())
            fallback_commit_events += 1
        ccaw_search_expansions += int(selection.search_expansions)
        history_observations += snapshot_history_observations
        history_stability_sum += snapshot_history_stability_sum

        if config.collect_trace:
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
                    "hard_block_left": hard_block_left,
                    "hard_block_right": hard_block_right,
                    "window_left": selection.window.left,
                    "window_right": selection.window.right,
                    "window_mask_capacity": selection.window.mask_capacity,
                    "window_active_masks": int(
                        selection.window.active_positions.numel()
                    ),
                    "qualified_count": selection.qualified_count,
                    "qualified_budget": int(config.ccaw_qualified_budget),
                    "history_anchor_position": history_anchor_position,
                    "history_anchor_qualified": history_anchor_qualified,
                    "history_anchor_consistent_observations": (
                        None
                        if history_anchor_position is None
                        else history_anchor_matches.get(history_anchor_position, 0)
                    ),
                    "history_anchor_forced_deferral": history_anchor_forced_deferral,
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
                    "window_pressure": {
                        "candidate_conflict": pressure.candidate_conflict,
                        "history_instability": pressure.history_instability,
                        "qualification_deficit": pressure.qualification_deficit,
                        "combined": pressure.combined,
                    },
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

        if config.ccaw_enabled:
            ccaw_pressure_sum += pressure.combined
            ccaw_pressure_count += 1
            if config.ccaw_mode == "inverse_window":
                update_inverse_ccaw_state(ccaw_state, pressure, config)
            else:
                update_ccaw_state(ccaw_state, pressure, config)
            ccaw_min_capacity = min(ccaw_min_capacity, ccaw_state.mask_capacity)
            ccaw_max_capacity = max(ccaw_max_capacity, ccaw_state.mask_capacity)
            if config.ccaw_mode == "hard_block":
                # Advance only after the current hard block is empty.
                remaining = state[0, decode_start:decode_end] == int(config.mask_id)
                block_remaining = remaining[hard_block_left:hard_block_right]
                if not bool(block_remaining.any()):
                    hard_block_start = hard_block_right

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
    decoder_name = (
        "cd_apc_triple_gate" if config.enable_g_gate else "vchd_fixed_dual_gate"
    )
    report: Dict[str, Any] = {
        "decoder": decoder_name,
        "enable_g_gate": bool(config.enable_g_gate),
        "enable_u_gate": bool(config.enable_u_gate),
        "enable_r_gate": bool(config.enable_r_gate),
        "tau_g": float(config.tau_g),
        "g_min_policy": str(config.g_min_policy),
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
            base_confidence_sum / selected_positions_scored
            if selected_positions_scored
            else 0.0
        ),
        "mean_contrast_confidence": (
            contrast_confidence_sum / selected_positions_scored
            if selected_positions_scored
            else 0.0
        ),
        "mean_contrast_reliability": (
            contrast_reliability_sum / selected_positions_scored
            if selected_positions_scored
            else 0.0
        ),
        "mean_history_reliability_penalty": (
            reliability_penalty_sum / selected_positions_scored
            if selected_positions_scored
            else 0.0
        ),
        "mean_apc_mass": (
            apc_mass_sum / selected_positions_scored
            if selected_positions_scored
            else 0.0
        ),
        "mean_visual_relevance": (
            visual_relevance_sum / selected_positions_scored
            if selected_positions_scored
            else 0.0
        ),
        "mean_visual_gain": (
            visual_gain_sum / selected_positions_scored
            if selected_positions_scored
            else 0.0
        ),
        "g_gate_block_rate": (
            g_gate_blocked_positions / g_gate_ur_pass_positions
            if g_gate_ur_pass_positions
            else 0.0
        ),
        "history_enabled": bool(config.history_enabled),
        "history_top_v_tokens": int(config.history_top_v_tokens),
        "history_ema_decay": float(config.history_ema_decay),
        "history_penalty_scale": float(config.history_penalty_scale),
        "history_anchor_min_consistent": int(config.history_anchor_min_consistent),
        "history_anchor_forced_deferrals": history_anchor_forced_deferrals,
        "history_observations": history_observations,
        "mean_history_stability": (
            history_stability_sum / history_observations
            if history_observations
            else 1.0
        ),
        "ccaw_enabled": bool(config.ccaw_enabled),
        "ccaw_mode": str(config.ccaw_mode),
        "ccaw_pressure_scale": float(config.ccaw_pressure_scale),
        "ccaw_block_size": int(config.ccaw_block_size),
        "ccaw_search_expansions": ccaw_search_expansions,
        "ccaw_final_mask_capacity": ccaw_state.mask_capacity,
        "ccaw_min_mask_capacity": ccaw_min_capacity,
        "ccaw_max_mask_capacity_seen": ccaw_max_capacity,
        "ccaw_mean_pressure": (
            ccaw_pressure_sum / ccaw_pressure_count
            if ccaw_pressure_count
            else 0.0
        ),
        "cache_pressure_refresh_requests": 0,
    }
    report.update(adapter.cache_report())
    if config.collect_trace:
        report["trace"] = trace
    return (output, report) if config.return_report else output
