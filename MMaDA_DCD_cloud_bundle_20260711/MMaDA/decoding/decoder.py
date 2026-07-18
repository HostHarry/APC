from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .config import (
    EOSTokenId,
    VCHDDecodeConfig,
    normalize_eos_token_ids,
)
from .contrast import compute_contrast_stats
from .history import (
    CounterfactualEvidenceHistory,
    FocusDwellObservation,
    FocusFrame,
    FocusLongTailObservation,
    SparseHistory,
    UnifiedCandidateTrajectory,
    UnifiedTrajectoryBatchObservation,
    UnifiedTrajectoryPosterior,
    compute_unified_trajectory_posterior,
    counterfactual_exposure_weights,
    focus_longtail_history_upper_bound,
    history_adjusted_reliability,
    observe_counterfactual_evidence_batch,
    observe_focus_dwell,
    observe_focus_longtail,
    observe_sparse_history,
    observe_unified_trajectory_batch,
)
from .mmada_adapter import MMaDAVisualAccessAdapter
from .selector import (
    FOCUS_DWELL_EMPTY_FALLBACK,
    MAX_WINDOW_TOP1_FALLBACK,
    THRESHOLD_COMMIT,
    build_fixed_window,
    select_ccaw_positions,
    select_counterfactual_exposure_positions,
    select_fixed_window_positions,
    select_focus_dwell_positions,
    select_unified_trajectory_positions,
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
    focus_dwell_frames: List[FocusFrame] = []
    counterfactual_history: Dict[int, CounterfactualEvidenceHistory] = {}
    unified_trajectory_history: Dict[
        int, Dict[int, UnifiedCandidateTrajectory]
    ] = {}
    focus_longtail_frames: List[FocusFrame] = []
    focus_longtail_history_depth = (
        focus_longtail_history_upper_bound(
            kernel_scale=config.focus_longtail_kernel_scale,
            kernel_shape=config.focus_longtail_kernel_shape,
            kernel_offset=config.focus_longtail_kernel_offset,
            epsilon=config.focus_longtail_history_epsilon,
            dwell_depth=config.focus_dwell_depth,
        )
        if config.focus_longtail_enabled
        else int(config.focus_dwell_depth)
    )
    previous_counterfactual_commit_positions = torch.empty(
        0, dtype=torch.long, device=state.device
    )
    previous_counterfactual_commit_relevance = torch.empty(
        0, dtype=torch.float32, device=state.device
    )
    previous_unified_commit_positions = torch.empty(
        0, dtype=torch.long, device=state.device
    )
    previous_unified_commit_relevance = torch.empty(
        0, dtype=torch.float32, device=state.device
    )
    previous_focus_longtail_commit_positions = torch.empty(
        0, dtype=torch.long, device=state.device
    )
    previous_focus_longtail_commit_relevance = torch.empty(
        0, dtype=torch.float32, device=state.device
    )
    history_observations = 0
    history_stability_sum = 0.0
    history_anchor_forced_deferrals = 0
    focus_dwell_eligible_positions_sum = 0
    focus_dwell_observations = 0
    focus_dwell_empty_fallbacks = 0
    focus_dwell_marginal_token_changes = 0
    focus_dwell_selected_marginal_token_changes = 0
    focus_dwell_marginal_entropy_sum = 0.0
    counterfactual_scored_positions = 0
    counterfactual_observations = 0
    counterfactual_support_positions = 0
    counterfactual_neutral_positions = 0
    counterfactual_opposed_positions = 0
    counterfactual_candidate_flips = 0
    counterfactual_evidence_sum = 0.0
    counterfactual_lower_bound_sum = 0.0
    counterfactual_effective_exposure_sum = 0.0
    counterfactual_vetoed_candidates = 0
    counterfactual_fallback_events = 0
    unified_scored_positions = 0
    unified_exposure_updates = 0
    unified_visually_informed_positions = 0
    unified_visual_active_positions = 0
    unified_token_replacements = 0
    unified_committed_token_replacements = 0
    unified_effective_exposure_sum = 0.0
    unified_effective_observations_sum = 0.0
    unified_visual_weight_sum = 0.0
    unified_entropy_sum = 0.0
    unified_dual_gate_fallbacks = 0
    unified_opposed_candidates = 0
    unified_selected_opposed = 0
    unified_semantic_only_commits = 0
    focus_longtail_scored_positions = 0
    focus_longtail_eligible_positions = 0
    focus_longtail_tail_active_positions = 0
    focus_longtail_token_replacements = 0
    focus_longtail_committed_token_replacements = 0
    focus_longtail_empty_dwell_fallbacks = 0
    focus_longtail_tail_activation_sum = 0.0
    focus_longtail_exposure_sum = 0.0
    focus_longtail_relevance_precision_sum = 0.0
    focus_longtail_conflict_sum = 0.0
    focus_longtail_long_tail_mass_sum = 0.0
    focus_longtail_current_weight_sum = 0.0
    focus_longtail_effective_dwell_depth_sum = 0.0
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
            return_dense_probs=(
                config.focus_dwell_enabled
                or config.focus_longtail_enabled
            ),
            trajectory_top_k=(
                config.unified_trajectory_top_k
                if config.unified_trajectory_enabled
                else None
            ),
        )
        decision_stats = stats
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

        focus_longtail_observation: Optional[FocusLongTailObservation] = None
        focus_longtail_token_overrides = stats.contrast_token.clone()
        focus_longtail_confidence = stats.contrast_confidence.clone()
        focus_longtail_base_confidence = stats.base_confidence.clone()
        focus_longtail_entropy = torch.full_like(
            stats.contrast_confidence, torch.inf
        )
        focus_longtail_margin = torch.zeros_like(stats.contrast_confidence)
        focus_longtail_tail_activation = torch.zeros_like(
            stats.contrast_confidence
        )
        focus_longtail_exposure = torch.zeros_like(stats.contrast_confidence)
        focus_longtail_relevance_precision = torch.zeros_like(
            stats.contrast_confidence
        )
        focus_longtail_conflict = torch.zeros_like(stats.contrast_confidence)
        focus_longtail_long_tail_mass = torch.zeros_like(
            stats.contrast_confidence
        )
        focus_longtail_current_weight = torch.ones_like(
            stats.contrast_confidence
        )
        focus_longtail_effective_dwell_depth = torch.ones_like(
            stats.contrast_token, dtype=torch.long
        )
        focus_longtail_eligible_mask = torch.zeros_like(mask)
        if config.focus_longtail_enabled:
            if stats.contrast_probs is None or stats.visual_probs is None:
                raise RuntimeError(
                    "Focus long-tail decoding requires full visual and "
                    "CD-APC distributions"
                )
            all_masked_positions = torch.nonzero(mask, as_tuple=True)[0]
            masked_exposure = counterfactual_exposure_weights(
                all_masked_positions,
                previous_focus_longtail_commit_positions,
                previous_focus_longtail_commit_relevance,
                distance_scale=(
                    config.counterfactual_exposure_distance_scale
                ),
                text_exposure_floor=(
                    config.counterfactual_exposure_text_exposure_floor
                ),
            )
            full_exposure = torch.zeros_like(stats.contrast_confidence)
            full_exposure[all_masked_positions] = masked_exposure
            focus_longtail_observation = observe_focus_longtail(
                focus_longtail_frames,
                stats.contrast_probs,
                stats.visual_probs,
                stats.contrast_confidence,
                stats.apc_mass,
                mask,
                full_exposure,
                stats.visual_relevance,
                dwell_depth=config.focus_dwell_depth,
                focus_capacity=config.focus_capacity,
                kernel_scale=config.focus_longtail_kernel_scale,
                kernel_shape=config.focus_longtail_kernel_shape,
                kernel_offset=config.focus_longtail_kernel_offset,
                mix_ceiling=config.focus_longtail_mix_ceiling,
                exposure_tau=config.focus_longtail_exposure_tau,
                relevance_tau=config.focus_longtail_relevance_tau,
                conflict_tau=config.focus_longtail_conflict_tau,
            )
            focus_longtail_token_overrides = (
                focus_longtail_observation.selected_token
            )
            focus_longtail_base_confidence = (
                focus_longtail_observation.base_confidence
            )
            focus_longtail_confidence = (
                focus_longtail_observation.contrast_confidence
            )
            focus_longtail_entropy = focus_longtail_observation.entropy
            focus_longtail_margin = focus_longtail_observation.margin
            focus_longtail_tail_activation = (
                focus_longtail_observation.tail_activation
            )
            focus_longtail_exposure = focus_longtail_observation.exposure
            focus_longtail_relevance_precision = (
                focus_longtail_observation.relevance_precision
            )
            focus_longtail_conflict = focus_longtail_observation.conflict
            focus_longtail_long_tail_mass = (
                focus_longtail_observation.long_tail_mass
            )
            focus_longtail_current_weight = (
                focus_longtail_observation.current_weight
            )
            focus_longtail_effective_dwell_depth = (
                focus_longtail_observation.effective_dwell_depth
            )
            focus_longtail_eligible_mask = (
                focus_longtail_observation.eligible_mask
            )
            decision_stats = replace(
                stats,
                contrast_token=focus_longtail_token_overrides,
                base_confidence=focus_longtail_base_confidence,
                contrast_confidence=focus_longtail_confidence,
            )
            contrast_reliability = focus_longtail_confidence

        unified_observation: Optional[UnifiedTrajectoryBatchObservation] = None
        unified_posterior: Optional[UnifiedTrajectoryPosterior] = None
        unified_token_overrides = stats.contrast_token.clone()
        unified_confidence = stats.contrast_confidence.clone()
        unified_base_confidence = stats.base_confidence.clone()
        unified_entropy = torch.zeros_like(stats.contrast_confidence)
        unified_margin = torch.zeros_like(stats.contrast_confidence)
        unified_visually_informed = torch.zeros_like(mask, dtype=torch.bool)
        unified_opposed = torch.zeros_like(mask, dtype=torch.bool)
        unified_effective_exposure = torch.zeros_like(
            stats.contrast_confidence
        )
        unified_effective_observations = torch.zeros_like(
            stats.contrast_confidence
        )
        unified_visual_weight = torch.zeros_like(stats.contrast_confidence)
        unified_updates: Dict[
            int, Dict[int, UnifiedCandidateTrajectory]
        ] = {}
        if config.unified_trajectory_enabled:
            trajectory_tensors = (
                stats.trajectory_token_ids,
                stats.trajectory_visual_log_probs,
                stats.trajectory_visual_probs,
                stats.trajectory_contrast_probs,
                stats.trajectory_gain,
                stats.trajectory_in_apc,
            )
            if any(value is None for value in trajectory_tensors):
                raise RuntimeError(
                    "Unified trajectory decoding requires top-K candidate "
                    "statistics"
                )
            all_masked_positions = torch.nonzero(mask, as_tuple=True)[0]
            exposure_weights = counterfactual_exposure_weights(
                all_masked_positions,
                previous_unified_commit_positions,
                previous_unified_commit_relevance,
                distance_scale=(
                    config.counterfactual_exposure_distance_scale
                ),
                text_exposure_floor=(
                    config.counterfactual_exposure_text_exposure_floor
                ),
            )
            unified_observation = observe_unified_trajectory_batch(
                unified_trajectory_history,
                all_masked_positions,
                stats.trajectory_token_ids[all_masked_positions],
                stats.trajectory_contrast_probs[
                    all_masked_positions
                ].clamp_min(torch.finfo(torch.float32).tiny).log(),
                stats.trajectory_gain[all_masked_positions],
                stats.trajectory_in_apc[all_masked_positions],
                exposure_weights,
                context_version=context_version,
                stale_decay=config.unified_trajectory_stale_decay,
                history_limit=config.unified_trajectory_history_limit,
                gain_uncertainty_scale=(
                    config.unified_trajectory_gain_uncertainty_scale
                ),
            )
            unified_posterior = compute_unified_trajectory_posterior(
                stats.trajectory_token_ids[all_masked_positions],
                stats.trajectory_visual_probs[all_masked_positions],
                stats.trajectory_contrast_probs[all_masked_positions],
                stats.trajectory_in_apc[all_masked_positions],
                stats.visual_relevance[all_masked_positions],
                unified_observation,
                semantic_std_scale=(
                    config.unified_trajectory_semantic_std_scale
                ),
                visual_weight=config.unified_trajectory_visual_weight,
                adaptive_visual_relevance=(
                    config.unified_trajectory_adaptive_visual_relevance
                ),
                observation_scale=(
                    config.unified_trajectory_observation_scale
                ),
                exposure_scale=config.unified_trajectory_exposure_scale,
                relevance_scale=(
                    config.unified_trajectory_relevance_scale
                ),
                uncertainty_scale=(
                    config.unified_trajectory_uncertainty_scale
                ),
                opposed_threshold=(
                    config.unified_trajectory_opposed_threshold
                ),
            )
            unified_token_overrides[all_masked_positions] = (
                unified_posterior.selected_token
            )
            unified_base_confidence[all_masked_positions] = (
                unified_posterior.selected_base_confidence
            )
            unified_confidence[all_masked_positions] = (
                unified_posterior.selected_confidence
            )
            unified_entropy[all_masked_positions] = (
                unified_posterior.selected_entropy
            )
            unified_margin[all_masked_positions] = (
                unified_posterior.selected_margin
            )
            unified_visually_informed[all_masked_positions] = (
                unified_posterior.selected_visually_informed
            )
            unified_opposed[all_masked_positions] = (
                unified_posterior.selected_opposed
            )
            unified_effective_exposure[all_masked_positions] = (
                unified_posterior.selected_effective_exposure
            )
            unified_effective_observations[all_masked_positions] = (
                unified_posterior.selected_effective_observations
            )
            unified_visual_weight[all_masked_positions] = (
                unified_posterior.candidate_visual_weight.gather(
                    -1,
                    unified_posterior.selected_index.unsqueeze(-1),
                ).squeeze(-1)
            )
            unified_updates = unified_observation.next_history
            decision_stats = replace(
                stats,
                contrast_token=unified_token_overrides,
                base_confidence=unified_base_confidence,
                contrast_confidence=unified_confidence,
            )
            contrast_reliability = unified_confidence

        counterfactual_lower_bound = torch.zeros_like(
            stats.contrast_confidence
        )
        counterfactual_effective_exposure = torch.zeros_like(
            stats.contrast_confidence
        )
        counterfactual_candidate_age = torch.zeros_like(
            stats.contrast_token, dtype=torch.long
        )
        counterfactual_candidate_flipped = torch.zeros_like(
            mask, dtype=torch.bool
        )
        counterfactual_updates: Dict[int, CounterfactualEvidenceHistory] = {}
        snapshot_counterfactual_observations = 0
        if config.counterfactual_exposure_mode != "off":
            if (
                stats.counterfactual_evidence is None
                or stats.ablated_competitor_token is None
            ):
                raise RuntimeError(
                    "Counterfactual exposure requires paired evidence outputs"
                )
            all_masked_positions = torch.nonzero(mask, as_tuple=True)[0]
            if config.counterfactual_exposure_mode == "exposure":
                dynamic_exposure_weights = counterfactual_exposure_weights(
                    all_masked_positions,
                    previous_counterfactual_commit_positions,
                    previous_counterfactual_commit_relevance,
                    distance_scale=(
                        config.counterfactual_exposure_distance_scale
                    ),
                    text_exposure_floor=(
                        config.counterfactual_exposure_text_exposure_floor
                    ),
                )
            else:
                dynamic_exposure_weights = torch.ones(
                    all_masked_positions.numel(),
                    dtype=torch.float32,
                    device=state.device,
                )

            counterfactual_batch = observe_counterfactual_evidence_batch(
                (
                    {}
                    if config.counterfactual_exposure_mode == "current"
                    else counterfactual_history
                ),
                all_masked_positions,
                stats.contrast_token[all_masked_positions],
                stats.counterfactual_evidence[all_masked_positions],
                dynamic_exposure_weights,
                context_version=context_version,
                flip_decay=config.counterfactual_exposure_flip_decay,
                lower_bound_scale=(
                    config.counterfactual_exposure_lower_bound_scale
                ),
            )
            counterfactual_lower_bound[all_masked_positions] = (
                counterfactual_batch.evidence_lower_bound
            )
            counterfactual_effective_exposure[all_masked_positions] = (
                counterfactual_batch.effective_exposure
            )
            counterfactual_candidate_age[all_masked_positions] = (
                counterfactual_batch.candidate_age
            )
            counterfactual_candidate_flipped[all_masked_positions] = (
                counterfactual_batch.candidate_flipped
            )
            if config.counterfactual_exposure_mode != "current":
                counterfactual_updates = counterfactual_batch.next_history
            snapshot_counterfactual_observations = (
                counterfactual_batch.updated_count
            )

        focus_dwell_observation: Optional[FocusDwellObservation] = None
        focus_dwell_empty = False
        if config.focus_dwell_enabled:
            if stats.contrast_probs is None or stats.visual_probs is None:
                raise RuntimeError(
                    "Focus dwell requires dense visual and contrast "
                    "distributions"
                )
            focus_dwell_observation = observe_focus_dwell(
                focus_dwell_frames,
                stats.contrast_probs,
                stats.visual_probs,
                stats.contrast_confidence,
                stats.apc_mass,
                mask,
                dwell_depth=config.focus_dwell_depth,
                focus_capacity=config.focus_capacity,
            )
            decision_stats = replace(
                stats,
                contrast_token=focus_dwell_observation.marginal_token,
                base_confidence=focus_dwell_observation.base_confidence,
                contrast_confidence=focus_dwell_observation.contrast_confidence,
            )
            contrast_reliability = focus_dwell_observation.contrast_confidence
            focus_dwell_empty = not bool(
                focus_dwell_observation.eligible_mask.any()
            )

        pressure = None
        commit_budget = int(config.max_commit_per_iteration)
        if config.focus_longtail_enabled and config.ccaw_enabled:
            # Combined path: focus_longtail supplies the marginalised token
            # posterior and eligibility mask, while CCAW acts as a
            # pressure-adaptive scheduler on the commit budget and (for
            # inverse_window) the mask capacity used to evaluate pressure.
            if focus_longtail_observation is None:
                raise RuntimeError(
                    "Focus long-tail observation was not constructed"
                )
            pressure_window = build_fixed_window(
                mask,
                mask_capacity=(
                    config.ccaw_max_mask_capacity
                    if config.ccaw_mode == "inverse_window"
                    else config.ccaw_block_size
                    if config.ccaw_mode == "hard_block"
                    else ccaw_state.mask_capacity
                ),
                max_physical_span=config.max_physical_span,
            )
            pressure = compute_window_pressure(
                decision_stats,
                history_stability,
                contrast_reliability,
                pressure_window,
                config,
            )
            if config.ccaw_mode == "hard_block":
                commit_budget = pressure_adaptive_commit_budget(
                    pressure,
                    config,
                )
                ccaw_state.mask_capacity = int(config.ccaw_block_size)
            elif config.ccaw_mode == "inverse_window":
                update_inverse_ccaw_state(ccaw_state, pressure, config)
            # legacy mode: state updated later in the shared accumulator
            selection = select_focus_dwell_positions(
                decision_stats,
                mask,
                config,
                eligible_mask=focus_longtail_eligible_mask,
                marginal_entropy=focus_longtail_entropy,
                max_commit_per_iteration=commit_budget,
            )
        elif config.focus_longtail_enabled:
            if focus_longtail_observation is None:
                raise RuntimeError(
                    "Focus long-tail observation was not constructed"
                )
            selection = select_focus_dwell_positions(
                decision_stats,
                mask,
                config,
                eligible_mask=focus_longtail_eligible_mask,
                marginal_entropy=focus_longtail_entropy,
            )
        elif config.unified_trajectory_enabled:
            if unified_posterior is None:
                raise RuntimeError(
                    "Unified trajectory posterior was not constructed"
                )
            selection = select_unified_trajectory_positions(
                decision_stats,
                mask,
                config,
                token_overrides=unified_token_overrides,
                trajectory_confidence=unified_confidence,
                trajectory_entropy=unified_entropy,
                trajectory_margin=unified_margin,
            )
        elif config.counterfactual_exposure_mode != "off":
            selection = select_counterfactual_exposure_positions(
                decision_stats,
                mask,
                config,
                evidence_lower_bound=counterfactual_lower_bound,
                effective_exposure=counterfactual_effective_exposure,
            )
        elif config.focus_dwell_enabled:
            if focus_dwell_observation is None:
                raise RuntimeError(
                    "Focus dwell observation was not constructed"
                )
            selection = select_focus_dwell_positions(
                decision_stats,
                mask,
                config,
                eligible_mask=focus_dwell_observation.eligible_mask,
                marginal_entropy=focus_dwell_observation.marginal_entropy,
            )
        elif config.ccaw_enabled and config.ccaw_mode == "hard_block":
            pressure_window = build_fixed_window(
                selection_mask,
                mask_capacity=config.ccaw_block_size,
                max_physical_span=config.ccaw_block_size,
            )
            pressure = compute_window_pressure(
                decision_stats,
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
                decision_stats,
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
            decision_stats.base_confidence[history_anchor_position]
            >= float(config.tau_base)
            and contrast_reliability[history_anchor_position]
            >= float(config.tau_contrast)
        )

        use_raw = (
            selection.reason == MAX_WINDOW_TOP1_FALLBACK
            and config.fallback_to_raw
        )
        selected_tokens = (
            selection.token_overrides
            if selection.token_overrides is not None
            else (
                decision_stats.raw_token[selected]
            if use_raw
                else decision_stats.contrast_token[selected]
            )
        )
        compute_pressure = config.ccaw_enabled or (
            config.cache_type == "dual"
            and config.cache_refresh_on_pressure
        )
        if pressure is None and compute_pressure:
            pressure = compute_window_pressure(
                decision_stats,
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
            decision_stats.base_confidence[masked_positions].sum().item()
        )
        contrast_confidence_sum += float(
            decision_stats.contrast_confidence[masked_positions].sum().item()
        )
        contrast_reliability_sum += float(
            contrast_reliability[masked_positions].sum().item()
        )
        apc_mass_sum += float(stats.apc_mass[masked_positions].sum().item())
        visual_relevance_sum += float(
            stats.visual_relevance[masked_positions].sum().item()
        )
        if config.counterfactual_exposure_mode != "off":
            if stats.counterfactual_evidence is None:
                raise RuntimeError(
                    "Counterfactual evidence unexpectedly missing at commit"
                )
            evidence_lower_bound = counterfactual_lower_bound[
                masked_positions
            ]
            support_mask = evidence_lower_bound >= float(
                config.counterfactual_exposure_positive_threshold
            )
            opposed_mask = evidence_lower_bound <= -float(
                config.counterfactual_exposure_negative_threshold
            )
            neutral_mask = ~(support_mask | opposed_mask)
            counterfactual_scored_positions += int(
                masked_positions.numel()
            )
            counterfactual_observations += (
                snapshot_counterfactual_observations
            )
            counterfactual_support_positions += int(
                support_mask.sum().item()
            )
            counterfactual_neutral_positions += int(
                neutral_mask.sum().item()
            )
            counterfactual_opposed_positions += int(
                opposed_mask.sum().item()
            )
            counterfactual_candidate_flips += int(
                counterfactual_candidate_flipped[masked_positions]
                .sum()
                .item()
            )
            counterfactual_evidence_sum += float(
                stats.counterfactual_evidence[masked_positions].sum().item()
            )
            counterfactual_lower_bound_sum += float(
                evidence_lower_bound.sum().item()
            )
            counterfactual_effective_exposure_sum += float(
                counterfactual_effective_exposure[masked_positions]
                .sum()
                .item()
            )
            counterfactual_vetoed_candidates += (
                selection.evidence_veto_count
            )
            counterfactual_fallback_events += int(
                selection.evidence_state == "fallback"
            )
        if config.focus_longtail_enabled:
            if focus_longtail_observation is None:
                raise RuntimeError(
                    "Focus long-tail observation missing at commit"
                )
            focus_longtail_eligible = torch.nonzero(
                focus_longtail_eligible_mask, as_tuple=True
            )[0]
            focus_longtail_scored_positions += int(masked_positions.numel())
            focus_longtail_eligible_positions += int(
                focus_longtail_eligible.numel()
            )
            focus_longtail_exposure_sum += float(
                focus_longtail_exposure[focus_longtail_eligible].sum().item()
            )
            focus_longtail_relevance_precision_sum += float(
                focus_longtail_relevance_precision[focus_longtail_eligible]
                .sum()
                .item()
            )
            focus_longtail_conflict_sum += float(
                focus_longtail_conflict[focus_longtail_eligible].sum().item()
            )
            focus_longtail_tail_activation_sum += float(
                focus_longtail_tail_activation[focus_longtail_eligible]
                .sum()
                .item()
            )
            focus_longtail_long_tail_mass_sum += float(
                focus_longtail_long_tail_mass[focus_longtail_eligible]
                .sum()
                .item()
            )
            focus_longtail_current_weight_sum += float(
                focus_longtail_current_weight[focus_longtail_eligible]
                .sum()
                .item()
            )
            focus_longtail_effective_dwell_depth_sum += float(
                focus_longtail_effective_dwell_depth[
                    focus_longtail_eligible
                ].sum().item()
            )
            focus_longtail_tail_active_positions += int(
                (
                    focus_longtail_tail_activation[focus_longtail_eligible]
                    > 0.0
                ).sum().item()
            )
            focus_longtail_token_replacements += int(
                (
                    focus_longtail_token_overrides[focus_longtail_eligible]
                    != stats.contrast_token[focus_longtail_eligible]
                )
                .sum()
                .item()
            )
            focus_longtail_committed_token_replacements += int(
                (
                    selected_tokens != stats.contrast_token[selected]
                )
                .sum()
                .item()
            )
            focus_longtail_empty_dwell_fallbacks += int(
                selection.reason == FOCUS_DWELL_EMPTY_FALLBACK
            )
        if config.unified_trajectory_enabled:
            if unified_observation is None:
                raise RuntimeError(
                    "Unified trajectory observation missing at commit"
                )
            unified_scored_positions += int(masked_positions.numel())
            unified_exposure_updates += unified_observation.updated_count
            unified_visually_informed_positions += int(
                unified_visually_informed[masked_positions].sum().item()
            )
            unified_visual_active_positions += int(
                (unified_visual_weight[masked_positions] > 0.0).sum().item()
            )
            unified_token_replacements += int(
                (
                    unified_token_overrides[masked_positions]
                    != stats.contrast_token[masked_positions]
                ).sum().item()
            )
            unified_committed_token_replacements += int(
                (
                    selected_tokens != stats.contrast_token[selected]
                ).sum().item()
            )
            unified_effective_exposure_sum += float(
                unified_effective_exposure[masked_positions].sum().item()
            )
            unified_effective_observations_sum += float(
                unified_effective_observations[masked_positions].sum().item()
            )
            unified_visual_weight_sum += float(
                unified_visual_weight[masked_positions].sum().item()
            )
            unified_entropy_sum += float(
                unified_entropy[masked_positions].sum().item()
            )
            unified_opposed_candidates += int(
                unified_opposed[masked_positions].sum().item()
            )
            unified_selected_opposed += int(
                unified_opposed[selected].sum().item()
            )
            unified_semantic_only_commits += int(
                (~unified_visually_informed[selected]).sum().item()
            )
            unified_dual_gate_fallbacks += int(
                selection.reason == MAX_WINDOW_TOP1_FALLBACK
            )
        if focus_dwell_observation is not None:
            eligible_positions = torch.nonzero(
                focus_dwell_observation.eligible_mask, as_tuple=True
            )[0]
            focus_dwell_observations += 1
            focus_dwell_eligible_positions_sum += int(
                eligible_positions.numel()
            )
            focus_dwell_empty_fallbacks += int(
                selection.reason == FOCUS_DWELL_EMPTY_FALLBACK
            )
            focus_dwell_marginal_token_changes += int(
                (
                    focus_dwell_observation.marginal_token[eligible_positions]
                    != stats.contrast_token[eligible_positions]
                ).sum().item()
            )
            focus_dwell_selected_marginal_token_changes += int(
                (
                    focus_dwell_observation.marginal_token[selected]
                    != stats.contrast_token[selected]
                ).sum().item()
            )
            focus_dwell_marginal_entropy_sum += float(
                focus_dwell_observation.marginal_entropy[
                    eligible_positions
                ].sum().item()
            )

        # Atomic commit: gather the complete decision before modifying state.
        absolute_positions = selected + int(decode_start)
        if not bool((state[0, absolute_positions] == config.mask_id).all()):
            raise RuntimeError("Selector attempted to overwrite a committed token")
        state[0, absolute_positions] = selected_tokens.to(state.dtype)
        if config.focus_longtail_enabled:
            if focus_longtail_observation is None:
                raise RuntimeError(
                    "Focus long-tail observation missing after commit"
                )
            focus_longtail_frames.append(
                focus_longtail_observation.current_frame
            )
            focus_longtail_frames = focus_longtail_frames[
                -focus_longtail_history_depth:
            ]

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
            focus_dwell_selected_entropy = []
            if focus_dwell_observation is not None:
                for position in selected.detach().cpu().tolist():
                    focus_dwell_selected_entropy.append(
                        float(
                            focus_dwell_observation.marginal_entropy[
                                position
                            ].item()
                        )
                        if bool(
                            focus_dwell_observation.eligible_mask[
                                position
                            ].item()
                        )
                        else None
                    )
            counterfactual_selected_evidence = []
            counterfactual_selected_lower_bound = []
            counterfactual_selected_exposure = []
            counterfactual_selected_age = []
            counterfactual_selected_flipped = []
            counterfactual_selected_states = []
            counterfactual_selected_competitors = []
            counterfactual_support_count = 0
            counterfactual_neutral_count = 0
            counterfactual_opposed_count = 0
            if config.counterfactual_exposure_mode != "off":
                if (
                    stats.counterfactual_evidence is None
                    or stats.ablated_competitor_token is None
                ):
                    raise RuntimeError(
                        "Counterfactual evidence missing from trace snapshot"
                    )
                active_lower_bound = counterfactual_lower_bound[
                    masked_positions
                ]
                support_mask = active_lower_bound >= float(
                    config.counterfactual_exposure_positive_threshold
                )
                opposed_mask = active_lower_bound <= -float(
                    config.counterfactual_exposure_negative_threshold
                )
                counterfactual_support_count = int(
                    support_mask.sum().item()
                )
                counterfactual_opposed_count = int(
                    opposed_mask.sum().item()
                )
                counterfactual_neutral_count = int(
                    (~(support_mask | opposed_mask)).sum().item()
                )
                for position in selected.detach().cpu().tolist():
                    lower_bound = float(
                        counterfactual_lower_bound[position].item()
                    )
                    if lower_bound >= float(
                        config.counterfactual_exposure_positive_threshold
                    ):
                        state_name = "support"
                    elif lower_bound <= -float(
                        config.counterfactual_exposure_negative_threshold
                    ):
                        state_name = "opposed"
                    else:
                        state_name = "neutral"
                    counterfactual_selected_evidence.append(
                        float(stats.counterfactual_evidence[position].item())
                    )
                    counterfactual_selected_lower_bound.append(lower_bound)
                    counterfactual_selected_exposure.append(
                        float(
                            counterfactual_effective_exposure[
                                position
                            ].item()
                        )
                    )
                    counterfactual_selected_age.append(
                        int(counterfactual_candidate_age[position].item())
                    )
                    counterfactual_selected_flipped.append(
                        bool(counterfactual_candidate_flipped[position].item())
                    )
                    counterfactual_selected_states.append(state_name)
                    counterfactual_selected_competitors.append(
                        int(
                            stats.ablated_competitor_token[position].item()
                        )
                    )
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
                    "counterfactual_exposure_mode": (
                        config.counterfactual_exposure_mode
                    ),
                    "counterfactual_support_count": (
                        counterfactual_support_count
                    ),
                    "counterfactual_neutral_count": (
                        counterfactual_neutral_count
                    ),
                    "counterfactual_opposed_count": (
                        counterfactual_opposed_count
                    ),
                    "counterfactual_evidence_veto_count": (
                        selection.evidence_veto_count
                    ),
                    "counterfactual_selection_state": (
                        selection.evidence_state
                    ),
                    "focus_longtail_enabled": (
                        config.focus_longtail_enabled
                    ),
                    "focus_longtail_dwell_depth": (
                        focus_longtail_observation.dwell_depth
                        if focus_longtail_observation is not None
                        else 0
                    ),
                    "focus_longtail_current_capacity_count": (
                        int(
                            focus_longtail_observation.current_frame.positions.numel()
                        )
                        if focus_longtail_observation is not None
                        else 0
                    ),
                    "focus_longtail_eligible_count": (
                        int(focus_longtail_eligible_mask.sum().item())
                    ),
                    "focus_longtail_selected_tail_activation": (
                        focus_longtail_tail_activation[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "focus_longtail_selected_exposure": (
                        focus_longtail_exposure[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "focus_longtail_selected_relevance_precision": (
                        focus_longtail_relevance_precision[selected]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "focus_longtail_selected_conflict": (
                        focus_longtail_conflict[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "focus_longtail_selected_long_tail_mass": (
                        focus_longtail_long_tail_mass[selected]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "focus_longtail_selected_current_weight": (
                        focus_longtail_current_weight[selected]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "focus_longtail_selected_effective_dwell_depth": (
                        focus_longtail_effective_dwell_depth[selected]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "focus_longtail_selected_entropy": [
                        v if math.isfinite(v) else None
                        for v in focus_longtail_entropy[selected]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "focus_longtail_selected_margin": (
                        focus_longtail_margin[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "focus_longtail_selected_token_changed": (
                        selected_tokens != stats.contrast_token[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "focus_longtail_anchor_current_token": int(
                        stats.contrast_token[history_anchor_position].item()
                    ),
                    "focus_longtail_anchor_marginal_token": int(
                        focus_longtail_token_overrides[
                            history_anchor_position
                        ].item()
                    ),
                    "focus_longtail_anchor_token_changed": bool(
                        focus_longtail_token_overrides[history_anchor_position]
                        != stats.contrast_token[history_anchor_position]
                    ),
                    "focus_longtail_anchor_eligible": bool(
                        focus_longtail_eligible_mask[
                            history_anchor_position
                        ].item()
                    ),
                    "focus_longtail_anchor_tail_activation": float(
                        focus_longtail_tail_activation[
                            history_anchor_position
                        ].item()
                    ),
                    "focus_longtail_anchor_exposure": float(
                        focus_longtail_exposure[history_anchor_position].item()
                    ),
                    "focus_longtail_anchor_relevance_precision": float(
                        focus_longtail_relevance_precision[
                            history_anchor_position
                        ].item()
                    ),
                    "focus_longtail_anchor_conflict": float(
                        focus_longtail_conflict[history_anchor_position].item()
                    ),
                    "focus_longtail_anchor_long_tail_mass": float(
                        focus_longtail_long_tail_mass[
                            history_anchor_position
                        ].item()
                    ),
                    "focus_longtail_anchor_current_weight": float(
                        focus_longtail_current_weight[
                            history_anchor_position
                        ].item()
                    ),
                    "focus_longtail_anchor_margin": float(
                        focus_longtail_margin[history_anchor_position].item()
                    ),
                    "focus_longtail_anchor_confidence": float(
                        focus_longtail_confidence[
                            history_anchor_position
                        ].item()
                    ),
                    "unified_trajectory_enabled": (
                        config.unified_trajectory_enabled
                    ),
                    "unified_selected_visually_informed": (
                        unified_visually_informed[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "unified_selected_opposed": unified_opposed[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "unified_selected_effective_exposure": (
                        unified_effective_exposure[selected]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "unified_selected_effective_observations": (
                        unified_effective_observations[selected]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "unified_selected_visual_weight": (
                        unified_visual_weight[selected]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "unified_selected_entropy": unified_entropy[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "unified_selected_margin": unified_margin[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "unified_selected_token_changed": (
                        selected_tokens != stats.contrast_token[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "focus_dwell_depth": (
                        focus_dwell_observation.dwell_depth
                        if focus_dwell_observation is not None
                        else 0
                    ),
                    "focus_dwell_current_capacity_count": (
                        int(
                            focus_dwell_observation.current_frame.positions.numel()
                        )
                        if focus_dwell_observation is not None
                        else 0
                    ),
                    "focus_dwell_eligible_count": (
                        int(focus_dwell_observation.eligible_mask.sum().item())
                        if focus_dwell_observation is not None
                        else 0
                    ),
                    "focus_dwell_empty_fallback": (
                        focus_dwell_empty
                    ),
                    "commit_budget": commit_budget,
                    "persistent_mask_capacity": ccaw_state.mask_capacity,
                    "search_expansions": selection.search_expansions,
                    "selected_positions": selected.detach().cpu().tolist(),
                    "selected_tokens": selected_tokens.detach().cpu().tolist(),
                    "commit_reason": selection.reason,
                    "base_confidence": decision_stats.base_confidence[selected]
                    .detach()
                    .cpu()
                    .tolist(),
                    "contrast_confidence": decision_stats.contrast_confidence[
                        selected
                    ]
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
                        decision_stats.raw_token[selected]
                        != decision_stats.contrast_token[selected]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "focus_dwell_marginal_token_changed": (
                        (
                            focus_dwell_observation.marginal_token[selected]
                            != stats.contrast_token[selected]
                        )
                        .detach()
                        .cpu()
                        .tolist()
                        if focus_dwell_observation is not None
                        else []
                    ),
                    "focus_dwell_marginal_entropy": (
                        focus_dwell_selected_entropy
                    ),
                    "counterfactual_selected_evidence": (
                        counterfactual_selected_evidence
                    ),
                    "counterfactual_selected_lower_bound": (
                        counterfactual_selected_lower_bound
                    ),
                    "counterfactual_selected_effective_exposure": (
                        counterfactual_selected_exposure
                    ),
                    "counterfactual_selected_candidate_age": (
                        counterfactual_selected_age
                    ),
                    "counterfactual_selected_candidate_flipped": (
                        counterfactual_selected_flipped
                    ),
                    "counterfactual_selected_states": (
                        counterfactual_selected_states
                    ),
                    "counterfactual_selected_ablated_competitors": (
                        counterfactual_selected_competitors
                    ),
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
        if config.focus_longtail_enabled:
            previous_focus_longtail_commit_positions = (
                selected.detach().clone()
            )
            previous_focus_longtail_commit_relevance = (
                stats.visual_relevance[selected].detach().float().clone()
            )
        if config.unified_trajectory_enabled:
            unified_trajectory_history = {
                position: update
                for position, update in unified_updates.items()
                if position not in selected_set
            }
            previous_unified_commit_positions = selected.detach().clone()
            previous_unified_commit_relevance = (
                stats.visual_relevance[selected].detach().float().clone()
            )
        if config.counterfactual_exposure_mode != "off":
            if config.counterfactual_exposure_mode != "current":
                counterfactual_history = {
                    position: update
                    for position, update in counterfactual_updates.items()
                    if position not in selected_set
                }
            previous_counterfactual_commit_positions = (
                selected.detach().clone()
            )
            previous_counterfactual_commit_relevance = (
                stats.visual_relevance[selected].detach().float().clone()
            )
        if focus_dwell_observation is not None:
            focus_dwell_frames.append(focus_dwell_observation.current_frame)
            focus_dwell_frames = focus_dwell_frames[
                -int(config.focus_dwell_depth) :
            ]
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
        "focus_dwell_enabled": bool(config.focus_dwell_enabled),
        "focus_dwell_depth": int(config.focus_dwell_depth),
        "focus_capacity": int(config.focus_capacity),
        "focus_dwell_observations": focus_dwell_observations,
        "focus_dwell_empty_fallbacks": (
            focus_dwell_empty_fallbacks
        ),
        "focus_dwell_mean_eligible_positions": (
            focus_dwell_eligible_positions_sum / focus_dwell_observations
            if focus_dwell_observations
            else 0.0
        ),
        "focus_dwell_mean_marginal_entropy": (
            focus_dwell_marginal_entropy_sum
            / focus_dwell_eligible_positions_sum
            if focus_dwell_eligible_positions_sum
            else 0.0
        ),
        "focus_dwell_marginal_token_changes": (
            focus_dwell_marginal_token_changes
        ),
        "focus_dwell_marginal_token_change_rate": (
            focus_dwell_marginal_token_changes
            / focus_dwell_eligible_positions_sum
            if focus_dwell_eligible_positions_sum
            else 0.0
        ),
        "focus_dwell_selected_marginal_token_changes": (
            focus_dwell_selected_marginal_token_changes
        ),
        "focus_longtail_enabled": bool(
            config.focus_longtail_enabled
        ),
        "focus_longtail_kernel_type": "shifted_loglogistic_survival",
        "focus_longtail_dwell_depth": int(
            config.focus_dwell_depth
        ),
        "focus_longtail_capacity": int(
            config.focus_capacity
        ),
        "focus_longtail_kernel_scale": float(
            config.focus_longtail_kernel_scale
        ),
        "focus_longtail_kernel_shape": float(
            config.focus_longtail_kernel_shape
        ),
        "focus_longtail_kernel_offset": float(
            config.focus_longtail_kernel_offset
        ),
        "focus_longtail_mix_ceiling": float(
            config.focus_longtail_mix_ceiling
        ),
        "focus_longtail_exposure_tau": float(
            config.focus_longtail_exposure_tau
        ),
        "focus_longtail_relevance_tau": float(
            config.focus_longtail_relevance_tau
        ),
        "focus_longtail_conflict_tau": float(
            config.focus_longtail_conflict_tau
        ),
        "focus_longtail_full_distribution": True,
        "focus_longtail_token_visual_residual": False,
        "focus_longtail_dwell_identity_at_zero_activation": True,
        "focus_longtail_fallback_policy": "focus_dwell",
        "focus_longtail_scored_positions": focus_longtail_scored_positions,
        "focus_longtail_eligible_positions": (
            focus_longtail_eligible_positions
        ),
        "focus_longtail_eligible_rate": (
            focus_longtail_eligible_positions
            / focus_longtail_scored_positions
            if focus_longtail_scored_positions
            else 0.0
        ),
        "focus_longtail_tail_activation_rate": (
            focus_longtail_tail_active_positions
            / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_mean_tail_activation": (
            focus_longtail_tail_activation_sum
            / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_mean_exposure": (
            focus_longtail_exposure_sum / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_mean_relevance_precision": (
            focus_longtail_relevance_precision_sum
            / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_mean_conflict": (
            focus_longtail_conflict_sum / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_mean_long_tail_mass": (
            focus_longtail_long_tail_mass_sum
            / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_mean_current_weight": (
            focus_longtail_current_weight_sum
            / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_mean_effective_dwell_depth": (
            focus_longtail_effective_dwell_depth_sum
            / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_token_replacements": (
            focus_longtail_token_replacements
        ),
        "focus_longtail_token_replacement_rate": (
            focus_longtail_token_replacements
            / focus_longtail_eligible_positions
            if focus_longtail_eligible_positions
            else 0.0
        ),
        "focus_longtail_committed_token_replacements": (
            focus_longtail_committed_token_replacements
        ),
        "focus_longtail_empty_dwell_fallbacks": (
            focus_longtail_empty_dwell_fallbacks
        ),
        "unified_trajectory_enabled": bool(
            config.unified_trajectory_enabled
        ),
        "unified_trajectory_top_k": int(config.unified_trajectory_top_k),
        "unified_trajectory_window_size": int(
            config.unified_trajectory_window_size
        ),
        "unified_trajectory_semantic_std_scale": float(
            config.unified_trajectory_semantic_std_scale
        ),
        "unified_trajectory_gain_uncertainty_scale": float(
            config.unified_trajectory_gain_uncertainty_scale
        ),
        "unified_trajectory_visual_weight": float(
            config.unified_trajectory_visual_weight
        ),
        "unified_trajectory_adaptive_visual_relevance": bool(
            config.unified_trajectory_adaptive_visual_relevance
        ),
        "unified_trajectory_relevance_scale": float(
            config.unified_trajectory_relevance_scale
        ),
        "unified_trajectory_observation_scale": float(
            config.unified_trajectory_observation_scale
        ),
        "unified_trajectory_exposure_scale": float(
            config.unified_trajectory_exposure_scale
        ),
        "unified_trajectory_uncertainty_scale": float(
            config.unified_trajectory_uncertainty_scale
        ),
        "unified_trajectory_stale_decay": float(
            config.unified_trajectory_stale_decay
        ),
        "unified_trajectory_history_limit": int(
            config.unified_trajectory_history_limit
        ),
        "unified_trajectory_opposed_threshold": float(
            config.unified_trajectory_opposed_threshold
        ),
        "unified_trajectory_scored_positions": unified_scored_positions,
        "unified_trajectory_exposure_updates": unified_exposure_updates,
        "unified_trajectory_visually_informed_rate": (
            unified_visually_informed_positions / unified_scored_positions
            if unified_scored_positions
            else 0.0
        ),
        "unified_trajectory_visual_activation_rate": (
            unified_visual_active_positions / unified_scored_positions
            if unified_scored_positions
            else 0.0
        ),
        "unified_trajectory_token_replacements": (
            unified_token_replacements
        ),
        "unified_trajectory_token_replacement_rate": (
            unified_token_replacements / unified_scored_positions
            if unified_scored_positions
            else 0.0
        ),
        "unified_trajectory_committed_token_replacements": (
            unified_committed_token_replacements
        ),
        "unified_trajectory_mean_effective_exposure": (
            unified_effective_exposure_sum / unified_scored_positions
            if unified_scored_positions
            else 0.0
        ),
        "unified_trajectory_mean_effective_observations": (
            unified_effective_observations_sum / unified_scored_positions
            if unified_scored_positions
            else 0.0
        ),
        "unified_trajectory_mean_visual_weight": (
            unified_visual_weight_sum / unified_scored_positions
            if unified_scored_positions
            else 0.0
        ),
        "unified_trajectory_mean_entropy": (
            unified_entropy_sum / unified_scored_positions
            if unified_scored_positions
            else 0.0
        ),
        "unified_trajectory_hard_readiness_gate": False,
        "unified_trajectory_dual_gate_fallbacks": (
            unified_dual_gate_fallbacks
        ),
        "unified_trajectory_opposed_candidates": (
            unified_opposed_candidates
        ),
        "unified_trajectory_selected_opposed": unified_selected_opposed,
        "unified_trajectory_semantic_only_commits": (
            unified_semantic_only_commits
        ),
        "counterfactual_exposure_mode": (
            config.counterfactual_exposure_mode
        ),
        "counterfactual_exposure_window_size": int(
            config.counterfactual_exposure_window_size
        ),
        "counterfactual_exposure_distance_scale": float(
            config.counterfactual_exposure_distance_scale
        ),
        "counterfactual_exposure_text_exposure_floor": float(
            config.counterfactual_exposure_text_exposure_floor
        ),
        "counterfactual_exposure_positive_threshold": float(
            config.counterfactual_exposure_positive_threshold
        ),
        "counterfactual_exposure_negative_threshold": float(
            config.counterfactual_exposure_negative_threshold
        ),
        "counterfactual_exposure_min_effective_exposure": float(
            config.counterfactual_exposure_min_effective_exposure
        ),
        "counterfactual_exposure_neutral_tau_contrast": float(
            config.counterfactual_exposure_neutral_tau_contrast
        ),
        "counterfactual_exposure_lower_bound_scale": float(
            config.counterfactual_exposure_lower_bound_scale
        ),
        "counterfactual_exposure_flip_decay": float(
            config.counterfactual_exposure_flip_decay
        ),
        "counterfactual_scored_positions": counterfactual_scored_positions,
        "counterfactual_observations": counterfactual_observations,
        "counterfactual_support_positions": (
            counterfactual_support_positions
        ),
        "counterfactual_neutral_positions": (
            counterfactual_neutral_positions
        ),
        "counterfactual_opposed_positions": (
            counterfactual_opposed_positions
        ),
        "counterfactual_support_rate": (
            counterfactual_support_positions
            / counterfactual_scored_positions
            if counterfactual_scored_positions
            else 0.0
        ),
        "counterfactual_neutral_rate": (
            counterfactual_neutral_positions
            / counterfactual_scored_positions
            if counterfactual_scored_positions
            else 0.0
        ),
        "counterfactual_opposed_rate": (
            counterfactual_opposed_positions
            / counterfactual_scored_positions
            if counterfactual_scored_positions
            else 0.0
        ),
        "counterfactual_candidate_flips": counterfactual_candidate_flips,
        "counterfactual_candidate_flip_rate": (
            counterfactual_candidate_flips
            / counterfactual_scored_positions
            if counterfactual_scored_positions
            else 0.0
        ),
        "counterfactual_mean_current_evidence": (
            counterfactual_evidence_sum / counterfactual_scored_positions
            if counterfactual_scored_positions
            else 0.0
        ),
        "counterfactual_mean_evidence_lower_bound": (
            counterfactual_lower_bound_sum / counterfactual_scored_positions
            if counterfactual_scored_positions
            else 0.0
        ),
        "counterfactual_mean_effective_exposure": (
            counterfactual_effective_exposure_sum
            / counterfactual_scored_positions
            if counterfactual_scored_positions
            else 0.0
        ),
        "counterfactual_vetoed_candidates": (
            counterfactual_vetoed_candidates
        ),
        "counterfactual_veto_rate": (
            counterfactual_vetoed_candidates
            / counterfactual_scored_positions
            if counterfactual_scored_positions
            else 0.0
        ),
        "counterfactual_fallback_events": counterfactual_fallback_events,
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
