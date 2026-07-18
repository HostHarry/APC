from __future__ import annotations

import math

import torch

from decoding import (
    CCAWState,
    ContrastStats,
    FocusFrame,
    UnifiedTrajectoryBatchObservation,
    VCHDDecodeConfig,
    WindowPressure,
    compute_contrast_stats,
    compute_focus_dwell_counter,
    compute_unified_trajectory_posterior,
    compute_window_pressure,
    focus_longtail_history_upper_bound,
    history_adjusted_reliability,
    loglogistic_survival_kernel,
    observe_focus_dwell,
    observe_focus_longtail,
    observe_sparse_history,
    observe_unified_trajectory_batch,
    pressure_adaptive_commit_budget,
    scope_next_hard_block,
    sparse_distribution_from_dense,
    sparse_jsd,
    update_ccaw_state,
    update_inverse_ccaw_state,
    visual_contrast_decode,
)
from decoding.history import (  # P0 helpers used by cache-behaviour tests
    _build_focus_lookup,
    _focus_frame_lookup,
    make_focus_frame,
)
from decoding.selector import (
    MAX_WINDOW_TOP1_FALLBACK,
    THRESHOLD_COMMIT,
    select_ccaw_positions,
    select_unified_trajectory_positions,
)


def _dense_jsd(first: torch.Tensor, second: torch.Tensor) -> float:
    mixture = 0.5 * (first + second)
    first_term = torch.where(
        first > 0,
        first * (first.log() - mixture.log()),
        torch.zeros_like(first),
    ).sum()
    second_term = torch.where(
        second > 0,
        second * (second.log() - mixture.log()),
        torch.zeros_like(second),
    ).sum()
    return float((0.5 * (first_term + second_term)).item())


def _observe(dense, history=None, version=0, top_v=3):
    ids, probs, other = sparse_distribution_from_dense(
        torch.tensor(dense, dtype=torch.float32), top_v
    )
    return observe_sparse_history(
        history,
        ids,
        probs,
        other,
        context_version=version,
        ema_decay=0.7,
        top_v_tokens=top_v,
    )


def test_first_history_observation_is_stable_and_normalized():
    observation = _observe([0.7, 0.2, 0.08, 0.02])
    assert observation.updated
    assert observation.stability.item() == 1.0
    history = observation.next_history
    assert math.isclose(
        float(history.probs.sum() + history.other_prob), 1.0, abs_tol=1e-6
    )


def test_same_context_does_not_update_history():
    first = _observe([0.7, 0.2, 0.08, 0.02])
    repeated = _observe(
        [0.7, 0.2, 0.08, 0.02],
        history=first.next_history,
        version=0,
    )
    assert not repeated.updated
    assert repeated.next_history is first.next_history
    assert repeated.stability.item() == 1.0


def test_history_counts_consecutive_top1_matches():
    first = _observe([0.7, 0.2, 0.08, 0.02])
    second = _observe(
        [0.65, 0.25, 0.08, 0.02],
        history=first.next_history,
        version=1,
    )
    third = _observe(
        [0.6, 0.3, 0.08, 0.02],
        history=second.next_history,
        version=2,
    )
    changed = _observe(
        [0.2, 0.7, 0.08, 0.02],
        history=third.next_history,
        version=3,
    )
    assert first.consecutive_top1_matches == 0
    assert second.consecutive_top1_matches == 1
    assert third.consecutive_top1_matches == 2
    assert changed.consecutive_top1_matches == 0


def test_distribution_change_lowers_stability_and_reliability():
    first = _observe([0.9, 0.08, 0.01, 0.01])
    changed = _observe(
        [0.08, 0.9, 0.01, 0.01],
        history=first.next_history,
        version=1,
    )
    assert changed.updated
    assert changed.stability.item() < 0.5

    contrast = torch.tensor([0.95])
    relevance = torch.tensor([0.8])
    stable = history_adjusted_reliability(
        contrast, relevance, torch.tensor([1.0])
    )
    unstable = history_adjusted_reliability(
        contrast, relevance, changed.stability.reshape(1)
    )
    assert 0.0 <= unstable.item() < stable.item() <= contrast.item()


def test_history_penalty_scale_can_veto_an_unstable_candidate():
    contrast = torch.tensor([0.95])
    relevance = torch.tensor([0.05])
    instability = torch.tensor([0.0])
    default = history_adjusted_reliability(
        contrast,
        relevance,
        instability,
    )
    amplified = history_adjusted_reliability(
        contrast,
        relevance,
        instability,
        penalty_scale=8.0,
    )
    assert default.item() > 0.9
    assert amplified.item() < 0.9


def test_sparse_jsd_tracks_dense_jsd_for_top_heavy_distributions():
    first = torch.tensor(
        [0.52, 0.24, 0.12, 0.06, 0.02, 0.015, 0.01, 0.005, 0.005, 0.005]
    )
    second = torch.tensor(
        [0.48, 0.27, 0.13, 0.055, 0.025, 0.015, 0.01, 0.005, 0.005, 0.005]
    )
    first = first / first.sum()
    second = second / second.sum()
    first_sparse = sparse_distribution_from_dense(first, 8)
    second_sparse = sparse_distribution_from_dense(second, 8)
    approximate = sparse_jsd(*first_sparse, *second_sparse)
    exact = _dense_jsd(first, second)
    assert abs(float(approximate.item()) - exact) < 1.0e-3


def _stats(base, contrast, relevance=None, changed=None):
    base = torch.tensor(base, dtype=torch.float32)
    contrast = torch.tensor(contrast, dtype=torch.float32)
    size = base.numel()
    raw = torch.zeros(size, dtype=torch.long)
    candidate = raw.clone()
    if changed is not None:
        candidate[torch.tensor(changed, dtype=torch.bool)] = 1
    return ContrastStats(
        raw_token=raw,
        contrast_token=candidate,
        raw_confidence=base,
        base_confidence=base,
        contrast_confidence=contrast,
        apc_mass=torch.ones(size),
        visual_relevance=torch.tensor(
            relevance if relevance is not None else [0.0] * size
        ),
        absolute_visual_gain=torch.zeros(size),
        relative_visual_advantage=torch.zeros(size),
    )


def test_ccaw_expands_same_snapshot_until_qualified():
    stats = _stats(
        base=[0.05, 0.05, 0.9, 0.9, 0.9, 0.9],
        contrast=[0.2, 0.2, 0.95, 0.95, 0.95, 0.95],
    )
    config = VCHDDecodeConfig(
        tau_base=0.1,
        tau_contrast=0.9,
        mask_capacity=2,
        max_physical_span=6,
        max_commit_per_iteration=2,
        ccaw_enabled=True,
        ccaw_max_mask_capacity=6,
        ccaw_expand_step=2,
    )
    selection = select_ccaw_positions(
        stats,
        torch.ones(6, dtype=torch.bool),
        config,
        contrast_reliability=stats.contrast_confidence,
        current_mask_capacity=2,
    )
    assert selection.reason == THRESHOLD_COMMIT
    assert selection.search_expansions == 1
    assert selection.positions.tolist() == [2, 3]


def test_ccaw_expands_until_qualified_budget_is_met():
    stats = _stats(
        base=[0.9, 0.9, 0.05, 0.05, 0.9, 0.9],
        contrast=[0.95, 0.95, 0.2, 0.2, 0.95, 0.95],
    )
    config = VCHDDecodeConfig(
        tau_base=0.1,
        tau_contrast=0.9,
        mask_capacity=4,
        max_physical_span=6,
        max_commit_per_iteration=4,
        ccaw_enabled=True,
        ccaw_qualified_budget=3,
        ccaw_max_mask_capacity=6,
        ccaw_expand_step=2,
    )
    selection = select_ccaw_positions(
        stats,
        torch.ones(6, dtype=torch.bool),
        config,
        contrast_reliability=stats.contrast_confidence,
        current_mask_capacity=4,
    )
    assert selection.reason == THRESHOLD_COMMIT
    assert selection.search_expansions == 1
    assert selection.qualified_count == 4
    assert selection.positions.tolist() == [0, 1, 4, 5]


def test_ccaw_falls_back_only_after_maximum_window():
    stats = _stats(base=[0.05] * 6, contrast=[0.2] * 6)
    config = VCHDDecodeConfig(
        tau_base=0.1,
        tau_contrast=0.9,
        mask_capacity=2,
        max_physical_span=6,
        ccaw_enabled=True,
        ccaw_max_mask_capacity=6,
        ccaw_expand_step=2,
    )
    selection = select_ccaw_positions(
        stats,
        torch.ones(6, dtype=torch.bool),
        config,
        contrast_reliability=stats.contrast_confidence,
        current_mask_capacity=2,
    )
    assert selection.reason == MAX_WINDOW_TOP1_FALLBACK
    assert selection.search_expansions == 2
    assert selection.window.active_positions.numel() == 6
    assert selection.positions.numel() == 1


def test_ccaw_pressure_update_respects_capacity_step():
    stats = _stats(
        base=[0.05, 0.05, 0.9, 0.9],
        contrast=[0.2, 0.2, 0.95, 0.95],
        relevance=[1.0, 1.0, 1.0, 1.0],
        changed=[True, True, True, True],
    )
    config = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=4,
        ccaw_enabled=True,
        ccaw_max_mask_capacity=8,
        ccaw_pressure_ema_decay=0.0,
        ccaw_expand_step=2,
        ccaw_shrink_step=1,
    )
    selection = select_ccaw_positions(
        stats,
        torch.ones(4, dtype=torch.bool),
        config,
        contrast_reliability=stats.contrast_confidence,
        current_mask_capacity=2,
    )
    pressure = compute_window_pressure(
        stats,
        torch.zeros(4),
        stats.contrast_confidence,
        selection.window,
        config,
    )
    state = CCAWState(mask_capacity=2)
    update_ccaw_state(state, pressure, config)
    assert state.mask_capacity == 4


def _pressure(value: float) -> WindowPressure:
    return WindowPressure(
        candidate_conflict=value,
        history_instability=value,
        qualification_deficit=value,
        combined=value,
    )


def test_hard_block_scope_does_not_advance_until_empty():
    mask = torch.tensor(
        [False, True, False, False, True, True, False, False]
    )
    scoped, left, right = scope_next_hard_block(
        mask,
        block_start=0,
        block_size=4,
    )
    assert (left, right) == (0, 4)
    assert torch.nonzero(scoped, as_tuple=True)[0].tolist() == [1]

    mask[1] = False
    scoped, left, right = scope_next_hard_block(
        mask,
        block_start=0,
        block_size=4,
    )
    assert (left, right) == (4, 8)
    assert torch.nonzero(scoped, as_tuple=True)[0].tolist() == [4, 5]


def test_high_pressure_reduces_hard_block_commit_budget():
    config = VCHDDecodeConfig(
        mask_capacity=8,
        max_commit_per_iteration=8,
        ccaw_min_commit_per_iteration=2,
        ccaw_max_mask_capacity=8,
    )
    low = pressure_adaptive_commit_budget(_pressure(0.0), config)
    medium = pressure_adaptive_commit_budget(_pressure(0.5), config)
    high = pressure_adaptive_commit_budget(_pressure(1.0), config)
    assert low == 8
    assert low > medium > high
    assert high == 2


def test_inverse_window_capacity_is_monotonic_with_pressure():
    config = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
    )
    capacities = []
    for value in (0.0, 0.5, 1.0):
        state = CCAWState(mask_capacity=2)
        update_inverse_ccaw_state(state, _pressure(value), config)
        capacities.append(state.mask_capacity)
    assert capacities == [8, 5, 2]


def test_inverse_window_ema_filter_smooths_response():
    """filter=ema should react more gradually than filter=none."""

    config_none = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
        ccaw_pressure_filter="none",
        ccaw_pressure_ema_decay=0.8,
    )
    config_ema = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
        ccaw_pressure_filter="ema",
        ccaw_pressure_ema_decay=0.8,
    )
    state_none = CCAWState(mask_capacity=8)
    state_ema = CCAWState(mask_capacity=8)
    update_inverse_ccaw_state(state_none, _pressure(1.0), config_none)
    update_inverse_ccaw_state(state_ema, _pressure(1.0), config_ema)
    # Raw filter contracts instantly to the minimum; EMA lags behind
    # because pressure_ema has only absorbed 0.2 of the spike.
    assert state_none.mask_capacity == 2
    assert state_ema.mask_capacity > state_none.mask_capacity


def test_pressure_scale_amplifies_shrinking():
    """pressure_scale > 1.0 saturates target at a lower raw pressure."""

    config_low = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
        ccaw_pressure_scale=1.0,
    )
    config_high = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
        ccaw_pressure_scale=3.0,
    )
    state_low = CCAWState(mask_capacity=8)
    state_high = CCAWState(mask_capacity=8)
    update_inverse_ccaw_state(state_low, _pressure(0.3), config_low)
    update_inverse_ccaw_state(state_high, _pressure(0.3), config_high)
    # 3.0 * 0.3 = 0.9 clamps close to full shrink; 1.0 * 0.3 = 0.3 shrinks less.
    assert state_high.mask_capacity < state_low.mask_capacity


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _FixedModel:
    def __init__(self):
        self.config = type(
            "Config", (), {"vocab_size": 6, "llm_vocab_size": 5}
        )()

    def __call__(
        self,
        *,
        input_ids,
        attention_mask=None,
        attention_bias=None,
        use_cache=False,
    ):
        del attention_mask, attention_bias, use_cache
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 6)
        logits[:, :, 1] = 8.0
        logits[:, :, 2] = 2.0
        logits[:, :, 5] = 20.0
        return _Output(logits)


def test_hard_block_decoder_reforwards_and_never_crosses_block():
    mask_id = 4
    tokens = torch.tensor([[0, 2, 3] + [mask_id] * 8])
    config = VCHDDecodeConfig(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        force_math_sdpa=False,
        truncate_at_eos=False,
        collect_trace=True,
        return_report=True,
        ccaw_enabled=True,
        ccaw_mode="hard_block",
        ccaw_block_size=4,
        ccaw_max_mask_capacity=8,
    )
    output, report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=11,
        image_span=(1, 2),
        config=config,
    )
    assert output[0, 3:].tolist() == [1] * 8
    trace = report["trace"]
    assert [item["hard_block_left"] for item in trace] == [0, 0, 4, 4]
    assert all(
        item["hard_block_left"] <= position < item["hard_block_right"]
        for item in trace
        for position in item["selected_positions"]
    )
    assert report["model_evaluations"] == 4
    assert report["threshold_commit_events"] == 4
    assert report["ccaw_search_expansions"] == 0
    assert all(item["history_anchor_qualified"] for item in trace)


def test_inverse_window_uses_current_snapshot_pressure():
    mask_id = 4
    tokens = torch.tensor([[0, 2, 3] + [mask_id] * 4])
    config = VCHDDecodeConfig(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=2,
        max_physical_span=4,
        max_commit_per_iteration=2,
        force_math_sdpa=False,
        truncate_at_eos=False,
        collect_trace=True,
        return_report=True,
        ccaw_enabled=True,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=4,
        ccaw_expand_step=2,
    )
    _, report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=7,
        image_span=(1, 2),
        config=config,
    )
    first = report["trace"][0]
    assert first["window_pressure"]["combined"] == 0.0
    assert first["persistent_mask_capacity"] == 4
    assert first["window_mask_capacity"] == 4
    assert report["ccaw_search_expansions"] == 0


def test_history_ccaw_decoder_terminates_and_reports_state():
    mask_id = 4
    tokens = torch.tensor([[0, 2, 3, mask_id, mask_id, mask_id]])
    config = VCHDDecodeConfig(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=2,
        max_physical_span=3,
        max_commit_per_iteration=2,
        force_math_sdpa=False,
        truncate_at_eos=False,
        return_report=True,
        history_enabled=True,
        history_top_v_tokens=2,
        history_anchor_min_consistent=2,
        ccaw_enabled=True,
        ccaw_max_mask_capacity=3,
        ccaw_expand_step=1,
    )
    output, report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=config,
    )
    assert output[0, 3:].tolist() == [1, 1, 1]
    assert report["history_enabled"]
    assert report["ccaw_enabled"]
    assert report["history_anchor_min_consistent"] == 2
    assert report["history_anchor_forced_deferrals"] == 2
    assert report["history_observations"] > 0
    assert report["context_versions"] == report["model_evaluations"]


def test_trajectory_top_k_excludes_invalid_vocabulary_entries():
    valid_vocab = torch.tensor([False, True, True, True, False])
    visual = torch.tensor([[50.0, 6.0, 4.0, 3.0, 40.0]])
    ablated = torch.tensor([[40.0, 3.0, 5.0, 2.0, 30.0]])

    stats = compute_contrast_stats(
        visual,
        ablated,
        valid_vocab,
        alpha=0.5,
        beta=0.1,
        trajectory_top_k=3,
    )

    assert stats.trajectory_token_ids is not None
    assert stats.trajectory_in_apc is not None
    assert bool(valid_vocab[stats.trajectory_token_ids].all())
    assert int(stats.trajectory_token_ids[0, 0]) == int(
        stats.contrast_token[0]
    )


def test_unified_trajectory_first_snapshot_is_semantic_baseline_only():
    positions = torch.tensor([3], dtype=torch.long)
    candidates = torch.tensor([[1, 2]], dtype=torch.long)
    log_probs = torch.tensor([[-0.2, -1.6]])
    gains = torch.tensor([[0.4, -0.2]])

    baseline = observe_unified_trajectory_batch(
        {},
        positions,
        candidates,
        log_probs,
        gains,
        torch.ones_like(candidates, dtype=torch.bool),
        torch.zeros(1),
        context_version=0,
        stale_decay=0.85,
        history_limit=4,
        gain_uncertainty_scale=1.0,
    )
    assert bool(baseline.baseline_only.all())
    assert torch.equal(
        baseline.effective_exposure, torch.zeros_like(baseline.effective_exposure)
    )
    assert torch.equal(
        baseline.effective_observations,
        torch.zeros_like(baseline.effective_observations),
    )

    exposed = observe_unified_trajectory_batch(
        baseline.next_history,
        positions,
        candidates,
        log_probs,
        gains,
        torch.ones_like(candidates, dtype=torch.bool),
        torch.full((1,), 0.5),
        context_version=1,
        stale_decay=0.85,
        history_limit=4,
        gain_uncertainty_scale=1.0,
    )
    assert not bool(exposed.baseline_only.any())
    assert torch.allclose(
        exposed.effective_exposure, torch.full_like(exposed.effective_exposure, 0.5)
    )
    assert torch.allclose(
        exposed.effective_observations,
        torch.ones_like(exposed.effective_observations),
    )


def test_unified_semantic_trajectory_updates_without_visual_exposure():
    positions = torch.tensor([3], dtype=torch.long)
    candidates = torch.tensor([[1, 2]], dtype=torch.long)
    in_apc = torch.ones_like(candidates, dtype=torch.bool)
    baseline = observe_unified_trajectory_batch(
        {},
        positions,
        candidates,
        torch.tensor([[-0.1, -2.0]]),
        torch.zeros(1, 2),
        in_apc,
        torch.zeros(1),
        context_version=0,
        stale_decay=0.85,
        history_limit=4,
        gain_uncertainty_scale=1.0,
    )
    updated = observe_unified_trajectory_batch(
        baseline.next_history,
        positions,
        candidates,
        torch.tensor([[-1.0, -0.2]]),
        torch.zeros(1, 2),
        in_apc,
        torch.zeros(1),
        context_version=1,
        stale_decay=0.85,
        history_limit=4,
        gain_uncertainty_scale=1.0,
    )
    assert not torch.equal(updated.semantic_mean, baseline.semantic_mean)
    assert torch.equal(
        updated.effective_exposure,
        torch.zeros_like(updated.effective_exposure),
    )
    assert updated.updated_count == 0


def test_unified_posterior_replaces_an_opposed_apc_candidate():
    candidates = torch.tensor([[1, 2]], dtype=torch.long)
    observation = UnifiedTrajectoryBatchObservation(
        next_history={},
        semantic_mean=torch.tensor([[-0.10, -0.15]]),
        semantic_std=torch.zeros(1, 2),
        gain_mean=torch.tensor([[-0.30, 0.50]]),
        gain_lower=torch.tensor([[-0.30, 0.50]]),
        gain_upper=torch.tensor([[-0.30, 0.50]]),
        effective_exposure=torch.ones(1, 2),
        effective_observations=torch.full((1, 2), 2.0),
        candidate_age=torch.ones(1, 2, dtype=torch.long),
        baseline_only=torch.zeros(1, 2, dtype=torch.bool),
        updated_count=2,
    )
    posterior = compute_unified_trajectory_posterior(
        candidates,
        torch.tensor([[0.95, 0.60]]),
        torch.tensor([[0.8, 0.2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1),
        observation,
        semantic_std_scale=0.25,
        visual_weight=1.0,
        adaptive_visual_relevance=True,
        observation_scale=1.0,
        exposure_scale=0.5,
        relevance_scale=0.1,
        uncertainty_scale=1.0,
        opposed_threshold=0.05,
    )
    assert posterior.candidate_opposed[0, 0]
    assert int(posterior.selected_token[0]) == 2

    config = VCHDDecodeConfig(
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=1,
        max_physical_span=1,
        max_commit_per_iteration=1,
        unified_trajectory_enabled=True,
        unified_trajectory_window_size=1,
    )
    selection = select_unified_trajectory_positions(
        _stats([0.9], [0.9]),
        torch.tensor([True]),
        config,
        token_overrides=posterior.selected_token,
        trajectory_confidence=posterior.selected_confidence,
        trajectory_entropy=posterior.selected_entropy,
        trajectory_margin=posterior.selected_margin,
    )
    assert selection.reason == THRESHOLD_COMMIT
    assert selection.token_overrides is not None
    assert selection.token_overrides.tolist() == [2]


def test_unified_zero_exposure_is_a_valid_semantic_posterior():
    candidates = torch.tensor([[1, 2]], dtype=torch.long)
    contrast_probs = torch.tensor([[0.95, 0.05]])
    observation = UnifiedTrajectoryBatchObservation(
        next_history={},
        semantic_mean=contrast_probs.log(),
        semantic_std=torch.zeros(1, 2),
        gain_mean=torch.tensor([[-10.0, 10.0]]),
        gain_lower=torch.tensor([[-10.0, 10.0]]),
        gain_upper=torch.tensor([[-10.0, 10.0]]),
        effective_exposure=torch.zeros(1, 2),
        effective_observations=torch.zeros(1, 2),
        candidate_age=torch.ones(1, 2, dtype=torch.long),
        baseline_only=torch.ones(1, 2, dtype=torch.bool),
        updated_count=0,
    )
    posterior = compute_unified_trajectory_posterior(
        candidates,
        torch.tensor([[0.95, 0.04]]),
        contrast_probs,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1),
        observation,
        semantic_std_scale=0.25,
        visual_weight=1.0,
        adaptive_visual_relevance=True,
        observation_scale=2.0,
        exposure_scale=0.1,
        relevance_scale=0.01,
        uncertainty_scale=1.0,
        opposed_threshold=0.05,
    )
    assert torch.allclose(
        posterior.candidate_posterior,
        contrast_probs,
        atol=1.0e-6,
    )
    assert torch.allclose(
        posterior.selected_confidence,
        torch.tensor([0.95]),
        atol=1.0e-6,
    )
    assert torch.equal(
        posterior.candidate_visual_weight,
        torch.zeros_like(posterior.candidate_visual_weight),
    )
    assert int(posterior.selected_token[0]) == 1

    selection = select_unified_trajectory_positions(
        _stats([0.95], [0.95]),
        torch.tensor([True]),
        VCHDDecodeConfig(
            tau_base=0.9,
            tau_contrast=0.9,
            mask_capacity=1,
            max_physical_span=1,
            max_commit_per_iteration=1,
            unified_trajectory_enabled=True,
            unified_trajectory_window_size=1,
        ),
        token_overrides=posterior.selected_token,
        trajectory_confidence=posterior.selected_confidence,
        trajectory_entropy=posterior.selected_entropy,
        trajectory_margin=posterior.selected_margin,
    )
    assert selection.reason == THRESHOLD_COMMIT


def test_unified_visual_precision_increases_smoothly_with_exposure():
    candidates = torch.tensor([[1, 2]], dtype=torch.long)

    def posterior(exposure: float, observations: float):
        observation = UnifiedTrajectoryBatchObservation(
            next_history={},
            semantic_mean=torch.tensor([[-0.2, -0.3]]),
            semantic_std=torch.zeros(1, 2),
            gain_mean=torch.tensor([[0.3, -0.2]]),
            gain_lower=torch.tensor([[0.2, -0.3]]),
            gain_upper=torch.tensor([[0.4, -0.1]]),
            effective_exposure=torch.full((1, 2), exposure),
            effective_observations=torch.full((1, 2), observations),
            candidate_age=torch.ones(1, 2, dtype=torch.long),
            baseline_only=torch.zeros(1, 2, dtype=torch.bool),
            updated_count=2,
        )
        return compute_unified_trajectory_posterior(
            candidates,
            torch.tensor([[0.6, 0.4]]),
            torch.tensor([[0.6, 0.4]]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.full((1,), 0.01),
            observation,
            semantic_std_scale=0.25,
            visual_weight=0.5,
            adaptive_visual_relevance=True,
            observation_scale=2.0,
            exposure_scale=0.1,
            relevance_scale=0.01,
            uncertainty_scale=1.0,
            opposed_threshold=0.05,
        )

    low = posterior(0.05, 1.0)
    high = posterior(0.5, 3.0)
    assert bool(
        (
            high.candidate_visual_weight
            > low.candidate_visual_weight
        ).all()
    )
    assert bool((high.candidate_visual_weight <= 0.5).all())


def test_unified_fixed_visual_ablation_removes_relevance_gating():
    candidates = torch.tensor([[1, 2]], dtype=torch.long)
    observation = UnifiedTrajectoryBatchObservation(
        next_history={},
        semantic_mean=torch.tensor([[-0.10, -0.15]]),
        semantic_std=torch.zeros(1, 2),
        gain_mean=torch.tensor([[-0.30, 0.50]]),
        gain_lower=torch.tensor([[-0.30, 0.50]]),
        gain_upper=torch.tensor([[-0.30, 0.50]]),
        effective_exposure=torch.ones(1, 2),
        effective_observations=torch.full((1, 2), 2.0),
        candidate_age=torch.ones(1, 2, dtype=torch.long),
        baseline_only=torch.zeros(1, 2, dtype=torch.bool),
        updated_count=2,
    )
    common = dict(
        semantic_std_scale=0.25,
        visual_weight=1.0,
        observation_scale=1.0,
        exposure_scale=0.5,
        relevance_scale=0.1,
        uncertainty_scale=1.0,
        opposed_threshold=0.05,
    )
    adaptive = compute_unified_trajectory_posterior(
        candidates,
        torch.tensor([[0.95, 0.60]]),
        torch.tensor([[0.8, 0.2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1),
        observation,
        adaptive_visual_relevance=True,
        **common,
    )
    fixed = compute_unified_trajectory_posterior(
        candidates,
        torch.tensor([[0.95, 0.60]]),
        torch.tensor([[0.8, 0.2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1),
        observation,
        adaptive_visual_relevance=False,
        **common,
    )
    assert int(adaptive.selected_token[0]) == 1
    assert int(fixed.selected_token[0]) == 2


def test_unified_posterior_never_selects_an_out_of_apc_candidate():
    candidates = torch.tensor([[1, 2]], dtype=torch.long)
    observation = UnifiedTrajectoryBatchObservation(
        next_history={},
        semantic_mean=torch.tensor([[-2.0, -0.01]]),
        semantic_std=torch.zeros(1, 2),
        gain_mean=torch.tensor([[0.0, 2.0]]),
        gain_lower=torch.tensor([[0.0, 2.0]]),
        gain_upper=torch.tensor([[0.0, 2.0]]),
        effective_exposure=torch.ones(1, 2),
        effective_observations=torch.full((1, 2), 2.0),
        candidate_age=torch.ones(1, 2, dtype=torch.long),
        baseline_only=torch.zeros(1, 2, dtype=torch.bool),
        updated_count=2,
    )
    posterior = compute_unified_trajectory_posterior(
        candidates,
        torch.tensor([[0.20, 0.80]]),
        torch.tensor([[0.2, 0.0]]),
        torch.tensor([[True, False]]),
        torch.ones(1),
        observation,
        semantic_std_scale=0.0,
        visual_weight=1.0,
        adaptive_visual_relevance=True,
        observation_scale=1.0,
        exposure_scale=0.5,
        relevance_scale=0.1,
        uncertainty_scale=1.0,
        opposed_threshold=0.05,
    )
    assert int(posterior.selected_token[0]) == 1


def test_unified_trajectory_isolation_rejects_legacy_modules():
    config = VCHDDecodeConfig(
        unified_trajectory_enabled=True,
        ccaw_enabled=True,
    )
    try:
        config.validate()
    except ValueError as error:
        assert "unified trajectory is isolated" in str(error)
    else:
        raise AssertionError("Unified trajectory must reject legacy CCAW")


def test_unified_trajectory_decoder_terminates_without_readiness_fallback():
    mask_id = 4
    tokens = torch.tensor([[0, 2, 3, mask_id, mask_id, mask_id]])
    config = VCHDDecodeConfig(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=3,
        max_physical_span=3,
        max_commit_per_iteration=1,
        force_math_sdpa=False,
        truncate_at_eos=False,
        collect_trace=True,
        return_report=True,
        unified_trajectory_enabled=True,
        unified_trajectory_top_k=2,
        unified_trajectory_window_size=3,
        unified_trajectory_observation_scale=1.0,
        unified_trajectory_exposure_scale=0.1,
    )
    output, report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=config,
    )
    assert output[0, 3:].tolist() == [1, 1, 1]
    assert report["unified_trajectory_enabled"]
    assert not report["unified_trajectory_hard_readiness_gate"]
    assert report["unified_trajectory_dual_gate_fallbacks"] == 0
    assert report["unified_trajectory_exposure_updates"] > 0
    assert report["context_versions"] == report["model_evaluations"]


def test_focus_longtail_loglogistic_kernel_has_sharp_long_tail():
    weights = loglogistic_survival_kernel(
        torch.arange(8),
        scale=3.20,
        shape=8.0,
        offset=1.0,
    )

    assert bool((weights[:-1] > weights[1:]).all())
    assert float(weights[2] / weights[3]) > 4.0
    assert float(weights[3] / weights[4]) > 4.0
    assert float(weights[-1]) > 0.0


def test_focus_dwell_counter_reflects_contiguous_dwell():
    frames = [
        FocusFrame(
            positions=torch.tensor([0, 1, 2]),
            distributions=torch.tensor(
                [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]
            ),
        ),
        FocusFrame(
            positions=torch.tensor([1, 0]),
            distributions=torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
        ),
    ]
    current_positions = torch.tensor([0, 1])
    counter = compute_focus_dwell_counter(
        frames, current_positions, total_positions=3
    )
    assert counter.tolist() == [3, 3, 0]
    empty_counter = compute_focus_dwell_counter(
        [], current_positions, total_positions=3
    )
    assert empty_counter.tolist() == [1, 1, 0]

    broken_frames = [
        FocusFrame(
            positions=torch.tensor([0]),
            distributions=torch.tensor([[0.9, 0.1]]),
        ),
        FocusFrame(
            positions=torch.tensor([1]),
            distributions=torch.tensor([[0.4, 0.6]]),
        ),
    ]
    broken_counter = compute_focus_dwell_counter(
        broken_frames, current_positions, total_positions=3
    )
    assert broken_counter.tolist() == [1, 2, 0]


def test_focus_longtail_zero_activation_matches_focus_dwell():
    frames = [
        FocusFrame(
            positions=torch.tensor([0, 1, 2]),
            distributions=torch.tensor(
                [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]
            ),
        ),
        FocusFrame(
            positions=torch.tensor([1, 0]),
            distributions=torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
        ),
    ]
    current = torch.tensor(
        [[0.4, 0.6], [0.7, 0.3], [0.1, 0.9]]
    )
    visual = torch.tensor(
        [[0.3, 0.7], [0.8, 0.2], [0.2, 0.8]]
    )
    confidence = torch.tensor([0.9, 0.8, 0.1])
    apc_mass = torch.ones(3)
    mask = torch.ones(3, dtype=torch.bool)
    dwell = observe_focus_dwell(
        frames,
        current,
        visual,
        confidence,
        apc_mass,
        mask,
        dwell_depth=2,
        focus_capacity=2,
    )
    longtail = observe_focus_longtail(
        frames,
        current,
        visual,
        confidence,
        apc_mass,
        mask,
        torch.zeros(3),
        torch.ones(3),
        dwell_depth=2,
        focus_capacity=2,
        kernel_scale=3.20,
        kernel_shape=8.0,
        kernel_offset=1.0,
        mix_ceiling=1.0,
        exposure_tau=0.10,
        relevance_tau=0.01,
        conflict_tau=0.002,
    )
    eligible = dwell.eligible_mask

    assert torch.equal(longtail.eligible_mask, eligible)
    assert torch.equal(longtail.selected_token, dwell.marginal_token)
    assert torch.allclose(
        longtail.base_confidence, dwell.base_confidence, atol=1e-7
    )
    assert torch.allclose(
        longtail.contrast_confidence,
        dwell.contrast_confidence,
        atol=1e-7,
    )
    assert torch.allclose(
        longtail.entropy[eligible],
        dwell.marginal_entropy[eligible],
        atol=1e-7,
    )
    assert bool((longtail.tail_activation[eligible] == 0.0).all())
    assert bool((longtail.long_tail_mass[eligible] == 0.0).all())
    assert torch.allclose(
        longtail.current_weight[eligible],
        torch.full((2,), 1.0 / 3.0),
        atol=1e-7,
    )


def test_focus_longtail_zero_mix_matches_focus_dwell_decoder_behavior():
    mask_id = 4
    tokens = torch.tensor([[0, 2, 3, mask_id, mask_id, mask_id]])
    common = dict(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=3,
        max_physical_span=3,
        max_commit_per_iteration=1,
        force_math_sdpa=False,
        truncate_at_eos=False,
        collect_trace=True,
        return_report=True,
        focus_dwell_depth=2,
        focus_capacity=3,
    )
    dwell_output, dwell_report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=VCHDDecodeConfig(**common, focus_dwell_enabled=True),
    )
    longtail_output, longtail_report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=VCHDDecodeConfig(
            **common,
            focus_longtail_enabled=True,
            focus_longtail_mix_ceiling=0.0,
        ),
    )

    assert torch.equal(longtail_output, dwell_output)
    dwell_decisions = [
        (
            item["selected_positions"],
            item["selected_tokens"],
            item["commit_reason"],
        )
        for item in dwell_report["trace"]
    ]
    longtail_decisions = [
        (
            item["selected_positions"],
            item["selected_tokens"],
            item["commit_reason"],
        )
        for item in longtail_report["trace"]
    ]
    assert longtail_decisions == dwell_decisions


def test_focus_longtail_activation_adds_nonzero_old_dwell_tail():
    recent = torch.tensor([[0.2, 0.8]])
    old = torch.tensor([[1.0, 0.0]])
    frames = [
        FocusFrame(torch.tensor([0]), old),
        FocusFrame(torch.tensor([0]), old),
        FocusFrame(torch.tensor([0]), recent),
        FocusFrame(torch.tensor([0]), recent),
    ]
    kwargs = dict(
        focus_frames=frames,
        contrast_distribution=torch.tensor([[0.0, 1.0]]),
        visual_distribution=torch.tensor([[0.4, 0.6]]),
        position_confidence=torch.ones(1),
        apc_mass=torch.ones(1),
        mask=torch.ones(1, dtype=torch.bool),
        exposure=torch.full((1,), 10.0),
        visual_relevance=torch.full((1,), 10.0),
        dwell_depth=2,
        focus_capacity=1,
        kernel_scale=3.20,
        kernel_shape=8.0,
        kernel_offset=1.0,
        exposure_tau=0.10,
        relevance_tau=0.01,
        conflict_tau=0.002,
    )
    dwell_equivalent = observe_focus_longtail(
        **kwargs, mix_ceiling=0.0
    )
    long_tail = observe_focus_longtail(
        **kwargs, mix_ceiling=1.0
    )

    assert float(long_tail.tail_activation[0]) > 0.99
    assert float(long_tail.long_tail_mass[0]) > 0.05
    assert long_tail.effective_dwell_depth[0].item() == 5
    assert long_tail.selected_token[0].item() == 1
    assert (
        float(long_tail.contrast_confidence[0])
        < float(dwell_equivalent.contrast_confidence[0])
    )


def test_focus_longtail_isolation_rejects_score_fusion():
    config = VCHDDecodeConfig(
        mask_id=4,
        focus_longtail_enabled=True,
        unified_trajectory_enabled=True,
    )
    try:
        config.validate()
    except ValueError as error:
        assert "focus long-tail posterior is isolated" in str(error)
    else:
        raise AssertionError(
            "Focus long-tail posterior must reject unified score fusion"
        )


def test_focus_longtail_decoder_reports_long_tail_diagnostics():
    mask_id = 4
    tokens = torch.tensor([[0, 2, 3, mask_id, mask_id, mask_id]])
    config = VCHDDecodeConfig(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=3,
        max_physical_span=3,
        max_commit_per_iteration=1,
        force_math_sdpa=False,
        truncate_at_eos=False,
        collect_trace=True,
        return_report=True,
        focus_longtail_enabled=True,
        focus_capacity=3,
    )
    output, report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=config,
    )

    assert output[0, 3:].tolist() == [1, 1, 1]
    assert report["focus_longtail_enabled"]
    assert report["focus_longtail_full_distribution"]
    assert not report["focus_longtail_token_visual_residual"]
    assert report["focus_longtail_dwell_identity_at_zero_activation"]
    assert (
        report["focus_longtail_kernel_type"]
        == "shifted_loglogistic_survival"
    )
    assert report["focus_longtail_eligible_positions"] > 0
    assert (
        report["focus_longtail_mean_effective_dwell_depth"] >= 1.0
    )
    assert report["focus_longtail_empty_dwell_fallbacks"] == 0
    assert report["context_versions"] == report["model_evaluations"]


def test_focus_longtail_history_upper_bound_matches_survival_epsilon():
    upper = focus_longtail_history_upper_bound(
        kernel_scale=3.2,
        kernel_shape=8.0,
        kernel_offset=1.0,
        epsilon=1.0e-4,
        dwell_depth=2,
    )
    assert upper >= 2
    kept = loglogistic_survival_kernel(
        torch.tensor([float(upper)]),
        scale=3.2,
        shape=8.0,
        offset=1.0,
    )
    beyond = loglogistic_survival_kernel(
        torch.tensor([float(upper + 4)]),
        scale=3.2,
        shape=8.0,
        offset=1.0,
    )
    assert float(kept.item()) >= 1.0e-4 / 4.0
    assert float(beyond.item()) < 1.0e-4
    assert (
        focus_longtail_history_upper_bound(
            kernel_scale=3.2,
            kernel_shape=8.0,
            kernel_offset=1.0,
            epsilon=1.0e-4,
            dwell_depth=64,
        )
        == 64
    )


def test_focus_longtail_config_rejects_nonint_depth_and_capacity():
    for bad in (float("inf"), float("nan"), 2.9, 0.5):
        try:
            VCHDDecodeConfig(
                mask_id=1,
                focus_dwell_enabled=True,
                focus_dwell_depth=bad,
            ).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"validate() must reject focus_dwell_depth={bad}"
            )
        try:
            VCHDDecodeConfig(
                mask_id=1,
                focus_dwell_enabled=True,
                focus_capacity=bad,
            ).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"validate() must reject focus_capacity={bad}"
            )


def test_focus_longtail_decoder_prunes_frames_below_survival_epsilon():
    mask_id = 4
    tokens = torch.tensor(
        [[0, 2, 3] + [mask_id] * 12]
    )
    config = VCHDDecodeConfig(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=12,
        max_physical_span=12,
        max_commit_per_iteration=1,
        force_math_sdpa=False,
        truncate_at_eos=False,
        collect_trace=False,
        return_report=True,
        focus_longtail_enabled=True,
        focus_capacity=6,
        focus_dwell_depth=2,
        focus_longtail_history_epsilon=1.0e-3,
    )
    max_history = focus_longtail_history_upper_bound(
        kernel_scale=config.focus_longtail_kernel_scale,
        kernel_shape=config.focus_longtail_kernel_shape,
        kernel_offset=config.focus_longtail_kernel_offset,
        epsilon=config.focus_longtail_history_epsilon,
        dwell_depth=config.focus_dwell_depth,
    )
    _, report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=15,
        image_span=(1, 2),
        config=config,
    )
    assert report["focus_longtail_enabled"]
    assert report["focus_longtail_scored_positions"] > max_history
    assert report["focus_longtail_eligible_positions"] > 0


def test_focus_longtail_zero_mix_is_bit_exact_focus_dwell():
    torch.manual_seed(0)
    positions = 5
    vocab = 7
    contrast = torch.rand(positions, vocab)
    contrast = contrast / contrast.sum(-1, keepdim=True)
    visual = torch.rand(positions, vocab)
    visual = visual / visual.sum(-1, keepdim=True)
    confidence = torch.tensor([0.9, 0.85, 0.8, 0.75, 0.1])
    apc_mass = torch.ones(positions)
    mask = torch.ones(positions, dtype=torch.bool)
    frames = [
        FocusFrame(
            positions=torch.tensor([0, 1, 2, 3]),
            distributions=(
                torch.rand(4, vocab)
                / torch.rand(4, vocab).sum(-1, keepdim=True)
            ),
        ),
        FocusFrame(
            positions=torch.tensor([1, 0, 2, 3]),
            distributions=(
                torch.rand(4, vocab)
                / torch.rand(4, vocab).sum(-1, keepdim=True)
            ),
        ),
    ]
    dwell = observe_focus_dwell(
        frames,
        contrast,
        visual,
        confidence,
        apc_mass,
        mask,
        dwell_depth=2,
        focus_capacity=4,
    )
    longtail = observe_focus_longtail(
        frames,
        contrast,
        visual,
        confidence,
        apc_mass,
        mask,
        torch.zeros(positions),
        torch.ones(positions),
        dwell_depth=2,
        focus_capacity=4,
        kernel_scale=3.2,
        kernel_shape=8.0,
        kernel_offset=1.0,
        mix_ceiling=0.0,
        exposure_tau=0.1,
        relevance_tau=0.01,
        conflict_tau=0.002,
    )
    assert torch.equal(dwell.eligible_mask, longtail.eligible_mask)
    assert torch.equal(dwell.marginal_token, longtail.selected_token)
    assert torch.equal(dwell.base_confidence, longtail.base_confidence)
    assert torch.equal(
        dwell.contrast_confidence, longtail.contrast_confidence
    )
    eligible = dwell.eligible_mask
    assert torch.equal(
        dwell.marginal_entropy[eligible], longtail.entropy[eligible]
    )


def test_make_focus_frame_precomputes_correct_inverse_lookup():
    positions = torch.tensor([2, 5, 7], dtype=torch.long)
    distributions = torch.rand(3, 6)
    frame = make_focus_frame(positions, distributions, total_positions=8)
    expected = torch.full((8,), -1, dtype=torch.long)
    expected[positions] = torch.arange(positions.numel(), dtype=torch.long)
    assert frame.lookup is not None
    assert torch.equal(frame.lookup, expected)
    # Composition invariant: frame.positions[lookup[i]] == i for present i,
    # so the inverse mapping is exact and independent of ordering.
    present = frame.lookup >= 0
    present_indices = torch.nonzero(present, as_tuple=True)[0]
    assert torch.equal(frame.positions[frame.lookup[present]], present_indices)


def test_build_focus_lookup_rejects_multi_dim_positions():
    positions = torch.zeros((2, 2), dtype=torch.long)
    try:
        _build_focus_lookup(positions, total_positions=8)
    except ValueError as exc:
        assert "one-dimensional" in str(exc)
    else:
        raise AssertionError("expected ValueError on multi-dim positions")


def test_focus_frame_lookup_fallback_matches_cached_path():
    """A raw ``FocusFrame`` without a cached lookup must produce the same
    observation as an equivalent frame built via ``make_focus_frame``. This
    locks the ``_focus_frame_lookup`` fallback to be strictly equivalent to
    the primary cache-hit path.
    """

    torch.manual_seed(1234)
    positions_count = 5
    vocab = 7
    contrast = torch.rand(positions_count, vocab)
    contrast = contrast / contrast.sum(-1, keepdim=True)
    visual = torch.rand(positions_count, vocab)
    visual = visual / visual.sum(-1, keepdim=True)
    confidence = torch.tensor([0.9, 0.85, 0.8, 0.75, 0.1])
    apc_mass = torch.ones(positions_count)
    mask = torch.ones(positions_count, dtype=torch.bool)

    focus_positions_a = torch.tensor([0, 1, 2, 3])
    focus_positions_b = torch.tensor([1, 0, 2, 3])
    dist_a = torch.rand(4, vocab)
    dist_a = dist_a / dist_a.sum(-1, keepdim=True)
    dist_b = torch.rand(4, vocab)
    dist_b = dist_b / dist_b.sum(-1, keepdim=True)

    raw_frames = [
        FocusFrame(positions=focus_positions_a, distributions=dist_a),
        FocusFrame(positions=focus_positions_b, distributions=dist_b),
    ]
    cached_frames = [
        make_focus_frame(focus_positions_a, dist_a, positions_count),
        make_focus_frame(focus_positions_b, dist_b, positions_count),
    ]
    assert raw_frames[0].lookup is None
    assert cached_frames[0].lookup is not None

    kwargs = dict(
        contrast_distribution=contrast,
        visual_distribution=visual,
        position_confidence=confidence,
        apc_mass=apc_mass,
        mask=mask,
        exposure=torch.full((positions_count,), 0.5),
        visual_relevance=torch.full((positions_count,), 0.5),
        dwell_depth=2,
        focus_capacity=4,
        kernel_scale=3.2,
        kernel_shape=8.0,
        kernel_offset=1.0,
        mix_ceiling=0.5,
        exposure_tau=0.4,
        relevance_tau=0.2,
        conflict_tau=0.1,
    )
    raw_obs = observe_focus_longtail(raw_frames, **kwargs)
    cached_obs = observe_focus_longtail(cached_frames, **kwargs)

    assert torch.equal(raw_obs.eligible_mask, cached_obs.eligible_mask)
    assert torch.equal(raw_obs.selected_token, cached_obs.selected_token)
    assert torch.equal(raw_obs.base_confidence, cached_obs.base_confidence)
    assert torch.equal(
        raw_obs.contrast_confidence, cached_obs.contrast_confidence
    )
    eligible = raw_obs.eligible_mask
    assert torch.equal(raw_obs.entropy[eligible], cached_obs.entropy[eligible])
    assert torch.equal(raw_obs.margin[eligible], cached_obs.margin[eligible])
    assert torch.equal(
        raw_obs.long_tail_mass[eligible], cached_obs.long_tail_mass[eligible]
    )


def test_focus_frame_lookup_rejects_mismatched_position_count():
    """A cached lookup whose length no longer matches ``position_count``
    must be rejected loudly rather than silently returning garbage indices.
    """

    positions = torch.tensor([0, 2], dtype=torch.long)
    distributions = torch.rand(2, 4)
    frame = make_focus_frame(positions, distributions, total_positions=8)
    try:
        _focus_frame_lookup(frame, total_positions=4, device=torch.device("cpu"))
    except ValueError as exc:
        assert "lookup" in str(exc).lower()
    else:
        raise AssertionError(
            "expected ValueError when total_positions disagrees with cache"
        )
