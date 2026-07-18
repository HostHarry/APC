from __future__ import annotations

import math

import torch

from decoding import (
    CCAWState,
    ContrastStats,
    VCHDDecodeConfig,
    WindowPressure,
    compute_window_pressure,
    history_adjusted_reliability,
    observe_sparse_history,
    pressure_adaptive_commit_budget,
    scope_next_hard_block,
    sparse_distribution_from_dense,
    sparse_jsd,
    update_ccaw_state,
    update_inverse_ccaw_state,
    visual_contrast_decode,
)
from decoding.selector import (
    MAX_WINDOW_TOP1_FALLBACK,
    THRESHOLD_COMMIT,
    select_ccaw_positions,
    select_fixed_window_positions,
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


def test_fallback_scope_does_not_limit_qualified_search():
    base = [0.05] * 64
    contrast = [0.2] * 64
    base[50] = 0.2
    contrast[50] = 0.95
    stats = _stats(base=base, contrast=contrast)
    config = VCHDDecodeConfig(
        tau_base=0.1,
        tau_contrast=0.9,
        mask_capacity=64,
        max_physical_span=64,
        fallback_mask_capacity=16,
    )
    selection = select_fixed_window_positions(
        stats, torch.ones(64, dtype=torch.bool), config
    )
    assert selection.reason == THRESHOLD_COMMIT
    assert selection.positions.tolist() == [50]


def test_fallback_scope_restricts_readiness_search_to_local_masks():
    base = [0.05] * 64
    contrast = [0.2] * 64
    base[7] = 0.09
    contrast[7] = 0.8
    base[50] = 0.099
    contrast[50] = 0.89
    stats = _stats(base=base, contrast=contrast)
    mask = torch.ones(64, dtype=torch.bool)

    unrestricted = VCHDDecodeConfig(
        tau_base=0.1,
        tau_contrast=0.9,
        mask_capacity=64,
        max_physical_span=64,
    )
    local = VCHDDecodeConfig(
        tau_base=0.1,
        tau_contrast=0.9,
        mask_capacity=64,
        max_physical_span=64,
        fallback_mask_capacity=16,
    )
    unrestricted_selection = select_fixed_window_positions(
        stats, mask, unrestricted
    )
    local_selection = select_fixed_window_positions(stats, mask, local)
    assert unrestricted_selection.reason == MAX_WINDOW_TOP1_FALLBACK
    assert unrestricted_selection.positions.tolist() == [50]
    assert local_selection.reason == MAX_WINDOW_TOP1_FALLBACK
    assert local_selection.positions.tolist() == [7]


def test_leftmost_fallback_policy_ignores_local_readiness_order():
    base = [0.05] * 32
    contrast = [0.2] * 32
    base[7] = 0.09
    contrast[7] = 0.8
    stats = _stats(base=base, contrast=contrast)
    config = VCHDDecodeConfig(
        tau_base=0.1,
        tau_contrast=0.9,
        mask_capacity=32,
        max_physical_span=32,
        fallback_mask_capacity=16,
        fallback_policy="leftmost",
    )
    selection = select_fixed_window_positions(
        stats, torch.ones(32, dtype=torch.bool), config
    )
    assert selection.reason == MAX_WINDOW_TOP1_FALLBACK
    assert selection.positions.tolist() == [0]


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


def test_inverse_window_pressure_filter_none_uses_raw_pressure():
    """filter='none' keeps the historical StrongShrink target formula.

    The raw pressure drives the target immediately, so a lone high-pressure
    step contracts as far as ``ccaw_shrink_step`` allows in that same step.
    """

    config = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
        ccaw_pressure_ema_decay=0.8,
        ccaw_pressure_filter="none",
    )
    state = CCAWState(mask_capacity=8)
    update_inverse_ccaw_state(state, _pressure(1.0), config)
    assert state.mask_capacity == 2
    # EMA is still maintained for observability even when unused.
    assert math.isclose(state.pressure_ema, 0.2, abs_tol=1e-6)


def test_inverse_window_pressure_filter_ema_smooths_target():
    """filter='ema' smooths pressure so the first high-pressure step is muted.

    With decay=0.8 the smoothed value after one step is 0.2 * pressure, so
    the target only pulls the capacity part-way toward the minimum, and a
    subsequent zero-pressure step lets it start climbing back.
    """

    config = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
        ccaw_pressure_ema_decay=0.8,
        ccaw_pressure_filter="ema",
    )
    state = CCAWState(mask_capacity=8)
    # Step 1: ema = 0.2, target = 8 - round(6 * 0.2) = 7.
    update_inverse_ccaw_state(state, _pressure(1.0), config)
    assert state.mask_capacity == 7
    assert math.isclose(state.pressure_ema, 0.2, abs_tol=1e-6)
    # Step 2: pressure drops to 0, ema decays to 0.16, target = 8 - 1 = 7.
    update_inverse_ccaw_state(state, _pressure(0.0), config)
    assert state.mask_capacity == 7
    assert math.isclose(state.pressure_ema, 0.16, abs_tol=1e-6)


def test_pressure_filter_validation_rejects_unknown_value():
    import pytest

    with pytest.raises(ValueError, match="ccaw_pressure_filter"):
        VCHDDecodeConfig(ccaw_pressure_filter="bogus").validate()


def test_inverse_window_pressure_scale_amplifies_shrinkage():
    baseline = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
        ccaw_pressure_scale=1.0,
    )
    amplified = VCHDDecodeConfig(
        mask_capacity=2,
        max_physical_span=8,
        max_commit_per_iteration=2,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=8,
        ccaw_expand_step=8,
        ccaw_shrink_step=8,
        ccaw_pressure_scale=3.0,
    )
    baseline_state = CCAWState(mask_capacity=2)
    amplified_state = CCAWState(mask_capacity=2)
    update_inverse_ccaw_state(baseline_state, _pressure(0.25), baseline)
    update_inverse_ccaw_state(amplified_state, _pressure(0.25), amplified)
    assert baseline_state.mask_capacity == 6
    assert amplified_state.mask_capacity == 4


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


def test_report_splits_pressure_by_qualified_vs_fallback():
    """New report strata separate qualified commits from fallback commits.

    On this configuration the model is confident enough to always qualify,
    so the fallback split is empty and the qualified split matches the
    total means.
    """

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
        ccaw_enabled=True,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=3,
        ccaw_expand_step=1,
    )
    _, report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=config,
    )
    qualified = report["ccaw_qualified_commit_count"]
    fallback = report["ccaw_fallback_commit_count"]
    assert qualified > 0
    assert fallback == 0
    assert qualified + fallback == report["context_versions"]
    assert report["ccaw_mean_pressure_fallback"] == 0.0
    assert report["ccaw_mean_qualification_deficit_fallback"] == 0.0
    assert math.isclose(
        report["ccaw_mean_pressure_qualified"],
        report["ccaw_mean_pressure"],
        abs_tol=1e-6,
    )
    assert math.isclose(
        report["ccaw_mean_qualification_deficit_qualified"],
        report["ccaw_mean_qualification_deficit"],
        abs_tol=1e-6,
    )
    assert report["ccaw_pressure_filter"] == "none"


def test_report_fallback_split_captures_fallback_only_run():
    """tau_contrast=1.0 forces every commit onto the fallback path."""

    mask_id = 4
    tokens = torch.tensor([[0, 2, 3, mask_id, mask_id, mask_id]])
    config = VCHDDecodeConfig(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        tau_base=0.0,
        tau_contrast=1.0,
        mask_capacity=2,
        max_physical_span=3,
        max_commit_per_iteration=2,
        force_math_sdpa=False,
        truncate_at_eos=False,
        return_report=True,
        ccaw_enabled=True,
        ccaw_mode="inverse_window",
        ccaw_max_mask_capacity=3,
        ccaw_expand_step=1,
    )
    _, report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=config,
    )
    assert report["ccaw_qualified_commit_count"] == 0
    assert report["ccaw_fallback_commit_count"] > 0
    assert report["ccaw_mean_pressure_qualified"] == 0.0
    assert report["ccaw_mean_qualification_deficit_qualified"] == 0.0
    assert math.isclose(
        report["ccaw_mean_pressure_fallback"],
        report["ccaw_mean_pressure"],
        abs_tol=1e-6,
    )


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
