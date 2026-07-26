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
