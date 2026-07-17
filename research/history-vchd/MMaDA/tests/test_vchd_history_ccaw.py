from __future__ import annotations

import math

import torch

from decoding import (
    CCAWState,
    CCDHistorySnapshot,
    ContrastStats,
    UnifiedTrajectoryBatchObservation,
    VCHDDecodeConfig,
    WindowPressure,
    compute_contrast_stats,
    compute_unified_trajectory_posterior,
    compute_window_pressure,
    history_adjusted_reliability,
    loglogistic_survival_kernel,
    observe_adaptive_temporal_history,
    observe_ccd_history,
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


def test_adaptive_temporal_loglogistic_kernel_has_sharp_long_tail():
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


def test_adaptive_temporal_zero_activation_is_exactly_ccd():
    history = [
        CCDHistorySnapshot(
            positions=torch.tensor([0, 1, 2]),
            distributions=torch.tensor(
                [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]
            ),
        ),
        CCDHistorySnapshot(
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
    ccd = observe_ccd_history(
        history,
        current,
        visual,
        confidence,
        apc_mass,
        mask,
        history_length=2,
        top_v_positions=2,
    )
    adaptive = observe_adaptive_temporal_history(
        history,
        current,
        visual,
        confidence,
        apc_mass,
        mask,
        torch.zeros(3),
        torch.ones(3),
        stability_length=2,
        top_v_positions=2,
        loglogistic_scale=3.20,
        loglogistic_shape=8.0,
        loglogistic_offset=1.0,
        tail_mix_max=1.0,
        exposure_scale=0.10,
        relevance_scale=0.01,
        conflict_scale=0.002,
    )
    eligible = ccd.eligible_mask

    assert torch.equal(adaptive.eligible_mask, eligible)
    assert torch.equal(adaptive.selected_token, ccd.marginal_token)
    assert torch.allclose(
        adaptive.base_confidence, ccd.base_confidence, atol=1e-7
    )
    assert torch.allclose(
        adaptive.contrast_confidence,
        ccd.contrast_confidence,
        atol=1e-7,
    )
    assert torch.allclose(
        adaptive.entropy[eligible],
        ccd.marginal_entropy[eligible],
        atol=1e-7,
    )
    assert bool((adaptive.tail_activation[eligible] == 0.0).all())
    assert bool((adaptive.long_tail_mass[eligible] == 0.0).all())
    assert torch.allclose(
        adaptive.current_weight[eligible],
        torch.full((2,), 1.0 / 3.0),
        atol=1e-7,
    )


def test_adaptive_temporal_zero_mix_matches_ccd_decoder_behavior():
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
        ccd_history_length=2,
        ccd_top_v_positions=3,
    )
    ccd_output, ccd_report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=VCHDDecodeConfig(**common, ccd_history_enabled=True),
    )
    adaptive_output, adaptive_report = visual_contrast_decode(
        _FixedModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=VCHDDecodeConfig(
            **common,
            adaptive_temporal_enabled=True,
            adaptive_temporal_tail_mix_max=0.0,
        ),
    )

    assert torch.equal(adaptive_output, ccd_output)
    ccd_decisions = [
        (
            item["selected_positions"],
            item["selected_tokens"],
            item["commit_reason"],
        )
        for item in ccd_report["trace"]
    ]
    adaptive_decisions = [
        (
            item["selected_positions"],
            item["selected_tokens"],
            item["commit_reason"],
        )
        for item in adaptive_report["trace"]
    ]
    assert adaptive_decisions == ccd_decisions


def test_adaptive_temporal_activation_adds_nonzero_old_history_tail():
    recent = torch.tensor([[0.2, 0.8]])
    old = torch.tensor([[1.0, 0.0]])
    history = [
        CCDHistorySnapshot(torch.tensor([0]), old),
        CCDHistorySnapshot(torch.tensor([0]), old),
        CCDHistorySnapshot(torch.tensor([0]), recent),
        CCDHistorySnapshot(torch.tensor([0]), recent),
    ]
    kwargs = dict(
        history=history,
        contrast_distribution=torch.tensor([[0.0, 1.0]]),
        visual_distribution=torch.tensor([[0.4, 0.6]]),
        position_confidence=torch.ones(1),
        apc_mass=torch.ones(1),
        mask=torch.ones(1, dtype=torch.bool),
        exposure=torch.full((1,), 10.0),
        visual_relevance=torch.full((1,), 10.0),
        stability_length=2,
        top_v_positions=1,
        loglogistic_scale=3.20,
        loglogistic_shape=8.0,
        loglogistic_offset=1.0,
        exposure_scale=0.10,
        relevance_scale=0.01,
        conflict_scale=0.002,
    )
    ccd_equivalent = observe_adaptive_temporal_history(
        **kwargs, tail_mix_max=0.0
    )
    long_tail = observe_adaptive_temporal_history(
        **kwargs, tail_mix_max=1.0
    )

    assert float(long_tail.tail_activation[0]) > 0.99
    assert float(long_tail.long_tail_mass[0]) > 0.05
    assert long_tail.effective_history_depth[0].item() == 5
    assert long_tail.selected_token[0].item() == 1
    assert (
        float(long_tail.contrast_confidence[0])
        < float(ccd_equivalent.contrast_confidence[0])
    )


def test_adaptive_temporal_isolation_rejects_score_fusion():
    config = VCHDDecodeConfig(
        mask_id=4,
        adaptive_temporal_enabled=True,
        unified_trajectory_enabled=True,
    )
    try:
        config.validate()
    except ValueError as error:
        assert "adaptive temporal posterior is isolated" in str(error)
    else:
        raise AssertionError(
            "Adaptive temporal posterior must reject unified score fusion"
        )


def test_adaptive_temporal_decoder_reports_long_tail_diagnostics():
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
        adaptive_temporal_enabled=True,
        ccd_top_v_positions=3,
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
    assert report["adaptive_temporal_enabled"]
    assert report["adaptive_temporal_full_distribution"]
    assert not report["adaptive_temporal_token_visual_residual"]
    assert report["adaptive_temporal_ccd_identity_at_zero_activation"]
    assert (
        report["adaptive_temporal_kernel"]
        == "shifted_loglogistic_survival"
    )
    assert report["adaptive_temporal_eligible_positions"] > 0
    assert (
        report["adaptive_temporal_mean_effective_history_depth"] >= 1.0
    )
    assert report["adaptive_temporal_empty_intersection_fallbacks"] == 0
    assert report["context_versions"] == report["model_evaluations"]
