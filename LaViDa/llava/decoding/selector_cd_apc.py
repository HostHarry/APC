"""CD-APC triple-gate position selection.

Uses the same left-anchored window as ``selector.select_fixed_window_positions``,
but eligibility is (u, r, g) with MC1-correct g-gate bypass semantics from
``decoding.cd_apc_gates``.
"""
from __future__ import annotations

from typing import Optional

import torch

from .cd_apc_gates import GMinPolicy, compute_g_min
from .config import VCHDDecodeConfig
from .contrast import ContrastStats
from .selector import (
    MAX_WINDOW_TOP1_FALLBACK,
    THRESHOLD_COMMIT,
    Selection,
    _normalized_gate_readiness,
    _stable_lexicographic_order,
    build_fixed_window,
    select_ccaw_positions as _select_ccaw_dual,
    select_fixed_window_positions as _select_fixed_dual,
)


def _g_min_vector(
    flipped: torch.BoolTensor,
    config: VCHDDecodeConfig,
) -> torch.Tensor:
    policy = GMinPolicy(str(config.g_min_policy))
    if (not config.enable_g_gate) or policy == GMinPolicy.BYPASS:
        return torch.full(
            (flipped.numel(),),
            float("-inf"),
            dtype=torch.float32,
            device=flipped.device,
        )
    tau_g = float(config.tau_g)
    if policy == GMinPolicy.ALWAYS_POS:
        return torch.full(
            (flipped.numel(),),
            tau_g,
            dtype=torch.float32,
            device=flipped.device,
        )
    if policy == GMinPolicy.ALWAYS_NEG:
        return torch.full(
            (flipped.numel(),),
            -tau_g,
            dtype=torch.float32,
            device=flipped.device,
        )
    return torch.where(
        flipped,
        torch.full_like(flipped, tau_g, dtype=torch.float32),
        torch.full_like(flipped, -tau_g, dtype=torch.float32),
    )


def position_gate_mask(
    stats: ContrastStats,
    positions: torch.LongTensor,
    config: VCHDDecodeConfig,
    *,
    contrast_reliability: torch.Tensor,
) -> torch.BoolTensor:
    base = stats.base_confidence[positions]
    reliability = contrast_reliability[positions]
    gain = stats.absolute_visual_gain[positions]
    flipped = stats.contrast_token[positions] != stats.raw_token[positions]
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
    g_min = _g_min_vector(flipped, config)
    g_ok = gain >= g_min
    return u_ok & r_ok & g_ok


def select_cd_apc_triple_gate_positions(
    stats: ContrastStats,
    mask: torch.BoolTensor,
    config: VCHDDecodeConfig,
    *,
    contrast_reliability: Optional[torch.Tensor] = None,
    mask_capacity: Optional[int] = None,
    max_commit_per_iteration: Optional[int] = None,
) -> Selection:
    """Triple-gate commit selection; falls back to dual-gate API if g-gate off."""

    if not (
        config.enable_g_gate and config.enable_u_gate and config.enable_r_gate
    ):
        # Preserve dual-gate semantics when g-gate is disabled.
        if not config.enable_g_gate:
            return _select_fixed_dual(
                stats,
                mask,
                config,
                contrast_reliability=contrast_reliability,
                mask_capacity=mask_capacity,
                max_commit_per_iteration=max_commit_per_iteration,
            )

    reliability = (
        stats.contrast_confidence
        if contrast_reliability is None
        else contrast_reliability
    )
    if reliability.shape != stats.contrast_confidence.shape:
        raise ValueError(
            "contrast_reliability must match per-position confidence shape"
        )
    window = build_fixed_window(
        mask,
        mask_capacity=(
            config.mask_capacity if mask_capacity is None else mask_capacity
        ),
        max_physical_span=config.max_physical_span,
    )
    active = window.active_positions
    qualified = position_gate_mask(
        stats,
        active,
        config,
        contrast_reliability=reliability,
    )
    qualified_positions = active[qualified]
    commit_limit = int(
        config.max_commit_per_iteration
        if max_commit_per_iteration is None
        else max_commit_per_iteration
    )
    if commit_limit < 1:
        raise ValueError("max_commit_per_iteration must be at least 1")

    if qualified_positions.numel() > 0:
        order = _stable_lexicographic_order(
            qualified_positions,
            reliability[qualified_positions],
            stats.base_confidence[qualified_positions],
        )
        selected = qualified_positions[order][:commit_limit]
        return Selection(
            positions=selected,
            reason=THRESHOLD_COMMIT,
            qualified_count=int(qualified_positions.numel()),
            window=window,
        )

    candidates_mask = torch.ones(
        active.numel(), dtype=torch.bool, device=active.device
    )
    if config.enable_g_gate:
        gain = stats.absolute_visual_gain[active]
        # Paper fallback: drop candidates with g <= -tau_g when g-gate is on.
        candidates_mask = gain > -float(config.tau_g)
        if not bool(candidates_mask.any()):
            candidates_mask = torch.ones_like(candidates_mask)
    cand = active[candidates_mask]
    base = stats.base_confidence[cand]
    contrast = reliability[cand]
    base_readiness = _normalized_gate_readiness(base, config.tau_base)
    contrast_readiness = _normalized_gate_readiness(
        contrast, config.tau_contrast
    )
    joint_readiness = torch.minimum(base_readiness, contrast_readiness)
    order = _stable_lexicographic_order(cand, joint_readiness, contrast)
    selected = cand[order[:1]]
    return Selection(
        positions=selected,
        reason=MAX_WINDOW_TOP1_FALLBACK,
        qualified_count=0,
        window=window,
    )


def select_ccaw_cd_apc_positions(
    stats: ContrastStats,
    mask: torch.BoolTensor,
    config: VCHDDecodeConfig,
    *,
    contrast_reliability: torch.Tensor,
    current_mask_capacity: int,
    max_commit_per_iteration: Optional[int] = None,
) -> Selection:
    if not config.enable_g_gate:
        return _select_ccaw_dual(
            stats,
            mask,
            config,
            contrast_reliability=contrast_reliability,
            current_mask_capacity=current_mask_capacity,
            max_commit_per_iteration=max_commit_per_iteration,
        )

    capacity = max(int(config.mask_capacity), int(current_mask_capacity))
    capacity = min(capacity, int(config.ccaw_max_mask_capacity))
    expansions = 0
    previous_active: Optional[torch.LongTensor] = None
    qualified_budget = int(config.ccaw_qualified_budget)

    while True:
        selection = select_cd_apc_triple_gate_positions(
            stats,
            mask,
            config,
            contrast_reliability=contrast_reliability,
            mask_capacity=capacity,
            max_commit_per_iteration=max_commit_per_iteration,
        )
        met_budget = (
            selection.reason == THRESHOLD_COMMIT
            and selection.qualified_count >= qualified_budget
        )
        if met_budget:
            return Selection(
                positions=selection.positions,
                reason=selection.reason,
                qualified_count=selection.qualified_count,
                window=selection.window,
                search_expansions=expansions,
            )

        active = selection.window.active_positions
        saturated = (
            previous_active is not None
            and torch.equal(previous_active, active)
        )
        if capacity >= int(config.ccaw_max_mask_capacity) or saturated:
            return Selection(
                positions=selection.positions,
                reason=selection.reason,
                qualified_count=selection.qualified_count,
                window=selection.window,
                search_expansions=expansions,
            )

        next_capacity = min(
            capacity + int(config.ccaw_expand_step),
            int(config.ccaw_max_mask_capacity),
        )
        if next_capacity == capacity:
            return Selection(
                positions=selection.positions,
                reason=selection.reason,
                qualified_count=selection.qualified_count,
                window=selection.window,
                search_expansions=expansions,
            )
        previous_active = active
        capacity = next_capacity
        expansions += 1
