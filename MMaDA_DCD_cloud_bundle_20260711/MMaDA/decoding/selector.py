from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .config import VCHDDecodeConfig
from .contrast import ContrastStats


THRESHOLD_COMMIT = "THRESHOLD"
MAX_WINDOW_TOP1_FALLBACK = "MAX_WINDOW_TOP1_FALLBACK"


@dataclass(frozen=True)
class Window:
    left: int
    right: int
    active_positions: torch.LongTensor
    mask_capacity: int


@dataclass(frozen=True)
class Selection:
    positions: torch.LongTensor
    reason: str
    qualified_count: int
    window: Window
    search_expansions: int = 0


def build_fixed_window(
    mask: torch.BoolTensor,
    *,
    mask_capacity: int,
    max_physical_span: int,
) -> Window:
    if mask.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {tuple(mask.shape)}")
    masked_positions = torch.nonzero(mask, as_tuple=True)[0]
    if masked_positions.numel() == 0:
        raise ValueError("Cannot build a window after all MASK positions are committed")

    left = int(masked_positions[0].item())
    within_span = masked_positions[
        masked_positions < left + int(max_physical_span)
    ]
    active = within_span[: int(mask_capacity)]
    if active.numel() == 0:
        raise RuntimeError("A left-anchored window must contain its first MASK")
    right = int(active[-1].item()) + 1
    return Window(
        left=left,
        right=right,
        active_positions=active,
        mask_capacity=int(mask_capacity),
    )


def _stable_lexicographic_order(
    positions: torch.LongTensor,
    primary: torch.Tensor,
    secondary: torch.Tensor,
) -> torch.LongTensor:
    """Order by primary desc, secondary desc, position asc."""

    order = torch.argsort(positions, stable=True)
    order = order[
        torch.argsort(secondary[order], descending=True, stable=True)
    ]
    order = order[
        torch.argsort(primary[order], descending=True, stable=True)
    ]
    return order


def _normalized_gate_readiness(
    values: torch.Tensor, threshold: float
) -> torch.Tensor:
    if threshold <= 0.0:
        return torch.full_like(values, torch.inf)
    return values / float(threshold)


def select_fixed_window_positions(
    stats: ContrastStats,
    mask: torch.BoolTensor,
    config: VCHDDecodeConfig,
    *,
    contrast_reliability: Optional[torch.Tensor] = None,
    mask_capacity: Optional[int] = None,
    max_commit_per_iteration: Optional[int] = None,
) -> Selection:
    """Apply the base/contrast dual gate, with one-token progress fallback."""

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
    base = stats.base_confidence[active]
    contrast = reliability[active]
    qualified = (
        (base >= float(config.tau_base))
        & (contrast >= float(config.tau_contrast))
    )
    qualified_positions = active[qualified]

    if qualified_positions.numel() > 0:
        qualified_base = stats.base_confidence[qualified_positions]
        qualified_contrast = reliability[qualified_positions]
        order = _stable_lexicographic_order(
            qualified_positions, qualified_contrast, qualified_base
        )
        commit_limit = (
            int(config.max_commit_per_iteration)
            if max_commit_per_iteration is None
            else int(max_commit_per_iteration)
        )
        if commit_limit < 1:
            raise ValueError("max_commit_per_iteration must be at least 1")
        selected = qualified_positions[order][
            :commit_limit
        ]
        return Selection(
            positions=selected,
            reason=THRESHOLD_COMMIT,
            qualified_count=int(qualified_positions.numel()),
            window=window,
        )

    fallback_capacity = int(config.fallback_mask_capacity)
    fallback_active = (
        active
        if fallback_capacity == 0
        else active[:fallback_capacity]
    )
    if fallback_active.numel() == 0:
        raise RuntimeError("Fallback scope must contain at least one MASK")
    if config.fallback_policy == "leftmost":
        selected = fallback_active[:1]
        return Selection(
            positions=selected,
            reason=MAX_WINDOW_TOP1_FALLBACK,
            qualified_count=0,
            window=window,
        )

    fallback_base = stats.base_confidence[fallback_active]
    fallback_contrast = reliability[fallback_active]
    base_readiness = _normalized_gate_readiness(
        fallback_base, config.tau_base
    )
    contrast_readiness = _normalized_gate_readiness(
        fallback_contrast, config.tau_contrast
    )
    joint_readiness = torch.minimum(base_readiness, contrast_readiness)
    order = _stable_lexicographic_order(
        fallback_active, joint_readiness, fallback_contrast
    )
    selected = fallback_active[order[:1]]
    return Selection(
        positions=selected,
        reason=MAX_WINDOW_TOP1_FALLBACK,
        qualified_count=0,
        window=window,
    )


def select_ccaw_positions(
    stats: ContrastStats,
    mask: torch.BoolTensor,
    config: VCHDDecodeConfig,
    *,
    contrast_reliability: torch.Tensor,
    current_mask_capacity: int,
) -> Selection:
    """Expand until the safe-token budget is met or the window saturates."""

    capacity = max(int(config.mask_capacity), int(current_mask_capacity))
    capacity = min(capacity, int(config.ccaw_max_mask_capacity))
    expansions = 0
    previous_active: Optional[torch.LongTensor] = None

    while True:
        selection = select_fixed_window_positions(
            stats,
            mask,
            config,
            contrast_reliability=contrast_reliability,
            mask_capacity=capacity,
        )
        active = selection.window.active_positions
        qualified_target = min(
            int(config.ccaw_qualified_budget),
            int(active.numel()),
        )
        if (
            selection.reason == THRESHOLD_COMMIT
            and selection.qualified_count >= qualified_target
        ):
            return Selection(
                positions=selection.positions,
                reason=selection.reason,
                qualified_count=selection.qualified_count,
                window=selection.window,
                search_expansions=expansions,
            )

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
