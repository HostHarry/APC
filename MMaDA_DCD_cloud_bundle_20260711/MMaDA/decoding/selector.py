from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .config import VCHDDecodeConfig
from .contrast import ContrastStats


THRESHOLD_COMMIT = "THRESHOLD"
MAX_WINDOW_TOP1_FALLBACK = "MAX_WINDOW_TOP1_FALLBACK"
FOCUS_DWELL_EMPTY_FALLBACK = "FOCUS_DWELL_EMPTY_FALLBACK"


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
    evidence_state: Optional[str] = None
    evidence_veto_count: int = 0
    token_overrides: Optional[torch.LongTensor] = None


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

    base_readiness = _normalized_gate_readiness(base, config.tau_base)
    contrast_readiness = _normalized_gate_readiness(
        contrast, config.tau_contrast
    )
    joint_readiness = torch.minimum(base_readiness, contrast_readiness)
    order = _stable_lexicographic_order(
        active, joint_readiness, contrast
    )
    selected = active[order[:1]]
    return Selection(
        positions=selected,
        reason=MAX_WINDOW_TOP1_FALLBACK,
        qualified_count=0,
        window=window,
    )


def select_focus_dwell_positions(
    stats: ContrastStats,
    mask: torch.BoolTensor,
    config: VCHDDecodeConfig,
    *,
    eligible_mask: torch.BoolTensor,
    marginal_entropy: torch.Tensor,
    max_commit_per_iteration: Optional[int] = None,
) -> Selection:
    """Select from the persistent-focus dwell window ranked by marginal entropy.

    ``max_commit_per_iteration`` overrides ``config.max_commit_per_iteration``
    when provided; used by the focus_longtail+CCAW combined path to feed a
    pressure-adaptive commit budget into the focus dwell selector.
    """

    if eligible_mask.shape != mask.shape or marginal_entropy.shape != mask.shape:
        raise ValueError(
            "Focus dwell selector inputs must match the response mask shape"
        )
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must be boolean")
    if bool((eligible_mask & ~mask).any()):
        raise ValueError(
            "Focus dwell eligibility cannot include committed positions"
        )

    commit_budget = (
        int(max_commit_per_iteration)
        if max_commit_per_iteration is not None
        else int(config.max_commit_per_iteration)
    )
    if commit_budget < 1:
        raise ValueError(
            "max_commit_per_iteration override must be at least 1, got "
            f"{commit_budget}"
        )

    if not bool(eligible_mask.any()):
        fallback = select_fixed_window_positions(
            stats,
            mask,
            config,
            mask_capacity=config.focus_capacity,
            max_commit_per_iteration=1,
        )
        return Selection(
            positions=fallback.positions,
            reason=FOCUS_DWELL_EMPTY_FALLBACK,
            qualified_count=0,
            window=fallback.window,
        )

    window = build_fixed_window(
        eligible_mask,
        mask_capacity=config.focus_capacity,
        max_physical_span=config.max_physical_span,
    )
    active = window.active_positions
    if not bool(torch.isfinite(marginal_entropy[active]).all()):
        raise FloatingPointError(
            "Eligible focus dwell marginal entropy is not finite"
        )
    base = stats.base_confidence[active]
    contrast = stats.contrast_confidence[active]
    qualified = (
        (base >= float(config.tau_base))
        & (contrast >= float(config.tau_contrast))
    )
    qualified_positions = active[qualified]

    if qualified_positions.numel() > 0:
        coherence = -marginal_entropy[qualified_positions]
        order = _stable_lexicographic_order(
            qualified_positions,
            coherence,
            stats.contrast_confidence[qualified_positions],
        )
        selected = qualified_positions[order][:commit_budget]
        return Selection(
            positions=selected,
            reason=THRESHOLD_COMMIT,
            qualified_count=int(qualified_positions.numel()),
            window=window,
        )

    base_readiness = _normalized_gate_readiness(base, config.tau_base)
    contrast_readiness = _normalized_gate_readiness(
        contrast, config.tau_contrast
    )
    joint_readiness = torch.minimum(base_readiness, contrast_readiness)
    order = _stable_lexicographic_order(
        active,
        joint_readiness,
        -marginal_entropy[active],
    )
    return Selection(
        positions=active[order[:1]],
        reason=MAX_WINDOW_TOP1_FALLBACK,
        qualified_count=0,
        window=window,
    )


def select_counterfactual_exposure_positions(
    stats: ContrastStats,
    mask: torch.BoolTensor,
    config: VCHDDecodeConfig,
    *,
    evidence_lower_bound: torch.Tensor,
    effective_exposure: torch.Tensor,
) -> Selection:
    """Commit only candidates with sufficient counterfactual visual evidence."""

    if (
        evidence_lower_bound.shape != mask.shape
        or effective_exposure.shape != mask.shape
    ):
        raise ValueError(
            "Counterfactual evidence inputs must match the response mask shape"
        )
    if not bool(torch.isfinite(evidence_lower_bound[mask]).all()):
        raise FloatingPointError(
            "Counterfactual evidence lower bound contains NaN or Inf"
        )
    if not bool(torch.isfinite(effective_exposure[mask]).all()):
        raise FloatingPointError(
            "Counterfactual effective exposure contains NaN or Inf"
        )

    window = build_fixed_window(
        mask,
        mask_capacity=config.counterfactual_exposure_window_size,
        max_physical_span=config.max_physical_span,
    )
    active = window.active_positions
    lower_bound = evidence_lower_bound[active]
    exposure = effective_exposure[active]
    base = stats.base_confidence[active]
    contrast = stats.contrast_confidence[active]
    standard_gate = (
        (base >= float(config.tau_base))
        & (contrast >= float(config.tau_contrast))
    )
    support = lower_bound >= float(
        config.counterfactual_exposure_positive_threshold
    )
    opposed = lower_bound <= -float(
        config.counterfactual_exposure_negative_threshold
    )
    neutral = ~(support | opposed)
    support_qualified = support & standard_gate
    neutral_qualified = (
        neutral
        & (base >= float(config.tau_base))
        & (
            contrast
            >= float(config.counterfactual_exposure_neutral_tau_contrast)
        )
        & (
            exposure
            >= float(
                config.counterfactual_exposure_min_effective_exposure
            )
        )
    )
    eligible = support_qualified | neutral_qualified
    evidence_veto_count = int((standard_gate & ~eligible).sum().item())

    if bool(eligible.any()):
        eligible_positions = active[eligible]
        eligible_lower_bound = lower_bound[eligible]
        eligible_contrast = contrast[eligible]
        eligible_support = support_qualified[eligible]
        order = torch.argsort(eligible_positions, stable=True)
        order = order[
            torch.argsort(
                eligible_contrast[order], descending=True, stable=True
            )
        ]
        order = order[
            torch.argsort(
                eligible_lower_bound[order], descending=True, stable=True
            )
        ]
        order = order[
            torch.argsort(
                eligible_support[order], descending=True, stable=True
            )
        ]
        selected = eligible_positions[order][
            : int(config.max_commit_per_iteration)
        ]
        selected_support = support_qualified[
            torch.searchsorted(active, selected)
        ]
        evidence_state = (
            "support"
            if bool(selected_support.all())
            else (
                "neutral"
                if not bool(selected_support.any())
                else "mixed"
            )
        )
        return Selection(
            positions=selected,
            reason=THRESHOLD_COMMIT,
            qualified_count=int(eligible.sum().item()),
            window=window,
            evidence_state=evidence_state,
            evidence_veto_count=evidence_veto_count,
        )

    base_readiness = _normalized_gate_readiness(base, config.tau_base)
    contrast_readiness = _normalized_gate_readiness(
        contrast, config.tau_contrast
    )
    joint_readiness = torch.minimum(base_readiness, contrast_readiness)
    order = _stable_lexicographic_order(
        active, joint_readiness, contrast
    )
    return Selection(
        positions=active[order[:1]],
        reason=MAX_WINDOW_TOP1_FALLBACK,
        qualified_count=0,
        window=window,
        evidence_state="fallback",
        evidence_veto_count=evidence_veto_count,
    )


def select_unified_trajectory_positions(
    stats: ContrastStats,
    mask: torch.BoolTensor,
    config: VCHDDecodeConfig,
    *,
    token_overrides: torch.LongTensor,
    trajectory_confidence: torch.Tensor,
    trajectory_entropy: torch.Tensor,
    trajectory_margin: torch.Tensor,
    window_size_override: Optional[int] = None,
    evidence_name: str = "posterior",
) -> Selection:
    """Select directly from the always-defined unified trajectory posterior."""

    expected_shape = mask.shape
    tensors = (
        token_overrides,
        trajectory_confidence,
        trajectory_entropy,
        trajectory_margin,
    )
    if any(value.shape != expected_shape for value in tensors):
        raise ValueError(
            "Unified trajectory selector inputs must match the response mask"
        )
    if token_overrides.dtype != torch.long:
        raise TypeError("token_overrides must be a long tensor")
    for value in (
        trajectory_confidence,
        trajectory_entropy,
        trajectory_margin,
    ):
        if not bool(torch.isfinite(value[mask]).all()):
            raise FloatingPointError(
                "Unified trajectory selector scores must be finite"
            )

    window = build_fixed_window(
        mask,
        mask_capacity=(
            config.unified_trajectory_window_size
            if window_size_override is None
            else int(window_size_override)
        ),
        max_physical_span=config.max_physical_span,
    )
    active = window.active_positions
    base = stats.base_confidence[active]
    confidence = trajectory_confidence[active]
    entropy = trajectory_entropy[active]
    margin = trajectory_margin[active]
    standard_gate = (
        (base >= float(config.tau_base))
        & (confidence >= float(config.tau_contrast))
    )
    qualified = standard_gate

    def order_candidates(
        positions: torch.LongTensor,
        *,
        primary: torch.Tensor,
        secondary: torch.Tensor,
    ) -> torch.LongTensor:
        return _stable_lexicographic_order(positions, primary, secondary)

    if bool(qualified.any()):
        qualified_positions = active[qualified]
        # Posterior confidence is primary; a wider top-1/top-2 margin and
        # lower posterior entropy break ties without changing the gate.
        quality = confidence[qualified] + 0.10 * margin[qualified]
        coherence = -entropy[qualified]
        order = order_candidates(
            qualified_positions,
            primary=quality,
            secondary=coherence,
        )
        selected = qualified_positions[order][
            : int(config.max_commit_per_iteration)
        ]
        return Selection(
            positions=selected,
            reason=THRESHOLD_COMMIT,
            qualified_count=int(qualified.sum().item()),
            window=window,
            evidence_state=evidence_name,
            token_overrides=token_overrides[selected],
        )

    base_readiness = _normalized_gate_readiness(base, config.tau_base)
    confidence_readiness = _normalized_gate_readiness(
        confidence, config.tau_contrast
    )
    readiness = torch.minimum(base_readiness, confidence_readiness)
    order = order_candidates(
        active,
        primary=readiness,
        secondary=margin,
    )
    selected = active[order[:1]]
    return Selection(
        positions=selected,
        reason=MAX_WINDOW_TOP1_FALLBACK,
        qualified_count=0,
        window=window,
        evidence_state=f"{evidence_name}_dual_gate_fallback",
        token_overrides=token_overrides[selected],
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
