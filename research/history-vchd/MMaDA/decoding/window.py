from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import VCHDDecodeConfig
from .contrast import ContrastStats
from .selector import Window


@dataclass
class CCAWState:
    mask_capacity: int
    pressure_ema: float = 0.0


@dataclass(frozen=True)
class WindowPressure:
    candidate_conflict: float
    history_instability: float
    qualification_deficit: float
    combined: float


def scope_next_hard_block(
    mask: torch.BoolTensor,
    *,
    block_start: int,
    block_size: int,
) -> tuple[torch.BoolTensor, int, int]:
    """Return the next non-empty fixed physical block mask."""

    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    start = max(0, int(block_start))
    length = int(mask.numel())
    while start < length:
        end = min(length, start + int(block_size))
        scoped = torch.zeros_like(mask)
        scoped[start:end] = mask[start:end]
        if bool(scoped.any()):
            return scoped, start, end
        start = end
    raise ValueError("No non-empty hard block remains")


def compute_window_pressure(
    stats: ContrastStats,
    history_stability: torch.Tensor,
    contrast_reliability: torch.Tensor,
    window: Window,
    config: VCHDDecodeConfig,
) -> WindowPressure:
    """Compute d/u/z/h on the final pre-commit active window."""

    active = window.active_positions
    if active.numel() == 0:
        raise ValueError("Cannot compute CCAW pressure on an empty window")
    if (
        history_stability.shape != stats.visual_relevance.shape
        or contrast_reliability.shape != stats.contrast_confidence.shape
    ):
        raise ValueError("CCAW score tensors must match response positions")

    relevance = stats.visual_relevance[active]
    changed = (
        stats.contrast_token[active] != stats.raw_token[active]
    ).float()
    candidate_conflict = (relevance * changed).mean()
    history_instability = (
        relevance * (1.0 - history_stability[active])
    ).mean()
    qualified = (
        stats.base_confidence[active] >= float(config.tau_base)
    ) & (
        contrast_reliability[active] >= float(config.tau_contrast)
    )
    qualification_deficit = 1.0 - qualified.float().mean()
    combined = (
        candidate_conflict
        + history_instability
        + qualification_deficit
    ) / 3.0

    values = (
        candidate_conflict,
        history_instability,
        qualification_deficit,
        combined,
    )
    if not all(bool(torch.isfinite(value)) for value in values):
        raise FloatingPointError("CCAW pressure produced NaN or Inf")
    return WindowPressure(
        candidate_conflict=float(candidate_conflict.clamp(0.0, 1.0).item()),
        history_instability=float(
            history_instability.clamp(0.0, 1.0).item()
        ),
        qualification_deficit=float(
            qualification_deficit.clamp(0.0, 1.0).item()
        ),
        combined=float(combined.clamp(0.0, 1.0).item()),
    )


def update_ccaw_state(
    state: CCAWState,
    pressure: WindowPressure,
    config: VCHDDecodeConfig,
) -> None:
    """Update persistent capacity after one atomic commit."""

    decay = float(config.ccaw_pressure_ema_decay)
    state.pressure_ema = (
        decay * float(state.pressure_ema)
        + (1.0 - decay) * float(pressure.combined)
    )
    base = int(config.mask_capacity)
    maximum = int(config.ccaw_max_mask_capacity)
    target = base + round((maximum - base) * state.pressure_ema)
    difference = target - int(state.mask_capacity)
    difference = max(
        -int(config.ccaw_shrink_step),
        min(int(config.ccaw_expand_step), difference),
    )
    state.mask_capacity = max(
        base, min(maximum, int(state.mask_capacity) + difference)
    )


def update_inverse_ccaw_state(
    state: CCAWState,
    pressure: WindowPressure,
    config: VCHDDecodeConfig,
) -> None:
    """Shrink on high pressure and expand on low pressure in the same step."""

    decay = float(config.ccaw_pressure_ema_decay)
    state.pressure_ema = (
        decay * float(state.pressure_ema)
        + (1.0 - decay) * float(pressure.combined)
    )
    minimum = int(config.mask_capacity)
    maximum = int(config.ccaw_max_mask_capacity)
    current_pressure = min(
        1.0, max(0.0, float(pressure.combined))
    )
    target = maximum - round(
        (maximum - minimum) * current_pressure
    )
    difference = target - int(state.mask_capacity)
    difference = max(
        -int(config.ccaw_shrink_step),
        min(int(config.ccaw_expand_step), difference),
    )
    state.mask_capacity = max(
        minimum,
        min(maximum, int(state.mask_capacity) + difference),
    )


def pressure_adaptive_commit_budget(
    pressure: WindowPressure,
    config: VCHDDecodeConfig,
) -> int:
    """Map high pressure to a smaller hard-block commit budget."""

    minimum = int(config.ccaw_min_commit_per_iteration)
    maximum = int(config.max_commit_per_iteration)
    pressure_value = min(1.0, max(0.0, float(pressure.combined)))
    budget = maximum - round((maximum - minimum) * pressure_value)
    return max(minimum, min(maximum, int(budget)))
