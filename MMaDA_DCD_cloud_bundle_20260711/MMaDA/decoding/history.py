from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SparseHistory:
    """Top-V EMA distribution plus a single omitted-mass bucket."""

    token_ids: torch.LongTensor
    probs: torch.FloatTensor
    other_prob: torch.FloatTensor
    last_observed_context_version: int
    last_top1_token: int
    consecutive_top1_matches: int


@dataclass(frozen=True)
class HistoryObservation:
    stability: torch.FloatTensor
    next_history: SparseHistory
    updated: bool
    consecutive_top1_matches: int


@dataclass(frozen=True)
class FocusFrame:
    """One iteration's captured contrast distributions on the focus positions.

    The decoder keeps a bounded queue of these frames. Each frame stores the
    ``focus_capacity`` positions that had the highest CD-APC confidence when
    the frame was recorded, together with the full-vocabulary contrast
    distribution at each of those positions. Downstream code interprets the
    queue through a per-position dwell counter (see
    :func:`compute_focus_dwell_counter`), so no set-intersection primitive is
    ever taken across frames.

    ``lookup`` is an optional pre-computed inverse map of length
    ``total_positions`` such that ``lookup[i]`` is the row index of position
    ``i`` in ``distributions`` (or ``-1`` when the position is absent). It is
    built once by :func:`make_focus_frame` and re-used by every downstream
    observation, avoiding the O(dwell_depth) full-vector re-allocations that
    the marginalization loop would otherwise trigger.
    """

    positions: torch.LongTensor
    distributions: torch.FloatTensor
    lookup: Optional[torch.LongTensor] = None


@dataclass(frozen=True)
class FocusDwellObservation:
    """Dwell-gated CD-APC marginal token statistics for one iteration.

    ``dwell_counter[i]`` is the length of the contiguous most-recent focus
    suffix containing position ``i`` (including the current iteration). A
    position is eligible when its dwell counter has reached the configured
    dwell depth; ineligible positions inherit the raw CD-APC decision.
    """

    current_frame: FocusFrame
    eligible_mask: torch.BoolTensor
    dwell_counter: torch.LongTensor
    marginal_token: torch.LongTensor
    marginal_entropy: torch.FloatTensor
    base_confidence: torch.FloatTensor
    contrast_confidence: torch.FloatTensor
    dwell_depth: int


@dataclass(frozen=True)
class CounterfactualEvidenceHistory:
    """Candidate-aligned, exposure-weighted evidence state for one MASK."""

    candidate_token: int
    candidate_age: int
    evidence_sum: torch.FloatTensor
    evidence_sq_sum: torch.FloatTensor
    exposure_sum: torch.FloatTensor
    exposure_sq_sum: torch.FloatTensor
    flip_count: int
    last_observed_context_version: int


@dataclass(frozen=True)
class CounterfactualEvidenceObservation:
    """One per-MASK update of the counterfactual evidence trajectory."""

    next_history: CounterfactualEvidenceHistory
    current_evidence: torch.FloatTensor
    observation_weight: torch.FloatTensor
    mean_evidence: torch.FloatTensor
    evidence_lower_bound: torch.FloatTensor
    effective_exposure: torch.FloatTensor
    effective_observations: torch.FloatTensor
    candidate_flipped: bool
    updated: bool


@dataclass(frozen=True)
class CounterfactualEvidenceBatchObservation:
    """Batched evidence updates for every currently unresolved MASK."""

    next_history: Dict[int, CounterfactualEvidenceHistory]
    evidence_lower_bound: torch.FloatTensor
    effective_exposure: torch.FloatTensor
    candidate_age: torch.LongTensor
    candidate_flipped: torch.BoolTensor
    updated_count: int


@dataclass(frozen=True)
class UnifiedCandidateTrajectory:
    """Sparse candidate trajectory for one unresolved MASK position."""

    token_id: int
    semantic_sum: float
    semantic_sq_sum: float
    semantic_weight_sum: float
    semantic_weight_sq_sum: float
    gain_sum: float
    gain_sq_sum: float
    exposure_sum: float
    exposure_sq_sum: float
    candidate_age: int
    last_seen_context_version: int


@dataclass(frozen=True)
class UnifiedTrajectoryBatchObservation:
    """Exposure-calibrated candidate histories for a compact top-K support."""

    next_history: Dict[int, Dict[int, UnifiedCandidateTrajectory]]
    semantic_mean: torch.FloatTensor
    semantic_std: torch.FloatTensor
    gain_mean: torch.FloatTensor
    gain_lower: torch.FloatTensor
    gain_upper: torch.FloatTensor
    effective_exposure: torch.FloatTensor
    effective_observations: torch.FloatTensor
    candidate_age: torch.LongTensor
    baseline_only: torch.BoolTensor
    updated_count: int


@dataclass(frozen=True)
class UnifiedTrajectoryPosterior:
    """Candidate-level posterior used to choose both MASK and token."""

    candidate_score: torch.FloatTensor
    candidate_posterior: torch.FloatTensor
    candidate_visually_informed: torch.BoolTensor
    candidate_opposed: torch.BoolTensor
    candidate_visual_weight: torch.FloatTensor
    selected_index: torch.LongTensor
    selected_token: torch.LongTensor
    selected_base_confidence: torch.FloatTensor
    selected_confidence: torch.FloatTensor
    selected_entropy: torch.FloatTensor
    selected_margin: torch.FloatTensor
    selected_visually_informed: torch.BoolTensor
    selected_opposed: torch.BoolTensor
    selected_gain_lower: torch.FloatTensor
    selected_gain_upper: torch.FloatTensor
    selected_effective_exposure: torch.FloatTensor
    selected_effective_observations: torch.FloatTensor


@dataclass(frozen=True)
class FocusLongTailObservation:
    """Dwell-gated positions with an exposure-modulated long-tail posterior.

    Fields mirror :class:`FocusDwellObservation` but the marginal distribution
    is computed by mixing a rectangular kernel over the current dwell depth
    with a shifted log-logistic survival kernel over the full contiguous
    dwell suffix of each eligible position. ``mix_ceiling == 0`` collapses
    the long-tail contribution to zero and reproduces the rectangular
    (dwell-only) baseline exactly.
    """

    current_frame: FocusFrame
    eligible_mask: torch.BoolTensor
    dwell_counter: torch.LongTensor
    selected_token: torch.LongTensor
    base_confidence: torch.FloatTensor
    contrast_confidence: torch.FloatTensor
    entropy: torch.FloatTensor
    margin: torch.FloatTensor
    tail_activation: torch.FloatTensor
    exposure: torch.FloatTensor
    relevance_precision: torch.FloatTensor
    conflict: torch.FloatTensor
    long_tail_mass: torch.FloatTensor
    current_weight: torch.FloatTensor
    effective_dwell_depth: torch.LongTensor
    dwell_depth: int


def loglogistic_survival_kernel(
    lags: torch.Tensor,
    *,
    scale: float,
    shape: float,
    offset: float,
) -> torch.FloatTensor:
    """Evaluate a shifted discrete log-logistic survival memory kernel.

    The kernel evaluates
    :math:`S(\\ell) = 1 / (1 + ((\\ell + \\delta) / \\lambda)^{\\kappa})` on
    integer lags. It is monotone non-increasing, strictly positive, and has a
    heavy power-law tail (kernel decays as :math:`\\ell^{-\\kappa}` for large
    :math:`\\ell`). This ensures that dwell-suffix marginalization keeps a
    non-vanishing weight on older observations rather than truncating them.
    """

    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("Log-logistic scale must be finite and positive")
    if not math.isfinite(float(shape)) or shape <= 1.0:
        raise ValueError(
            "Log-logistic shape must be finite and greater than 1"
        )
    if not math.isfinite(float(offset)) or offset < 0.0:
        raise ValueError(
            "Log-logistic offset must be finite and non-negative"
        )
    if not bool(torch.isfinite(lags).all()) or bool((lags < 0).any()):
        raise ValueError("Log-logistic lags must be finite and non-negative")
    scaled_age = (lags.float() + float(offset)) / float(scale)
    return 1.0 / (1.0 + scaled_age.pow(float(shape)))


def focus_longtail_history_upper_bound(
    *,
    kernel_scale: float,
    kernel_shape: float,
    kernel_offset: float,
    epsilon: float,
    dwell_depth: int,
) -> int:
    """Smallest history depth that keeps every log-logistic weight ``>= epsilon``.

    The long-tail kernel decays as ``lag^{-kappa}`` so it never reaches zero
    but the contribution becomes numerically negligible quickly. We solve
    ``S(lag) >= epsilon`` in closed form and clamp to ``dwell_depth`` (the
    rectangular window must always be retained). Callers use this to bound
    the stored frame ring buffer without truncating any weight above
    ``epsilon``.
    """

    if not math.isfinite(float(epsilon)) or not 0.0 < float(epsilon) < 1.0:
        raise ValueError("epsilon must be finite and in (0, 1)")
    max_useful_lag = math.ceil(
        float(kernel_scale)
        * (1.0 / float(epsilon) - 1.0) ** (1.0 / float(kernel_shape))
        - float(kernel_offset)
    )
    return max(int(dwell_depth), max_useful_lag)


def _build_focus_lookup(
    positions: torch.LongTensor, total_positions: int
) -> torch.LongTensor:
    """Return ``lookup[i] = row_of(i)`` (or ``-1`` if ``i`` is absent).

    Building the inverse index once amortises the ``O(total_positions)``
    allocation across every marginalization pass that consults the same
    frame. Downstream code should treat the result as immutable.
    """

    if positions.ndim != 1:
        raise ValueError("Focus positions must be one-dimensional")
    lookup = torch.full(
        (int(total_positions),),
        -1,
        dtype=torch.long,
        device=positions.device,
    )
    lookup[positions] = torch.arange(
        positions.numel(), dtype=torch.long, device=positions.device
    )
    return lookup


def make_focus_frame(
    positions: torch.LongTensor,
    distributions: torch.Tensor,
    total_positions: int,
) -> FocusFrame:
    """Materialise a :class:`FocusFrame` with a cached inverse lookup.

    The observe functions call this helper once when a new frame enters the
    ring buffer, so the reverse lookup used inside the marginalization loop
    is built exactly once per frame rather than once per ``lag``.
    """

    positions = positions.detach().clone()
    distributions = distributions.detach().clone()
    lookup = _build_focus_lookup(positions, total_positions)
    return FocusFrame(
        positions=positions, distributions=distributions, lookup=lookup
    )


def _focus_frame_lookup(
    frame: FocusFrame, total_positions: int, device: torch.device
) -> torch.LongTensor:
    """Return ``frame.lookup`` if it exists on the right device, else rebuild.

    We keep the fallback so callers that construct raw :class:`FocusFrame`
    instances without a lookup still work; :func:`make_focus_frame` should
    be preferred for new code.
    """

    lookup = frame.lookup
    if lookup is not None and lookup.device == device:
        if int(lookup.numel()) != int(total_positions):
            raise ValueError(
                "Cached focus frame lookup length disagrees with the current "
                "position count; rebuild the frame via ``make_focus_frame``"
            )
        return lookup
    return _build_focus_lookup(frame.positions.to(device), total_positions)


def _current_focus_positions(
    position_confidence: torch.Tensor,
    mask: torch.BoolTensor,
    focus_capacity: int,
) -> torch.LongTensor:
    """Return the top ``focus_capacity`` unresolved positions by confidence.

    These positions form the *current focus set* used to update the dwell
    counters. They are ordered by descending contrast confidence with a
    stable secondary key so decisions are deterministic on ties.
    """

    masked_positions = torch.nonzero(mask, as_tuple=True)[0]
    if masked_positions.numel() == 0:
        raise ValueError(
            "Cannot identify the current focus set without masked positions"
        )
    order = torch.argsort(
        position_confidence[masked_positions],
        descending=True,
        stable=True,
    )
    retain = min(int(focus_capacity), int(masked_positions.numel()))
    return masked_positions[order[:retain]]


def compute_focus_dwell_counter(
    focus_frames: Sequence[FocusFrame],
    current_positions: torch.LongTensor,
    *,
    total_positions: int,
) -> torch.LongTensor:
    """Per-position dwell counter: length of the contiguous newest suffix of
    focus frames containing each position, plus one for the current frame.

    This is the *only* eligibility primitive used by the decoder. It reads as
    "how many consecutive most-recent iterations has this position stayed in
    the model's high-confidence focus region", and it replaces any explicit
    set-intersection language across frames. Two positions whose dwell
    counters have the same value have appeared in exactly the same suffix of
    focus frames.
    """

    device = current_positions.device
    dwell = torch.zeros(int(total_positions), dtype=torch.long, device=device)
    if current_positions.numel() == 0:
        return dwell
    dwell[current_positions] = 1
    chain = torch.zeros(int(total_positions), dtype=torch.bool, device=device)
    chain[current_positions] = True
    for frame in reversed(list(focus_frames)):
        if frame.positions.device != device:
            raise ValueError(
                "Focus frames must live on the same device as the mask"
            )
        membership = torch.zeros(
            int(total_positions), dtype=torch.bool, device=device
        )
        membership[frame.positions] = True
        chain &= membership
        if not bool(chain.any()):
            break
        dwell += chain.long()
    return dwell


def observe_focus_longtail(
    focus_frames: Sequence[FocusFrame],
    contrast_distribution: torch.Tensor,
    visual_distribution: torch.Tensor,
    position_confidence: torch.Tensor,
    apc_mass: torch.Tensor,
    mask: torch.BoolTensor,
    exposure: torch.Tensor,
    visual_relevance: torch.Tensor,
    *,
    dwell_depth: int,
    focus_capacity: int,
    kernel_scale: float,
    kernel_shape: float,
    kernel_offset: float,
    mix_ceiling: float,
    exposure_tau: float,
    relevance_tau: float,
    conflict_tau: float,
) -> FocusLongTailObservation:
    """Marginalize a dwell-gated trajectory with a log-logistic long-tail kernel.

    Eligibility is decided by the per-position dwell counter over the ``dwell_depth``
    most-recent focus frames plus the current iteration. Ineligible positions
    return their raw CD-APC decision. For each eligible position ``i`` the
    posterior mixes a rectangular kernel over the current dwell window with a
    shifted log-logistic survival kernel over the entire dwell suffix. The
    mixing weight is exposure-, relevance-, and conflict-modulated and clamps
    to ``[0, mix_ceiling]``; setting ``mix_ceiling = 0`` collapses the
    long-tail contribution to zero and the posterior becomes the pure
    dwell-window rectangular baseline.
    """

    if contrast_distribution.ndim != 2:
        raise ValueError(
            "Focus long-tail contrast distribution must be two-dimensional"
        )
    if visual_distribution.shape != contrast_distribution.shape:
        raise ValueError(
            "Focus long-tail visual and contrast distributions must match"
        )
    position_count = int(contrast_distribution.shape[0])
    expected_shape = (position_count,)
    if (
        position_confidence.shape != expected_shape
        or apc_mass.shape != expected_shape
        or mask.shape != expected_shape
        or exposure.shape != expected_shape
        or visual_relevance.shape != expected_shape
    ):
        raise ValueError(
            "Focus long-tail position statistics must match distributions"
        )
    if mask.dtype != torch.bool:
        raise TypeError("Focus long-tail mask must be boolean")
    if dwell_depth < 1:
        raise ValueError("dwell_depth must be at least 1")
    if focus_capacity < 1:
        raise ValueError("focus_capacity must be at least 1")
    if not (
        math.isfinite(float(mix_ceiling))
        and 0.0 <= float(mix_ceiling) <= 1.0
    ):
        raise ValueError("mix_ceiling must be finite and in [0, 1]")
    for name, value in (
        ("exposure_tau", exposure_tau),
        ("relevance_tau", relevance_tau),
        ("conflict_tau", conflict_tau),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    tensors = (
        contrast_distribution,
        visual_distribution,
        position_confidence,
        apc_mass,
        exposure,
        visual_relevance,
    )
    # One host synchronisation instead of six: build the six per-tensor finite
    # reductions eagerly (they run as concurrent CUDA kernels) and only issue
    # a single ``.item()`` at the end.
    if not bool(
        torch.stack(
            [torch.isfinite(value).all() for value in tensors]
        ).all().item()
    ):
        raise FloatingPointError(
            "Focus long-tail observations must be finite"
        )
    if bool((exposure < 0.0).any()):
        raise ValueError("Focus long-tail exposure must be non-negative")

    current_positions = _current_focus_positions(
        position_confidence, mask, focus_capacity
    )
    current_frame = make_focus_frame(
        current_positions,
        contrast_distribution[current_positions],
        position_count,
    )

    # Validate each stored frame once; the dwell counter walks them again in
    # ``compute_focus_dwell_counter``.
    for frame in focus_frames:
        if frame.distributions.ndim != 2:
            raise ValueError(
                "Focus frame distributions must be two-dimensional"
            )
        if frame.distributions.shape[0] != frame.positions.numel():
            raise ValueError(
                "Focus frame position/distribution counts must match"
            )
        if frame.distributions.shape[1] != contrast_distribution.shape[1]:
            raise ValueError(
                "Focus frame vocabulary size changed between iterations"
            )
        if frame.positions.device != mask.device:
            raise ValueError(
                "Focus frames and current mask must share a device"
            )

    recent_focus_frames = list(focus_frames)[-int(dwell_depth) :]
    dwell_counter = compute_focus_dwell_counter(
        focus_frames,
        current_positions,
        total_positions=position_count,
    )
    # Warmup semantics: before ``dwell_depth`` frames exist we require the
    # position to appear in every stored frame plus the current one, so the
    # first ``dwell_depth`` iterations accept eligibility on shorter suffixes.
    # This is *not* the strict ``D >= dwell_depth + 1`` gate: it degenerates
    # to that only once the ring buffer has been filled.
    dwell_gate = 1 + len(recent_focus_frames)
    eligible_mask = dwell_counter >= dwell_gate

    current = contrast_distribution.float()
    visual = visual_distribution.float()
    tiny = torch.finfo(torch.float32).tiny
    relevance_precision = -torch.expm1(
        -visual_relevance.float().clamp_min(0.0)
        / float(relevance_tau)
    )
    exposure_precision = -torch.expm1(
        -exposure.float().clamp_min(0.0) / float(exposure_tau)
    )

    selected_token = current.argmax(dim=-1)
    base_confidence = visual.gather(
        -1, selected_token.unsqueeze(-1)
    ).squeeze(-1)
    contrast_confidence = apc_mass.float() * current.gather(
        -1, selected_token.unsqueeze(-1)
    ).squeeze(-1)
    entropy = torch.full_like(position_confidence.float(), torch.inf)
    margin = torch.zeros_like(position_confidence.float())
    tail_activation = torch.zeros_like(position_confidence.float())
    conflict = torch.zeros_like(position_confidence.float())
    long_tail_mass = torch.zeros_like(position_confidence.float())
    current_weight = torch.ones_like(position_confidence.float())
    effective_dwell_depth = torch.ones_like(
        selected_token, dtype=torch.long
    )

    eligible_positions = torch.nonzero(eligible_mask, as_tuple=True)[0]
    if eligible_positions.numel() > 0:
        current_eligible = current[eligible_positions]
        available_focus_frames = list(focus_frames)
        # Memoise the inverse lookup for a given frame within this call.
        # ``FocusFrame.lookup`` covers the common decoder path (frames enter
        # through :func:`make_focus_frame`); raw frames construct once and
        # cache in the local dict so the dwell-reference pass and the main
        # marginalization loop share the result without an upfront allocation
        # storm on histories that break early.
        _local_lookup_cache: Dict[int, torch.LongTensor] = {}

        def _lookup_for(frame_index: int) -> torch.LongTensor:
            frame = available_focus_frames[frame_index]
            cached = frame.lookup
            if cached is not None and cached.device == mask.device:
                return cached
            cached = _local_lookup_cache.get(frame_index)
            if cached is not None:
                return cached
            fresh = _build_focus_lookup(frame.positions, position_count)
            _local_lookup_cache[frame_index] = fresh
            return fresh

        recent_start = len(available_focus_frames) - len(recent_focus_frames)
        # Dwell-window reference distribution: average over the current frame
        # and the ``dwell_depth`` most-recent frames. Eligibility guarantees
        # that every eligible position appears in each of these frames.
        window_distributions = [current_eligible]
        for offset in range(len(recent_focus_frames) - 1, -1, -1):
            frame_index = recent_start + offset
            lookup = _lookup_for(frame_index)
            frame = available_focus_frames[frame_index]
            rows = lookup[eligible_positions]
            if bool((rows < 0).any()):
                raise RuntimeError(
                    "Focus dwell window lost an eligible position"
                )
            window_distributions.append(
                frame.distributions[rows].float()
            )
        dwell_reference = torch.stack(window_distributions, dim=0).mean(
            dim=0
        )
        dwell_reference = dwell_reference / dwell_reference.sum(
            dim=-1, keepdim=True
        ).clamp_min(tiny)
        eligible_conflict = 0.5 * torch.abs(
            current_eligible - dwell_reference
        ).sum(dim=-1)
        conflict_precision = -torch.expm1(
            -eligible_conflict / float(conflict_tau)
        )
        eligible_activation = (
            float(mix_ceiling)
            * exposure_precision[eligible_positions]
            * relevance_precision[eligible_positions]
            * conflict_precision
        ).clamp(0.0, 1.0)

        numerator = torch.zeros_like(current_eligible)
        denominator = torch.zeros_like(eligible_activation)
        tail_numerator = torch.zeros_like(eligible_activation)
        depth = torch.zeros_like(eligible_positions)
        # A position leaves the marginalization chain the first time it is
        # missing from a historical focus frame. This preserves the invariant
        # that only contiguous dwell suffixes contribute weight.
        chain_active = torch.ones_like(
            eligible_positions, dtype=torch.bool
        )
        max_lag = len(available_focus_frames)
        lags = torch.arange(
            max_lag + 1,
            dtype=torch.float32,
            device=mask.device,
        )
        survival = loglogistic_survival_kernel(
            lags,
            scale=kernel_scale,
            shape=kernel_shape,
            offset=kernel_offset,
        )

        for lag in range(max_lag + 1):
            rectangular = 1.0 if lag <= int(dwell_depth) else 0.0
            lag_weight = (
                (1.0 - eligible_activation) * rectangular
                + eligible_activation * survival[lag]
            )
            if lag == 0:
                active_indices = torch.arange(
                    eligible_positions.numel(),
                    device=mask.device,
                )
                lag_distribution = current_eligible
            else:
                frame_index = len(available_focus_frames) - lag
                frame = available_focus_frames[frame_index]
                # ``_lookup_for`` returns the cached ``FocusFrame.lookup``
                # when present (the common decoder path) or memoises a fresh
                # build otherwise. Either way each frame is materialised at
                # most once per call, and only when actually visited.
                lookup = _lookup_for(frame_index)
                rows = lookup[eligible_positions]
                chain_active &= rows >= 0
                active_indices = torch.nonzero(
                    chain_active, as_tuple=True
                )[0]
                if active_indices.numel() == 0:
                    break
                lag_distribution = frame.distributions[
                    rows[active_indices]
                ].float()
            active_weight = lag_weight[active_indices]
            numerator[active_indices] += (
                active_weight.unsqueeze(-1) * lag_distribution
            )
            denominator[active_indices] += active_weight
            depth[active_indices] += (active_weight > tiny).long()
            if lag > int(dwell_depth):
                tail_numerator[active_indices] += active_weight

        posterior = numerator / denominator.unsqueeze(-1).clamp_min(tiny)
        posterior = posterior / posterior.sum(
            dim=-1, keepdim=True
        ).clamp_min(tiny)
        if not bool(torch.isfinite(posterior).all()):
            raise FloatingPointError(
                "Focus long-tail posterior produced NaN or Inf"
            )
        eligible_token = posterior.argmax(dim=-1)
        eligible_probability = posterior.gather(
            -1, eligible_token.unsqueeze(-1)
        ).squeeze(-1)
        eligible_base = visual[eligible_positions].gather(
            -1, eligible_token.unsqueeze(-1)
        ).squeeze(-1)
        eligible_entropy = -torch.xlogy(posterior, posterior).sum(dim=-1)
        top_two = torch.topk(
            posterior,
            k=2,
            dim=-1,
            largest=True,
            sorted=True,
        ).values

        selected_token[eligible_positions] = eligible_token
        base_confidence[eligible_positions] = eligible_base
        contrast_confidence[eligible_positions] = (
            apc_mass[eligible_positions].float() * eligible_probability
        )
        entropy[eligible_positions] = eligible_entropy
        margin[eligible_positions] = top_two[:, 0] - top_two[:, 1]
        tail_activation[eligible_positions] = eligible_activation
        conflict[eligible_positions] = eligible_conflict
        long_tail_mass[eligible_positions] = (
            tail_numerator / denominator.clamp_min(tiny)
        )
        current_weight[eligible_positions] = (
            (
                (1.0 - eligible_activation)
                + eligible_activation * survival[0]
            )
            / denominator.clamp_min(tiny)
        )
        effective_dwell_depth[eligible_positions] = depth

    outputs = (
        base_confidence,
        contrast_confidence,
        margin,
        tail_activation,
        conflict,
        relevance_precision,
        long_tail_mass,
        current_weight,
    )
    # Single-sync variant of eight ``bool(...).all()`` short-circuit checks:
    # every per-output finite reduction runs concurrently on the device and
    # the host waits exactly once at the aggregate ``.item()``.
    if not bool(
        torch.stack(
            [torch.isfinite(value[mask]).all() for value in outputs]
        ).all().item()
    ):
        raise FloatingPointError(
            "Focus long-tail decision statistics produced NaN or Inf"
        )
    return FocusLongTailObservation(
        current_frame=current_frame,
        eligible_mask=eligible_mask,
        dwell_counter=dwell_counter,
        selected_token=selected_token,
        base_confidence=base_confidence,
        contrast_confidence=contrast_confidence,
        entropy=entropy,
        margin=margin,
        tail_activation=tail_activation,
        exposure=exposure.float(),
        relevance_precision=relevance_precision,
        conflict=conflict,
        long_tail_mass=long_tail_mass,
        current_weight=current_weight,
        effective_dwell_depth=effective_dwell_depth,
        dwell_depth=len(recent_focus_frames),
    )


def counterfactual_exposure_weights(
    positions: torch.LongTensor,
    previous_commit_positions: torch.LongTensor,
    previous_commit_relevance: torch.Tensor,
    *,
    distance_scale: float,
    text_exposure_floor: float,
) -> torch.FloatTensor:
    """Measure new local context exposure after the preceding commit."""

    if positions.ndim != 1 or previous_commit_positions.ndim != 1:
        raise ValueError("Counterfactual exposure positions must be one-dimensional")
    if previous_commit_relevance.ndim != 1:
        raise ValueError(
            "Counterfactual commit relevance must be one-dimensional"
        )
    if previous_commit_positions.numel() != previous_commit_relevance.numel():
        raise ValueError(
            "Counterfactual commit positions and relevance must have equal lengths"
        )
    if positions.device != previous_commit_positions.device:
        raise ValueError(
            "Counterfactual exposure positions must share a device"
        )
    if previous_commit_relevance.device != positions.device:
        raise ValueError(
            "Counterfactual relevance must share the positions device"
        )
    if not math.isfinite(float(distance_scale)) or distance_scale <= 0.0:
        raise ValueError("distance_scale must be finite and positive")
    if not 0.0 <= float(text_exposure_floor) <= 1.0:
        raise ValueError("text_exposure_floor must be in [0, 1]")
    if not bool(torch.isfinite(previous_commit_relevance).all()):
        raise FloatingPointError(
            "Counterfactual commit relevance contains NaN or Inf"
        )
    if previous_commit_positions.numel() == 0:
        return torch.zeros(
            positions.numel(),
            dtype=torch.float32,
            device=positions.device,
        )

    relevance = previous_commit_relevance.float().clamp(0.0, 1.0)
    distance = (
        positions[:, None].long() - previous_commit_positions[None, :].long()
    ).abs().float()
    influence = torch.exp(-distance / float(distance_scale)) * (
        float(text_exposure_floor)
        + (1.0 - float(text_exposure_floor)) * relevance[None, :]
    )
    return (1.0 - torch.exp(-influence.sum(dim=-1))).clamp(0.0, 1.0)


def _counterfactual_evidence_summary(
    evidence_sum: torch.Tensor,
    evidence_sq_sum: torch.Tensor,
    exposure_sum: torch.Tensor,
    exposure_sq_sum: torch.Tensor,
    *,
    lower_bound_scale: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Return mean evidence, conservative lower bound, and effective count."""

    if not math.isfinite(float(lower_bound_scale)) or lower_bound_scale < 0.0:
        raise ValueError("lower_bound_scale must be finite and non-negative")
    epsilon = torch.finfo(torch.float32).eps
    sum_weight = exposure_sum.float().clamp_min(0.0)
    sum_weight_sq = exposure_sq_sum.float().clamp_min(0.0)
    active = sum_weight > epsilon
    mean = torch.where(
        active,
        evidence_sum.float() / sum_weight.clamp_min(epsilon),
        torch.zeros_like(sum_weight),
    )
    mean_square = torch.where(
        active,
        evidence_sq_sum.float() / sum_weight.clamp_min(epsilon),
        torch.zeros_like(sum_weight),
    )
    variance = (mean_square - mean.square()).clamp_min(0.0)
    effective_observations = torch.where(
        sum_weight_sq > epsilon,
        sum_weight.square() / sum_weight_sq,
        torch.zeros_like(sum_weight),
    )
    standard_error = torch.sqrt(
        variance / effective_observations.clamp_min(1.0)
    )
    lower_bound = torch.where(
        active,
        mean - float(lower_bound_scale) * standard_error,
        torch.zeros_like(mean),
    )
    if not bool(
        torch.isfinite(mean).all()
        and torch.isfinite(lower_bound).all()
        and torch.isfinite(effective_observations).all()
    ):
        raise FloatingPointError(
            "Counterfactual evidence summary produced NaN or Inf"
        )
    return mean, lower_bound, effective_observations


def observe_counterfactual_evidence(
    history: Optional[CounterfactualEvidenceHistory],
    *,
    candidate_token: int,
    current_evidence: torch.Tensor,
    observation_weight: torch.Tensor,
    context_version: int,
    flip_decay: float,
    lower_bound_scale: float,
) -> CounterfactualEvidenceObservation:
    """Update one MASK's candidate-aligned visual evidence trajectory."""

    if candidate_token < 0:
        raise ValueError("candidate_token must be non-negative")
    if context_version < 0:
        raise ValueError("context_version must be non-negative")
    if not 0.0 <= float(flip_decay) <= 1.0:
        raise ValueError("flip_decay must be in [0, 1]")
    evidence = current_evidence.reshape(()).float()
    weight = observation_weight.reshape(()).float().clamp(0.0, 1.0)
    if not bool(torch.isfinite(evidence)) or not bool(torch.isfinite(weight)):
        raise FloatingPointError(
            "Counterfactual evidence and observation weight must be finite"
        )

    if history is not None:
        if history.last_observed_context_version > context_version:
            raise ValueError(
                "Counterfactual history context version is newer than current"
            )
        if history.last_observed_context_version == context_version:
            mean, lower_bound, effective_observations = (
                _counterfactual_evidence_summary(
                    history.evidence_sum,
                    history.evidence_sq_sum,
                    history.exposure_sum,
                    history.exposure_sq_sum,
                    lower_bound_scale=lower_bound_scale,
                )
            )
            return CounterfactualEvidenceObservation(
                next_history=history,
                current_evidence=evidence,
                observation_weight=weight,
                mean_evidence=mean,
                evidence_lower_bound=lower_bound,
                effective_exposure=history.exposure_sum,
                effective_observations=effective_observations,
                candidate_flipped=False,
                updated=False,
            )

    candidate_flipped = (
        history is not None and int(candidate_token) != history.candidate_token
    )
    if history is None:
        prior_evidence_sum = torch.zeros_like(evidence)
        prior_evidence_sq_sum = torch.zeros_like(evidence)
        prior_exposure_sum = torch.zeros_like(weight)
        prior_exposure_sq_sum = torch.zeros_like(weight)
        candidate_age = 1
        flip_count = 0
    elif candidate_flipped:
        decay = float(flip_decay)
        prior_evidence_sum = history.evidence_sum * decay
        prior_evidence_sq_sum = history.evidence_sq_sum * decay
        prior_exposure_sum = history.exposure_sum * decay
        prior_exposure_sq_sum = history.exposure_sq_sum * (decay * decay)
        candidate_age = 1
        flip_count = history.flip_count + 1
    else:
        prior_evidence_sum = history.evidence_sum
        prior_evidence_sq_sum = history.evidence_sq_sum
        prior_exposure_sum = history.exposure_sum
        prior_exposure_sq_sum = history.exposure_sq_sum
        candidate_age = history.candidate_age + 1
        flip_count = history.flip_count

    next_history = CounterfactualEvidenceHistory(
        candidate_token=int(candidate_token),
        candidate_age=candidate_age,
        evidence_sum=(
            prior_evidence_sum + weight * evidence
        ).detach().clone(),
        evidence_sq_sum=(
            prior_evidence_sq_sum + weight * evidence.square()
        ).detach().clone(),
        exposure_sum=(prior_exposure_sum + weight).detach().clone(),
        exposure_sq_sum=(
            prior_exposure_sq_sum + weight.square()
        ).detach().clone(),
        flip_count=flip_count,
        last_observed_context_version=int(context_version),
    )
    mean, lower_bound, effective_observations = (
        _counterfactual_evidence_summary(
            next_history.evidence_sum,
            next_history.evidence_sq_sum,
            next_history.exposure_sum,
            next_history.exposure_sq_sum,
            lower_bound_scale=lower_bound_scale,
        )
    )
    return CounterfactualEvidenceObservation(
        next_history=next_history,
        current_evidence=evidence,
        observation_weight=weight,
        mean_evidence=mean,
        evidence_lower_bound=lower_bound,
        effective_exposure=next_history.exposure_sum,
        effective_observations=effective_observations,
        candidate_flipped=candidate_flipped,
        updated=True,
    )


def _counterfactual_history_scalar(value: torch.Tensor) -> float:
    """Read a scalar state value without assuming its storage device."""

    if value.numel() != 1:
        raise ValueError("Counterfactual history values must be scalar")
    result = float(value.item())
    if not math.isfinite(result):
        raise FloatingPointError(
            "Counterfactual history contains a non-finite scalar"
        )
    return result


def observe_counterfactual_evidence_batch(
    history: Dict[int, CounterfactualEvidenceHistory],
    positions: torch.LongTensor,
    candidate_tokens: torch.LongTensor,
    current_evidence: torch.Tensor,
    observation_weights: torch.Tensor,
    *,
    context_version: int,
    flip_decay: float,
    lower_bound_scale: float,
) -> CounterfactualEvidenceBatchObservation:
    """Update all unresolved MASK trajectories with one host/device exchange.

    The persistent per-MASK dictionary remains intentionally sparse and
    candidate-aligned. Scalar arithmetic runs on host-resident state so the
    decoder does not launch and synchronize hundreds of tiny CUDA operations
    per diffusion step.
    """

    expected_shape = positions.shape
    if (
        positions.ndim != 1
        or candidate_tokens.shape != expected_shape
        or current_evidence.shape != expected_shape
        or observation_weights.shape != expected_shape
    ):
        raise ValueError(
            "Batched counterfactual inputs must have equal one-dimensional "
            "shapes"
        )
    if positions.dtype != torch.long or candidate_tokens.dtype != torch.long:
        raise TypeError(
            "Counterfactual positions and candidate tokens must be long"
        )
    if (
        candidate_tokens.device != positions.device
        or current_evidence.device != positions.device
        or observation_weights.device != positions.device
    ):
        raise ValueError(
            "Batched counterfactual inputs must share a device"
        )
    if context_version < 0:
        raise ValueError("context_version must be non-negative")
    if not 0.0 <= float(flip_decay) <= 1.0:
        raise ValueError("flip_decay must be in [0, 1]")
    if not math.isfinite(float(lower_bound_scale)) or lower_bound_scale < 0.0:
        raise ValueError("lower_bound_scale must be finite and non-negative")
    if bool((candidate_tokens < 0).any()):
        raise ValueError("candidate_tokens must be non-negative")
    if not bool(torch.isfinite(current_evidence).all()):
        raise FloatingPointError(
            "Batched counterfactual evidence contains NaN or Inf"
        )
    if not bool(torch.isfinite(observation_weights).all()):
        raise FloatingPointError(
            "Batched counterfactual observation weights contain NaN or Inf"
        )

    # A single compact transfer avoids per-MASK CUDA synchronizations. Token
    # IDs are below 2**24 in the MMaDA vocabulary and remain exact in float32.
    payload = torch.stack(
        (
            positions.float(),
            candidate_tokens.float(),
            current_evidence.float(),
            observation_weights.float().clamp(0.0, 1.0),
        ),
        dim=-1,
    ).detach().cpu().tolist()
    next_history: Dict[int, CounterfactualEvidenceHistory] = {}
    lower_bounds = []
    exposures = []
    ages = []
    flipped = []
    updated_count = 0
    epsilon = float(torch.finfo(torch.float32).eps)
    decay = float(flip_decay)

    for position_value, candidate_value, evidence, weight in payload:
        position = int(position_value)
        candidate = int(candidate_value)
        evidence = float(evidence)
        weight = float(weight)
        previous = history.get(position)
        if (
            previous is not None
            and previous.last_observed_context_version > context_version
        ):
            raise ValueError(
                "Counterfactual history context version is newer than current"
            )

        if (
            previous is not None
            and previous.last_observed_context_version == context_version
        ):
            evidence_sum = _counterfactual_history_scalar(
                previous.evidence_sum
            )
            evidence_sq_sum = _counterfactual_history_scalar(
                previous.evidence_sq_sum
            )
            exposure_sum = _counterfactual_history_scalar(
                previous.exposure_sum
            )
            exposure_sq_sum = _counterfactual_history_scalar(
                previous.exposure_sq_sum
            )
            next_state = previous
            did_flip = False
        else:
            did_flip = previous is not None and candidate != previous.candidate_token
            if previous is None:
                prior_evidence_sum = 0.0
                prior_evidence_sq_sum = 0.0
                prior_exposure_sum = 0.0
                prior_exposure_sq_sum = 0.0
                age = 1
                flip_count = 0
                # The first value creates a candidate-specific baseline.
                weight = 1.0
            elif did_flip:
                prior_evidence_sum = (
                    _counterfactual_history_scalar(previous.evidence_sum)
                    * decay
                )
                prior_evidence_sq_sum = (
                    _counterfactual_history_scalar(previous.evidence_sq_sum)
                    * decay
                )
                prior_exposure_sum = (
                    _counterfactual_history_scalar(previous.exposure_sum)
                    * decay
                )
                prior_exposure_sq_sum = (
                    _counterfactual_history_scalar(previous.exposure_sq_sum)
                    * decay
                    * decay
                )
                age = 1
                flip_count = previous.flip_count + 1
            else:
                prior_evidence_sum = _counterfactual_history_scalar(
                    previous.evidence_sum
                )
                prior_evidence_sq_sum = _counterfactual_history_scalar(
                    previous.evidence_sq_sum
                )
                prior_exposure_sum = _counterfactual_history_scalar(
                    previous.exposure_sum
                )
                prior_exposure_sq_sum = _counterfactual_history_scalar(
                    previous.exposure_sq_sum
                )
                age = previous.candidate_age + 1
                flip_count = previous.flip_count

            evidence_sum = prior_evidence_sum + weight * evidence
            evidence_sq_sum = (
                prior_evidence_sq_sum + weight * evidence * evidence
            )
            exposure_sum = prior_exposure_sum + weight
            exposure_sq_sum = prior_exposure_sq_sum + weight * weight
            next_state = CounterfactualEvidenceHistory(
                candidate_token=candidate,
                candidate_age=age,
                evidence_sum=torch.tensor(evidence_sum, dtype=torch.float32),
                evidence_sq_sum=torch.tensor(
                    evidence_sq_sum, dtype=torch.float32
                ),
                exposure_sum=torch.tensor(exposure_sum, dtype=torch.float32),
                exposure_sq_sum=torch.tensor(
                    exposure_sq_sum, dtype=torch.float32
                ),
                flip_count=flip_count,
                last_observed_context_version=int(context_version),
            )
            updated_count += 1

        if exposure_sum > epsilon:
            mean = evidence_sum / exposure_sum
            variance = max(
                evidence_sq_sum / exposure_sum - mean * mean,
                0.0,
            )
            effective_observations = (
                exposure_sum * exposure_sum
                / max(exposure_sq_sum, epsilon)
            )
            lower_bound = mean - float(lower_bound_scale) * math.sqrt(
                variance / max(effective_observations, 1.0)
            )
        else:
            lower_bound = 0.0
        if not math.isfinite(lower_bound):
            raise FloatingPointError(
                "Batched counterfactual lower bound is non-finite"
            )
        next_history[position] = next_state
        lower_bounds.append(lower_bound)
        exposures.append(exposure_sum)
        ages.append(next_state.candidate_age)
        flipped.append(did_flip)

    device = current_evidence.device
    return CounterfactualEvidenceBatchObservation(
        next_history=next_history,
        evidence_lower_bound=torch.tensor(
            lower_bounds, dtype=torch.float32, device=device
        ),
        effective_exposure=torch.tensor(
            exposures, dtype=torch.float32, device=device
        ),
        candidate_age=torch.tensor(ages, dtype=torch.long, device=device),
        candidate_flipped=torch.tensor(
            flipped, dtype=torch.bool, device=device
        ),
        updated_count=updated_count,
    )


def _trajectory_summary(
    trajectory: UnifiedCandidateTrajectory,
    *,
    gain_uncertainty_scale: float,
) -> Tuple[float, float, float, float, float, float]:
    """Return semantic mean/std, gain mean/bounds, and effective samples."""

    if gain_uncertainty_scale < 0.0 or not math.isfinite(
        float(gain_uncertainty_scale)
    ):
        raise ValueError(
            "gain_uncertainty_scale must be finite and non-negative"
        )
    epsilon = float(torch.finfo(torch.float32).eps)
    semantic_weight = max(trajectory.semantic_weight_sum, epsilon)
    semantic_mean = trajectory.semantic_sum / semantic_weight
    semantic_variance = max(
        trajectory.semantic_sq_sum / semantic_weight
        - semantic_mean * semantic_mean,
        0.0,
    )
    semantic_std = math.sqrt(semantic_variance)

    exposure = max(trajectory.exposure_sum, 0.0)
    if exposure <= epsilon:
        return semantic_mean, semantic_std, 0.0, 0.0, 0.0, 0.0

    gain_mean = trajectory.gain_sum / exposure
    gain_variance = max(
        trajectory.gain_sq_sum / exposure - gain_mean * gain_mean,
        0.0,
    )
    effective_observations = (
        exposure * exposure
        / max(trajectory.exposure_sq_sum, epsilon)
    )
    gain_standard_error = math.sqrt(
        gain_variance / max(effective_observations, 1.0)
    )
    shrinkage = float(gain_uncertainty_scale) * gain_standard_error
    return (
        semantic_mean,
        semantic_std,
        gain_mean,
        gain_mean - shrinkage,
        gain_mean + shrinkage,
        effective_observations,
    )


def _decay_unified_trajectory(
    trajectory: UnifiedCandidateTrajectory,
    *,
    factor: float,
) -> UnifiedCandidateTrajectory:
    """Age a dormant candidate without changing its token identity."""

    if not math.isfinite(float(factor)) or not 0.0 <= factor <= 1.0:
        raise ValueError("trajectory decay factor must be in [0, 1]")
    return UnifiedCandidateTrajectory(
        token_id=trajectory.token_id,
        semantic_sum=trajectory.semantic_sum * factor,
        semantic_sq_sum=trajectory.semantic_sq_sum * factor,
        semantic_weight_sum=trajectory.semantic_weight_sum * factor,
        semantic_weight_sq_sum=trajectory.semantic_weight_sq_sum
        * factor
        * factor,
        gain_sum=trajectory.gain_sum * factor,
        gain_sq_sum=trajectory.gain_sq_sum * factor,
        exposure_sum=trajectory.exposure_sum * factor,
        exposure_sq_sum=trajectory.exposure_sq_sum * factor * factor,
        candidate_age=trajectory.candidate_age,
        last_seen_context_version=trajectory.last_seen_context_version,
    )


def observe_unified_trajectory_batch(
    history: Dict[int, Dict[int, UnifiedCandidateTrajectory]],
    positions: torch.LongTensor,
    candidate_tokens: torch.LongTensor,
    semantic_log_probs: torch.Tensor,
    gains: torch.Tensor,
    candidate_in_apc: torch.BoolTensor,
    observation_weights: torch.Tensor,
    *,
    context_version: int,
    stale_decay: float,
    history_limit: int,
    gain_uncertainty_scale: float,
) -> UnifiedTrajectoryBatchObservation:
    """Update sparse candidate trajectories under each new context.

    Semantic log-probability is recursively updated on every observation.
    Counterfactual gain is updated only after genuine context exposure. The
    first APC appearance therefore creates a semantic baseline without
    manufacturing visual evidence.
    """

    expected_shape = candidate_tokens.shape
    if (
        positions.ndim != 1
        or candidate_tokens.ndim != 2
        or candidate_tokens.shape[0] != positions.numel()
        or semantic_log_probs.shape != expected_shape
        or gains.shape != expected_shape
        or candidate_in_apc.shape != expected_shape
        or observation_weights.shape != positions.shape
    ):
        raise ValueError(
            "Unified trajectory inputs must be [positions, top_k] with "
            "one exposure weight per position"
        )
    if candidate_tokens.dtype != torch.long or positions.dtype != torch.long:
        raise TypeError("Unified trajectory positions and tokens must be long")
    if (
        candidate_tokens.device != positions.device
        or semantic_log_probs.device != positions.device
        or gains.device != positions.device
        or candidate_in_apc.device != positions.device
        or observation_weights.device != positions.device
    ):
        raise ValueError("Unified trajectory inputs must share a device")
    if candidate_tokens.shape[1] < 2:
        raise ValueError("Unified trajectory support must contain at least two tokens")
    if candidate_in_apc.dtype != torch.bool:
        raise TypeError("candidate_in_apc must be boolean")
    if not bool(candidate_in_apc.any(dim=-1).all()):
        raise ValueError("Every trajectory position must contain an APC token")
    if context_version < 0:
        raise ValueError("context_version must be non-negative")
    if not math.isfinite(float(stale_decay)) or not 0.0 <= stale_decay <= 1.0:
        raise ValueError("stale_decay must be in [0, 1]")
    if history_limit < int(candidate_tokens.shape[1]):
        raise ValueError("history_limit cannot be smaller than candidate top-K")
    if bool((candidate_tokens < 0).any()):
        raise ValueError("Unified trajectory candidate tokens must be non-negative")
    if not bool(
        torch.isfinite(semantic_log_probs).all()
        and torch.isfinite(gains).all()
        and torch.isfinite(observation_weights).all()
    ):
        raise FloatingPointError(
            "Unified trajectory observations must be finite"
        )
    sorted_tokens = torch.sort(candidate_tokens, dim=-1).values
    if bool((sorted_tokens[:, 1:] == sorted_tokens[:, :-1]).any()):
        raise ValueError("Unified trajectory candidates must be unique per MASK")

    # One compact transfer preserves the batched CUDA behavior of the original
    # counterfactual path while keeping sparse dictionaries on the host.
    payload = torch.stack(
        (
            candidate_tokens.float(),
            semantic_log_probs.float(),
            gains.float(),
            candidate_in_apc.float(),
        ),
        dim=-1,
    ).detach().cpu().tolist()
    position_values = positions.detach().cpu().tolist()
    weight_values = (
        observation_weights.float().clamp(0.0, 1.0).detach().cpu().tolist()
    )

    next_history: Dict[int, Dict[int, UnifiedCandidateTrajectory]] = {}
    semantic_means = []
    semantic_stds = []
    gain_means = []
    gain_lowers = []
    gain_uppers = []
    exposures = []
    effective_observations = []
    ages = []
    baseline_only = []
    updated_count = 0
    epsilon = float(torch.finfo(torch.float32).eps)

    for position_value, candidate_rows, weight_value in zip(
        position_values, payload, weight_values
    ):
        position = int(position_value)
        weight = float(weight_value)
        previous_by_token = history.get(position, {})
        current_tokens = {
            int(row[0]) for row in candidate_rows if bool(row[3])
        }
        updated_by_token: Dict[int, UnifiedCandidateTrajectory] = {}
        row_semantic_means = []
        row_semantic_stds = []
        row_gain_means = []
        row_gain_lowers = []
        row_gain_uppers = []
        row_exposures = []
        row_effective_observations = []
        row_ages = []
        row_baseline_only = []

        for (
            token_value,
            log_semantic_value,
            gain_value,
            in_apc_value,
        ) in candidate_rows:
            token_id = int(token_value)
            log_semantic = float(log_semantic_value)
            gain = float(gain_value)
            in_apc = bool(in_apc_value)
            previous = previous_by_token.get(token_id) if in_apc else None

            if previous is None:
                # A candidate's first APC appearance establishes only its
                # semantic state. Counterfactual precision requires a later
                # context-changing observation of the same candidate.
                trajectory = UnifiedCandidateTrajectory(
                    token_id=token_id,
                    semantic_sum=log_semantic,
                    semantic_sq_sum=log_semantic * log_semantic,
                    semantic_weight_sum=1.0,
                    semantic_weight_sq_sum=1.0,
                    gain_sum=0.0,
                    gain_sq_sum=0.0,
                    exposure_sum=0.0,
                    exposure_sq_sum=0.0,
                    candidate_age=1,
                    last_seen_context_version=int(context_version),
                )
            elif previous.last_seen_context_version == context_version:
                trajectory = previous
            else:
                if previous.last_seen_context_version > context_version:
                    raise ValueError(
                        "Unified trajectory context version is newer than "
                        "the current snapshot"
                    )
                context_gap = max(
                    1,
                    int(context_version)
                    - previous.last_seen_context_version,
                )
                trajectory = _decay_unified_trajectory(
                    previous,
                    factor=float(stale_decay) ** context_gap,
                )
                gain_weight = weight if weight > epsilon else 0.0
                trajectory = UnifiedCandidateTrajectory(
                    token_id=token_id,
                    semantic_sum=trajectory.semantic_sum + log_semantic,
                    semantic_sq_sum=(
                        trajectory.semantic_sq_sum
                        + log_semantic * log_semantic
                    ),
                    semantic_weight_sum=(
                        trajectory.semantic_weight_sum + 1.0
                    ),
                    semantic_weight_sq_sum=(
                        trajectory.semantic_weight_sq_sum + 1.0
                    ),
                    gain_sum=trajectory.gain_sum + gain_weight * gain,
                    gain_sq_sum=(
                        trajectory.gain_sq_sum
                        + gain_weight * gain * gain
                    ),
                    exposure_sum=(
                        trajectory.exposure_sum + gain_weight
                    ),
                    exposure_sq_sum=(
                        trajectory.exposure_sq_sum
                        + gain_weight * gain_weight
                    ),
                    candidate_age=trajectory.candidate_age + 1,
                    last_seen_context_version=int(context_version),
                )
                if gain_weight > 0.0:
                    updated_count += 1

            if in_apc:
                updated_by_token[token_id] = trajectory
            (
                semantic_mean,
                semantic_std,
                gain_mean,
                gain_lower,
                gain_upper,
                effective_count,
            ) = _trajectory_summary(
                trajectory,
                gain_uncertainty_scale=gain_uncertainty_scale,
            )
            row_semantic_means.append(semantic_mean)
            row_semantic_stds.append(semantic_std)
            row_gain_means.append(gain_mean)
            row_gain_lowers.append(gain_lower)
            row_gain_uppers.append(gain_upper)
            row_exposures.append(trajectory.exposure_sum)
            row_effective_observations.append(effective_count)
            row_ages.append(trajectory.candidate_age)
            row_baseline_only.append(trajectory.exposure_sum <= epsilon)

        # Preserve a bounded dormant support so candidates that briefly leave
        # top-K can re-enter with decayed, rather than erased, trajectories.
        retained = dict(updated_by_token)
        remaining = max(0, int(history_limit) - len(retained))
        dormant = sorted(
            (
                trajectory
                for token_id, trajectory in previous_by_token.items()
                if token_id not in current_tokens
            ),
            key=lambda value: (
                value.last_seen_context_version,
                value.candidate_age,
            ),
            reverse=True,
        )
        for trajectory in dormant[:remaining]:
            retained[trajectory.token_id] = trajectory
        next_history[position] = retained
        semantic_means.append(row_semantic_means)
        semantic_stds.append(row_semantic_stds)
        gain_means.append(row_gain_means)
        gain_lowers.append(row_gain_lowers)
        gain_uppers.append(row_gain_uppers)
        exposures.append(row_exposures)
        effective_observations.append(row_effective_observations)
        ages.append(row_ages)
        baseline_only.append(row_baseline_only)

    device = semantic_log_probs.device
    return UnifiedTrajectoryBatchObservation(
        next_history=next_history,
        semantic_mean=torch.tensor(
            semantic_means, dtype=torch.float32, device=device
        ),
        semantic_std=torch.tensor(
            semantic_stds, dtype=torch.float32, device=device
        ),
        gain_mean=torch.tensor(gain_means, dtype=torch.float32, device=device),
        gain_lower=torch.tensor(
            gain_lowers, dtype=torch.float32, device=device
        ),
        gain_upper=torch.tensor(
            gain_uppers, dtype=torch.float32, device=device
        ),
        effective_exposure=torch.tensor(
            exposures, dtype=torch.float32, device=device
        ),
        effective_observations=torch.tensor(
            effective_observations, dtype=torch.float32, device=device
        ),
        candidate_age=torch.tensor(ages, dtype=torch.long, device=device),
        baseline_only=torch.tensor(
            baseline_only, dtype=torch.bool, device=device
        ),
        updated_count=updated_count,
    )


def compute_unified_trajectory_posterior(
    candidate_tokens: torch.LongTensor,
    candidate_visual_probs: torch.Tensor,
    candidate_contrast_probs: torch.Tensor,
    candidate_in_apc: torch.BoolTensor,
    visual_relevance: torch.Tensor,
    observation: UnifiedTrajectoryBatchObservation,
    *,
    semantic_std_scale: float,
    visual_weight: float,
    adaptive_visual_relevance: bool,
    observation_scale: float,
    exposure_scale: float,
    relevance_scale: float,
    uncertainty_scale: float,
    opposed_threshold: float,
) -> UnifiedTrajectoryPosterior:
    """Fuse semantic history and exposure-calibrated visual residuals.

    For candidate ``c`` at MASK ``i``, the posterior energy is

    ``E(i,c) = mean(log q(c)) - kappa * std(log q(c))
                + lambda(i,c) * residual_gain(i,c)``.

    ``lambda`` is a continuous precision derived from counterfactual exposure,
    effective observations, relevance, and gain uncertainty. At zero exposure
    it is exactly zero, leaving a valid recursive semantic posterior rather
    than triggering a separate fallback decoder.
    """

    expected_shape = candidate_tokens.shape
    if (
        candidate_tokens.ndim != 2
        or candidate_visual_probs.shape != expected_shape
        or candidate_contrast_probs.shape != expected_shape
        or candidate_in_apc.shape != expected_shape
        or visual_relevance.shape != expected_shape[:1]
    ):
        raise ValueError(
            "Unified posterior inputs must share [positions, top_k] support"
        )
    observation_tensors = (
        observation.semantic_mean,
        observation.semantic_std,
        observation.gain_mean,
        observation.gain_lower,
        observation.gain_upper,
        observation.effective_exposure,
        observation.effective_observations,
    )
    if any(value.shape != expected_shape for value in observation_tensors):
        raise ValueError(
            "Unified trajectory observations must match candidate support"
        )
    for name, value in (
        ("semantic_std_scale", semantic_std_scale),
        ("visual_weight", visual_weight),
        ("observation_scale", observation_scale),
        ("exposure_scale", exposure_scale),
        ("relevance_scale", relevance_scale),
        ("uncertainty_scale", uncertainty_scale),
        ("opposed_threshold", opposed_threshold),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not bool(
        torch.isfinite(candidate_visual_probs).all()
        and torch.isfinite(candidate_contrast_probs).all()
        and torch.isfinite(visual_relevance).all()
        and all(torch.isfinite(value).all() for value in observation_tensors)
    ):
        raise FloatingPointError("Unified posterior inputs must be finite")
    if not bool(candidate_in_apc.any(dim=-1).all()):
        raise RuntimeError("Every unified trajectory support needs an APC token")

    gain_residual = torch.where(
        observation.gain_mean >= 0.0,
        observation.gain_lower.clamp_min(0.0),
        -(-observation.gain_upper).clamp_min(0.0),
    )
    exposure = observation.effective_exposure.float().clamp_min(0.0)
    effective_count = observation.effective_observations.float().clamp_min(
        0.0
    )
    exposure_precision = (
        (exposure > 0.0).float()
        if float(exposure_scale) == 0.0
        else -torch.expm1(-exposure / float(exposure_scale))
    )
    observation_precision = (
        (effective_count > 0.0).float()
        if float(observation_scale) == 0.0
        else effective_count / (
            effective_count + float(observation_scale)
        )
    )
    gain_uncertainty = 0.5 * (
        observation.gain_upper.float() - observation.gain_lower.float()
    ).clamp_min(0.0)
    uncertainty_precision = torch.exp(
        -float(uncertainty_scale) * gain_uncertainty
    )
    raw_relevance = visual_relevance.float().clamp(0.0, 1.0).unsqueeze(-1)
    relevance_precision = (
        (
            (raw_relevance > 0.0).float()
            if float(relevance_scale) == 0.0
            else -torch.expm1(
                -raw_relevance / float(relevance_scale)
            )
        )
        if adaptive_visual_relevance
        else torch.ones_like(raw_relevance)
    )
    visual_precision = (
        relevance_precision
        * exposure_precision
        * observation_precision
        * uncertainty_precision
    )
    candidate_visual_weight = float(visual_weight) * visual_precision
    visually_informed = candidate_visual_weight > 0.0
    candidate_score = (
        observation.semantic_mean.float()
        - float(semantic_std_scale) * observation.semantic_std.float()
        + candidate_visual_weight * gain_residual
    )
    candidate_score = candidate_score.masked_fill(
        ~candidate_in_apc,
        -torch.inf,
    )
    candidate_opposed = (
        (candidate_visual_weight > 0.0)
        & (observation.gain_upper <= -float(opposed_threshold))
    )
    candidate_posterior = torch.softmax(candidate_score, dim=-1)
    if not bool(torch.isfinite(candidate_posterior).all()):
        raise FloatingPointError(
            "Unified trajectory posterior produced NaN or Inf"
        )

    selected_index = candidate_posterior.argmax(dim=-1)
    selected_token = candidate_tokens.gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    selected_base_confidence = candidate_visual_probs.gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    support_mass = candidate_contrast_probs.float().masked_fill(
        ~candidate_in_apc,
        0.0,
    ).sum(dim=-1).clamp(0.0, 1.0)
    selected_confidence = support_mass * candidate_posterior.gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    selected_visually_informed = visually_informed.gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    selected_opposed = candidate_opposed.gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    selected_gain_lower = observation.gain_lower.gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    selected_gain_upper = observation.gain_upper.gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    selected_effective_exposure = observation.effective_exposure.gather(
        -1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    selected_effective_observations = (
        observation.effective_observations.gather(
            -1, selected_index.unsqueeze(-1)
        ).squeeze(-1)
    )
    selected_entropy = -torch.xlogy(
        candidate_posterior,
        candidate_posterior,
    ).sum(dim=-1)
    top_two = torch.topk(
        candidate_posterior,
        k=2,
        dim=-1,
        largest=True,
        sorted=True,
    ).values
    selected_margin = top_two[:, 0] - top_two[:, 1]
    return UnifiedTrajectoryPosterior(
        candidate_score=candidate_score,
        candidate_posterior=candidate_posterior,
        candidate_visually_informed=visually_informed,
        candidate_opposed=candidate_opposed,
        candidate_visual_weight=candidate_visual_weight,
        selected_index=selected_index,
        selected_token=selected_token,
        selected_base_confidence=selected_base_confidence,
        selected_confidence=selected_confidence,
        selected_entropy=selected_entropy,
        selected_margin=selected_margin,
        selected_visually_informed=selected_visually_informed,
        selected_opposed=selected_opposed,
        selected_gain_lower=selected_gain_lower,
        selected_gain_upper=selected_gain_upper,
        selected_effective_exposure=selected_effective_exposure,
        selected_effective_observations=selected_effective_observations,
    )


def observe_focus_dwell(
    focus_frames: Sequence[FocusFrame],
    contrast_distribution: torch.Tensor,
    visual_distribution: torch.Tensor,
    position_confidence: torch.Tensor,
    apc_mass: torch.Tensor,
    mask: torch.BoolTensor,
    *,
    dwell_depth: int,
    focus_capacity: int,
) -> FocusDwellObservation:
    """Marginalize contrast distributions over the dwell window at each MASK.

    This is the pure rectangular-kernel variant used both for its own ablation
    and as the strict fallback of :func:`observe_focus_longtail`. Each position
    contributes the average of its distributions over the current frame and
    the ``dwell_depth`` most-recent frames it has been observed in. Only the
    dwell counter decides eligibility; there is no explicit intersection of
    top-focus position sets.
    """

    if contrast_distribution.ndim != 2:
        raise ValueError("contrast_distribution must have shape [positions, vocab]")
    if visual_distribution.shape != contrast_distribution.shape:
        raise ValueError("visual and contrast distributions must have equal shapes")
    position_count = int(contrast_distribution.shape[0])
    expected_shape = (position_count,)
    if (
        position_confidence.shape != expected_shape
        or apc_mass.shape != expected_shape
        or mask.shape != expected_shape
    ):
        raise ValueError(
            "Focus dwell per-position inputs must match distribution positions"
        )
    if mask.dtype != torch.bool:
        raise TypeError("Focus dwell mask must be boolean")
    if dwell_depth < 1:
        raise ValueError("dwell_depth must be at least 1")
    if focus_capacity < 1:
        raise ValueError("focus_capacity must be at least 1")
    if not bool(torch.isfinite(contrast_distribution).all()):
        raise FloatingPointError(
            "Focus dwell contrast distribution contains NaN or Inf"
        )
    if not bool(torch.isfinite(visual_distribution).all()):
        raise FloatingPointError(
            "Focus dwell visual distribution contains NaN or Inf"
        )

    current_positions = _current_focus_positions(
        position_confidence, mask, focus_capacity
    )
    current_frame = make_focus_frame(
        current_positions,
        contrast_distribution[current_positions],
        position_count,
    )

    for frame in focus_frames:
        if frame.distributions.ndim != 2:
            raise ValueError("Focus frame distributions must be two-dimensional")
        if frame.distributions.shape[0] != frame.positions.numel():
            raise ValueError(
                "Focus frame position/distribution counts must match"
            )
        if frame.distributions.shape[1] != contrast_distribution.shape[1]:
            raise ValueError(
                "Focus frame vocabulary size changed between iterations"
            )
        if frame.positions.device != mask.device:
            raise ValueError(
                "Focus frames and current mask must share a device"
            )

    recent_focus_frames = list(focus_frames)[-int(dwell_depth) :]
    dwell_counter = compute_focus_dwell_counter(
        focus_frames,
        current_positions,
        total_positions=position_count,
    )
    # See ``observe_focus_longtail`` for the warmup rationale: the gate
    # equals ``1 + <stored frames>``, so the strict ``D >= dwell_depth + 1``
    # requirement kicks in only after the ring buffer has been filled.
    dwell_gate = 1 + len(recent_focus_frames)
    eligible_mask = dwell_counter >= dwell_gate

    current_token = contrast_distribution.argmax(dim=-1)
    base_confidence = visual_distribution.gather(
        -1, current_token.unsqueeze(-1)
    ).squeeze(-1)
    contrast_confidence = apc_mass * contrast_distribution.gather(
        -1, current_token.unsqueeze(-1)
    ).squeeze(-1)
    marginal_token = current_token.clone()
    marginal_entropy = torch.full_like(
        position_confidence.float(), torch.inf
    )

    eligible_positions = torch.nonzero(eligible_mask, as_tuple=True)[0]
    if eligible_positions.numel() > 0:
        # Accumulate in the same order ``observe_focus_longtail`` uses
        # (current, newest-stored-frame, ..., oldest-stored-frame). Sharing
        # the reduction order makes both paths bit-exact when the long-tail
        # activation is zero.
        marginalized = contrast_distribution[eligible_positions].float().clone()
        frame_count = 1
        for frame in reversed(recent_focus_frames):
            # Prefer the frame's precomputed inverse lookup (the common case
            # for frames created via :func:`make_focus_frame`); raw frames
            # fall back to the on-the-spot build path that mirrors the
            # pre-P0 reference behaviour with no extra allocation ceremony.
            lookup = frame.lookup
            if lookup is None or lookup.device != mask.device:
                lookup = _build_focus_lookup(frame.positions, position_count)
            frame_rows = lookup[eligible_positions]
            if bool((frame_rows < 0).any()):
                raise RuntimeError(
                    "Focus dwell window lost an eligible position"
                )
            marginalized = marginalized + frame.distributions[frame_rows].float()
            frame_count += 1
        marginalized = marginalized / float(frame_count)
        marginalized = marginalized / marginalized.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).tiny)
        eligible_tokens = marginalized.argmax(dim=-1)
        eligible_probs = marginalized.gather(
            -1, eligible_tokens.unsqueeze(-1)
        ).squeeze(-1)
        eligible_base = visual_distribution[eligible_positions].gather(
            -1, eligible_tokens.unsqueeze(-1)
        ).squeeze(-1)
        entropy = -torch.xlogy(marginalized, marginalized).sum(dim=-1)

        marginal_token[eligible_positions] = eligible_tokens
        marginal_entropy[eligible_positions] = entropy
        base_confidence[eligible_positions] = eligible_base
        contrast_confidence[eligible_positions] = (
            apc_mass[eligible_positions] * eligible_probs
        )

    return FocusDwellObservation(
        current_frame=current_frame,
        eligible_mask=eligible_mask,
        dwell_counter=dwell_counter,
        marginal_token=marginal_token,
        marginal_entropy=marginal_entropy,
        base_confidence=base_confidence,
        contrast_confidence=contrast_confidence,
        dwell_depth=len(recent_focus_frames),
    )


def _normalized_parts(
    token_ids: torch.LongTensor,
    probs: torch.Tensor,
    other_prob: torch.Tensor,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    token_ids = token_ids.reshape(-1).long()
    probs = probs.reshape(-1).float().clamp_min(0.0)
    other_prob = other_prob.reshape(()).float().clamp_min(0.0)
    if token_ids.numel() != probs.numel():
        raise ValueError("Sparse token_ids and probs must have equal lengths")
    if token_ids.numel() == 0:
        raise ValueError("A sparse distribution must retain at least one token")
    if torch.unique(token_ids).numel() != token_ids.numel():
        raise ValueError("Sparse token_ids must be unique")
    total = probs.sum() + other_prob
    if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
        raise FloatingPointError("Sparse distribution has invalid total mass")
    return token_ids, probs / total, other_prob / total


def sparse_distribution_from_dense(
    distribution: torch.Tensor, top_v_tokens: int
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compress a normalized dense distribution into top-V plus OTHER."""

    if distribution.ndim != 1:
        raise ValueError("Dense distribution must be one-dimensional")
    if top_v_tokens < 2:
        raise ValueError("top_v_tokens must be at least 2")
    dense = distribution.float().clamp_min(0.0)
    total = dense.sum()
    if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
        raise FloatingPointError("Dense distribution has invalid total mass")
    dense = dense / total
    top_v = min(int(top_v_tokens), int(dense.numel()))
    probs, token_ids = torch.topk(
        dense, k=top_v, largest=True, sorted=True
    )
    other_prob = (1.0 - probs.sum()).clamp(0.0, 1.0)
    return _normalized_parts(token_ids, probs, other_prob)


def _aligned_sparse_vectors(
    first_ids: torch.LongTensor,
    first_probs: torch.Tensor,
    first_other: torch.Tensor,
    second_ids: torch.LongTensor,
    second_probs: torch.Tensor,
    second_other: torch.Tensor,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    first_ids, first_probs, first_other = _normalized_parts(
        first_ids, first_probs, first_other
    )
    second_ids, second_probs, second_other = _normalized_parts(
        second_ids, second_probs, second_other
    )
    if first_ids.device != second_ids.device:
        raise ValueError("Sparse distributions must be on the same device")

    support = torch.unique(
        torch.cat((first_ids, second_ids)), sorted=True
    )
    first = torch.zeros(
        support.numel() + 1, dtype=torch.float32, device=support.device
    )
    second = torch.zeros_like(first)
    first[torch.searchsorted(support, first_ids)] = first_probs
    second[torch.searchsorted(support, second_ids)] = second_probs
    first[-1] = first_other
    second[-1] = second_other
    return support, first, second


def sparse_jsd(
    first_ids: torch.LongTensor,
    first_probs: torch.Tensor,
    first_other: torch.Tensor,
    second_ids: torch.LongTensor,
    second_probs: torch.Tensor,
    second_other: torch.Tensor,
) -> torch.FloatTensor:
    """JSD on the union of retained tokens and one shared OTHER bucket."""

    _, first, second = _aligned_sparse_vectors(
        first_ids,
        first_probs,
        first_other,
        second_ids,
        second_probs,
        second_other,
    )
    mixture = 0.5 * (first + second)

    def contribution(values: torch.Tensor) -> torch.Tensor:
        return torch.where(
            values > 0.0,
            values * (torch.log(values) - torch.log(mixture)),
            torch.zeros_like(values),
        ).sum()

    divergence = 0.5 * (contribution(first) + contribution(second))
    if not bool(torch.isfinite(divergence)):
        raise FloatingPointError("Sparse history JSD produced NaN or Inf")
    return divergence.clamp(0.0, math.log(2.0))


def _ema_history(
    history: SparseHistory,
    current_ids: torch.LongTensor,
    current_probs: torch.Tensor,
    current_other: torch.Tensor,
    *,
    context_version: int,
    ema_decay: float,
    top_v_tokens: int,
) -> SparseHistory:
    support, current, previous = _aligned_sparse_vectors(
        current_ids,
        current_probs,
        current_other,
        history.token_ids,
        history.probs,
        history.other_prob,
    )
    merged = float(ema_decay) * previous + (
        1.0 - float(ema_decay)
    ) * current
    explicit = merged[:-1]
    order = torch.argsort(explicit, descending=True, stable=True)
    keep_count = min(int(top_v_tokens), int(order.numel()))
    keep = order[:keep_count]
    dropped = order[keep_count:]
    kept_ids = support[keep]
    kept_probs = explicit[keep]
    other_prob = merged[-1] + explicit[dropped].sum()
    kept_ids, kept_probs, other_prob = _normalized_parts(
        kept_ids, kept_probs, other_prob
    )
    return SparseHistory(
        token_ids=kept_ids,
        probs=kept_probs,
        other_prob=other_prob,
        last_observed_context_version=int(context_version),
        last_top1_token=int(current_ids[0].item()),
        consecutive_top1_matches=(
            history.consecutive_top1_matches + 1
            if int(current_ids[0].item()) == history.last_top1_token
            else 0
        ),
    )


def observe_sparse_history(
    history: Optional[SparseHistory],
    current_ids: torch.LongTensor,
    current_probs: torch.Tensor,
    current_other: torch.Tensor,
    *,
    context_version: int,
    ema_decay: float,
    top_v_tokens: int,
) -> HistoryObservation:
    """Score against pre-update history and plan at most one versioned update."""

    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if top_v_tokens < 2:
        raise ValueError("top_v_tokens must be at least 2")
    current_ids, current_probs, current_other = _normalized_parts(
        current_ids, current_probs, current_other
    )

    if history is None:
        next_history = SparseHistory(
            token_ids=current_ids[:top_v_tokens].clone(),
            probs=current_probs[:top_v_tokens].clone(),
            other_prob=(
                current_other + current_probs[top_v_tokens:].sum()
            ).clone(),
            last_observed_context_version=int(context_version),
            last_top1_token=int(current_ids[0].item()),
            consecutive_top1_matches=0,
        )
        return HistoryObservation(
            stability=torch.ones(
                (), dtype=torch.float32, device=current_probs.device
            ),
            next_history=next_history,
            updated=True,
            consecutive_top1_matches=0,
        )

    if history.last_observed_context_version > context_version:
        raise ValueError(
            "History context version is newer than the current snapshot"
        )
    divergence = sparse_jsd(
        current_ids,
        current_probs,
        current_other,
        history.token_ids,
        history.probs,
        history.other_prob,
    )
    stability = (1.0 - divergence / math.log(2.0)).clamp(0.0, 1.0)
    if history.last_observed_context_version == context_version:
        return HistoryObservation(
            stability=stability,
            next_history=history,
            updated=False,
            consecutive_top1_matches=history.consecutive_top1_matches,
        )

    next_history = _ema_history(
        history,
        current_ids,
        current_probs,
        current_other,
        context_version=context_version,
        ema_decay=ema_decay,
        top_v_tokens=top_v_tokens,
    )
    return HistoryObservation(
        stability=stability,
        next_history=next_history,
        updated=True,
        consecutive_top1_matches=next_history.consecutive_top1_matches,
    )


def history_adjusted_reliability(
    contrast_confidence: torch.Tensor,
    visual_relevance: torch.Tensor,
    history_stability: torch.Tensor,
    *,
    penalty_scale: float = 1.0,
) -> torch.FloatTensor:
    """R = C_contrast * [1 - clamp(gamma * rho * (1 - T), 0, 1)]."""

    if not (
        contrast_confidence.shape
        == visual_relevance.shape
        == history_stability.shape
    ):
        raise ValueError("Reliability inputs must have identical shapes")
    if not math.isfinite(float(penalty_scale)) or penalty_scale < 0.0:
        raise ValueError("penalty_scale must be finite and non-negative")
    penalty = (
        float(penalty_scale)
        * visual_relevance.float()
        * (1.0 - history_stability.float())
    ).clamp(0.0, 1.0)
    reliability = contrast_confidence.float() * (
        1.0 - penalty
    )
    if not bool(torch.isfinite(reliability).all()):
        raise FloatingPointError("History reliability produced NaN or Inf")
    return reliability.clamp(0.0, 1.0)
