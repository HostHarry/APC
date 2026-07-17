"""Defer-only CV: visual-gain-based veto functions.

Four pure veto functions transform ``base_conf`` into ``eff_conf`` using the
visual gain of the base argmax token. They differ in how strictly visual
support is required:

Variant       Formula                                        Semantics
----------    -------------------------------------------    -----------------------------------
``hard``      ``eff = base_conf if gain >= tau else 0``      Positions with insufficient visual
                                                              support are forced to defer.
``mult``      ``eff = base_conf * sigmoid(beta*(gain-tau))`` Smooth version of ``hard``. NOTE:
                                                              zero-point pathology -- at gain=0
                                                              eff drops to base_conf/2. Kept for
                                                              backwards compatibility / ablation.
``min``       ``eff = min(base_conf, sigmoid(beta*gain))``   Two-sided consensus. Same zero-point
                                                              pathology as ``mult``.
``soft``      ``eff = base_conf * exp(-beta*max(tau-gain,0))``  Asymmetric penalty. gain >= tau =>
                                                                   no penalty; gain < tau =>
                                                                   smooth exponential decay. This
                                                                   is the recommended smooth
                                                                   variant (fixes ``mult``'s
                                                                   zero-point issue).

Gain formulation
----------------
The ``compute_visual_gain`` helper supports two definitions of ``gain``:

- ``logit`` (default, v3.2-compatible): ``base_logit[x] - drop_logit[x]``.
  The raw logit difference at the base argmax token. This is what v3.2's
  debug column ``visual_gain_at_x0`` actually stored. Median-absolute value
  for shuffle drop is ~0.5, wide enough for beta=1 sigmoid/exp to act.

- ``logprob``: ``log_softmax(base)[x] - log_softmax(drop)[x]``. The
  normalized log-likelihood ratio. Matches Li et al. (2022) CD paper. For
  shuffle drop this is much smaller (median ~0), because the global
  ``logsumexp`` difference cancels most of the raw logit diff. Use ONLY with
  ``text_only`` or ``mask`` drop where the two distributions differ globally.

Key property (verified by ``test_defer_only.py``):
    ``apply_veto_*`` NEVER exceeds ``base_conf``. Visual gain can only
    *suppress* the commit signal, never amplify it. This is the whole
    point of "defer-only" -- the argmax stays exactly where base_logits
    puts it, we just delay some positions.
"""
from __future__ import annotations

from enum import Enum

import torch


class VetoType(str, Enum):
    HARD = "hard"
    MULT = "mult"
    MIN = "min"
    SOFT = "soft"


class GainType(str, Enum):
    """Which formula ``compute_visual_gain`` uses."""

    LOGIT = "logit"     # base_logit[x] - drop_logit[x]   (v3.2 default)
    LOGPROB = "logprob" # log_softmax(base)[x] - log_softmax(drop)[x]   (CD paper)


def compute_visual_gain(
    base_logits: torch.Tensor,
    drop_logits: torch.Tensor,
    x0: torch.Tensor,
    causal_clip: float,
    gain_type: str = "logit",
) -> torch.Tensor:
    """Clipped gain at the base argmax token.

    Shapes:
        base_logits : (B, L, V)
        drop_logits : (B, L, V)
        x0          : (B, L) long
        output      : (B, L) float64, clipped to [-clip, +clip]

    Parameters
    ----------
    gain_type : str
        ``'logit'`` (default) -> ``base_logit[x] - drop_logit[x]``. Matches
        v3.2's stored ``visual_gain_at_x0`` column; magnitudes ~ [-1.5, +1.5]
        for shuffle drop.

        ``'logprob'`` -> ``log_softmax(base)[x] - log_softmax(drop)[x]``.
        The CD paper's log-likelihood ratio; magnitudes ~ [-0.05, +0.05]
        for shuffle (LSE largely cancels the raw diff). Use with ``text_only``
        drop for a stronger signal.

    Memory notes
    ------------
    A naive implementation would call ``base_logits.to(torch.float64)`` (a
    ``(B, L, V)`` allocation, ~1-2 GB for MMaDA at V=134656 with L=2048),
    which caused OOM on 47 GiB GPUs when the base decode already uses ~22 GiB
    of activations. We avoid this by:

    - Gathering per-token values in the original dtype first (result is
      ``(B, L)`` -- tiny), THEN promoting to float64.
    - For ``'logprob'`` using ``torch.logsumexp`` (a fused kernel that does
      NOT materialize a full exp tensor), so peak memory stays at
      ``O(B * L)`` instead of ``O(B * L * V * 8 bytes)``.
    """
    gt = str(gain_type).lower()

    # Gather in the input dtype (typically bfloat16 or float32) so the
    # intermediate is (B, L) rather than (B, L, V). Cast the small result up
    # to float64 for stable arithmetic.
    base_val = torch.gather(
        base_logits, dim=-1, index=x0.unsqueeze(-1)
    ).squeeze(-1).to(torch.float64)
    drop_val = torch.gather(
        drop_logits, dim=-1, index=x0.unsqueeze(-1)
    ).squeeze(-1).to(torch.float64)

    if gt == GainType.LOGIT.value:
        gain = base_val - drop_val
    elif gt == GainType.LOGPROB.value:
        # logsumexp is a fused CUDA kernel; it does NOT allocate a full
        # (B, L, V) exp tensor, so peak memory is O(B * L).
        base_lse = torch.logsumexp(base_logits, dim=-1).to(torch.float64)
        drop_lse = torch.logsumexp(drop_logits, dim=-1).to(torch.float64)
        gain = (base_val - base_lse) - (drop_val - drop_lse)
    else:
        raise ValueError(
            f"unknown gain_type={gain_type!r}; expected 'logit' | 'logprob'"
        )
    if causal_clip is not None and causal_clip > 0:
        gain = gain.clamp(min=-float(causal_clip), max=float(causal_clip))
    return gain


def apply_veto_hard(
    base_conf: torch.Tensor, gain: torch.Tensor, tau: float
) -> torch.Tensor:
    """``eff = base_conf where gain >= tau else 0``."""
    keep = gain >= tau
    return torch.where(keep, base_conf, torch.zeros_like(base_conf))


def apply_veto_mult(
    base_conf: torch.Tensor, gain: torch.Tensor, tau: float, beta: float
) -> torch.Tensor:
    """``eff = base_conf * sigmoid(beta * (gain - tau))``.

    NOTE: This variant halves ``base_conf`` at ``gain == tau`` because
    ``sigmoid(0) = 0.5``. Prefer ``apply_veto_soft`` for a smooth variant
    without this pathology. Retained for ablation / backwards compatibility.
    """
    scale = torch.sigmoid(beta * (gain - tau))
    return base_conf * scale


def apply_veto_min(
    base_conf: torch.Tensor, gain: torch.Tensor, beta: float
) -> torch.Tensor:
    """``eff = min(base_conf, sigmoid(beta * gain))``.

    NOTE: Same zero-point pathology as ``apply_veto_mult`` -- at ``gain == 0``
    the visual term is 0.5, capping ``eff`` at 0.5 regardless of
    ``base_conf``. Retained for ablation.
    """
    visual_conf = torch.sigmoid(beta * gain)
    return torch.minimum(base_conf, visual_conf)


def apply_veto_soft(
    base_conf: torch.Tensor, gain: torch.Tensor, tau: float, beta: float
) -> torch.Tensor:
    """``eff = base_conf * exp(-beta * clamp(tau - gain, min=0))``.

    Asymmetric variant:
        * gain >= tau => ``eff = base_conf``           (no penalty)
        * gain <  tau => ``eff = base_conf * exp(-beta*(tau - gain)) < base_conf``

    This is the smooth analogue of ``apply_veto_hard`` -- it agrees with hard
    on the sign of the effect (never penalize positive gain) but decays
    smoothly for negative gain, giving DCD's ``_select_transfer`` a
    graded confidence signal instead of an all-or-nothing zero.
    """
    neg = torch.clamp(tau - gain, min=0.0)
    penalty = torch.exp(-beta * neg)
    return base_conf * penalty


def apply_veto(
    base_conf: torch.Tensor,
    gain: torch.Tensor,
    veto_type: str,
    tau: float,
    beta: float,
) -> torch.Tensor:
    """Dispatcher; pure function returning ``[B, L]`` in ``[0, 1]``.

    Raises ValueError on unknown ``veto_type``.
    """
    vt = str(veto_type).lower()
    if vt == VetoType.HARD.value:
        return apply_veto_hard(base_conf, gain, tau)
    if vt == VetoType.MULT.value:
        return apply_veto_mult(base_conf, gain, tau, beta)
    if vt == VetoType.MIN.value:
        return apply_veto_min(base_conf, gain, beta)
    if vt == VetoType.SOFT.value:
        return apply_veto_soft(base_conf, gain, tau, beta)
    raise ValueError(
        f"unknown defer veto_type={veto_type!r}; "
        f"expected 'hard' | 'mult' | 'min' | 'soft'"
    )
