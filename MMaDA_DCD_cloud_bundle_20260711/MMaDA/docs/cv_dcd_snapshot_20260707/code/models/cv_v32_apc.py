"""CV-DCD v3.2 — Adaptive Plausibility Constraint (APC) + logit blending.

Pure math helpers for contrastive decoding with an APC candidate mask. All
functions here are stateless: no dependency on ``MMaDADecodeConfig``. The
higher-level dispatcher (``cv_v32_dispatcher``) is responsible for reading the
config and forwarding the right scalars.

Design references
-----------------
- Li et al., 2022, "Contrastive Decoding: Open-ended Text Generation as
  Optimization" — introduced adaptive plausibility with ``alpha``.
- O'Brien & Lewis, 2023 — refined ``alpha`` + ``beta`` recipe.
- DoLa, 2023 — layer-wise variant; uses the same alpha-cut.

APC mask
    valid(v) = base_logit(v) >= max_v base_logit(v) + log(alpha)

CD blending
    cd_logit(v) = base_logit(v) + beta * (base_logit(v) - drop_logit(v))
                = (1 + beta) * base_logit(v) - beta * drop_logit(v)

The v3.2 change vs v3.1: we return the **unmasked** blended logits alongside
the masked selection logits. Downstream confidence code must run softmax on
the unmasked tensor to avoid the ``v_valid_size == 1 -> softmax == 1.0``
false-confidence pathology.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def apc_plausible_mask(base_logits: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return the adaptive-plausibility mask over the vocabulary axis.

    Parameters
    ----------
    base_logits : Tensor
        Base-model logits, shape ``(..., V)``.
    alpha : float
        Plausibility factor in ``(0, 1]``. A token survives iff
        ``prob(token) >= alpha * max_prob``, i.e., ``logit(token) >= max_logit
        + log(alpha)``. Larger alpha = **stricter** mask (fewer survivors);
        alpha=1.0 keeps only the argmax; alpha=0.1 (v3.1 default) keeps any
        token with at least 10% of the max probability.

    Returns
    -------
    Tensor of bool with the same shape as ``base_logits``. ``True`` marks
    tokens that survive the APC cut.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    max_base = base_logits.amax(dim=-1, keepdim=True)
    return base_logits >= (max_base + math.log(alpha))


def blend_cd_logits(
    base_logits: torch.Tensor,
    drop_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return CD-blended logits ``base + beta * (base - drop)`` (no masking)."""
    return base_logits + beta * (base_logits - drop_logits)


def apply_cd_apc(
    base_logits: torch.Tensor,
    drop_logits: torch.Tensor,
    beta: float,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine APC mask + CD blend and return two tensors.

    Returns
    -------
    selection_logits : Tensor
        Blended logits with non-plausible tokens set to ``-inf``. Use this for
        the argmax / sampling step.
    blended_logits : Tensor
        The same blended logits **without** the ``-inf`` mask. Use this for
        confidence softmax to avoid false-high-confidence when only 1-2
        tokens survive APC.

    Notes
    -----
    - When ``beta == 0`` we short-circuit and return ``base_logits`` for both,
      matching the "CD off" fallback.
    - The APC mask is always computed from the **base** logits (not blended),
      as prescribed by Li et al. 2022. This anchors the plausible set to what
      the unperturbed model considers reasonable.
    """
    if beta == 0.0:
        return base_logits, base_logits

    blended = blend_cd_logits(base_logits, drop_logits, beta)
    plausible = apc_plausible_mask(base_logits, alpha)
    neg_inf = torch.finfo(blended.dtype).min
    selection = torch.where(plausible, blended, torch.full_like(blended, neg_inf))
    return selection, blended
