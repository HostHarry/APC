"""Defer-only CV: pick-transfer orchestration.

Public entry point ``pick_transfer_defer_only`` has the exact same signature
as ``_pick_transfer`` in ``mmada_decode.py``, so ``_pick_transfer_cv`` can
route to it when ``config.cv_mode == 'defer_only'``.

Design invariants (verified by tests/test_defer_only.py):
    (a) ``x0`` is ALWAYS ``argmax(base_logits)`` at masked positions
        (with optional Gumbel when ``temperature > 0``). No CD blending.
    (b) When ``drop_logits is None`` OR ``causal_lambda == 0``, this
        function is BIT-IDENTICAL to plain DCD's ``_pick_transfer``:
        - same Gumbel formula (identical RNG consumption),
        - same ``F.softmax(logits.to(float64))`` confidence formula,
        - same ``_select_transfer`` semantics.
        The strict-parity test ``test_lambda_zero_bit_identical_to_plain_dcd``
        asserts this holds for random inputs at every masked position.
    (c) ``visual_gain`` NEVER boosts confidence; only suppresses. Any
        commit that happens under defer_only would also have happened
        under DCD baseline given the same base_logits.
    (d) ``causal_lambda`` is INTERVENTION STRENGTH (v5+ semantic):
            eff_conf = (1 - lambda) * base_conf + lambda * veto_result
        - lambda = 0 -> no intervention (matches plain DCD)
        - lambda = 1 -> full veto (matches v1-v4 defer_only)
        - lambda in (0,1) -> partial veto, linear blend
        This decouples intervention FREQUENCY (controlled by tau) from
        intervention STRENGTH (controlled by lambda).

Why this direction exists
-------------------------
v3.2 Phase B showed CD's argmax modifications damage the LM's structural
priors (repetition suppression via masked-diffusion parallel decoding).
Defer-only keeps DCD's token selection intact and uses the visual signal
only to *delay* commits at positions where the image doesn't help --
letting subsequent diffusion refresh rounds re-evaluate them.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import veto as _veto


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Local copy of ``mmada_decode.add_gumbel_noise`` to avoid a circular import."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _select_transfer(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    decode_algo: str,
    decode_param: float,
    refresh_count: int,
) -> torch.Tensor:
    """Threshold / topk / factor selection (mirrors ``_pick_transfer`` semantics).

    Kept as a private local copy rather than importing from
    ``cv_v32_dispatcher`` so this module stays self-contained.
    """
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
    for batch_idx in range(confidence.shape[0]):
        masked_positions = mask_index[batch_idx].nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            continue
        if decode_algo == "threshold":
            chosen = masked_positions[
                confidence[batch_idx, masked_positions] >= float(decode_param)
            ]
            if chosen.numel() == 0:
                chosen = masked_positions[
                    torch.topk(confidence[batch_idx, masked_positions], k=1).indices
                ]
        elif decode_algo == "factor":
            k = max(1, math.ceil(masked_positions.numel() * float(decode_param)))
            rel = torch.topk(
                confidence[batch_idx, masked_positions],
                k=min(k, masked_positions.numel()),
            ).indices
            chosen = masked_positions[rel]
        else:
            k = (
                1
                if masked_positions.numel() == 1
                else max(1, masked_positions.numel() // max(1, refresh_count))
            )
            rel = torch.topk(
                confidence[batch_idx, masked_positions],
                k=min(k, masked_positions.numel()),
            ).indices
            chosen = masked_positions[rel]
        transfer_index[batch_idx, chosen] = True
    return transfer_index


def _log_defer_debug(
    debug_records: List[Dict[str, Any]],
    *,
    step_idx: int,
    batch_idx: int,
    positions: torch.Tensor,
    x0: torch.Tensor,
    base_conf: torch.Tensor,
    eff_conf: torch.Tensor,
    gain: Optional[torch.Tensor],
    veto_type: str,
    tau: float,
    beta: float,
    gain_type: str,
    causal_lambda: float,
    base_logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
) -> None:
    """Emit one debug record per committed position."""
    for pos_t in positions.tolist():
        pos = int(pos_t)
        token_id = int(x0[batch_idx, pos].item())
        base_conf_val = float(base_conf[batch_idx, pos].item())
        eff_conf_val = float(eff_conf[batch_idx, pos].item())

        gain_val = None
        if gain is not None:
            gain_val = float(gain[batch_idx, pos].item())

        # ``defer_active`` = did the veto actually reduce this position's conf?
        defer_active = bool(gain is not None and eff_conf_val < base_conf_val - 1e-6)

        debug_records.append({
            "step": step_idx,
            "batch": batch_idx,
            "position": pos,
            "x0_token": token_id,
            # base argmax IS x0 in defer-only, so argmax_changed is always False.
            "base_argmax": token_id,
            "argmax_changed": False,
            # populate the v3.2 schema fields so downstream analysis tools work
            "raw_conf": eff_conf_val,          # what DCD threshold saw
            "cv_score": eff_conf_val,          # kept identical for schema parity
            "raw_conf_base": base_conf_val,
            "raw_conf_effective": eff_conf_val,
            "conf_from_base": base_conf_val,
            "conf_from_blended": None,         # no blended logits in defer-only
            "conf_from_selection_v31bug": None,
            "conf_source_used": f"defer_{veto_type}",
            "gate_active": defer_active,       # reused slot: "was defer engaged?"
            "v_valid_size": None,              # not applicable
            "visual_gain_at_x0": gain_val,
            # defer-only-specific fields
            "defer_veto_type": veto_type,
            "defer_tau": float(tau),
            "defer_beta": float(beta),
            "defer_gain_type": gain_type,
            "defer_lambda": float(causal_lambda),
            "defer_base_conf": base_conf_val,
            "defer_eff_conf": eff_conf_val,
            "defer_gain": gain_val,
            "defer_active": defer_active,
        })


def pick_transfer_defer_only(
    base_logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
    config,
    mask_index: torch.Tensor,
    current_tokens: torch.Tensor,
    debug_records: Optional[List[Dict[str, Any]]] = None,
    step_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Defer-only pick-transfer.

    Pipeline:
        1. x0     = argmax(base_logits) [+ optional Gumbel]
        2. base_c = softmax(base_logits)[x0]
        3. if drop_logits is available AND causal_lambda > 0:
             gain     = clamp(compute_visual_gain(base, drop, x0, gain_type), +/- clip)
             veto_c   = apply_veto(base_c, gain, veto_type, tau, beta)
             eff_c    = (1 - lambda) * base_c + lambda * veto_c   # linear blend
           else:
             eff_c    = base_c            # equivalent to baseline DCD
        4. transfer_index = _select_transfer(eff_c, mask_index, ...)

    Config fields read:
        defer_veto_type : 'hard' | 'mult' | 'min' | 'soft'
        defer_tau       : gain threshold (controls WHICH positions get intervention)
        defer_beta      : penalty steepness (for soft/mult/min variants)
        defer_gain_type : 'logit' (v3.2-compatible) | 'logprob' (CD paper)
        causal_clip     : abs bound on gain
        causal_lambda   : intervention STRENGTH in [0, 1]
                          - 0 -> no veto (== plain DCD; sanity)
                          - 1 -> full veto (== v1-v4 defer_only)
                          - (0, 1) -> partial veto (linear blend)
    """
    veto_type = str(getattr(config, "defer_veto_type", "mult")).lower()
    tau = float(getattr(config, "defer_tau", 0.0))
    beta_sig = float(getattr(config, "defer_beta", 1.0))
    gain_type = str(getattr(config, "defer_gain_type", "logit")).lower()
    causal_clip = float(getattr(config, "causal_clip", 4.0))
    temperature = float(config.temperature)
    causal_lambda = float(getattr(config, "causal_lambda", 0.0))

    # (a) x0 from base_logits (with optional Gumbel).
    if temperature != 0.0:
        base_scored = _add_gumbel_noise(base_logits, temperature=temperature)
        x0 = torch.argmax(base_scored, dim=-1)
    else:
        x0 = torch.argmax(base_logits, dim=-1)

    # (b) base_conf at the (now-committed) argmax token.
    # CRITICAL: this MUST match ``mmada_decode._confidence_from_logits`` bit-
    # for-bit so that ``causal_lambda == 0`` is a true byte-identical sanity
    # for plain DCD (``cv_mode == 'off'``). Prior versions used a memory-
    # efficient ``logsumexp`` variant, but that computes the LSE in the input
    # dtype (bfloat16) before casting to float64 -- ~2^-7 precision loss
    # relative to plain DCD's ``F.softmax(logits.to(float64))``, enough to
    # occasionally flip a conf across the 0.9 threshold and diverge outputs.
    # For dual-cache decode (block-sized L <= block_size) the fp64 softmax
    # materialization is small (tens of MB) and safe.
    probs = F.softmax(base_logits.to(torch.float64), dim=-1)
    base_conf = torch.gather(
        probs, dim=-1, index=x0.unsqueeze(-1)
    ).squeeze(-1)
    del probs

    # (c) veto with intervention-strength blend (v5 semantic).
    veto_active = drop_logits is not None and causal_lambda > 0.0
    if veto_active:
        gain = _veto.compute_visual_gain(
            base_logits, drop_logits, x0, causal_clip, gain_type=gain_type
        )
        veto_conf = _veto.apply_veto(base_conf, gain, veto_type, tau, beta_sig)
        # Linear blend: causal_lambda in [0,1] controls intervention strength.
        # lambda=0 -> eff_conf = base_conf (identity); lambda=1 -> full veto.
        eff_conf = (1.0 - causal_lambda) * base_conf + causal_lambda * veto_conf
    else:
        gain = None
        eff_conf = base_conf

    # (d) mask non-masked positions out of consideration (identical to _pick_transfer).
    x0 = torch.where(mask_index, x0, current_tokens)
    confidence = torch.where(mask_index, eff_conf, torch.full_like(eff_conf, -np.inf))

    transfer_index = _select_transfer(
        confidence, mask_index,
        decode_algo=config.decode_algo,
        decode_param=float(config.decode_param),
        refresh_count=int(config.refresh_count),
    )

    # (e) Debug logging.
    if debug_records is not None and config.return_debug:
        for batch_idx in range(transfer_index.shape[0]):
            committed = transfer_index[batch_idx].nonzero(as_tuple=True)[0]
            if committed.numel() == 0:
                continue
            _log_defer_debug(
                debug_records,
                step_idx=step_idx,
                batch_idx=batch_idx,
                positions=committed,
                x0=x0,
                base_conf=base_conf,
                eff_conf=eff_conf,
                gain=gain,
                veto_type=veto_type,
                tau=tau,
                beta=beta_sig,
                gain_type=gain_type,
                causal_lambda=causal_lambda,
                base_logits=base_logits,
                drop_logits=drop_logits,
            )

    return x0, transfer_index
