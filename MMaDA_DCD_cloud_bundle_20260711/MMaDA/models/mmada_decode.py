from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MMaDADecodeConfig:
    window_type: str = "sliding"
    initial_window_length: int = 32
    block_size: int = 32
    decode_algo: str = "threshold"
    decode_param: float = 0.9
    temperature: float = 0.0
    remasking: str = "low_confidence"
    cfg_scale: float = 0.0
    mask_id: int = 126336
    debug: bool = False
    cache_type: str = "none"
    refresh_count: int = 1
    # CV-DCD (Causal-Visual Deferred Commitment)
    visual_token_start: int = 2
    visual_token_end: int = 1026
    causal_lambda: float = 0.5
    causal_clip: float = 4.0
    cv_stride: int = 1
    image_drop_strategy: str = "mask"  # mask | shuffle | random_mask | mean_token | neutral
    return_debug: bool = False
    cv_alpha: float = 0.1
    cv_mode: str = "cd_apc"  # cd_apc | cd_naive | defer_only | legacy_score | off
    # cv_conf_source / cv_gate_tau are consumed by cv_mode in
    # {'cd_apc', 'cd_naive'}. They decouple "which token CD picks"
    # from "what confidence the DCD threshold sees", which is the main fix for
    # CD hurting commit-based diffusion decoding.
    #   - cv_conf_source: which softmax feeds the DCD threshold.
    #       'blended'          -> softmax(unmasked blended)[x0]        (helper's v1)
    #       'min_base_blended' -> min(base_conf, blended_conf)          (conservative)
    #       'base'             -> softmax(base_logits)[x0]              (cleanest semantics)
    #   - cv_gate_tau: if > 0, bypass CD at positions where base_conf >= tau
    #     (0.0 disables gating; the whole step goes through CD).
    cv_conf_source: str = "min_base_blended"  # was "blended"; conservative default
    cv_gate_tau: float = 0.0
    # v4 direction B (defer-only CV): activated by cv_mode == 'defer_only'.
    # argmax is ALWAYS argmax(base_logits) -- visual gain only modulates confidence.
    #   defer_veto_type in {'hard', 'mult', 'min', 'soft'}
    #     - hard: eff_conf = base_conf if gain >= tau else 0
    #     - mult: eff_conf = base_conf * sigmoid(beta * (gain - tau))    (deprecated)
    #     - min : eff_conf = min(base_conf, sigmoid(beta * gain))         (deprecated)
    #     - soft: eff_conf = base_conf * exp(-beta * max(tau - gain, 0))  (recommended smooth variant)
    #   defer_tau: gain threshold (only used by 'hard', 'mult', 'soft')
    #   defer_beta: penalty steepness (used by 'mult', 'min', 'soft')
    #   defer_gain_type in {'logit', 'logprob'}
    #     - logit  : gain = base_logit[x] - drop_logit[x]           (v3.2-compatible; ~[-1.5, +1.5])
    #     - logprob: gain = log_softmax(base)[x] - log_softmax(drop)[x]  (CD paper; ~[-0.05, +0.05] for shuffle)
    defer_veto_type: str = "soft"  # was "mult" (deprecated); soft is the smooth variant
    defer_tau: float = 0.0
    defer_beta: float = 1.0
    defer_gain_type: str = "logit"
    _drop_perm: Optional[List[torch.Tensor]] = field(default=None, repr=False)
    # Neutral-image ablation: pre-encoded VQ codes of a gray reference image.
    # Shape: 1-D tensor of length (visual_token_end - visual_token_start), stored
    # in the model's global vocab space (i.e., already offset by vocab_offset when
    # applicable). Populated by the wrapper at load time.
    neutral_image_tokens: Optional[torch.Tensor] = field(default=None, repr=False)
    # v4 addition: image_drop_strategy='text_only' replaces the image span with
    # this token id (typically tokenizer.pad_token_id or eos_token_id). Populated
    # by the wrapper when the strategy is selected.
    text_only_fill_id: Optional[int] = field(default=None, repr=False)


def decode_config_from_dict(d: Dict) -> MMaDADecodeConfig:
    if d is None:
        return MMaDADecodeConfig()
    cfg = MMaDADecodeConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


from .cv_common import image_drop as _image_drop_mod
from .cv_common import log_prob as _log_prob_mod
from .cv_common import paired_forward as _paired_forward_mod


def _maybe_expand_attention_bias(attention_bias: Optional[torch.Tensor], repeat: int) -> Optional[torch.Tensor]:
    return _paired_forward_mod._maybe_expand_attention_bias(attention_bias, repeat)


def _confidence_from_logits(logits: torch.Tensor, x0: torch.Tensor, remasking: str) -> torch.Tensor:
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    raise NotImplementedError(remasking)


def _logp_of_x0(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return _log_prob_mod.logp_of_tokens(logits, x0)


def _build_dropped_image(x: torch.Tensor, config: MMaDADecodeConfig) -> torch.Tensor:
    """Construct x_drop by ablating the visual token span (thin wrapper)."""
    return _image_drop_mod.build_dropped_image(x, config)


def _cv_scores_from_paired_logits(
    logits: torch.Tensor,
    drop_logits: torch.Tensor,
    x0: torch.Tensor,
    config: MMaDADecodeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (raw_confidence, cv_score) for candidate tokens x0.

    cv_score lives in the same [0, 1]-ish probability space as raw_conf so
    that downstream threshold checks (e.g. ``decode_algo='threshold'`` with
    ``decode_param=0.9``) remain meaningful. Concretely::

        cv_score = raw_conf * exp(λ · clamp(base_logp - drop_logp))

    which equals ``exp(base_logp + λ · visual_gain)``. With ``visual_gain >= 0``
    the token can exceed 1.0 (extra image support), while a negative gain
    pushes it below raw_conf. When ``causal_clip`` is large this can still
    saturate above 1, but that is fine: any value ``>=0.9`` passes the
    threshold, which is what we want.
    """
    raw_conf = _confidence_from_logits(logits, x0, config.remasking)
    base_logp = _logp_of_x0(logits, x0)
    drop_logp = _logp_of_x0(drop_logits, x0)
    visual_gain = (base_logp - drop_logp).clamp(
        min=-config.causal_clip, max=config.causal_clip
    )
    cv_score = raw_conf * torch.exp(config.causal_lambda * visual_gain)
    return raw_conf, cv_score


def _apc_plausible_mask(logits: torch.Tensor, config: MMaDADecodeConfig) -> torch.Tensor:
    """Return Li/O'Brien/DoLa adaptive plausibility mask for CD logits."""
    alpha = float(config.cv_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"cv_alpha must be in (0, 1], got {alpha}")
    max_base = logits.amax(dim=-1, keepdim=True)
    return logits >= (max_base + math.log(alpha))


def _apply_cd_style(
    logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
    config: MMaDADecodeConfig,
) -> torch.Tensor:
    """Apply APC-style contrastive decoding in logit space.

    This follows the Li et al. / O'Brien-Lewis / DoLa recipe:
        valid = base_logits >= max(base_logits) + log(alpha)
        cd_logits = (1 + beta) * base_logits - beta * drop_logits
    with invalid tokens masked out. Here beta is config.causal_lambda.
    """
    if drop_logits is None or config.causal_lambda == 0.0:
        return logits
    plausible = _apc_plausible_mask(logits, config)
    blended = logits + config.causal_lambda * (logits - drop_logits)
    neg_inf = torch.finfo(blended.dtype).min
    return torch.where(plausible, blended, torch.full_like(blended, neg_inf))


def _apply_naive_contrastive(
    logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
    config: MMaDADecodeConfig,
) -> torch.Tensor:
    """Apply contrastive logit blending without APC; ablation only."""
    if drop_logits is None or config.causal_lambda == 0.0:
        return logits
    return logits + config.causal_lambda * (logits - drop_logits)


def _prob_at(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """softmax(logits)[x0], computed in float64 for stable confidences."""
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    return torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)


def _cd_pick_with_conf(
    logits: torch.Tensor,
    drop_logits: torch.Tensor,
    config: MMaDADecodeConfig,
    *,
    apc: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contrastive selection with a *decoupled* commit-confidence.

    Returns ``(effective_logits, x0, conf)`` where:

    * ``effective_logits`` / ``x0`` come from the contrastive distribution
      (APC-masked when ``apc=True``, naive blend otherwise) -- i.e. CD still
      decides *which* token to place;
    * ``conf`` is the ranking/threshold signal used by DCD. Its scale is chosen
      by ``config.cv_conf_source`` and computed on the BASE model where
      requested, so the ``decode_param`` threshold stays calibrated instead of
      being inflated by the CD blend + APC masking.

    ``config.cv_gate_tau > 0`` lets positions the base model is already sure
    about bypass CD entirely, so easy tokens are never irreversibly flipped by
    the contrastive term (flips are unrecoverable in masked-diffusion decoding).
    """
    if apc:
        eff = _apply_cd_style(logits, drop_logits, config)
    else:
        eff = _apply_naive_contrastive(logits, drop_logits, config)

    x0_cd = torch.argmax(add_gumbel_noise(eff, config.temperature), dim=-1)
    x0_base = torch.argmax(add_gumbel_noise(logits, config.temperature), dim=-1)

    # Gate: where the base model is already confident, keep its own token.
    tau = float(config.cv_gate_tau)
    base_conf_base = _prob_at(logits, x0_base)
    if tau > 0.0:
        gate = base_conf_base >= tau
    else:
        gate = torch.zeros_like(base_conf_base, dtype=torch.bool)
    x0 = torch.where(gate, x0_base, x0_cd)

    if config.remasking != "low_confidence":
        # Non-probabilistic ranking (e.g. 'random'): nothing to recalibrate.
        conf = _confidence_from_logits(eff, x0, config.remasking)
        return eff, x0, conf

    src = (config.cv_conf_source or "blended").lower()
    base_conf_x0 = _prob_at(logits, x0)
    if src == "base":
        conf = base_conf_x0
    elif src == "min_base_blended":
        conf = torch.minimum(base_conf_x0, _prob_at(eff, x0))
    elif src == "blended":
        conf = _prob_at(eff, x0)
    else:
        raise ValueError(f"Unknown cv_conf_source: {config.cv_conf_source}")
    # Gated positions are always ranked by their (high) base confidence.
    conf = torch.where(gate, base_conf_base, conf)
    return eff, x0, conf


def _pick_transfer(
    logits: torch.Tensor,
    config: MMaDADecodeConfig,
    mask_index: torch.Tensor,
    current_tokens: torch.Tensor,
    confidence_override: Optional[torch.Tensor] = None,
    debug_records: Optional[List[Dict[str, Any]]] = None,
    step_idx: int = 0,
    base_logits: Optional[torch.Tensor] = None,
    drop_logits: Optional[torch.Tensor] = None,
    x0_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x0_override is not None:
        x0 = x0_override
    else:
        logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
    if confidence_override is not None:
        conf = confidence_override
    else:
        conf = _confidence_from_logits(logits, x0, config.remasking)
    x0 = torch.where(mask_index, x0, current_tokens)
    confidence = torch.where(mask_index, conf, torch.full_like(conf, -np.inf))

    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
    for batch_idx in range(confidence.shape[0]):
        masked_positions = mask_index[batch_idx].nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            continue
        if config.decode_algo == "threshold":
            threshold = float(config.decode_param)
            chosen = masked_positions[confidence[batch_idx, masked_positions] >= threshold]
            if chosen.numel() == 0:
                chosen = masked_positions[torch.topk(confidence[batch_idx, masked_positions], k=1).indices]
        elif config.decode_algo == "factor":
            k = max(1, math.ceil(masked_positions.numel() * float(config.decode_param)))
            rel = torch.topk(confidence[batch_idx, masked_positions], k=min(k, masked_positions.numel())).indices
            chosen = masked_positions[rel]
        else:
            k = 1 if masked_positions.numel() == 1 else max(1, masked_positions.numel() // max(1, config.refresh_count))
            rel = torch.topk(confidence[batch_idx, masked_positions], k=min(k, masked_positions.numel())).indices
            chosen = masked_positions[rel]
        transfer_index[batch_idx, chosen] = True

        if debug_records is not None and config.return_debug:
            raw_conf = _confidence_from_logits(logits, x0, config.remasking)
            for pos in chosen.tolist():
                base_argmax = None
                argmax_changed = None
                raw_conf_base = None
                raw_conf_effective = None
                v_valid_size = None
                visual_gain_at_x0 = None
                if base_logits is not None:
                    base_pos = base_logits[batch_idx, pos].to(torch.float64)
                    base_probs = F.softmax(base_pos, dim=-1)
                    base_argmax_t = torch.argmax(base_pos)
                    base_argmax = int(base_argmax_t.item())
                    argmax_changed = bool(base_argmax != int(x0[batch_idx, pos].item()))
                    raw_conf_base = float(base_probs.max().item())
                    if config.cv_mode == "cd_apc":
                        alpha = float(config.cv_alpha)
                        if 0.0 < alpha <= 1.0:
                            threshold = base_pos.max() + math.log(alpha)
                            v_valid_size = int((base_pos >= threshold).sum().item())

                effective_probs = F.softmax(logits[batch_idx, pos].to(torch.float64), dim=-1)
                raw_conf_effective = float(effective_probs.max().item())
                if base_logits is not None and drop_logits is not None:
                    token_id = int(x0[batch_idx, pos].item())
                    visual_gain_at_x0 = float(
                        (base_logits[batch_idx, pos, token_id] - drop_logits[batch_idx, pos, token_id]).item()
                    )

                debug_records.append({
                    "step": step_idx,
                    "batch": batch_idx,
                    "position": int(pos),
                    "x0_token": int(x0[batch_idx, pos].item()),
                    "raw_conf": float(raw_conf[batch_idx, pos].item()),
                    "cv_score": float(conf[batch_idx, pos].item()),
                    "base_argmax": base_argmax,
                    "argmax_changed": argmax_changed,
                    "raw_conf_base": raw_conf_base,
                    "raw_conf_effective": raw_conf_effective,
                    "v_valid_size": v_valid_size,
                    "visual_gain_at_x0": visual_gain_at_x0,
                })

    return x0, transfer_index


def _resolve_cv_confidence(
    logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
    x0: torch.Tensor,
    config: MMaDADecodeConfig,
    step_idx: int,
) -> torch.Tensor:
    """Pick commit ranking signal: raw conf, or causal-visual score."""
    if config.causal_lambda == 0.0 or drop_logits is None:
        return _confidence_from_logits(logits, x0, config.remasking)
    raw_conf, cv_score = _cv_scores_from_paired_logits(logits, drop_logits, x0, config)
    return cv_score


def _pick_transfer_cv(
    logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
    config: MMaDADecodeConfig,
    mask_index: torch.Tensor,
    current_tokens: torch.Tensor,
    debug_records: Optional[List[Dict[str, Any]]] = None,
    step_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick transfer tokens for CV-DCD according to the selected CV mode."""
    mode = (config.cv_mode or "cd_apc").lower()
    if mode == "defer_only":
        from .defer_only.dispatcher import pick_transfer_defer_only
        return pick_transfer_defer_only(
            logits, drop_logits, config, mask_index, current_tokens,
            debug_records=debug_records,
            step_idx=step_idx,
        )
    if mode in ("cd_apc", "cd_naive"):
        # When drop logits are unavailable this step (cv_stride skip or
        # causal_lambda == 0), fall back to a plain base pick so the commit
        # threshold sees the same (base) confidence scale on every step.
        if drop_logits is None or config.causal_lambda == 0.0:
            return _pick_transfer(
                logits, config, mask_index, current_tokens,
                debug_records=debug_records,
                step_idx=step_idx,
                base_logits=logits,
                drop_logits=None,
            )
        eff, x0, conf = _cd_pick_with_conf(
            logits, drop_logits, config, apc=(mode == "cd_apc")
        )
        return _pick_transfer(
            eff, config, mask_index, current_tokens,
            confidence_override=conf,
            x0_override=x0,
            debug_records=debug_records,
            step_idx=step_idx,
            base_logits=logits,
            drop_logits=drop_logits,
        )
    if mode == "legacy_score":
        logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
        x0_cand = torch.argmax(logits_with_noise, dim=-1)
        cv_conf = _resolve_cv_confidence(logits, drop_logits, x0_cand, config, step_idx)
        return _pick_transfer(
            logits, config, mask_index, current_tokens,
            confidence_override=cv_conf,
            debug_records=debug_records,
            step_idx=step_idx,
            base_logits=logits,
            drop_logits=drop_logits,
        )
    if mode == "off":
        return _pick_transfer(
            logits, config, mask_index, current_tokens,
            debug_records=debug_records,
            step_idx=step_idx,
            base_logits=logits,
            drop_logits=drop_logits,
        )
    raise NotImplementedError(f"Unknown cv_mode: {config.cv_mode}")


def _should_compute_drop_forward(config: MMaDADecodeConfig, step_idx: int) -> bool:
    if config.causal_lambda == 0.0:
        return False
    stride = max(1, config.cv_stride)
    return step_idx % stride == 0


def _iter_block_slices(decode_start: int, decode_end: int, block_size: int):
    pos = decode_start
    while pos < decode_end:
        nxt = min(pos + block_size, decode_end)
        yield pos, nxt
        pos = nxt


@torch.no_grad()
def dcd_decode_text(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
):
    x = tokens.clone()
    nfe = 0
    for block_start, block_end in _iter_block_slices(decode_start, decode_end, max(1, config.block_size)):
        while (x[:, block_start:block_end] == config.mask_id).any():
            mask_index = x == config.mask_id
            mask_index[:, block_end:] = False
            if config.cfg_scale > 0.0 and prompt_index is not None:
                un_x = x.clone()
                un_x[prompt_index] = config.mask_id
                x_ = torch.cat([x, un_x], dim=0)
                ab = _maybe_expand_attention_bias(attention_bias, 2)
                logits = model(x_, attention_bias=ab).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (config.cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_bias=attention_bias).logits
            x0, transfer_index = _pick_transfer(logits, config, mask_index, x)
            x[transfer_index] = x0[transfer_index]
            nfe += 1
    return (x, nfe) if config.debug else x


@torch.no_grad()
def dcd_decode_text_prefix_cache(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
):
    if config.cfg_scale > 0.0 and prompt_index is not None:
        return dcd_decode_text(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)

    x = tokens.clone()
    nfe = 0
    for block_start, block_end in _iter_block_slices(decode_start, decode_end, max(1, config.block_size)):
        out = model(x, attention_bias=attention_bias, use_cache=True)
        past_key_values = out.past_key_values
        mask_index = x == config.mask_id
        mask_index[:, block_end:] = False
        x0, transfer_index = _pick_transfer(out.logits, config, mask_index, x)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        prefix_cache = []
        for layer_cache in past_key_values:
            prefix_cache.append(tuple(cache[:, :, :block_start] for cache in layer_cache))
        prefix_cache = prefix_cache

        while (x[:, block_start:block_end] == config.mask_id).any():
            block_tokens = x[:, block_start:]
            block_mask = block_tokens == config.mask_id
            block_mask[:, block_end - block_start :] = False
            logits = model(
                block_tokens,
                attention_bias=attention_bias,
                past_key_values=prefix_cache,
                use_cache=True,
            ).logits
            x0, transfer_index = _pick_transfer(logits, config, block_mask, block_tokens)
            x[:, block_start:][transfer_index] = x0[transfer_index]
            nfe += 1
    return (x, nfe) if config.debug else x


@torch.no_grad()
def dcd_decode_text_dual_cache(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
):
    if config.cfg_scale > 0.0 and prompt_index is not None:
        return dcd_decode_text(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)

    x = tokens.clone()
    nfe = 0
    for block_start, block_end in _iter_block_slices(decode_start, decode_end, max(1, config.block_size)):
        out = model(x, attention_bias=attention_bias, use_cache=True)
        past_key_values = out.past_key_values
        mask_index = x == config.mask_id
        mask_index[:, block_end:] = False
        x0, transfer_index = _pick_transfer(out.logits, config, mask_index, x)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = True
        while (x[:, block_start:block_end] == config.mask_id).any():
            block_tokens = x[:, block_start:block_end]
            block_mask = block_tokens == config.mask_id
            out = model(
                block_tokens,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            )
            past_key_values = out.past_key_values
            x0, transfer_index = _pick_transfer(out.logits, config, block_mask, block_tokens)
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
            nfe += 1
    return (x, nfe) if config.debug else x


def _image_logits(logits: torch.Tensor, vocab_offset: int, codebook_size: int) -> torch.Tensor:
    return logits[..., vocab_offset : vocab_offset + codebook_size]


def _current_image_tokens(tokens: torch.Tensor, mask_id: int, vocab_offset: int) -> torch.Tensor:
    return torch.where(tokens == mask_id, tokens, tokens - vocab_offset)


@torch.no_grad()
def dcd_decode_image(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    vocab_offset: int,
    codebook_size: int,
    attention_bias: Optional[torch.Tensor] = None,
):
    x = tokens.clone()
    nfe = 0
    for block_start, block_end in _iter_block_slices(decode_start, decode_end, max(1, config.block_size)):
        while (x[:, block_start:block_end] == config.mask_id).any():
            logits = model(x, attention_bias=attention_bias).logits[:, block_start:block_end]
            logits = _image_logits(logits, vocab_offset, codebook_size)
            current_tokens = _current_image_tokens(x[:, block_start:block_end], config.mask_id, vocab_offset)
            mask_index = x[:, block_start:block_end] == config.mask_id
            x0, transfer_index = _pick_transfer(logits, config, mask_index, current_tokens)
            x[:, block_start:block_end][transfer_index] = x0[transfer_index] + vocab_offset
            nfe += 1
    result = x[:, decode_start:decode_end] - vocab_offset
    return (result, nfe) if config.debug else result


@torch.no_grad()
def dcd_decode_image_dual_cache(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    vocab_offset: int,
    codebook_size: int,
    attention_bias: Optional[torch.Tensor] = None,
):
    x = tokens.clone()
    nfe = 0
    for block_start, block_end in _iter_block_slices(decode_start, decode_end, max(1, config.block_size)):
        out = model(x, attention_bias=attention_bias, use_cache=True)
        past_key_values = out.past_key_values
        logits = _image_logits(out.logits[:, block_start:block_end], vocab_offset, codebook_size)
        current_tokens = _current_image_tokens(x[:, block_start:block_end], config.mask_id, vocab_offset)
        mask_index = x[:, block_start:block_end] == config.mask_id
        x0, transfer_index = _pick_transfer(logits, config, mask_index, current_tokens)
        x[:, block_start:block_end][transfer_index] = x0[transfer_index] + vocab_offset
        nfe += 1

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = True
        while (x[:, block_start:block_end] == config.mask_id).any():
            block_tokens = x[:, block_start:block_end]
            block_mask = block_tokens == config.mask_id
            out = model(
                block_tokens,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            )
            past_key_values = out.past_key_values
            logits = _image_logits(out.logits, vocab_offset, codebook_size)
            current_tokens = _current_image_tokens(block_tokens, config.mask_id, vocab_offset)
            x0, transfer_index = _pick_transfer(logits, config, block_mask, current_tokens)
            x[:, block_start:block_end][transfer_index] = x0[transfer_index] + vocab_offset
            nfe += 1
    result = x[:, decode_start:decode_end] - vocab_offset
    return (result, nfe) if config.debug else result


def _paired_forward_logits(
    model,
    x: torch.Tensor,
    x_drop: torch.Tensor,
    attention_bias: Optional[torch.Tensor],
    past_key_values=None,
    use_cache: bool = False,
    replace_position=None,
):
    """Run base and drop-image forwards (thin wrapper around cv_common)."""
    return _paired_forward_mod.paired_forward_logits(
        model, x, x_drop, attention_bias,
        past_key_values=past_key_values,
        use_cache=use_cache,
        replace_position=replace_position,
    )


def _cv_decode_return(x, nfe, debug_records, config):
    if config.return_debug:
        return x, {"nfe": nfe, "debug_records": debug_records}
    if config.debug:
        return x, nfe
    return x


@torch.no_grad()
def dcd_decode_text_cv(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
):
    if config.cfg_scale > 0.0 and prompt_index is not None:
        return dcd_decode_text(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)

    x = tokens.clone()
    nfe = 0
    step_idx = 0
    debug_records: List[Dict[str, Any]] = []
    config._drop_perm = None

    for block_start, block_end in _iter_block_slices(decode_start, decode_end, max(1, config.block_size)):
        while (x[:, block_start:block_end] == config.mask_id).any():
            mask_index = x == config.mask_id
            mask_index[:, block_end:] = False
            x_drop = _build_dropped_image(x, config)

            drop_logits = None
            if _should_compute_drop_forward(config, step_idx):
                logits, drop_logits, _, _ = _paired_forward_logits(
                    model, x, x_drop, attention_bias, use_cache=False
                )
                nfe += 2
            else:
                logits = model(x, attention_bias=attention_bias).logits
                nfe += 1

            x0, transfer_index = _pick_transfer_cv(
                logits, drop_logits, config, mask_index, x,
                debug_records=debug_records,
                step_idx=step_idx,
            )
            x[transfer_index] = x0[transfer_index]
            step_idx += 1

    return _cv_decode_return(x, nfe, debug_records, config)


@torch.no_grad()
def dcd_decode_text_cv_prefix_cache(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
):
    if config.cfg_scale > 0.0 and prompt_index is not None:
        return dcd_decode_text(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)

    x = tokens.clone()
    nfe = 0
    step_idx = 0
    debug_records: List[Dict[str, Any]] = []
    config._drop_perm = None

    for block_start, block_end in _iter_block_slices(decode_start, decode_end, max(1, config.block_size)):
        use_drop = config.causal_lambda > 0.0
        if use_drop:
            x_drop = _build_dropped_image(x, config)
            logits, drop_logits, past_key_values, past_key_values_drop = _paired_forward_logits(
                model, x, x_drop, attention_bias, use_cache=True
            )
            nfe += 2
        else:
            out = model(x, attention_bias=attention_bias, use_cache=True)
            logits = out.logits
            drop_logits = None
            past_key_values = out.past_key_values
            past_key_values_drop = None
            nfe += 1

        mask_index = x == config.mask_id
        mask_index[:, block_end:] = False
        drop_for_step = drop_logits if _should_compute_drop_forward(config, step_idx) else None
        x0, transfer_index = _pick_transfer_cv(
            logits, drop_for_step, config, mask_index, x,
            debug_records=debug_records,
            step_idx=step_idx,
        )
        x[transfer_index] = x0[transfer_index]
        step_idx += 1

        prefix_cache = []
        for layer_cache in past_key_values:
            prefix_cache.append(tuple(c[:, :, :block_start] for c in layer_cache))
        prefix_cache_drop = []
        if use_drop:
            for layer_cache_drop in past_key_values_drop:
                prefix_cache_drop.append(tuple(c[:, :, :block_start] for c in layer_cache_drop))

        while (x[:, block_start:block_end] == config.mask_id).any():
            block_tokens = x[:, block_start:]
            block_mask = block_tokens == config.mask_id
            block_mask[:, block_end - block_start :] = False

            if use_drop:
                x_drop_block = _build_dropped_image(x, config)[:, block_start:]
                logits = model(
                    block_tokens, attention_bias=attention_bias,
                    past_key_values=prefix_cache, use_cache=True,
                ).logits
                drop_logits_raw = model(
                    x_drop_block, attention_bias=attention_bias,
                    past_key_values=prefix_cache_drop, use_cache=True,
                ).logits
                drop_logits = drop_logits_raw if _should_compute_drop_forward(config, step_idx) else None
                nfe += 2
            else:
                logits = model(
                    block_tokens, attention_bias=attention_bias,
                    past_key_values=prefix_cache, use_cache=True,
                ).logits
                drop_logits = None
                nfe += 1

            x0, transfer_index = _pick_transfer_cv(
                logits, drop_logits, config, block_mask, block_tokens,
                debug_records=debug_records,
                step_idx=step_idx,
            )
            x[:, block_start:][transfer_index] = x0[transfer_index]
            step_idx += 1

    return _cv_decode_return(x, nfe, debug_records, config)


@torch.no_grad()
def dcd_decode_text_cv_dual_cache(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
):
    if config.cfg_scale > 0.0 and prompt_index is not None:
        return dcd_decode_text(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)

    x = tokens.clone()
    nfe = 0
    step_idx = 0
    debug_records: List[Dict[str, Any]] = []
    config._drop_perm = None

    for block_start, block_end in _iter_block_slices(decode_start, decode_end, max(1, config.block_size)):
        use_drop = config.causal_lambda > 0.0
        if use_drop:
            x_drop = _build_dropped_image(x, config)
            logits, drop_logits, past_key_values, past_key_values_drop = _paired_forward_logits(
                model, x, x_drop, attention_bias, use_cache=True
            )
            nfe += 2
        else:
            out = model(x, attention_bias=attention_bias, use_cache=True)
            logits = out.logits
            drop_logits = None
            past_key_values = out.past_key_values
            past_key_values_drop = None
            nfe += 1

        mask_index = x == config.mask_id
        mask_index[:, block_end:] = False
        drop_for_step = drop_logits if _should_compute_drop_forward(config, step_idx) else None
        x0, transfer_index = _pick_transfer_cv(
            logits, drop_for_step, config, mask_index, x,
            debug_records=debug_records,
            step_idx=step_idx,
        )
        x[transfer_index] = x0[transfer_index]
        step_idx += 1

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = True

        while (x[:, block_start:block_end] == config.mask_id).any():
            block_tokens = x[:, block_start:block_end]
            block_mask = block_tokens == config.mask_id

            if use_drop:
                x_drop_block = _build_dropped_image(x, config)[:, block_start:block_end]
                out = model(
                    block_tokens,
                    attention_bias=attention_bias,
                    past_key_values=past_key_values,
                    use_cache=True,
                    replace_position=replace_position,
                )
                past_key_values = out.past_key_values
                logits = out.logits
                out_drop = model(
                    x_drop_block,
                    attention_bias=attention_bias,
                    past_key_values=past_key_values_drop,
                    use_cache=True,
                    replace_position=replace_position,
                )
                past_key_values_drop = out_drop.past_key_values
                drop_logits = out_drop.logits if _should_compute_drop_forward(config, step_idx) else None
                nfe += 2
            else:
                out = model(
                    block_tokens,
                    attention_bias=attention_bias,
                    past_key_values=past_key_values,
                    use_cache=True,
                    replace_position=replace_position,
                )
                past_key_values = out.past_key_values
                logits = out.logits
                drop_logits = None
                nfe += 1

            x0, transfer_index = _pick_transfer_cv(
                logits, drop_logits, config, block_mask, block_tokens,
                debug_records=debug_records,
                step_idx=step_idx,
            )
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
            step_idx += 1

    return _cv_decode_return(x, nfe, debug_records, config)


def dispatch_cv_dcd_decode_text(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
):
    if config.cache_type == "dual":
        return dcd_decode_text_cv_dual_cache(
            model, tokens, decode_start, decode_end, config, attention_bias, prompt_index
        )
    if config.cache_type == "prefix":
        return dcd_decode_text_cv_prefix_cache(
            model, tokens, decode_start, decode_end, config, attention_bias, prompt_index
        )
    return dcd_decode_text_cv(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)


def dispatch_dcd_decode_text(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
):
    if config.cache_type == "prefix":
        return dcd_decode_text_prefix_cache(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)
    if config.cache_type == "dual":
        return dcd_decode_text_dual_cache(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)
    return dcd_decode_text(model, tokens, decode_start, decode_end, config, attention_bias, prompt_index)


def dispatch_dcd_decode_image(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: MMaDADecodeConfig,
    vocab_offset: int,
    codebook_size: int,
    attention_bias: Optional[torch.Tensor] = None,
):
    if config.cache_type == "dual":
        return dcd_decode_image_dual_cache(
            model, tokens, decode_start, decode_end, config, vocab_offset, codebook_size, attention_bias
        )
    return dcd_decode_image(model, tokens, decode_start, decode_end, config, vocab_offset, codebook_size, attention_bias)
