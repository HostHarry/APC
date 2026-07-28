"""Pure VCD decode loop (Leng et al. CVPR 2024) on LLaDA block/step schedule.

Uses the same confidence-based transfer schedule as ``llada_generate`` /
original decoding, but selects tokens from CD-APC contrast logits produced by
a paired visual / noised-image forward.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .config import VCHDDecodeConfig
from .contrast import compute_contrast_stats
from .vocabulary import build_valid_text_vocab


def _resolve_text_vocab_size(model, config: VCHDDecodeConfig, vocab_size: int) -> Optional[int]:
    if config.text_vocab_size is not None:
        return int(config.text_vocab_size)
    for attr in ("llm_vocab_size", "vocab_size"):
        value = getattr(getattr(model, "config", None), attr, None)
        if value is not None:
            return int(value)
    return vocab_size


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = max(1, int(steps))
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def visual_contrastive_decode_vcd(
    model,
    tokens: torch.LongTensor,
    *,
    decode_start: int,
    decode_end: int,
    config: VCHDDecodeConfig,
    adapter,
    block_length: Optional[int] = None,
    steps: Optional[int] = None,
    step_per_block: Optional[int] = None,
    temperature: float = 0.0,
) -> torch.LongTensor:
    """Run original-schedule VCD with CD-APC contrast selection."""

    config.validate()
    if config.negative_branch != "noise_image":
        raise ValueError(
            "visual_contrastive_decode_vcd requires "
            "negative_branch='noise_image'"
        )
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("VCD decoder supports batch_size=1 only")
    if not 0 <= decode_start < decode_end <= tokens.shape[1]:
        raise ValueError(
            f"Invalid decode range [{decode_start}, {decode_end}) for "
            f"sequence length {tokens.shape[1]}"
        )

    state = tokens.clone()
    response_length = decode_end - decode_start
    block_length = int(
        block_length
        or config.vcd_block_length
        or response_length
    )
    if response_length % block_length != 0:
        raise ValueError(
            f"response_length ({response_length}) must be divisible by "
            f"block_length ({block_length})"
        )
    num_blocks = response_length // block_length
    if step_per_block is not None and int(step_per_block) > 0:
        steps_per_block = int(step_per_block)
    elif steps is not None and int(steps) > 0:
        total_steps = int(steps)
        if total_steps % num_blocks != 0:
            raise ValueError(
                f"steps ({total_steps}) must be divisible by num_blocks "
                f"({num_blocks})"
            )
        steps_per_block = total_steps // num_blocks
    elif config.vcd_step_per_block > 0:
        steps_per_block = int(config.vcd_step_per_block)
    elif config.vcd_steps > 0:
        if config.vcd_steps % num_blocks != 0:
            raise ValueError(
                f"vcd_steps ({config.vcd_steps}) must be divisible by "
                f"num_blocks ({num_blocks})"
            )
        steps_per_block = int(config.vcd_steps) // num_blocks
    else:
        steps_per_block = block_length
    steps_per_block = min(steps_per_block, block_length)

    valid_text_vocab: Optional[torch.BoolTensor] = None
    mask_id = int(config.mask_id)

    for num_block in range(num_blocks):
        block_left = decode_start + num_block * block_length
        block_right = decode_start + (num_block + 1) * block_length
        block_mask_index = state[:, block_left:block_right] == mask_id
        num_transfer_tokens = _get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )
        for step_i in range(steps_per_block):
            mask_index = state == mask_id
            if not bool(mask_index.any()):
                return state

            paired = adapter.paired_forward(state)
            visual = paired.visual
            ablated = paired.ablated
            # paired_forward returns response-only logits [R, V]
            if visual.ndim != 2:
                raise RuntimeError(
                    f"Unexpected visual logits shape {tuple(visual.shape)}"
                )
            if valid_text_vocab is None:
                vocab_size = int(visual.shape[-1])
                valid_text_vocab = build_valid_text_vocab(
                    vocab_size,
                    text_vocab_size=_resolve_text_vocab_size(
                        model, config, vocab_size
                    ),
                    forbidden_token_ids=config.forbidden_token_ids,
                    eos_token_id=config.eos_token_id,
                    device=visual.device,
                )

            stats = compute_contrast_stats(
                visual,
                ablated,
                valid_text_vocab,
                alpha=config.alpha,
                beta=config.beta,
            )
            # Reconstruct full-sequence token predictions for transfer.
            x0_response = stats.contrast_token
            # Confidence = APC-aware contrast confidence (paper uses CD logits;
            # low-confidence remasking uses p_cd of the chosen token).
            conf_response = stats.contrast_confidence

            x0 = state.clone()
            x0[:, decode_start:decode_end] = torch.where(
                mask_index[:, decode_start:decode_end],
                x0_response,
                state[:, decode_start:decode_end],
            )
            confidence = torch.full(
                state.shape, -np.inf, device=state.device, dtype=torch.float64
            )
            confidence[:, decode_start:decode_end] = torch.where(
                mask_index[:, decode_start:decode_end],
                conf_response.to(torch.float64),
                torch.tensor(-np.inf, device=state.device, dtype=torch.float64),
            )
            # Only transfer within the active block (and already-unmasked prefix).
            confidence[:, block_right:] = -np.inf

            k = int(num_transfer_tokens[0, step_i].item())
            if k <= 0:
                continue
            _, select_index = torch.topk(confidence[0], k=k)
            state[0, select_index] = x0[0, select_index]

            if config.truncate_at_eos and config.eos_token_id is not None:
                eos_ids = (
                    (int(config.eos_token_id),)
                    if isinstance(config.eos_token_id, int)
                    else tuple(int(x) for x in config.eos_token_id)
                )
                resp = state[0, decode_start:decode_end]
                for eos_id in eos_ids:
                    hits = (resp == eos_id).nonzero(as_tuple=False)
                    if hits.numel() > 0:
                        first = int(hits[0].item())
                        # Keep committed eos; leave remaining masks.
                        break

    return state
