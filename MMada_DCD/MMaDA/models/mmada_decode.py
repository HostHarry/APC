from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


def _maybe_expand_attention_bias(attention_bias: Optional[torch.Tensor], repeat: int) -> Optional[torch.Tensor]:
    if attention_bias is None:
        return None
    if attention_bias.shape[0] == 1:
        return attention_bias.repeat(repeat, 1, 1, 1)
    return torch.cat([attention_bias] * repeat, dim=0)


def _confidence_from_logits(logits: torch.Tensor, x0: torch.Tensor, remasking: str) -> torch.Tensor:
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    raise NotImplementedError(remasking)


def _pick_transfer(
    logits: torch.Tensor,
    config: MMaDADecodeConfig,
    mask_index: torch.Tensor,
    current_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
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
    return x0, transfer_index


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
