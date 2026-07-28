from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .mmada_decode import (
    MMaDADecodeConfig,
    dcd_decode_text_official,
    decode_config_from_dict,
    dispatch_dcd_decode_image,
    dispatch_dcd_decode_text,
    dispatch_cv_dcd_decode_text,
    vcd_decode_text,
)
from decoding import (
    VCHDDecodeConfig,
    vchd_config_from_dict,
    visual_contrast_decode,
)
from decoding.mmada_adapter import build_paired_attention_bias
from decoding.swd import SwdState, apply_swd_to_confidence, response_span_token_probs
from decoding.thinking_diffusion import (
    ThinkingSwdDecodeConfig,
    apply_psp,
    apply_vrg,
    resolve_thinking_swd_config,
)
from .sampling import cosine_schedule, mask_by_random_topk
from transformers import PretrainedConfig

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def _infer_visual_token_span(idx: torch.Tensor, soi_id: int = 126084, eoi_id: int = 126085):
    """Infer [visual_token_start, visual_token_end) from prompt layout [mmu, soi, img*, eoi, text]."""
    seq = idx[0].tolist()
    try:
        soi_pos = seq.index(soi_id)
        eoi_pos = seq.index(eoi_id, soi_pos + 1)
        return soi_pos + 1, eoi_pos
    except ValueError:
        return 2, 1026


def _merge_attention_bias_with_vrg(
    branch_bias: torch.Tensor,
    attention_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if attention_bias is None:
        return branch_bias
    if attention_bias.dtype == torch.bool:
        additive = torch.zeros(
            attention_bias.shape,
            dtype=branch_bias.dtype,
            device=branch_bias.device,
        )
        additive = additive.masked_fill(
            ~attention_bias, torch.finfo(branch_bias.dtype).min
        )
    else:
        additive = attention_bias.to(
            dtype=branch_bias.dtype, device=branch_bias.device
        )
    if additive.shape[0] == 1:
        additive = additive.expand(branch_bias.shape[0], *additive.shape[1:])
    return branch_bias + additive


def _original_path_logits(
    model,
    x: torch.Tensor,
    *,
    attention_bias: Optional[torch.Tensor],
    cfg_scale: float,
    prompt_index: torch.Tensor,
    thinking_cfg: ThinkingSwdDecodeConfig,
    image_span: Tuple[int, int],
    mask_id: int,
) -> torch.Tensor:
    if thinking_cfg.vrg_enabled:
        if cfg_scale > 0.0:
            raise ValueError(
                "VRG visual guidance cannot be combined with text cfg_scale>0"
            )
        seq_len = int(x.shape[1])
        branch_bias = build_paired_attention_bias(
            seq_len, image_span, device=x.device
        )
        branch_bias = _merge_attention_bias_with_vrg(branch_bias, attention_bias)
        pair_logits = model(x.repeat(2, 1), attention_bias=branch_bias).logits
        logits_c, logits_u = torch.chunk(pair_logits, 2, dim=0)
        return apply_vrg(logits_c, logits_u, thinking_cfg.vrg_scale)
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        logits = model(torch.cat([x, un_x], dim=0)).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        return un_logits + (cfg_scale + 1) * (logits - un_logits)
    return model(x, attention_bias=attention_bias).logits


def _prepare_cv_dcd_decode_config(decode_config, mask_id, temperature, cfg_scale, remasking, block_length, idx):
    if decode_config is None:
        decode_config = MMaDADecodeConfig(
            mask_id=mask_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            block_size=block_length,
            cache_type="dual",
        )
    elif isinstance(decode_config, dict):
        decode_config = decode_config_from_dict(decode_config)
    decode_config.mask_id = mask_id
    if decode_config.visual_token_start == 2 and decode_config.visual_token_end == 1026 and idx is not None:
        vs, ve = _infer_visual_token_span(idx)
        decode_config.visual_token_start = vs
        decode_config.visual_token_end = ve
    return decode_config


class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])



class MMadaModelLM(LLaDAModelLM):
    config_class = MMadaConfig
    base_model_prefix = "model"
    def __init__(self, config: MMadaConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

        # # resize token embeddings
        # print(f"Resizing token embeddings to {config.new_vocab_size}")
        # self.resize_token_embeddings(config.new_vocab_size)

    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            decode_strategy='original',
            decode_config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)

        if attention_mask is not None:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None

        if decode_strategy == 'dcd':
            if decode_config is None:
                decode_config = MMaDADecodeConfig(
                    mask_id=mask_token_id,
                    temperature=temperature,
                    cfg_scale=guidance_scale,
                )
            elif isinstance(decode_config, dict):
                decode_config = decode_config_from_dict(decode_config)
            decode_config.mask_id = mask_token_id
            vocab_offset = len(uni_prompting.text_tokenizer) + num_new_special_tokens
            result = dispatch_dcd_decode_image(
                model=self,
                tokens=input_ids.clone(),
                decode_start=input_ids.shape[1] - (num_vq_tokens + 1),
                decode_end=input_ids.shape[1] - 1,
                config=decode_config,
                vocab_offset=vocab_offset,
                codebook_size=codebook_size,
                attention_bias=attention_bias,
            )
            return result if not decode_config.debug else result[0]

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids
    
    def forward_process(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            p_mask_lm=None,
            p_mask_mmu=None,
            answer_lengths=None,
            t2i_masks=None,
            answer_lengths_lm=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[batch_size_t2i:batch_size_t2i + batch_size_lm]
        masked_indices_mmu = masked_indices[-batch_size_mmu:]
        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)       
        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        loss_lm = F.cross_entropy(
            logits[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]

        if answer_lengths_lm is not None:
            loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0])  
        else:
            loss_lm = loss_lm.sum() / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0] * logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[1])     

        loss_mmu = F.cross_entropy(
            logits[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[-batch_size_mmu:].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu

    def forward_process_with_r2i(
            self,
            input_ids, 
            labels,
            t2i_masks=None,
            max_seq_length=128,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_r2i=0,
            p_mask_lm=None,
            p_mask_mmu=None,
            p_mask_r2i=None,
            answer_lengths=None,
            answer_lengths_lm=None,
            answer_lengths_r2i=None,
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            # t2i loss
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        # llada loss  

        start_lm = batch_size_t2i
        end_lm = start_lm + batch_size_lm
        start_mmu = end_lm
        end_mmu = start_mmu + batch_size_mmu
        start_r2i = end_mmu
        end_r2i = start_r2i + batch_size_r2i

        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[start_lm:end_lm]
        masked_indices_mmu = masked_indices[start_mmu:end_mmu]
        masked_indices_r2i = masked_indices[start_r2i:end_r2i]

        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
        p_mask_r2i = p_mask_r2i.to(masked_indices_r2i.device)

        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
        answer_lengths_r2i = answer_lengths_r2i.to(masked_indices_r2i.device)

        loss_lm = F.cross_entropy(
            logits[start_lm:end_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]

        if answer_lengths_lm is not None:
            loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[start_lm:end_lm].shape[0]) 
        else:
            loss_lm = loss_lm.sum() / (logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1])

        loss_mmu = F.cross_entropy(
            logits[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[start_mmu:end_mmu].shape[0])
        
        loss_r2i = F.cross_entropy(
            logits[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1, self.output_size),
            labels[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_r2i[masked_indices_r2i]
        loss_r2i = torch.sum(loss_r2i/answer_lengths_r2i[masked_indices_r2i]) / (logits[start_r2i:end_r2i].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu, loss_r2i


    def forward_t2i(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            max_seq_length=128,
            t2i_masks=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        return loss_t2i





    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, decode_strategy='original', decode_config=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if decode_strategy in ('vchd', 'vchd_fixed'):
            if idx is None:
                raise ValueError("VCHD decoding requires token input_ids")
            if input_embeddings is not None:
                raise ValueError("VCHD decoding does not support input_embeddings")
            if cfg_scale > 0.0:
                raise ValueError("VCHD decoding does not support MMU CFG")
            if decode_config is None:
                decode_config = VCHDDecodeConfig(mask_id=mask_id)
            elif isinstance(decode_config, dict):
                decode_config = vchd_config_from_dict(decode_config)
            elif not isinstance(decode_config, VCHDDecodeConfig):
                raise TypeError(
                    "decode_config must be VCHDDecodeConfig or dict for VCHD"
                )
            decode_config.mask_id = mask_id
            decode_config.validate()

            batch_size = idx.shape[0]
            if batch_size != 1:
                raise ValueError("The phase 0--2 VCHD decoder supports batch_size=1")
            x = torch.full(
                (batch_size, idx.shape[1] + max_new_tokens),
                mask_id,
                dtype=torch.long,
                device=self.device,
            )
            x[:, :idx.shape[1]] = idx.clone()
            visual_span = _infer_visual_token_span(idx)
            return visual_contrast_decode(
                model=self,
                tokens=x,
                decode_start=idx.shape[1],
                decode_end=idx.shape[1] + max_new_tokens,
                image_span=visual_span,
                config=decode_config,
                attention_mask=attention_mask,
            )

        if decode_strategy in ('dcd', 'official_dcd'):
            if decode_config is None:
                decode_config = MMaDADecodeConfig(mask_id=mask_id, temperature=temperature, cfg_scale=cfg_scale, remasking=remasking, block_size=block_length)
            elif isinstance(decode_config, dict):
                decode_config = decode_config_from_dict(decode_config)
            decode_config.mask_id = mask_id
            batch_size = idx.shape[0]
            x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
            x[:, :idx.shape[1]] = idx.clone()
            prompt_index = (x != mask_id)
            if attention_mask is not None and 0.0 in attention_mask:
                ab = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            else:
                ab = None
            if decode_strategy == 'official_dcd':
                if cfg_scale > 0.0:
                    raise ValueError("Official DCD reproduction does not use MMU CFG")
                result = dcd_decode_text_official(
                    model=self,
                    tokens=x,
                    decode_start=idx.shape[1],
                    decode_end=idx.shape[1] + max_new_tokens,
                    config=decode_config,
                    attention_bias=ab,
                )
            else:
                result = dispatch_dcd_decode_text(
                    model=self,
                    tokens=x,
                    decode_start=idx.shape[1],
                    decode_end=idx.shape[1] + max_new_tokens,
                    config=decode_config,
                    attention_bias=ab,
                    prompt_index=prompt_index,
                )
            return result if not decode_config.debug else result[0]

        if decode_strategy in ('cv_dcd', 'causal_dcd', 'grounded_dcd'):
            decode_config = _prepare_cv_dcd_decode_config(
                decode_config, mask_id, temperature, cfg_scale, remasking, block_length, idx
            )
            batch_size = idx.shape[0]
            x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
            x[:, :idx.shape[1]] = idx.clone()
            prompt_index = (x != mask_id)
            if attention_mask is not None and 0.0 in attention_mask:
                ab = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            else:
                ab = None
            result = dispatch_cv_dcd_decode_text(
                model=self,
                tokens=x,
                decode_start=idx.shape[1],
                decode_end=idx.shape[1] + max_new_tokens,
                config=decode_config,
                attention_bias=ab,
                prompt_index=prompt_index,
            )
            if decode_config.return_debug and isinstance(result, tuple):
                tokens_out, debug_info = result
                return tokens_out, debug_info
            return result if not decode_config.debug else result[0]

        if decode_strategy == 'vcd':
            if idx is None:
                raise ValueError("VCD decoding requires token input_ids")
            if cfg_scale > 0.0:
                raise ValueError("VCD decoding does not support MMU CFG")
            if decode_config is None:
                decode_config = MMaDADecodeConfig(
                    mask_id=mask_id,
                    temperature=temperature,
                    remasking=remasking,
                    block_size=block_length,
                    causal_lambda=1.0,
                    cv_alpha=0.1,
                    cv_mode="cd_apc",
                    image_drop_strategy="gaussian_noise",
                    visual_token_start=2,
                    visual_token_end=1026,
                )
            elif isinstance(decode_config, dict):
                decode_config = decode_config_from_dict(decode_config)
            decode_config.mask_id = mask_id
            decode_config.image_drop_strategy = "gaussian_noise"
            if decode_config.noise_image_tokens is None:
                raise ValueError(
                    "VCD requires decode_config.noise_image_tokens "
                    "(populate via add_diffusion_noise → get_code)"
                )
            batch_size = idx.shape[0]
            x = torch.full(
                (batch_size, idx.shape[1] + max_new_tokens),
                mask_id,
                dtype=torch.long,
            ).to(self.device)
            x[:, : idx.shape[1]] = idx.clone()
            if attention_mask is not None and 0.0 in attention_mask:
                ab = (
                    attention_mask[:, :, None] & attention_mask[:, None, :]
                ).bool().unsqueeze(1)
            else:
                ab = None
            return vcd_decode_text(
                model=self,
                tokens=x,
                decode_start=idx.shape[1],
                decode_end=idx.shape[1] + max_new_tokens,
                steps=steps,
                block_length=block_length,
                config=decode_config,
                attention_bias=ab,
                temperature=temperature,
            )

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        thinking_cfg = resolve_thinking_swd_config(
            decode_config
            if isinstance(decode_config, (ThinkingSwdDecodeConfig, dict))
            or decode_config is None
            else None
        )

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        image_span = _infer_visual_token_span(idx)
        swd_state = SwdState() if thinking_cfg.swd_enabled else None
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        total_steps = num_blocks * steps
        response_start = int(idx.shape[1])
        response_end = response_start + int(max_new_tokens)
        
        # print(f"num_blocks: {num_blocks}, steps: {steps}")
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
            for i in range(steps):
                mask_index = (x == mask_id)
                logits = _original_path_logits(
                    self,
                    x,
                    attention_bias=attention_bias,
                    cfg_scale=cfg_scale,
                    prompt_index=prompt_index,
                    thinking_cfg=thinking_cfg,
                    image_span=image_span,
                    mask_id=mask_id,
                )
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                swd_p_curr = None
                if remasking == 'low_confidence':
                    if thinking_cfg.swd_enabled:
                        # SWD path: never materialize full-sequence float64 softmax.
                        x0_p, swd_p_curr = response_span_token_probs(
                            logits,
                            x0,
                            response_start=response_start,
                            response_end=response_end,
                        )
                    else:
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                global_step = num_block * steps + i
                if thinking_cfg.psp_enabled and remasking == 'low_confidence':
                    confidence = apply_psp(
                        confidence,
                        step_index=global_step,
                        num_steps=total_steps,
                        response_start=response_start,
                        response_end=response_end,
                        gamma=thinking_cfg.psp_gamma,
                    )
                if (
                    thinking_cfg.swd_enabled
                    and remasking == 'low_confidence'
                    and swd_state is not None
                ):
                    confidence = apply_swd_to_confidence(
                        confidence,
                        logits,
                        swd_state,
                        lambda_=thinking_cfg.swd_lambda,
                        response_start=response_start,
                        response_end=response_end,
                        mask_index=mask_index,
                        p_curr=swd_p_curr,
                    )

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                
            
            # logits = logits[:, -1, :] / temperature
            # # optionally crop the logits to only the top k options
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float('Inf')
            # # apply softmax to convert logits to (normalized) probabilities
            # probs = F.softmax(logits, dim=-1)
            # # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1)
            # result.append(idx_next[0][0])
            # # append sampled index to the running sequence and continue
            # if self.config.w_clip_vit:
            #     idx_next_embeddings = self.mmada.model.embed_tokens(idx_next)
            #     input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            # else:
            #     idx = torch.cat((idx, idx_next), dim=1)

            # if eot_token is not None and idx_next.cpu() == eot_token:
            #     break

        return x

    @torch.no_grad()
    def mmu_generate_fast(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, decode_strategy='original', decode_config=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if decode_strategy in ('dcd', 'vchd', 'vchd_fixed'):
            return self.mmu_generate(
                idx=idx,
                input_embeddings=input_embeddings,
                max_new_tokens=max_new_tokens,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                top_k=top_k,
                eot_token=eot_token,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
                attention_mask=attention_mask,
                decode_strategy=decode_strategy,
                decode_config=decode_config,
            )

        if decode_strategy in ('cv_dcd', 'causal_dcd', 'grounded_dcd'):
            return self.mmu_generate(
                idx=idx,
                input_embeddings=input_embeddings,
                max_new_tokens=max_new_tokens,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                top_k=top_k,
                eot_token=eot_token,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
                attention_mask=attention_mask,
                decode_strategy=decode_strategy,
                decode_config=decode_config,
            )

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        thinking_cfg = resolve_thinking_swd_config(
            decode_config
            if isinstance(decode_config, (ThinkingSwdDecodeConfig, dict))
            or decode_config is None
            else None
        )

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        image_span = _infer_visual_token_span(idx)
        swd_state = SwdState() if thinking_cfg.swd_enabled else None
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        total_steps = num_blocks * steps
        response_start = int(idx.shape[1])
        response_end = response_start + int(max_new_tokens)
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id)
                logits = _original_path_logits(
                    self,
                    x,
                    attention_bias=attention_bias,
                    cfg_scale=cfg_scale,
                    prompt_index=prompt_index,
                    thinking_cfg=thinking_cfg,
                    image_span=image_span,
                    mask_id=mask_id,
                )
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                swd_p_curr = None
                if remasking == 'low_confidence':
                    if thinking_cfg.swd_enabled:
                        x0_p, swd_p_curr = response_span_token_probs(
                            logits,
                            x0,
                            response_start=response_start,
                            response_end=response_end,
                        )
                    else:
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                global_step = num_block * steps + i
                if thinking_cfg.psp_enabled and remasking == 'low_confidence':
                    confidence = apply_psp(
                        confidence,
                        step_index=global_step,
                        num_steps=total_steps,
                        response_start=response_start,
                        response_end=response_end,
                        gamma=thinking_cfg.psp_gamma,
                    )
                if (
                    thinking_cfg.swd_enabled
                    and remasking == 'low_confidence'
                    and swd_state is not None
                ):
                    confidence = apply_swd_to_confidence(
                        confidence,
                        logits,
                        swd_state,
                        lambda_=thinking_cfg.swd_lambda,
                        response_start=response_start,
                        response_end=response_end,
                        mask_index=mask_index,
                        p_curr=swd_p_curr,
                    )

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        return x

    @torch.no_grad()
    def t2i_generate_decoding_stepwise(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            vq_model = None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            current_image_vq_indices = sampled_ids.clone()
            # print(f"current_image_vq_indices: {current_image_vq_indices}")
            current_image_vq_indices = torch.clamp(current_image_vq_indices, 0, 8192 - 1)
            current_image = vq_model.decode_code(current_image_vq_indices)
            images = torch.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = Image.fromarray(images[0]) 
            yield pil_images, f"Step {step + 1}/{timesteps}"
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            

        return sampled_ids
    

AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
