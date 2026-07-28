
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from .llada import *
from transformers import AutoConfig, AutoModelForCausalLM

from torch.nn import CrossEntropyLoss

from .llada.modeling_llada import LLaDAModel,LLaDAModelLM,LLaDAConfig,create_model_config_from_pretrained_config
from .llada.generate import generate as llada_generate
from llava.model.language_model.llada.log_likelyhood import get_log_likelihood as get_log_likelihood
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
import os
from accelerate.utils import reduce
ENFORCE_NUM_ITEMIN_BATCH = os.environ.get("ENFORCE_NUM_ITEMIN_BATCH", False)
class LlavaLladaConfig(LLaDAConfig):
    model_type = "llava_llada"
    # temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    # max_new_tokens: int = 1024
    # do_sample: bool = False
    # top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}
    
    
class LlavaLladaModel(LlavaMetaModel,LLaDAModel):
    config_class = LlavaLladaConfig
    dtype = torch.bfloat16 # hack

    def __init__(self, pretrained_config,llada_config,init_params=None,vision_kwargs=None):
        # breakpoint()
        
        LLaDAModel.__init__(self, llada_config)
        LlavaMetaModel.__init__(self, pretrained_config,vision_kwargs=vision_kwargs,skip_init=True)
        
    def embed_tokens(self, x):
        return self.transformer.wte(x)

def sample_t(b,device,policy='uniform',policy_args=None):
    if policy == 'uniform':
        return torch.rand(b, device=device)
    elif policy == 'logit_normal':
        if policy_args is None:
            policy_args = dict(logit_mean=0.0,logit_std=1.0)
        u = torch.normal(mean=policy_args['logit_mean'], std=policy_args['logit_std'], size=(b,), device="cpu")
        u = torch.nn.functional.sigmoid(u).to(device=device)
        return u
    elif policy == "mode":
        u = torch.rand(size=(b,), device="cpu")
        u = 1 - u - policy_args['mode_scale'] * (torch.cos(torch.pi * u / 2) ** 2 - 1 + u)
        return u
        
def forward_process(bsz,seq_len,device, eps=1e-3,policy='uniform',policy_args=None):
    b, l = bsz,seq_len
    t = sample_t(b,device,policy=policy,policy_args=policy_args)
    # t = torch.sigmoid(t)
    p_mask = (1 - eps) * t + eps
    
    p_mask = p_mask[:, None]#.repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=device)
    mask_cutoff =  torch.max(p_mask,masked_indices.min(-1,keepdim=True).values)
    masked_indices = masked_indices <= mask_cutoff
    # mask at least one token
    # 126336 is used for [MASK] token
    #noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    return masked_indices, p_mask
import os
LOG_BATCH_LENGTH = os.environ.get('LOG_BATCH_LENGTH', False)
DEBUG_PRINT_IMAGE_RES = os.environ.get("DEBUG_PRINT_IMAGE_RES", False)

class LlavaLladaForMaskedDiffusion(LLaDAModelLM,LlavaMetaForCausalLM):
    
    config_class = LlavaLladaConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False,vision_kwargs=None,prefix_lm=False,**kwargs):
        LLaDAModelLM.__init__(self, config,model,init_params)

        # configure default generation settings
        config.model_type = "llava_llada"
        # config.rope_scaling = None
        self.prefix_lm = prefix_lm

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LlavaLladaModel(config,model_config, init_params=init_params,vision_kwargs=vision_kwargs)
        else:
            self.model = model
        self.model.set_activation_checkpointing('whole_layer')
        
        self.post_init() # TODO
        # self.eos_id = 126081 # hack
        # self.mask_id = 126336 # hack
        
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        policy='uniform',
        policy_args=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        eos_id = 126081 # hack
        mask_id = 126336
        fim_id = 126085
        raw_inputs_ids = input_ids
        attention_mask_raw = attention_mask.clone()
        non_padding = ~(raw_inputs_ids==eos_id)
        attention_mask[raw_inputs_ids==eos_id] = True # no sequence attention mask per Sec B.1
        labels[raw_inputs_ids==eos_id] = eos_id # revert target
        # fix attention mask
        input_ids == input_ids
        # pad_len = torch.randint(0,pad_len_max,(1,)).item()
        # padding = torch.full((bsz,pad_len),eos_id,dtype=labels.dtype,device=labels.device) 
                
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels,new_input_ids) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,return_inputs=True)
        #prompt_lengths = 
        #breakpoint()
        # hack starts here
        # 1. Get the mask of trget tokens 
        # if we have labels, run forward process
        # prefix_length = 
        # 
        prompt_len = None
        if labels is not None:
            assert labels.min() == -100
            labels_mask = ~(labels == -100) # targets mask
            infill_token_pos = labels==fim_id
            # find index of the first non zero mask
            # labels_mask = labels_mask.cumsum(-1).eq(1)
            if self.prefix_lm:
                # breakpoint()
                prompt_len = labels_mask.float().argmax(dim=1)
                # print(prompt_len)
            noise_embeddings = self.get_model().transformer.wte(torch.tensor([mask_id]).to(raw_inputs_ids))
            # noise_embeddings is 1, 4096
            bsz,seq_len = labels_mask.shape
            noise_embeddings = noise_embeddings.view(1,1,-1)#.repeat(bsz,seq_len,1)
            # t = torch.rand(b, device=input_ids.device)
            masked_indices, p_mask = forward_process(bsz,seq_len,raw_inputs_ids.device,policy=policy,policy_args=policy_args)
            # torch.where()
            final_masked_indices = masked_indices&labels_mask & (~infill_token_pos)
            final_masked_indices_inv = (~masked_indices)&labels_mask & (~infill_token_pos)
            # breakpoint()
            # breakpoint()
            # boardcast goingon here
            # final_masked_indices: B X L X 1
            # noise_embeddings: 1 X 1 X D
            # inputs_embeds:  B X L X D
            inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            inputs_embeds = torch.where(final_masked_indices.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            # inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            # print(final_masked_indices.float().mean(-1).cpu())
            # new_input_ids
            # breakpoint()
            
            labels_inv = labels.clone()
            labels_inv[~final_masked_indices_inv] = -100
            labels[~final_masked_indices] = -100
            labels[labels==fim_id] = -100 # kill infill token so we don't predict it
            
            inputs_embeds = torch.cat([inputs_embeds,inputs_embeds_inv])
            labels =  torch.cat([labels,labels_inv])
            if self.prefix_lm:
                prompt_len = prompt_len.repeat(2,1)
            final_masked_indices = torch.cat([final_masked_indices,final_masked_indices_inv])
            seq_len = labels.shape[-1]
            # print(seq_len)
            if LOG_BATCH_LENGTH:
                print("Batch Length",seq_len)
            CUFOFF=30720
            if seq_len > CUFOFF:
                print(seq_len,labels.shape)
                labels = labels[:,:CUFOFF]
                inputs_embeds = inputs_embeds[:,:CUFOFF]
                attention_mask = attention_mask[:,:CUFOFF]
                if position_ids is not None:
                    position_ids = position_ids[:,:CUFOFF]
                assert input_ids is None
                assert past_key_values is None
            elif seq_len < CUFOFF:
                pass
                # raise ValueError("Out of Length")
                # pad_len_max = 128 #torch.randint(0, 128, (1,)).item()
                # if pad_len_max > 0:
                #     pad_len = torch.randint(0,pad_len_max,(1,)).item()
                #     padding = torch.full((bsz,pad_len),eos_id,dtype=labels.dtype,device=labels.device) 
                #     labels = torch.cat([labels,padding],dim=-1)
                #     new_input_ids  = torch.cat([new_input_ids,padding],dim=-1)
                #     padding = torch.full((bsz,pad_len,inputs_embeds.shape[-1]),0,dtype=inputs_embeds.dtype,device=inputs_embeds.device)
                #     inputs_embeds = torch.cat([inputs_embeds,padding],dim=-2)
                #     padding = torch.full((bsz,pad_len),1,dtype=attention_mask.dtype,device=attention_mask.device)
                #     attention_mask = torch.cat([attention_mask,padding],dim=-1)
                #     if position_ids is not None:
                #         padding = torch.full((bsz,padding),0,dtype=position_ids.dtype,device=position_ids.device)
                #         position_ids = torch.cat([position_ids,padding],dim=-1)
        if dpo_forward:
            raise NotImplementedError() 
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            #assert attention_mask is None or torch.all(attention_mask)
            attention_mask = None
            num_items_in_batch = None
            if ENFORCE_NUM_ITEMIN_BATCH:
                num_items_in_batch = labels.ne(-100).float().sum()
                num_items_in_batch = reduce(num_items_in_batch)
                num_items_in_batch = num_items_in_batch.long()
            output =  super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                prompt_len=prompt_len,
                num_items_in_batch=num_items_in_batch,
            )
            output['new_input_ids']=new_input_ids
            output['labels'] = labels
            output['final_masked_indices']=final_masked_indices
            output['p_mask'] = p_mask
            return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        from llava.decoding.generate_utils import (
            coerce_thinking_config,
            coerce_vchd_config,
            extract_decode_options,
        )
        from llava.decoding import (
            LaViDaVisualAccessAdapter,
            infer_visual_mask_from_expanded_ids,
            visual_contrast_decode,
        )
        from llava.decoding.vcd_decoder import visual_contrastive_decode_vcd
        from llava.decoding.vcd_noise import noise_images

        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        decode_strategy, decode_config = extract_decode_options(kwargs)
        tokenizer = kwargs.pop("tokenizer", None)
        prefix_lm = bool(kwargs.get("prefix_lm", False))

        if decode_strategy in ("vchd", "vchd_fixed", "vcd"):
            if images is None:
                raise ValueError(f"{decode_strategy} decoding requires visual inputs")
            if float(kwargs.get("cfg_scale", 0.0) or 0.0) > 0.0:
                raise ValueError(f"{decode_strategy} decoding does not support cfg_scale > 0")
            if inputs is not None and inputs.shape[0] != 1:
                raise ValueError(f"{decode_strategy} decoding supports batch_size=1 only")

            raw_position_ids = position_ids
            raw_attention_mask = attention_mask
            (
                _input_ids,
                position_ids,
                attention_mask,
                _past,
                inputs_embeds,
                _labels,
                expanded_ids,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                raw_position_ids,
                raw_attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
                return_inputs=True,
            )
            prompt_len = int(inputs_embeds.shape[1])
            max_new_tokens = int(kwargs.pop("max_new_tokens", 128))
            block_length = kwargs.pop("block_length", None)
            step_per_block = kwargs.pop("step_per_block", None)
            steps = kwargs.pop("steps", None)
            kwargs.pop("step_ratio", None)
            kwargs.pop("schedule", None)
            kwargs.pop("schedule_kwargs", None)
            kwargs.pop("remasking", None)
            kwargs.pop("prefix_lm", None)
            kwargs.pop("draft_tokens", None)
            kwargs.pop("verbose", None)
            kwargs.pop("do_sample", None)
            kwargs.pop("top_p", None)
            kwargs.pop("num_beams", None)
            kwargs.pop("pad_token_id", None)
            kwargs.pop("use_cache", None)
            kwargs.pop("stopping_criteria", None)
            temperature = float(kwargs.pop("temperature", 0.0) or 0.0)
            if temperature != 0.0:
                # Contrast path is greedy over CD-APC scores; keep API tolerant.
                pass

            mask_id = int(kwargs.pop("mask_id", 126336))
            eos_token_id = kwargs.pop("eos_token_id", None)
            if eos_token_id is None and tokenizer is not None:
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                eos_token_id = 126081
            text_vocab_size = None
            if tokenizer is not None:
                text_vocab_size = len(tokenizer)
            config = coerce_vchd_config(
                decode_config,
                mask_id=mask_id,
                eos_token_id=eos_token_id,
                text_vocab_size=text_vocab_size,
                forbidden_token_ids=(mask_id,),
            )
            if config.prefix_prompt_cache and not prefix_lm:
                raise ValueError(
                    f"{decode_strategy} paired prompt cache requires prefix_lm=True"
                )

            visual_mask = infer_visual_mask_from_expanded_ids(expanded_ids[0])
            if not bool(visual_mask.any()):
                raise ValueError(
                    f"{decode_strategy} requires IMAGE_TOKEN_INDEX positions "
                    "in the expanded prompt"
                )

            prompt_embeds_negative = None
            if decode_strategy == "vcd" or config.negative_branch == "noise_image":
                config.negative_branch = "noise_image"
                images_cd = noise_images(images, config.noise_step)
                (
                    _ids_cd,
                    _pos_cd,
                    _attn_cd,
                    _past_cd,
                    inputs_embeds_cd,
                    _labels_cd,
                    _expanded_ids_cd,
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    raw_position_ids,
                    raw_attention_mask,
                    None,
                    None,
                    images_cd,
                    modalities,
                    image_sizes=image_sizes,
                    return_inputs=True,
                )
                if inputs_embeds_cd.shape != inputs_embeds.shape:
                    raise RuntimeError(
                        "Noised-image prompt embeds shape mismatch: "
                        f"{tuple(inputs_embeds_cd.shape)} vs "
                        f"{tuple(inputs_embeds.shape)}"
                    )
                prompt_embeds_negative = inputs_embeds_cd

            tokens = torch.full(
                (1, prompt_len + max_new_tokens),
                mask_id,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            tokens[:, :prompt_len] = expanded_ids.to(device=tokens.device)

            adapter = LaViDaVisualAccessAdapter(
                self.get_model(),
                prompt_embeds=inputs_embeds,
                visual_mask=visual_mask,
                decode_start=prompt_len,
                decode_end=prompt_len + max_new_tokens,
                mask_id=mask_id,
                attention_mask=attention_mask,
                force_math_sdpa=config.force_math_sdpa,
                backend="llada",
                prefix_lm=prefix_lm,
                prefix_prompt_cache=config.prefix_prompt_cache,
                prompt_embeds_negative=prompt_embeds_negative,
            )
            if decode_strategy == "vcd" or config.negative_branch == "noise_image":
                # Keep original L/T/B transfer schedule for fair comparison.
                bl = int(block_length) if block_length is not None else max_new_tokens
                spb = int(step_per_block) if step_per_block is not None else 0
                total_steps = int(steps) if steps is not None else 0
                if spb <= 0 and total_steps <= 0:
                    # Default: total_steps == max_new_tokens (LLaDA-style).
                    total_steps = max_new_tokens
                output_tokens = visual_contrastive_decode_vcd(
                    self.get_model(),
                    tokens,
                    decode_start=prompt_len,
                    decode_end=prompt_len + max_new_tokens,
                    config=config,
                    adapter=adapter,
                    block_length=bl,
                    steps=total_steps if total_steps > 0 else None,
                    step_per_block=spb if spb > 0 else None,
                    temperature=temperature,
                )
                self._last_vchd_report = None
                return output_tokens[:, prompt_len:]

            # VCHD pops block/step schedule; length controlled by max_new_tokens.
            result = visual_contrast_decode(
                self.get_model(),
                tokens,
                decode_start=prompt_len,
                decode_end=prompt_len + max_new_tokens,
                visual_mask=visual_mask,
                config=config,
                attention_mask=attention_mask,
                adapter=adapter,
            )
            if config.return_report:
                output_tokens, report = result
                self._last_vchd_report = report
            else:
                output_tokens = result
                self._last_vchd_report = None
            return output_tokens[:, prompt_len:]

        thinking_config = coerce_thinking_config(
            decode_strategy, decode_config
        )
        visual_mask = None
        if thinking_config.vrg_enabled:
            if images is None:
                raise ValueError("VRG decoding requires visual inputs")
            if inputs is not None and inputs.shape[0] != 1:
                raise ValueError("VRG decoding currently supports batch_size=1")

        if images is not None and thinking_config.vrg_enabled:
            (
                _input_ids,
                position_ids,
                attention_mask,
                _past,
                inputs_embeds,
                _labels,
                expanded_ids,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
                return_inputs=True,
            )
            visual_mask = infer_visual_mask_from_expanded_ids(expanded_ids[0])
            if not bool(visual_mask.any()):
                raise ValueError(
                    "VRG requires IMAGE_TOKEN_INDEX positions in the "
                    "expanded multimodal prompt"
                )
        elif images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            # breakpoint()
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # if DEBUG_PRINT_IMAGE_RES:
        #     print("Seq len:",inputs_embeds.shape[1])

        #return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        generation_output = llada_generate(
            self.get_model(),
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            thinking_config=thinking_config,
            visual_mask=visual_mask,
            **kwargs,
        )
        if thinking_config.vrg_enabled:
            response_length = int(kwargs.get("max_new_tokens", 128))
            if isinstance(generation_output, tuple):
                tokens, history = generation_output
                return tokens[:, -response_length:], history
            return generation_output[:, -response_length:]
        return generation_output
    
    
    @torch.no_grad()
    def log_likelyhood_inference(
        self,
        inputs: Optional[torch.Tensor] = None,
        answer: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        mc_num=128,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        max_seq_len = 5000
        #if inputs_embeds.shape[1] > max_seq_len:
        max_seq_len = max_seq_len[:,-max_seq_len:]
        answer = answer[:300]
        return get_log_likelihood(self.get_model(), None,inputs_embeds=inputs_embeds, answer=answer, mc_num=mc_num,**kwargs)
        #return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        return llada_generate(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_llada", LlavaLladaConfig)
AutoModelForCausalLM.register(LlavaLladaConfig, LlavaLladaForMaskedDiffusion)

    
    
            
    
