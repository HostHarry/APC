import os
import sys
import torch
import warnings
import numpy as np
import pandas as pd
import string
from PIL import Image
from transformers import AutoTokenizer, AutoConfig

# Add MMaDA models path to sys.path
mmada_path = os.path.join(os.path.dirname(__file__), '../../../../..')
sys.path.insert(0, mmada_path)

try:
    from models import MAGVITv2, MMadaModelLM
    from models.mmada_decode import MMaDADecodeConfig, decode_config_from_dict
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform, image_transform_squash
except ImportError as e:
    warnings.warn(f"Failed to import MMaDA modules: {e}")
    sys.path.append('/path/to/mmada')
    from models import MAGVITv2, MMadaConfig, MMadaModelLM
    from models.mmada_decode import MMaDADecodeConfig, decode_config_from_dict
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform, image_transform_squash

from .utils import load_mmada_image, reorganize_mmada_prompt
from .dataset_configs import get_dataset_config
from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY
from ...smp import *

from .utils import (build_multi_choice_prompt,
                    build_mcq_cot_prompt,
                    build_qa_cot_prompt)


class MMaDA(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    
    def __init__(self,
                 model_path='./work_dirs/mmada/',
                 tokenizer_path=None,
                 vq_model_path=None,
                 vq_model_type='magvitv2',
                 resolution=256,
                 max_new_tokens=1024,
                 steps=512,
                 block_length=1024,
                 temperature=0.8,
                 top_k=1,
                 use_config_file=True,
                 custom_configs=None,
                 decode_strategy='original',
                 cache_type='none',
                 dcd_window_type='sliding',
                 dcd_initial_window_length=32,
                 dcd_block_size=32,
                 dcd_decode_algo='threshold',
                 dcd_decode_param=0.9,
                 dcd_temperature=0.0,
                 dcd_remasking='low_confidence',
                 **kwargs):
        self.use_cot = (os.getenv('USE_COT') == '1')
        print(f"use_cot: {self.use_cot}")
        self.cot_prompt = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"
        
        self.model_path = model_path
        self.resolution = resolution
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.block_length = block_length
        self.temperature = temperature
        self.top_k = top_k
        self.use_config_file = use_config_file
        self.custom_configs = custom_configs or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decode_strategy = os.getenv('MMADA_DECODE_STRATEGY', decode_strategy)
        self.dcd_config = None
        if self.decode_strategy == 'dcd':
            self.dcd_config = MMaDADecodeConfig(
                window_type=dcd_window_type,
                initial_window_length=dcd_initial_window_length,
                block_size=dcd_block_size,
                decode_algo=dcd_decode_algo,
                decode_param=dcd_decode_param,
                temperature=dcd_temperature,
                remasking=dcd_remasking,
                cache_type=os.getenv('MMADA_CACHE_TYPE', cache_type),
            )
        print(f"[MMaDA] decode_strategy={self.decode_strategy}, "
              f"cache_type={self.dcd_config.cache_type if self.dcd_config else 'N/A'}")
        
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        except Exception as e:
            warnings.warn(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        
        self.uni_prompting = UniversalPrompting(
            self.tokenizer, 
            max_text_len=2048,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100, 
            cond_dropout_prob=0.0, 
            use_reserved_token=True
        )
        
        if vq_model_type == "magvitv2":
            if vq_model_path is None:
                vq_model_path = "multimodalart/MAGVIT2"
            
            self.vq_model = MAGVITv2.from_pretrained(vq_model_path).to(self.device)
            self.vq_model.requires_grad_(False)
            self.vq_model.eval()
        else:
            raise ValueError(f"VQ model type {vq_model_type} not supported.")
        
        self.model = MMadaModelLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        
        self.mask_token_id = self.model.config.mask_token_id if hasattr(self.model.config, 'mask_token_id') else None
        
        warnings.warn(f'MMaDA initialized with model_path: {model_path}')

    def get_generation_kwargs(self, dataset=None):
        base_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "steps": self.steps,
            "block_length": self.block_length,
            "temperature": self.temperature,
            "top_k": self.top_k,
        }

        if not self.use_config_file or dataset is None:
            if dataset and dataset in self.custom_configs:
                base_kwargs.update(self.custom_configs[dataset])
            return base_kwargs

        dataset_config = get_dataset_config(dataset)
        if dataset_config:
            base_kwargs.update(dataset_config)
        custom = self.custom_configs.get(dataset, {})
        if custom:
            base_kwargs.update(custom)
        return base_kwargs

   
    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True
        return True

    def build_prompt(self, line, dataset=None):
        """Build prompt for MMaDA evaluation"""
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        
        tgt_path = self.dump_image(line, dataset)
        
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        return message

    def set_max_num(self, dataset):
        """Set maximum number of images based on dataset"""
        self.total_max_num = 16  # Conservative limit for MMaDA
        if dataset is None:
            self.max_num = 1  # MMaDA typically works with single images
            return None
        
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        else:
            self.max_num = 1  # Start with single image support

    @torch.no_grad()
    def generate_mmada(self, message, dataset=None):
        """Generate response using MMaDA model"""
        image_num = len([x for x in message if x['type'] == 'image'])
        
        if image_num == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            return "I need an image to provide a meaningful response."
        
        if image_num > 1:
            warnings.warn(f"Multiple images ({image_num}) detected, using the first one.")

        image_path = [x['value'] for x in message if x['type'] == 'image'][0]
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        
        try:
            image_ori = Image.open(image_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load image from {image_path}: {e}")
            image_ori = Image.new("RGB", (self.resolution, self.resolution), (255, 255, 255))
        
        squash_tags = ('ai2d', 'clevr', 'docvqa', 'geo', 'llava')
        if any(tag in os.path.basename(image_path).lower() for tag in squash_tags):
            image = image_transform_squash(image_ori, resolution=self.resolution).to(self.device)
        else:
            image = image_transform(image_ori, resolution=self.resolution).to(self.device)
        image = image.unsqueeze(0)
        
        image_tokens = self.vq_model.get_code(image) + len(self.uni_prompting.text_tokenizer)

        messages = [{"role": "user", "content": prompt}]
        text_token_ids = self.uni_prompting.text_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        batch_size = image_tokens.shape[0]
        input_ids = torch.cat([
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device),
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            image_tokens,
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            text_token_ids,
        ], dim=1).long()
        
        generation_kwargs = self.get_generation_kwargs(dataset)

        if self.decode_strategy == 'dcd' and self.dcd_config is not None:
            generation_kwargs['decode_strategy'] = 'dcd'
            generation_kwargs['decode_config'] = self.dcd_config
        
        if dataset:
            warnings.warn(f"Using generation config for {dataset}: {generation_kwargs}")
        
        output_ids = self.model.mmu_generate(
            input_ids, 
            **generation_kwargs  
        )
        
        response_text = self.uni_prompting.text_tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        response_text = self.post_process_response(response_text, dataset)
        
        return response_text.strip()

    def post_process_response(self, response, dataset=None):
        if dataset is None:
            return response
            
        if DATASET_TYPE(dataset) == 'Y/N':
            response_lower = response.lower()
            if 'yes' in response_lower and 'no' not in response_lower:
                return 'Yes'
            elif 'no' in response_lower and 'yes' not in response_lower:
                return 'No'
            elif response_lower.strip().startswith('yes'):
                return 'Yes'
            elif response_lower.strip().startswith('no'):
                return 'No'
                
        elif DATASET_TYPE(dataset) == 'MCQ':
            import re
            matches = re.findall(r'\b([A-E])\b', response)
            if matches:
                return matches[-1]
                
        if listinstr(['MME'], dataset):
            if len(response) > 10:
                words = response.split()
                if len(words) > 3:
                    return ' '.join(words[:3])
                    
        return response

    def generate_inner(self, message, dataset=None):
        """Main generation function called by VLMEvalKit"""
        self.set_max_num(dataset)
        return self.generate_mmada(message, dataset) 