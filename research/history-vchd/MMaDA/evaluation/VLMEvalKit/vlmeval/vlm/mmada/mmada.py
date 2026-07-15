import json as _json
import os
import sys
import time
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
    from decoding import VCHDDecodeConfig
    from models.attention_hooks import collect_attentions
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform, image_transform_squash
except ImportError as e:
    warnings.warn(f"Failed to import MMaDA modules: {e}")
    sys.path.append('/path/to/mmada')
    from models import MAGVITv2, MMadaConfig, MMadaModelLM
    from models.mmada_decode import MMaDADecodeConfig, decode_config_from_dict
    from decoding import VCHDDecodeConfig
    from models.attention_hooks import collect_attentions
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform, image_transform_squash

from .utils import load_mmada_image, reorganize_mmada_prompt
from .dataset_configs import DEFAULT_KWARGS, get_dataset_config, merge_configs
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
                 cv_causal_lambda=0.5,
                 cv_causal_clip=4.0,
                 cv_stride=1,
                 cv_image_drop='mask',
                 cv_alpha=0.1,
                 cv_mode='cd_apc',
                 cv_conf_source='min_base_blended',
                 cv_gate_tau=0.0,
                 defer_veto_type='soft',
                 defer_tau=0.0,
                 defer_beta=1.0,
                 defer_gain_type='logit',
                 vchd_tau_base=0.10,
                 vchd_tau_contrast=0.90,
                 vchd_mask_capacity=16,
                 vchd_max_commit=16,
                 vchd_fallback_to_raw=False,
                 vchd_alpha=0.5,
                 vchd_beta=0.1,
                 vchd_history_enabled=False,
                 vchd_history_top_v=8,
                 vchd_history_ema_decay=0.7,
                 vchd_history_penalty_scale=1.0,
                 vchd_history_anchor_min_consistent=0,
                 vchd_ccaw_enabled=False,
                 vchd_ccaw_mode='legacy',
                 vchd_ccaw_block_size=32,
                 vchd_ccaw_min_commit=1,
                 vchd_ccaw_qualified_budget=1,
                 vchd_ccaw_max_capacity=64,
                 vchd_ccaw_pressure_decay=0.8,
                 vchd_ccaw_expand_step=8,
                 vchd_ccaw_shrink_step=4,
                 vchd_cache_type='none',
                 vchd_cache_refresh_interval=8,
                 vchd_cache_refresh_on_pressure=True,
                 vchd_cache_pressure_threshold=0.60,
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
        self.vchd_config = None
        self.cv_causal_lambda = float(os.getenv('MMADA_CV_LAMBDA', cv_causal_lambda))
        self.cv_causal_clip = float(os.getenv('MMADA_CV_CLIP', cv_causal_clip))
        self.cv_stride = int(os.getenv('MMADA_CV_STRIDE', cv_stride))
        self.cv_image_drop = os.getenv('MMADA_CV_DROP', cv_image_drop)
        self.cv_alpha = float(os.getenv('MMADA_CV_ALPHA', cv_alpha))
        self.cv_mode = os.getenv('MMADA_CV_MODE', cv_mode)
        self.cv_conf_source = os.getenv('MMADA_CV_CONF_SOURCE', cv_conf_source)
        self.cv_gate_tau = float(os.getenv('MMADA_CV_GATE_TAU', cv_gate_tau))
        # v4 defer-only knobs (only consumed when cv_mode == 'defer_only').
        self.defer_veto_type = os.getenv('MMADA_DEFER_VETO', defer_veto_type)
        self.defer_tau = float(os.getenv('MMADA_DEFER_TAU', defer_tau))
        self.defer_beta = float(os.getenv('MMADA_DEFER_BETA', defer_beta))
        self.defer_gain_type = os.getenv('MMADA_DEFER_GAIN_TYPE', defer_gain_type)
        self.vchd_tau_base = float(
            os.getenv('MMADA_VCHD_TAU_BASE', vchd_tau_base)
        )
        self.vchd_tau_contrast = float(
            os.getenv('MMADA_VCHD_TAU_CONTRAST', vchd_tau_contrast)
        )
        self.vchd_mask_capacity = int(
            os.getenv('MMADA_VCHD_MASK_CAPACITY', vchd_mask_capacity)
        )
        self.vchd_max_commit = int(
            os.getenv('MMADA_VCHD_MAX_COMMIT', vchd_max_commit)
        )
        self.vchd_fallback_to_raw = (
            os.getenv(
                'MMADA_VCHD_FALLBACK_TO_RAW',
                '1' if vchd_fallback_to_raw else '0',
            )
            == '1'
        )
        self.vchd_alpha = float(
            os.getenv('MMADA_VCHD_ALPHA', vchd_alpha)
        )
        self.vchd_beta = float(
            os.getenv('MMADA_VCHD_BETA', vchd_beta)
        )
        self.vchd_history_enabled = (
            os.getenv(
                'MMADA_VCHD_HISTORY',
                '1' if vchd_history_enabled else '0',
            )
            == '1'
        )
        self.vchd_history_top_v = int(
            os.getenv('MMADA_VCHD_HISTORY_TOP_V', vchd_history_top_v)
        )
        self.vchd_history_ema_decay = float(
            os.getenv(
                'MMADA_VCHD_HISTORY_EMA_DECAY',
                vchd_history_ema_decay,
            )
        )
        self.vchd_history_penalty_scale = float(
            os.getenv(
                'MMADA_VCHD_HISTORY_PENALTY_SCALE',
                vchd_history_penalty_scale,
            )
        )
        self.vchd_history_anchor_min_consistent = int(
            os.getenv(
                'MMADA_VCHD_HISTORY_ANCHOR_MIN_CONSISTENT',
                vchd_history_anchor_min_consistent,
            )
        )
        self.vchd_ccaw_enabled = (
            os.getenv(
                'MMADA_VCHD_CCAW',
                '1' if vchd_ccaw_enabled else '0',
            )
            == '1'
        )
        self.vchd_ccaw_mode = os.getenv(
            'MMADA_VCHD_CCAW_MODE',
            vchd_ccaw_mode,
        )
        self.vchd_ccaw_block_size = int(
            os.getenv(
                'MMADA_VCHD_CCAW_BLOCK_SIZE',
                vchd_ccaw_block_size,
            )
        )
        self.vchd_ccaw_min_commit = int(
            os.getenv(
                'MMADA_VCHD_CCAW_MIN_COMMIT',
                vchd_ccaw_min_commit,
            )
        )
        self.vchd_ccaw_qualified_budget = int(
            os.getenv(
                'MMADA_VCHD_CCAW_QUALIFIED_BUDGET',
                vchd_ccaw_qualified_budget,
            )
        )
        self.vchd_ccaw_max_capacity = int(
            os.getenv(
                'MMADA_VCHD_CCAW_MAX_CAPACITY',
                vchd_ccaw_max_capacity,
            )
        )
        self.vchd_ccaw_pressure_decay = float(
            os.getenv(
                'MMADA_VCHD_CCAW_PRESSURE_DECAY',
                vchd_ccaw_pressure_decay,
            )
        )
        self.vchd_ccaw_expand_step = int(
            os.getenv(
                'MMADA_VCHD_CCAW_EXPAND_STEP',
                vchd_ccaw_expand_step,
            )
        )
        self.vchd_ccaw_shrink_step = int(
            os.getenv(
                'MMADA_VCHD_CCAW_SHRINK_STEP',
                vchd_ccaw_shrink_step,
            )
        )
        self.vchd_cache_type = os.getenv(
            'MMADA_VCHD_CACHE_TYPE',
            vchd_cache_type,
        )
        self.vchd_cache_refresh_interval = int(
            os.getenv(
                'MMADA_VCHD_CACHE_REFRESH_INTERVAL',
                vchd_cache_refresh_interval,
            )
        )
        self.vchd_cache_refresh_on_pressure = (
            os.getenv(
                'MMADA_VCHD_CACHE_REFRESH_ON_PRESSURE',
                '1' if vchd_cache_refresh_on_pressure else '0',
            )
            == '1'
        )
        self.vchd_cache_pressure_threshold = float(
            os.getenv(
                'MMADA_VCHD_CACHE_PRESSURE_THRESHOLD',
                vchd_cache_pressure_threshold,
            )
        )
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
        elif self.decode_strategy in ('cv_dcd', 'causal_dcd', 'grounded_dcd'):
            self.dcd_config = MMaDADecodeConfig(
                window_type=dcd_window_type,
                initial_window_length=dcd_initial_window_length,
                block_size=dcd_block_size,
                decode_algo=dcd_decode_algo,
                decode_param=dcd_decode_param,
                temperature=dcd_temperature,
                remasking=dcd_remasking,
                cache_type=os.getenv('MMADA_CACHE_TYPE', cache_type),
                causal_lambda=self.cv_causal_lambda,
                causal_clip=self.cv_causal_clip,
                cv_stride=self.cv_stride,
                image_drop_strategy=self.cv_image_drop,
                return_debug=(os.getenv('MMADA_CV_RETURN_DEBUG', '0') == '1'),
                cv_alpha=self.cv_alpha,
                cv_mode=self.cv_mode,
                cv_conf_source=self.cv_conf_source,
                cv_gate_tau=self.cv_gate_tau,
                defer_veto_type=self.defer_veto_type,
                defer_tau=self.defer_tau,
                defer_beta=self.defer_beta,
                defer_gain_type=self.defer_gain_type,
                visual_token_start=2,
                visual_token_end=1026,
            )
            # For image_drop_strategy='text_only', populate the filler token id
            # from the tokenizer's pad_token_id (fallback: eos_token_id).
            # This happens later during __init__ when tokenizer is ready; here
            # we just record the intent in the config's docstring position.
            # See __init__ tail for the actual assignment.
        _defer_info = (
            f", defer_veto={self.defer_veto_type}, defer_tau={self.defer_tau}, "
            f"defer_beta={self.defer_beta}, defer_gain_type={self.defer_gain_type}"
            if self.cv_mode == 'defer_only' else ""
        )
        print(f"[MMaDA] decode_strategy={self.decode_strategy}, "
              f"cache_type={self.dcd_config.cache_type if self.dcd_config else 'N/A'}"
              + (f", cv_lambda={self.cv_causal_lambda}, cv_drop={self.cv_image_drop}, "
                 f"cv_alpha={self.cv_alpha}, cv_mode={self.cv_mode}, "
                 f"cv_conf_source={self.cv_conf_source}, cv_gate_tau={self.cv_gate_tau}"
                 + _defer_info
                 if self.decode_strategy in ('cv_dcd', 'causal_dcd', 'grounded_dcd') else ""))
        
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

        # For CV-DCD with image_drop='neutral', precompute the VQ codes of a
        # mid-gray reference image. This gives us a stable, in-distribution
        # ablation baseline (unlike replacing image tokens with the text
        # mask_id, which is out-of-distribution and produces unstable
        # visual_gain estimates -- see docs/cv_dcd_report_v1.md).
        if self.decode_strategy in ('cv_dcd', 'causal_dcd', 'grounded_dcd') \
                and self.dcd_config is not None \
                and self.cv_image_drop == 'neutral':
            self._populate_neutral_image_tokens()

        # v4 addition: populate text_only_fill_id from the tokenizer so
        # image_drop_strategy='text_only' works out of the box. Prefer
        # pad_token_id (idle in text-side training), then eos_token_id.
        if self.decode_strategy in ('cv_dcd', 'causal_dcd', 'grounded_dcd') \
                and self.dcd_config is not None:
            fill_id = None
            for attr in ('pad_token_id', 'eos_token_id'):
                v = getattr(self.tokenizer, attr, None)
                if v is not None:
                    fill_id = int(v)
                    break
            fill_env = os.getenv('MMADA_TEXT_ONLY_FILL_ID')
            if fill_env is not None and fill_env.strip():
                fill_id = int(fill_env.strip())
            self.dcd_config.text_only_fill_id = fill_id
            if self.cv_image_drop == 'text_only' and fill_id is None:
                warnings.warn(
                    "[MMaDA] cv_image_drop='text_only' but no pad/eos token id "
                    "could be resolved. Set MMADA_TEXT_ONLY_FILL_ID explicitly."
                )
        
        self.mask_token_id = self.model.config.mask_token_id if hasattr(self.model.config, 'mask_token_id') else None

        if self.decode_strategy in ('vchd', 'vchd_fixed'):
            mask_id = int(self.mask_token_id or 126336)
            tokenizer_vocab = self.tokenizer.get_vocab()
            eos_ids = {
                int(token_id)
                for token_id in (
                    getattr(self.tokenizer, 'eos_token_id', None),
                    tokenizer_vocab.get('<|eot|>'),
                    tokenizer_vocab.get('<|eot_id|>'),
                )
                if token_id is not None
            }

            # All tokenizer/prompt control markers are illegal answer tokens,
            # except recognized EOS variants which remain valid terminators.
            forbidden_ids = {mask_id}
            forbidden_ids.update(
                int(token_id)
                for token_id in getattr(self.tokenizer, 'all_special_ids', ())
                if int(token_id) not in eos_ids
            )
            forbidden_ids.update(
                int(token_id)
                for token_id, token in getattr(
                    self.tokenizer, 'added_tokens_decoder', {}
                ).items()
                if getattr(token, 'special', False)
                and int(token_id) not in eos_ids
            )
            forbidden_ids.update(
                int(token_id)
                for token_id in self.uni_prompting.sptids_dict.values()
                if int(token_id) not in eos_ids
            )
            for token_name in (
                '<|start_header_id|>',
                '<|end_header_id|>',
                '[iPAD]',
                '<|r2i|>',
            ):
                token_id = tokenizer_vocab.get(token_name)
                if token_id is not None and int(token_id) not in eos_ids:
                    forbidden_ids.add(int(token_id))

            normalized_eos = tuple(sorted(eos_ids))
            self.vchd_config = VCHDDecodeConfig(
                mask_id=mask_id,
                eos_token_id=normalized_eos or None,
                # Image VQ ids are offset by this exact tokenizer length in
                # generate_mmada; model.config.llm_vocab_size is larger and
                # would admit the first image-code ids as text candidates.
                text_vocab_size=len(self.uni_prompting.text_tokenizer),
                forbidden_token_ids=tuple(sorted(forbidden_ids)),
                alpha=self.vchd_alpha,
                beta=self.vchd_beta,
                tau_base=self.vchd_tau_base,
                tau_contrast=self.vchd_tau_contrast,
                mask_capacity=self.vchd_mask_capacity,
                max_physical_span=max(
                    self.max_new_tokens,
                    self.vchd_mask_capacity,
                    self.vchd_ccaw_block_size,
                    self.vchd_ccaw_max_capacity,
                ),
                max_commit_per_iteration=self.vchd_max_commit,
                fallback_to_raw=self.vchd_fallback_to_raw,
                force_math_sdpa=(
                    os.getenv('MMADA_VCHD_FORCE_MATH_SDPA', '1') == '1'
                ),
                cache_type=self.vchd_cache_type,
                cache_refresh_interval=self.vchd_cache_refresh_interval,
                cache_refresh_on_pressure=(
                    self.vchd_cache_refresh_on_pressure
                ),
                cache_pressure_threshold=(
                    self.vchd_cache_pressure_threshold
                ),
                collect_trace=(
                    os.getenv('MMADA_VCHD_COLLECT_TRACE', '0') == '1'
                ),
                return_report=(
                    os.getenv('MMADA_VCHD_RETURN_REPORT', '0') == '1'
                ),
                history_enabled=self.vchd_history_enabled,
                history_top_v_tokens=self.vchd_history_top_v,
                history_ema_decay=self.vchd_history_ema_decay,
                history_penalty_scale=self.vchd_history_penalty_scale,
                history_anchor_min_consistent=(
                    self.vchd_history_anchor_min_consistent
                ),
                ccaw_enabled=self.vchd_ccaw_enabled,
                ccaw_mode=self.vchd_ccaw_mode,
                ccaw_block_size=self.vchd_ccaw_block_size,
                ccaw_min_commit_per_iteration=(
                    self.vchd_ccaw_min_commit
                ),
                ccaw_qualified_budget=self.vchd_ccaw_qualified_budget,
                ccaw_max_mask_capacity=self.vchd_ccaw_max_capacity,
                ccaw_pressure_ema_decay=self.vchd_ccaw_pressure_decay,
                ccaw_expand_step=self.vchd_ccaw_expand_step,
                ccaw_shrink_step=self.vchd_ccaw_shrink_step,
            )
            self.vchd_config.validate()
            warnings.warn(
                "[MMaDA] History-VCHD enabled: "
                f"alpha={self.vchd_config.alpha}, "
                f"beta={self.vchd_config.beta}, "
                f"tau_base={self.vchd_config.tau_base}, "
                f"tau_contrast={self.vchd_config.tau_contrast}, "
                f"window={self.vchd_config.mask_capacity}, "
                f"max_commit={self.vchd_config.max_commit_per_iteration}, "
                f"history={self.vchd_config.history_enabled}, "
                f"history_scale={self.vchd_config.history_penalty_scale}, "
                f"anchor_consistency="
                f"{self.vchd_config.history_anchor_min_consistent}, "
                f"ccaw={self.vchd_config.ccaw_enabled}, "
                f"ccaw_mode={self.vchd_config.ccaw_mode}, "
                f"ccaw_block={self.vchd_config.ccaw_block_size}, "
                f"ccaw_min_commit="
                f"{self.vchd_config.ccaw_min_commit_per_iteration}, "
                f"qualified_budget={self.vchd_config.ccaw_qualified_budget}, "
                f"cache={self.vchd_config.cache_type}, "
                f"eos={normalized_eos}"
            )

        # Optional attention collection (env-controlled, OFF by default).
        # Useful for inspecting how DCD attends across the prompt; enabling on a
        # full benchmark will produce hundreds of GB of dumps, so it is gated.
        self.collect_attn = (os.getenv('MMADA_COLLECT_ATTENTION', '0') == '1')
        self.attn_dir = os.getenv('MMADA_ATTN_DIR', './attention_dumps')
        self.attn_layers = os.getenv('MMADA_ATTN_LAYERS', 'first_mid_last')
        self.attn_max_samples = int(os.getenv('MMADA_ATTN_MAX_SAMPLES', '0'))  # 0 = unlimited
        self.attn_steps = os.getenv('MMADA_ATTN_STEPS', '')
        self.attn_head_mean = (os.getenv('MMADA_ATTN_HEAD_MEAN', '0') == '1')
        # MMADA_RUN_ID lets multiple inference runs share the same MMADA_ATTN_DIR
        # without overwriting each other; auto-fall back to a wall-clock stamp so
        # the default behaviour is also collision-free.
        self.run_id = (os.getenv('MMADA_RUN_ID') or '').strip() or time.strftime('%Y%m%d_%H%M%S')
        self._attn_count = 0
        self._vchd_report_count = 0
        if self.collect_attn:
            os.makedirs(self.attn_dir, exist_ok=True)
            warnings.warn(
                f"[MMaDA] attention collection ENABLED -> {self.attn_dir} "
                f"(run_id={self.run_id}, layers={self.attn_layers}, "
                f"steps={self.attn_steps or 'all'}, "
                f"head_mean={int(self.attn_head_mean)}, "
                f"max_samples={self.attn_max_samples or 'inf'})"
            )

        warnings.warn(f'MMaDA initialized with model_path: {model_path}')

    @torch.no_grad()
    def _populate_neutral_image_tokens(self):
        """Encode a mid-gray reference image and cache the resulting VQ codes.

        The codes are placed on the same offset as real image inputs so that
        the decoder's ``_build_dropped_image`` can splice them directly into
        the visual span. They are stored in ``self.dcd_config.neutral_image_tokens``
        as a flat 1-D LongTensor of length ``visual_token_end - visual_token_start``.
        """
        gray = Image.new("RGB", (self.resolution, self.resolution), (127, 127, 127))
        image = image_transform(gray, resolution=self.resolution).to(self.device).unsqueeze(0)
        vq_offset = len(self.uni_prompting.text_tokenizer)
        neutral = self.vq_model.get_code(image).squeeze(0).long() + vq_offset
        expected = int(self.dcd_config.visual_token_end - self.dcd_config.visual_token_start)
        if neutral.numel() != expected:
            warnings.warn(
                f"[MMaDA] neutral_image_tokens length mismatch: "
                f"vq produced {neutral.numel()}, expected {expected}. "
                f"Falling back to image_drop='mask'."
            )
            self.dcd_config.image_drop_strategy = 'mask'
            return
        self.dcd_config.neutral_image_tokens = neutral.detach().cpu()
        warnings.warn(
            f"[MMaDA] cached {neutral.numel()} neutral image tokens for CV-DCD "
            f"(gray 127x127x127, resolution={self.resolution}, "
            f"first 8 codes={neutral[:8].tolist()})"
        )

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
        return merge_configs(
            DEFAULT_KWARGS,
            dataset_config,
            self.custom_configs.get(dataset, {}),
        )

   
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
            elif listinstr(['HallusionBench', 'AMBER', 'POPE'], dataset):
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
        elif self.decode_strategy in ('cv_dcd', 'causal_dcd', 'grounded_dcd') and self.dcd_config is not None:
            generation_kwargs['decode_strategy'] = self.decode_strategy
            generation_kwargs['decode_config'] = self.dcd_config
        elif self.decode_strategy in ('vchd', 'vchd_fixed') and self.vchd_config is not None:
            generation_kwargs['decode_strategy'] = self.decode_strategy
            generation_kwargs['decode_config'] = self.vchd_config
        
        if dataset:
            warnings.warn(f"Using generation config for {dataset}: {generation_kwargs}")

        do_collect = (
            self.collect_attn
            and (self.attn_max_samples == 0 or self._attn_count < self.attn_max_samples)
        )
        if do_collect:
            n_layers = self.model.model.config.n_layers
            if self.attn_layers == 'first_mid_last':
                layers = [0, n_layers // 2, n_layers - 1]
            elif self.attn_layers == 'all':
                layers = list(range(n_layers))
            else:
                layers = [int(x) for x in self.attn_layers.split(',') if x.strip()]
            save_steps = None
            save_last = False
            if self.attn_steps:
                save_steps = []
                for item in self.attn_steps.split(','):
                    item = item.strip().lower()
                    if not item:
                        continue
                    if item in {'last', '-1'}:
                        save_last = True
                    else:
                        save_steps.append(int(item))
            sample_id = self._attn_count
            sample_dir = os.path.join(
                self.attn_dir, f"sample_{self.run_id}_{sample_id:05d}"
            )
            os.makedirs(sample_dir, exist_ok=True)
            with collect_attentions(
                self.model,
                layers=layers,
                stream_dir=sample_dir,
                save_steps=save_steps,
                save_last=save_last,
                head_mean=self.attn_head_mean,
            ) as bag:
                gen_out = self.model.mmu_generate(input_ids, **generation_kwargs)
            debug_info = None
            if isinstance(gen_out, tuple) and len(gen_out) == 2:
                output_ids, debug_info = gen_out
            else:
                output_ids = gen_out
            input_len = int(input_ids.shape[1])
            tokenizer = self.uni_prompting.text_tokenizer
            response_preview = tokenizer.decode(
                output_ids[0, input_len:].tolist(), skip_special_tokens=True,
            )
            with open(os.path.join(sample_dir, 'meta.txt'), 'w') as f:
                f.write(f"dataset={dataset}\n"
                        f"dataset_index={os.getenv('MMADA_CURRENT_INDEX', '')}\n"
                        f"run_id={self.run_id}\n"
                        f"image={image_path}\nprompt={prompt}\n"
                        f"input_length={input_len}\n"
                        f"layers={layers}\n"
                        f"saved_steps={self.attn_steps or 'all'}\n"
                        f"head_mean={int(self.attn_head_mean)}\n"
                        f"num_steps={bag.num_steps}\n"
                        f"response={response_preview}\n")

            all_ids = output_ids[0].tolist()
            IMG_START, IMG_END = 2, 1026
            token_strings = []
            for pos, tid in enumerate(all_ids):
                if IMG_START <= pos < IMG_END:
                    token_strings.append("[img]")
                else:
                    token_strings.append(tokenizer.decode([tid]))
            with open(os.path.join(sample_dir, 'tokens.json'), 'w',
                      encoding='utf-8') as f:
                _json.dump({"input_length": input_len,
                            "total_length": len(all_ids),
                            "tokens": token_strings}, f, ensure_ascii=False, indent=2)
                f.write("\n")

            self._attn_count += 1
        else:
            gen_out = self.model.mmu_generate(input_ids, **generation_kwargs)
            debug_info = None
            if isinstance(gen_out, tuple) and len(gen_out) == 2:
                output_ids, debug_info = gen_out
            else:
                output_ids = gen_out

        if (
            debug_info is not None
            and isinstance(debug_info, dict)
            and "debug_records" in debug_info
            and self.decode_strategy in ('cv_dcd', 'causal_dcd', 'grounded_dcd')
        ):
            try:
                from attention_analysis.cv_debug_io import save_cv_debug_npz
            except ImportError:
                import sys
                attn_root = os.path.join(
                    os.path.dirname(__file__), '../../../attention_analysis'
                )
                sys.path.insert(0, os.path.abspath(attn_root))
                from cv_debug_io import save_cv_debug_npz
            debug_dir = os.getenv(
                'MMADA_CV_DEBUG_DIR',
                os.path.join(self.attn_dir, 'cv_debug'),
            )
            os.makedirs(debug_dir, exist_ok=True)
            sample_tag = os.getenv('MMADA_CURRENT_INDEX', str(self._attn_count))
            npz_path = os.path.join(debug_dir, f"cv_debug_{self.run_id}_{sample_tag}.npz")
            save_cv_debug_npz(
                debug_info["debug_records"],
                npz_path,
                meta={
                    "dataset": dataset,
                    "run_id": self.run_id,
                    "image": image_path,
                    "prompt": prompt,
                    "causal_lambda": self.cv_causal_lambda,
                    "cv_alpha": self.cv_alpha,
                    "cv_mode": self.cv_mode,
                    "cv_conf_source": self.cv_conf_source,
                    "cv_gate_tau": self.cv_gate_tau,
                    "image_drop": self.cv_image_drop,
                    "defer_veto_type": self.defer_veto_type,
                    "defer_tau": self.defer_tau,
                    "defer_beta": self.defer_beta,
                    "defer_gain_type": self.defer_gain_type,
                    "nfe": debug_info.get("nfe"),
                },
            )

        if (
            debug_info is not None
            and isinstance(debug_info, dict)
            and self.decode_strategy in ('vchd', 'vchd_fixed')
        ):
            report_dir = os.getenv('MMADA_VCHD_REPORT_DIR')
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
                sample_tag = os.getenv(
                    'MMADA_CURRENT_INDEX', str(self._attn_count)
                )
                safe_sample_tag = str(sample_tag).replace(os.sep, '_')
                report_path = os.path.join(
                    report_dir,
                    f'vchd_report_{self.run_id}_{safe_sample_tag}_'
                    f'{self._vchd_report_count:06d}.json',
                )
                temporary_path = report_path + '.tmp'
                payload = {
                    "dataset": dataset,
                    "dataset_index": sample_tag,
                    "run_id": self.run_id,
                    "image": image_path,
                    "prompt": prompt,
                    "model_path": self.model_path,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "input_length": int(input_ids.shape[1]),
                    "image_span": [
                        2,
                        2 + int(image_tokens.shape[1]),
                    ],
                    "vchd_config": {
                        "alpha": self.vchd_config.alpha,
                        "beta": self.vchd_config.beta,
                        "tau_base": self.vchd_config.tau_base,
                        "tau_contrast": self.vchd_config.tau_contrast,
                        "mask_capacity": self.vchd_config.mask_capacity,
                        "max_commit_per_iteration": (
                            self.vchd_config.max_commit_per_iteration
                        ),
                        "history_enabled": (
                            self.vchd_config.history_enabled
                        ),
                        "history_penalty_scale": (
                            self.vchd_config.history_penalty_scale
                        ),
                        "history_anchor_min_consistent": (
                            self.vchd_config.history_anchor_min_consistent
                        ),
                        "ccaw_enabled": self.vchd_config.ccaw_enabled,
                        "ccaw_mode": self.vchd_config.ccaw_mode,
                        "ccaw_block_size": (
                            self.vchd_config.ccaw_block_size
                        ),
                        "ccaw_min_commit_per_iteration": (
                            self.vchd_config.ccaw_min_commit_per_iteration
                        ),
                        "ccaw_qualified_budget": (
                            self.vchd_config.ccaw_qualified_budget
                        ),
                        "cache_type": self.vchd_config.cache_type,
                        "force_math_sdpa": (
                            self.vchd_config.force_math_sdpa
                        ),
                        "text_vocab_size": (
                            self.vchd_config.text_vocab_size
                        ),
                        "eos_token_id": self.vchd_config.eos_token_id,
                    },
                    "report": debug_info,
                }
                with open(temporary_path, 'w', encoding='utf-8') as handle:
                    _json.dump(
                        payload,
                        handle,
                        ensure_ascii=False,
                        separators=(',', ':'),
                    )
                    handle.write('\n')
                os.replace(temporary_path, report_path)
                self._vchd_report_count += 1
        
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
