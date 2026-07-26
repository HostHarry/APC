from __future__ import annotations

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.decoding import (
    VCHDDecodeConfig,
    build_paired_attention_bias_from_mask,
    build_paired_prefix_attention_bias,
    infer_visual_mask_from_expanded_ids,
    visual_contrast_decode,
)
from llava.decoding.lavida_adapter import LaViDaVisualAccessAdapter
from llava.model.language_model.llada.configuration_llada import ModelConfig
from llava.model.language_model.llada.modeling_llada import LLaDAModel


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _EmbedModel:
    def __init__(self, hidden=4, vocab=6):
        self.hidden = hidden
        self.vocab = vocab
        self.config = type("Config", (), {"vocab_size": vocab, "llm_vocab_size": 5})()
        self.transformer = type("T", (), {})()
        weight = torch.randn(vocab, hidden)
        self.transformer.wte = lambda ids: weight[ids]

    def __call__(
        self,
        input_ids=None,
        input_embeddings=None,
        attention_mask=None,
        attention_bias=None,
        use_cache=False,
    ):
        del input_ids, attention_mask, use_cache
        assert input_embeddings is not None
        batch, seq_len, _ = input_embeddings.shape
        logits = torch.zeros(batch, seq_len, self.vocab)
        marker = torch.zeros(batch)
        if attention_bias is not None:
            blocked = attention_bias.reshape(batch, -1) < -1.0e20
            marker = blocked.any(dim=-1).float()
        logits[:, :, 1] = 8.0 - 3.0 * marker[:, None]
        logits[:, :, 2] = 2.0
        logits[:, :, 5] = 20.0
        return _Output(logits)


def test_multi_span_visual_mask_blocks_only_non_visual_queries():
    mask = torch.tensor([False, True, True, False, True, False])
    bias = build_paired_attention_bias_from_mask(mask)
    blocked = bias[1, 0] < -1.0e20
    assert blocked[0, 1:3].all()
    assert blocked[0, 4]
    assert blocked[3, 1:3].all()
    assert blocked[5, 4]
    assert not blocked[1:3].any()
    assert not blocked[4].any()
    assert torch.equal(bias[0], torch.zeros_like(bias[0]))


def test_infer_visual_mask_from_expanded_ids():
    ids = torch.tensor([1, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 7, IMAGE_TOKEN_INDEX, 9])
    mask = infer_visual_mask_from_expanded_ids(ids)
    assert mask.tolist() == [False, True, True, False, True, False]


def test_paired_prefix_bias_is_bidirectional_only_inside_prompt():
    visual_mask = torch.tensor([False, True, False, False, False])
    bias = build_paired_prefix_attention_bias(
        visual_mask,
        prompt_length=3,
    )
    visual_blocked = bias[0, 0] < -1.0e20
    assert visual_blocked[:3, 3:].all()
    assert not visual_blocked[:3, :3].any()
    assert visual_blocked[3, 4]
    assert not visual_blocked[4].any()

    ablated_blocked = bias[1, 0] < -1.0e20
    assert ablated_blocked[0, 1]
    assert ablated_blocked[3, 1]
    assert not ablated_blocked[1, 1]


def test_lavida_adapter_decode_terminates():
    model = _EmbedModel()
    prompt_len = 3
    decode_len = 3
    prompt_embeds = torch.randn(1, prompt_len, model.hidden)
    visual_mask = torch.tensor([False, True, False])
    adapter = LaViDaVisualAccessAdapter(
        model,
        prompt_embeds=prompt_embeds,
        visual_mask=visual_mask,
        decode_start=prompt_len,
        decode_end=prompt_len + decode_len,
        mask_id=4,
        force_math_sdpa=False,
        backend="llada",
    )
    tokens = torch.tensor([[0, 2, 3, 4, 4, 4]])
    config = VCHDDecodeConfig(
        mask_id=4,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, 4),
        tau_base=0.5,
        tau_contrast=0.5,
        mask_capacity=3,
        max_physical_span=3,
        max_commit_per_iteration=2,
        force_math_sdpa=False,
        truncate_at_eos=False,
        return_report=True,
        cache_type="none",
    )
    output, report = visual_contrast_decode(
        model,
        tokens,
        decode_start=prompt_len,
        decode_end=prompt_len + decode_len,
        config=config,
        adapter=adapter,
    )
    assert output[0, prompt_len:].tolist() == [1, 1, 1]
    assert report["cache_type"] == "none"


def test_dual_cache_rejected():
    try:
        VCHDDecodeConfig(cache_type="dual").validate()
    except Exception:
        pass
    tokens = torch.tensor([[0, 2, 3, 4, 4]])
    config = VCHDDecodeConfig(
        mask_id=4,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, 4),
        cache_type="dual",
        force_math_sdpa=False,
    )
    # validate allows dual in config, decoder rejects at runtime
    config.validate()
    raised = False
    try:
        visual_contrast_decode(
            _EmbedModel(),
            tokens,
            decode_start=3,
            decode_end=5,
            image_span=(1, 2),
            config=config,
        )
    except ValueError as exc:
        raised = "cache_type" in str(exc)
    assert raised


class _DreamDtypeModel:
    """Captures Dream-path attention bias dtype from the adapter."""

    def __init__(self, hidden=4, vocab=6, dtype=torch.bfloat16):
        self.hidden = hidden
        self.vocab = vocab
        self.dtype = dtype
        self.config = type("Config", (), {"vocab_size": vocab, "llm_vocab_size": 5})()
        self.transformer = type("T", (), {})()
        weight = torch.randn(vocab, hidden, dtype=dtype)
        self.transformer.wte = lambda ids: weight[ids]
        self.seen_bias_dtypes = []

    def forward_dream(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        use_cache=False,
        num_logits_to_keep=0,
    ):
        del input_ids, use_cache, num_logits_to_keep
        assert inputs_embeds is not None
        if attention_mask is not None:
            self.seen_bias_dtypes.append(attention_mask.dtype)
            assert attention_mask.dtype == inputs_embeds.dtype
        batch, seq_len, _ = inputs_embeds.shape
        logits = torch.zeros(batch, seq_len, self.vocab, dtype=torch.float32)
        logits[:, :, 1] = 8.0
        logits[:, :, 2] = 2.0
        return _Output(logits)


def test_dream_attention_bias_cast_to_model_dtype():
    model = _DreamDtypeModel(dtype=torch.bfloat16)
    prompt_len = 3
    decode_len = 2
    prompt_embeds = torch.randn(1, prompt_len, model.hidden, dtype=torch.bfloat16)
    visual_mask = torch.tensor([False, True, False])
    adapter = LaViDaVisualAccessAdapter(
        model,
        prompt_embeds=prompt_embeds,
        visual_mask=visual_mask,
        decode_start=prompt_len,
        decode_end=prompt_len + decode_len,
        mask_id=4,
        force_math_sdpa=False,
        backend="dream",
    )
    tokens = torch.tensor([[0, 2, 3, 4, 4]])
    paired = adapter.paired_forward(tokens)
    assert paired.visual.shape == (decode_len, model.vocab)
    assert model.seen_bias_dtypes
    assert all(dtype == torch.bfloat16 for dtype in model.seen_bias_dtypes)


def test_unknown_decode_strategy_raises():
    from llava.decoding.generate_utils import extract_decode_options

    raised = False
    try:
        extract_decode_options({"decode_strategy": "vchdd"})
    except ValueError as exc:
        raised = "decode_strategy" in str(exc)
    assert raised
    strategy, _ = extract_decode_options({"decode_strategy": "vchd"})
    assert strategy == "vchd"


class _PrefixCacheOutput:
    def __init__(self, logits, attn_key_values=None):
        self.logits = logits
        self.attn_key_values = attn_key_values


class _PrefixCacheAttentionModel:
    """One-layer attention model with LLaDA-compatible prompt KV caching."""

    def __init__(self, embed_weight=None, output_weight=None):
        hidden = 4
        vocab = 6
        self.hidden = hidden
        self.vocab = vocab
        self.embed_weight = (
            torch.randn(vocab, hidden)
            if embed_weight is None
            else embed_weight.clone()
        )
        self.output_weight = (
            torch.randn(hidden, vocab)
            if output_weight is None
            else output_weight.clone()
        )
        self.config = type(
            "Config", (), {"vocab_size": vocab, "llm_vocab_size": vocab}
        )()
        self.transformer = type("T", (), {})()
        self.transformer.wte = lambda ids: self.embed_weight[ids]
        self.calls = []

    def __call__(
        self,
        input_ids=None,
        input_embeddings=None,
        attention_mask=None,
        attention_bias=None,
        past_key_values=None,
        use_cache=False,
        last_logits_only=False,
    ):
        del input_ids, attention_mask
        assert input_embeddings is not None
        batch, query_len, hidden = input_embeddings.shape
        query = input_embeddings[:, None]
        current_key = input_embeddings[:, None]
        current_value = input_embeddings[:, None]
        if past_key_values is None:
            key = current_key
            value = current_value
        else:
            past_key, past_value = past_key_values[0]
            key = torch.cat([past_key, current_key], dim=-2)
            value = torch.cat([past_value, current_value], dim=-2)
        key_len = int(key.shape[-2])
        scores = torch.matmul(query, key.transpose(-1, -2)) / hidden**0.5
        if attention_bias is not None:
            scores = scores + attention_bias[
                :, :, key_len - query_len : key_len, :key_len
            ]
        context = torch.matmul(torch.softmax(scores, dim=-1), value)
        logits = context[:, 0] @ self.output_weight
        cache = ((key.clone(), value.clone()),) if use_cache else None
        if last_logits_only:
            logits = logits[:, -1:]
        self.calls.append(
            {
                "batch": batch,
                "query_len": query_len,
                "key_len": key_len,
                "use_cache": use_cache,
                "has_past": past_key_values is not None,
            }
        )
        return _PrefixCacheOutput(logits, cache)


def test_paired_prefix_prompt_cache_matches_full_prefix_forward():
    torch.manual_seed(7)
    full_model = _PrefixCacheAttentionModel()
    cached_model = _PrefixCacheAttentionModel(
        full_model.embed_weight,
        full_model.output_weight,
    )
    prompt_len = 3
    response_len = 3
    prompt_embeds = torch.randn(1, prompt_len, full_model.hidden)
    visual_mask = torch.tensor([False, True, False])
    common = {
        "prompt_embeds": prompt_embeds,
        "visual_mask": visual_mask,
        "decode_start": prompt_len,
        "decode_end": prompt_len + response_len,
        "mask_id": 4,
        "force_math_sdpa": False,
        "backend": "llada",
        "prefix_lm": True,
    }
    full = LaViDaVisualAccessAdapter(
        full_model,
        prefix_prompt_cache=False,
        **common,
    )
    cached = LaViDaVisualAccessAdapter(
        cached_model,
        prefix_prompt_cache=True,
        **common,
    )
    tokens = torch.tensor([[0, 2, 3, 4, 4, 4]])

    full_first = full.paired_forward(tokens, context_version=0)
    cached_first = cached.paired_forward(tokens, context_version=0)
    assert torch.allclose(cached_first.visual, full_first.visual, atol=1.0e-6)
    assert torch.allclose(cached_first.ablated, full_first.ablated, atol=1.0e-6)
    assert cached_first.cache_event == "prompt_prefill"

    tokens[0, prompt_len] = 1
    full_second = full.paired_forward(tokens, context_version=1)
    cached_second = cached.paired_forward(tokens, context_version=1)
    assert torch.allclose(cached_second.visual, full_second.visual, atol=1.0e-6)
    assert torch.allclose(cached_second.ablated, full_second.ablated, atol=1.0e-6)
    assert cached_second.cache_event == "prompt_reuse"

    report = cached.cache_report()
    assert report["cache_type"] == "none"
    assert report["prompt_cache_type"] == "paired_prefix"
    assert report["prompt_cache_prefills"] == 1
    assert [call["query_len"] for call in cached_model.calls] == [
        prompt_len,
        response_len,
        response_len,
    ]
    assert report["cache_logical_query_tokens"] < full.cache_report()[
        "cache_logical_query_tokens"
    ]


def test_real_llada_honors_paired_bias_and_prompt_cache_parity():
    torch.manual_seed(17)
    config = ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=2,
        mlp_hidden_size=32,
        rope=True,
        max_sequence_length=16,
        vocab_size=8,
        embedding_size=8,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        init_device="cpu",
        init_std=0.2,
    )
    model = LLaDAModel(config, init_params=True).eval()
    prompt_len = 3
    response_len = 3
    prompt_ids = torch.tensor([[0, 1, 2]])
    prompt_embeds = model.transformer.wte(prompt_ids)
    visual_mask = torch.tensor([False, True, False])
    common = {
        "prompt_embeds": prompt_embeds,
        "visual_mask": visual_mask,
        "decode_start": prompt_len,
        "decode_end": prompt_len + response_len,
        "mask_id": 4,
        "force_math_sdpa": True,
        "backend": "llada",
        "prefix_lm": True,
    }
    full = LaViDaVisualAccessAdapter(
        model,
        prefix_prompt_cache=False,
        **common,
    )
    cached = LaViDaVisualAccessAdapter(
        model,
        prefix_prompt_cache=True,
        **common,
    )
    tokens = torch.tensor([[0, 1, 2, 4, 5, 6]])

    full_output = full.paired_forward(tokens)
    cached_output = cached.paired_forward(tokens)

    assert not torch.allclose(
        full_output.visual,
        full_output.ablated,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert torch.allclose(
        cached_output.visual,
        full_output.visual,
        atol=1.0e-5,
        rtol=1.0e-4,
    )
    assert torch.allclose(
        cached_output.ablated,
        full_output.ablated,
        atol=1.0e-5,
        rtol=1.0e-4,
    )
