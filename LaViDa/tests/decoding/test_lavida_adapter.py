from __future__ import annotations

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.decoding import (
    VCHDDecodeConfig,
    build_paired_attention_bias_from_mask,
    infer_visual_mask_from_expanded_ids,
    visual_contrast_decode,
)
from llava.decoding.lavida_adapter import LaViDaVisualAccessAdapter


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
