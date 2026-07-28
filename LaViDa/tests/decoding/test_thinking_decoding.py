from __future__ import annotations

import math

import torch

from llava.decoding.generate_utils import (
    coerce_thinking_config,
    extract_decode_options,
)
from llava.decoding.thinking import (
    SwdState,
    ThinkingDecodeConfig,
    apply_psp,
    apply_swd,
    apply_vrg,
    kl_prev_vs_curr,
    response_span_token_probs,
    stability_weight,
    visual_guided_logits,
)
from llava.model.language_model.llada.generate import generate


class _Output:
    def __init__(self, logits, attn_key_values=None):
        self.logits = logits
        self.attn_key_values = attn_key_values


class _ToyLlada:
    def __init__(self, vocab_size=5, hidden_size=3):
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        weight = torch.arange(
            vocab_size * hidden_size, dtype=torch.float32
        ).reshape(vocab_size, hidden_size)
        self.transformer = type("Transformer", (), {})()
        self.transformer.wte = lambda token_ids: weight[token_ids]
        self.last_batch = None
        self.last_attention_bias = None

    def __call__(
        self,
        input_ids=None,
        input_embeddings=None,
        attention_mask=None,
        attention_bias=None,
        past_key_values=None,
        use_cache=False,
    ):
        del input_ids, attention_mask, past_key_values
        batch, seq_len, _ = input_embeddings.shape
        self.last_batch = batch
        self.last_attention_bias = attention_bias
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        logits[:, :, 1] = 5.0
        if attention_bias is not None:
            blocked = (attention_bias.reshape(batch, -1) < -1.0e20).sum(-1)
            ablated = blocked > blocked.min()
            logits[ablated, :, 1] = 0.0
            logits[ablated, :, 2] = 5.0
        cache = None
        if use_cache:
            cache_tensor = torch.zeros(batch, 1, seq_len, 1)
            cache = [(cache_tensor, cache_tensor.clone())]
        return _Output(logits, cache)


def test_strategy_presets_and_flat_overrides():
    kwargs = {
        "decode_strategy": "psp_vrg",
        "thinking__psp_gamma": "0.25",
        "thinking__vrg_scale": "0.75",
    }
    strategy, raw_config = extract_decode_options(kwargs)
    config = coerce_thinking_config(strategy, raw_config)
    assert kwargs == {}
    assert strategy == "psp_vrg"
    assert config.psp_enabled
    assert config.vrg_enabled
    assert config.psp_gamma == 0.25
    assert config.vrg_scale == 0.75


def test_vrg_zero_scale_recovers_visual_logits():
    visual = torch.tensor([[2.0, 0.0]])
    ablated = torch.tensor([[0.0, 3.0]])
    assert torch.allclose(apply_vrg(visual, ablated, 0.0), visual)


def test_psp_penalizes_late_positions_early():
    confidence = torch.ones(1, 8)
    early = apply_psp(
        confidence,
        step_index=0,
        num_steps=8,
        response_start=0,
        response_end=8,
        gamma=0.5,
    )
    final = apply_psp(
        confidence,
        step_index=7,
        num_steps=8,
        response_start=0,
        response_end=8,
        gamma=0.5,
    )
    assert early[0, -1] < early[0, 0]
    assert torch.allclose(final, confidence)


def test_swd_uses_previous_to_current_kl_and_updates_state():
    previous = torch.tensor([[[0.7, 0.3]]])
    current = torch.tensor([[[0.2, 0.8]]])
    expected = 0.7 * math.log(0.7 / 0.2) + 0.3 * math.log(0.3 / 0.8)
    assert abs(float(kl_prev_vs_curr(previous, current)) - expected) < 1.0e-5
    assert torch.allclose(
        stability_weight(current, current, 5.0), torch.ones(1, 1)
    )

    logits = torch.randn(1, 4, 3)
    predicted = logits.argmax(dim=-1)
    confidence, probabilities = response_span_token_probs(
        logits, predicted, response_start=1, response_end=4
    )
    state = SwdState()
    output = apply_swd(
        confidence,
        logits,
        state,
        lambda_=1.0,
        response_start=1,
        response_end=4,
        mask_index=torch.tensor([[False, True, True, True]]),
        current_probs=probabilities,
    )
    assert output.shape == confidence.shape
    assert state.prev_probs is not None
    assert state.prev_probs.shape == (1, 3, 3)


def test_visual_guided_logits_runs_paired_access_branches():
    model = _ToyLlada()
    embeddings = torch.randn(1, 5, 3)
    logits = visual_guided_logits(
        model,
        embeddings,
        torch.tensor([False, True]),
        scale=0.5,
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        force_math_sdpa=False,
    )
    assert logits.shape == (1, 5, 5)
    assert model.last_batch == 2
    assert model.last_attention_bias.shape == (2, 1, 5, 5)
    assert torch.equal(logits.argmax(dim=-1), torch.ones(1, 5, dtype=torch.long))


def test_visual_guided_logits_no_causal_overlay_in_cache_path():
    """VRG must NOT AND a causal mask onto branch_bias when using cache.

    LaViDa's Prefix-DLM keeps answer tokens bidirectional even in the
    cached path, so an earlier attempt to torch.minimum branch_bias with
    a causal bias was wrong. This test guards against re-adding it.
    """

    model = _ToyLlada()
    embeddings = torch.randn(1, 4, 3)
    # Dummy cache with two prompt-length keys so past_len=2.
    fake_cache = [(torch.zeros(2, 1, 2, 1), torch.zeros(2, 1, 2, 1))]

    _ = visual_guided_logits(
        model,
        embeddings,
        visual_mask=torch.tensor([False, True]),
        scale=0.0,
        past_key_values=fake_cache,
        force_math_sdpa=False,
    )
    bias = model.last_attention_bias
    seq_len = 2 + 4  # past_len + query_len
    assert bias.shape == (2, 1, seq_len, seq_len)
    finfo = torch.finfo(bias.dtype)
    # Branch 0 (visual) must be fully zero -- no causal, no ablation.
    assert torch.all(bias[0] == 0.0)
    # Branch 1 (ablated) must ONLY block non-visual queries from the
    # visual column (index 1 in the visual_mask). Every other position
    # -- including q>k pairs that a causal overlay would block -- stays
    # exactly zero.
    ablated = bias[1, 0]
    for q in range(seq_len):
        for k in range(seq_len):
            expected = 0.0
            if k == 1 and q != 1:
                expected = finfo.min
            assert float(ablated[q, k]) == expected, (
                f"branch-1 bias mismatch at ({q},{k}): "
                f"got {float(ablated[q, k])}, expected {expected}"
            )


def test_llada_generate_integrates_cached_vrg_and_completes_response():
    model = _ToyLlada()
    output = generate(
        model,
        inputs_embeds=torch.randn(1, 2, 3),
        max_new_tokens=4,
        block_length=4,
        temperature=0.0,
        prefix_lm=True,
        mask_id=4,
        thinking_config=ThinkingDecodeConfig(
            vrg_enabled=True,
            vrg_scale=0.5,
            force_math_sdpa=False,
        ),
        visual_mask=torch.tensor([False, True]),
    )
    assert output.shape == (1, 4)
    assert torch.equal(output, torch.ones(1, 4, dtype=torch.long))
    assert model.last_batch == 2
