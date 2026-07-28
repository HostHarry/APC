"""Forward-level attention semantics for the LaViDa LLaDA backbone.

These tests exercise the mask-merging logic in ``LLaDAModel.forward``
using a minimal-parameter model so we can verify what
``attention_bias``/``block_mask`` combinations the transformer blocks
actually receive. They protect the following invariants:

* When only ``past_key_values`` is provided, no causal bias should be
  auto-generated (MDM's cached prefix path is bidirectional).
* When ``prefix_length`` is provided (no other mask), a Flex
  ``block_mask`` is created and no additive bias reaches the blocks.
* When an explicit ``attention_bias`` is provided, it must reach the
  transformer blocks unchanged.
* When ``attention_mask`` (a 2D 0/1 pad mask) is provided, it must be
  folded into a proper additive bias, even without an explicit
  ``attention_bias``.
* When only ``attention_bias`` is provided together with
  ``past_key_values``, the bias must reach the blocks unchanged with no
  causal contamination.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch
from torch import nn

from llava.model.language_model.llada.configuration_llada import (
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
)
from llava.model.language_model.llada.modeling_llada import (
    LLaDAModel,
)
from llava.model.language_model.llada import modeling_llada as llada_module


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        d_model=16,
        n_heads=2,
        n_kv_heads=None,
        n_layers=1,
        mlp_ratio=1,
        activation_type=ActivationType.swiglu,
        block_type=BlockType.sequential,
        block_group_size=1,
        alibi=False,
        rope=True,
        rope_full_precision=True,
        flash_attention=False,
        attention_dropout=0.0,
        attention_layer_norm=False,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        layer_norm_type=LayerNormType.default,
        input_emb_norm=False,
        max_sequence_length=64,
        rope_theta=10000.0,
        include_qkv_bias=False,
        include_bias=False,
        scale_logits=False,
        vocab_size=32,
        embedding_size=None,
        weight_tying=True,
        eos_token_id=0,
        pad_token_id=0,
        mask_token_id=0,
        init_device="cpu",
        precision=None,
    )


def _make_model() -> LLaDAModel:
    torch.manual_seed(0)
    model = LLaDAModel(_tiny_config())
    model.eval()
    return model


class _SpyBlock(nn.Module):
    """Records the arguments received by an LLaDABlock's forward."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.calls = []

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        block_mask=None,
    ):
        self.calls.append(
            {
                "attention_bias": None if attention_bias is None else attention_bias.detach().clone(),
                "block_mask": block_mask,
                "layer_past_shape": None if layer_past is None else tuple(layer_past[0].shape),
                "x_shape": tuple(x.shape),
            }
        )
        # Return unchanged hidden state and an empty cache tuple when requested.
        if use_cache:
            batch, seq_len, _ = x.shape
            zeros = torch.zeros(batch, 1, seq_len, 1, device=x.device, dtype=x.dtype)
            cache = (zeros, zeros.clone())
        else:
            cache = None
        return x, cache


def _install_spy(model: LLaDAModel) -> _SpyBlock:
    spy = _SpyBlock(model.config.d_model)
    model.transformer.blocks = nn.ModuleList([spy])  # type: ignore[assignment]
    return spy


def _tokens(seq_len: int) -> torch.LongTensor:
    return torch.zeros(1, seq_len, dtype=torch.long)


def _fake_past(seq_len: int, hidden: int, batch: int = 1):
    key = torch.zeros(batch, 1, seq_len, hidden)
    value = torch.zeros(batch, 1, seq_len, hidden)
    return [(key, value)]


def test_forward_with_only_past_kv_does_not_generate_causal_bias():
    """MDM cached prefix path must reach SDPA with attention_bias=None."""
    model = _make_model()
    spy = _install_spy(model)

    past = _fake_past(seq_len=4, hidden=1)
    with torch.no_grad():
        model.forward(_tokens(3), past_key_values=past)

    assert len(spy.calls) == 1
    assert spy.calls[0]["attention_bias"] is None, (
        "past_key_values alone must NOT synthesize a causal bias; "
        "otherwise MDM's cached prefix path silently becomes causal."
    )
    assert spy.calls[0]["block_mask"] is None


def test_forward_with_explicit_bias_forwards_it_unchanged():
    model = _make_model()
    spy = _install_spy(model)

    seq_len = 3
    bias = torch.zeros(1, 1, seq_len, seq_len)
    bias[..., 0, 2] = torch.finfo(bias.dtype).min
    with torch.no_grad():
        model.forward(_tokens(seq_len), attention_bias=bias)

    assert len(spy.calls) == 1
    received = spy.calls[0]["attention_bias"]
    assert received is not None
    assert torch.equal(received, bias.to(torch.float))
    assert spy.calls[0]["block_mask"] is None


def test_forward_with_explicit_bias_and_cache_keeps_bias():
    """When callers supply both `attention_bias` and `past_key_values`,
    the bias must reach the blocks intact (no causal overlay)."""
    model = _make_model()
    spy = _install_spy(model)

    seq_len = 3
    past_len = 2
    bias = torch.zeros(1, 1, past_len + seq_len, past_len + seq_len)
    bias[..., -seq_len:, 0] = torch.finfo(bias.dtype).min
    with torch.no_grad():
        model.forward(
            _tokens(seq_len),
            attention_bias=bias,
            past_key_values=_fake_past(seq_len=past_len, hidden=1),
        )

    assert len(spy.calls) == 1
    received = spy.calls[0]["attention_bias"]
    assert received is not None
    assert torch.equal(received, bias.to(torch.float))


@pytest.mark.skipif(
    llada_module.create_block_mask is None,
    reason="Flex attention (create_block_mask) not enabled; set USE_FLEX_ATTENTION=1 to run.",
)
def test_forward_with_prefix_length_uses_block_mask_only():
    """Prefix-DLM training uses Flex; no additive bias should be materialised."""
    model = _make_model()
    spy = _install_spy(model)

    seq_len = 4
    with torch.no_grad():
        model.forward(
            _tokens(seq_len),
            prefix_length=torch.tensor([2], dtype=torch.long),
        )

    assert len(spy.calls) == 1
    assert spy.calls[0]["attention_bias"] is None
    assert spy.calls[0]["block_mask"] is not None


def test_forward_with_pad_mask_folds_into_additive_bias():
    model = _make_model()
    spy = _install_spy(model)

    seq_len = 4
    # `1` = valid, `0` = padded.
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    with torch.no_grad():
        model.forward(_tokens(seq_len), attention_mask=attention_mask)

    assert len(spy.calls) == 1
    bias = spy.calls[0]["attention_bias"]
    assert bias is not None
    # Padded column should be masked to -inf; valid columns must remain 0.
    finfo = torch.finfo(bias.dtype)
    assert torch.all(bias[..., :, :3] == 0.0)
    assert torch.all(bias[..., :, 3] == finfo.min)


def test_forward_with_no_mask_at_all_passes_none_through():
    model = _make_model()
    spy = _install_spy(model)

    with torch.no_grad():
        model.forward(_tokens(3))

    assert len(spy.calls) == 1
    assert spy.calls[0]["attention_bias"] is None
    assert spy.calls[0]["block_mask"] is None
