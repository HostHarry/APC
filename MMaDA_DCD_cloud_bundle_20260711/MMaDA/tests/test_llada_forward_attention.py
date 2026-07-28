"""Forward-level attention semantics for MMaDA's LLaDA backbone.

These tests use a minimal-parameter ``LLaDAModel`` with monkey-patched
blocks to observe exactly what ``attention_bias`` reaches each layer
under different call patterns. They protect the following invariants
after the 2026-07-27 mask-merge cleanup:

* No mask kwargs at all -> pure bidirectional; blocks receive
  ``attention_bias=None``.
* ``past_key_values`` alone (with ``kv_grows=True``) must NOT synthesize
  a causal bias -- MDM's cached prefix path must stay bidirectional
  regardless of whether the caller opted into ``replace_position``.
* ``past_key_values`` + ``replace_position`` (dual-delay / VCHD path)
  also stays bidirectional; the ``replace_position`` argument only
  changes how KV updates are written, not the attention pattern.
* An explicit ``attention_bias`` reaches the block intact.
* A 2D 0/1 ``attention_mask`` (HuggingFace-style pad mask) is folded
  into a proper additive bias with ``-inf`` in the padded columns.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.configuration_llada import (  # noqa: E402
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
)
from models.modeling_llada import LLaDAModel  # noqa: E402


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


class _SpyBlock(nn.Module):
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
        replace_position: Optional[torch.Tensor] = None,
    ):
        self.calls.append(
            {
                "attention_bias": None if attention_bias is None else attention_bias.detach().clone(),
                "replace_position": None if replace_position is None else replace_position.detach().clone(),
                "layer_past_shape": None if layer_past is None else tuple(layer_past[0].shape),
                "x_shape": tuple(x.shape),
            }
        )
        if use_cache:
            batch, seq_len, _ = x.shape
            zeros = torch.zeros(batch, 1, seq_len, 1, device=x.device, dtype=x.dtype)
            cache = (zeros, zeros.clone())
        else:
            cache = None
        return x, cache


def _make_model() -> LLaDAModel:
    torch.manual_seed(0)
    model = LLaDAModel(_tiny_config())
    model.eval()
    return model


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


def test_forward_no_mask_at_all_yields_none_bias():
    model = _make_model()
    spy = _install_spy(model)
    with torch.no_grad():
        model.forward(_tokens(3))
    assert len(spy.calls) == 1
    assert spy.calls[0]["attention_bias"] is None


def test_forward_past_kv_alone_stays_bidirectional_kv_grows():
    """Growing-cache path (`replace_position=None`) must not auto-causal."""
    model = _make_model()
    spy = _install_spy(model)
    with torch.no_grad():
        model.forward(_tokens(3), past_key_values=_fake_past(seq_len=4, hidden=1))
    assert spy.calls[0]["attention_bias"] is None


def test_forward_past_kv_with_replace_position_stays_bidirectional():
    """In-place cache path used by dual / dual-delay / VCHD must stay bidirectional."""
    model = _make_model()
    spy = _install_spy(model)
    seq_len = 3
    replace = torch.zeros(1, 4, dtype=torch.bool)
    replace[0, :seq_len] = True
    with torch.no_grad():
        model.forward(
            _tokens(seq_len),
            past_key_values=_fake_past(seq_len=4, hidden=1),
            replace_position=replace,
        )
    assert spy.calls[0]["attention_bias"] is None
    assert spy.calls[0]["replace_position"] is not None


def test_forward_explicit_bias_is_forwarded():
    model = _make_model()
    spy = _install_spy(model)
    seq_len = 3
    bias = torch.zeros(1, 1, seq_len, seq_len)
    bias[..., 0, 2] = torch.finfo(bias.dtype).min
    with torch.no_grad():
        model.forward(_tokens(seq_len), attention_bias=bias)
    received = spy.calls[0]["attention_bias"]
    assert received is not None
    assert torch.equal(received, bias.to(torch.float))


def test_forward_explicit_bias_and_cache_keeps_bias():
    """MMaDA VCHD supplies both `attention_bias` and `past_key_values`."""
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
    received = spy.calls[0]["attention_bias"]
    assert received is not None
    assert torch.equal(received, bias.to(torch.float))


def test_forward_pad_mask_folds_into_additive_bias():
    model = _make_model()
    spy = _install_spy(model)
    seq_len = 4
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    with torch.no_grad():
        model.forward(_tokens(seq_len), attention_mask=attention_mask)
    bias = spy.calls[0]["attention_bias"]
    assert bias is not None
    finfo = torch.finfo(bias.dtype)
    assert torch.all(bias[..., :, :3] == 0.0)
    assert torch.all(bias[..., :, 3] == finfo.min)
