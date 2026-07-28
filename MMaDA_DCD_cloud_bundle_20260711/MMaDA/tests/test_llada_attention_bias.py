"""Block-level attention semantics for MMaDA's LLaDA backbone.

These tests mirror LaViDa's ``test_llada_attention_bias.py`` for the
MMaDA fork. They guard against the two regressions we saw:

* An explicit additive ``attention_bias`` reaching SDPA (the LaViDa
  code hardcoded ``attn_mask=None`` and silently dropped it -- MMaDA
  already passes ``attn_mask`` through, so this is a permanent
  regression fence, not a fix).
* No auto-generated causal bias when the model is called with only
  ``past_key_values`` (used by ``cache_type=prefix``).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.modeling_llada import LLaDABlock  # noqa: E402


def _minimal_attention_block() -> LLaDABlock:
    block = object.__new__(LLaDABlock)
    nn.Module.__init__(block)
    block.config = SimpleNamespace(
        n_heads=1,
        effective_n_kv_heads=1,
        rope=False,
        attention_dropout=0.0,
    )
    block.q_norm = None
    block.k_norm = None
    block.flash_attn_func = None
    block.attn_out = nn.Identity()
    block.eval()
    return block


def test_mmada_attention_applies_additive_bias():
    block = _minimal_attention_block()
    q = torch.tensor([[[1.0], [1.0]]])
    k = torch.tensor([[[1.0], [2.0]]])
    v = torch.tensor([[[3.0], [9.0]]])
    attention_bias = torch.zeros(1, 1, 2, 2)
    attention_bias[..., 1] = torch.finfo(attention_bias.dtype).min

    masked, _ = block.attention(q, k, v, attention_bias=attention_bias)
    unmasked, _ = block.attention(q, k, v)

    assert torch.allclose(masked, torch.full_like(masked, 3.0))
    assert not torch.allclose(masked, unmasked)


def test_mmada_attention_bidirectional_when_no_bias():
    block = _minimal_attention_block()
    q = torch.tensor([[[1.0], [1.0]]])
    k = torch.tensor([[[1.0], [1.0]]])
    v = torch.tensor([[[3.0], [9.0]]])
    output, _ = block.attention(q, k, v)
    expected = torch.tensor([[[6.0], [6.0]]])
    assert torch.allclose(output, expected)


def test_mmada_attention_zero_bias_equals_no_bias():
    block = _minimal_attention_block()

    q = torch.tensor([[[1.0], [1.0]]])
    k = torch.tensor([[[1.0], [2.0]]])
    v = torch.tensor([[[3.0], [9.0]]])
    biased, _ = block.attention(q, k, v, attention_bias=torch.zeros(1, 1, 2, 2))
    unbiased, _ = block.attention(q, k, v)
    assert torch.allclose(biased, unbiased)


def test_mmada_attention_layer_past_concat_when_no_replace():
    """Without `replace_position`, MMaDA concatenates past K/V (growing cache)."""
    block = _minimal_attention_block()

    # Past K/V are stored in their 4D post-view shape (B, n_kv_heads, T_past, head_dim).
    past_k = torch.tensor([[[[5.0]]]])
    past_v = torch.tensor([[[[7.0]]]])
    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[5.0]]])
    v = torch.tensor([[[7.0]]])
    output, present = block.attention(
        q, k, v, layer_past=(past_k, past_v), use_cache=True
    )
    assert torch.allclose(output, torch.tensor([[[7.0]]]))
    assert present is not None
    past_k_present, past_v_present = present
    assert past_k_present.shape[-2] == 2
    assert past_v_present.shape[-2] == 2


def test_mmada_attention_replace_position_writes_in_place():
    """With `replace_position` set, MMaDA writes into the existing cache."""
    block = _minimal_attention_block()

    past_k = torch.zeros(1, 1, 4, 1)
    past_v = torch.zeros(1, 1, 4, 1)
    past_k[0, 0, 2, 0] = 5.0  # existing value at slot 2
    past_v[0, 0, 2, 0] = 7.0

    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[10.0]]])
    v = torch.tensor([[[20.0]]])
    replace_position = torch.zeros(1, 4, dtype=torch.bool)
    replace_position[0, 3] = True  # write to slot 3

    _, present = block.attention(
        q,
        k,
        v,
        layer_past=(past_k, past_v),
        use_cache=True,
        replace_position=replace_position,
    )

    assert present is not None
    updated_k, updated_v = present
    # Slot 3 should have been overwritten with the new K/V.
    assert updated_k.shape[-2] == 4  # no cache growth
    assert updated_k[0, 0, 3, 0].item() == 10.0
    assert updated_v[0, 0, 3, 0].item() == 20.0
    # Previously written slot 2 must be preserved.
    assert updated_k[0, 0, 2, 0].item() == 5.0
    assert updated_v[0, 0, 2, 0].item() == 7.0
