from types import SimpleNamespace

import torch
from torch import nn

from llava.model.language_model.llada.modeling_llada import LLaDABlock


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


def test_llada_attention_applies_additive_bias():
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


def test_llada_attention_bias_not_dropped_when_block_mask_present():
    """Additive bias must win over flex block_mask to avoid silent ablations."""
    block = _minimal_attention_block()

    def _boom(*_args, **_kwargs):
        raise AssertionError("flex path must not run when attention_bias is set")

    block._flex_attention = _boom  # type: ignore[method-assign]
    q = torch.tensor([[[1.0], [1.0]]])
    k = torch.tensor([[[1.0], [2.0]]])
    v = torch.tensor([[[3.0], [9.0]]])
    attention_bias = torch.zeros(1, 1, 2, 2)
    attention_bias[..., 1] = torch.finfo(attention_bias.dtype).min

    masked, _ = block.attention(
        q, k, v, attention_bias=attention_bias, block_mask=object()
    )
    assert torch.allclose(masked, torch.full_like(masked, 3.0))


def test_llada_attention_bidirectional_when_no_bias():
    """No bias, no block_mask must yield fully bidirectional attention."""
    block = _minimal_attention_block()

    def _boom(*_args, **_kwargs):
        raise AssertionError("flex path must not run without block_mask")

    block._flex_attention = _boom  # type: ignore[method-assign]

    q = torch.tensor([[[1.0], [1.0]]])
    k = torch.tensor([[[1.0], [1.0]]])
    v = torch.tensor([[[3.0], [9.0]]])
    output, _ = block.attention(q, k, v)
    # With identical queries and keys, both positions produce (v0 + v1) / 2.
    expected = torch.tensor([[[6.0], [6.0]]])
    assert torch.allclose(output, expected)


def test_llada_attention_zero_bias_equals_no_bias():
    """A pure-zero additive bias is identity, matching the unbiased call."""
    block = _minimal_attention_block()

    q = torch.tensor([[[1.0], [1.0]]])
    k = torch.tensor([[[1.0], [2.0]]])
    v = torch.tensor([[[3.0], [9.0]]])

    zero_bias = torch.zeros(1, 1, 2, 2)
    biased, _ = block.attention(q, k, v, attention_bias=zero_bias)
    unbiased, _ = block.attention(q, k, v)
    assert torch.allclose(biased, unbiased)


def test_llada_attention_with_layer_past_extends_keys():
    """Passing `layer_past` must concat past K/V and attend over the full span."""
    block = _minimal_attention_block()

    # LLaDABlock.attention expects past_key/past_value to already be 4D
    # (batch, n_kv_heads, seq_len, head_dim), matching the current K/V
    # after the view+transpose inside the method.
    past_k = torch.tensor([[[[5.0]]]])
    past_v = torch.tensor([[[[7.0]]]])

    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[5.0]]])
    v = torch.tensor([[[7.0]]])
    output, present = block.attention(
        q, k, v, layer_past=(past_k, past_v), use_cache=True
    )
    # Past and current keys are identical; average of the two values is 7.
    assert torch.allclose(output, torch.tensor([[[7.0]]]))
    assert present is not None
    past_key_present, past_value_present = present
    assert past_key_present.shape[-2] == 2  # past + current key
    assert past_value_present.shape[-2] == 2


def test_llada_attention_paired_bias_ablated_blocks_visual_only():
    """The VRG/VCHD paired branch must block only answer→visual edges."""

    from llava.decoding.lavida_adapter import build_paired_attention_bias_from_mask

    block = _minimal_attention_block()
    # Positions: 0=visual, 1=answer, 2=answer.
    visual_mask = torch.tensor([True, False, False])
    branch_bias = build_paired_attention_bias_from_mask(visual_mask)
    assert branch_bias.shape == (2, 1, 3, 3)

    # Visual branch (branch 0) should be fully bidirectional -- all zeros.
    assert torch.all(branch_bias[0] == 0.0)

    # Ablated branch (branch 1) blocks non-visual queries from attending to
    # the visual column only. Answer→answer must remain open.
    inf = torch.finfo(branch_bias.dtype).min
    assert branch_bias[1, 0, 1, 0] == inf  # answer q -> visual k blocked
    assert branch_bias[1, 0, 2, 0] == inf
    assert branch_bias[1, 0, 1, 1] == 0.0  # answer q -> answer k open
    assert branch_bias[1, 0, 1, 2] == 0.0
    assert branch_bias[1, 0, 2, 1] == 0.0
    assert branch_bias[1, 0, 0, 0] == 0.0  # visual q -> visual k open
    assert branch_bias[1, 0, 0, 1] == 0.0  # visual q -> answer k open
    assert branch_bias[1, 0, 0, 2] == 0.0
