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
