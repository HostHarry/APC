"""Paired conditional/unconditional forward pass.

Extracted from ``mmada_decode.py::_paired_forward_logits``. Behaviour is
bit-exact with v3.2 -- this is a pure refactor.

Two operating modes:
- ``past_key_values is None``  -> concat (x, x_drop) along batch dim, one
  forward, then split. Used by the initial-block forward in dual-cache
  decoding.
- ``past_key_values is not None`` -> two separate forwards sharing
  ``past_key_values`` (dual-cache block iteration).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def _maybe_expand_attention_bias(
    attention_bias: Optional[torch.Tensor], repeat: int
) -> Optional[torch.Tensor]:
    """Broadcast a 1-batch attention bias to the paired batch dimension."""
    if attention_bias is None:
        return None
    if attention_bias.shape[0] == 1:
        return attention_bias.repeat(repeat, 1, 1, 1)
    return torch.cat([attention_bias] * repeat, dim=0)


def paired_forward_logits(
    model,
    x: torch.Tensor,
    x_drop: torch.Tensor,
    attention_bias: Optional[torch.Tensor],
    past_key_values: Optional[Any] = None,
    use_cache: bool = False,
    replace_position: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
    """Run base and drop-image forwards; returns ``(logits, drop_logits, past_kv, past_kv_drop)``.

    Contract identical to the original ``_paired_forward_logits``:
    - When ``past_key_values is None``, concat & split; ``past_kv_drop`` is
      ``None`` if ``use_cache`` is False (the caller only splits caches on demand).
    - When ``past_key_values is not None``, runs two sequential forwards.
    """
    if past_key_values is None:
        pair_x = torch.cat([x, x_drop], dim=0)
        ab = _maybe_expand_attention_bias(attention_bias, 2)
        out = model(pair_x, attention_bias=ab, use_cache=use_cache)
        pair_logits = out.logits
        logits, drop_logits = torch.chunk(pair_logits, 2, dim=0)
        if use_cache and out.past_key_values is not None:
            past_kv = []
            past_kv_drop = []
            for layer_cache in out.past_key_values:
                k, v = layer_cache
                b = k.shape[0] // 2
                past_kv.append((k[:b], v[:b]))
                past_kv_drop.append((k[b:], v[b:]))
            return logits, drop_logits, past_kv, past_kv_drop
        return logits, drop_logits, out.past_key_values, None

    # Dual-cache block iteration: separate forwards sharing no batch dim.
    kwargs = dict(attention_bias=attention_bias, past_key_values=past_key_values, use_cache=True)
    if replace_position is not None:
        kwargs["replace_position"] = replace_position
    out = model(x, **kwargs)
    kwargs_drop = dict(
        attention_bias=attention_bias,
        past_key_values=past_key_values,
        use_cache=True,
    )
    if replace_position is not None:
        kwargs_drop["replace_position"] = replace_position
    out_drop = model(x_drop, **kwargs_drop)
    return out.logits, out_drop.logits, out.past_key_values, out_drop.past_key_values
