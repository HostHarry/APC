"""Image-token ablation strategies.

Extracted from ``mmada_decode.py::_build_dropped_image``. Behaviour of the
five original strategies (``mask`` / ``shuffle`` / ``random_mask`` /
``mean_token`` / ``neutral``) is bit-exact with v3.2. v4 adds one new
strategy:

- ``text_only``: replace the entire visual token span with a text-side
  filler token (typically ``pad_token_id`` or ``eos_token_id``). Unlike
  ``mask`` (which uses the diffusion text mask_id and is OOD), this uses a
  token the model has actually seen in text-side training, giving a
  cleaner "language prior" signal for the CFG rerank uncond forward.

The wrapper (``mmada.py``) is responsible for populating
``config.text_only_fill_id`` at model-load time. If unset, ``text_only``
raises RuntimeError with a descriptive message.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:  # pragma: no cover
    from ..mmada_decode import MMaDADecodeConfig


def build_dropped_image(x: torch.Tensor, config: "MMaDADecodeConfig") -> torch.Tensor:
    """Return ``x_drop`` where the visual span [start:end) is ablated.

    ``config._drop_perm`` is used to *cache* the shuffle permutation across
    steps so that a given (batch, sample) always sees the same ablated
    image; callers must reset it (``config._drop_perm = None``) before each
    new decode.
    """
    x_drop = x.clone()
    s, e = config.visual_token_start, config.visual_token_end
    if s >= e or s < 0 or e > x.shape[1]:
        return x_drop

    strategy = config.image_drop_strategy
    span = x[:, s:e].clone()

    if strategy == "mask":
        x_drop[:, s:e] = config.mask_id

    elif strategy == "shuffle":
        if config._drop_perm is None:
            config._drop_perm = []  # type: ignore[assignment]
        perms: List[torch.Tensor] = config._drop_perm  # type: ignore[assignment]
        batch_size = x.shape[0]
        while len(perms) < batch_size:
            perms.append(torch.randperm(span.shape[1], device=x.device))
        for b in range(batch_size):
            x_drop[b, s:e] = span[b, perms[b]]

    elif strategy == "random_mask":
        drop_rate = 0.5
        rand_mask = torch.rand(span.shape, device=x.device) < drop_rate
        x_drop[:, s:e] = torch.where(
            rand_mask, torch.full_like(span, config.mask_id), span
        )

    elif strategy == "mean_token":
        mean_tok = int(span.float().mean().item())
        x_drop[:, s:e] = mean_tok

    elif strategy == "neutral":
        neutral = config.neutral_image_tokens
        if neutral is None:
            raise RuntimeError(
                "image_drop_strategy='neutral' requires config.neutral_image_tokens "
                "to be populated by the wrapper (see mmada.py::_populate_neutral_image_tokens)."
            )
        expected = e - s
        if neutral.shape[-1] != expected:
            raise ValueError(
                f"neutral_image_tokens has length {neutral.shape[-1]} but the "
                f"visual span is {expected} tokens ([{s}, {e}))."
            )
        neutral = neutral.to(device=x.device, dtype=x.dtype)
        x_drop[:, s:e] = neutral.expand(x.shape[0], -1)

    elif strategy == "text_only":
        fill_id = getattr(config, "text_only_fill_id", None)
        if fill_id is None:
            raise RuntimeError(
                "image_drop_strategy='text_only' requires config.text_only_fill_id "
                "(the model's pad_token_id or eos_token_id) to be populated by the "
                "wrapper. Set MMADA_TEXT_ONLY_FILL_ID env var or fix the wrapper."
            )
        x_drop[:, s:e] = int(fill_id)

    else:
        raise NotImplementedError(strategy)

    return x_drop
