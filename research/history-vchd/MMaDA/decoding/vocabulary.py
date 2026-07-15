from __future__ import annotations

from typing import Iterable, Optional

import torch

from .config import EOSTokenId, normalize_eos_token_ids


def build_valid_text_vocab(
    vocab_size: int,
    *,
    text_vocab_size: Optional[int],
    forbidden_token_ids: Iterable[int],
    eos_token_id: EOSTokenId,
    device: torch.device,
) -> torch.BoolTensor:
    """Return the model-vocabulary mask allowed for answer generation.

    MMaDA places image codebook entries after ``llm_vocab_size``. Restricting
    the valid prefix removes those image tokens, while ``forbidden_token_ids``
    removes structural tokens that still live inside the text-side prefix.
    EOS tokens are restored last because they are legal answer tokens.
    """

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    text_end = vocab_size if text_vocab_size is None else int(text_vocab_size)
    if not 0 < text_end <= vocab_size:
        raise ValueError(
            f"text_vocab_size must be in [1, {vocab_size}], got {text_end}"
        )

    valid = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    valid[:text_end] = True

    for token_id in forbidden_token_ids:
        token_id = int(token_id)
        if 0 <= token_id < vocab_size:
            valid[token_id] = False

    for eos_token_id_value in normalize_eos_token_ids(eos_token_id):
        if not 0 <= eos_token_id_value < text_end:
            raise ValueError(
                f"eos_token_id={eos_token_id_value} is outside text vocabulary "
                f"[0, {text_end})"
            )
        valid[eos_token_id_value] = True

    if not bool(valid.any()):
        raise ValueError("Text-vocabulary filtering removed every token")
    return valid


def mask_invalid_logits(
    logits: torch.Tensor, valid_text_vocab: torch.BoolTensor
) -> torch.Tensor:
    if logits.shape[-1] != valid_text_vocab.numel():
        raise ValueError(
            "Logit vocabulary does not match valid_text_vocab: "
            f"{logits.shape[-1]} != {valid_text_vocab.numel()}"
        )
    return logits.masked_fill(~valid_text_vocab, -torch.inf)
