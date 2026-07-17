"""Smoke tests for CV-DCD v3.1 APC-style contrastive decoding."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mmada_decode import (  # noqa: E402
    MMaDADecodeConfig,
    _apply_cd_style,
    _apply_naive_contrastive,
    _pick_transfer,
    _pick_transfer_cv,
    _resolve_cv_confidence,
)


def _cfg(**kwargs) -> MMaDADecodeConfig:
    cfg = MMaDADecodeConfig(
        mask_id=126336,
        decode_algo="threshold",
        decode_param=0.9,
        temperature=0.0,
        remasking="low_confidence",
        causal_lambda=0.5,
        cv_alpha=0.1,
        cv_mode="cd_apc",
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_cd_style_applies_apc_mask_and_blend():
    logits = torch.tensor([[[4.0, 2.0, 1.0, -5.0]]])
    drop_logits = torch.tensor([[[3.0, 3.0, -1.0, -5.0]]])
    cfg = _cfg(causal_lambda=0.5, cv_alpha=0.1)

    out = _apply_cd_style(logits, drop_logits, cfg)

    expected_blended = logits + 0.5 * (logits - drop_logits)
    assert torch.allclose(out[..., 0], expected_blended[..., 0])
    assert torch.allclose(out[..., 1], expected_blended[..., 1])
    # alpha=0.1 keeps logits >= max + log(0.1) ~= 1.697, so tokens 2 and 3 are masked.
    assert out[..., 2].item() == torch.finfo(out.dtype).min
    assert out[..., 3].item() == torch.finfo(out.dtype).min


def test_cd_naive_blends_without_apc_mask():
    logits = torch.tensor([[[4.0, 2.0, 1.0, -5.0]]])
    drop_logits = torch.tensor([[[3.0, 3.0, -1.0, -5.0]]])
    cfg = _cfg(causal_lambda=0.5)

    out = _apply_naive_contrastive(logits, drop_logits, cfg)

    assert torch.allclose(out, logits + 0.5 * (logits - drop_logits))


def test_contrastive_helpers_are_identity_without_drop_or_lambda():
    logits = torch.randn(2, 3, 5)
    drop_logits = torch.randn(2, 3, 5)

    assert _apply_cd_style(logits, None, _cfg(causal_lambda=0.5)) is logits
    assert _apply_naive_contrastive(logits, None, _cfg(causal_lambda=0.5)) is logits
    assert _apply_cd_style(logits, drop_logits, _cfg(causal_lambda=0.0)) is logits
    assert _apply_naive_contrastive(logits, drop_logits, _cfg(causal_lambda=0.0)) is logits


def test_legacy_score_mode_matches_original_cv_confidence_path():
    logits = torch.tensor([[[4.0, 1.0, 0.0, -1.0], [0.0, 2.0, 5.0, -1.0]]])
    drop_logits = torch.tensor([[[3.0, 2.0, 0.0, -1.0], [0.0, 4.0, 1.0, -1.0]]])
    mask_index = torch.tensor([[True, True]])
    current_tokens = torch.full((1, 2), 126336, dtype=torch.long)
    cfg = _cfg(cv_mode="legacy_score", causal_lambda=0.5, decode_param=0.2)

    x0_cand = torch.argmax(logits, dim=-1)
    cv_conf = _resolve_cv_confidence(logits, drop_logits, x0_cand, cfg, step_idx=0)
    expected_x0, expected_transfer = _pick_transfer(
        logits,
        cfg,
        mask_index,
        current_tokens,
        confidence_override=cv_conf,
        step_idx=0,
        base_logits=logits,
        drop_logits=drop_logits,
    )

    actual_x0, actual_transfer = _pick_transfer_cv(
        logits,
        drop_logits,
        cfg,
        mask_index,
        current_tokens,
        step_idx=0,
    )

    assert torch.equal(actual_x0, expected_x0)
    assert torch.equal(actual_transfer, expected_transfer)
