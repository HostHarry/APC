"""Smoke tests for Causal-Visual DCD decoding."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mmada_decode import (  # noqa: E402
    MMaDADecodeConfig,
    _build_dropped_image,
    _cv_scores_from_paired_logits,
    dispatch_cv_dcd_decode_text,
    dispatch_dcd_decode_text,
)


MASK_ID = 126336
SOI_ID = 126084
EOI_ID = 126085


class _MockOut:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


class MockMMaDAModel:
    """Minimal LM stub with dual-cache compatible forward."""

    def __init__(self, vocab_size: int = 200000, n_layers: int = 2):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.config = type("Cfg", (), {"n_layers": n_layers})()

    def _logits_from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len = x.shape
        logits = torch.randn(b, seq_len, self.vocab_size, dtype=torch.float32)
        for bi in range(b):
            for si in range(seq_len):
                tid = int(x[bi, si].item())
                logits[bi, si, :] = -5.0
                logits[bi, si, tid % self.vocab_size] = 8.0
                if tid == MASK_ID:
                    logits[bi, si, 42] = 9.0
        return logits

    def _make_past(self, x: torch.Tensor):
        b, seq_len = x.shape
        past = []
        for _ in range(self.n_layers):
            k = torch.randn(b, 4, seq_len, 16)
            v = torch.randn(b, 4, seq_len, 16)
            past.append((k, v))
        return past

    def __call__(
        self,
        x,
        attention_bias=None,
        use_cache=False,
        past_key_values=None,
        replace_position=None,
    ):
        logits = self._logits_from_tokens(x)
        past = self._make_past(x) if use_cache else None
        return _MockOut(logits, past)


def _make_prompt_and_tokens(prompt_len: int = 1100, gen_len: int = 32):
    """Layout: [mmu, soi, img(16), eoi, text..., mask gen region]."""
    text_tail = list(range(200, 200 + (prompt_len - 19)))
    seq = [1, SOI_ID] + list(range(100, 116)) + [EOI_ID] + text_tail
    assert len(seq) == prompt_len, f"seq len {len(seq)} != prompt_len {prompt_len}"
    idx = torch.tensor([seq], dtype=torch.long)
    mask_id = MASK_ID
    x = torch.full((1, prompt_len + gen_len), mask_id, dtype=torch.long)
    x[:, :prompt_len] = idx
    return x, prompt_len, prompt_len + gen_len


def _base_config(**kwargs) -> MMaDADecodeConfig:
    cfg = MMaDADecodeConfig(
        mask_id=MASK_ID,
        block_size=16,
        decode_algo="threshold",
        decode_param=0.5,
        temperature=0.0,
        cache_type="dual",
        visual_token_start=2,
        visual_token_end=18,
        causal_lambda=0.0,
        causal_clip=4.0,
        cv_stride=1,
        image_drop_strategy="mask",
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_build_dropped_image_strategies():
    x, _, _ = _make_prompt_and_tokens()
    for strategy in ("mask", "shuffle", "random_mask", "mean_token"):
        cfg = _base_config(image_drop_strategy=strategy)
        cfg._drop_perm = None
        x_drop = _build_dropped_image(x, cfg)
        assert x_drop.shape == x.shape
        if strategy == "mask":
            assert (x_drop[:, cfg.visual_token_start : cfg.visual_token_end] == MASK_ID).all()


def test_cv_scores_finite():
    logits = torch.randn(1, 8, 500)
    drop_logits = torch.randn(1, 8, 500)
    x0 = torch.randint(0, 500, (1, 8))
    cfg = _base_config(causal_lambda=0.5)
    raw, cv = _cv_scores_from_paired_logits(logits, drop_logits, x0, cfg)
    assert torch.isfinite(raw).all()
    assert torch.isfinite(cv).all()


def test_lambda_zero_matches_dcd_dual_cache():
    model = MockMMaDAModel()
    x, decode_start, decode_end = _make_prompt_and_tokens()
    dcd_cfg = _base_config(causal_lambda=0.0)
    cv_cfg = _base_config(causal_lambda=0.0)

    out_dcd = dispatch_dcd_decode_text(
        model, x.clone(), decode_start, decode_end, dcd_cfg
    )
    out_cv = dispatch_cv_dcd_decode_text(
        model, x.clone(), decode_start, decode_end, cv_cfg
    )
    assert torch.equal(out_dcd, out_cv), "lambda=0 CV-DCD must match DCD token-wise"


def test_lambda_half_mask_runs_without_nan():
    model = MockMMaDAModel()
    x, decode_start, decode_end = _make_prompt_and_tokens()
    cfg = _base_config(causal_lambda=0.5, image_drop_strategy="mask")
    out = dispatch_cv_dcd_decode_text(model, x.clone(), decode_start, decode_end, cfg)
    assert torch.isfinite(out.float()).all()


def test_all_drop_strategies_dual_cache():
    model = MockMMaDAModel()
    x, decode_start, decode_end = _make_prompt_and_tokens()
    for strategy in ("mask", "shuffle", "random_mask"):
        cfg = _base_config(causal_lambda=0.5, image_drop_strategy=strategy)
        cfg._drop_perm = None
        out = dispatch_cv_dcd_decode_text(model, x.clone(), decode_start, decode_end, cfg)
        assert out.shape == x.shape


def test_return_debug_records():
    model = MockMMaDAModel()
    x, decode_start, decode_end = _make_prompt_and_tokens()
    cfg = _base_config(causal_lambda=0.5, return_debug=True)
    tokens_out, debug_info = dispatch_cv_dcd_decode_text(
        model, x.clone(), decode_start, decode_end, cfg
    )
    assert tokens_out.shape == x.shape
    assert "debug_records" in debug_info
    assert isinstance(debug_info["debug_records"], list)
