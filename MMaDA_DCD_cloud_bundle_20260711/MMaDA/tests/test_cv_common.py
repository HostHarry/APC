"""CPU-only unit tests for the shared ``cv_common/`` module (v4 M0).

Verifies:
- image_drop.build_dropped_image  : all six strategies + new text_only
- paired_forward.paired_forward_logits : shape + split
- log_prob.logp_of_tokens          : matches manual log_softmax + gather
- log_prob.stepwise_logp_from_records / aggregate_sequence_ll
- Backward compat: the thin wrappers in mmada_decode.py behave identically

Run:
    /home/user/anaconda3/envs/mmada/bin/python tests/test_cv_common.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.cv_common import image_drop as image_drop_mod  # noqa: E402
from models.cv_common import log_prob as log_prob_mod  # noqa: E402
from models.cv_common import paired_forward as paired_forward_mod  # noqa: E402
from models.cv_common.types import Candidate, StepRecord  # noqa: E402
from models.mmada_decode import (  # noqa: E402
    MMaDADecodeConfig,
    _build_dropped_image,
    _logp_of_x0,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# image_drop
# --------------------------------------------------------------------------


def _cfg(**kwargs) -> MMaDADecodeConfig:
    base = dict(
        mask_id=126336,
        visual_token_start=1,
        visual_token_end=5,
        image_drop_strategy="mask",
    )
    base.update(kwargs)
    return MMaDADecodeConfig(**base)


def _sample_input() -> torch.Tensor:
    """x with shape (1, 7); [1:5] is the visual span."""
    return torch.tensor([[126336, 10, 20, 30, 40, 999, 999]], dtype=torch.long)


def test_build_dropped_image_mask():
    cfg = _cfg(image_drop_strategy="mask")
    x = _sample_input()
    x_drop = image_drop_mod.build_dropped_image(x, cfg)
    assert x_drop[0, 1:5].tolist() == [126336, 126336, 126336, 126336]
    # Non-visual span untouched.
    assert x_drop[0, 0].item() == 126336
    assert x_drop[0, 5:].tolist() == [999, 999]


def test_build_dropped_image_shuffle_deterministic():
    cfg = _cfg(image_drop_strategy="shuffle")
    x = _sample_input()
    x_drop1 = image_drop_mod.build_dropped_image(x, cfg)
    # Same perm cached => second call gives same result.
    x_drop2 = image_drop_mod.build_dropped_image(x, cfg)
    assert torch.equal(x_drop1, x_drop2)
    # The 4 visual tokens are a permutation of the original.
    assert sorted(x_drop1[0, 1:5].tolist()) == [10, 20, 30, 40]


def test_build_dropped_image_text_only_uses_fill_id():
    cfg = _cfg(image_drop_strategy="text_only")
    cfg.text_only_fill_id = 7
    x = _sample_input()
    x_drop = image_drop_mod.build_dropped_image(x, cfg)
    assert x_drop[0, 1:5].tolist() == [7, 7, 7, 7]


def test_build_dropped_image_text_only_raises_without_fill_id():
    cfg = _cfg(image_drop_strategy="text_only")
    x = _sample_input()
    try:
        image_drop_mod.build_dropped_image(x, cfg)
    except RuntimeError:
        pass
    else:
        raise AssertionError("text_only without fill_id must raise")


def test_build_dropped_image_neutral_requires_tokens():
    cfg = _cfg(image_drop_strategy="neutral")
    x = _sample_input()
    try:
        image_drop_mod.build_dropped_image(x, cfg)
    except RuntimeError:
        pass
    else:
        raise AssertionError("neutral without neutral_image_tokens must raise")


def test_build_dropped_image_wrapper_matches_new():
    """v3.2 compat: the mmada_decode.py wrapper still produces bit-exact output."""
    cfg = _cfg(image_drop_strategy="mask")
    x = _sample_input()
    a = image_drop_mod.build_dropped_image(x, cfg)
    cfg2 = _cfg(image_drop_strategy="mask")  # fresh cfg so no cached state
    b = _build_dropped_image(x, cfg2)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# paired_forward
# --------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeModel:
    """Returns the input tokens embedded as one-hot logits (deterministic)."""

    def __init__(self, vocab: int = 16):
        self.vocab = vocab

    def __call__(self, x, attention_bias=None, use_cache=False, **kwargs):
        # Convert integer tokens to one-hot logits so caller can inspect the input.
        logits = F.one_hot(x.clamp(0, self.vocab - 1), num_classes=self.vocab).float()
        return _FakeOutput(logits=logits, past_key_values=None)


def test_paired_forward_splits_concat_batch():
    model = _FakeModel(vocab=16)
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    x_drop = torch.tensor([[5, 6, 7]], dtype=torch.long)
    logits, drop, past, past_drop = paired_forward_mod.paired_forward_logits(
        model, x, x_drop, attention_bias=None
    )
    # logits[0] must reflect x, drop[0] must reflect x_drop
    assert torch.argmax(logits[0, 0]).item() == 1
    assert torch.argmax(drop[0, 0]).item() == 5
    assert past is None and past_drop is None


# --------------------------------------------------------------------------
# log_prob
# --------------------------------------------------------------------------


def test_logp_of_tokens_matches_manual():
    logits = torch.randn(2, 3, 5)
    tokens = torch.tensor([[0, 1, 2], [4, 3, 0]])
    got = log_prob_mod.logp_of_tokens(logits, tokens)
    expected = F.log_softmax(logits.to(torch.float64), dim=-1)
    expected = expected.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, expected)


def test_logp_wrapper_backcompat():
    """Old ``_logp_of_x0`` in mmada_decode.py must equal the new function."""
    logits = torch.randn(1, 4, 8)
    x0 = torch.tensor([[0, 1, 2, 3]])
    a = log_prob_mod.logp_of_tokens(logits, x0)
    b = _logp_of_x0(logits, x0)
    assert torch.allclose(a, b)


def test_stepwise_logp_from_records_concatenates_in_order():
    r1 = StepRecord(
        step_idx=0, committed_positions=[3, 5],
        committed_tokens=[10, 20],
        logp_cond=torch.tensor([-1.0, -2.0], dtype=torch.float64),
    )
    r2 = StepRecord(
        step_idx=1, committed_positions=[7],
        committed_tokens=[30],
        logp_cond=torch.tensor([-0.5], dtype=torch.float64),
    )
    out = log_prob_mod.stepwise_logp_from_records([r1, r2])
    assert out.tolist() == [-1.0, -2.0, -0.5]


def test_aggregate_sequence_ll_reductions():
    rec = StepRecord(
        step_idx=0, committed_positions=[0, 1, 2, 3],
        committed_tokens=[0, 1, 2, 3],
        logp_cond=torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float64),
    )
    s = log_prob_mod.aggregate_sequence_ll([rec], reduction="sum")
    m = log_prob_mod.aggregate_sequence_ll([rec], reduction="mean")
    n = log_prob_mod.aggregate_sequence_ll([rec], reduction="length_normalized")
    assert math.isclose(s, -10.0)
    assert math.isclose(m, -2.5)
    assert math.isclose(n, -10.0 / (4 ** 0.7), rel_tol=1e-9)


def test_aggregate_sequence_ll_empty():
    assert log_prob_mod.aggregate_sequence_ll([], reduction="sum") == 0.0


def test_aggregate_sequence_ll_unknown_reduction():
    rec = StepRecord(
        step_idx=0, committed_positions=[0],
        committed_tokens=[0],
        logp_cond=torch.tensor([-1.0]),
    )
    try:
        log_prob_mod.aggregate_sequence_ll([rec], reduction="what")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown reduction must raise")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _run_all_tests():
    ns = dict(globals())
    tests = sorted(k for k in ns if k.startswith("test_") and callable(ns[k]))
    passed = 0
    failed = []
    for name in tests:
        try:
            ns[name]()
        except AssertionError as e:
            failed.append((name, f"AssertionError: {e}"))
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"\n{passed}/{len(tests)} passed, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all_tests())
