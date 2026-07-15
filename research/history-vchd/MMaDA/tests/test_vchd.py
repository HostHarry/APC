"""CPU-only tests for the phase 0--2 VCHD dual-gate decoder."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decoding import (  # noqa: E402
    MMaDAVisualAccessAdapter,
    VCHDDecodeConfig,
    build_paired_attention_bias,
    build_valid_text_vocab,
    compute_contrast_stats,
    visual_contrast_decode,
)
from decoding.selector import (  # noqa: E402
    MAX_WINDOW_TOP1_FALLBACK,
    THRESHOLD_COMMIT,
    select_fixed_window_positions,
)


class _Output:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _BiasAwareModel:
    """Tiny attention-like model used to verify visual-access invariance."""

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
        self.config = type(
            "Config",
            (),
            {"vocab_size": vocab_size, "llm_vocab_size": vocab_size},
        )()

    def __call__(
        self,
        *,
        input_ids,
        attention_mask=None,
        attention_bias=None,
        use_cache=False,
    ):
        del attention_mask, use_cache
        values = torch.nn.functional.one_hot(
            input_ids.remainder(self.vocab_size),
            num_classes=self.vocab_size,
        ).float()
        if attention_bias is None:
            allowed = torch.ones(
                input_ids.shape[0],
                input_ids.shape[1],
                input_ids.shape[1],
                dtype=torch.bool,
            )
        else:
            allowed = attention_bias[:, 0] > -1.0e20
        logits = torch.matmul(allowed.float(), values)
        return _Output(logits)


class _FixedBranchModel:
    """Returns deterministic visual/ablated answer distributions."""

    def __init__(self):
        self.config = type(
            "Config", (), {"vocab_size": 6, "llm_vocab_size": 5}
        )()

    def __call__(
        self,
        *,
        input_ids,
        attention_mask=None,
        attention_bias=None,
        use_cache=False,
    ):
        del attention_mask, attention_bias, use_cache
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 6)
        logits[0, :, 1] = 8.0
        logits[0, :, 2] = 2.0
        logits[1, :, 1] = 5.0
        logits[1, :, 2] = 2.0
        logits[:, :, 5] = 20.0  # Image-vocabulary token; must be filtered.
        return _Output(logits)


def _all_text(vocab_size: int) -> torch.BoolTensor:
    return torch.ones(vocab_size, dtype=torch.bool)


def test_paired_bias_blocks_only_non_image_queries_to_image_keys():
    bias = build_paired_attention_bias(6, (1, 3), device=torch.device("cpu"))
    assert bias.shape == (2, 1, 6, 6)
    assert torch.equal(bias[0], torch.zeros_like(bias[0]))

    blocked = bias[1, 0] < -1.0e20
    assert blocked[0, 1:3].all()
    assert blocked[3:, 1:3].all()
    assert not blocked[1:3].any()
    assert not blocked[:, :1].any()
    assert not blocked[:, 3:].any()


def test_ablated_logits_are_invariant_to_vq_replacement():
    model = _BiasAwareModel()
    adapter = MMaDAVisualAccessAdapter(
        model,
        decode_start=4,
        decode_end=5,
        image_span=(1, 3),
        force_math_sdpa=False,
    )
    first = torch.tensor([[0, 5, 6, 3, 4]])
    second = torch.tensor([[0, 9, 10, 3, 4]])

    first_logits = adapter.paired_forward(first)
    second_logits = adapter.paired_forward(second)
    assert torch.equal(first_logits.ablated, second_logits.ablated)
    assert not torch.equal(first_logits.visual, second_logits.visual)


def test_alpha_zero_recovers_normal_visual_confidence():
    visual = torch.tensor([[2.0, 1.0, -1.0]])
    ablated = torch.tensor([[-2.0, 3.0, 0.0]])
    stats = compute_contrast_stats(
        visual, ablated, _all_text(3), alpha=0.0, beta=0.1
    )
    expected = torch.softmax(visual, dim=-1).max(dim=-1).values
    assert torch.equal(stats.raw_token, stats.contrast_token)
    assert torch.allclose(stats.base_confidence, expected)
    assert torch.allclose(stats.contrast_confidence, expected)


def test_dual_gate_rejects_contrast_sharpened_low_base_candidate():
    visual_prob = torch.tensor([[0.60, 0.08, 0.32]])
    ablated_prob = torch.tensor([[0.59, 1.0e-8, 0.41]])
    stats = compute_contrast_stats(
        visual_prob.log(),
        ablated_prob.log(),
        _all_text(3),
        alpha=0.5,
        beta=0.1,
    )
    assert stats.contrast_token.item() == 1
    assert stats.base_confidence.item() < 0.10
    assert stats.contrast_confidence.item() > 0.50

    config = VCHDDecodeConfig(
        tau_base=0.10,
        tau_contrast=0.50,
        mask_capacity=1,
        max_physical_span=1,
    )
    selection = select_fixed_window_positions(
        stats, torch.tensor([True]), config
    )
    assert selection.reason == MAX_WINDOW_TOP1_FALLBACK

    config.tau_base = 0.05
    selection = select_fixed_window_positions(
        stats, torch.tensor([True]), config
    )
    assert selection.reason == THRESHOLD_COMMIT


def test_text_vocab_filter_removes_image_and_control_tokens_but_keeps_eos():
    valid = build_valid_text_vocab(
        10,
        text_vocab_size=7,
        forbidden_token_ids=(0, 2, 6),
        eos_token_id=2,
        device=torch.device("cpu"),
    )
    assert valid.tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_fixed_decoder_commits_multiple_tokens_and_terminates():
    model = _FixedBranchModel()
    mask_id = 4
    tokens = torch.tensor([[0, 2, 3, mask_id, mask_id, mask_id]])
    config = VCHDDecodeConfig(
        mask_id=mask_id,
        text_vocab_size=5,
        forbidden_token_ids=(0, 3, mask_id),
        alpha=0.5,
        beta=0.1,
        tau_base=0.50,
        tau_contrast=0.50,
        mask_capacity=3,
        max_physical_span=3,
        max_commit_per_iteration=2,
        force_math_sdpa=False,
        truncate_at_eos=False,
        collect_trace=True,
        return_report=True,
    )
    output, report = visual_contrast_decode(
        model,
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=config,
    )
    assert output[0, 3:].tolist() == [1, 1, 1]
    assert report["model_evaluations"] == 2
    assert report["threshold_commits"] == 3
    assert report["fallback_commits"] == 0
    assert len(report["trace"]) == 2


def test_contrast_confidence_is_not_assumed_to_be_base_confidence():
    # A regression guard for the known blended-confidence inflation failure.
    visual_prob = torch.tensor([[0.51, 0.06, 0.43]])
    ablated_prob = torch.tensor([[0.50, 1.0e-12, 0.50]])
    stats = compute_contrast_stats(
        visual_prob.log(),
        ablated_prob.log(),
        _all_text(3),
        alpha=0.5,
        beta=0.1,
    )
    assert stats.contrast_token.item() == 1
    assert math.isclose(stats.base_confidence.item(), 0.06, rel_tol=1.0e-5)
    assert stats.contrast_confidence.item() > stats.base_confidence.item()


def _run_all_tests():
    namespace = dict(globals())
    tests = sorted(
        name
        for name, value in namespace.items()
        if name.startswith("test_") and callable(value)
    )
    failures = []
    for name in tests:
        try:
            namespace[name]()
        except Exception as error:  # noqa: BLE001
            failures.append((name, error))
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all_tests())
