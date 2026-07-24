"""Unit tests for Thinking Diffusion (PSP/VRG) and optimized SWD."""

from __future__ import annotations

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from decoding.swd import (  # noqa: E402
    SwdState,
    apply_swd_to_confidence,
    kl_prev_vs_curr,
    response_span_token_probs,
    stability_weight,
    uniform_prior_like,
)
from decoding.thinking_diffusion import (  # noqa: E402
    ThinkingSwdDecodeConfig,
    apply_psp,
    apply_vrg,
    resolve_thinking_swd_config,
)


def test_vrg_zero_scale_recovers_conditional():
    logits_c = torch.tensor([[1.0, 2.0, 0.5]])
    logits_u = torch.tensor([[0.0, 1.0, 3.0]])
    assert torch.allclose(apply_vrg(logits_c, logits_u, s_vrg=0.0), logits_c)


def test_vrg_positive_scale_amplifies_delta():
    logits_c = torch.tensor([[2.0, 0.0]])
    logits_u = torch.tensor([[0.0, 0.0]])
    out = apply_vrg(logits_c, logits_u, s_vrg=1.0)
    assert torch.allclose(out, torch.tensor([[4.0, 0.0]]))


def test_psp_early_step_penalizes_late_positions_more():
    conf = torch.ones(1, 8)
    early = apply_psp(
        conf, step_index=0, num_steps=10,
        response_start=0, response_end=8, gamma=0.5,
    )
    late = apply_psp(
        conf, step_index=9, num_steps=10,
        response_start=0, response_end=8, gamma=0.5,
    )
    assert early[0, -1].item() < early[0, 0].item()
    assert abs(late[0, -1].item() - 1.0) < 0.05
    assert early[0, -1].item() < late[0, -1].item()


def test_psp_preserves_non_finite():
    conf = torch.tensor([[0.5, float("-inf"), 0.8]])
    out = apply_psp(
        conf, step_index=0, num_steps=4,
        response_start=0, response_end=3, gamma=0.5,
    )
    assert math.isinf(out[0, 1].item()) and out[0, 1].item() < 0


def test_swd_identical_distributions_weight_one():
    p = torch.softmax(torch.randn(2, 4, 8), dim=-1)
    w = stability_weight(p, p, lambda_=5.0)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-5)


def test_swd_high_kl_near_zero_weight():
    p_curr = torch.zeros(1, 2, 4)
    p_curr[..., 0] = 1.0
    p_prev = torch.zeros(1, 2, 4)
    p_prev[..., -1] = 1.0
    assert float(stability_weight(p_curr, p_prev, lambda_=5.0).max()) < 1e-3


def test_swd_response_span_only_and_updates_state():
    # Full sequence length 6; response is positions [2, 5).
    conf = torch.tensor([[0.1, 0.2, 0.9, 0.8, 0.7, 0.05]], dtype=torch.float64)
    logits = torch.randn(1, 6, 5)
    mask = torch.tensor([[False, False, True, True, True, False]])
    state = SwdState()
    out = apply_swd_to_confidence(
        conf, logits, state, lambda_=1.0,
        response_start=2, response_end=5, mask_index=mask,
    )
    assert state.prev_probs is not None
    assert state.prev_probs.shape == (1, 3, 5)  # response only
    # Prompt/image positions untouched.
    assert out[0, 0].item() == conf[0, 0].item()
    assert out[0, 1].item() == conf[0, 1].item()
    # Second call with same logits -> near identity on response.
    out2 = apply_swd_to_confidence(
        conf, logits, state, lambda_=1.0,
        response_start=2, response_end=5, mask_index=mask,
    )
    assert torch.allclose(out2[:, 2:5], conf[:, 2:5], atol=1e-5)


def test_response_span_token_probs_matches_slice_softmax():
    logits = torch.randn(2, 10, 7)
    x0 = torch.randint(0, 7, (2, 10))
    x0_p, p_resp = response_span_token_probs(
        logits, x0, response_start=3, response_end=8
    )
    assert p_resp.shape == (2, 5, 7)
    assert torch.isneginf(x0_p[:, :3]).all()
    assert torch.isneginf(x0_p[:, 8:]).all()
    expect = torch.gather(
        torch.softmax(logits[:, 3:8, :].float(), dim=-1),
        dim=-1,
        index=x0[:, 3:8].unsqueeze(-1),
    ).squeeze(-1)
    assert torch.allclose(x0_p[:, 3:8], expect)


def test_swd_accepts_precomputed_p_curr():
    conf = torch.ones(1, 4)
    logits = torch.randn(1, 4, 6)
    p_curr = torch.softmax(logits[:, 1:3, :].float(), dim=-1)
    state = SwdState()
    out = apply_swd_to_confidence(
        conf, logits, state, lambda_=1.0,
        response_start=1, response_end=3, p_curr=p_curr,
    )
    assert state.prev_probs is not None
    assert torch.allclose(state.prev_probs, p_curr)
    assert out.shape == conf.shape


def test_kl_direction_matches_algorithm_one():
    p_prev = torch.tensor([[[0.7, 0.3]]])
    p_curr = torch.tensor([[[0.2, 0.8]]])
    expected = 0.7 * math.log(0.7 / 0.2) + 0.3 * math.log(0.3 / 0.8)
    assert abs(float(kl_prev_vs_curr(p_prev, p_curr)) - expected) < 1e-5


def test_uniform_prior_sums_to_one():
    u = uniform_prior_like(torch.randn(1, 3, 10))
    assert torch.allclose(u.sum(dim=-1), torch.ones(1, 3))


def test_resolve_config_from_dict():
    cfg = resolve_thinking_swd_config(
        {"psp_enabled": True, "swd_enabled": True, "swd_lambda": 2.5}
    )
    assert cfg.psp_enabled and cfg.swd_enabled and cfg.swd_lambda == 2.5
    assert isinstance(cfg, ThinkingSwdDecodeConfig)
