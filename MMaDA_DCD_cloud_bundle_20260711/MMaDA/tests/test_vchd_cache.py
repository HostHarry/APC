"""CPU-only tests for VCHD's disjoint visual/ablated KV caches."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decoding import MMaDAVisualAccessAdapter, VCHDDecodeConfig  # noqa: E402
from decoding import visual_contrast_decode  # noqa: E402


class _CacheOutput:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


class _CacheAwareBranchModel:
    """Small model that implements MMaDA's in-place replace_position API."""

    def __init__(self):
        self.config = type(
            "Config", (), {"vocab_size": 6, "llm_vocab_size": 5}
        )()
        self.calls = []

    @staticmethod
    def _branch_marker(attention_bias, batch, device):
        if attention_bias is None:
            return torch.zeros(batch, dtype=torch.float32, device=device)
        blocked = attention_bias.reshape(batch, -1) < -1.0e20
        return blocked.any(dim=-1).float()

    def __call__(
        self,
        *,
        input_ids,
        attention_mask=None,
        attention_bias=None,
        past_key_values=None,
        use_cache=False,
        replace_position=None,
    ):
        del attention_mask
        batch, query_len = input_ids.shape
        marker = self._branch_marker(
            attention_bias, batch, input_ids.device
        )
        past_pointer = (
            None
            if past_key_values is None
            else past_key_values[0][0].untyped_storage().data_ptr()
        )
        self.calls.append(
            {
                "batch": batch,
                "marker": marker.detach().cpu().tolist(),
                "past_pointer": past_pointer,
                "query_len": query_len,
            }
        )

        encoded = input_ids.float()[:, None, :, None]
        marker_4d = marker[:, None, None, None]
        if past_key_values is None:
            key = encoded + 100.0 * marker_4d
            value = encoded + 200.0 * marker_4d
        else:
            key, value = past_key_values[0]
            if replace_position is None:
                raise AssertionError("Cached test forwards require replacement")
            for batch_index in range(batch):
                positions = torch.nonzero(
                    replace_position[batch_index], as_tuple=True
                )[0]
                if positions.numel() != query_len:
                    raise AssertionError("Replacement/query lengths diverged")
                key[batch_index, 0, positions, 0] = (
                    input_ids[batch_index].float()
                    + 100.0 * marker[batch_index]
                )
                value[batch_index, 0, positions, 0] = (
                    input_ids[batch_index].float()
                    + 200.0 * marker[batch_index]
                )

        logits = torch.zeros(batch, query_len, 6)
        logits[:, :, 1] = 8.0 - 3.0 * marker[:, None]
        logits[:, :, 2] = 2.0
        logits[:, :, 5] = 20.0
        cache = [(key, value)] if use_cache else None
        return _CacheOutput(logits, cache)


def _decode_config(cache_type, **overrides):
    values = {
        "mask_id": 4,
        "text_vocab_size": 5,
        "forbidden_token_ids": (0, 3, 4),
        "tau_base": 0.5,
        "tau_contrast": 0.5,
        "mask_capacity": 3,
        "max_physical_span": 3,
        "max_commit_per_iteration": 1,
        "force_math_sdpa": False,
        "cache_type": cache_type,
        "cache_refresh_interval": 100,
        "cache_refresh_on_pressure": False,
        "truncate_at_eos": False,
        "return_report": True,
    }
    values.update(overrides)
    return VCHDDecodeConfig(**values)


def _storage_pointer(tensor):
    return tensor.untyped_storage().data_ptr()


def test_dual_cache_branches_have_disjoint_storage_and_forwards():
    model = _CacheAwareBranchModel()
    adapter = MMaDAVisualAccessAdapter(
        model,
        decode_start=3,
        decode_end=5,
        image_span=(1, 2),
        force_math_sdpa=False,
        mask_id=4,
        cache_type="dual",
        cache_refresh_interval=8,
    )
    tokens = torch.tensor([[0, 2, 3, 4, 4]])
    seed = adapter.paired_forward(tokens, context_version=0)
    assert seed.cache_event == "full_refresh"

    state = adapter.cache_state
    assert state is not None
    for visual_layer, ablated_layer in zip(
        state.visual.past_key_values,
        state.ablated.past_key_values,
    ):
        for visual_tensor, ablated_tensor in zip(
            visual_layer, ablated_layer
        ):
            assert _storage_pointer(visual_tensor) != _storage_pointer(
                ablated_tensor
            )

    tokens[0, 3] = 1
    cached = adapter.paired_forward(tokens, context_version=1)
    assert cached.cache_event == "partial_refresh"
    assert cached.model_forward_calls == 2
    assert [call["marker"] for call in model.calls[-2:]] == [[0.0], [1.0]]
    assert (
        model.calls[-2]["past_pointer"]
        != model.calls[-1]["past_pointer"]
    )
    state = adapter.cache_state
    assert state.visual.context_version == 1
    assert state.ablated.context_version == 1


def test_context_interval_rebuilds_both_branch_caches():
    model = _CacheAwareBranchModel()
    adapter = MMaDAVisualAccessAdapter(
        model,
        decode_start=3,
        decode_end=5,
        image_span=(1, 2),
        force_math_sdpa=False,
        mask_id=4,
        cache_type="dual",
        cache_refresh_interval=1,
    )
    tokens = torch.tensor([[0, 2, 3, 4, 4]])
    adapter.paired_forward(tokens, context_version=0)
    tokens[0, 3] = 1
    refreshed = adapter.paired_forward(tokens, context_version=1)

    assert refreshed.cache_event == "full_refresh"
    assert refreshed.cache_refresh_reason == "context_interval"
    assert model.calls[-1]["batch"] == 2
    report = adapter.cache_report()
    assert report["cache_full_refreshes"] == 2
    assert report["cache_partial_refreshes"] == 0


def test_cache_on_and_off_produce_identical_tokens():
    tokens = torch.tensor([[0, 2, 3, 4, 4, 4]])
    exact, exact_report = visual_contrast_decode(
        _CacheAwareBranchModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=_decode_config("none"),
    )
    cached, cached_report = visual_contrast_decode(
        _CacheAwareBranchModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=_decode_config("dual"),
    )

    assert torch.equal(cached, exact)
    assert cached[0, 3:].tolist() == [1, 1, 1]
    assert exact_report["cache_type"] == "none"
    assert cached_report["cache_full_refreshes"] == 1
    assert cached_report["cache_partial_refreshes"] == 2


def test_pressure_requests_exact_retry_before_commit():
    tokens = torch.tensor([[0, 2, 3, 4, 4, 4]])
    output, report = visual_contrast_decode(
        _CacheAwareBranchModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=_decode_config(
            "dual",
            cache_refresh_on_pressure=True,
            cache_pressure_threshold=0.0,
        ),
    )

    assert output[0, 3:].tolist() == [1, 1, 1]
    assert report["cache_pressure_refresh_requests"] == 2
    assert report["cache_full_refreshes"] == 3
    assert report["cache_partial_refreshes"] == 2
    assert report["cache_refresh_reasons"]["conflict_pressure"] == 2
    assert report["model_evaluations"] == 5


def test_dual_cache_composes_with_history_and_ccaw():
    tokens = torch.tensor([[0, 2, 3, 4, 4, 4]])
    output, report = visual_contrast_decode(
        _CacheAwareBranchModel(),
        tokens,
        decode_start=3,
        decode_end=6,
        image_span=(1, 2),
        config=_decode_config(
            "dual",
            history_enabled=True,
            history_top_v_tokens=2,
            ccaw_enabled=True,
            ccaw_max_mask_capacity=3,
            ccaw_expand_step=1,
        ),
    )

    assert output[0, 3:].tolist() == [1, 1, 1]
    assert report["history_enabled"]
    assert report["ccaw_enabled"]
    assert report["history_observations"] > 0
    assert report["cache_full_refreshes"] == 1
    assert report["cache_partial_refreshes"] == 2
