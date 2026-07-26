"""CPU tests for the faithful shuyingte/DCD multimodal port."""

from __future__ import annotations

import torch

from models.mmada_decode import MMaDADecodeConfig, dcd_decode_text_official


class _Output:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


class _CacheAwareModel:
    def __init__(self, *, peaked=True):
        self.peaked = peaked
        self.calls = []

    def __call__(
        self,
        input_ids,
        attention_bias=None,
        past_key_values=None,
        use_cache=False,
        replace_position=None,
    ):
        del attention_bias
        batch, query_length = input_ids.shape
        self.calls.append(
            {
                "query_length": query_length,
                "cached": past_key_values is not None,
                "replace_count": (
                    0
                    if replace_position is None
                    else int(replace_position.sum().item())
                ),
            }
        )
        logits = torch.zeros(batch, query_length, 8)
        if self.peaked:
            logits[..., 1] = 8.0
            logits[..., 6] = 12.0
        if use_cache:
            if past_key_values is None:
                cache_length = query_length
                key = torch.zeros(batch, 1, cache_length, 1)
                cache = ((key, key.clone()),)
            else:
                assert replace_position is not None
                assert int(replace_position.sum().item()) == query_length
                cache = past_key_values
        else:
            cache = None
        return _Output(logits, cache)


def _config(cache_type, **overrides):
    values = {
        "window_type": "sliding",
        "initial_window_length": 2,
        "max_window_length": 4,
        "decode_algo": "threshold",
        "decode_param": 0.9,
        "temperature": 0.0,
        "cache_type": cache_type,
        "refresh_count": 100,
        "mask_id": 4,
        "text_vocab_size": 5,
        "debug": True,
    }
    values.update(overrides)
    return MMaDADecodeConfig(**values)


def test_official_dual_delay_matches_no_cache_and_uses_partial_refreshes():
    tokens = torch.tensor([[3, 3, 4, 4, 4, 4, 4, 4]])
    exact_model = _CacheAwareModel()
    cached_model = _CacheAwareModel()

    exact, exact_nfe = dcd_decode_text_official(
        exact_model, tokens, 2, 8, _config("none")
    )
    cached, cached_nfe = dcd_decode_text_official(
        cached_model, tokens, 2, 8, _config("dual-delay2")
    )

    assert torch.equal(cached, exact)
    assert cached[0, 2:].tolist() == [1] * 6
    assert cached_nfe == exact_nfe
    assert cached_model.calls[0] == {
        "query_length": 8,
        "cached": False,
        "replace_count": 0,
    }
    assert any(call["cached"] for call in cached_model.calls[1:])
    assert all(
        call["replace_count"] == call["query_length"]
        for call in cached_model.calls
        if call["cached"]
    )


def test_official_dcd_masks_multimodal_non_text_tokens():
    tokens = torch.tensor([[3, 4, 4]])
    output, _ = dcd_decode_text_official(
        _CacheAwareModel(),
        tokens,
        1,
        3,
        _config("none"),
    )
    assert output[0, 1:].tolist() == [1, 1]


def test_official_threshold_fallback_commits_one_token():
    tokens = torch.tensor([[3, 4, 4, 4, 4]])
    output, nfe = dcd_decode_text_official(
        _CacheAwareModel(peaked=False),
        tokens,
        1,
        5,
        _config("none"),
    )
    assert not (output[:, 1:] == 4).any()
    assert nfe == 4
