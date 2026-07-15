from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral
from typing import Any, List, Mapping, Optional, Tuple, Union


EOSTokenId = Optional[Union[int, Tuple[int, ...], List[int]]]


def normalize_eos_token_ids(eos_token_id: EOSTokenId) -> Tuple[int, ...]:
    """Normalize a scalar or sequence of EOS IDs to a unique tuple."""

    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, Integral) and not isinstance(eos_token_id, bool):
        values = (eos_token_id,)
    elif isinstance(eos_token_id, (tuple, list)):
        values = eos_token_id
    else:
        raise TypeError(
            "eos_token_id must be an int, tuple/list of ints, or None"
        )

    normalized = []
    seen = set()
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TypeError("Every eos_token_id value must be an int")
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError(
                f"eos_token_id values must be non-negative, got {token_id}"
            )
        if token_id not in seen:
            normalized.append(token_id)
            seen.add(token_id)
    return tuple(normalized)


@dataclass
class VCHDDecodeConfig:
    """Configuration for the fixed-window VCHD decoder.

    The decoder supports fixed or adaptive windows, sparse history, and an
    optional DCD-style approximate KV cache with isolated visual/ablated
    branches.
    """

    mask_id: int = 126336
    eos_token_id: EOSTokenId = None
    text_vocab_size: Optional[int] = None
    forbidden_token_ids: Tuple[int, ...] = ()

    alpha: float = 0.5
    beta: float = 0.1
    jsd_epsilon: float = 1.0e-8

    tau_base: float = 0.10
    tau_contrast: float = 0.90
    max_commit_per_iteration: int = 16

    mask_capacity: int = 16
    max_physical_span: int = 128
    fallback_to_raw: bool = False

    force_math_sdpa: bool = True
    cache_type: str = "none"
    cache_refresh_interval: int = 8
    cache_refresh_on_pressure: bool = True
    cache_pressure_threshold: float = 0.60
    truncate_at_eos: bool = True
    collect_trace: bool = False
    return_report: bool = False

    history_enabled: bool = False
    history_top_v_tokens: int = 8
    history_ema_decay: float = 0.7
    history_penalty_scale: float = 1.0
    history_anchor_min_consistent: int = 0
    ccaw_enabled: bool = False
    ccaw_mode: str = "legacy"
    ccaw_block_size: int = 32
    ccaw_min_commit_per_iteration: int = 1
    ccaw_qualified_budget: int = 1
    ccaw_max_mask_capacity: int = 64
    ccaw_pressure_ema_decay: float = 0.8
    ccaw_expand_step: int = 8
    ccaw_shrink_step: int = 4

    def validate(self) -> None:
        if self.mask_id < 0:
            raise ValueError(f"mask_id must be non-negative, got {self.mask_id}")
        normalize_eos_token_ids(self.eos_token_id)
        if self.text_vocab_size is not None and self.text_vocab_size <= 0:
            raise ValueError(
                f"text_vocab_size must be positive or None, got {self.text_vocab_size}"
            )
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")
        if not 0.0 <= self.tau_base <= 1.0:
            raise ValueError(f"tau_base must be in [0, 1], got {self.tau_base}")
        if not 0.0 <= self.tau_contrast <= 1.0:
            raise ValueError(
                f"tau_contrast must be in [0, 1], got {self.tau_contrast}"
            )
        if self.jsd_epsilon <= 0.0:
            raise ValueError(
                f"jsd_epsilon must be positive, got {self.jsd_epsilon}"
            )
        if self.max_commit_per_iteration < 1:
            raise ValueError("max_commit_per_iteration must be at least 1")
        if self.mask_capacity < 1:
            raise ValueError("mask_capacity must be at least 1")
        if self.max_physical_span < self.mask_capacity:
            raise ValueError(
                "max_physical_span must be at least mask_capacity "
                f"({self.max_physical_span} < {self.mask_capacity})"
            )
        if self.history_top_v_tokens < 2:
            raise ValueError(
                "history_top_v_tokens must be at least 2, got "
                f"{self.history_top_v_tokens}"
            )
        if not 0.0 <= self.history_ema_decay < 1.0:
            raise ValueError(
                "history_ema_decay must be in [0, 1), got "
                f"{self.history_ema_decay}"
            )
        if self.history_penalty_scale < 0.0:
            raise ValueError(
                "history_penalty_scale must be non-negative, got "
                f"{self.history_penalty_scale}"
            )
        if self.history_anchor_min_consistent < 0:
            raise ValueError(
                "history_anchor_min_consistent must be non-negative"
            )
        if self.history_anchor_min_consistent and not self.history_enabled:
            raise ValueError(
                "history_anchor_min_consistent requires history_enabled"
            )
        if self.ccaw_qualified_budget < 1:
            raise ValueError("ccaw_qualified_budget must be at least 1")
        if self.ccaw_mode not in {
            "legacy",
            "hard_block",
            "inverse_window",
        }:
            raise ValueError(
                "ccaw_mode must be 'legacy', 'hard_block', or "
                f"'inverse_window', got {self.ccaw_mode!r}"
            )
        if self.ccaw_block_size < 1:
            raise ValueError("ccaw_block_size must be at least 1")
        if (
            self.ccaw_mode == "hard_block"
            and self.ccaw_block_size > self.max_physical_span
        ):
            raise ValueError(
                "ccaw_block_size cannot exceed max_physical_span "
                f"({self.ccaw_block_size} > {self.max_physical_span})"
            )
        if not (
            1
            <= self.ccaw_min_commit_per_iteration
            <= self.max_commit_per_iteration
        ):
            raise ValueError(
                "ccaw_min_commit_per_iteration must be between 1 and "
                "max_commit_per_iteration"
            )
        if (
            self.ccaw_mode == "hard_block"
            and self.max_commit_per_iteration > self.ccaw_block_size
        ):
            raise ValueError(
                "max_commit_per_iteration cannot exceed ccaw_block_size "
                "in hard_block mode"
            )
        if self.ccaw_max_mask_capacity < self.mask_capacity:
            raise ValueError(
                "ccaw_max_mask_capacity must be at least mask_capacity "
                f"({self.ccaw_max_mask_capacity} < {self.mask_capacity})"
            )
        if self.ccaw_qualified_budget > self.ccaw_max_mask_capacity:
            raise ValueError(
                "ccaw_qualified_budget cannot exceed ccaw_max_mask_capacity "
                f"({self.ccaw_qualified_budget} > "
                f"{self.ccaw_max_mask_capacity})"
            )
        if not 0.0 <= self.ccaw_pressure_ema_decay < 1.0:
            raise ValueError(
                "ccaw_pressure_ema_decay must be in [0, 1), got "
                f"{self.ccaw_pressure_ema_decay}"
            )
        if self.ccaw_expand_step < 1:
            raise ValueError("ccaw_expand_step must be at least 1")
        if self.ccaw_shrink_step < 1:
            raise ValueError("ccaw_shrink_step must be at least 1")
        if self.max_commit_per_iteration > self.ccaw_max_mask_capacity:
            raise ValueError(
                "max_commit_per_iteration cannot exceed "
                "ccaw_max_mask_capacity"
            )
        if self.cache_type not in {"none", "dual"}:
            raise ValueError(
                "cache_type must be 'none' or 'dual', got "
                f"{self.cache_type!r}"
            )
        if self.cache_refresh_interval < 1:
            raise ValueError("cache_refresh_interval must be at least 1")
        if not 0.0 <= self.cache_pressure_threshold <= 1.0:
            raise ValueError(
                "cache_pressure_threshold must be in [0, 1], got "
                f"{self.cache_pressure_threshold}"
            )


def vchd_config_from_dict(values: Mapping[str, Any]) -> VCHDDecodeConfig:
    """Build a strict config so misspelled experimental knobs fail fast."""

    valid_names = {field.name for field in fields(VCHDDecodeConfig)}
    unknown = sorted(set(values) - valid_names)
    if unknown:
        raise ValueError(f"Unknown VCHD config fields: {', '.join(unknown)}")

    kwargs = dict(values)
    if "forbidden_token_ids" in kwargs:
        kwargs["forbidden_token_ids"] = tuple(
            int(token_id) for token_id in kwargs["forbidden_token_ids"]
        )
    if "eos_token_id" in kwargs:
        original_eos = kwargs["eos_token_id"]
        normalized_eos = normalize_eos_token_ids(original_eos)
        if original_eos is None:
            kwargs["eos_token_id"] = None
        elif isinstance(original_eos, Integral) and not isinstance(
            original_eos, bool
        ):
            kwargs["eos_token_id"] = normalized_eos[0]
        else:
            kwargs["eos_token_id"] = normalized_eos
    config = VCHDDecodeConfig(**kwargs)
    config.validate()
    return config
