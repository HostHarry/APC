from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from .config import VCHDDecodeConfig, vchd_config_from_dict
from .thinking import (
    ThinkingDecodeConfig,
    resolve_thinking_config,
)


def _coerce_flat_value(name: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    field_types = {
        f.name: f.type
        for config_type in (VCHDDecodeConfig, ThinkingDecodeConfig)
        for f in fields(config_type)
    }
    annotation = str(field_types.get(name, ""))
    if "Tuple[int" in annotation or name in {"forbidden_token_ids", "eos_token_id"}:
        if text.startswith("(") or text.startswith("["):
            inner = text.strip("()[]")
            if not inner:
                return ()
            return tuple(int(part.strip()) for part in inner.split(",") if part.strip())
        if "," in text:
            return tuple(int(part.strip()) for part in text.split(",") if part.strip())
        if name == "eos_token_id":
            return int(text)
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return value


def extract_decode_options(
    kwargs: MutableMapping[str, Any],
) -> Tuple[str, Optional[Any]]:
    """Pop decoding strategy and flattened VCHD / Thinking knobs."""

    decode_strategy = str(kwargs.pop("decode_strategy", "original"))
    decode_config = kwargs.pop("decode_config", None)
    vchd_strategies = {"vchd", "vchd_fixed"}
    thinking_strategies = {"thinking", "swd", "psp", "vrg", "psp_vrg"}
    allowed_strategies = {"original"} | vchd_strategies | thinking_strategies
    if decode_strategy not in allowed_strategies:
        raise ValueError(
            f"Unknown decode_strategy={decode_strategy!r}; "
            f"expected one of {sorted(allowed_strategies)}"
        )

    vchd_flat: Dict[str, Any] = {}
    thinking_flat: Dict[str, Any] = {}
    for key in list(kwargs.keys()):
        if key.startswith("vchd__"):
            name = key[len("vchd__") :]
            vchd_flat[name] = _coerce_flat_value(name, kwargs.pop(key))
        elif key.startswith("thinking__"):
            name = key[len("thinking__") :]
            thinking_flat[name] = _coerce_flat_value(name, kwargs.pop(key))

    if vchd_flat and decode_strategy not in vchd_strategies:
        raise ValueError(
            "vchd__* options require decode_strategy='vchd' or 'vchd_fixed'"
        )
    if thinking_flat and decode_strategy in vchd_strategies:
        raise ValueError("thinking__* options cannot be combined with VCHD")

    flat = vchd_flat if decode_strategy in vchd_strategies else thinking_flat
    if flat:
        if decode_config is None:
            decode_config = dict(flat)
        elif isinstance(decode_config, Mapping):
            decode_config = {**decode_config, **flat}
        elif isinstance(decode_config, (VCHDDecodeConfig, ThinkingDecodeConfig)):
            decode_config = {
                **{
                    field.name: getattr(decode_config, field.name)
                    for field in fields(decode_config)
                },
                **flat,
            }
        else:
            raise TypeError(
                "decode_config must be a decoding dataclass, mapping, or None "
                "when flattened overrides are provided"
            )

    presets = {
        "swd": {"swd_enabled": True},
        "psp": {"psp_enabled": True},
        "vrg": {"vrg_enabled": True},
        "psp_vrg": {"psp_enabled": True, "vrg_enabled": True},
    }
    if decode_strategy in presets:
        if decode_config is None:
            decode_config = {}
        elif isinstance(decode_config, ThinkingDecodeConfig):
            decode_config = decode_config.to_dict()
        elif not isinstance(decode_config, Mapping):
            raise TypeError(
                "Thinking decode_config must be ThinkingDecodeConfig, "
                "a mapping, or None"
            )
        decode_config = {**presets[decode_strategy], **dict(decode_config)}
    return decode_strategy, decode_config


def coerce_thinking_config(
    decode_strategy: str,
    decode_config: Optional[Any],
) -> ThinkingDecodeConfig:
    """Resolve presets and explicit settings to a validated config."""

    thinking_strategies = {"thinking", "swd", "psp", "vrg", "psp_vrg"}
    if decode_strategy not in thinking_strategies | {"original"}:
        raise ValueError(
            f"decode_strategy={decode_strategy!r} is not a Thinking strategy"
        )
    if decode_config is None:
        return ThinkingDecodeConfig()
    return resolve_thinking_config(decode_config)


def coerce_vchd_config(
    decode_config: Optional[Any],
    *,
    mask_id: int,
    eos_token_id=None,
    text_vocab_size: Optional[int] = None,
    forbidden_token_ids=(),
) -> VCHDDecodeConfig:
    if decode_config is None:
        config = VCHDDecodeConfig(mask_id=mask_id)
    elif isinstance(decode_config, VCHDDecodeConfig):
        config = decode_config
    elif isinstance(decode_config, Mapping):
        config = vchd_config_from_dict(decode_config)
    else:
        raise TypeError(
            "decode_config must be VCHDDecodeConfig, dict, or None"
        )

    config.mask_id = int(mask_id)
    if config.eos_token_id is None and eos_token_id is not None:
        config.eos_token_id = eos_token_id
    if config.text_vocab_size is None and text_vocab_size is not None:
        config.text_vocab_size = int(text_vocab_size)
    if not config.forbidden_token_ids and forbidden_token_ids:
        config.forbidden_token_ids = tuple(int(x) for x in forbidden_token_ids)
    config.validate()
    if config.cache_type != "none":
        raise ValueError(
            "LaViDa VCHD supports cache_type='none' only; "
            f"got {config.cache_type!r}"
        )
    return config
