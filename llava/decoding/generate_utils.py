from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from .config import VCHDDecodeConfig, vchd_config_from_dict


def _coerce_flat_value(name: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    field_types = {f.name: f.type for f in fields(VCHDDecodeConfig)}
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
    """Pop decode_strategy / decode_config / flattened vchd__* knobs from kwargs."""

    decode_strategy = str(kwargs.pop("decode_strategy", "original"))
    decode_config = kwargs.pop("decode_config", None)
    allowed_strategies = {"original", "vchd", "vchd_fixed"}
    if decode_strategy not in allowed_strategies:
        raise ValueError(
            f"Unknown decode_strategy={decode_strategy!r}; "
            f"expected one of {sorted(allowed_strategies)}"
        )

    flat: Dict[str, Any] = {}
    for key in list(kwargs.keys()):
        if not key.startswith("vchd__"):
            continue
        name = key[len("vchd__") :]
        flat[name] = _coerce_flat_value(name, kwargs.pop(key))

    if flat:
        if decode_config is None:
            decode_config = flat
        elif isinstance(decode_config, Mapping):
            merged = dict(decode_config)
            merged.update(flat)
            decode_config = merged
        elif isinstance(decode_config, VCHDDecodeConfig):
            merged = {
                field.name: getattr(decode_config, field.name)
                for field in fields(decode_config)
            }
            merged.update(flat)
            decode_config = merged
        else:
            raise TypeError(
                "decode_config must be VCHDDecodeConfig, dict, or None when "
                "passing vchd__* overrides"
            )
    return decode_strategy, decode_config


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
