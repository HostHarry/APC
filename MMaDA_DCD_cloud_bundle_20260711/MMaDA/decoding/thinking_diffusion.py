"""Thinking Diffusion inference modules: PSP and VRG (training-free).

Position & Step Penalty (PSP) and Visual Reasoning Guidance (VRG) from
Kim et al., arXiv:2604.05497. For the MMaDA ``original`` low-confidence path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Union

import torch


@dataclass
class ThinkingSwdDecodeConfig:
    """Optional score modulators for original remasking."""

    psp_enabled: bool = False
    psp_gamma: float = 0.5
    vrg_enabled: bool = False
    vrg_scale: float = 0.5
    swd_enabled: bool = False
    swd_lambda: float = 5.0

    def validate(self) -> None:
        if not (0.0 <= float(self.psp_gamma) <= 1.0):
            raise ValueError(
                f"psp_gamma must be in [0, 1], got {self.psp_gamma}"
            )
        if float(self.vrg_scale) < 0.0:
            raise ValueError(
                f"vrg_scale must be non-negative, got {self.vrg_scale}"
            )
        if float(self.swd_lambda) < 0.0:
            raise ValueError(
                f"swd_lambda must be non-negative, got {self.swd_lambda}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def thinking_swd_config_from_dict(
    values: Mapping[str, Any],
) -> ThinkingSwdDecodeConfig:
    known = {
        f.name
        for f in ThinkingSwdDecodeConfig.__dataclass_fields__.values()
    }
    filtered = {k: v for k, v in values.items() if k in known}
    config = ThinkingSwdDecodeConfig(**filtered)
    config.validate()
    return config


def resolve_thinking_swd_config(
    decode_config: Optional[Union[ThinkingSwdDecodeConfig, Mapping[str, Any]]],
) -> ThinkingSwdDecodeConfig:
    if decode_config is None:
        return ThinkingSwdDecodeConfig()
    if isinstance(decode_config, ThinkingSwdDecodeConfig):
        decode_config.validate()
        return decode_config
    if isinstance(decode_config, Mapping):
        return thinking_swd_config_from_dict(decode_config)
    raise TypeError(
        "decode_config for original Thinking/SWD path must be "
        "ThinkingSwdDecodeConfig, dict, or None"
    )


def apply_vrg(
    logits_c: torch.Tensor,
    logits_u: torch.Tensor,
    s_vrg: float = 0.5,
) -> torch.Tensor:
    """Visual Reasoning Guidance (paper Eq. 6).

    ``logits_vrg = logits_u + (s_vrg + 1) * (logits_c - logits_u)``.
    At ``s_vrg=0`` this recovers ``logits_c``.
    """

    if logits_c.shape != logits_u.shape:
        raise ValueError(
            "VRG requires matching logits shapes, got "
            f"{tuple(logits_c.shape)} vs {tuple(logits_u.shape)}"
        )
    scale = float(s_vrg) + 1.0
    return logits_u + scale * (logits_c - logits_u)


def apply_psp(
    confidence: torch.Tensor,
    *,
    step_index: int,
    num_steps: int,
    response_start: int,
    response_end: int,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Position & Step Penalty (paper Eq. 4).

    ``tilde{C}_j = C_j * [1 - gamma * (1 - tau) * rel(j)]`` with
    ``tau = (step_index + 1) / num_steps`` and ``rel(j)`` normalized
    inside ``[response_start, response_end)``.
    """

    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if not 0 <= step_index < num_steps:
        raise ValueError(
            f"step_index must be in [0, {num_steps}), got {step_index}"
        )
    if response_end <= response_start:
        raise ValueError(
            "response_end must be greater than response_start, got "
            f"[{response_start}, {response_end})"
        )

    tau = float(step_index + 1) / float(num_steps)
    span = max(1, int(response_end) - int(response_start) - 1)
    seq_len = confidence.shape[-1]
    positions = torch.arange(
        seq_len, device=confidence.device, dtype=torch.float32
    )
    rel = ((positions - float(response_start)) / float(span)).clamp(0.0, 1.0)
    in_response = (positions >= float(response_start)) & (
        positions < float(response_end)
    )
    rel = torch.where(in_response, rel, torch.zeros_like(rel))
    factor = 1.0 - float(gamma) * (1.0 - tau) * rel
    factor = factor.to(dtype=confidence.dtype)
    while factor.ndim < confidence.ndim:
        factor = factor.unsqueeze(0)
    finite = torch.isfinite(confidence)
    return torch.where(finite, confidence * factor, confidence)
