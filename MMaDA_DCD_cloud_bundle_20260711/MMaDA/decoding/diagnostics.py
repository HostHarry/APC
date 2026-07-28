"""Opt-in, zero-storage hooks for VCHD distribution diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch

from .mmada_adapter import PairedLogits


@dataclass(frozen=True)
class VCHDDiagnosticStep:
    """Borrowed tensors from one VCHD model evaluation.

    Callbacks run synchronously. Consumers that retain tensors after the
    callback returns must detach, move, and/or clone them themselves.
    """

    model_evaluation: int
    context_version: int
    state: torch.LongTensor
    response_mask: torch.BoolTensor
    selection_mask: torch.BoolTensor
    valid_text_vocab: torch.BoolTensor
    paired: PairedLogits


VCHDDiagnosticObserver = Callable[[VCHDDiagnosticStep], None]

_OBSERVER: ContextVar[Optional[VCHDDiagnosticObserver]] = ContextVar(
    "vchd_diagnostic_observer",
    default=None,
)


@contextmanager
def observe_vchd_steps(
    observer: VCHDDiagnosticObserver,
) -> Iterator[None]:
    """Send VCHD paired logits to ``observer`` within this context."""

    if not callable(observer):
        raise TypeError("observer must be callable")
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def emit_vchd_step(step: VCHDDiagnosticStep) -> None:
    """Emit a step when diagnostics are enabled; otherwise do nothing."""

    observer = _OBSERVER.get()
    if observer is not None:
        observer(step)
