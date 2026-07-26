"""CPU tests for opt-in VCHD diagnostic hooks."""

import torch

from decoding.diagnostics import (
    VCHDDiagnosticStep,
    emit_vchd_step,
    observe_vchd_steps,
)
from decoding.mmada_adapter import PairedLogits


def _step():
    return VCHDDiagnosticStep(
        model_evaluation=1,
        context_version=0,
        state=torch.tensor([[1, 4, 4]]),
        response_mask=torch.tensor([True, True]),
        selection_mask=torch.tensor([True, False]),
        valid_text_vocab=torch.tensor([True, True, False, False, False]),
        paired=PairedLogits(
            visual=torch.zeros(2, 5),
            ablated=torch.ones(2, 5),
        ),
    )


def test_observer_is_scoped_and_receives_borrowed_step():
    seen = []
    step = _step()

    emit_vchd_step(step)
    with observe_vchd_steps(seen.append):
        emit_vchd_step(step)
    emit_vchd_step(step)

    assert seen == [step]


def test_nested_observers_restore_outer_observer():
    outer, inner = [], []
    step = _step()

    with observe_vchd_steps(outer.append):
        emit_vchd_step(step)
        with observe_vchd_steps(inner.append):
            emit_vchd_step(step)
        emit_vchd_step(step)

    assert outer == [step, step]
    assert inner == [step]
