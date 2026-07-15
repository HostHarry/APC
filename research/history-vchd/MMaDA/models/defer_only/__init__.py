"""Defer-only CV (CV-DCD v4 direction B).

Argmax always follows base_logits (never modified by CD). The paired
forward's ``drop_logits`` is used only to compute a per-position visual
gain, which then modulates the DCD threshold's confidence signal via one
of three "veto" variants (hard / mult / min). Nothing is ever boosted.

Public entry point: ``pick_transfer_defer_only`` in ``dispatcher.py``.
"""
from . import dispatcher, veto

__all__ = ["dispatcher", "veto"]
