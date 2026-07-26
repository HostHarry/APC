"""Restored from bytecode snapshot (source was deleted)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_pyc = Path(__file__).resolve().parents[2] / "_bytecode" / "mmada.cpython-311.pyc"
_spec = importlib.util.spec_from_file_location(__name__, _pyc)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _mod
_spec.loader.exec_module(_mod)
