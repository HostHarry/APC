"""Lightweight checks for the local MMBench two-cycle manifest generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "VLind-Bench" / "scripts" / "make_mmbench_two_cycle.py"
SPEC = importlib.util.spec_from_file_location("make_mmbench_two_cycle", GENERATOR)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_two_cycle_rotates_normal_and_skips_abnormal_rows(tmp_path):
    source = tmp_path / "MMBench_DEV_EN.tsv"
    output = tmp_path / "MMBench_DEV_EN_2C.tsv"
    data = pd.DataFrame(
        [
            {
                "index": 1,
                "image": "x" * 80,
                "question": "normal",
                "answer": "A",
                "A": "Cat",
                "B": "Dog",
                "C": "Bird",
            },
            {
                "index": 2,
                "image": "y" * 80,
                "question": "label-valued option",
                "answer": "B",
                "A": "A",
                "B": "Blue",
                "C": "Green",
            },
        ]
    )
    data.to_csv(source, sep="\t", index=False)

    MODULE.build_two_cycle(source, output)

    result = pd.read_csv(output, sep="\t")
    assert result["index"].tolist() == [1, 2, MODULE.OFFSET + 1]
    assert result["g_index"].tolist() == [1, 2, 1]
    rotated = result.iloc[-1]
    assert str(rotated["image"]) == "1"
    assert rotated["answer"] == "C"
    assert (rotated["A"], rotated["B"], rotated["C"]) == (
        "Dog",
        "Bird",
        "Cat",
    )


def test_build_two_cycle_rejects_preoffset_indices(tmp_path):
    source = tmp_path / "source.tsv"
    output = tmp_path / "output.tsv"
    pd.DataFrame(
        [
            {
                "index": MODULE.OFFSET,
                "image": "x" * 80,
                "question": "bad index",
                "answer": "A",
                "A": "Yes",
                "B": "No",
            }
        ]
    ).to_csv(source, sep="\t", index=False)

    with pytest.raises(ValueError, match="Source indices must be below"):
        MODULE.build_two_cycle(source, output)
