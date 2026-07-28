from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from lmms_eval.tasks.m3cot.utils import extract_m3cot_answer
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator
from lmms_eval.tasks.mmmu.utils import parse_multi_choice_response


def test_m3cot_prefers_final_boxed_answer() -> None:
    response = r"An intermediate possibility is (A), but finally \boxed{C}."
    assert extract_m3cot_answer(response, ["one", "two", "three", "four"]) == "C"


def test_mmmu_prefers_final_boxed_answer() -> None:
    response = r"Consider (A) first. The final answer is \boxed{\text{D}}."
    assert (
        parse_multi_choice_response(
            response,
            ["A", "B", "C", "D"],
            {"A": "one", "B": "two", "C": "three", "D": "four"},
        )
        == "D"
    )


def test_mmmu_unparseable_is_empty_not_random() -> None:
    # No letter / option content → must not invent a random choice.
    assert (
        parse_multi_choice_response(
            "I am not sure about this question.",
            ["A", "B", "C", "D"],
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        )
        == ""
    )


def test_mmbench_extracts_boxed_answer_without_api() -> None:
    evaluator = MMBench_Evaluator()
    response = r"After comparing A and C, the final answer is \boxed{B}."
    assert evaluator.can_infer_option(response, num_choice=4) == "B"
