"""lmms-eval adapter for the local M3CoT test split."""

from __future__ import annotations

import math
import re
import string
from pathlib import Path
from typing import Any

from PIL import Image


ANSWER_LABELS = list(string.ascii_uppercase)


def m3cot_process_docs(dataset):
    """Keep only the official M3CoT test split from the mixed TSV."""
    return dataset.filter(
        lambda doc: str(doc.get("split", "")).strip().lower() == "test"
    )


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in {"", "nan", "none"}


def _choices(doc: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (label, str(doc[label]).strip())
        for label in ANSWER_LABELS
        if label in doc and _is_present(doc[label])
    ]


def m3cot_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    image_path = Path(str(doc["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"M3CoT image not found: {image_path}")
    with Image.open(image_path) as image:
        return [image.convert("RGB")]


def m3cot_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None
) -> str:
    sections = []
    context = doc.get("context")
    if _is_present(context):
        sections.append(f"[Context]\n{str(context).strip()}")
    sections.append(f"[Question]\n{str(doc['question']).strip()}")

    choices = _choices(doc)
    if not choices:
        raise ValueError(f"M3CoT sample {doc.get('id')} has no choices")
    option_text = "\n".join(f"({label}) {choice}" for label, choice in choices)
    sections.append(f"[Choices]\n{option_text}")

    prompt_mode = (lmms_eval_specific_kwargs or {}).get("prompt_mode", "cot")
    if prompt_mode == "cot":
        suffix = (
            "\n\nLet's think step-by-step! End your response with the final "
            "option in parentheses, for example (A)."
        )
    elif prompt_mode == "direct":
        suffix = (
            "\n\nSelect the correct answer and respond only with its option "
            "in parentheses, for example (A)."
        )
    else:
        raise ValueError(f"Unknown M3CoT prompt mode: {prompt_mode}")
    return "\n".join(sections) + suffix


def extract_m3cot_answer(text: str, choices: list[str]) -> str:
    """Apply the official M3CoT ``judge_answer`` extraction rules."""
    text = str(text)
    if "[Answer]" in text:
        text = (
            text.split("[Answer]")[-1]
            .split("[Rationale]")[0]
            .split("[Context]")[0]
        )

    matches = re.findall(r"\(([A-Za-z])\)", text)
    if matches:
        return matches[-1].upper()

    matched_labels = []
    for index, choice in enumerate(choices):
        if str(choice).lower() in text.lower():
            matched_labels.append(ANSWER_LABELS[index])
    if matched_labels:
        return matched_labels[-1]

    normalized = re.sub(r"[\n.,!?]", " ", text)
    tokens = normalized.split(" ")
    for index in range(len(choices)):
        if ANSWER_LABELS[index] in tokens:
            matched_labels.append(ANSWER_LABELS[index])
    if matched_labels:
        return matched_labels[-1]

    for index in range(len(choices)):
        if ANSWER_LABELS[index].lower() in tokens:
            matched_labels.append(ANSWER_LABELS[index])
    return matched_labels[-1] if matched_labels else "FAILED"


def m3cot_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, float]:
    choices = [choice for _, choice in _choices(doc)]
    prediction = extract_m3cot_answer(results[0], choices)
    answer = str(doc["answer"]).strip().upper()
    return {"m3cot_accuracy": float(prediction == answer)}
