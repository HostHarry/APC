#!/usr/bin/env python3
"""Correct saved MMMU samples/results with final boxed-answer extraction.

The original artifacts are preserved next to each modified file with the
``.pre_boxed_fix`` marker. Only multiple-choice rows containing an explicit
final ``\\boxed{X}`` are changed; all other parsed predictions are preserved.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import string
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from lmms_eval.tasks.mmmu.utils import mmmu_aggregate_results


BOXED_CHOICE_RE = re.compile(
    r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?\(?\s*([A-Za-z])\s*\)?"
    r"(?:\s*\})?\s*\}"
)


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.pre_boxed_fix{path.suffix}")


def ensure_backup(path: Path) -> Path:
    backup = backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def valid_labels(doc: dict[str, Any]) -> set[str]:
    options = doc.get("options", [])
    if isinstance(options, str):
        options = ast.literal_eval(options)
    return set(string.ascii_uppercase[: len(options)])


def final_boxed_choice(response: str, labels: set[str]) -> str | None:
    matches = [
        match.upper()
        for match in BOXED_CHOICE_RE.findall(str(response))
        if match.upper() in labels
    ]
    return matches[-1] if matches else None


def correct_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    boxed_rows = 0
    changed_rows = 0
    old_correct = 0
    corrected_correct = 0

    for row in rows:
        metric = row["mmmu_acc"]
        gold = str(metric["answer"])
        old_prediction = metric["parsed_pred"][0]
        if old_prediction == gold:
            old_correct += 1

        if metric["question_type"] == "multiple-choice":
            response = str(row["filtered_resps"][0])
            boxed = final_boxed_choice(response, valid_labels(row["doc"]))
            if boxed is not None:
                boxed_rows += 1
                if old_prediction != boxed:
                    changed_rows += 1
                row["mmmu_acc"]["parsed_pred"] = [boxed]
                if "mmmu_acc_pass_at_k" in row:
                    row["mmmu_acc_pass_at_k"]["parsed_pred"] = [boxed]

        if row["mmmu_acc"]["parsed_pred"][0] == gold:
            corrected_correct += 1

    ensure_backup(path)
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    return rows, {
        "samples": len(rows),
        "boxed_rows": boxed_rows,
        "changed_rows": changed_rows,
        "old_exact_letter_matches": old_correct,
        "corrected_exact_letter_matches": corrected_correct,
        "backup": str(backup_path(path)),
    }


def correct_mode(mode_dir: Path) -> dict[str, Any]:
    result_paths = list(mode_dir.rglob("*_results.json"))
    if len(result_paths) != 1:
        raise RuntimeError(
            f"Expected one results JSON under {mode_dir}, found {result_paths}"
        )
    result_path = result_paths[0]
    result_data = json.loads(result_path.read_text())
    result_backup = ensure_backup(result_path)

    mode_summary: dict[str, Any] = {
        "result_file": str(result_path),
        "result_backup": str(result_backup),
        "splits": {},
    }
    for split, task in (("dev", "mmmu_dev"), ("val", "mmmu_val")):
        sample_paths = list(mode_dir.rglob(f"*_samples_mmmu_{split}.jsonl"))
        if len(sample_paths) != 1:
            raise RuntimeError(
                f"Expected one {split} sample JSONL under {mode_dir}, "
                f"found {sample_paths}"
            )
        rows, split_summary = correct_samples(sample_paths[0])
        corrected_score = mmmu_aggregate_results(
            [row["mmmu_acc"] for row in rows]
        )
        metric_key = "mmmu_acc,none"
        old_score = result_data["results"][task][metric_key]
        result_data["results"][task][metric_key] = corrected_score
        split_summary.update(
            {
                "old_score": old_score,
                "corrected_score": corrected_score,
                "sample_file": str(sample_paths[0]),
            }
        )
        mode_summary["splits"][split] = split_summary

    result_data["answer_extraction_correction"] = {
        "method": "final_valid_boxed_choice_first",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "mode_summary": mode_summary,
    }
    atomic_write_text(
        result_path,
        json.dumps(result_data, indent=2, ensure_ascii=False) + "\n",
    )
    return mode_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_root",
        nargs="?",
        type=Path,
        default=(
            ROOT
            / "eval/logs/lavida_thinking_3bench_no_original_20260724/mmmu"
        ),
    )
    args = parser.parse_args()

    summary = {
        "method": "final_valid_boxed_choice_first",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "modes": {},
    }
    for mode in ("swd", "psp", "psp_vrg"):
        summary["modes"][mode] = correct_mode(args.run_root / mode)

    summary_path = args.run_root / "boxed_answer_correction_summary.json"
    atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
