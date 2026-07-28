#!/usr/bin/env python3
"""Re-parse MMMU MC samples without random.choice fallback.

Uses the current ``parse_multi_choice_response`` (boxed-first, unparseable→"").
Backs up originals with ``.pre_no_random`` if not already present.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import string
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from lmms_eval.tasks.mmmu.utils import (  # noqa: E402
    mmmu_aggregate_results,
    parse_multi_choice_response,
)


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.pre_no_random{path.suffix}")


def ensure_backup(path: Path) -> Path:
    backup = backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def choice_maps(doc: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    options = doc.get("options", [])
    if isinstance(options, str):
        options = ast.literal_eval(options)
    labels = list(string.ascii_uppercase[: len(options)])
    index2ans = {lab: str(opt) for lab, opt in zip(labels, options)}
    return labels, index2ans


def correct_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    changed = 0
    unparseable = 0
    old_correct = 0
    new_correct = 0

    for row in rows:
        metric = row["mmmu_acc"]
        gold = str(metric["answer"])
        old_pred = metric["parsed_pred"][0]
        if old_pred == gold:
            old_correct += 1

        if metric["question_type"] == "multiple-choice":
            labels, index2ans = choice_maps(row["doc"])
            response = str(row["filtered_resps"][0])
            parsed = parse_multi_choice_response(response, labels, index2ans)
            if parsed == "":
                unparseable += 1
            if parsed != old_pred:
                changed += 1
            row["mmmu_acc"]["parsed_pred"] = [parsed]
            if "mmmu_acc_pass_at_k" in row:
                row["mmmu_acc_pass_at_k"]["parsed_pred"] = [parsed]

        if row["mmmu_acc"]["parsed_pred"][0] == gold:
            new_correct += 1

    ensure_backup(path)
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    return rows, {
        "samples": len(rows),
        "changed_rows": changed,
        "unparseable_mc": unparseable,
        "old_exact_letter_matches": old_correct,
        "corrected_exact_letter_matches": new_correct,
        "backup": str(backup_path(path)),
    }


def correct_mode(mode_dir: Path) -> dict[str, Any]:
    result_paths = list(mode_dir.rglob("*_results.json"))
    # Prefer non-backup results
    result_paths = [
        p
        for p in result_paths
        if ".pre_" not in p.name and not p.name.endswith(".tmp")
    ]
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
        sample_paths = [
            p
            for p in mode_dir.rglob(f"*_samples_mmmu_{split}.jsonl")
            if ".pre_" not in p.name
        ]
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
        "method": "boxed_first_no_random_fallback",
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
        "mode_dir",
        type=Path,
        help="Directory containing *_results.json and samples jsonl "
        "(e.g. .../mmmu/vcd_prefix_cache)",
    )
    args = parser.parse_args()
    summary = correct_mode(args.mode_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
