#!/usr/bin/env python3
"""Offline M3CoT rescoring with the boxed-first extractor.

The original ``extract_m3cot_answer`` (official judge_answer rules) misreads
reasoning-style outputs that end in ``\\boxed{B}``: rule 1 wants ``(B)``,
rule 2 matches the LAST option-text mention inside the CoT (near random when
the CoT discusses every option), rule 3 wants standalone letter tokens.
This script rescoreseach samples_m3cot_full.jsonl with the patched extractor
(boxed -> "answer is X" -> official rules).

Usage:
  PYTHONPATH=eval python3 eval/rescore_m3cot_boxed_first.py --runs DIR [DIR...]
Each DIR must contain *samples_m3cot_full.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

from lmms_eval.tasks.m3cot.utils import extract_m3cot_answer


def response_text(row: dict) -> str:
    x = row.get("filtered_resps") or row.get("resps")
    while isinstance(x, list):
        x = x[0] if x else ""
    return str(x or "")


def score_file(path: str) -> dict:
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    correct = 0
    unparsed = 0
    official = 0.0
    by_domain: Counter = Counter()
    by_domain_ok: Counter = Counter()
    details = []
    for row in rows:
        doc = row["doc"]
        choices = [
            doc[k]
            for k in ("A", "B", "C", "D", "E")
            if doc.get(k) not in (None, "") and str(doc.get(k)) != "nan"
        ]
        pred = extract_m3cot_answer(response_text(row), choices)
        target = str(doc["answer"]).strip().upper()
        hit = pred == target
        correct += hit
        unparsed += pred == "FAILED"
        official += float(row.get("m3cot_accuracy", 0) or 0)
        domain = doc.get("domain", "?")
        by_domain[domain] += 1
        by_domain_ok[domain] += hit
        details.append(
            {"id": doc.get("id"), "target": target, "pred": pred, "ok": hit}
        )
    return {
        "n": n,
        "official_acc": round(official / n * 100, 2),
        "boxed_first_acc": round(correct / n * 100, 2),
        "unparsed": unparsed,
        "by_domain": {
            k: round(by_domain_ok[k] / v * 100, 2) for k, v in by_domain.items()
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    summary = {}
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        files = sorted(glob.glob(os.path.join(run, "**", "*samples_m3cot_full.jsonl"), recursive=True))
        if not files:
            print(f"SKIP (no samples): {run}")
            continue
        label = Path(run).name
        result = score_file(files[-1])
        details = result.pop("details")
        summary[label] = result
        print(
            f"{label:36} n={result['n']:5} official={result['official_acc']:6.2f}% "
            f"boxed-first={result['boxed_first_acc']:6.2f}% unparsed={result['unparsed']}"
        )
        print(f"{'':36} by_domain={result['by_domain']}")
        if args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            with (out / f"{label}__details.jsonl").open("w") as fh:
                for row in details:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.out_dir:
        with (Path(args.out_dir) / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
