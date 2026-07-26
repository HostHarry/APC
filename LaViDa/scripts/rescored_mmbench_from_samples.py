#!/usr/bin/env python3
"""Re-score MMBench samples jsonl without re-running model inference.

Uses the existing MMBench evaluator path:
  1) rule-based option extraction (can_infer / prefetch)
  2) GPT only when rule extraction fails
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_jsonl", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("GPT_EVAL_MODEL_NAME", "gpt-4o"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "eval"))

    from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    evaluator = MMBench_Evaluator(
        sys_prompt="There are several options:",
        API_KEY=api_key,
        API_URL=api_url,
        model_version=args.model,
    )

    results = []
    rule_hits = 0
    with args.samples_jsonl.open() as f:
        for line in f:
            obj = json.loads(line)
            item = dict(obj["gpt_eval_score"])
            # Ensure option fields exist for can_infer / eval_result.
            for c in "ABCD":
                item.setdefault(c, obj.get("doc", {}).get(c, "nan"))
            results.append(item)
            choices = evaluator.build_choices(item)
            if evaluator.can_infer(str(item.get("prediction", "")), choices):
                rule_hits += 1

    print(f"loaded={len(results)} rule_prefetchable≈{rule_hits} model={args.model}")
    overall_acc, category_acc, l2_category_acc = evaluator.eval_result(results, eval_method="openai")
    details = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
        "n": len(results),
        "rule_prefetchable": rule_hits,
        "model": args.model,
        "source_samples": str(args.samples_jsonl),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n")
    print(f"overall_acc%={overall_acc * 100:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
