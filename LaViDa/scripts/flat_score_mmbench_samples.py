#!/usr/bin/env python3
"""Flat MMBench accuracy: per-row hit mean, no circular aggregation.

judged_prediction = can_infer(prediction) or GPT extract
row_hit = judged_prediction == answer
flat_accuracy = mean(row_hit)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_jsonl", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(
            "/root/autodl-tmp/lladav_vchd_server_bundle/VLMEvalKit/.env"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GPT_EVAL_MODEL_NAME", "gpt-4o"),
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    if not os.environ.get("http_proxy") and not os.environ.get("HTTP_PROXY"):
        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"
        os.environ["HTTP_PROXY"] = os.environ["http_proxy"]
        os.environ["HTTPS_PROXY"] = os.environ["https_proxy"]

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "eval"))
    from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_url = os.environ.get(
        "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
    )
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    evaluator = MMBench_Evaluator(
        sys_prompt="There are several options:",
        API_KEY=api_key,
        API_URL=api_url,
        model_version=args.model,
    )

    rows = []
    with args.samples_jsonl.open() as f:
        for line in f:
            obj = json.loads(line)
            item = dict(obj["gpt_eval_score"])
            for c in "ABCD":
                item.setdefault(c, obj.get("doc", {}).get(c, "nan"))
            rows.append(item)

    hits = 0
    rule = 0
    gpt = 0
    for i, item in enumerate(rows):
        choices = evaluator.build_choices(item)
        pred = evaluator.can_infer(str(item.get("prediction", "")), choices)
        source = "rule"
        if not pred:
            pred, _ = evaluator.extract_answer_from_item(item)
            source = "gpt"
            gpt += 1
        else:
            rule += 1
        gold = str(item["answer"]).strip().upper()
        hit = str(pred).strip().upper() == gold
        hits += int(hit)
        item["judged_prediction"] = pred
        item["judge_source"] = source
        item["row_hit"] = hit
        if (i + 1) % 500 == 0 or (i + 1) == len(rows):
            print(
                f"progress {i + 1}/{len(rows)} "
                f"flat_so_far={hits / (i + 1) * 100:.2f}% "
                f"rule={rule} gpt={gpt}",
                flush=True,
            )

    flat = hits / len(rows) if rows else 0.0
    details = {
        "metric": "flat_accuracy",
        "definition": "mean(row_hit); no circular aggregation",
        "flat_accuracy": flat,
        "flat_accuracy_percent": flat * 100,
        "hits": hits,
        "n": len(rows),
        "rule_resolved": rule,
        "gpt_resolved": gpt,
        "model": args.model,
        "source_samples": str(args.samples_jsonl),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n")
    print(
        f"FLAT={flat * 100:.4f}% ({hits}/{len(rows)}) "
        f"rule={rule} gpt={gpt}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
