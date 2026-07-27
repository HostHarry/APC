#!/usr/bin/env python3
"""Offline GPT judge for deferred LLaVA-Bench-COCO sample logs.

Reads samples_*.jsonl written with LLAVA_JUDGE_SKIP=1 (scores=[-999,-999]),
calls the judge API using the saved `content` field, and writes:
  - judged samples jsonl
  - summary json with relative scores (model/gpt4 * 100), matching lmms-eval
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests


NUM_SECONDS_TO_SLEEP = 0.5
METRIC_KEYS = [
    "gpt_eval_llava_conv",
    "gpt_eval_llava_detail",
    "gpt_eval_llava_complex",
    "gpt_eval_llava_all",
]


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def ensure_proxy() -> None:
    if os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"):
        return
    os.environ["http_proxy"] = "http://127.0.0.1:7897"
    os.environ["https_proxy"] = "http://127.0.0.1:7897"
    os.environ["HTTP_PROXY"] = os.environ["http_proxy"]
    os.environ["HTTPS_PROXY"] = os.environ["https_proxy"]


def parse_score(review: str) -> list[float]:
    try:
        score_pair = review.split("\n")[0].replace(",", " ")
        sp = score_pair.split()
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
    except Exception:
        pass
    return [-1.0, -1.0]


def get_eval(
    content: str,
    *,
    api_url: str,
    api_key: str,
    model: str,
    max_tokens: int = 1024,
    retries: int = 5,
) -> tuple[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful and precise assistant for checking "
                    "the quality of the answer."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    last_err = ""
    for attempt in range(retries):
        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=90
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            used = data.get("model", model)
            if text:
                return text, used
            last_err = "empty content"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP * (attempt + 1))
    print(f"judge failed after {retries} retries: {last_err}", flush=True)
    return "", ""


def relative_score(pairs: list[list[float]]) -> float | None:
    if not pairs:
        return None
    stats = np.asarray(pairs).mean(0).tolist()
    if stats[0] == 0:
        return None
    return round(stats[1] / stats[0] * 100, 1)


def judge_one(
    row: dict,
    *,
    api_url: str,
    api_key: str,
    model: str,
) -> dict:
    payload = row.get("gpt_eval_llava_all") or {}
    content = payload.get("content") or ""
    if not content:
        raise ValueError("sample missing gpt_eval_llava_all.content")

    # Already judged?
    scores = payload.get("scores") or []
    if (
        len(scores) == 2
        and -999 not in scores
        and payload.get("eval_model") not in (None, "", "deferred", "Failed Request")
        and payload.get("review")
        and "deferred" not in str(payload.get("review", "")).lower()
    ):
        return row

    review, model_name = get_eval(
        content, api_url=api_url, api_key=api_key, model=model
    )
    if not review:
        scores = [-1.0, -1.0]
        model_name = model_name or "Failed Request"
        review = "Failed to Get a Proper Review."
    else:
        scores = parse_score(review)

    category = payload.get("category", "")
    metric = f"gpt_eval_llava_{category.replace('llava_bench_', '')}"
    updated = {
        "question": payload.get("question", ""),
        "ans1": payload.get("ans1", ""),
        "ans2": payload.get("ans2", ""),
        "context": payload.get("context", ""),
        "category": category,
        "review": review,
        "scores": scores,
        "eval_model": model_name,
        "content": content,
    }
    for key in METRIC_KEYS:
        if key == "gpt_eval_llava_all" or key == metric:
            row[key] = dict(updated)
        else:
            non = dict(updated)
            non["scores"] = [-999, -999]
            row[key] = non
    return row


def aggregate(rows: list[dict]) -> dict:
    by_cat: dict[str, list[list[float]]] = defaultdict(list)
    all_pairs: list[list[float]] = []
    failed = 0
    for row in rows:
        payload = row["gpt_eval_llava_all"]
        scores = payload.get("scores") or [-1, -1]
        if -999 in scores or -1 in scores:
            failed += 1
            continue
        all_pairs.append(scores)
        cat = payload.get("category", "").replace("llava_bench_", "")
        by_cat[cat].append(scores)

    summary = {
        "n": len(rows),
        "judged_ok": len(all_pairs),
        "failed_or_skipped": failed,
        "gpt_eval_llava_all": relative_score(all_pairs),
        "gpt_eval_llava_conv": relative_score(by_cat.get("conv", [])),
        "gpt_eval_llava_detail": relative_score(by_cat.get("detail", [])),
        "gpt_eval_llava_complex": relative_score(by_cat.get("complex", [])),
        "mean_scores_all": (
            [round(x, 3) for x in np.asarray(all_pairs).mean(0).tolist()]
            if all_pairs
            else None
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_jsonl", type=Path, nargs="+")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(
            "/root/autodl-tmp/lladav_vchd_server_bundle/VLMEvalKit/.env"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLAVA_JUDGE_MODEL", os.getenv("GPT_EVAL_MODEL_NAME", "gpt-4o")),
    )
    parser.add_argument(
        "--out-suffix",
        default="_judged_gpt4o",
        help="appended to samples stem before .jsonl",
    )
    parser.add_argument(
        "--summary-name",
        default="llava_judge_gpt4o.json",
        help="summary filename written next to samples",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    ensure_proxy()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_url = os.environ.get(
        "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
    )
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    for samples_path in args.samples_jsonl:
        rows = [json.loads(line) for line in samples_path.open()]
        print(f"=== {samples_path} ({len(rows)} samples) model={args.model}", flush=True)
        judged = []
        for i, row in enumerate(rows):
            judged.append(
                judge_one(row, api_url=api_url, api_key=api_key, model=args.model)
            )
            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                print(f"progress {i + 1}/{len(rows)}", flush=True)

        out_jsonl = samples_path.with_name(
            samples_path.stem + args.out_suffix + ".jsonl"
        )
        with out_jsonl.open("w") as f:
            for row in judged:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = aggregate(judged)
        summary["model"] = args.model
        summary["source_samples"] = str(samples_path)
        summary["judged_samples"] = str(out_jsonl)
        out_summary = samples_path.parent / args.summary_name
        out_summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"wrote {out_jsonl}", flush=True)
        print(f"wrote {out_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
