#!/usr/bin/env python3
"""Score MMBench submissions with local prefetch + GPT-4o judge.

Inputs (any one):
  - submissions/mmbench_en_dev_results.xlsx
  - *samples_mmbench*.jsonl

Outputs under --output-dir (default: <input_parent>/gpt4o_score):
  - judge_cache.jsonl
  - details.xlsx
  - circular_main.xlsx
  - summary.json   (circular primary + flat_accuracy*)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

CIRCULAR_MOD = int(1e6)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        column: column.lower() if column not in "ABCDE" else column
        for column in frame.columns
    }
    return frame.rename(columns=rename).sort_values("index").reset_index(drop=True)


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                doc = row["doc"]
                prediction = row.get("filtered_resps", row.get("resps"))
                if isinstance(prediction, list):
                    prediction = prediction[0]
                    if isinstance(prediction, list):
                        prediction = prediction[0]
                item = {
                    "index": doc["index"],
                    "question": doc["question"],
                    "answer": doc.get("answer"),
                    "prediction": prediction,
                    "hint": doc.get("hint"),
                    "source": doc.get("source"),
                    "split": doc.get("split"),
                    "category": doc.get("category"),
                    "l2-category": doc.get("L2-category", doc.get("l2-category")),
                }
                for letter in "ABCDE":
                    item[letter] = doc.get(letter)
                rows.append(item)
        frame = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported input: {path}")
    return normalize_frame(frame)


def prediction_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            key = f"{item['index']}::{item['prediction_hash']}::{item['model']}"
            cache[key] = item
    return cache


def load_detail_cache(path: Optional[Path]) -> Dict[str, str]:
    """Reuse completed judgments when the input index and prediction match."""
    if path is None or not path.exists():
        return {}
    frame = normalize_frame(pd.read_excel(path))
    if not {"index", "prediction", "judged_prediction"} <= set(frame.columns):
        return {}
    cache: Dict[str, str] = {}
    for _, item in frame.iterrows():
        letter = str(item["judged_prediction"]).strip().upper()
        if letter not in "ABCDE":
            continue
        key = f"{int(item['index'])}::{prediction_hash(item['prediction'])}"
        cache[key] = letter
    return cache


def append_cache(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_option_str(item: pd.Series) -> str:
    parts = []
    for letter in "ABCD":
        value = item.get(letter)
        if pd.isna(value) or str(value) == "nan":
            continue
        parts.append(f"{letter}. {value}")
    return " ".join(parts)


def extract_letter(text: str) -> Optional[str]:
    text = str(text).strip()
    if not text:
        return None
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else None


class OpenAIJudge:
    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str,
        timeout: int = 60,
        max_retries: int = 5,
        sleep_s: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_s = sleep_s
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._usage_lock = threading.Lock()
        self._thread_local = threading.local()

    def judge(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                session = getattr(self._thread_local, "session", None)
                if session is None:
                    session = requests.Session()
                    self._thread_local.session = session
                response = session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                raw = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage") or {}
                with self._usage_lock:
                    for key in self.usage:
                        self.usage[key] += int(usage.get(key, 0) or 0)
                return {
                    "status": "ok",
                    "letter": extract_letter(raw),
                    "raw": raw,
                    "usage": usage,
                    "attempts": attempt,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                time.sleep(self.sleep_s * attempt)
        return {
            "status": "error",
            "letter": None,
            "raw": last_error or "Failed to obtain answer via API",
            "usage": {},
            "attempts": self.max_retries,
        }


def circular_hit_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Official MMBench circular aggregation over main indices (< 1e6)."""
    work = frame.copy()
    work["main_index"] = work["index"].astype(int) % CIRCULAR_MOD
    work["row_hit"] = work["judged_prediction"].astype(str) == work["answer"].astype(
        str
    )
    rows = []
    for main_index, group in work.groupby("main_index", sort=True):
        hit = bool(group["row_hit"].all())
        first = group.iloc[0]
        rows.append(
            {
                "index": int(main_index),
                "hit": int(hit),
                "n_variants": int(len(group)),
                "category": first.get("category"),
                "l2-category": first.get("l2-category"),
                "answer": first.get("answer"),
            }
        )
    return pd.DataFrame(rows)


def score_frame(
    frame: pd.DataFrame,
    *,
    judge: OpenAIJudge,
    cache_path: Path,
    evaluator: MMBench_Evaluator,
    detail_cache_path: Optional[Path] = None,
    workers: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cache = load_cache(cache_path)
    detail_cache = load_detail_cache(detail_cache_path)
    judged: list[Optional[dict]] = [None] * len(frame)
    pending = []
    locally_inferred = 0
    gpt_judged = 0
    detail_reused = 0
    cache_reused = 0
    api_calls = 0

    for position, (_, item) in enumerate(frame.iterrows()):
        choices = evaluator.build_choices(item)
        local = evaluator.can_infer(str(item["prediction"]), choices)
        if local:
            letter = str(local).upper()
            source = "local"
            locally_inferred += 1
        else:
            pred_hash = prediction_hash(item["prediction"])
            detail_key = f"{int(item['index'])}::{pred_hash}"
            cache_key = f"{int(item['index'])}::{pred_hash}::{judge.model}"
            if detail_key in detail_cache:
                letter = detail_cache[detail_key]
                source = "details"
                detail_reused += 1
                gpt_judged += 1
            elif (
                (cached := cache.get(cache_key))
                and cached.get("status") == "ok"
                and cached.get("letter")
            ):
                letter = str(cached["letter"]).upper()
                source = "cache"
                cache_reused += 1
                gpt_judged += 1
            else:
                prompt = evaluator.build_prompt(
                    item["question"],
                    build_option_str(item),
                    item["prediction"],
                )
                pending.append(
                    (position, item.to_dict(), pred_hash, cache_key, prompt)
                )
                continue

        out = item.to_dict()
        out["judged_prediction"] = letter
        out["hit"] = int(str(letter) == str(item["answer"]).strip().upper())
        out["judge_source"] = source
        judged[position] = out

    if pending:
        print(
            f"GPT requests: {len(pending)} with {workers} workers "
            f"(details reused={detail_reused}, cache reused={cache_reused})",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(judge.judge, prompt): (
                    position,
                    item,
                    pred_hash,
                    cache_key,
                )
                for position, item, pred_hash, cache_key, prompt in pending
            }
            completed = 0
            for future in as_completed(futures):
                position, item, pred_hash, cache_key = futures[future]
                result = future.result()
                letter = (result.get("letter") or "E").upper()
                record = {
                    "index": int(item["index"]),
                    "prediction_hash": pred_hash,
                    "model": judge.model,
                    "status": result["status"],
                    "letter": letter if result["status"] == "ok" else None,
                    "raw": result.get("raw"),
                    "usage": result.get("usage"),
                    "attempts": result.get("attempts"),
                }
                append_cache(cache_path, record)
                cache[cache_key] = record
                if result["status"] != "ok":
                    letter = "E"
                out = dict(item)
                out["judged_prediction"] = letter
                out["hit"] = int(
                    str(letter) == str(item["answer"]).strip().upper()
                )
                out["judge_source"] = "api"
                judged[position] = out
                gpt_judged += 1
                api_calls += 1
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"GPT progress: {completed}/{len(pending)}",
                        flush=True,
                    )

    if any(item is None for item in judged):
        raise RuntimeError("Scoring left unresolved rows")
    details = pd.DataFrame(judged)
    details["row_hit"] = details["judged_prediction"].astype(str) == details[
        "answer"
    ].astype(str)
    details["main_index"] = details["index"].astype(int) % CIRCULAR_MOD
    circular = circular_hit_table(details)
    details["circular_hit"] = details["main_index"].map(
        circular.set_index("index")["hit"]
    )

    category_accuracy = (
        circular.groupby("category")["hit"].mean().dropna().to_dict()
        if "category" in circular
        else {}
    )
    l2_accuracy = (
        circular.groupby("l2-category")["hit"].mean().dropna().to_dict()
        if "l2-category" in circular
        else {}
    )

    summary = {
        "model": judge.model,
        "rows": int(len(details)),
        "locally_inferred": int(locally_inferred),
        "gpt_judged": int(gpt_judged),
        "detail_reused": int(detail_reused),
        "cache_reused": int(cache_reused),
        "api_calls": int(api_calls),
        "workers": int(workers),
        "cached_api_usage": dict(judge.usage),
        "protocol": "circular",
        "n_main": int(len(circular)),
        "accuracy": float(circular["hit"].mean()) if len(circular) else 0.0,
        "accuracy_percent": float(circular["hit"].mean() * 100) if len(circular) else 0.0,
        "flat_accuracy": float(details["row_hit"].mean()) if len(details) else 0.0,
        "flat_accuracy_percent": float(details["row_hit"].mean() * 100)
        if len(details)
        else 0.0,
        "category_accuracy": category_accuracy,
        "l2_category_accuracy": l2_accuracy,
    }
    return details, circular, summary


def resolve_default_output(input_path: Path) -> Path:
    # .../mode/submissions/file.xlsx -> .../mode/gpt4o_score
    if input_path.parent.name == "submissions":
        return input_path.parent.parent / "gpt4o_score"
    return input_path.parent / "gpt4o_score"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="MMBench xlsx or samples jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sibling gpt4o_score/)",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument(
        "--api-url",
        default=os.getenv(
            "OPENAI_API_URL",
            os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions"),
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="Or set OPENAI_API_KEY",
    )
    parser.add_argument("--mode-name", default=None, help="Stored in summary.json")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("MMBENCH_GPT_WORKERS", "16")),
        help="Concurrent GPT requests (default: 16)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.api_key:
        raise SystemExit("Missing API key: set OPENAI_API_KEY or pass --api-key")

    input_path = args.input.resolve()
    output_dir = (args.output_dir or resolve_default_output(input_path)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_input(input_path)
    evaluator = MMBench_Evaluator(
        API_KEY=args.api_key,
        API_URL=args.api_url,
        model_version=args.model,
    )
    judge = OpenAIJudge(api_key=args.api_key, api_url=args.api_url, model=args.model)
    details, circular, summary = score_frame(
        frame,
        judge=judge,
        cache_path=output_dir / "judge_cache.jsonl",
        evaluator=evaluator,
        detail_cache_path=output_dir / "details.xlsx",
        workers=args.workers,
    )
    if args.mode_name:
        summary["mode"] = args.mode_name

    details.to_excel(output_dir / "details.xlsx", index=False)
    circular.to_excel(output_dir / "circular_main.xlsx", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
