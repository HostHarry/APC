#!/usr/bin/env python3
"""Batch GPT scoring for the CV-DCD sweep using VLMEvalKit's LLaVABench.evaluate().

Reads OPENAI_API_KEY (and optionally OPENAI_API_BASE) from environment.
For each config directory, invokes the LLaVABench class-level evaluate method,
which uses build_judge (OpenAI-compatible) + build_prompt + LLaVABench_atomeval
under the hood — exactly the same pipeline VLMEvalKit's `run.py --mode all` uses.

Outputs, per config:
  - <name>_LLaVABench_<judge>.xlsx   (row-level scores, columns: gpt4_score, score)
  - <name>_LLaVABench_score.csv      (summary: Relative Score, VLM Score, GPT4 Score)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VLMEVAL_ROOT = HERE.parent
sys.path.insert(0, str(VLMEVAL_ROOT))

from vlmeval.dataset import LLaVABench  # noqa: E402


def find_pred_xlsx(config_dir: Path) -> Path | None:
    matches = list(config_dir.rglob("*_LLaVABench.xlsx"))
    return matches[0] if matches else None


def score_one(config_dir: Path, judge_model: str, nproc: int,
              retry: int, verbose: bool = False, force: bool = False) -> dict:
    pred_xlsx = find_pred_xlsx(config_dir)
    if pred_xlsx is None:
        return {"config": config_dir.name, "status": "no_pred_file"}

    stem = pred_xlsx.stem
    parent = pred_xlsx.parent
    result_xlsx = parent / f"{stem}_{judge_model}.xlsx"
    score_csv = parent / f"{stem}_score.csv"

    if result_xlsx.exists() and not force:
        return {
            "config": config_dir.name,
            "status": "already_scored",
            "result": str(result_xlsx),
        }

    judge_kwargs = {
        "model": judge_model,
        "nproc": nproc,
        "retry": retry,
        "verbose": verbose,
    }

    t0 = time.time()
    try:
        ret_df = LLaVABench.evaluate(str(pred_xlsx), **judge_kwargs)
    except Exception as exc:
        return {
            "config": config_dir.name,
            "status": f"eval_error: {exc}",
            "elapsed_s": round(time.time() - t0, 1),
        }
    elapsed = round(time.time() - t0, 1)

    return {
        "config": config_dir.name,
        "status": "ok",
        "elapsed_s": elapsed,
        "summary": ret_df.to_dict(orient="records") if hasattr(ret_df, "to_dict") else str(ret_df),
        "result": str(result_xlsx) if result_xlsx.exists() else "(check parent dir)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=str,
                    default="outputs/cvdcd_sweep/cvdcd_llava_20260705_032934",
                    help="Directory with lam*_drop subdirectories.")
    ap.add_argument("--also-baseline", type=str, default="",
                    help="Optional: baseline dir to (re-)score for consistency.")
    ap.add_argument("--judge", type=str, default="gpt-4o-mini",
                    help="Judge model name (must be OpenAI-compatible chat model).")
    ap.add_argument("--api-nproc", type=int, default=8,
                    help="Parallel API calls per config.")
    ap.add_argument("--retry", type=int, default=3, help="Retry count on API failure.")
    ap.add_argument("--force", action="store_true",
                    help="Re-score even if <name>_<judge>.xlsx already exists.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--configs", type=str, nargs="*", default=None,
                    help="Optional: only score these subdir names (space-separated).")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY is not set. Run:\n"
              "  export OPENAI_API_KEY='sk-...'\n"
              "and (optionally) OPENAI_API_BASE.")
        sys.exit(2)

    print(f"[INFO] Judge model    : {args.judge}")
    print(f"[INFO] API_BASE       : {os.environ.get('OPENAI_API_BASE', 'OFFICIAL')}")
    print(f"[INFO] API parallelism: {args.api_nproc}")
    print(f"[INFO] Retry count    : {args.retry}")
    print()

    sweep = Path(args.sweep_dir)
    targets: list[Path] = []
    if args.also_baseline:
        targets.append(Path(args.also_baseline))
    for sub in sorted(sweep.iterdir()):
        if not sub.is_dir():
            continue
        if args.configs and sub.name not in args.configs:
            continue
        if not sub.name.startswith("lam"):
            continue
        targets.append(sub)

    if not targets:
        print("[ERROR] No targets found.")
        sys.exit(1)

    print(f"[INFO] Will score {len(targets)} config(s):")
    for t in targets:
        print(f"  - {t.name}")
    print()

    results = []
    for i, t in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] Scoring: {t.name}")
        r = score_one(t, args.judge, args.api_nproc, args.retry,
                      args.verbose, args.force)
        results.append(r)
        print(f"  → status={r['status']}", flush=True)
        if r.get("summary"):
            for row in r["summary"]:
                print(f"      {row}")
        print()

    print("=" * 70)
    print("Summary:")
    for r in results:
        ok = "✓" if r["status"] == "ok" else ("○" if r["status"] == "already_scored" else "✗")
        print(f"  {ok} {r['config']:<28} {r['status']}"
              + (f"  ({r.get('elapsed_s')}s)" if 'elapsed_s' in r else ""))


if __name__ == "__main__":
    main()
