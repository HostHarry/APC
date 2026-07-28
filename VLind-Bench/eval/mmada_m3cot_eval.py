#!/usr/bin/env python3
"""MMaDA × M3CoT harness wired to the official LightChen233/M3CoT scorer.

Generation uses cloud_bundle MMaDA (VLMEvalKit wrapper). Scoring uses
``third_party/M3CoT/evaluate.py --setting custom`` (official
``utils.metric.judge_answer``), not a private heuristic.

Official refs:
  - Gen-Verse/MMaDA VLM: VLMEvalKit (no M3CoT dataset registered upstream)
  - LightChen233/M3CoT: https://github.com/LightChen233/M3CoT ``evaluate.py``
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

THIRD_PARTY_M3COT = Path("/root/autodl-tmp/third_party/M3CoT")


def build_prompt(doc: dict, *, prompt_style: str = "cot") -> str:
    """Match LightChen zero-shot direct/cot style prompts."""
    question = doc.get("question") or ""
    choices = doc.get("choices") or doc.get("options") or []
    if isinstance(choices, dict):
        choice_lines = [f"({k}) {v}" for k, v in choices.items()]
    else:
        choice_lines = [
            f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)
        ]
    body = question
    if choice_lines:
        body = f"{question}\n" + "\n".join(choice_lines)
    if prompt_style == "direct":
        return f"{body}\nAnswer with the option's letter from the given choices."
    # cot (default): official M3CoT CoT cue
    return (
        f"{body}\nLet's think step by step.\n"
        "Put the final answer in the form (X) where X is the option letter."
    )


def target_letter(doc: dict) -> str:
    ans = doc.get("answer")
    if isinstance(ans, str) and len(ans.strip()) == 1 and ans.strip().isalpha():
        return ans.strip().upper()
    if isinstance(ans, int):
        return chr(ord("A") + ans)
    choices = doc.get("choices") or doc.get("options") or []
    if isinstance(choices, list) and ans in choices:
        return chr(ord("A") + choices.index(ans))
    return str(ans).strip().upper()[:1]


def to_official_jsonl_row(doc: dict, pred: str, idx: int) -> dict:
    """Schema required by LightChen233/M3CoT evaluate.py --setting custom."""
    choices = doc.get("choices") or doc.get("options") or []
    if isinstance(choices, dict):
        choice_list = list(choices.values())
    else:
        choice_list = list(choices)
    sample_id = doc.get("id") or doc.get("qid") or str(idx)
    return {
        "id": str(sample_id),
        "choices": choice_list,
        "answer": target_letter(doc),
        "domain": doc.get("domain", "unknown"),
        "topic": doc.get("topic", "unknown"),
        "messages": [
            doc.get("question") or "",
            pred,
        ],
    }


def run_official_metric(jsonl_path: Path) -> dict:
    """Invoke official evaluate.py --setting custom and parse Total line."""
    if not (THIRD_PARTY_M3COT / "evaluate.py").is_file():
        raise FileNotFoundError(
            f"Official M3CoT repo missing at {THIRD_PARTY_M3COT}"
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(THIRD_PARTY_M3COT) + (
        (os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(THIRD_PARTY_M3COT / "evaluate.py"),
            "--setting",
            "custom",
            "--metric_path",
            str(jsonl_path),
            "--metric_by",
            "all",
        ],
        cwd=str(THIRD_PARTY_M3COT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    acc = None
    total = None
    for line in out.splitlines():
        if line.startswith("Total:"):
            # Total: N, Correct: xx.xx%
            try:
                parts = line.replace("%", "").split(",")
                total = int(parts[0].split(":")[1].strip())
                acc = float(parts[1].split(":")[1].strip()) / 100.0
            except Exception:
                pass
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "accuracy": acc,
        "n": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmada-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/MMaDA_DCD_cloud_bundle_20260711/MMaDA"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/MMaDA-8B-MixCoT"),
    )
    parser.add_argument(
        "--vq-model-path",
        type=Path,
        default=Path("/root/autodl-tmp/MMaDA-8B-MixCoT"),
    )
    parser.add_argument(
        "--strategy",
        choices=("original", "vcd", "vchd", "dcd", "cv_dcd"),
        default="vcd",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt-style",
        choices=("cot", "direct"),
        default="cot",
        help="LightChen zero-shot prompt style",
    )
    parser.add_argument(
        "--skip-official-metric",
        action="store_true",
        help="Only write JSONL; do not call third_party/M3CoT/evaluate.py",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.mmada_root / "evaluation" / "VLMEvalKit"))
    os.environ["MMADA_DECODE_STRATEGY"] = args.strategy
    from vlmeval.vlm.mmada.mmada import MMaDA

    # Prefer magvitv2 path if present beside MixCoT.
    vq = args.vq_model_path
    magvit = Path("/root/autodl-tmp/MMaDA-8B-MixCoT")  # may contain or sibling
    alt = Path("/root/autodl-tmp/magvitv2")
    if (alt / "config.json").exists() or alt.is_dir():
        # keep user override unless default-ish
        pass

    model = MMaDA(
        model_path=str(args.model_path),
        tokenizer_path=str(args.model_path),
        vq_model_path=str(vq),
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
        decode_strategy=args.strategy,
        use_config_file=False,
    )

    ds = load_dataset("LightChen2333/M3CoT", split="test")
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".official.jsonl")
    rows = []
    t0 = time.time()
    with open(jsonl_path, "w") as jf:
        for i, doc in enumerate(tqdm(ds, desc=f"m3cot/{args.strategy}")):
            image = doc.get("image")
            pred = ""
            if image is not None:
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                else:
                    image = image.convert("RGB")
                tmp = args.output.parent / f"_tmp_m3cot_{os.getpid()}_{i}.png"
                image.save(tmp)
                message = [
                    {"type": "image", "value": str(tmp)},
                    {
                        "type": "text",
                        "value": build_prompt(
                            doc, prompt_style=args.prompt_style
                        ),
                    },
                ]
                pred = model.generate_mmada(message, dataset="M3CoT")
                try:
                    tmp.unlink()
                except OSError:
                    pass
            official = to_official_jsonl_row(doc, pred, i)
            jf.write(json.dumps(official, ensure_ascii=False) + "\n")
            jf.flush()
            rows.append(official)

    summary = {
        "strategy": args.strategy,
        "n": len(rows),
        "elapsed_s": time.time() - t0,
        "schedule": {
            "max_new_tokens": args.max_new_tokens,
            "steps": args.steps,
            "block_length": args.block_length,
        },
        "prompt_style": args.prompt_style,
        "official_jsonl": str(jsonl_path),
        "official_scorer": str(THIRD_PARTY_M3COT / "evaluate.py"),
    }
    if not args.skip_official_metric:
        metric = run_official_metric(jsonl_path)
        summary["official_metric"] = {
            "accuracy": metric["accuracy"],
            "n": metric["n"],
            "returncode": metric["returncode"],
            "stdout_tail": (metric["stdout"] or "")[-2000:],
        }
        print(
            f"[m3cot-official] strategy={args.strategy} "
            f"acc={metric['accuracy']} n={metric['n']} "
            f"elapsed={summary['elapsed_s']:.1f}s jsonl={jsonl_path}"
        )
    else:
        print(
            f"[m3cot] wrote {jsonl_path} n={len(rows)}; "
            "skip official metric"
        )

    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
