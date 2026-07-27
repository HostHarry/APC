#!/usr/bin/env python3
"""Offline MMMU rescoring: boxed-first, no random fallback.

Priority for multiple-choice:
  1) last \\boxed{...} (closed or unclosed), letter or option text
  2) "answer is A" / "final answer A"
  3) last (A) / [A]
  4) last bare letter A-E
  5) unique option-text match
  else: unparseable (counted wrong; never random.choice)

Open answers: last \\boxed{...} content via parse_open_response, else official parsed_pred.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lmms_eval.tasks.mmmu.utils import eval_open, parse_open_response


def get_text(o: dict) -> str:
    r = o.get("filtered_resps") or o.get("resps")
    while isinstance(r, list):
        r = r[0] if r else ""
    return str(r or "")


def parse_options(options) -> List[str]:
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except Exception:
            options = []
    return list(options) if options else []


def last_boxed_content(text: str) -> Optional[Tuple[str, str]]:
    """Return (content, kind) for the last \\boxed{...}, kind in {boxed, unclosed_boxed}."""
    last = None
    for m in re.finditer(r"\\boxed\{", text):
        rest = text[m.end() :]
        cm = re.match(r"([^}]*)\}", rest)
        if cm:
            last = (cm.group(1).strip(), "boxed")
        else:
            content = re.split(r"[\n\\]", rest, maxsplit=1)[0].strip()
            content = re.sub(r"[\[\]]+$", "", content).strip()
            last = (content, "unclosed_boxed")
    return last


def extract_mcq(response: str, all_choices: List[str], index2ans: Dict[str, str]) -> Tuple[Optional[str], str]:
    text = str(response)

    boxed = last_boxed_content(text)
    if boxed:
        content, kind = boxed
        m = re.search(r"\b([A-E])\b", content) or re.fullmatch(r"[\(\[]?([A-E])[\)\]]?", content)
        if m and m.group(1) in all_choices:
            return m.group(1), f"{kind}_letter"
        bl = content.lower().strip()
        for idx, ans in index2ans.items():
            if ans and ans.lower().strip() == bl:
                return idx, f"{kind}_text"

    for pat in [
        r"(?i)correct answer is\s*[:：]?\s*[\(\[]?([A-E])[\)\]]?",
        r"(?i)(?:the\s+)?answer is\s*[:：]?\s*[\(\[]?([A-E])[\)\]]?",
        r"(?i)final answer\s*[:：]?\s*[\(\[]?([A-E])[\)\]]?",
    ]:
        ms = list(re.finditer(pat, text))
        if ms and ms[-1].group(1) in all_choices:
            return ms[-1].group(1), "answer_is"

    ms = list(re.finditer(r"[\(\[]([A-E])[\)\]]", text))
    if ms and ms[-1].group(1) in all_choices:
        return ms[-1].group(1), "paren"

    ms = list(re.finditer(r"(?:^|[^A-Za-z])([A-E])(?:[^A-Za-z]|$)", text))
    if ms and ms[-1].group(1) in all_choices:
        return ms[-1].group(1), "letter"

    # option text only when no letter signal
    if not re.search(r"[\(\[][A-E][\)\]]", text) and not re.search(
        r"(?:^|[^A-Za-z])([A-E])(?:[^A-Za-z]|$)", text
    ):
        hits = []
        tl = text.lower()
        for idx, ans in index2ans.items():
            a = str(ans).strip()
            if len(a) >= 2 and a.lower() in tl:
                hits.append(idx)
        if len(hits) == 1:
            return hits[0], "option_text"
        if len(hits) > 1:
            last_idx, last_pos = None, -1
            for idx in hits:
                pos = tl.rfind(str(index2ans[idx]).lower())
                if pos > last_pos:
                    last_pos, last_idx = pos, idx
            return last_idx, "option_text_multi"

    return None, "none"


def official_pred(o: dict) -> Any:
    p = o.get("mmmu_acc", {}).get("parsed_pred")
    if isinstance(p, list):
        return p[0] if p else None
    return p


def as_open_pred_list(pred: Any) -> list:
    """eval_open expects a list of already-normalized preds."""
    if pred is None:
        return []
    if isinstance(pred, list):
        # already a list of norms, or a list of raw strings
        if pred and all(isinstance(x, (str, int, float)) for x in pred):
            # if looks like parse_open_response output (str/float mix), use as-is
            if any(isinstance(x, float) for x in pred) or all(isinstance(x, str) for x in pred):
                return pred
        # nested: flatten one level of parse results
        out = []
        for p in pred:
            out.extend(as_open_pred_list(p))
        return out
    if isinstance(pred, (int, float)):
        return [pred]
    return parse_open_response(str(pred))


def score_open(gt: str, text: str, official: Any) -> Tuple[bool, Any, str]:
    boxed = last_boxed_content(text)
    if boxed:
        content, kind = boxed
        parsed = parse_open_response(content)
        ok = bool(eval_open(gt, parsed))
        return ok, parsed, f"{kind}_open"
    if official is None:
        return False, None, "open_none"
    pred_list = as_open_pred_list(official)
    ok = bool(eval_open(gt, pred_list)) if pred_list else False
    return ok, official, "open_official"


def score_file(path: Path) -> dict:
    n_all = n_mcq = n_open = 0
    off_all = box_all = off_mcq = box_mcq = off_open = box_open = 0
    methods: Counter = Counter()
    rows_out = []

    for line in path.open():
        if not line.strip():
            continue
        o = json.loads(line)
        doc = o["doc"]
        gt = str(doc["answer"]).strip()
        qtype = doc.get("question_type", "multiple-choice")
        text = get_text(o)
        official = official_pred(o)
        n_all += 1

        if qtype != "multiple-choice":
            n_open += 1
            o_list = as_open_pred_list(official)
            o_ok = bool(eval_open(gt, o_list)) if o_list else False
            b_ok, pred, method = score_open(gt, text, official)
            off_all += o_ok
            box_all += b_ok
            off_open += o_ok
            box_open += b_ok
            methods[method] += 1
            rows_out.append(
                {
                    "id": doc.get("id"),
                    "split_hint": path.name,
                    "question_type": qtype,
                    "answer": gt,
                    "official_pred": official,
                    "boxed_pred": pred,
                    "method": method,
                    "official_ok": o_ok,
                    "boxed_ok": b_ok,
                }
            )
            continue

        n_mcq += 1
        opts = parse_options(doc.get("options"))
        letters = [chr(ord("A") + i) for i in range(len(opts))] or list("ABCD")
        index2ans = {letters[i]: str(opts[i]) for i in range(len(letters))}
        pred, method = extract_mcq(text, letters, index2ans)
        o_ok = str(official) == gt if official is not None else False
        b_ok = pred == gt
        off_all += o_ok
        box_all += b_ok
        off_mcq += o_ok
        box_mcq += b_ok
        methods[method] += 1
        rows_out.append(
            {
                "id": doc.get("id"),
                "split_hint": path.name,
                "question_type": qtype,
                "answer": gt,
                "official_pred": official,
                "boxed_pred": pred,
                "method": method,
                "official_ok": o_ok,
                "boxed_ok": b_ok,
                "has_boxed": r"\boxed{" in text,
            }
        )

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0

    return {
        "path": str(path),
        "n_all": n_all,
        "n_mcq": n_mcq,
        "n_open": n_open,
        "official_all": pct(off_all, n_all),
        "boxed_all": pct(box_all, n_all),
        "official_mcq": pct(off_mcq, n_mcq),
        "boxed_mcq": pct(box_mcq, n_mcq),
        "official_open": pct(off_open, n_open),
        "boxed_open": pct(box_open, n_open),
        "counts": {
            "official_all": off_all,
            "boxed_all": box_all,
            "official_mcq": off_mcq,
            "boxed_mcq": box_mcq,
            "official_open": off_open,
            "boxed_open": box_open,
        },
        "methods": dict(methods),
        "rows": rows_out,
    }


def merge_splits(dev: dict, val: dict) -> dict:
    """Sample-weighted merge of two split summaries (without rows)."""

    def add(a, b, key):
        return a["counts"][key] + b["counts"][key]

    def addn(a, b, key):
        return a[key] + b[key]

    n_all = addn(dev, val, "n_all")
    n_mcq = addn(dev, val, "n_mcq")
    n_open = addn(dev, val, "n_open")
    methods = Counter(dev["methods"])
    methods.update(val["methods"])

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0

    return {
        "n_all": n_all,
        "n_mcq": n_mcq,
        "n_open": n_open,
        "official_all": pct(add(dev, val, "official_all"), n_all),
        "boxed_all": pct(add(dev, val, "boxed_all"), n_all),
        "official_mcq": pct(add(dev, val, "official_mcq"), n_mcq),
        "boxed_mcq": pct(add(dev, val, "boxed_mcq"), n_mcq),
        "official_open": pct(add(dev, val, "official_open"), n_open),
        "boxed_open": pct(add(dev, val, "boxed_open"), n_open),
        "delta_all_pp": round(
            pct(add(dev, val, "boxed_all"), n_all) - pct(add(dev, val, "official_all"), n_all), 2
        ),
        "methods": dict(methods),
    }


def discover_run(run_dir: Path) -> Dict[str, Path]:
    out = {}
    for split in ("dev", "val"):
        hits = sorted(run_dir.glob(f"*samples_mmmu_{split}*.jsonl"))
        if hits:
            out[split] = hits[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Directories containing samples_mmmu_{dev,val}*.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        default="/root/autodl-tmp/LaViDa/eval/logs/mmmu_boxed_first_rescore_20260727",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"runs": {}}
    print(
        f"{'run':48} {'split':5} {'n':>5} {'off_all':>8} {'box_all':>8} {'Δ':>7} {'off_mcq':>8} {'box_mcq':>8}"
    )
    for run in args.runs:
        run_dir = Path(run)
        label = run_dir.parent.name if run_dir.name.startswith("lavida") else run_dir.name
        # nicer label: .../<mode>/lavida-ckpts...
        if run_dir.name.startswith("lavida-ckpts"):
            label = f"{run_dir.parent.parent.name}/{run_dir.parent.name}"
        splits = discover_run(run_dir)
        if not splits:
            print(f"SKIP (no samples): {run_dir}")
            continue
        run_res = {}
        for split, path in splits.items():
            s = score_file(path)
            # drop heavy rows from console merge; keep on disk
            rows = s.pop("rows")
            run_res[split] = s
            delta = round(s["boxed_all"] - s["official_all"], 2)
            print(
                f"{label:48} {split:5} {s['n_all']:5} {s['official_all']:7.2f}% {s['boxed_all']:7.2f}% {delta:+6.2f} {s['official_mcq']:7.2f}% {s['boxed_mcq']:7.2f}%"
            )
            detail_path = out_dir / f"{label.replace('/', '__')}__{split}.jsonl"
            with detail_path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            s["detail"] = str(detail_path)
        if "dev" in run_res and "val" in run_res:
            merged = merge_splits(run_res["dev"], run_res["val"])
            run_res["weighted"] = merged
            print(
                f"{label:48} {'wtd':5} {merged['n_all']:5} {merged['official_all']:7.2f}% {merged['boxed_all']:7.2f}% {merged['delta_all_pp']:+6.2f} {merged['official_mcq']:7.2f}% {merged['boxed_mcq']:7.2f}%"
            )
        summary["runs"][label] = run_res

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # compact markdown
    lines = [
        "# MMMU boxed-first offline rescore",
        "",
        "Rules: last `\\boxed{}` (incl. unclosed) → answer-is → `(A)` → bare letter → option text; unparseable = wrong (no random).",
        "",
        "| Run | Split | n | Official all | Boxed all | Δ | Official MCQ | Boxed MCQ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, run_res in summary["runs"].items():
        for split in ("dev", "val", "weighted"):
            if split not in run_res:
                continue
            s = run_res[split]
            delta = s.get("delta_all_pp", round(s["boxed_all"] - s["official_all"], 2))
            lines.append(
                f"| {label} | {split} | {s['n_all']} | {s['official_all']:.2f}% | {s['boxed_all']:.2f}% | {delta:+.2f} | {s['official_mcq']:.2f}% | {s['boxed_mcq']:.2f}% |"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
