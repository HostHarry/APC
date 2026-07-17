"""Byte-parity comparison for E0 sanity vs plain DCD baseline.

After the ``defer_only/dispatcher.py`` fix (softmax(fp64) for base_conf),
defer_only + causal_lambda=0 should produce BIT-IDENTICAL LLaVABench
predictions to plain DCD (``dcd_decode_text_dual_cache``).

This script diffs two run directories (e.g. B0_plain_dcd vs E0_defer_l0)
and reports:
    - #rows exactly equal
    - #rows differing (with first diff shown)
    - if scores are available, mean-score delta

Usage:
    python scripts/compare_e0_parity.py \
        --base ./outputs/cvdcd_sweep/<run>/B0_plain_dcd \
        --alt  ./outputs/cvdcd_sweep/<run>/E0_defer_l0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def find_pred_xlsx(root: Path) -> Path:
    """Return the *_LLaVABench.xlsx file under a run directory."""
    hits = sorted(root.rglob("*_LLaVABench.xlsx"))
    hits = [h for h in hits if "openai_result" not in h.name]
    if not hits:
        raise FileNotFoundError(
            f"no *_LLaVABench.xlsx (non-openai) under {root}"
        )
    return hits[0]


def find_score_csv(root: Path):
    hits = sorted(root.rglob("*_LLaVABench_score.csv"))
    if not hits:
        return None
    return hits[0]


def load_preds(root: Path) -> pd.DataFrame:
    xlsx = find_pred_xlsx(root)
    df = pd.read_excel(xlsx)
    df = df.sort_values("index").reset_index(drop=True)
    return df, xlsx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline run directory (e.g. B0_plain_dcd)")
    ap.add_argument("--alt", required=True, help="alt run directory (e.g. E0_defer_l0)")
    ap.add_argument("--show-first-diffs", type=int, default=3,
                    help="how many diff samples to print (default: 3)")
    args = ap.parse_args()

    base_root = Path(args.base)
    alt_root = Path(args.alt)

    print(f"== E0-parity comparison ==")
    print(f"  base : {base_root}")
    print(f"  alt  : {alt_root}")
    print()

    base_df, base_xlsx = load_preds(base_root)
    alt_df, alt_xlsx = load_preds(alt_root)
    print(f"  base xlsx: {base_xlsx.name}")
    print(f"  alt  xlsx: {alt_xlsx.name}")

    if len(base_df) != len(alt_df):
        print(f"ROW-COUNT MISMATCH: base={len(base_df)}, alt={len(alt_df)}")
        return 2

    print(f"  n_rows   : {len(base_df)}")

    if not (base_df["index"].values == alt_df["index"].values).all():
        print("INDEX MISMATCH: sample indices differ")
        return 2

    diffs = []
    for i in range(len(base_df)):
        b_pred = str(base_df.iloc[i]["prediction"])
        a_pred = str(alt_df.iloc[i]["prediction"])
        if b_pred != a_pred:
            diffs.append((i, int(base_df.iloc[i]["index"]), b_pred, a_pred))

    n_diff = len(diffs)
    n_same = len(base_df) - n_diff
    print()
    print(f"  identical: {n_same}/{len(base_df)}")
    print(f"  differing: {n_diff}/{len(base_df)}")

    if n_diff == 0:
        print()
        print("PARITY HOLDS: E0 == plain DCD byte-for-byte.")
    else:
        print()
        print(f"PARITY BROKEN: {n_diff} rows differ. First {min(args.show_first_diffs, n_diff)}:")
        for row_i, sample_idx, b_pred, a_pred in diffs[: args.show_first_diffs]:
            print(f"  --- sample idx={sample_idx} (row {row_i}) ---")
            print(f"      base: {b_pred[:200]!r}...")
            print(f"      alt : {a_pred[:200]!r}...")

    base_score = find_score_csv(base_root)
    alt_score = find_score_csv(alt_root)
    if base_score and alt_score:
        print()
        print("== Score CSVs ==")
        b = pd.read_csv(base_score)
        a = pd.read_csv(alt_score)
        merged = b.merge(a, on=b.columns[0], suffixes=("_base", "_alt"))
        print(merged.to_string(index=False))

    return 0 if n_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
