"""
Convert HuggingFace parquet datasets (lmms-lab/POPE, lmms-lab/MME,
lmms-lab/MMBench_EN) into VLMEval-style TSVs stored at ~/LMUData/<name>.tsv.

Why this script exists:
- The stock VLMEvalKit downloads TSVs from opencompass.openxlab.space, which
  is unreachable from this workspace. HuggingFace, however, is reachable, so
  we mirror the parquet payloads from lmms-lab and re-encode them as the
  base64-image TSV format that VLMEval loaders expect.

Target TSV schema (per VLMEval ImageBaseDataset.dump_image, image_yorn.py,
image_mcq.py):
- Required for all:   index, question, image (base64 JPEG string, no header)
- Y/N tasks (POPE, MME): + answer (Yes/No), image_path, category
- MCQ tasks (MMBench):   + answer (A/B/C/D...), A, B, C, D, [E], category,
                          [l2-category], [hint]

Run:
    python hf_to_vlmeval_tsv.py --dataset POPE   --parquet /tmp/hfd/pope_random.parquet
    python hf_to_vlmeval_tsv.py --dataset MME    --parquet /tmp/hfd/mme_test_0.parquet
    python hf_to_vlmeval_tsv.py --dataset MMBench_DEV_EN --parquet /tmp/hfd/mmbench_en_dev.parquet
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


# -----------------------------------------------------------------------------
# Image encoding
# -----------------------------------------------------------------------------

def _bytes_to_base64_jpeg(image_bytes: bytes, quality: int = 90) -> str:
    """Encode raw image bytes (any format Pillow can read) to base64 JPEG."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_image_bytes(row_image) -> bytes:
    """Extract raw image bytes from a parquet 'image' cell (dict or bytes)."""
    if isinstance(row_image, dict) and "bytes" in row_image and row_image["bytes"] is not None:
        return row_image["bytes"]
    if isinstance(row_image, (bytes, bytearray)):
        return bytes(row_image)
    raise ValueError(f"unsupported image cell type: {type(row_image)}")


# -----------------------------------------------------------------------------
# Per-dataset converters
# -----------------------------------------------------------------------------

def convert_pope(df: pd.DataFrame) -> pd.DataFrame:
    """POPE parquet -> VLMEval TSV.

    HF cols: id, question_id, question, answer(yes/no lower), image_source,
             image({bytes,path}), category(random|popular|adversarial)
    """
    rows = []
    for i, row in df.iterrows():
        img_b64 = _bytes_to_base64_jpeg(_extract_image_bytes(row["image"]))
        # POPE_rating expects answer as 'Yes'/'No' (title-case).
        ans_raw = str(row["answer"]).strip().lower()
        answer = "Yes" if ans_raw.startswith("y") else "No"
        # image_path uniquely identifies each COCO image -> use image_source.
        image_path = str(row.get("image_source") or f"pope_{i:06d}") + ".jpg"
        rows.append({
            "index": int(i),
            "question": str(row["question"]),
            "answer": answer,
            "image": img_b64,
            "image_path": image_path,
            "category": str(row["category"]),
        })
    return pd.DataFrame(rows)


def convert_mme(df: pd.DataFrame) -> pd.DataFrame:
    """MME parquet -> VLMEval TSV.

    HF cols: question_id (e.g. 'code_reasoning/0020.png'), image({bytes,path}),
             question, answer(Yes/No), category (14 fine-grained cats)
    """
    rows = []
    for i, row in df.iterrows():
        img_b64 = _bytes_to_base64_jpeg(_extract_image_bytes(row["image"]))
        # question_id is already 'category/filename.png' -> perfect image_path.
        image_path = str(row["question_id"]).replace("/", "_")
        rows.append({
            "index": int(i),
            "question": str(row["question"]),
            "answer": str(row["answer"]).strip().title(),  # -> Yes/No
            "image": img_b64,
            "image_path": image_path,
            "category": str(row["category"]),
        })
    return pd.DataFrame(rows)


def _norm_cell(v) -> str:
    """Treat pandas NaN and the literal string 'nan' as empty."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def convert_mmbench(df: pd.DataFrame) -> pd.DataFrame:
    """MMBench_EN parquet -> VLMEval TSV.

    HF cols (lmms-lab/MMBench_EN): index, question, hint, A, B, C, D, answer,
    category, image({bytes,path}), source, l2-category, comment, split.
    """
    have_hint = "hint" in df.columns
    have_l2 = "l2-category" in df.columns or "l2_category" in df.columns
    have_split = "split" in df.columns
    letters = [L for L in list("ABCDE") if L in df.columns]

    rows = []
    for i, row in df.iterrows():
        img_b64 = _bytes_to_base64_jpeg(_extract_image_bytes(row["image"]))
        rec = {
            "index": int(row.get("index", i)) if pd.notna(row.get("index", None)) else int(i),
            "question": _norm_cell(row["question"]),
            "image": img_b64,
            "answer": _norm_cell(row["answer"]).upper(),
        }
        for L in letters:
            rec[L] = _norm_cell(row[L])
        rec["category"] = _norm_cell(row["category"]) if "category" in df.columns else ""
        if have_l2:
            key = "l2-category" if "l2-category" in df.columns else "l2_category"
            rec["l2-category"] = _norm_cell(row[key])
        if have_hint:
            rec["hint"] = _norm_cell(row["hint"])
        if have_split:
            rec["split"] = _norm_cell(row["split"])
        rows.append(rec)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

_CONVERTERS = {
    "POPE": convert_pope,
    "MME": convert_mme,
    "MMBench_DEV_EN": convert_mmbench,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(_CONVERTERS))
    ap.add_argument("--parquet", required=True, nargs="+",
                    help="one or more parquet files to concatenate")
    default_out = os.environ.get(
        "LMUData",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "LMUData",
        ),
    )
    ap.add_argument("--out-dir", default=default_out)
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only convert first N rows (for smoke tests)")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="if >0, shuffle rows with this seed so category-"
                         "stratified subsampling (MMADA_INDICES=0..N-1) yields a "
                         "diverse subset instead of a single-category block")
    args = ap.parse_args()

    frames = []
    for p in args.parquet:
        print(f"[read] {p}")
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)

    if args.shuffle_seed:
        print(f"[shuffle] seed={args.shuffle_seed}, n_rows={len(df)}")
        df = df.sample(frac=1.0, random_state=args.shuffle_seed).reset_index(drop=True)

    if args.limit and args.limit < len(df):
        df = df.head(args.limit).reset_index(drop=True)

    print(f"[convert] {args.dataset}: {len(df)} rows")
    out = _CONVERTERS[args.dataset](df)

    # Rewrite index to be a contiguous 0..N-1 after shuffle so the downstream
    # `MMADA_INDICES=0-N-1` env selects the first N shuffled rows.
    out = out.reset_index(drop=True)
    out["index"] = out.index.astype(int)

    os.makedirs(args.out_dir, exist_ok=True)
    tsv_path = os.path.join(args.out_dir, f"{args.dataset}.tsv")
    print(f"[write] {tsv_path} ({len(out)} rows, cols={list(out.columns)})")
    out.to_csv(tsv_path, sep="\t", index=False)

    size_mb = os.path.getsize(tsv_path) / (1024 ** 2)
    print(f"[done] wrote {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
