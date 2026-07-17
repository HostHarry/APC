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


def convert_mmmu(df: pd.DataFrame) -> pd.DataFrame:
    """MMMU (lmms-lab/MMMU) parquet -> VLMEval TSV (MMMU_DEV_VAL flavour).

    HF cols: id (e.g. 'dev_Accounting_1'), question, options (list[str]),
             explanation, image_1..image_7 ({bytes,path} or None),
             img_type, answer (letter for MC, str for open),
             topic_difficulty, question_type ('multiple-choice'|'open'),
             subfield.

    Filter policy (documented in TSV as `notes` column):
    * multi-choice questions ONLY (drop 'open' - MMaDA wrapper is MCQ-tuned)
    * single-image questions ONLY (MMaDA wrapper uses first image only)
    * 4 options exactly (drop the ~2-option minority to keep MC scoring clean)

    Question text keeps `<image 1>` marker so downstream MMMU
    `split_MMMU` can insert the image at the right position; secondary
    `<image N>` markers are removed since we drop those images.
    """
    import ast
    import re
    IMG_COLS = [f"image_{i}" for i in range(1, 8)]

    def n_images(row) -> int:
        n = 0
        for c in IMG_COLS:
            v = row[c]
            if isinstance(v, dict) and v.get("bytes"):
                n += 1
        return n

    def normalize_options(x):
        # lmms-lab/MMMU stores `options` as a Python `repr` string, e.g.
        # "['$63,020', '$58,410', '$71,320', '$77,490']". Parse safely.
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except (ValueError, SyntaxError):
                pass
        return []

    df = df.copy()
    df["_n_imgs"] = df.apply(n_images, axis=1)
    df["_opts"] = df["options"].apply(normalize_options)
    df["_n_opts"] = df["_opts"].apply(len)

    n0 = len(df)
    df = df[df["question_type"] == "multiple-choice"]
    df = df[df["_n_imgs"] == 1]
    df = df[df["_n_opts"] == 4]
    print(f"[MMMU] {n0} -> {len(df)} after filters (single-image, MC, 4-option)")

    rows = []
    for _, row in df.iterrows():
        img_b64 = _bytes_to_base64_jpeg(row["image_1"]["bytes"])
        # Drop <image 2..7> markers, keep <image 1> for correct positioning.
        q = str(row["question"])
        q = re.sub(r"<image\s*[2-7]\s*>", "", q)
        opts = row["_opts"]
        answer_letter = str(row["answer"]).strip().upper()
        # Sanity: answer must be one of A/B/C/D
        if answer_letter not in ("A", "B", "C", "D"):
            continue
        # id looks like 'dev_Accounting_1' / 'validation_Math_3'
        rid = str(row["id"])
        if rid.startswith("dev_"):
            split_val = "dev"
        elif rid.startswith("validation_"):
            split_val = "validation"
        else:
            split_val = "unknown"
        rows.append({
            "index": 0,  # rewritten below
            "id": rid,
            "question": q,
            "image": img_b64,
            "A": _norm_cell(opts[0]),
            "B": _norm_cell(opts[1]),
            "C": _norm_cell(opts[2]),
            "D": _norm_cell(opts[3]),
            "answer": answer_letter,
            "category": _norm_cell(row.get("subfield", "")),
            "split": split_val,
        })
    return pd.DataFrame(rows)


def convert_mathvista(df: pd.DataFrame) -> pd.DataFrame:
    """MathVista (AI4Math/MathVista) parquet -> VLMEval TSV (MathVista_MINI).

    HF cols: pid, question, image (path str), decoded_image ({bytes,path}),
             choices (numpy array or None), unit, precision, answer,
             question_type ('multi_choice'|'free_form'),
             answer_type ('text'|'integer'|'float'|'list'),
             metadata (dict: task, category, context, grade, skills,
                              source, split, language, img_height, img_width),
             query (pre-formatted prompt with Hint + Choices).

    VLMEval's MathVista scorer needs at TSV level:
      question, answer, question_type, answer_type, answer_option (for MC),
      choices (str repr of list), unit, precision, task, skills.

    We use `query` (already has A/B/C/D format hint) as the prompt sent to
    the model.  For MC we compute `answer_option` by matching `answer`
    against the choice list.
    """
    import numpy as np

    rows = []
    for _, row in df.iterrows():
        img_dict = row["decoded_image"]
        if not (isinstance(img_dict, dict) and img_dict.get("bytes")):
            continue
        img_b64 = _bytes_to_base64_jpeg(img_dict["bytes"])

        qtype = str(row["question_type"])
        atype = str(row["answer_type"])
        answer = row["answer"]

        # Choices: numpy array of strings OR None.
        choices_raw = row["choices"]
        if choices_raw is None or (isinstance(choices_raw, float) and pd.isna(choices_raw)):
            choices_list = []
        elif isinstance(choices_raw, (list, tuple, np.ndarray)):
            choices_list = [str(x) for x in list(choices_raw)]
        else:
            choices_list = []

        # answer_option: letter for MC, empty for free_form.
        answer_option = ""
        if qtype == "multi_choice" and choices_list:
            for i, ch in enumerate(choices_list):
                if str(ch).strip() == str(answer).strip():
                    answer_option = chr(ord("A") + i)
                    break

        meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
        skills_val = meta.get("skills", [])
        # numpy arrays -> plain list
        if isinstance(skills_val, np.ndarray):
            skills_val = list(skills_val)
        skills_str = repr([str(s) for s in list(skills_val)]) if skills_val else "[]"

        rows.append({
            "index": 0,  # rewritten below
            "pid": int(row["pid"]) if row["pid"] is not None else -1,
            # Use `query` (pre-formatted prompt with format hint + choices)
            # as the actual question sent to the model. Falls back to raw
            # question if query missing.
            "question": _norm_cell(row.get("query") or row["question"]),
            "image": img_b64,
            "choices": repr(choices_list),
            "answer": _norm_cell(answer),
            "answer_option": answer_option,
            "question_type": qtype,
            "answer_type": atype,
            "unit": _norm_cell(row.get("unit")),
            "precision": _norm_cell(row.get("precision")),
            "task": _norm_cell(meta.get("task", "")),
            "category": _norm_cell(meta.get("category", "")),
            "grade": _norm_cell(meta.get("grade", "")),
            "source": _norm_cell(meta.get("source", "")),
            "context": _norm_cell(meta.get("context", "")),
            "skills": skills_str,
            "split": _norm_cell(meta.get("split", "testmini")),
            "language": _norm_cell(meta.get("language", "en")),
        })
    return pd.DataFrame(rows)


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
    "MMMU_DEV_VAL": convert_mmmu,
    "MathVista_MINI": convert_mathvista,
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
