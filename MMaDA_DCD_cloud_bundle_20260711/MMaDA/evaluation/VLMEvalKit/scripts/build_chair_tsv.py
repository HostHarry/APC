#!/usr/bin/env python
"""Build a VLMEval-compatible TSV for the CHAIR hallucination benchmark.

Reads MSCOCO val2017 images + instance annotations from a local path, samples
N images (stratified so we cover as many object categories as possible) and
emits ``$LMUData/CHAIR.tsv`` with columns:

    index, image, image_path, image_id, question, answer

- ``image``     : base64 JPEG (VLMEval standard).
- ``image_id``  : the COCO image_id (needed downstream by the scorer for GT
                  lookup — kept separate from ``index`` so subsampling is
                  reversible).
- ``question``  : the CHAIR-standard "please describe" prompt.
- ``answer``    : JSON-encoded list of GT object canonical names (for
                  human-readability; the scorer re-reads the JSON annotations
                  as source of truth).

Usage
-----
    python build_chair_tsv.py \
        --coco-root /home/user/大模型/LLava/data/coco \
        --n 200 --shuffle-seed 42

If ``LMUData`` env var is set, the TSV is written to ``$LMUData/CHAIR.tsv``,
else to ``./LMUData/CHAIR.tsv``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
from collections import defaultdict

from PIL import Image
import pandas as pd

# Ensure we can import the VLMEval scorer for its category list, so we stay
# in sync with the 80-class canonical names.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..')))
from vlmeval.dataset.utils.chair import _COCO80_NAMES  # noqa: E402


PROMPT = "Please describe this image in detail."


def _encode_jpeg_b64(path: str, max_side: int = 512, quality: int = 90) -> str:
    """Load, optionally downsample, re-encode as JPEG, return base64 str."""
    img = Image.open(path).convert('RGB')
    if max(img.size) > max_side:
        r = max_side / max(img.size)
        img = img.resize((int(img.size[0] * r), int(img.size[1] * r)),
                         Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _stratified_sample(image_to_cats: dict[int, set[str]],
                       n: int, seed: int) -> list[int]:
    """Pick ``n`` image_ids such that we cover as many categories as possible.

    Simple round-robin: iterate categories, at each round pick one still-
    available image containing that category. Fill any residual slots by
    uniform random draws.
    """
    rng = random.Random(seed)
    remaining = dict(image_to_cats)
    cat_to_images: dict[str, list[int]] = defaultdict(list)
    for img_id, cats in remaining.items():
        for c in cats:
            cat_to_images[c].append(img_id)
    for lst in cat_to_images.values():
        rng.shuffle(lst)

    picked: list[int] = []
    seen: set[int] = set()
    cat_iter = list(cat_to_images.keys())
    rng.shuffle(cat_iter)

    while len(picked) < n:
        progressed = False
        for c in cat_iter:
            while cat_to_images[c] and cat_to_images[c][-1] in seen:
                cat_to_images[c].pop()
            if not cat_to_images[c]:
                continue
            img_id = cat_to_images[c].pop()
            if img_id in seen:
                continue
            picked.append(img_id)
            seen.add(img_id)
            progressed = True
            if len(picked) >= n:
                break
        if not progressed:
            break

    # Fill any residual slots with random draws from unseen images
    if len(picked) < n:
        pool = [i for i in remaining if i not in seen]
        rng.shuffle(pool)
        picked.extend(pool[: n - len(picked)])
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco-root', required=True,
                    help='Path to a COCO2017 root containing '
                         'annotations/instances_val2017.json and '
                         'images/val2017/*.jpg')
    ap.add_argument('--n', type=int, default=200,
                    help='Number of images to sample (default: 200).')
    ap.add_argument('--shuffle-seed', type=int, default=42)
    ap.add_argument('--out-dir', default=None,
                    help='Output dir. Defaults to $LMUData or ./LMUData.')
    ap.add_argument('--max-side', type=int, default=512,
                    help='Max side length for down-sampled JPEG (default: 512).')
    args = ap.parse_args()

    out_dir = args.out_dir or os.environ.get('LMUData', './LMUData')
    os.makedirs(out_dir, exist_ok=True)
    out_tsv = os.path.join(out_dir, 'CHAIR.tsv')

    instances_json = os.path.join(
        args.coco_root, 'annotations', 'instances_val2017.json')
    img_root = os.path.join(args.coco_root, 'images', 'val2017')
    if not os.path.exists(instances_json):
        raise SystemExit(f'Missing {instances_json}')
    if not os.path.isdir(img_root):
        raise SystemExit(f'Missing {img_root}')

    print(f'[chair-tsv] loading {instances_json}')
    with open(instances_json, 'r') as f:
        coco = json.load(f)
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    file_by_id = {im['id']: im['file_name'] for im in coco['images']}

    image_to_cats: dict[int, set[str]] = defaultdict(set)
    for ann in coco['annotations']:
        image_to_cats[ann['image_id']].add(cat_id_to_name[ann['category_id']])

    for c in cat_id_to_name.values():
        if c not in _COCO80_NAMES:
            raise ValueError(
                f"COCO category '{c}' not in CHAIR's 80-class list.")

    print(f'[chair-tsv] total val2017 images with annotations: '
          f'{len(image_to_cats)}')
    picked = _stratified_sample(image_to_cats, args.n, args.shuffle_seed)
    print(f'[chair-tsv] picked {len(picked)} image_ids '
          f'(covering {len(set().union(*(image_to_cats[i] for i in picked)))} '
          f'distinct categories)')

    rows = []
    for idx, img_id in enumerate(picked):
        fname = file_by_id[img_id]
        fpath = os.path.join(img_root, fname)
        img_b64 = _encode_jpeg_b64(fpath, max_side=args.max_side)
        rows.append({
            'index': idx,
            'image': img_b64,
            'image_path': fname,
            'image_id': img_id,
            'question': PROMPT,
            'answer': json.dumps(sorted(image_to_cats[img_id])),
        })

    df = pd.DataFrame(rows, columns=[
        'index', 'image', 'image_path', 'image_id', 'question', 'answer'
    ])
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'[chair-tsv] wrote {out_tsv}  '
          f'rows={len(df)}  size={os.path.getsize(out_tsv) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
