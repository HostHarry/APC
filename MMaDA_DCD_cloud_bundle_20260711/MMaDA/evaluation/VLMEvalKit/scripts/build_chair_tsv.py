#!/usr/bin/env python
"""Build a VLMEval-compatible TSV for the CHAIR hallucination benchmark.

Supports either a sampled MSCOCO val2017 benchmark or the canonical full
COCO2014 Karpathy Test split (5,000 images), and emits
``$LMUData/CHAIR.tsv`` with columns:

    index, image, image_path, image_id, question, answer

- ``image``     : base64 JPEG (VLMEval standard), omitted with
                  ``--path-only``.
- ``image_path``: source filename, or an absolute source path with
                  ``--path-only``.
- ``image_id``  : the COCO image_id (needed downstream by the scorer for GT
                  lookup — kept separate from ``index`` so subsampling is
                  reversible).
- ``question``  : the CHAIR-standard "please describe" prompt.
- ``answer``    : JSON-encoded list of GT object canonical names (for
                  human-readability; the scorer re-reads the JSON annotations
                  as source of truth).

Usage
-----
    # Sampled COCO val2017
    python build_chair_tsv.py \
        --coco-root /home/user/大模型/LLava/data/coco \
        --n 200 --shuffle-seed 42

    # Canonical full CHAIR protocol: COCO2014 Karpathy Test, all 5,000 images
    python build_chair_tsv.py \
        --coco-root /path/to/coco2014_karpathy \
        --karpathy-split /path/to/karpathy_test.parquet \
        --all --path-only

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


def _encode_jpeg_b64(source, max_side: int = 512, quality: int = 90) -> str:
    """Load a path or HF image struct and return base64 image data.

    ``max_side=0`` preserves the source bytes exactly. This is preferred for
    the canonical Karpathy Test protocol so model-specific preprocessing sees
    the original COCO JPEG.
    """
    if isinstance(source, dict):
        if source.get('bytes') is not None:
            source = io.BytesIO(source['bytes'])
        elif source.get('path'):
            source = source['path']
        else:
            raise ValueError('Image struct contains neither bytes nor path.')
    if max_side <= 0:
        if isinstance(source, io.BytesIO):
            raw = source.getvalue()
        else:
            with open(source, 'rb') as f:
                raw = f.read()
        return base64.b64encode(raw).decode('ascii')
    img = Image.open(source).convert('RGB')
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
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--coco-root',
                        help='Path to a COCO2017 root containing '
                             'annotations/instances_val2017.json and '
                             'images/val2017/*.jpg')
    source.add_argument('--parquet',
                        help='MM-Hallu/CHAIR parquet containing image, '
                             'image_id, and objects columns.')
    ap.add_argument(
        '--karpathy-split',
        help='Karpathy Test parquet containing filename, cocoid, filepath, '
             'and split columns. Requires --coco-root and selects the fixed '
             'COCO2014 test split in file order.')
    ap.add_argument('--n', type=int, default=200,
                    help='Number of images to sample (default: 200).')
    ap.add_argument('--all', action='store_true',
                    help='Use every source image (all 5,000 for Karpathy Test).')
    ap.add_argument('--shuffle-seed', type=int, default=42)
    ap.add_argument('--out-dir', default=None,
                    help='Output dir. Defaults to $LMUData or ./LMUData.')
    ap.add_argument('--max-side', type=int, default=512,
                    help='Max JPEG side (default: 512); 0 preserves originals.')
    ap.add_argument(
        '--path-only', action='store_true',
        help='Store absolute local image paths instead of base64 data. This '
             'preserves original bytes and keeps a full 5K TSV small.')
    args = ap.parse_args()

    if args.karpathy_split and not args.coco_root:
        ap.error('--karpathy-split requires --coco-root')
    if args.path_only and not args.coco_root:
        ap.error('--path-only requires --coco-root')
    if args.n <= 0:
        ap.error('--n must be positive')

    out_dir = args.out_dir or os.environ.get('LMUData', './LMUData')
    os.makedirs(out_dir, exist_ok=True)
    out_tsv = os.path.join(out_dir, 'CHAIR.tsv')

    image_to_cats: dict[int, set[str]] = defaultdict(set)
    image_by_id = {}
    file_by_id = {}
    if args.coco_root:
        if args.karpathy_split:
            instances_json = os.path.join(
                args.coco_root, 'annotations', 'instances_val2014.json')
            image_candidates = [
                os.path.join(
                    args.coco_root, 'images_mscoco_2014_5k_test'),
                os.path.join(args.coco_root, 'images', 'val2014'),
                os.path.join(args.coco_root, 'val2014'),
            ]
            img_root = next(
                (p for p in image_candidates if os.path.isdir(p)),
                image_candidates[0],
            )
        else:
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
        # Retain images without instance masks: canonical CHAIR also obtains
        # their GT objects from human captions.
        for image_id in file_by_id:
            image_to_cats[image_id]
        for ann in coco['annotations']:
            image_to_cats[ann['image_id']].add(
                cat_id_to_name[ann['category_id']])
        category_names = set(cat_id_to_name.values())
    else:
        print(f'[chair-tsv] loading {args.parquet}')
        df = pd.read_parquet(
            args.parquet, columns=['image', 'image_id', 'objects'])
        for row in df.itertuples(index=False):
            image_id = int(row.image_id)
            objects = json.loads(row.objects) if isinstance(
                row.objects, str) else list(row.objects)
            image_to_cats[image_id].update(objects)
            image_by_id[image_id] = row.image
            image_path = row.image.get('path') if isinstance(
                row.image, dict) else None
            file_by_id[image_id] = image_path or f'{image_id:012d}.jpg'
        category_names = set().union(*image_to_cats.values())

    for c in category_names:
        if c not in _COCO80_NAMES:
            raise ValueError(
                f"COCO category '{c}' not in CHAIR's 80-class list.")

    if args.karpathy_split:
        split_df = pd.read_parquet(args.karpathy_split)
        required = {'filename', 'cocoid', 'filepath', 'split'}
        missing = required - set(split_df.columns)
        if missing:
            raise ValueError(
                f'Karpathy split is missing columns: {sorted(missing)}')
        split_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)
        if (len(split_df) != 5000
                or split_df['cocoid'].nunique() != 5000
                or split_df['filename'].nunique() != 5000
                or set(split_df['filepath']) != {'val2014'}):
            raise ValueError(
                'Expected the canonical COCO2014 Karpathy Test split: '
                '5,000 unique val2014 images.')
        ordered_ids = [int(i) for i in split_df['cocoid']]
        missing_ids = set(ordered_ids) - set(image_to_cats)
        if missing_ids:
            raise ValueError(
                f'{len(missing_ids)} Karpathy image IDs are absent from '
                f'{instances_json}.')
        file_by_id.update({
            int(row.cocoid): row.filename
            for row in split_df.itertuples(index=False)
        })
        picked = ordered_ids if args.all else ordered_ids[:args.n]
        protocol = 'COCO2014 Karpathy Test'
    elif args.all:
        picked = list(image_to_cats)
        protocol = 'all source images'
    else:
        picked = _stratified_sample(
            image_to_cats, args.n, args.shuffle_seed)
        protocol = f'stratified sample (seed={args.shuffle_seed})'

    print(f'[chair-tsv] source images: {len(image_to_cats)}')
    covered_categories = set()
    for image_id in picked:
        covered_categories.update(image_to_cats[image_id])
    print(f'[chair-tsv] picked {len(picked)} image_ids '
          f'using {protocol} (covering {len(covered_categories)} '
          f'distinct categories)')

    rows = []
    for idx, img_id in enumerate(picked):
        fname = file_by_id[img_id]
        source = (os.path.join(img_root, fname) if args.coco_root
                  else image_by_id[img_id])
        row = {
            'index': idx,
            'image_path': os.path.abspath(source) if args.path_only else fname,
            'image_id': img_id,
            'question': PROMPT,
            'answer': json.dumps(sorted(image_to_cats[img_id])),
        }
        if not args.path_only:
            row['image'] = _encode_jpeg_b64(
                source, max_side=args.max_side)
        rows.append(row)

    columns = ['index']
    if not args.path_only:
        columns.append('image')
    columns.extend(['image_path', 'image_id', 'question', 'answer'])
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'[chair-tsv] wrote {out_tsv}  '
          f'rows={len(df)}  size={os.path.getsize(out_tsv) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
