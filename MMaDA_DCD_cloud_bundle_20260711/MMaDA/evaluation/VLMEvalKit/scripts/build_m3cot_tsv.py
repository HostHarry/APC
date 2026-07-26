#!/usr/bin/env python
"""Convert Hugging Face M3CoT parquet shards to a local VLMEval TSV.

The converter extracts the embedded images once and stores absolute image
paths in the TSV. By default only the official test split is exported.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import string
from pathlib import Path

import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = {
    'id', 'category', 'image_id', 'question', 'choices', 'context', 'answer',
    'rationale', 'split', 'image', 'domain', 'topic',
}


def _safe_filename(sample_id: str, source_name: str | None) -> str:
    safe_id = re.sub(r'[^A-Za-z0-9._-]+', '_', sample_id).strip('._')
    suffix = Path(source_name or '').suffix.lower()
    if suffix not in {'.jpg', '.jpeg', '.png', '.webp'}:
        suffix = '.png'
    return f'{safe_id}{suffix}'


def _extract_image(image_obj: dict, target: Path) -> None:
    raw = image_obj.get('bytes')
    if raw is None:
        source = image_obj.get('path')
        if not source or not os.path.isfile(source):
            raise ValueError('Image contains neither embedded bytes nor a valid path.')
        raw = Path(source).read_bytes()

    # Verify the encoded image before persisting it.
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()

    if not target.exists() or target.stat().st_size != len(raw):
        target.write_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--parquet-dir',
        default='/root/autodl-tmp/datasets/M3CoT/data',
        help='Directory containing M3CoT split parquet shards.',
    )
    parser.add_argument(
        '--split', default='test', choices=['train', 'validation', 'test'])
    parser.add_argument(
        '--out-dir',
        default=None,
        help='TSV output directory (defaults to $LMUData or ./LMUData).',
    )
    parser.add_argument(
        '--image-dir',
        default=None,
        help='Extracted image directory (defaults beside the parquet data).',
    )
    parser.add_argument('--dataset-name', default='M3CoT')
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir).resolve()
    shards = sorted(parquet_dir.glob(f'{args.split}-*.parquet'))
    if not shards:
        raise SystemExit(
            f'No {args.split} parquet shards found in {parquet_dir}.')

    frames = []
    for shard in shards:
        frame = pd.read_parquet(shard)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f'{shard.name} is missing columns: {sorted(missing)}')
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    if set(data['split']) != {args.split}:
        raise ValueError(
            f'Expected only split={args.split}, got {set(data["split"])}.')
    if data['id'].nunique() != len(data):
        raise ValueError('M3CoT sample IDs are not unique.')

    out_dir = Path(
        args.out_dir or os.environ.get('LMUData', './LMUData')).resolve()
    image_dir = Path(
        args.image_dir
        or parquet_dir.parent / 'images' / args.split
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    max_choices = max(len(choices) for choices in data['choices'])
    if max_choices > len(string.ascii_uppercase):
        raise ValueError(f'Too many answer choices: {max_choices}.')
    option_labels = list(string.ascii_uppercase[:max_choices])

    rows = []
    for index, sample in enumerate(data.itertuples(index=False)):
        choices = [str(choice) for choice in sample.choices]
        valid_labels = option_labels[:len(choices)]
        answer = str(sample.answer).strip().upper()
        if answer not in valid_labels:
            raise ValueError(
                f'Invalid answer {answer!r} for sample {sample.id!r}.')

        source_name = sample.image.get('path')
        image_path = image_dir / _safe_filename(
            str(sample.id), source_name)
        _extract_image(sample.image, image_path)

        row = {
            'index': index,
            'id': str(sample.id),
            'image_path': str(image_path),
            'image_id': str(sample.image_id),
            'question': str(sample.question),
            'context': '' if pd.isna(sample.context) else str(sample.context),
            'answer': answer,
            'domain': str(sample.domain),
            'topic': str(sample.topic),
            'category': str(sample.category),
            'split': str(sample.split),
            'choices': json.dumps(choices, ensure_ascii=False),
            'rationale': str(sample.rationale),
        }
        for label, choice in zip(valid_labels, choices):
            row[label] = choice
        rows.append(row)

    columns = [
        'index', 'id', 'image_path', 'image_id', 'question', 'context',
        *option_labels, 'answer', 'domain', 'topic', 'category', 'split',
        'choices', 'rationale',
    ]
    output = out_dir / f'{args.dataset_name}.tsv'
    pd.DataFrame(rows, columns=columns).to_csv(
        output, sep='\t', index=False)

    print(f'[m3cot-tsv] split={args.split} rows={len(rows)}')
    print(f'[m3cot-tsv] images={image_dir}')
    print(
        f'[m3cot-tsv] wrote {output} '
        f'({output.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
