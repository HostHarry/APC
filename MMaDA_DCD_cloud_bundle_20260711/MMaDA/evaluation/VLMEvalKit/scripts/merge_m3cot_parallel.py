#!/usr/bin/env python
"""Merge two M3CoT inference shards and run one unified evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vlmeval.dataset import build_dataset


def _normalize_index(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith('.0') else text


def _find_prediction(worker_root: Path, model: str, dataset: str) -> Path:
    name = f'{model}_{dataset}.xlsx'
    candidates = list(worker_root.rglob(name))
    if not candidates:
        raise FileNotFoundError(
            f'No {name} found below {worker_root}.')
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--expected', type=int, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    requested_indices = []
    frames = []
    for worker in range(2):
        index_file = run_root / 'indices' / f'worker{worker}.csv'
        requested_indices.extend(
            pd.read_csv(index_file)['index'].map(_normalize_index).tolist())

        prediction_file = _find_prediction(
            run_root / f'worker{worker}', args.model, args.dataset)
        frame = pd.read_excel(prediction_file)
        if 'index' not in frame or 'prediction' not in frame:
            raise ValueError(
                f'{prediction_file} lacks index or prediction columns.')
        frames.append(frame)
        print(
            f'[merge] worker{worker}: {len(frame)} rows from '
            f'{prediction_file}')

    merged = pd.concat(frames, ignore_index=True)
    normalized = merged['index'].map(_normalize_index)
    if normalized.duplicated().any():
        duplicates = normalized[normalized.duplicated()].unique().tolist()
        raise ValueError(f'Duplicate prediction indices: {duplicates[:10]}')

    predicted_indices = set(normalized)
    requested_set = set(requested_indices)
    missing = requested_set - predicted_indices
    unexpected = predicted_indices - requested_set
    if missing or unexpected:
        raise ValueError(
            f'Shard mismatch: missing={sorted(missing)[:10]}, '
            f'unexpected={sorted(unexpected)[:10]}')
    if len(merged) != args.expected:
        raise ValueError(
            f'Expected {args.expected} merged rows, got {len(merged)}.')

    order = {
        index: position
        for position, index in enumerate(requested_indices)
    }
    merged['_parallel_order'] = normalized.map(order)
    merged = (
        merged.sort_values('_parallel_order')
        .drop(columns=['_parallel_order'])
        .reset_index(drop=True)
    )

    merged_dir = run_root / 'merged'
    merged_dir.mkdir(parents=True, exist_ok=True)
    output_file = merged_dir / f'{args.model}_{args.dataset}.xlsx'
    merged.to_excel(output_file, index=False)
    print(f'[merge] wrote {len(merged)} rows to {output_file}')

    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise RuntimeError(f'Failed to build dataset {args.dataset}.')
    dataset.evaluate(str(output_file))
    print(f'[merge] unified M3CoT evaluation completed: {output_file}')


if __name__ == '__main__':
    main()
