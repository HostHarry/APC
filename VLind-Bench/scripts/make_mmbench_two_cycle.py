"""Create a two-cycle MMBench TSV for stricter circular evaluation."""

from __future__ import annotations

import argparse
import string
from collections import Counter
from pathlib import Path

import pandas as pd


OFFSET = 1_000_000


def _is_abnormal(row: pd.Series, labels: list[str]) -> bool:
    choices = {label: str(row[label]) for label in labels}
    has_label = False
    for label, value in choices.items():
        normalized = value
        for char in set(value):
            if char not in string.ascii_letters and char != " ":
                normalized = normalized.replace(char, " ")
        hit_labels = {word for word in normalized.split() if word in choices}
        if len(hit_labels) > 1:
            return True
        if value in string.ascii_uppercase:
            has_label = True
    return has_label


def _labels(row: pd.Series) -> list[str]:
    labels = []
    for label in string.ascii_uppercase:
        if label not in row.index or pd.isna(row[label]):
            break
        labels.append(label)
    return labels


def build_two_cycle(source: Path, output: Path) -> None:
    data = pd.read_csv(source, sep="\t")
    if int(data["index"].max()) >= OFFSET:
        raise ValueError(f"Source indices must be below {OFFSET}")

    original = data.copy()
    original["g_index"] = original["index"].astype(int)
    rotated_rows = []
    counts = Counter()

    for _, row in data.iterrows():
        labels = _labels(row)
        answer = str(row["answer"])
        if len(labels) < 2 or answer not in labels or _is_abnormal(row, labels):
            counts["single_cycle_abnormal"] += 1
            continue

        rotated_labels = [labels[-1], *labels[:-1]]
        choice_map = dict(zip(labels, rotated_labels))
        rotated = row.copy()
        index = int(row["index"])
        rotated["index"] = index + OFFSET
        rotated["g_index"] = index
        rotated["image"] = str(index)
        rotated["answer"] = choice_map[answer]
        for source_label, target_label in choice_map.items():
            rotated[target_label] = row[source_label]
        rotated_rows.append(rotated)
        counts[f"{len(labels)}_choice_rotated"] += 1

    rotated = pd.DataFrame(rotated_rows)
    combined = pd.concat([original, rotated], ignore_index=True)
    combined["index"] = combined["index"].astype(int)
    combined["g_index"] = combined["g_index"].astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, sep="\t", index=False)

    print(f"source={len(original)} rotated={len(rotated)} total={len(combined)}")
    print(f"groups={combined['g_index'].nunique()} counts={dict(counts)}")
    print(f"output={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_two_cycle(args.source, args.output)


if __name__ == "__main__":
    main()
