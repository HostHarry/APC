#!/usr/bin/env python3
"""Convert the official VLind-Bench release into an lmms-eval TSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ROOT = Path(
    "/root/autodl-tmp/datasets/VLind-Bench/VLind-Bench Dataset"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/MMaDA_DCD_cloud_bundle_20260711/MMaDA/"
    "evaluation/VLMEvalKit/LMUData/VLind-Bench.tsv"
)


def image_prompt(statement: str, prompt_type: str) -> str:
    if prompt_type == "common_sense":
        return (
            f"Statement: {statement}\nBased on common sense, is the given "
            "statement true or false? Only respond in True or False."
        )
    if prompt_type == "simple":
        return (
            f"Statement: {statement}\nBased on the image, is the given "
            "statement true or false? Only respond in True or False."
        )
    if prompt_type == "detailed":
        return (
            f"Statement: {statement}\nBased on the image, is the given "
            "statement true or false? Forget real-world common sense and "
            "just follow the information provided in the image. Only respond "
            "in True or False."
        )
    raise ValueError(f"Unknown prompt type: {prompt_type}")


def context_prompt(context: str, statement: str) -> str:
    return (
        f"Context: {context}\nStatement: {statement}\nBased on the context, "
        "is the given statement true or false? Forget real-world common "
        "sense and just follow the information provided in the context. "
        "Only respond in True or False."
    )


def good_images(record: dict[str, Any], vote_threshold: int) -> list[str]:
    return [
        str(image_id)
        for image_id, votes in record[
            "aggregated_human_label_good_images"
        ].items()
        if int(votes) >= vote_threshold
    ]


def rows_for_record(
    record: dict[str, Any],
    dataset_root: Path,
    vote_threshold: int,
) -> Iterable[dict[str, Any]]:
    concept = str(record["concept"])
    global_id = int(record["global_id"])
    counterfactual_dir = (
        dataset_root
        / "images"
        / "counterfactual"
        / concept
        / f"{record['context_id']}_{record['context']}"
    )
    factual_image = (
        dataset_root
        / "images"
        / "factual"
        / concept
        / f"{record['context_id']}_{record['factual_context']}"
        / "0.jpg"
    )
    counterfactual_image = (
        counterfactual_dir / f"{record['best_img_id']}.jpg"
    )

    fixed_queries = [
        (
            "a1",
            factual_image,
            image_prompt(record["false_statement"], "common_sense"),
            "True",
        ),
        (
            "a2",
            factual_image,
            image_prompt(record["true_statement"], "common_sense"),
            "False",
        ),
        (
            "b1",
            counterfactual_image,
            image_prompt(
                f"There is {record['existent_noun']} in the given image.",
                "simple",
            ),
            "True",
        ),
        (
            "b2",
            counterfactual_image,
            image_prompt(
                f"There is {record['non-existent_noun']} in the given image.",
                "simple",
            ),
            "False",
        ),
        (
            "c1",
            counterfactual_image,
            context_prompt(record["context"], record["true_statement"]),
            "True",
        ),
        (
            "c2",
            counterfactual_image,
            context_prompt(record["context"], record["false_statement"]),
            "False",
        ),
    ]
    for query_key, image_path, question, answer in fixed_queries:
        yield {
            "global_id": global_id,
            "concept": concept,
            "query_key": query_key,
            "image_id": "",
            "image_path": str(image_path),
            "question": question,
            "answer": answer,
        }

    for image_id in good_images(record, vote_threshold):
        image_path = counterfactual_dir / f"{image_id}.jpg"
        yield {
            "global_id": global_id,
            "concept": concept,
            "query_key": "d1",
            "image_id": image_id,
            "image_path": str(image_path),
            "question": image_prompt(record["true_statement"], "detailed"),
            "answer": "True",
        }
        yield {
            "global_id": global_id,
            "concept": concept,
            "query_key": "d2",
            "image_id": image_id,
            "image_path": str(image_path),
            "question": image_prompt(record["false_statement"], "detailed"),
            "answer": "False",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vote-threshold", type=int, default=2)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping records with missing release images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(
        (args.dataset_root / "data.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    skipped_records: list[tuple[int, list[str]]] = []
    for record in records:
        record_rows = list(
            rows_for_record(record, args.dataset_root, args.vote_threshold)
        )
        missing_images = sorted(
            {
                row["image_path"]
                for row in record_rows
                if not Path(row["image_path"]).is_file()
            }
        )
        if missing_images:
            skipped_records.append((int(record["global_id"]), missing_images))
            continue
        for row in record_rows:
            row["index"] = len(rows)
            rows.append(row)

    if skipped_records and args.strict:
        examples = "\n".join(
            path
            for _, missing in skipped_records[:10]
            for path in missing[:1]
        )
        raise FileNotFoundError(
            f"{len(skipped_records)} VLind records reference missing release "
            f"images, including:\n{examples}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "global_id",
        "concept",
        "query_key",
        "image_id",
        "image_path",
        "question",
        "answer",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)} queries from "
        f"{len(records) - len(skipped_records)}/{len(records)} complete "
        f"VLind instances to {args.output}; skipped "
        f"{len(skipped_records)} records with missing release images"
    )


if __name__ == "__main__":
    main()
