"""lmms-eval adapter and official pipeline metrics for local VLind-Bench."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def infer_true_or_false(response: str) -> str:
    """Use the official VLind token-level True/False extractor."""
    words = (
        str(response)
        .lower()
        .replace("\n", " ")
        .replace(",", "")
        .replace(".", "")
        .split(" ")
    )
    for word in words:
        if word == "true":
            return "True"
        if word == "false":
            return "False"
    return "NA"


def _target(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    normalized = str(value).strip().lower()
    if normalized == "true":
        return "True"
    if normalized == "false":
        return "False"
    raise ValueError(f"Invalid VLind target: {value!r}")


def _identifier(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def vlind_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    image_path = Path(str(doc["image_path"]))
    if image_path.is_dir():
        image_path = image_path / "0.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"VLind image not found: {image_path}")
    with Image.open(image_path) as image:
        return [image.convert("RGB")]


def vlind_doc_to_text(doc: dict[str, Any]) -> str:
    return str(doc["question"]).strip()


def vlind_doc_to_target(doc: dict[str, Any]) -> str:
    return _target(doc["answer"])


def vlind_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, dict[str, Any]]:
    prediction = str(results[0])
    extracted = infer_true_or_false(prediction)
    target = _target(doc["answer"])
    record = {
        "index": _identifier(doc["index"]),
        "global_id": _identifier(doc["global_id"]),
        "concept": str(doc["concept"]),
        "query_key": str(doc["query_key"]),
        "image_id": _identifier(doc.get("image_id")),
        "prediction": prediction,
        "extracted": extracted,
        "answer": target,
        "hit": int(extracted == target),
    }
    return {
        "vlind_query_acc": record,
        "vlind_ck_acc": record,
        "vlind_vp_acc": record,
        "vlind_cb_acc": record,
        "vlind_lp_raw_macro": record,
        "vlind_lp_acc": record,
    }


def score_vlind_pipeline(results: list[dict[str, Any]]) -> dict[str, float]:
    concept2num_instance: defaultdict[str, int] = defaultdict(int)
    concept2num_image: defaultdict[str, int] = defaultdict(int)
    concept2scores: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["global_id"])].append(row)

    for group in grouped.values():
        by_key: dict[str, Any] = {}
        for row in group:
            key = str(row["query_key"])
            prediction = str(row["extracted"])
            if key in {"d1", "d2"}:
                by_key.setdefault(key, {})[str(row["image_id"])] = prediction
            else:
                by_key[key] = prediction

        concept = str(group[0]["concept"])
        good_images = sorted(by_key.get("d1", {}).keys())
        if not good_images:
            continue

        concept2num_instance[concept] += 1
        concept2num_instance["total"] += 1
        concept2num_image[concept] += len(good_images)
        concept2num_image["total"] += len(good_images)

        a_pass = int(by_key.get("a1") == "True" and by_key.get("a2") == "False")
        b_pass = int(by_key.get("b1") == "True" and by_key.get("b2") == "False")
        c_raw = int(by_key.get("c1") == "True" and by_key.get("c2") == "False")
        c_pass = int(c_raw and a_pass)
        bc_pass = int(b_pass and c_pass)

        d_flags = []
        d_pass_flags = []
        for image_id in good_images:
            d_ok = int(
                by_key.get("d1", {}).get(image_id) == "True"
                and by_key.get("d2", {}).get(image_id) == "False"
            )
            d_flags.append(d_ok)
            d_pass_flags.append(int(d_ok and bc_pass))
        d_macro = sum(d_flags) / len(good_images)
        d_pass_macro = sum(d_pass_flags) / len(good_images)

        for bucket in (concept, "total"):
            scores = concept2scores[bucket]
            scores["a"] += a_pass
            scores["b"] += b_pass
            scores["c"] += c_raw
            scores["d_macro"] += d_macro
            scores["a_pass"] += a_pass
            scores["c_pass"] += c_pass
            scores["bc_pass"] += bc_pass
            scores["d_pass_macro"] += d_pass_macro

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    total = concept2scores.get("total", {})
    instance_count = concept2num_instance.get("total", 0)
    query_accuracy = (
        100.0 * sum(int(row["hit"]) for row in results) / len(results)
        if results
        else 0.0
    )
    return {
        "query_acc": query_accuracy,
        "CK_acc": 100.0 * ratio(total.get("a", 0.0), instance_count),
        "VP_acc": 100.0 * ratio(total.get("b", 0.0), instance_count),
        "CB_acc": 100.0 * ratio(total.get("c", 0.0), instance_count),
        "LP_raw_macro": 100.0
        * ratio(total.get("d_macro", 0.0), instance_count),
        "LP_acc": 100.0
        * ratio(total.get("d_pass_macro", 0.0), total.get("bc_pass", 0.0)),
    }


def vlind_aggregate_query_acc(results: list[dict[str, Any]]) -> float:
    return score_vlind_pipeline(results)["query_acc"]


def vlind_aggregate_ck_acc(results: list[dict[str, Any]]) -> float:
    return score_vlind_pipeline(results)["CK_acc"]


def vlind_aggregate_vp_acc(results: list[dict[str, Any]]) -> float:
    return score_vlind_pipeline(results)["VP_acc"]


def vlind_aggregate_cb_acc(results: list[dict[str, Any]]) -> float:
    return score_vlind_pipeline(results)["CB_acc"]


def vlind_aggregate_lp_raw_macro(results: list[dict[str, Any]]) -> float:
    return score_vlind_pipeline(results)["LP_raw_macro"]


def vlind_aggregate_lp_acc(results: list[dict[str, Any]]) -> float:
    return score_vlind_pipeline(results)["LP_acc"]
