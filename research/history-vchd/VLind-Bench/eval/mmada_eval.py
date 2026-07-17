"""Evaluate local MMaDA decoding strategies on a selected VLind subset.

This runner intentionally mirrors VLind's A/B/C/D prompt protocol while
keeping the evaluation set explicit through ``--global-ids``.  It also writes
one recoverable JSON output after each context, so interrupted long runs can
be resumed without discarding completed model calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = (
    Path("/root/autodl-tmp/datasets/VLind-Bench/VLind-Bench Dataset/data.json")
)
DEFAULT_COUNTERFACTUAL_IMAGES = Path(
    "/root/autodl-tmp/datasets/VLind-Bench/VLind-Bench Dataset/images/"
    "counterfactual"
)
DEFAULT_FACTUAL_IMAGES = Path(
    "/root/autodl-tmp/datasets/VLind-Bench/VLind-Bench Dataset/images/factual"
)
DEFAULT_MMADA_ROOT = Path(
    "/root/autodl-tmp/MMaDA_DCD_cloud_bundle_20260711/MMaDA"
)
DEFAULT_MODEL_PATH = Path("/root/autodl-tmp/MMaDA-8B-MixCoT")
DEFAULT_VQ_MODEL_PATH = Path("/root/autodl-tmp/magvitv2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local MMaDA decoding on VLind-Bench."
    )
    parser.add_argument(
        "--strategy",
        choices=("original", "dcd", "cv_dcd", "vchd", "vchd_fixed"),
        default="vchd",
    )
    parser.add_argument(
        "--vchd-profile",
        choices=(
            "plain",
            "history",
            "ccd_history",
            "adaptive_temporal",
            "ccaw",
            "history_ccaw",
            "counterfactual_current",
            "counterfactual_uniform",
            "counterfactual_exposure",
            "unified_trajectory",
            "unified_trajectory_history_only",
            "unified_trajectory_fixed_visual",
        ),
        default="plain",
    )
    parser.add_argument("--model-identifier", required=True)
    parser.add_argument("--global-ids", default="")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--counterfactual-image-dir",
        type=Path,
        default=DEFAULT_COUNTERFACTUAL_IMAGES,
    )
    parser.add_argument(
        "--factual-image-dir", type=Path, default=DEFAULT_FACTUAL_IMAGES
    )
    parser.add_argument("--mmada-root", type=Path, default=DEFAULT_MMADA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--tokenizer-path", type=Path, default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "--vq-model-path", type=Path, default=DEFAULT_VQ_MODEL_PATH
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cache-type", default="none")
    parser.add_argument("--vote-thres", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")

    parser.add_argument("--vchd-alpha", type=float, default=0.5)
    parser.add_argument("--vchd-beta", type=float, default=0.1)
    parser.add_argument("--vchd-tau-base", type=float, default=0.10)
    parser.add_argument("--vchd-tau-contrast", type=float, default=0.90)
    parser.add_argument("--vchd-mask-capacity", type=int, default=16)
    parser.add_argument("--vchd-max-commit", type=int, default=16)
    parser.add_argument("--vchd-cache-type", default="none")
    parser.add_argument("--vchd-save-reports", action="store_true")
    parser.add_argument("--vchd-collect-trace", action="store_true")
    parser.add_argument("--vchd-report-dir", type=Path)

    parser.add_argument("--vchd-history-top-v", type=int, default=8)
    parser.add_argument("--vchd-history-ema-decay", type=float, default=0.7)
    parser.add_argument("--vchd-history-penalty-scale", type=float, default=1.0)
    parser.add_argument(
        "--vchd-history-anchor-min-consistent", type=int, default=0
    )
    parser.add_argument("--vchd-ccd-history-length", type=int, default=2)
    parser.add_argument("--vchd-ccd-top-v-positions", type=int, default=64)
    parser.add_argument(
        "--vchd-adaptive-temporal-loglogistic-scale",
        type=float,
        default=3.20,
    )
    parser.add_argument(
        "--vchd-adaptive-temporal-loglogistic-shape",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--vchd-adaptive-temporal-loglogistic-offset",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vchd-adaptive-temporal-tail-mix-max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vchd-adaptive-temporal-exposure-scale", type=float, default=0.10
    )
    parser.add_argument(
        "--vchd-adaptive-temporal-relevance-scale", type=float, default=0.01
    )
    parser.add_argument(
        "--vchd-adaptive-temporal-conflict-scale",
        type=float,
        default=0.002,
    )
    parser.add_argument("--vchd-ccaw-mode", default="legacy")
    parser.add_argument("--vchd-ccaw-block-size", type=int, default=32)
    parser.add_argument("--vchd-ccaw-min-commit", type=int, default=1)
    parser.add_argument("--vchd-ccaw-qualified-budget", type=int, default=1)
    parser.add_argument("--vchd-ccaw-max-capacity", type=int, default=64)
    parser.add_argument("--vchd-ccaw-pressure-decay", type=float, default=0.8)
    parser.add_argument("--vchd-ccaw-expand-step", type=int, default=8)
    parser.add_argument("--vchd-ccaw-shrink-step", type=int, default=4)

    parser.add_argument(
        "--vchd-counterfactual-exposure-window-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--vchd-counterfactual-exposure-distance-scale",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--vchd-counterfactual-exposure-text-exposure-floor",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--vchd-counterfactual-exposure-positive-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--vchd-counterfactual-exposure-negative-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--vchd-counterfactual-exposure-min-effective-exposure",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vchd-counterfactual-exposure-neutral-tau-contrast",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--vchd-counterfactual-exposure-lower-bound-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vchd-counterfactual-exposure-flip-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-top-k",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-window-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-semantic-std-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-gain-uncertainty-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-visual-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-relevance-scale",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-observation-scale",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-exposure-scale",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-uncertainty-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-stale-decay",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-history-limit",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--vchd-unified-trajectory-opposed-threshold",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def infer_true_or_false(response: str) -> str:
    for word in (
        response.lower()
        .replace("\n", " ")
        .replace(",", "")
        .replace(".", "")
        .split(" ")
    ):
        if word == "true":
            return "True"
        if word == "false":
            return "False"
    return "NA"


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
    raise ValueError(f"Unknown image prompt type: {prompt_type}")


def context_prompt(context: str, statement: str) -> str:
    return (
        f"Context: {context}\nStatement: {statement}\nBased on the context, "
        "is the given statement true or false? Forget real-world common "
        "sense and just follow the information provided in the context. "
        "Only respond in True or False."
    )


def profile_settings(args: argparse.Namespace) -> dict[str, Any]:
    profile = args.vchd_profile
    counterfactual_mode = {
        "counterfactual_current": "current",
        "counterfactual_uniform": "uniform",
        "counterfactual_exposure": "exposure",
    }.get(profile, "off")
    unified_profiles = {
        "unified_trajectory",
        "unified_trajectory_history_only",
        "unified_trajectory_fixed_visual",
    }
    return {
        "history_enabled": profile in {"history", "history_ccaw"},
        "ccd_history_enabled": profile == "ccd_history",
        "adaptive_temporal_enabled": profile == "adaptive_temporal",
        "ccaw_enabled": profile in {"ccaw", "history_ccaw"},
        "counterfactual_exposure_mode": counterfactual_mode,
        "unified_trajectory_enabled": profile in unified_profiles,
        "unified_trajectory_adaptive_visual_relevance": (
            profile != "unified_trajectory_fixed_visual"
        ),
        "unified_trajectory_visual_weight": (
            0.0
            if profile == "unified_trajectory_history_only"
            else args.vchd_unified_trajectory_visual_weight
        ),
    }


def configure_report_environment(args: argparse.Namespace) -> None:
    report_dir = args.vchd_report_dir
    if report_dir is None and (
        args.vchd_save_reports or args.vchd_collect_trace
    ):
        report_dir = ROOT / "outputs" / f"{args.model_identifier}_reports"
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MMADA_VCHD_REPORT_DIR"] = str(report_dir)
    else:
        os.environ.pop("MMADA_VCHD_REPORT_DIR", None)
    os.environ["MMADA_VCHD_COLLECT_TRACE"] = (
        "1" if args.vchd_collect_trace else "0"
    )
    os.environ["MMADA_VCHD_RETURN_REPORT"] = (
        "1" if (args.vchd_save_reports or args.vchd_collect_trace) else "0"
    )


def load_selected_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.global_ids.strip():
        raise ValueError("--global-ids is required for this runner")
    requested_ids = [
        int(value.strip())
        for value in args.global_ids.split(",")
        if value.strip()
    ]
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("--global-ids contains duplicates")
    source = json.loads(args.data_path.read_text(encoding="utf-8"))
    by_id = {int(item["global_id"]): item for item in source}
    missing = [global_id for global_id in requested_ids if global_id not in by_id]
    if missing:
        raise ValueError(f"Unknown global IDs: {missing}")
    return [copy.deepcopy(by_id[global_id]) for global_id in requested_ids]


def atomic_write(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def good_images(instance: dict[str, Any], vote_threshold: int) -> list[str]:
    return [
        str(image_id)
        for image_id, votes in instance[
            "aggregated_human_label_good_images"
        ].items()
        if int(votes) >= vote_threshold
    ]


def expected_prediction_keys(
    identifier: str, instance: dict[str, Any], vote_threshold: int
) -> list[str]:
    prefix = f"{identifier}_pred"
    keys = [
        f"{prefix}_a1_image",
        f"{prefix}_a2_image",
        f"{prefix}_b1",
        f"{prefix}_b2",
        f"{prefix}_c1_image",
        f"{prefix}_c2_image",
    ]
    for image_id in good_images(instance, vote_threshold):
        keys.extend((f"{prefix}_d1:{image_id}", f"{prefix}_d2:{image_id}"))
    return keys


def is_complete(
    identifier: str, instance: dict[str, Any], vote_threshold: int
) -> bool:
    prefix = f"{identifier}_pred"
    for key in expected_prediction_keys(identifier, instance, vote_threshold):
        if ":" in key:
            group, image_id = key.split(":", 1)
            if image_id not in instance.get(group, {}):
                return False
        elif key not in instance:
            return False
    return True


def prediction_message(image: Path, prompt: str) -> list[dict[str, str]]:
    return [
        {"type": "text", "value": prompt},
        {"type": "image", "value": str(image)},
    ]


def run_prediction(
    model: Any,
    *,
    image: Path,
    prompt: str,
    report_tag: str,
) -> dict[str, str]:
    if not image.is_file():
        raise FileNotFoundError(f"VLind image does not exist: {image}")
    os.environ["MMADA_CURRENT_INDEX"] = report_tag
    response = model.generate_inner(prediction_message(image, prompt))
    return {"response": response, "answer": infer_true_or_false(response)}


def evaluate_instance(
    model: Any, args: argparse.Namespace, instance: dict[str, Any]
) -> None:
    identifier = args.model_identifier
    prefix = f"{identifier}_pred"
    global_id = int(instance["global_id"])
    counterfactual_dir = (
        args.counterfactual_image_dir
        / instance["concept"]
        / f"{instance['context_id']}_{instance['context']}"
    )
    factual_dir = (
        args.factual_image_dir
        / instance["concept"]
        / f"{instance['context_id']}_{instance['factual_context']}"
    )
    factual_image = factual_dir / "0.jpg"
    counterfactual_image = counterfactual_dir / f"{instance['best_img_id']}.jpg"

    tasks: list[tuple[str, Path, str]] = [
        (
            f"{prefix}_a1_image",
            factual_image,
            image_prompt(instance["false_statement"], "common_sense"),
        ),
        (
            f"{prefix}_a2_image",
            factual_image,
            image_prompt(instance["true_statement"], "common_sense"),
        ),
        (
            f"{prefix}_b1",
            counterfactual_image,
            image_prompt(
                f"There is {instance['existent_noun']} in the given image.",
                "simple",
            ),
        ),
        (
            f"{prefix}_b2",
            counterfactual_image,
            image_prompt(
                f"There is {instance['non-existent_noun']} in the given image.",
                "simple",
            ),
        ),
        (
            f"{prefix}_c1_image",
            counterfactual_image,
            context_prompt(instance["context"], instance["true_statement"]),
        ),
        (
            f"{prefix}_c2_image",
            counterfactual_image,
            context_prompt(instance["context"], instance["false_statement"]),
        ),
    ]
    for key, image, prompt in tasks:
        instance[key] = run_prediction(
            model,
            image=image,
            prompt=prompt,
            report_tag=f"{global_id}_{key}",
        )

    instance[f"{prefix}_d1"] = {}
    instance[f"{prefix}_d2"] = {}
    for image_id in good_images(instance, args.vote_thres):
        image = counterfactual_dir / f"{image_id}.jpg"
        instance[f"{prefix}_d1"][image_id] = run_prediction(
            model,
            image=image,
            prompt=image_prompt(instance["true_statement"], "detailed"),
            report_tag=f"{global_id}_{prefix}_d1_{image_id}",
        )
        instance[f"{prefix}_d2"][image_id] = run_prediction(
            model,
            image=image,
            prompt=image_prompt(instance["false_statement"], "detailed"),
            report_tag=f"{global_id}_{prefix}_d2_{image_id}",
        )


def import_mmada(mmada_root: Path) -> type[Any]:
    vlmeval_root = mmada_root / "evaluation" / "VLMEvalKit"
    if not vlmeval_root.is_dir():
        raise FileNotFoundError(f"MMaDA evaluation root not found: {vlmeval_root}")
    sys.path.insert(0, str(vlmeval_root))
    from vlmeval.vlm.mmada.mmada import MMaDA

    return MMaDA


def instantiate_model(args: argparse.Namespace) -> Any:
    MMaDA = import_mmada(args.mmada_root)
    settings = profile_settings(args)
    configure_report_environment(args)
    os.environ["MMADA_DECODE_STRATEGY"] = args.strategy
    return MMaDA(
        model_path=str(args.model_path),
        tokenizer_path=str(args.tokenizer_path),
        vq_model_path=str(args.vq_model_path),
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_length=args.block_length,
        resolution=args.resolution,
        temperature=args.temperature,
        decode_strategy=args.strategy,
        cache_type=args.cache_type,
        vchd_alpha=args.vchd_alpha,
        vchd_beta=args.vchd_beta,
        vchd_tau_base=args.vchd_tau_base,
        vchd_tau_contrast=args.vchd_tau_contrast,
        vchd_mask_capacity=args.vchd_mask_capacity,
        vchd_max_commit=args.vchd_max_commit,
        vchd_cache_type=args.vchd_cache_type,
        vchd_history_enabled=settings["history_enabled"],
        vchd_history_top_v=args.vchd_history_top_v,
        vchd_history_ema_decay=args.vchd_history_ema_decay,
        vchd_history_penalty_scale=args.vchd_history_penalty_scale,
        vchd_history_anchor_min_consistent=(
            args.vchd_history_anchor_min_consistent
        ),
        vchd_ccd_history_enabled=settings["ccd_history_enabled"],
        vchd_ccd_history_length=args.vchd_ccd_history_length,
        vchd_ccd_top_v_positions=args.vchd_ccd_top_v_positions,
        vchd_adaptive_temporal_enabled=(
            settings["adaptive_temporal_enabled"]
        ),
        vchd_adaptive_temporal_loglogistic_scale=(
            args.vchd_adaptive_temporal_loglogistic_scale
        ),
        vchd_adaptive_temporal_loglogistic_shape=(
            args.vchd_adaptive_temporal_loglogistic_shape
        ),
        vchd_adaptive_temporal_loglogistic_offset=(
            args.vchd_adaptive_temporal_loglogistic_offset
        ),
        vchd_adaptive_temporal_tail_mix_max=(
            args.vchd_adaptive_temporal_tail_mix_max
        ),
        vchd_adaptive_temporal_exposure_scale=(
            args.vchd_adaptive_temporal_exposure_scale
        ),
        vchd_adaptive_temporal_relevance_scale=(
            args.vchd_adaptive_temporal_relevance_scale
        ),
        vchd_adaptive_temporal_conflict_scale=(
            args.vchd_adaptive_temporal_conflict_scale
        ),
        vchd_unified_trajectory_enabled=(
            settings["unified_trajectory_enabled"]
        ),
        vchd_unified_trajectory_top_k=(
            args.vchd_unified_trajectory_top_k
        ),
        vchd_unified_trajectory_window_size=(
            args.vchd_unified_trajectory_window_size
        ),
        vchd_unified_trajectory_semantic_std_scale=(
            args.vchd_unified_trajectory_semantic_std_scale
        ),
        vchd_unified_trajectory_gain_uncertainty_scale=(
            args.vchd_unified_trajectory_gain_uncertainty_scale
        ),
        vchd_unified_trajectory_visual_weight=(
            settings["unified_trajectory_visual_weight"]
        ),
        vchd_unified_trajectory_adaptive_visual_relevance=(
            settings["unified_trajectory_adaptive_visual_relevance"]
        ),
        vchd_unified_trajectory_relevance_scale=(
            args.vchd_unified_trajectory_relevance_scale
        ),
        vchd_unified_trajectory_observation_scale=(
            args.vchd_unified_trajectory_observation_scale
        ),
        vchd_unified_trajectory_exposure_scale=(
            args.vchd_unified_trajectory_exposure_scale
        ),
        vchd_unified_trajectory_uncertainty_scale=(
            args.vchd_unified_trajectory_uncertainty_scale
        ),
        vchd_unified_trajectory_stale_decay=(
            args.vchd_unified_trajectory_stale_decay
        ),
        vchd_unified_trajectory_history_limit=(
            args.vchd_unified_trajectory_history_limit
        ),
        vchd_unified_trajectory_opposed_threshold=(
            args.vchd_unified_trajectory_opposed_threshold
        ),
        vchd_ccaw_enabled=settings["ccaw_enabled"],
        vchd_ccaw_mode=args.vchd_ccaw_mode,
        vchd_ccaw_block_size=args.vchd_ccaw_block_size,
        vchd_ccaw_min_commit=args.vchd_ccaw_min_commit,
        vchd_ccaw_qualified_budget=args.vchd_ccaw_qualified_budget,
        vchd_ccaw_max_capacity=args.vchd_ccaw_max_capacity,
        vchd_ccaw_pressure_decay=args.vchd_ccaw_pressure_decay,
        vchd_ccaw_expand_step=args.vchd_ccaw_expand_step,
        vchd_ccaw_shrink_step=args.vchd_ccaw_shrink_step,
        vchd_counterfactual_exposure_mode=(
            settings["counterfactual_exposure_mode"]
        ),
        vchd_counterfactual_exposure_window_size=(
            args.vchd_counterfactual_exposure_window_size
        ),
        vchd_counterfactual_exposure_distance_scale=(
            args.vchd_counterfactual_exposure_distance_scale
        ),
        vchd_counterfactual_exposure_text_exposure_floor=(
            args.vchd_counterfactual_exposure_text_exposure_floor
        ),
        vchd_counterfactual_exposure_positive_threshold=(
            args.vchd_counterfactual_exposure_positive_threshold
        ),
        vchd_counterfactual_exposure_negative_threshold=(
            args.vchd_counterfactual_exposure_negative_threshold
        ),
        vchd_counterfactual_exposure_min_effective_exposure=(
            args.vchd_counterfactual_exposure_min_effective_exposure
        ),
        vchd_counterfactual_exposure_neutral_tau_contrast=(
            args.vchd_counterfactual_exposure_neutral_tau_contrast
        ),
        vchd_counterfactual_exposure_lower_bound_scale=(
            args.vchd_counterfactual_exposure_lower_bound_scale
        ),
        vchd_counterfactual_exposure_flip_decay=(
            args.vchd_counterfactual_exposure_flip_decay
        ),
    )


def merge_resume_records(
    selected: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if not args.resume or not args.output_path.is_file():
        return selected
    previous = json.loads(args.output_path.read_text(encoding="utf-8"))
    previous_by_id = {int(item["global_id"]): item for item in previous}
    return [
        copy.deepcopy(previous_by_id.get(int(item["global_id"]), item))
        for item in selected
    ]


def main() -> None:
    args = parse_args()
    if args.output_path is None:
        args.output_path = (
            ROOT / "outputs" / f"data_{args.model_identifier}.json"
        )
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be combined")
    selected = load_selected_records(args)
    if args.validate_only:
        print(
            f"Validated {len(selected)} VLind contexts for "
            f"{args.vchd_profile}."
        )
        return

    records = merge_resume_records(selected, args)
    model = instantiate_model(args)
    total = len(records)
    start = time.perf_counter()
    for index, record in enumerate(records, 1):
        if is_complete(args.model_identifier, record, args.vote_thres):
            print(
                f"[{index}/{total}] global_id={record['global_id']} already complete"
            )
            continue
        print(f"[{index}/{total}] evaluating global_id={record['global_id']}")
        evaluate_instance(model, args, record)
        atomic_write(args.output_path, records)
        elapsed = time.perf_counter() - start
        print(
            f"[{index}/{total}] saved {args.output_path.name} "
            f"after {elapsed:.1f}s"
        )

    atomic_write(args.output_path, records)
    print(f"Completed {total} VLind contexts: {args.output_path}")


if __name__ == "__main__":
    main()
