#!/usr/bin/env python3
"""Single-image smoke for LaViDa-Dream original / VCHD / VCHD+CCAW."""
from __future__ import annotations

import argparse
import copy
import time

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def build_prompt(conv_template: str) -> str:
    question = DEFAULT_IMAGE_TOKEN + "\nDescribe the image briefly."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_once(model, tokenizer, input_ids, image_tensor, image_sizes, mode: str, max_new_tokens: int):
    common = dict(
        images=image_tensor,
        image_sizes=image_sizes,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
    )
    if mode == "original":
        kwargs = dict(
            temperature=0.0,
            top_p=0.95,
            alg="entropy",
            steps=max_new_tokens,
            **common,
        )
    elif mode == "vchd":
        kwargs = dict(
            decode_strategy="vchd",
            prefix_lm=False,
            vchd__ccaw_enabled=False,
            vchd__enable_g_gate=False,
            vchd__mask_capacity=16,
            vchd__tau_base=0.1,
            vchd__tau_contrast=0.9,
            vchd__return_report=True,
            **common,
        )
    elif mode == "vchd_ccaw":
        kwargs = dict(
            decode_strategy="vchd",
            prefix_lm=False,
            vchd__ccaw_enabled=True,
            vchd__ccaw_mode="inverse_window",
            vchd__enable_g_gate=False,
            vchd__mask_capacity=16,
            vchd__ccaw_max_mask_capacity=64,
            vchd__tau_base=0.1,
            vchd__tau_contrast=0.9,
            vchd__return_report=True,
            **common,
        )
    else:
        raise ValueError(mode)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    out = model.generate(input_ids, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.time() - t0
    sequences = out.sequences if hasattr(out, "sequences") else out
    text = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
    report = getattr(model, "_last_vchd_report", None)
    return text, dt, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="lavida-ckpts/lavida-dream-hd")
    parser.add_argument("--image", default="images/dog.png")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--modes",
        default="original,vchd,vchd_ccaw",
        help="comma-separated modes",
    )
    args = parser.parse_args()

    vision_kwargs = dict(
        mm_vision_tower="google/siglip-so400m-patch14-384",
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        use_mm_proj=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        None,
        "llava_dream",
        device_map="cuda:0",
        vision_kwargs=vision_kwargs,
        torch_dtype="bfloat16",
    )
    model.eval()
    model.to(torch.bfloat16)

    prompt = build_prompt("dream")
    image = Image.open(args.image).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device="cuda") for _image in image_tensor]
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")
    image_sizes = [image.size]

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        text, dt, report = run_once(
            model,
            tokenizer,
            input_ids,
            image_tensor,
            image_sizes,
            mode,
            args.max_new_tokens,
        )
        print("=" * 60)
        print(f"mode={mode} latency={dt:.3f}s")
        print(text.replace("\n", " ")[:500])
        if report is not None:
            print(
                "report:",
                {
                    k: report.get(k)
                    for k in (
                        "decoder",
                        "model_evaluations",
                        "threshold_commits",
                        "fallback_commits",
                        "ccaw_enabled",
                        "ccaw_mode",
                        "output_tokens",
                    )
                },
            )
            assert report.get("output_tokens", 0) >= 0
        assert "<|mdm_mask|>" not in text
        assert text.strip(), f"empty output for mode={mode}"


if __name__ == "__main__":
    main()
