#!/usr/bin/env python3
"""Single-image LaViDa-LLaDA smoke for SWD, PSP, VRG, and PSP+VRG."""

from __future__ import annotations

import argparse
import copy
import time
from typing import Optional

import torch
from PIL import Image, ImageDraw

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def build_prompt(tokenizer) -> str:
    conversation = copy.deepcopy(conv_templates["llada"])
    conversation.tokenizer = tokenizer
    conversation.append_message(
        conversation.roles[0],
        DEFAULT_IMAGE_TOKEN + "\nDescribe the image briefly.",
    )
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def load_image(path: Optional[str]) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    image = Image.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 64, 336, 320), fill=(40, 110, 210))
    draw.ellipse((128, 112, 256, 240), fill=(245, 190, 45))
    return image


def mode_kwargs(mode: str, max_new_tokens: int) -> dict:
    common = {
        "max_new_tokens": max_new_tokens,
        "block_length": max_new_tokens,
        "step_per_block": max_new_tokens,
        "temperature": 0.0,
    }
    if mode == "original":
        return {**common, "prefix_lm": True}
    if mode == "swd":
        return {
            **common,
            "decode_strategy": "swd",
            "prefix_lm": True,
            "thinking__swd_lambda": 5.0,
        }
    if mode == "psp":
        return {
            **common,
            "decode_strategy": "psp",
            "prefix_lm": True,
            "thinking__psp_gamma": 0.5,
        }
    if mode == "vrg":
        return {
            **common,
            "decode_strategy": "vrg",
            "prefix_lm": True,
            "thinking__vrg_scale": 0.5,
        }
    if mode == "psp_vrg":
        return {
            **common,
            "decode_strategy": "psp_vrg",
            "prefix_lm": True,
            "thinking__psp_gamma": 0.5,
            "thinking__vrg_scale": 0.5,
        }
    raise ValueError(f"Unknown mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", default="lavida-ckpts/lavida-llada-hd-reason"
    )
    parser.add_argument(
        "--vision-tower",
        default="/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384",
    )
    parser.add_argument("--image")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--modes", default="swd,psp,vrg,psp_vrg")
    args = parser.parse_args()

    vision_kwargs = {
        "mm_vision_tower": args.vision_tower,
        "mm_resampler_type": None,
        "mm_projector_type": "mlp2x_gelu",
        "mm_hidden_size": 1152,
        "use_mm_proj": True,
    }
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        None,
        "llava_llada",
        device_map="cuda:0",
        vision_kwargs=vision_kwargs,
        torch_dtype="bfloat16",
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)

    image = load_image(args.image)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [
        tensor.to(dtype=torch.bfloat16, device="cuda")
        for tensor in image_tensor
    ]
    input_ids = tokenizer_image_token(
        build_prompt(tokenizer),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to("cuda")

    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        kwargs = mode_kwargs(mode, args.max_new_tokens)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            tokenizer=tokenizer,
            **kwargs,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = output[0] if isinstance(output, tuple) else output
        text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        if "<|mdm_mask|>" in text:
            raise RuntimeError(f"Unresolved mask token in {mode} output")
        print(
            f"mode={mode} latency={elapsed:.3f}s "
            f"token_ids={tokens[0].tolist()} output={text!r}"
        )


if __name__ == "__main__":
    main()
