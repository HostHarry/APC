#!/usr/bin/env python3
"""Diagnose OpenAI API connectivity for LLaVABench judging.

Runs the same probe that VLMEvalKit's `model.working()` uses,
plus dumps the raw HTTP response so we can see auth/URL/model errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VLMEVAL_ROOT = HERE.parent
sys.path.insert(0, str(VLMEVAL_ROOT))


def _mask(s: str, keep: int = 6) -> str:
    if not s:
        return "(unset)"
    if len(s) <= keep + 4:
        return s[:2] + "***" + s[-2:]
    return s[:keep] + "..." + s[-4:]


def dump_env():
    print("=" * 66)
    print("[env]")
    for k in ["OPENAI_API_KEY", "OPENAI_API_BASE",
              "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
              "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION",
              "BOYUE_API_KEY", "BOYUE_API_BASE",
              "OPENAI_ORGANIZATION"]:
        v = os.environ.get(k, "")
        if "KEY" in k or "SECRET" in k or "TOKEN" in k:
            v = _mask(v)
        print(f"  {k:<35s} = {v!r}")
    print()


def probe_direct(model_name: str) -> None:
    """Direct raw HTTPS probe using requests, bypasses VLMEvalKit."""
    import requests

    api_base = os.environ.get("OPENAI_API_BASE", "").strip()
    if not api_base:
        api_base = "https://api.openai.com/v1/chat/completions"
        print("[direct probe] OPENAI_API_BASE unset → default OpenAI official.")
    print(f"[direct probe] api_base = {api_base}")

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("[direct probe] ERROR: OPENAI_API_KEY is empty.")
        return
    print(f"[direct probe] key     = {_mask(key)}")

    if not api_base.rstrip("/").endswith("chat/completions"):
        print("[direct probe] WARN: api_base doesn't look like a chat completions endpoint.\n"
              "               VLMEvalKit expects the FULL URL, e.g. https://.../v1/chat/completions")

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say 'pong' if you receive this."}],
        "max_tokens": 8,
        "temperature": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    print(f"[direct probe] posting model={model_name} ...")
    try:
        r = requests.post(api_base, headers=headers,
                          data=json.dumps(payload), timeout=30)
    except Exception as exc:
        print(f"[direct probe] EXCEPTION: {type(exc).__name__}: {exc}")
        return

    print(f"[direct probe] HTTP {r.status_code}")
    text = r.text
    if len(text) > 1200:
        text = text[:1200] + "\n... (truncated)"
    print(text)
    print()


def probe_via_vlmeval(model_name: str) -> None:
    """Use VLMEvalKit's OpenAIWrapper (same code path as build_judge)."""
    import logging
    logging.getLogger("ChatAPI").setLevel(logging.WARNING)
    logging.getLogger("BaseAPI").setLevel(logging.WARNING)

    print("[vlmeval probe] importing OpenAIWrapper ...")
    try:
        from vlmeval.api import OpenAIWrapper
    except Exception as exc:
        print(f"[vlmeval probe] import ERROR: {type(exc).__name__}: {exc}")
        return

    print(f"[vlmeval probe] instantiating model={model_name} verbose=False ...")
    try:
        model = OpenAIWrapper(model_name, verbose=False, retry=1)
    except Exception as exc:
        print(f"[vlmeval probe] init ERROR: {type(exc).__name__}: {exc}")
        return

    msgs = [dict(type="text", value="Say 'pong' if you receive this.")]
    print("[vlmeval probe] calling generate_inner(...) ...")
    try:
        code, answer, resp = model.generate_inner(msgs)
    except Exception as exc:
        print(f"[vlmeval probe] EXCEPTION: {type(exc).__name__}: {exc}")
        return

    print(f"[vlmeval probe] return_code = {code}")
    print(f"[vlmeval probe] answer      = {answer!r}")
    if isinstance(resp, dict):
        print(f"[vlmeval probe] resp keys   = {list(resp.keys())}")
    print()

    try:
        is_ok = model.working()
    except Exception as exc:
        print(f"[vlmeval probe] model.working() EXCEPTION: {type(exc).__name__}: {exc}")
        return
    print(f"[vlmeval probe] model.working() = {is_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt-4o-mini",
                    help="Judge model name to test (e.g. gpt-4o-mini, gpt-4o).")
    ap.add_argument("--skip-direct", action="store_true",
                    help="Skip raw requests probe, only use VLMEvalKit wrapper.")
    args = ap.parse_args()

    dump_env()
    if not args.skip_direct:
        probe_direct(args.model)
    probe_via_vlmeval(args.model)


if __name__ == "__main__":
    main()
