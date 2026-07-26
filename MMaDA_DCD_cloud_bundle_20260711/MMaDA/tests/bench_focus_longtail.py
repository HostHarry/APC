"""Micro-benchmark for ``observe_focus_longtail``.

Runs a realistic-scale call in a tight loop and reports the mean per-call
latency. Compatible with both the pre-P0 and post-P0 ``history.py`` because
it only uses the public ``FocusFrame(positions=, distributions=)`` two-field
constructor — this lets the same script benchmark both versions without any
API dependency mismatch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def _add_repo_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=134656)
    parser.add_argument("--focus-capacity", type=int, default=64)
    parser.add_argument("--history-depth", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--frame-mode",
        choices=("raw", "cached"),
        default="raw",
        help="raw: pre-P0-compatible two-field FocusFrame. cached: P0-only "
        "make_focus_frame (pre-caches lookup, models the real decoder path).",
    )
    args = parser.parse_args()

    _add_repo_to_path()
    from decoding.history import FocusFrame, observe_focus_longtail

    try:
        from decoding.history import make_focus_frame as _make_frame_cached
    except ImportError:
        _make_frame_cached = None

    device = torch.device(args.device)
    torch.manual_seed(0)

    contrast = torch.rand(args.positions, args.vocab, device=device)
    contrast = contrast / contrast.sum(-1, keepdim=True)
    visual = torch.rand(args.positions, args.vocab, device=device)
    visual = visual / visual.sum(-1, keepdim=True)
    confidence = torch.rand(args.positions, device=device)
    apc_mass = torch.rand(args.positions, device=device) * 0.5 + 0.5
    mask = torch.zeros(args.positions, dtype=torch.bool, device=device)
    mask[: args.positions * 5 // 8] = True
    exposure = torch.rand(args.positions, device=device)
    relevance = torch.rand(args.positions, device=device) * 0.8

    frames = []
    for _ in range(args.history_depth):
        scores = confidence * mask.float()
        top_idx = torch.topk(scores, k=args.focus_capacity).indices.sort().values
        dist = contrast[top_idx].clone()
        dist = dist / dist.sum(-1, keepdim=True)
        if args.frame_mode == "cached":
            if _make_frame_cached is None:
                raise SystemExit(
                    "cached frame_mode requires P0's make_focus_frame; "
                    "current history.py does not expose it."
                )
            frames.append(_make_frame_cached(top_idx, dist, args.positions))
        else:
            # Two-field ctor is compatible with pre-P0 FocusFrame.
            frames.append(FocusFrame(positions=top_idx, distributions=dist))

    kwargs = dict(
        contrast_distribution=contrast,
        visual_distribution=visual,
        position_confidence=confidence,
        apc_mass=apc_mass,
        mask=mask,
        exposure=exposure,
        visual_relevance=relevance,
        dwell_depth=2,
        focus_capacity=args.focus_capacity,
        kernel_scale=3.2,
        kernel_shape=8.0,
        kernel_offset=0.0,
        mix_ceiling=0.35,
        exposure_tau=1.0,
        relevance_tau=1.0,
        conflict_tau=1.0,
    )

    for _ in range(args.warmup):
        _ = observe_focus_longtail(frames, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure three separate windows to catch clock noise.
    latencies = []
    for _ in range(3):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = observe_focus_longtail(frames, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) / args.iters * 1000.0)

    latencies.sort()
    print(
        f"device={args.device}  iters={args.iters}  vocab={args.vocab}  "
        f"history_depth={args.history_depth}  frame_mode={args.frame_mode}"
    )
    print(f"per-call latency (ms): min={latencies[0]:.3f}  median={latencies[1]:.3f}  max={latencies[2]:.3f}")


if __name__ == "__main__":
    main()
