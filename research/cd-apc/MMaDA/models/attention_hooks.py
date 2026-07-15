# coding=utf-8
"""Non-invasive attention collection for MMaDA / LLaDA blocks.

Replaces ``LLaDABlock._scaled_dot_product_attention`` at runtime with a manual
softmax(QK^T / sqrt(d)) implementation that additionally stashes the attention
weights into an ``AttentionBag``. Original methods are restored on exit.

The DCD decoding logic (KV cache, ``replace_position``, block-wise RoPE) is
untouched; only the kernel that computes attention scores is swapped, and the
output of the swap is mathematically equivalent to the fused
``F.scaled_dot_product_attention`` call.

Memory note: a single attention tensor for one layer/step has shape
``[B, H, q_len, k_len]``. For an 8B MMaDA with 32 heads at 4k context this is
roughly 1 GB in fp16 per layer per step, so by default we collect only a few
layers and downcast to fp16 / move to CPU immediately.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F


class AttentionBag:
    """Collects per-layer attention tensors and groups them by decode step.

    A "step" is closed when every layer in ``layers_to_collect`` has fired
    exactly once. By default each closed step is appended to ``self.steps`` as
    a tensor of shape ``[L, B, H, q, k]`` where ``L = len(layers_to_collect)``.

    Streaming mode (``stream_dir`` set): each closed step is immediately written
    to ``stream_dir/step_XXXX.pt`` (or via ``stream_save_fn``) and dropped from
    memory. ``self.steps`` then stays empty and ``self.num_steps`` tracks the
    number of saved steps so meta files can still report it.
    """

    def __init__(
        self,
        layers_to_collect: Sequence[int],
        stream_dir: Optional[str] = None,
        stream_save_fn: Optional[Callable[[int, torch.Tensor], None]] = None,
        save_steps: Optional[Sequence[int]] = None,
        save_last: bool = False,
        head_mean: bool = False,
    ):
        self.layers_to_collect = list(layers_to_collect)
        self.layer_pos = {l: i for i, l in enumerate(self.layers_to_collect)}
        self.steps: List[torch.Tensor] = []
        self._buf: Dict[int, torch.Tensor] = {}
        self.stream_dir = stream_dir
        self.stream_save_fn = stream_save_fn
        self.save_steps: Optional[Set[int]] = None if save_steps is None else {int(s) for s in save_steps}
        self.save_last = bool(save_last)
        self.head_mean = bool(head_mean)
        self._last_step_idx: Optional[int] = None
        self._last_step_tensor: Optional[torch.Tensor] = None
        self._saved_steps: Set[int] = set()
        self.num_steps = 0
        if self.stream_dir is not None:
            os.makedirs(self.stream_dir, exist_ok=True)

    def _save_step(self, step_idx: int, tensor: torch.Tensor) -> None:
        if self.stream_save_fn is not None:
            self.stream_save_fn(step_idx, tensor)
        elif self.stream_dir is not None:
            path = os.path.join(self.stream_dir, f"step_{step_idx:04d}.pt")
            torch.save(tensor, path)
        else:
            self.steps.append(tensor)
        self._saved_steps.add(step_idx)

    def _should_save_step(self, step_idx: int) -> bool:
        return self.save_steps is None or step_idx in self.save_steps

    def _emit_step(self, tensor: torch.Tensor) -> None:
        step_idx = self.num_steps
        if self.save_last:
            self._last_step_idx = step_idx
            self._last_step_tensor = tensor
        if self._should_save_step(step_idx):
            self._save_step(step_idx, tensor)
        self.num_steps += 1

    def finalize(self) -> None:
        """Save the final collected step when requested by ``save_last``."""
        if (
            self.save_last
            and self._last_step_idx is not None
            and self._last_step_tensor is not None
            and self._last_step_idx not in self._saved_steps
        ):
            self._save_step(self._last_step_idx, self._last_step_tensor)
        self._last_step_tensor = None

    def collect(self, layer_id: int, attn: torch.Tensor) -> None:
        if layer_id not in self.layer_pos:
            return
        tensor = attn.detach()
        if self.head_mean:
            # Reduce H -> 1 immediately on GPU to keep memory low; output stays
            # 4D ([B, 1, q, k]) so downstream stack/save logic is unchanged.
            tensor = tensor.to(dtype=torch.float32).mean(dim=1, keepdim=True)
        tensor = tensor.to(dtype=torch.float16).cpu()
        self._buf[layer_id] = tensor
        if len(self._buf) == len(self.layers_to_collect):
            ordered = [self._buf[l] for l in self.layers_to_collect]
            stacked = torch.stack(ordered, dim=0)
            self._buf.clear()
            self._emit_step(stacked)
            del stacked

    def flush_partial(self) -> None:
        """Promote any partial buffer to a step (used when a generation ends
        mid-layer-loop, which should not normally happen)."""
        if self._buf:
            ordered = [self._buf[l] for l in self.layers_to_collect if l in self._buf]
            stacked = torch.stack(ordered, dim=0)
            self._buf.clear()
            self._emit_step(stacked)
            del stacked


def _patched_sdpa(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Drop-in replacement for ``LLaDABlock._scaled_dot_product_attention``.

    Computes attention exactly as the original (no flash, no fused SDPA) so we
    can intercept the softmax probabilities. Output is the attention-weighted
    value tensor with the same shape/dtype as the original.
    """
    # GQA expansion (mirrors original logic)
    assert k.size(1) == v.size(1)
    num_kv_heads = k.size(1)
    num_q_heads = q.size(1)
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        nrep = num_q_heads // num_kv_heads
        k = k.repeat_interleave(nrep, dim=1, output_size=num_q_heads)
        v = v.repeat_interleave(nrep, dim=1, output_size=num_q_heads)

    head_dim = q.size(-1)
    scale = 1.0 / math.sqrt(head_dim)

    # Cast to float32 for numerically stable softmax, then back.
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale  # [B, H, q, k]
    if attn_mask is not None:
        # attn_mask comes pre-shaped to [B, 1, q, k] (or broadcastable) by caller.
        scores = scores + attn_mask.to(scores.dtype)
    attn = F.softmax(scores, dim=-1)

    # Stash attention weights into the bag attached to this block.
    bag: Optional[AttentionBag] = getattr(self, "_attn_bag", None)
    if bag is not None:
        bag.collect(self.layer_id, attn)

    if dropout_p > 0.0 and self.training:
        attn = F.dropout(attn, p=dropout_p)

    out = torch.matmul(attn.to(v.dtype), v)
    return out


def _iter_blocks(model):
    """Yields all LLaDABlock instances regardless of block_group_size."""
    inner = getattr(model, "model", model)  # LLaDAModelLM -> LLaDAModel
    transformer = inner.transformer
    if hasattr(transformer, "blocks") and len(getattr(transformer, "blocks", [])):
        for blk in transformer.blocks:
            yield blk
    elif hasattr(transformer, "block_groups"):
        for grp in transformer.block_groups:
            for blk in grp:
                yield blk


def _resolve_layers(model, layers: Optional[Sequence[int]]):
    blocks = list(_iter_blocks(model))
    n_layers = len(blocks)
    if layers is None:
        # default: first / middle / last
        mid = n_layers // 2
        layers = [0, mid, n_layers - 1]
    layers = sorted(set(int(l) for l in layers))
    for l in layers:
        if not (0 <= l < n_layers):
            raise ValueError(f"layer index {l} out of range [0, {n_layers})")
    return blocks, layers


@contextmanager
def collect_attentions(
    model,
    layers: Optional[Sequence[int]] = None,
    stream_dir: Optional[str] = None,
    stream_save_fn: Optional[Callable[[int, torch.Tensor], None]] = None,
    save_steps: Optional[Sequence[int]] = None,
    save_last: bool = False,
    head_mean: bool = False,
):
    """Context manager that monkey-patches LLaDABlocks to collect attention.

    Usage::

        with collect_attentions(model, layers=[0, 16, 31]) as bag:
            output_ids = model.mmu_generate(..., decode_strategy='dcd', ...)
        # bag.steps : List[Tensor[L, B, H, q_len, k_len]]

    If ``stream_dir`` (or ``stream_save_fn``) is provided, each completed step
    is written to disk immediately and dropped from memory. ``bag.num_steps``
    will reflect how many steps were saved.
    """
    blocks, layers = _resolve_layers(model, layers)
    bag = AttentionBag(
        layers,
        stream_dir=stream_dir,
        stream_save_fn=stream_save_fn,
        save_steps=save_steps,
        save_last=save_last,
        head_mean=head_mean,
    )

    saved = []
    for blk in blocks:
        saved.append((blk, blk._scaled_dot_product_attention))
        blk._attn_bag = bag if blk.layer_id in bag.layer_pos else None
        # bind patched function as a method on this specific block
        blk._scaled_dot_product_attention = _patched_sdpa.__get__(blk, type(blk))

    try:
        yield bag
        bag.flush_partial()
        bag.finalize()
    finally:
        for blk, original in saved:
            blk._scaled_dot_product_attention = original
            if hasattr(blk, "_attn_bag"):
                delattr(blk, "_attn_bag")
