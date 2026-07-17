from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PairedLogits:
    visual: torch.FloatTensor
    ablated: torch.FloatTensor
    cache_event: str = "disabled"
    cache_refresh_reason: Optional[str] = None
    query_start: int = 0
    query_tokens: int = 0
    model_forward_calls: int = 1


PastKeyValue = Tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class BranchKVCache:
    """One branch's private DCD-style per-layer KV cache."""

    past_key_values: Tuple[PastKeyValue, ...]
    context_version: int


@dataclass(frozen=True)
class PairedKVCache:
    """Strictly disjoint visual and ablated branch caches."""

    visual: BranchKVCache
    ablated: BranchKVCache
    token_snapshot: torch.LongTensor
    sequence_length: int
    last_full_refresh_version: int


def build_paired_attention_bias(
    seq_len: int,
    image_span: Tuple[int, int],
    *,
    device: torch.device,
) -> torch.FloatTensor:
    """Build same-shape visual and visual-access-ablated additive biases."""

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    image_left, image_right = (int(image_span[0]), int(image_span[1]))
    if not 0 <= image_left < image_right <= seq_len:
        raise ValueError(
            f"Invalid image span [{image_left}, {image_right}) for length {seq_len}"
        )

    bias = torch.zeros(
        2, 1, seq_len, seq_len, dtype=torch.float32, device=device
    )
    non_image_queries = torch.ones(seq_len, dtype=torch.bool, device=device)
    non_image_queries[image_left:image_right] = False
    blocked = (
        non_image_queries[:, None]
        & (
            (torch.arange(seq_len, device=device) >= image_left)
            & (torch.arange(seq_len, device=device) < image_right)
        )[None, :]
    )
    bias[1, 0].masked_fill_(blocked, torch.finfo(torch.float32).min)
    return bias


def _normalize_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise ValueError(
            "The phase 0--2 adapter expects a [1, sequence] attention mask"
        )
    attention_mask = attention_mask.to(device=device)
    mask_len = attention_mask.shape[1]
    if mask_len > seq_len:
        raise ValueError(
            f"attention_mask length {mask_len} exceeds sequence length {seq_len}"
        )
    if mask_len < seq_len:
        response_mask = torch.ones(
            1,
            seq_len - mask_len,
            dtype=attention_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat([attention_mask, response_mask], dim=1)
    return attention_mask


@contextmanager
def _math_sdpa_context(enabled: bool) -> Iterator[None]:
    if not enabled or not torch.cuda.is_available():
        with nullcontext():
            yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError):
        SDPBackend = None
        sdpa_kernel = None

    if SDPBackend is not None and sdpa_kernel is not None:
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return

    # Compatibility with older PyTorch releases used by some MMaDA setups.
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    ):
        yield


class MMaDAVisualAccessAdapter:
    """Backend-matched paired forward with optional disjoint branch caches."""

    def __init__(
        self,
        model,
        *,
        decode_start: int,
        decode_end: int,
        image_span: Tuple[int, int],
        attention_mask: Optional[torch.Tensor] = None,
        force_math_sdpa: bool = True,
        mask_id: int = 126336,
        cache_type: str = "none",
        cache_refresh_interval: int = 8,
    ) -> None:
        if not 0 <= decode_start < decode_end:
            raise ValueError(
                f"Invalid decode range [{decode_start}, {decode_end})"
            )
        if cache_type not in {"none", "dual"}:
            raise ValueError(
                f"cache_type must be 'none' or 'dual', got {cache_type!r}"
            )
        if cache_refresh_interval < 1:
            raise ValueError("cache_refresh_interval must be at least 1")
        self.model = model
        self.decode_start = int(decode_start)
        self.decode_end = int(decode_end)
        self.image_span = (int(image_span[0]), int(image_span[1]))
        self.attention_mask = attention_mask
        self.force_math_sdpa = bool(force_math_sdpa)
        self.mask_id = int(mask_id)
        self.cache_type = str(cache_type)
        self.cache_refresh_interval = int(cache_refresh_interval)

        self._branch_bias: Optional[torch.Tensor] = None
        self._single_attention_mask: Optional[torch.Tensor] = None
        self._cache_state: Optional[PairedKVCache] = None
        self._cached_visual_logits: Optional[torch.Tensor] = None
        self._cached_ablated_logits: Optional[torch.Tensor] = None
        self._pending_full_refresh_reason: Optional[str] = None
        self._full_refreshes = 0
        self._partial_refreshes = 0
        self._model_forward_calls = 0
        self._branch_evaluations = 0
        self._logical_query_tokens = 0
        self._refresh_reasons: Dict[str, int] = {}

    @property
    def cache_state(self) -> Optional[PairedKVCache]:
        """Expose cache metadata for diagnostics without merging the branches."""

        return self._cache_state

    def request_full_refresh(self, reason: str) -> None:
        """Force the next dual-cache snapshot to rebuild both branch caches."""

        reason = str(reason).strip()
        if not reason:
            raise ValueError("A cache refresh reason must be non-empty")
        if self.cache_type == "dual":
            self._pending_full_refresh_reason = reason

    def cache_report(self) -> Dict[str, Any]:
        state = self._cache_state
        return {
            "cache_type": self.cache_type,
            "cache_full_refreshes": self._full_refreshes,
            "cache_partial_refreshes": self._partial_refreshes,
            "cache_model_forward_calls": self._model_forward_calls,
            "cache_branch_evaluations": self._branch_evaluations,
            "cache_logical_query_tokens": self._logical_query_tokens,
            "cache_refresh_reasons": dict(self._refresh_reasons),
            "cache_visual_context_version": (
                None if state is None else state.visual.context_version
            ),
            "cache_ablated_context_version": (
                None if state is None else state.ablated.context_version
            ),
            "cache_last_full_refresh_version": (
                None if state is None else state.last_full_refresh_version
            ),
        }

    def _validate_tokens(self, tokens: torch.LongTensor) -> int:
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(
                "The VCHD reference adapter supports batch_size=1 only"
            )
        seq_len = int(tokens.shape[1])
        if self.decode_end > seq_len:
            raise ValueError(
                f"decode_end={self.decode_end} exceeds sequence length {seq_len}"
            )
        if self.cache_type == "dual" and self.decode_end != seq_len:
            raise ValueError(
                "DCD-style VCHD caching requires the decode range to be the "
                "sequence suffix"
            )
        return seq_len

    def _ensure_static_inputs(
        self, tokens: torch.LongTensor, seq_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._branch_bias is None:
            self._branch_bias = build_paired_attention_bias(
                seq_len, self.image_span, device=tokens.device
            )
            self._single_attention_mask = _normalize_attention_mask(
                self.attention_mask,
                seq_len=seq_len,
                device=tokens.device,
            )
        elif (
            self._branch_bias.shape[-1] != seq_len
            or self._branch_bias.device != tokens.device
        ):
            raise ValueError(
                "An adapter instance cannot be reused across sequence lengths "
                "or devices"
            )
        return self._branch_bias, self._single_attention_mask

    @staticmethod
    def _compact_response_logits(
        pair_logits: torch.Tensor,
        *,
        decode_start: int,
        decode_end: int,
        seq_len: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if pair_logits.shape[0] != 2 or pair_logits.shape[1] != seq_len:
            raise RuntimeError(
                "Unexpected paired logits shape: "
                f"{tuple(pair_logits.shape)} for sequence length {seq_len}"
            )
        visual, ablated = pair_logits.chunk(2, dim=0)
        # clone() prevents a response slice from retaining full-sequence logits.
        return (
            visual[0, decode_start:decode_end].float().clone(),
            ablated[0, decode_start:decode_end].float().clone(),
        )

    @staticmethod
    def _split_private_branch_caches(
        past_key_values: Optional[Sequence[PastKeyValue]],
        *,
        context_version: int,
    ) -> Tuple[BranchKVCache, BranchKVCache]:
        if not past_key_values:
            raise RuntimeError(
                "The model did not return past_key_values for VCHD dual cache"
            )
        visual_layers = []
        ablated_layers = []
        for layer_index, layer_cache in enumerate(past_key_values):
            if len(layer_cache) != 2:
                raise RuntimeError(
                    f"Layer {layer_index} cache is not a (key, value) pair"
                )
            key, value = layer_cache
            if key.shape[0] != 2 or value.shape[0] != 2:
                raise RuntimeError(
                    "A paired cache seed must have batch dimension 2"
                )
            # Separate clones are intentional: the two branches must not share
            # storage because replace_position updates cache tensors in place.
            visual_layers.append(
                (key[0:1].clone(), value[0:1].clone())
            )
            ablated_layers.append(
                (key[1:2].clone(), value[1:2].clone())
            )
        return (
            BranchKVCache(
                past_key_values=tuple(visual_layers),
                context_version=int(context_version),
            ),
            BranchKVCache(
                past_key_values=tuple(ablated_layers),
                context_version=int(context_version),
            ),
        )

    def _record_refresh(
        self,
        *,
        full: bool,
        reason: Optional[str],
        query_tokens: int,
        model_forward_calls: int,
    ) -> None:
        if full:
            self._full_refreshes += 1
            if reason is not None:
                self._refresh_reasons[reason] = (
                    self._refresh_reasons.get(reason, 0) + 1
                )
        else:
            self._partial_refreshes += 1
        self._model_forward_calls += int(model_forward_calls)
        self._branch_evaluations += 2
        self._logical_query_tokens += 2 * int(query_tokens)

    def _full_forward(
        self,
        tokens: torch.LongTensor,
        *,
        seq_len: int,
        context_version: int,
        use_cache: bool,
        refresh_reason: Optional[str],
    ) -> PairedLogits:
        branch_bias, single_attention_mask = self._ensure_static_inputs(
            tokens, seq_len
        )
        pair_tokens = tokens.repeat(2, 1)
        pair_attention_mask = (
            None
            if single_attention_mask is None
            else single_attention_mask.repeat(2, 1)
        )

        with torch.inference_mode(), _math_sdpa_context(self.force_math_sdpa):
            output = self.model(
                input_ids=pair_tokens,
                attention_mask=pair_attention_mask,
                attention_bias=branch_bias,
                use_cache=use_cache,
            )

        visual_logits, ablated_logits = self._compact_response_logits(
            output.logits,
            decode_start=self.decode_start,
            decode_end=self.decode_end,
            seq_len=seq_len,
        )
        if not use_cache:
            self._model_forward_calls += 1
            self._branch_evaluations += 2
            self._logical_query_tokens += 2 * seq_len
            return PairedLogits(
                visual=visual_logits,
                ablated=ablated_logits,
                cache_event="disabled",
                query_start=0,
                query_tokens=seq_len,
                model_forward_calls=1,
            )

        visual_cache, ablated_cache = self._split_private_branch_caches(
            output.past_key_values,
            context_version=context_version,
        )
        self._cache_state = PairedKVCache(
            visual=visual_cache,
            ablated=ablated_cache,
            token_snapshot=tokens.clone(),
            sequence_length=seq_len,
            last_full_refresh_version=int(context_version),
        )
        self._cached_visual_logits = visual_logits
        self._cached_ablated_logits = ablated_logits
        self._pending_full_refresh_reason = None
        self._record_refresh(
            full=True,
            reason=refresh_reason,
            query_tokens=seq_len,
            model_forward_calls=1,
        )
        return PairedLogits(
            visual=visual_logits,
            ablated=ablated_logits,
            cache_event="full_refresh",
            cache_refresh_reason=refresh_reason,
            query_start=0,
            query_tokens=seq_len,
            model_forward_calls=1,
        )

    def _partial_forward(
        self,
        tokens: torch.LongTensor,
        *,
        seq_len: int,
        context_version: int,
    ) -> PairedLogits:
        state = self._cache_state
        if state is None:
            raise RuntimeError("Cannot run a partial refresh without a cache seed")
        if (
            self._cached_visual_logits is None
            or self._cached_ablated_logits is None
        ):
            raise RuntimeError("Cached branch logits are missing")
        if state.visual.context_version != state.ablated.context_version:
            raise RuntimeError("Visual and ablated cache versions diverged")
        if context_version != state.visual.context_version + 1:
            raise RuntimeError(
                "A partial cache refresh requires exactly one new context "
                f"version ({state.visual.context_version} -> {context_version})"
            )

        changed = tokens[0] != state.token_snapshot[0]
        changed_positions = torch.nonzero(changed, as_tuple=True)[0]
        if changed_positions.numel() == 0:
            raise RuntimeError(
                "Context version advanced without any token-state change"
            )
        if bool(
            (
                (changed_positions < self.decode_start)
                | (changed_positions >= self.decode_end)
            ).any()
        ):
            return self._full_forward(
                tokens,
                seq_len=seq_len,
                context_version=context_version,
                use_cache=True,
                refresh_reason="out_of_range_change",
            )

        remaining_masks = torch.nonzero(
            tokens[0, self.decode_start : self.decode_end] == self.mask_id,
            as_tuple=True,
        )[0]
        if remaining_masks.numel() == 0:
            raise RuntimeError("Partial refresh requested after decoding completed")
        first_mask = self.decode_start + int(remaining_masks[0].item())
        refresh_start = min(
            int(changed_positions[0].item()),
            first_mask,
        )
        suffix_tokens = tokens[:, refresh_start:self.decode_end]
        query_tokens = int(suffix_tokens.shape[1])
        replace_position = torch.zeros_like(tokens, dtype=torch.bool)
        replace_position[:, refresh_start:self.decode_end] = True
        branch_bias, single_attention_mask = self._ensure_static_inputs(
            tokens, seq_len
        )

        with torch.inference_mode(), _math_sdpa_context(self.force_math_sdpa):
            visual_output = self.model(
                input_ids=suffix_tokens,
                attention_mask=single_attention_mask,
                attention_bias=branch_bias[0:1],
                past_key_values=state.visual.past_key_values,
                use_cache=True,
                replace_position=replace_position,
            )
            ablated_output = self.model(
                input_ids=suffix_tokens,
                attention_mask=single_attention_mask,
                attention_bias=branch_bias[1:2],
                past_key_values=state.ablated.past_key_values,
                use_cache=True,
                replace_position=replace_position,
            )

        if visual_output.logits.shape[:2] != (1, query_tokens):
            raise RuntimeError(
                "Unexpected visual cached logits shape: "
                f"{tuple(visual_output.logits.shape)}"
            )
        if ablated_output.logits.shape[:2] != (1, query_tokens):
            raise RuntimeError(
                "Unexpected ablated cached logits shape: "
                f"{tuple(ablated_output.logits.shape)}"
            )
        response_offset = refresh_start - self.decode_start
        self._cached_visual_logits[response_offset:].copy_(
            visual_output.logits[0].float()
        )
        self._cached_ablated_logits[response_offset:].copy_(
            ablated_output.logits[0].float()
        )
        visual_cache = BranchKVCache(
            past_key_values=tuple(visual_output.past_key_values),
            context_version=int(context_version),
        )
        ablated_cache = BranchKVCache(
            past_key_values=tuple(ablated_output.past_key_values),
            context_version=int(context_version),
        )
        self._cache_state = PairedKVCache(
            visual=visual_cache,
            ablated=ablated_cache,
            token_snapshot=tokens.clone(),
            sequence_length=seq_len,
            last_full_refresh_version=state.last_full_refresh_version,
        )
        self._record_refresh(
            full=False,
            reason=None,
            query_tokens=query_tokens,
            model_forward_calls=2,
        )
        return PairedLogits(
            visual=self._cached_visual_logits,
            ablated=self._cached_ablated_logits,
            cache_event="partial_refresh",
            query_start=refresh_start,
            query_tokens=query_tokens,
            model_forward_calls=2,
        )

    def paired_forward(
        self,
        tokens: torch.LongTensor,
        *,
        context_version: int = 0,
    ) -> PairedLogits:
        seq_len = self._validate_tokens(tokens)
        context_version = int(context_version)
        if context_version < 0:
            raise ValueError("context_version must be non-negative")
        if self.cache_type == "none":
            return self._full_forward(
                tokens,
                seq_len=seq_len,
                context_version=context_version,
                use_cache=False,
                refresh_reason=None,
            )

        state = self._cache_state
        refresh_reason = self._pending_full_refresh_reason
        if state is None:
            refresh_reason = refresh_reason or "initial"
        elif state.sequence_length != seq_len:
            refresh_reason = "sequence_length_change"
        elif context_version < state.visual.context_version:
            raise ValueError(
                "context_version cannot move backwards relative to the cache"
            )
        elif (
            context_version - state.last_full_refresh_version
            >= self.cache_refresh_interval
        ):
            refresh_reason = refresh_reason or "context_interval"
        elif context_version > state.visual.context_version + 1:
            refresh_reason = refresh_reason or "context_gap"
        elif context_version == state.visual.context_version:
            if not torch.equal(tokens, state.token_snapshot):
                refresh_reason = refresh_reason or "same_version_change"
            elif refresh_reason is None:
                return PairedLogits(
                    visual=self._cached_visual_logits,
                    ablated=self._cached_ablated_logits,
                    cache_event="reuse",
                    query_start=self.decode_start,
                    query_tokens=0,
                    model_forward_calls=0,
                )

        if refresh_reason is not None:
            return self._full_forward(
                tokens,
                seq_len=seq_len,
                context_version=context_version,
                use_cache=True,
                refresh_reason=refresh_reason,
            )
        return self._partial_forward(
            tokens,
            seq_len=seq_len,
            context_version=context_version,
        )
