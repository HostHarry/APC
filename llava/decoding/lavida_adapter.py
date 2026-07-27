from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import torch

from llava.constants import IMAGE_TOKEN_INDEX


@dataclass(frozen=True)
class PairedLogits:
    visual: torch.FloatTensor
    ablated: torch.FloatTensor
    cache_event: str = "disabled"
    cache_refresh_reason: Optional[str] = None
    query_start: int = 0
    query_tokens: int = 0
    model_forward_calls: int = 1


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
    visual_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    visual_mask[image_left:image_right] = True
    return build_paired_attention_bias_from_mask(visual_mask)


def build_paired_attention_bias_from_mask(
    visual_mask: torch.BoolTensor,
) -> torch.FloatTensor:
    """Build paired biases from an arbitrary visual-token boolean mask."""

    if visual_mask.ndim != 1:
        raise ValueError(
            f"visual_mask must be 1-D, got shape {tuple(visual_mask.shape)}"
        )
    seq_len = int(visual_mask.numel())
    if seq_len <= 0:
        raise ValueError("visual_mask must be non-empty")
    if not bool(visual_mask.any()):
        raise ValueError("visual_mask must contain at least one visual position")

    device = visual_mask.device
    bias = torch.zeros(2, 1, seq_len, seq_len, dtype=torch.float32, device=device)
    non_visual_queries = ~visual_mask
    blocked = non_visual_queries[:, None] & visual_mask[None, :]
    bias[1, 0].masked_fill_(blocked, torch.finfo(torch.float32).min)
    return bias


def build_paired_prefix_attention_bias(
    visual_mask: torch.BoolTensor,
    *,
    prompt_length: int,
    cached: bool = False,
) -> torch.FloatTensor:
    """Build visual/ablated Prefix-LM bias for a masked-diffusion response.

    Two regimes, matching LaViDa's own reference behaviour:

    * ``cached=False`` (single-forward Prefix-LM, no ``past_key_values``): the
      response block attends bidirectionally over both prompt and response
      tokens (``prefix_lm_dllm`` in ``modeling_llada.py`` lets every response
      query see every key). The only Prefix-LM restriction we impose is that
      *prompt* queries may not attend to *response* keys, so the future
      response never leaks into the cached prompt KV.

    * ``cached=True`` (paired-prefix prompt KV cache decode): LaViDa's cached
      generate.py path deliberately turns the response block causal on top of
      a bidirectional prompt cache. This is documented in the upstream
      ``visual_guided_logits`` (``llava/decoding/thinking.py`` on the
      ``hostharry/apc:43133-lavida`` branch) as "Match LaViDa's cached prefix
      path, which becomes causal over the response tokens once past keys are
      supplied." We therefore add response-only causal (``k > q`` within the
      response block) on top of the ``cached=False`` bias. Prompt<->prompt
      attention stays bidirectional so that a full-sequence forward with this
      bias produces the same prompt KVs as the paired prefill call.

    Ablation still blocks non-visual queries from visual keys on branch 1 in
    both regimes.
    """

    seq_len = int(visual_mask.numel())
    prompt_length = int(prompt_length)
    if not 0 < prompt_length <= seq_len:
        raise ValueError(
            f"prompt_length must be in [1, {seq_len}], got {prompt_length}"
        )
    bias = build_paired_attention_bias_from_mask(visual_mask)
    positions = torch.arange(seq_len, device=visual_mask.device)
    query = positions[:, None]
    key = positions[None, :]
    prompt_to_response = (query < prompt_length) & (key >= prompt_length)
    bias[:, 0].masked_fill_(prompt_to_response, torch.finfo(torch.float32).min)
    if cached:
        # Within the response block only, additionally block ``k > q`` so the
        # response is causal on top of a bidirectional prompt. Prompt<->prompt
        # attention stays unrestricted so the reference full-sequence forward
        # produces the same KVs as the prefill call, and response queries can
        # still attend to every prompt key.
        within_response = (query >= prompt_length) & (key >= prompt_length)
        response_causal = within_response & (key > query)
        bias[:, 0].masked_fill_(response_causal, torch.finfo(torch.float32).min)
    return bias


def infer_visual_mask_from_expanded_ids(
    expanded_ids: torch.LongTensor,
) -> torch.BoolTensor:
    """Infer visual positions from expanded multimodal token ids."""

    if expanded_ids.ndim != 1:
        if expanded_ids.ndim == 2 and expanded_ids.shape[0] == 1:
            expanded_ids = expanded_ids[0]
        else:
            raise ValueError(
                "expanded_ids must have shape [S] or [1, S], got "
                f"{tuple(expanded_ids.shape)}"
            )
    return expanded_ids == int(IMAGE_TOKEN_INDEX)


def image_span_from_visual_mask(
    visual_mask: torch.BoolTensor,
) -> Tuple[int, int]:
    """Return a covering [left, right) span for diagnostics / simple adapters."""

    positions = torch.nonzero(visual_mask, as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("Cannot infer image span from an empty visual mask")
    return int(positions[0].item()), int(positions[-1].item()) + 1


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
            "The VCHD adapter expects a [1, sequence] attention mask"
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

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    ):
        yield


class TokenVisualAccessAdapter:
    """Paired forward for token-id models (unit tests / MMaDA-style backends)."""

    def __init__(
        self,
        model,
        *,
        decode_start: int,
        decode_end: int,
        image_span: Optional[Tuple[int, int]] = None,
        visual_mask: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        force_math_sdpa: bool = True,
        mask_id: int = 126336,
        cache_type: str = "none",
        cache_refresh_interval: int = 8,
    ) -> None:
        if cache_type != "none":
            raise ValueError(
                "LaViDa VCHD currently supports cache_type='none' only; "
                f"got {cache_type!r}"
            )
        if not 0 <= decode_start < decode_end:
            raise ValueError(
                f"Invalid decode range [{decode_start}, {decode_end})"
            )
        if visual_mask is None and image_span is None:
            raise ValueError("Either visual_mask or image_span is required")
        self.model = model
        self.decode_start = int(decode_start)
        self.decode_end = int(decode_end)
        self.image_span = (
            None if image_span is None else (int(image_span[0]), int(image_span[1]))
        )
        self.visual_mask = None if visual_mask is None else visual_mask.clone()
        self.attention_mask = attention_mask
        self.force_math_sdpa = bool(force_math_sdpa)
        self.mask_id = int(mask_id)
        self.cache_type = "none"
        self.cache_refresh_interval = int(cache_refresh_interval)
        self._branch_bias: Optional[torch.Tensor] = None
        self._single_attention_mask: Optional[torch.Tensor] = None
        self._model_forward_calls = 0
        self._branch_evaluations = 0
        self._logical_query_tokens = 0

    def request_full_refresh(self, reason: str) -> None:
        del reason
        return None

    def cache_report(self) -> Dict[str, Any]:
        return {
            "cache_type": self.cache_type,
            "cache_full_refreshes": 0,
            "cache_partial_refreshes": 0,
            "cache_model_forward_calls": self._model_forward_calls,
            "cache_branch_evaluations": self._branch_evaluations,
            "cache_logical_query_tokens": self._logical_query_tokens,
            "cache_refresh_reasons": {},
            "cache_visual_context_version": None,
            "cache_ablated_context_version": None,
            "cache_last_full_refresh_version": None,
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
        return seq_len

    def _ensure_static_inputs(
        self, tokens: torch.LongTensor, seq_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._branch_bias is None:
            if self.visual_mask is not None:
                mask = self.visual_mask.to(device=tokens.device)
                if int(mask.numel()) != seq_len:
                    # Visual mask covers the prompt prefix; pad response as non-visual.
                    if int(mask.numel()) > seq_len:
                        raise ValueError(
                            "visual_mask is longer than the current sequence"
                        )
                    pad = torch.zeros(
                        seq_len - int(mask.numel()),
                        dtype=torch.bool,
                        device=tokens.device,
                    )
                    mask = torch.cat([mask, pad], dim=0)
                self._branch_bias = build_paired_attention_bias_from_mask(mask)
            else:
                assert self.image_span is not None
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

    def paired_forward(
        self,
        tokens: torch.LongTensor,
        *,
        context_version: int = 0,
    ) -> PairedLogits:
        del context_version
        seq_len = self._validate_tokens(tokens)
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
                use_cache=False,
            )
        logits = output.logits
        if logits.shape[0] != 2 or logits.shape[1] != seq_len:
            raise RuntimeError(
                "Unexpected paired logits shape: "
                f"{tuple(logits.shape)} for sequence length {seq_len}"
            )
        visual, ablated = logits.chunk(2, dim=0)
        self._model_forward_calls += 1
        self._branch_evaluations += 2
        self._logical_query_tokens += 2 * seq_len
        return PairedLogits(
            visual=visual[0, self.decode_start : self.decode_end].float().clone(),
            ablated=ablated[0, self.decode_start : self.decode_end].float().clone(),
            cache_event="disabled",
            query_start=0,
            query_tokens=seq_len,
            model_forward_calls=1,
        )


class LaViDaVisualAccessAdapter:
    """Paired visual/ablated forward for LaViDa embedding-based multimodal prompts."""

    def __init__(
        self,
        model,
        *,
        prompt_embeds: torch.FloatTensor,
        visual_mask: torch.BoolTensor,
        decode_start: int,
        decode_end: int,
        mask_id: int,
        attention_mask: Optional[torch.Tensor] = None,
        force_math_sdpa: bool = True,
        backend: str = "llada",
        prefix_lm: bool = False,
        prefix_prompt_cache: bool = False,
    ) -> None:
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
            raise ValueError(
                "prompt_embeds must have shape [1, prompt_len, hidden]"
            )
        if visual_mask.ndim != 1:
            raise ValueError("visual_mask must be 1-D over the prompt length")
        if int(visual_mask.numel()) != int(prompt_embeds.shape[1]):
            raise ValueError(
                "visual_mask length must match prompt_embeds length: "
                f"{int(visual_mask.numel())} != {int(prompt_embeds.shape[1])}"
            )
        if not bool(visual_mask.any()):
            raise ValueError(
                "VCHD requires at least one visual token in the prompt"
            )
        if not 0 <= decode_start < decode_end:
            raise ValueError(
                f"Invalid decode range [{decode_start}, {decode_end})"
            )
        if backend not in {"llada", "dream"}:
            raise ValueError(f"Unknown backend {backend!r}")
        if prefix_prompt_cache and not prefix_lm:
            raise ValueError(
                "prefix_prompt_cache=True requires prefix_lm=True"
            )
        if prefix_prompt_cache and backend != "llada":
            raise ValueError(
                "Paired Prefix-LM prompt cache currently supports backend='llada' only"
            )
        self.model = model
        self.prompt_embeds = prompt_embeds
        self.visual_mask = visual_mask.bool().clone()
        self.decode_start = int(decode_start)
        self.decode_end = int(decode_end)
        self.mask_id = int(mask_id)
        self.attention_mask = attention_mask
        self.force_math_sdpa = bool(force_math_sdpa)
        self.backend = backend
        self.prefix_lm = bool(prefix_lm)
        self.prefix_prompt_cache = bool(prefix_prompt_cache)
        self._prompt_cache: Optional[
            Sequence[Tuple[torch.Tensor, torch.Tensor]]
        ] = None
        self._prompt_cache_prefills = 0
        self._model_forward_calls = 0
        self._branch_evaluations = 0
        self._logical_query_tokens = 0
        self.image_span = image_span_from_visual_mask(self.visual_mask)

    def request_full_refresh(self, reason: str) -> None:
        del reason
        return None

    def cache_report(self) -> Dict[str, Any]:
        return {
            "cache_type": "none",
            "cache_full_refreshes": 0,
            "cache_partial_refreshes": 0,
            "cache_model_forward_calls": self._model_forward_calls,
            "cache_branch_evaluations": self._branch_evaluations,
            "cache_logical_query_tokens": self._logical_query_tokens,
            "cache_refresh_reasons": {},
            "cache_visual_context_version": None,
            "cache_ablated_context_version": None,
            "cache_last_full_refresh_version": None,
            "prefix_lm": self.prefix_lm,
            "prompt_cache_type": (
                "paired_prefix" if self.prefix_prompt_cache else "none"
            ),
            "prompt_cache_prefills": self._prompt_cache_prefills,
        }

    def _embed_tokens(self, token_ids: torch.LongTensor) -> torch.FloatTensor:
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte(token_ids)
        if hasattr(self.model, "get_model"):
            inner = self.model.get_model()
            if hasattr(inner, "embed_tokens"):
                return inner.embed_tokens(token_ids)
            if hasattr(inner, "get_input_embeddings"):
                return inner.get_input_embeddings()(token_ids)
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens(token_ids)
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()(token_ids)
        raise AttributeError(
            "LaViDaVisualAccessAdapter could not locate an embedding layer"
        )

    def _forward_pair(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        past_key_values: Optional[
            Sequence[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
    ):
        pair_embeds = inputs_embeds.repeat(2, 1, 1)
        pair_mask = (
            None if attention_mask is None else attention_mask.repeat(2, 1)
        )
        with torch.inference_mode(), _math_sdpa_context(self.force_math_sdpa):
            if self.backend == "llada":
                forward_kwargs = {
                    "input_embeddings": pair_embeds,
                    "attention_mask": pair_mask,
                    "attention_bias": attention_bias,
                    "use_cache": use_cache,
                }
                if past_key_values is not None:
                    forward_kwargs["past_key_values"] = past_key_values
                if last_logits_only:
                    forward_kwargs["last_logits_only"] = True
                return self.model(None, **forward_kwargs)
            if past_key_values is not None or use_cache:
                raise ValueError(
                    "Paired Prefix-LM prompt cache is not implemented for Dream"
                )
            # Dream SDPA requires attn bias dtype to match query/value dtype
            # (typically bf16). float32 bias raises "invalid dtype for bias" on CUDA.
            target_dtype = pair_embeds.dtype
            dream_bias = attention_bias.to(dtype=target_dtype)
            if pair_mask is not None:
                dream_bias = dream_bias + _mask_to_bias(pair_mask, target_dtype)
            if hasattr(self.model, "forward_dream"):
                return self.model.forward_dream(
                    input_ids=None,
                    inputs_embeds=pair_embeds,
                    attention_mask=dream_bias,
                    use_cache=False,
                    num_logits_to_keep=0,
                )
            return self.model(
                input_ids=None,
                inputs_embeds=pair_embeds,
                attention_mask=dream_bias,
                use_cache=False,
            )

    def _prefill_prompt_cache(self) -> None:
        if self._prompt_cache is not None:
            return
        prompt_len = int(self.prompt_embeds.shape[1])
        prompt_bias = build_paired_attention_bias_from_mask(
            self.visual_mask.to(device=self.prompt_embeds.device)
        )
        prompt_attention_mask = _normalize_attention_mask(
            self.attention_mask,
            seq_len=prompt_len,
            device=self.prompt_embeds.device,
        )
        output = self._forward_pair(
            self.prompt_embeds,
            prompt_bias,
            prompt_attention_mask,
            use_cache=True,
            last_logits_only=True,
        )
        cache = getattr(output, "attn_key_values", None)
        if not cache:
            raise RuntimeError(
                "LLaDA Prefix-LM prefill did not return attn_key_values"
            )
        for layer_index, layer_cache in enumerate(cache):
            if len(layer_cache) != 2:
                raise RuntimeError(
                    f"Prompt cache layer {layer_index} is not a (key, value) pair"
                )
            key, value = layer_cache
            if key.shape[0] != 2 or value.shape[0] != 2:
                raise RuntimeError(
                    "Paired prompt cache must preserve batch dimension 2"
                )
            if key.shape[-2] != prompt_len or value.shape[-2] != prompt_len:
                raise RuntimeError(
                    "Paired prompt cache length does not match prompt length"
                )
        self._prompt_cache = tuple(cache)
        self._prompt_cache_prefills += 1
        self._model_forward_calls += 1
        self._branch_evaluations += 2
        self._logical_query_tokens += 2 * prompt_len

    def paired_forward(
        self,
        tokens: torch.LongTensor,
        *,
        context_version: int = 0,
    ) -> PairedLogits:
        del context_version
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("LaViDa VCHD supports batch_size=1 only")
        seq_len = int(tokens.shape[1])
        prompt_len = int(self.prompt_embeds.shape[1])
        if seq_len != self.decode_end:
            raise ValueError(
                "LaViDa VCHD expects tokens to cover the full "
                f"[0, decode_end) range; got length {seq_len}"
            )
        if prompt_len != self.decode_start:
            raise ValueError(
                "prompt_embeds length must equal decode_start: "
                f"{prompt_len} != {self.decode_start}"
            )

        response_ids = tokens[:, self.decode_start : self.decode_end]
        response_embeds = self._embed_tokens(response_ids)

        full_visual_mask = torch.zeros(
            seq_len, dtype=torch.bool, device=tokens.device
        )
        full_visual_mask[:prompt_len] = self.visual_mask.to(device=tokens.device)
        # LaViDa's LLaDA base checkpoint was trained (and its paper numbers
        # collected) with an SDPA path that silently ignored the additive
        # attention bias (Bug A). That accidentally kept both the cached and
        # non-cached prefix-LM decode paths fully bidirectional. Once Bug A
        # is fixed we must be careful not to *add* a response-causal mask on
        # the cached path, or else the paired forward diverges from the
        # regime the weights were tuned for. We therefore always request the
        # bidirectional ``cached=False`` bias here; the ``cached=True`` branch
        # remains available in ``build_paired_prefix_attention_bias`` for AR
        # backends (e.g. Dream) that genuinely need response-causal cached
        # decode.
        branch_bias = (
            build_paired_prefix_attention_bias(
                full_visual_mask,
                prompt_length=prompt_len,
                cached=False,
            )
            if self.prefix_lm
            else build_paired_attention_bias_from_mask(full_visual_mask)
        )
        single_attention_mask = _normalize_attention_mask(
            self.attention_mask,
            seq_len=seq_len,
            device=tokens.device,
        )

        if self.prefix_prompt_cache:
            prefilled = self._prompt_cache is None
            self._prefill_prompt_cache()
            output = self._forward_pair(
                response_embeds,
                branch_bias,
                single_attention_mask,
                past_key_values=self._prompt_cache,
            )
            logits = output.logits
            response_len = self.decode_end - self.decode_start
            if logits.shape[0] != 2 or logits.shape[1] != response_len:
                raise RuntimeError(
                    "Unexpected cached paired logits shape: "
                    f"{tuple(logits.shape)} for response length {response_len}"
                )
            visual, ablated = logits.chunk(2, dim=0)
            self._model_forward_calls += 1
            self._branch_evaluations += 2
            self._logical_query_tokens += 2 * response_len
            return PairedLogits(
                visual=visual[0].float().clone(),
                ablated=ablated[0].float().clone(),
                cache_event=(
                    "prompt_prefill" if prefilled else "prompt_reuse"
                ),
                query_start=self.decode_start,
                query_tokens=response_len,
                model_forward_calls=2 if prefilled else 1,
            )

        inputs_embeds = torch.cat([self.prompt_embeds, response_embeds], dim=1)
        output = self._forward_pair(
            inputs_embeds, branch_bias, single_attention_mask
        )
        logits = output.logits
        if logits.shape[0] != 2 or logits.shape[1] != seq_len:
            raise RuntimeError(
                "Unexpected paired logits shape: "
                f"{tuple(logits.shape)} for sequence length {seq_len}"
            )
        visual, ablated = logits.chunk(2, dim=0)
        self._model_forward_calls += 1
        self._branch_evaluations += 2
        self._logical_query_tokens += 2 * seq_len
        return PairedLogits(
            visual=visual[0, self.decode_start : self.decode_end].float().clone(),
            ablated=ablated[0, self.decode_start : self.decode_end].float().clone(),
            cache_event=(
                "prefix_no_cache" if self.prefix_lm else "disabled"
            ),
            query_start=0,
            query_tokens=seq_len,
            model_forward_calls=1,
        )


def _mask_to_bias(
    attention_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    # attention_mask: [B, S] with 1=keep → additive [B, 1, S, S]
    keep = attention_mask.to(dtype=dtype)
    bias = (1.0 - keep)[:, None, None, :] * torch.finfo(torch.float32).min
    return bias.to(dtype=dtype)


# Back-compat alias used by ported tests.
MMaDAVisualAccessAdapter = TokenVisualAccessAdapter
