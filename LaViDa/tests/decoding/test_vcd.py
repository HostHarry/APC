"""CPU-only tests for pure VCD (noised-image negative branch)."""
from __future__ import annotations

import torch

from llava.decoding import (
    LaViDaVisualAccessAdapter,
    VCHDDecodeConfig,
    add_diffusion_noise,
    build_paired_attention_bias_from_mask,
    build_paired_shared_attention_bias_from_mask,
    visual_contrastive_decode_vcd,
)
from llava.decoding.generate_utils import extract_decode_options


class _Output:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.attn_key_values = None


class _DualEmbedModel:
    """Model whose logits depend on prompt-embed content (detects noise branch)."""

    def __init__(self, vocab_size: int = 16, hidden: int = 4):
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.config = type(
            "Config",
            (),
            {"vocab_size": vocab_size, "llm_vocab_size": vocab_size},
        )()
        self.transformer = type("T", (), {})()
        # Embedding table: token id -> one-hot-ish vector in first vocab dims.
        weight = torch.zeros(vocab_size, hidden)
        for i in range(min(vocab_size, hidden)):
            weight[i, i % hidden] = 1.0
        self.transformer.wte = lambda ids: weight[ids]

    def __call__(self, _input_ids=None, **kwargs):
        embeds = kwargs["input_embeddings"]  # [2, S, H]
        # Map mean of each row's embeds to a preferred token logit.
        scores = embeds.mean(dim=(1, 2))  # [2]
        seq_len = embeds.shape[1]
        logits = torch.zeros(2, seq_len, self.vocab_size)
        # Branch 0 prefers token 1; branch 1 prefers token 2 when embeds differ.
        logits[0, :, 1] = 5.0 + scores[0]
        logits[0, :, 2] = 1.0
        logits[1, :, 1] = 1.0
        logits[1, :, 2] = 5.0 + scores[1]
        # Also allow token 3 as low-prob filler.
        logits[:, :, 3] = 0.5
        return _Output(logits)


def test_add_diffusion_noise_changes_tensor():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8, 8)
    y = add_diffusion_noise(x, noise_step=500)
    assert y.shape == x.shape
    assert not torch.allclose(x, y)


def test_shared_bias_has_no_ablation():
    mask = torch.tensor([False, True, True, False])
    ablated = build_paired_attention_bias_from_mask(mask, ablate_visual_access=True)
    shared = build_paired_shared_attention_bias_from_mask(mask)
    assert torch.allclose(ablated[0], shared[0])
    assert not torch.allclose(ablated[1], shared[1])
    assert torch.allclose(shared[0], shared[1])


def test_extract_vcd_preset():
    kwargs = {"decode_strategy": "vcd", "vcd__noise_step": "400"}
    strategy, cfg = extract_decode_options(kwargs)
    assert strategy == "vcd"
    assert cfg["negative_branch"] == "noise_image"
    assert cfg["noise_step"] == 400
    assert cfg["alpha"] == 1.0
    assert cfg["beta"] == 0.1
    assert cfg["enable_u_gate"] is False


def test_dual_embeds_change_ablated_logits():
    torch.manual_seed(0)
    model = _DualEmbedModel()
    prompt_len = 4
    hidden = 4
    visual_mask = torch.tensor([False, True, True, False])
    prompt = torch.randn(1, prompt_len, hidden)
    prompt_neg = prompt + 1.5  # clearly different
    adapter = LaViDaVisualAccessAdapter(
        model,
        prompt_embeds=prompt,
        visual_mask=visual_mask,
        decode_start=prompt_len,
        decode_end=prompt_len + 4,
        mask_id=0,
        prompt_embeds_negative=prompt_neg,
    )
    assert adapter.ablate_visual_access is False
    tokens = torch.zeros(1, prompt_len + 4, dtype=torch.long)
    paired = adapter.paired_forward(tokens)
    # With different prompt embeds, visual and ablated logits must differ.
    assert not torch.allclose(paired.visual, paired.ablated)


def test_vcd_decode_commits_tokens():
    torch.manual_seed(0)
    model = _DualEmbedModel()
    prompt_len = 4
    max_new = 4
    visual_mask = torch.tensor([False, True, True, False])
    prompt = torch.randn(1, prompt_len, 4)
    prompt_neg = torch.randn(1, prompt_len, 4)
    adapter = LaViDaVisualAccessAdapter(
        model,
        prompt_embeds=prompt,
        visual_mask=visual_mask,
        decode_start=prompt_len,
        decode_end=prompt_len + max_new,
        mask_id=0,
        prompt_embeds_negative=prompt_neg,
    )
    tokens = torch.zeros(1, prompt_len + max_new, dtype=torch.long)
    config = VCHDDecodeConfig(
        mask_id=0,
        alpha=1.0,
        beta=0.1,
        negative_branch="noise_image",
        enable_u_gate=False,
        enable_r_gate=False,
        enable_g_gate=False,
        text_vocab_size=16,
        forbidden_token_ids=(0,),
    )
    out = visual_contrastive_decode_vcd(
        model,
        tokens,
        decode_start=prompt_len,
        decode_end=prompt_len + max_new,
        config=config,
        adapter=adapter,
        block_length=4,
        step_per_block=4,
    )
    response = out[0, prompt_len:]
    assert (response != 0).all()
