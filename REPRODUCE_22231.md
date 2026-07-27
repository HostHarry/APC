# 22231-lavida: LaViDa + VCHD / CCAW / Paired Prefix-KV-Cache (reproduction guide)

Snapshot of the working tree on the 22231 machine (2026-07-27). This is the code that
produced the LLaVA-Bench-COCO and MMMU numbers below. Base: `jacklishufan/LaViDa` @
`24220c0` + all local changes committed here.

## 1. Environment / weights

- conda env: python 3.10 + torch 2.x + the upstream LaViDa `eval/` requirements
  (`lmms-eval` vendored under `eval/lmms_eval`). We used env name `CrossMatch`.
- checkpoint: `hbXNov/lavida-llada-reason`, placed at `lavida-ckpts/lavida-llada-reason`
  (same weights as the dir mislabeled `lavida-llada-hd-reason` on the 43133 machine).
- GPT judge / extraction: export `OPENAI_API_KEY` yourself. All hardcoded keys in
  scripts were replaced with `REPLACE_WITH_YOUR_OPENAI_API_KEY`.

## 2. What is modified vs upstream LaViDa (code map)

| Area | File(s) | Change |
|---|---|---|
| Bug A: SDPA ignored attention bias | `llava/model/language_model/llada/modeling_llada.py` | `_scaled_dot_product_attention` now actually passes `attn_mask` / `is_causal` to `F.scaled_dot_product_attention`. Without this, `attention_bias` (visual ablation, prefix masks) was silently dropped. |
| Bug B: auto-causal bias with KV cache | same file | When the caller supplies no bias/padding/alibi, the auto-synthesized *causal* bias is suppressed (broad suppression, cached path included). LaViDa was trained under the Bug-A regime where cached inference was effectively bidirectional; re-enabling causal masking collapses generation (empty / `<end>` outputs). |
| VCHD decoding | `llava/decoding/` (`decoder.py`, `selector.py`, `selector_cd_apc.py`, `window.py`, `config.py`) | Visual-Contrast Hallucination-Detection gating (dual branch visual/ablated, contrast gate `tau_contrast`, `alpha`), CCAW adaptive window (`ccaw_mode=inverse_window`), history stability. |
| Paired Prefix-LM prompt KV cache | `llava/decoding/lavida_adapter.py` | `batch=2` (visual + ablated) prompt cache: one prefill writes both branches' KVs with `build_paired_attention_bias_from_mask`; decode steps feed only response embeddings and reuse the pair cache. Response stays **bidirectional** (`cached=False` in `build_paired_prefix_attention_bias`) to match LaViDa's trained behavior. |
| Model glue | `eval/lmms_eval/models/llava_llada.py`, `llava/model/language_model/llava_llada.py` | `decode_strategy=vchd`, `vchd__*` gen_kwargs plumbing, prefix cache toggle. |
| Tests | `tests/decoding/test_lavida_adapter.py` | bias construction, cache/no-cache logit parity, real-LLaDA regression tests for Bugs A/B. |
| Scoring | `eval/rescore_mmmu_boxed_first.py` | Strict offline MMMU rescoring: boxed-first extraction, **no random fallback** (unparseable = wrong). Use this for any reasoning-style checkpoint; the stock lmms-eval MMMU parser misses `\boxed{X}` and falls back to `random.choice`. |

## 3. Headline results (this tree, lavida-llada-reason)

### LLaVA-Bench-COCO, 90 samples, gpt-4o relative judge
Protocol: `max_new_tokens=256`, `block_length=256`, 128 steps, prefix_lm.

| mode | all | conv | detail | complex |
|---|---|---|---|---|
| original | 54.5 | 68.1 | 45.7 | 51.1 |
| SWD / PSP / PSP+VRG (43133 tree, same judge, same ckpt) | 60.1 / 56.3 / 61.2 | | | |
| **vchd_prefix_cache** | **80.3** | 69.7 | 74.1 | 94.5 |
| vchd_ccaw_prefix_cache | 72.1 | 64.5 | 71.9 | 78.8 |

Mechanism of the win: VCHD's contrast gate delays premature end-token commits;
truncation rate 10/90 vs 40-52/90 for the other modes. Known remaining issue:
repetition in long answers (~45/90).

### MMMU dev/val (mc-aligned protocol: 128 tok, until `\n\n`, block 64, steps 64)
Scored with `rescore_mmmu_boxed_first.py` (strict, no random fallback), weighted dev+val:

| mode | dev | val | weighted |
|---|---|---|---|
| original | 36.00 | 35.78 | 35.81 |
| vchd_prefix_cache (bidir-fix run) | 38.67 | 35.44 | 35.90 |
| vchd_ccaw_prefix_cache | 39.33 | 34.33 | 35.05 |
| 43133 SWD / PSP / PSP+VRG (their uploaded samples, same script) | | | 33.52 / 35.90 / 35.81 |

i.e. **multiple-choice parity across all methods**; the 36-40% "official" numbers on the
43133 branch come from (a) a boxed-first patch inside their `mmmu/utils.py` plus
(b) `random.choice` fallback on the 17-25% truncated answers (+3-6pp noise).
McNemar original vs VCHD+CCAW: 48/40 flips, p≈0.46. CCAW's entire -0.76pp is a
net 8 questions where the wide adaptive window (16→64) fails to reach `\boxed{}`
within the 128-token budget; accuracy on co-answered questions is identical.

Raw artifacts: `eval/reported_results/` (samples, judged files, judge summaries,
per-question rescore details, `mmmu_boxed_first_rescore/summary.md` with the full verdict).

## 4. How to run

```bash
# LLaVA-Bench-COCO, our three modes (writes eval/logs/llavabench_coco_ours_*/)
bash eval/run_llavabench_coco_ours.sh

# judge offline with gpt-4o (per samples file)
python scripts/judge_llava_bench_samples.py \
  --samples <...samples_llava_bench_coco.jsonl> --model gpt-4o

# MMMU + MMBench, aligned MC protocol (original / vchd_prefix_cache / vchd_ccaw_prefix_cache)
bash eval/run_mc_aligned_ours.sh

# strict MMMU rescoring (boxed-first, no random fallback)
python eval/rescore_mmmu_boxed_first.py --runs <dir with samples_mmmu_{dev,val}.jsonl> ...
```

Key gen_kwargs for the flagship mode (`vchd_prefix_cache`):
`decode_strategy=vchd, prefix_lm=True, vchd__prefix_prompt_cache=True,
vchd__alpha=0.25, vchd__beta=0.1, vchd__tau_base=0.1, vchd__tau_contrast=0.9,
vchd__mask_capacity=16`. Add `vchd__ccaw_enabled=True, vchd__ccaw_mode=inverse_window,
vchd__ccaw_max_mask_capacity=64` for the CCAW variant (not recommended as default;
see MMMU/LLaVA numbers above).

## 5. Caveats

- Do NOT re-enable causal masking on the cached path (either the auto-bias in
  `modeling_llada.py` or `cached=True` in `build_paired_prefix_attention_bias`):
  the reason checkpoint collapses to empty/`<end>` outputs (verified twice).
- `tests/decoding/test_lavida_adapter.py` covers the bias/cache invariants; run it
  after touching `modeling_llada.py` or `lavida_adapter.py`.
- The mc-aligned full MMMU rerun of `vchd_prefix_cache` on exactly this tree was still
  queued when this snapshot was taken; the table above uses the bidir-fix run
  (same code paths for the decoder; only earlier snapshot of eval scripts).
