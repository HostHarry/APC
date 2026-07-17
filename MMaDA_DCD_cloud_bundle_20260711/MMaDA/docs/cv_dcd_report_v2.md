# CV-DCD for MMaDA — Experiments Report v2

> **Status: paused. Diagnostic phase complete; hypothesis on original mechanism partially falsified.**
> Generated: 2026-07-05
> Follow-up to `cv_dcd_report_v1.md` (design & plan)

---

## TL;DR

1. **Original CV-DCD sweep on LLaVABench (9 configs, λ ∈ {0.25,0.5,1.0} × drop ∈ {mask, shuffle, random_mask}) all regressed vs DCD baseline** — from -3pt to -12pt overall. Detailed table in §3.
2. **Attention analysis on all 60 LLaVABench samples** revealed the failure mode split (§4):
   - ~32% of DCD failures are "degenerate stubs" (`"Jim"`, `"Grapefruit"`, `"5"`) that **do show under-image-attention** with Cohen d = -2.47, p < 10⁻¹⁵.
   - ~35% are long-form hallucinations that show **no attention deficit**.
   - Uniform λ (all-token boost) helps the first group but noise-injects the rest → net negative.
3. **Four bugs identified in CV-DCD implementation** (§5):
   - **B1 (critical)**: `cv_score` in log-space but `decode_algo='threshold'` compares against 0.9 (probability space) → threshold **never fires** → fallback to top-1 commit per step → DCD's multi-token throughput annihilated.
   - **B2 (latent)**: independent gumbel-noise between `x0_cand` and `x0` breaks alignment when `temperature > 0`; not currently triggered (`dcd_temperature=0`).
   - **B3 (severe)**: `image_drop='mask'` fills the visual span with the text mask token (`126336`), an out-of-distribution input for the diffusion model → `drop_logp` is undefined-behaviour, not a valid causal control.
   - **B4 (architectural)**: CV-DCD only modifies commit **timing**, never commit **token**. Base-logit argmax happens first; CV score just re-ranks. Cannot repair language-prior errors at the argmax stage.
4. **Fix 1 (probability-space cv_score) + Fix 3 (neutral VQ-encoded gray image tokens) applied and validated at code level.** Empirically **fixes crashed the sweep further** (-17pt vs baseline) due to a newly-exposed semantic incompatibility: multi-token commit now allows low-`raw_conf` tokens to bypass DCD's confidence gate, causing repetition-collapse (`"items items items items ..."`, `"yellow yellow jacket"`).
5. **Conclusion**: CV-DCD as currently formulated is at best a targeted intervention for a 32% subgroup, and cannot be applied uniformly. Two viable next steps are proposed (§8): **demote-only CV** for safety, or **logit-blending contrastive decoding** for a true fix.

---

## 1. Original Sweep Results (LLaVABench, GPT-4-mini judge)

Full 3×3 = 9 configs, DCD baseline included.

| config             | overall | conv | complex | detail |
|--------------------|---------|------|---------|--------|
| **DCD baseline**   | **44.1**| 32.7 | 50.9    | 45.1   |
| λ=0.25 mask        | 39.9    | 23.7 | 48.3    | 44.3   |
| λ=0.25 random_mask | 39.0    | 25.0 | 48.0    | 40.2   |
| λ=0.25 shuffle     | 44.1    | 35.6 | 49.1    | 44.7   |
| λ=0.5  mask        | 37.1    | 17.4 | 48.5    | 40.7   |
| λ=0.5  random_mask | 39.5    | 24.5 | 47.8    | 42.1   |
| λ=0.5  shuffle     | 41.1    | 26.8 | 46.8    | 47.6   |
| λ=1.0  mask        | 32.4    | -    | -       | -      |
| λ=1.0  random_mask | 37.2    | -    | -       | -      |
| λ=1.0  shuffle     | 34.6    | -    | -       | -      |

**Observations**:

- Monotone dose-response: larger λ → worse.
- `shuffle` (mildest ablation, preserves token distribution) is uniformly least harmful.
- `mask` (strongest ablation, but see §5-B3: it's out-of-distribution, not a clean causal control) is uniformly most harmful.
- `random_mask` is a noisy in-between due to per-step `torch.rand`.
- Detail category anomalously ROSE for `λ=0.5 shuffle` (+2.5pt on detail), suggesting shuffle's disruption of spatial info marginally helped spatial-referring tokens. This is a small signal, not a paradigm shift.

**Interpretation**: uniform-λ CV-DCD is a strict net negative. But we needed to know whether the flaw is "wrong formula" or "wrong target". §4 answers that.

---

## 2. Refuting the Naive Hypothesis via NPZ Debug Dumps

The original CV-DCD proposal assumed: **DCD under-attends to image; adding VCS boosts image-supported tokens; this repairs high-language-prior commits.**

Prediction if true: `visual_gain = base_logp - drop_logp` should be **small (≈0)** for the majority of committed tokens (because DCD is language-driven), with a large positive tail for the few tokens that need image.

**Actual measurement** (drop=mask, λ>0, per commit record, 1280 tokens):

```
raw_conf p50 = 1.0000    ← DCD only commits when very confident
base_logp p50 = 0.0000

VCS quantiles:
    p05 = -0.091   p25 = -0.000   p50 = +0.000
    p75 = +0.000   p95 = +0.617   p99 = +3.760
    mean = +0.117   |VCS|mean = 0.142

    frac(|VCS| < 0.1) = 83.4%   ← image barely matters for these commits
    frac(VCS > 0.5)   =  5.8%   ← minority actually needs image
```

Then, stratified by **commit timing** (early vs late in the diffusion trajectory):

```
tier             raw_conf p50    |VCS| mean    VCS>0.5    |VCS|<0.1
earliest 20%       1.0000          0.44         18.4%       67.5%
early 20%          1.0000          0.16          6.7%       77.2%
mid   20%          1.0000          0.06          1.6%       86.3%
late  20%          1.0000          0.05          2.0%       87.8%
latest 20%         1.0000          0.007         0.4%       98.1%    ← language filler
```

`Pearson(raw_conf, |VCS|) = −0.58` — image dependence is HIGHER for the (few) low-confidence tokens.

Zoom into low-confidence: `raw_conf < 0.5` (n=118): `|VCS| mean = 1.03, VCS>0.5 = 38.1%`.

**Reading**: image IS used, and its use IS concentrated on genuinely-uncertain tokens, but the model has already internalized this information into `raw_conf`. Adding λ·VCS on top mostly re-scores what raw_conf already knows.

---

## 3. Attention Analysis on all 60 LLaVABench Samples

Reused the existing per-sample rendered attention maps in `rendered_all_layers_v3/` (5 layers × ~6 steps × 60 samples ≈ 1735 layer/step observations). Each sample was auto-classified using its (`pred_len`, GPT-4 `score`) tuple:

| group | criterion | n |
|---|---|---|
| degenerate | `pred_len ≤ 60` AND `score ≤ 3` | 19 |
| normal | `score ≥ 6` | 20 |
| error_long | `pred_len > 60` AND `score ≤ 3` | 12 |
| borderline | else (mid scores) | 9 |

Per-sample aggregation (mean over layer/step within each sample, then Welch t-test between groups; `n = # samples`):

### 3.1 `left_half_frac` (fraction of attention on image-heavy left half of sequence)

| group vs normal | Δ (pp) | t | p | Cohen d |
|---|---|---|---|---|
| **degenerate**  | **-4.53** | **-8.01** | **1.3×10⁻¹⁵** *** | **-2.54** |
| error_long | +0.45 | +0.67 | 0.50 ns | +0.23 |
| borderline | -0.54 | -0.55 | 0.58 ns | -0.22 |

### 3.2 Middle-layer subset (layers 8/16/24), same metric

| group | Δ vs normal | Cohen d |
|---|---|---|
| **degenerate** | **-5.83pp** | **-2.47** *** |
| error_long | +0.84pp | +0.16 ns |
| borderline | -0.75pp | -0.24 ns |

### 3.3 Attention entropy (attention diffuseness)

| group | Δ vs normal | p |
|---|---|---|
| degenerate | **+37.5pp** (more diffuse) | 0.008 ** |
| error_long | -8.7pp | 0.56 ns |
| borderline | -5.1pp | 0.72 ns |

**Conclusion**: **~32% of LLaVABench failures ARE the "under-attend to image" pathology proposed by the original CV-DCD hypothesis.** The other ~2/3 (long-form errors, partial-credit borderline) show no attention deficit and are unlikely to benefit from image-conditioned intervention.

---

## 4. Sweep Results Re-interpreted

Given §3, the original sweep numbers make sense:

- CV-DCD's uniform-λ boost helps ~32% of samples (degenerate).
- On ~68% (normal + error_long + borderline), it noise-injects the score → damages tie-break, reorders commits.
- Damage on 68% > help on 32% → net negative.
- `shuffle` was least harmful because it minimally perturbs the score (VCS distribution shifted least from mask; see §2).

The `conv` category, dominated by degenerate answers, showed the largest downward move — because that's where the uniform λ was doing the most damage to samples whose issue is elsewhere.

---

## 5. Code-Level Bug Diagnosis

Deep-dive into `models/mmada_decode.py`. Four issues found; two are critical for the observed behaviour, one is latent, one is architectural.

### 5.1 B1 — cv_score / threshold scale mismatch (CRITICAL)

The threshold path in `_pick_transfer`:

```python
if config.decode_algo == "threshold":
    threshold = float(config.decode_param)      # LLaVABench default = 0.9
    chosen = masked_positions[confidence[batch_idx, masked_positions] >= threshold]
    if chosen.numel() == 0:
        chosen = masked_positions[torch.topk(confidence[batch_idx, masked_positions], k=1).indices]
```

`confidence` when `λ=0` is a probability in `[0, 1]`. When `λ>0`, `_resolve_cv_confidence` returns:

```python
cv_score = base_logp + λ · visual_gain
```

which is in log-probability space (mean ≈ 0, span ~±4 given `causal_clip=4`). The `≥0.9` compare fires almost never, so the fallback branch runs every step → **1 token commits per NFE** → DCD's fast multi-token-per-step commit is silently annihilated.

Empirical measurement from NPZ dumps:

| config | raw_conf ≥ 0.9 | cv_score ≥ 0.9 (buggy) | (would-be) cv_score ≥ log(0.9) |
|---|---|---|---|
| mask λ=0.5 | 79.4% | **0.55%** | 82.5% |
| shuffle λ=0.5 | 86.0% | 0.62% | 88.9% |
| random_mask λ=0.5 | 76.7% | 32.3% | 78.1% |

**Fix 1 applied**: rewrite as `cv_score = raw_conf * exp(λ · visual_gain)` (probability-space).

### 5.2 B2 — Double gumbel-noise argmax mismatch (LATENT, not triggered)

`dcd_decode_text_cv` samples `x0_cand`:

```python
logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)   # noise sample #1
x0_cand = torch.argmax(logits_with_noise, dim=-1)
cv_conf = _resolve_cv_confidence(logits, drop_logits, x0_cand, config, step_idx)  # score for x0_cand
```

then `_pick_transfer` internally re-samples:

```python
logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)   # noise sample #2
x0 = torch.argmax(logits_with_noise, dim=-1)                                    # possibly ≠ x0_cand
```

When `temperature > 0` the two argmax draws will disagree → `cv_conf` scores a token that never gets committed. Currently OK because `MMaDA-MixCoT-CV-DCD` uses `dcd_temperature = 0.0` (identity on `add_gumbel_noise`). Silent trap for any future run with `temperature > 0`.

### 5.3 B3 — `mask_id` is not a neutral image (SEVERE)

```python
if strategy == "mask":
    x_drop[:, s:e] = config.mask_id   # 126336 -- text mask token
```

Replacing 1024 image tokens with the text mask token yields an input the model **has never seen during training**. `drop_logp` is therefore not "log-prob when image is neutral" — it is "log-prob under an out-of-distribution input". `visual_gain = base_logp - drop_logp` is not a valid causal effect estimate.

Empirical signature: `shuffle > mask > random_mask` in the sweep, in the opposite direction of what causal-effect strength would predict.

**Fix 3 applied**: added `image_drop='neutral'`. `MMaDA.__init__` now encodes a mid-gray image via the VQ tokenizer (`Image.new('RGB', (res, res), (127,127,127))`), producing 1024 in-distribution VQ codes cached in `MMaDADecodeConfig.neutral_image_tokens`. `_build_dropped_image` splices these into the visual span for the drop forward.

Verified: gray image collapses to 53 unique codes out of 1024 (uniform gray → same VQ code for same patch), first codes `[126356, 131282, 130770, 126612, 130706, 126612, 130706, 130708, ...]`, all in valid image-token space.

### 5.4 B4 — Only commit ORDER, never commit TOKEN (ARCHITECTURAL)

Decode step:

```python
logits = model(x).logits
logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
x0 = torch.argmax(logits_with_noise, dim=-1)          # ← token picked from BASE logits
cv_conf = ...                                          # score for x0
transfer_index = threshold(cv_conf)                    # decides WHICH positions commit
x[transfer_index] = x0[transfer_index]                 # commits the base-picked token
```

If `argmax(base_logits) = "Grapefruit"` (wrong, language-prior-driven), CV-DCD can only choose "commit `Grapefruit` now" or "commit `Grapefruit` later" — it cannot swap it for `"an apple pie"`. This is why CV-DCD is fundamentally unable to repair the degenerate cases at the base-logit level; it can only defer them.

Fixing this requires **logit-level blending** (classifier-free-guidance / contrastive decoding), applied BEFORE `argmax`:

```python
blended = logits + λ · (logits - drop_logits)          # PMI / CFG-style
x0 = torch.argmax(blended, dim=-1)
```

Left as an unfinished path in this report.

---

## 6. Fix 1 + Fix 3 Rerun (mini-sweep, λ=0.5 × {mask, shuffle, neutral})

After applying Fix 1 (probability-space cv_score) and Fix 3 (neutral VQ tokens), we reran a 3-config minisweep on LLaVABench, GPT scored:

| config | overall | Δ vs baseline | mean_score | avg_len |
|---|---|---|---|---|
| DCD baseline | 44.1 | — | 3.67 | 384 |
| OLD λ=0.5 mask (buggy) | 37.1 | -7.0 | 3.13 | 330 |
| OLD λ=0.5 shuffle (buggy) | 41.1 | -3.0 | 3.45 | 356 |
| **NEW λ=0.5 mask (fixed)** | **26.7** | **-17.4** | 2.37 | 344 |
| **NEW λ=0.5 shuffle (fixed)** | **27.4** | **-16.7** | 2.42 | 368 |
| **NEW λ=0.5 neutral (fixed)** | **26.6** | **-17.5** | 2.32 | 355 |

**Fix 1 was mechanically correct but semantically wrong — the sweep got worse, not better.**

### Root cause of the regression

With Fix 1, `cv_score` now correctly passes the 0.9 threshold (82.5% of positions for mask λ=0.5). This restores DCD's multi-token-per-step commit throughput. **But** it also does something DCD's design never intended: promoting `raw_conf < 0.9` tokens above threshold whenever their `visual_gain > 0`.

DCD's `raw_conf ≥ 0.9` gate is a "commit only when the model is confident given the current partial context" invariant. Under CV-DCD-fixed, an uncertain token can bypass this gate if the image happens to slightly prefer it. Committing an uncertain token places it into a sequence where surrounding positions are still masked; the model's global consistency breaks; **subsequent tokens fit around this premature commit**, producing:

- `"riding a bicycle bicycle through a city street"`
- `"wearing a yellow yellow jacket"`
- `"a refrigerator-organized with items items items items items items items items items items items"`
- `"Sub serene sereneies"` (Subway → collapsed to noise)
- `"The made of"` (a 326-char answer degenerated to 3 words)

Repetition-rate diagnostic:

| config | mean consec-repeat rate | frac samples with rep | max rate |
|---|---|---|---|
| baseline | 0.00% | 0% | 0% |
| OLD mask (buggy, top-1 fallback) | 0.68% | 3.3% | 40% |
| **NEW mask (fixed)** | **6.65%** | **20%** | **96%** |
| **NEW neutral (fixed)** | 3.60% | 20% | 97% |

### But Fix 1 DID help the specific target group

Per-sample deltas of `NEW mask (fixed)` vs baseline, stratified by baseline score:

| baseline group | n | mean Δ | notable cases |
|---|---|---|---|
| degenerate (score ≤ 2) | 26 | **-0.23** | #12: `"It"` → `"The Lion King"` (1 → **7**), #48: `"My Joke"` → `"My joke is fake!"` |
| middle (3-5) | 18 | -2.00 |  |
| good (≥ 6) | 16 | -2.25 | #7: 9→2, #54: 6→1, #50: 6→1, #41: 6→1 (collapse to `"The made of"`) |

**The intervention lands on the right group** (degenerate samples improve on average, occasionally by +6 points) **but simultaneously damages the ~35% good group** where DCD was already correct. The two effects roughly cancel on the degenerate subset and net-negative on everything else.

This is exactly the outcome predicted by §3 (uniform λ on a heterogeneous failure population).

---

## 7. What We Learned

1. **The original CV-DCD hypothesis is 1/3 correct**: about 32% of LLaVABench failures have a measurable attention deficit toward image (Cohen d = -2.47, extremely large). But the remaining 2/3 do not.
2. **Uniform-λ CV-DCD is fundamentally incompatible with a heterogeneous failure population**. It helps the target subgroup and hurts everything else.
3. **Even after fixing the scale bug (B1), CV-DCD as currently designed breaks DCD's confidence-gating invariant**, causing repetition-collapse on ~20% of samples.
4. **`mask` is not a valid causal control** — it's out-of-distribution. `neutral` (VQ-encoded gray image) is the theoretically correct choice; the code now supports both.
5. **The commit-order-only architecture (B4)** is the deepest issue — CV-DCD cannot repair base-logit argmax errors. A true fix requires logit-level blending, which is essentially classifier-free guidance / contrastive decoding.

---

## 8. Two Viable Next Directions (Not Implemented)

### 8.1 Demote-only CV (safe)

```python
visual_gain = (base_logp - drop_logp).clamp(min=-config.causal_clip, max=0.0)  # upper clamp = 0
cv_score = raw_conf * torch.exp(config.causal_lambda * visual_gain)             # ≤ raw_conf always
```

Semantics: image can only VETO a would-be commit (`raw_conf ≥ 0.9` demoted to `< 0.9`). It cannot promote a low-`raw_conf` token past the gate. Preserves DCD's confidence invariant.

Expected outcome: near-baseline overall, degenerate NOT rescued (because those need PROMOTION, not veto), but no regression.

### 8.2 Logit-level contrastive decoding (real fix)

```python
blended = logits + λ · (logits - drop_logits)          # (1+λ)·conditional  − λ·null
x0 = torch.argmax(blended, dim=-1)
raw_conf = softmax_of_x0(blended)                       # gate uses blended
```

Semantics: changes WHICH token wins argmax. Direct implementation of PMI/CFG decoding. Preserves DCD's threshold gate (blended `raw_conf` still ≥ 0.9 to commit). Can genuinely swap `"Grapefruit"` for `"an apple pie"` if image support is strong.

Expected outcome: real improvement on degenerate, possible over-boost on already-correct samples. Requires λ tuning per benchmark.

### 8.3 Recommendation

If the goal is a **safe production knob** → 8.1.
If the goal is **validating the visual-grounding-for-decoding thesis** → 8.2. This is the same intervention as Contrastive Decoding (Li et al. 2022), which has literature backing.

---

## 9. Code Changes in This Iteration

| File | Change |
|---|---|
| `models/mmada_decode.py` | Fix 1: `cv_score = raw_conf * exp(λ · visual_gain)` (probability-space) |
| `models/mmada_decode.py` | Fix 3a: added `image_drop_strategy='neutral'` branch + `MMaDADecodeConfig.neutral_image_tokens` field |
| `vlmeval/vlm/mmada/mmada.py` | Fix 3b: `_populate_neutral_image_tokens()` VQ-encodes a gray reference at load time |
| `attention_analysis/full60_attention_analysis.py` | New: n=60 attention analysis across all LLaVABench samples |
| `attention_analysis/full60_per_sample_corrected.json` | New: per-sample aggregates (used for §3 tables) |
| `scripts/run_cvdcd_min3.sh` | New: minimal 3-config sweep runner for Fix validation |

---

## 10. Files & Artefacts Produced

```
docs/
  cv_dcd_report_v1.md               # design & plan (pre-experiment)
  cv_dcd_report_v2.md               # this file

attention_analysis/
  full60_attention_analysis.py      # n=60 attention analysis (this iteration)
  full60_raw_stats.json             # raw per-(layer,step) rows across all 60 samples
  full60_per_sample.json            # per-sample aggregates (initial, buggy groups)
  full60_per_sample_corrected.json  # per-sample aggregates (post score correction)
  full60_auto_classification.csv    # sample → {degenerate,normal,error_long,borderline}
  cv_debug_dumps/                   # NPZ dumps used in §2 (per-commit raw_conf, cv_score)

outputs/cvdcd_sweep/
  cvdcd_llava_20260705_032934/                            # ORIGINAL 9-config sweep (buggy)
  cvdcd_min3_fix1_neutral_20260705_224750/                # Fix 1 + Fix 3 mini-sweep

scripts/
  run_cvdcd_llavabench.sh          # original 9-config sweep runner
  run_cvdcd_min3.sh                # new 3-config Fix-validation runner
```

---

## 11. Reproducing the §6 mini-sweep

```bash
cd MMaDA/evaluation/VLMEvalKit
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/run_cvdcd_min3.sh
# Auto-triggers GPT scoring during run.py evaluation phase.
# Results appear in outputs/cvdcd_sweep/cvdcd_min3_*/.../MMaDA-MixCoT-CV-DCD_LLaVABench_openai_result.xlsx
```

---

## 12. Reproducing the §3 attention analysis

```bash
cd MMaDA/evaluation/VLMEvalKit
python attention_analysis/full60_attention_analysis.py
# reads rendered PNG heatmaps in rendered_all_layers_v3/
# writes full60_raw_stats.json + full60_per_sample.json
# ~15 minutes of PNG processing on CPU
```

---

## 13. Open Questions

- Would 8.2 (logit-level contrastive) actually help, or would the (1+λ) boost push language-prior-CORRECT commits into over-confident errors?
- Is the 32% under-attend subgroup a stable population, or does it shift across benchmarks? (Only tested on LLaVABench; MMBench / ScienceQA untested with CV-DCD.)
- Is the "commit uncertain tokens too early" failure mode (§6 root cause) specific to CV-DCD, or a general concern with any DCD confidence-modifier?
