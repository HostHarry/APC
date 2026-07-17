# CV-DCD v3.1 Design: APC-Style Contrastive Decoding for MMaDA

> **Purpose**: Align v3 with the standard Contrastive Decoding recipe from
> Li et al. 2022, O'Brien & Lewis 2023, and Chuang et al. 2023 (DoLa).
> Replaces the ad-hoc `base-conf gate` in v3 with the well-tested
> **Adaptive Plausibility Constraint (APC)**.
>
> Supersedes `cv_dcd_v3_gated_contrastive_design.md`.
> Follows `cv_dcd_report_v2.md` (empirical findings).
> Generated: 2026-07-06

---

## 0. Why v3.1 (What Changed from v3)

After reading the CD literature more carefully, one of v3's design choices was ad-hoc and needs replacement:

| Aspect | v3 (my ad-hoc design) | v3.1 (literature-aligned) |
|---|---|---|
| Where the guard sits | per-**position** gate: `softmax(base).max() < cv_gate` | per-**token** filter: `logits[i] >= max_k logits[k] + log(α)` |
| Terminology | `cv_gate` | Adaptive Plausibility Constraint (APC), threshold α |
| Default value | `cv_gate=0.9` (I made it up) | `α=0.1` (Li 2022, O'Brien 2023, Chuang 2023 all use this) |
| Blend strength | `causal_lambda ∈ {0.1, 0.3, 0.5}` (guessed) | `β=0.5` (literature default across many benchmarks) |
| What it prevents | over-boost on already-correct positions | (1) implausible-token false positive, (2) confident-easy-token false negative |
| Literature support | none | Li et al. 2022, O'Brien & Lewis 2023, Chuang et al. 2023 |

**The APC does something my per-position gate could not**: it protects against nonsense-token selection (false positive) even at low-confidence positions. This is important because for our "degenerate" case, base can be uncertain, and a strong drop signal on a random garbage token could win the argmax without APC.

---

## 1. The Recipe (from Li 2022 / O'Brien 2023 / DoLa 2023)

### 1.1 Formula in logit space

Let $s_e^{(i)}$ = base logits (with image), $s_a^{(i)}$ = drop logits (image ablated). Then:

$$
V_{\text{valid}} = \left\{ j \in V : s_e^{(j)} \geq \log \alpha + \max_{k \in V} s_e^{(k)} \right\}
$$

$$
s_{\text{CD}}^{(i)} = \begin{cases} (1+\beta) \cdot s_e^{(i)} - \beta \cdot s_a^{(i)}, & i \in V_{\text{valid}} \\ -\infty, & i \notin V_{\text{valid}} \end{cases}
$$

Then argmax on $s_{\text{CD}}$, and DCD's raw_conf gate applies to `softmax(s_CD)[argmax]`.

### 1.2 Recommended defaults

- **α = 0.1** — keep tokens whose base probability is at least 10% of the top-1 probability
- **β = 0.5** — moderate contrastive strength (this is our `causal_lambda`)

Both papers report these defaults are robust across many benchmarks (α insensitive as long as β < 1).

### 1.3 Two failure modes APC guards against

Directly quoted from DoLa §2.3:

> **False positive**: an implausible token with an extremely low score may be rewarded with a high score after contrast, due to the unstable low probability range on these implausible tokens.
>
> **False negative**: when the model is very confident about an easy decision, the output probability of a high-score token does not change much and results in low scores after contrast, so we need to force the model still select from these high-score tokens.

Both modes are addressed by restricting argmax to $V_{\text{valid}}$.

### 1.4 Probabilistic interpretation

$$
p_{\text{CD}}(i) \propto p_e(i) \cdot \left( \frac{p_e(i)}{p_a(i)} \right)^{\beta}, \quad i \in V_{\text{valid}}
$$

- As $\beta \to 0$: recovers base model (no contrastive effect)
- As $\beta \to \infty$: collapses to $\arg\max_i (p_e / p_a)$ (original Li 2022 CD)

---

## 2. Implementation for MMaDA

### 2.1 New helper (replaces `_apply_gated_contrastive`)

Add to `models/mmada_decode.py`:

```python
import math

def _apply_cd_style(
    logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
    config: MMaDADecodeConfig,
) -> torch.Tensor:
    """CD-style contrastive decoding with adaptive plausibility constraint (APC).

    Implements the O'Brien & Lewis 2023 formulation:
        V_valid = { j : logits_j >= max_k(logits_k) + log(alpha) }
        cd_logits = (1+beta) * logits - beta * drop_logits    (inside V_valid)
                    -inf                                       (outside V_valid)

    Equivalent to `logits + beta * (logits - drop_logits)` inside V_valid.

    References:
      - Li et al. 2022, "Contrastive Decoding: Open-ended Text Generation
                          as Optimization" (arXiv:2210.15097)
      - O'Brien & Lewis 2023, "Contrastive Decoding Improves Reasoning in
                                Large Language Models" (arXiv:2309.09117)
      - Chuang et al. 2023, "DoLa: Decoding by Contrasting Layers Improves
                             Factuality in Large Language Models" (arXiv:2309.03883)
    """
    if drop_logits is None or config.causal_lambda == 0.0:
        return logits

    log_alpha = math.log(config.cv_alpha)                             # e.g., log(0.1) = -2.303
    max_base = logits.amax(dim=-1, keepdim=True)                      # [B, T, 1]
    plausible = logits >= (max_base + log_alpha)                      # [B, T, V]

    blended = logits + config.causal_lambda * (logits - drop_logits)  # (1+beta)*e - beta*a

    neg_inf = torch.finfo(blended.dtype).min
    return torch.where(plausible, blended, torch.full_like(blended, neg_inf))
```

### 2.2 Config additions

```python
@dataclass
class MMaDADecodeConfig:
    ...
    # v3.1 APC-style contrastive decoding
    cv_alpha: float = 0.1                      # APC threshold (literature default)
    cv_mode: str = "cd_apc"                    # {cd_apc, cd_naive, legacy_score, off}
    ...
```

- `cd_apc`: v3.1 default (APC + contrastive blending)
- `cd_naive`: blending without APC (for ablation, ~expects worse due to false positives)
- `legacy_score`: v2's `raw_conf * exp(λ·gain)` path (regression reference)
- `off`: identity (base DCD)

`causal_lambda` is reused as O'Brien's $\beta$ (contrastive strength). Default value **0.5** (was previously 0.5 in the existing wrapper, so no change to defaults).

### 2.3 Replacement pattern for the 4 CV decode loops

For each of `dcd_decode_text_cv`, `dcd_decode_text_cv_prefix_cache`, and the two mid-block iterations in the cache variants:

**Before** (v2 / Fix 1):

```python
logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
x0_cand = torch.argmax(logits_with_noise, dim=-1)
cv_conf = _resolve_cv_confidence(logits, drop_logits, x0_cand, config, step_idx)

x0, transfer_index = _pick_transfer(
    logits, config, mask_index, x,
    confidence_override=cv_conf,
    debug_records=debug_records, step_idx=step_idx,
)
```

**After** (v3.1):

```python
if config.cv_mode == "cd_apc":
    effective_logits = _apply_cd_style(logits, drop_logits, config)
elif config.cv_mode == "cd_naive":
    effective_logits = _apply_naive_contrastive(logits, drop_logits, config)
elif config.cv_mode == "legacy_score":
    # v2 Fix 1 path -- kept for regression testing only
    ...
    # (same as before, uses confidence_override)
    x0, transfer_index = _pick_transfer(logits, config, mask_index, x,
                                          confidence_override=cv_conf,
                                          debug_records=debug_records, step_idx=step_idx)
    ...
    continue    # skip the new-code path below
else:  # "off"
    effective_logits = logits

x0, transfer_index = _pick_transfer(
    effective_logits, config, mask_index, x,
    debug_records=debug_records, step_idx=step_idx,
)
```

Notes:
- The `x0_cand` intermediate is gone (fixes v2 B2 automatically)
- `confidence_override` no longer used in the new path
- `_pick_transfer` computes raw_conf internally from `effective_logits`, so DCD's threshold gate is applied to the CD-shifted distribution — exactly what we want

### 2.4 Environment variables

```python
# In mmada.py __init__:
self.cv_alpha = float(os.getenv('MMADA_CV_ALPHA', 0.1))
self.cv_mode = os.getenv('MMADA_CV_MODE', 'cd_apc')
# (self.cv_causal_lambda remains as beta)
```

---

## 3. Case Analysis Under v3.1

Re-examining the 4 cases from v3 with the APC recipe:

### 3.1 Case A — successful rescue (LLaVABench #12)

```
base logits:  {"It":   3.5, "The": 2.5, "This": 2.0}    → base_probs {"It":0.65, "The":0.24, "This":0.11}
drop logits:  {"It":   4.0, "The": 1.5, "This": 2.0}
Δ:            {"It":  -0.5, "The":+1.0, "This": 0.0}

APC check (α=0.1):
  threshold = log(0.1) = -2.303 (added to max_base = 3.5)
  keep tokens with logit >= 3.5 - 2.303 = 1.197
  V_valid = {"It", "The", "This"}  (all pass)

Blended (β=0.5):
  "It":   3.5 + 0.5·(-0.5) = 3.25
  "The":  2.5 + 0.5·(+1.0) = 3.00
  "This": 2.0 + 0.5·(0.0)  = 2.00

argmax = "It"    ← WITHIN β=0.5 range, blend not strong enough to flip here

For β=1.0:
  "It":  3.5 + 1.0·(-0.5) = 3.00
  "The": 2.5 + 1.0·(+1.0) = 3.50    ← wins
```

**Trade-off learned**: β=0.5 (literature default) may not always flip clear language-prior errors. β=1.0 is more aggressive but risks Case D. For our MMaDA-degenerate use case, we may want to try **both β=0.5 and β=1.0** in Phase A.

### 3.2 Case B — confident language prior error

```
base_probs: {"My": 0.94, "The": 0.05, "Your": 0.03}
APC:  threshold = 0.1 × 0.94 = 0.094
      V_valid = {"My"}  ← only one token survives APC filter

argmax necessarily = "My" (still wrong)
```

**Cannot be rescued by any α ∈ [0.05, 0.15]**. Case B is an inherent limitation of CD-family methods. Out of scope for v3.1.

### 3.3 Case C — misleading drop signal

If neutral drop retains too much visual info, `Δ ≈ 0` everywhere, blended ≈ base, no effect. Sweep may reveal by comparing `drop=neutral` vs `drop=shuffle` — if they perform similarly, drop signal is not carrying information.

### 3.4 Case D — blend flips a correct base

```
base logits: {"and": 4.0, "or": 3.5, "with": 3.0}    → base_probs {"and":0.44, "or":0.27, "with":0.17}
drop logits: {"and": 4.2, "or": 2.5, "with": 3.5}
Δ:           {"and":-0.2, "or":+1.0, "with":-0.5}

APC (α=0.1): threshold = 0.1 × 0.44 = 0.044
      V_valid = {"and", "or", "with"}  (all pass)

Blended (β=0.5):
  "and":  4.0 + 0.5·(-0.2) = 3.90
  "or":   3.5 + 0.5·(+1.0) = 4.00    ← wins (bad: base was right)
  "with": 3.0 + 0.5·(-0.5) = 2.75

argmax = "or"    ← flipped correct base to wrong

DCD threshold: softmax(blended) → P("or") ≈ 0.41 < 0.9 → defer
```

**DCD threshold catches Case D** — the blended distribution isn't confident enough for the flipped token to commit right away. On the next step, context may evolve to disambiguate. In some fraction of samples this may still fail, but DCD's deferral provides real protection.

---

## 4. Ablation Design

### 4.1 Phase A: 3-config core test (~1.5h, LLaVABench 60)

| Config | β (causal_lambda) | α (cv_alpha) | drop | Purpose |
|---|---|---|---|---|
| Baseline (DCD) | — | — | — | Reference (44.1 overall) |
| A1 | 0.5 | 0.1 | neutral | Literature default (main candidate) |
| A2 | 1.0 | 0.1 | neutral | More aggressive contrast (see Case A analysis) |
| A3 | 0.5 | 0.1 | shuffle | Sanity: shuffle should be similar to neutral |

Success criterion: A1 or A2 shows any of:
- Overall ≥ baseline (break-even proves no catastrophic damage)
- Degenerate subgroup +1.5 mean score (mechanism validated)
- Ideal: overall +1.0pp AND degenerate +2.0

### 4.2 Phase B: ablation of APC vs no-APC (~1h)

| Config | Purpose |
|---|---|
| A1 (repeat) | reference |
| B1: `cd_naive` (α → 0, no APC) | prove APC matters |
| B2: `cd_apc` with α=0.05 | more permissive APC |
| B3: `cd_apc` with α=0.3 | more restrictive APC |

Expected: A1 > B1 (APC helps by preventing false positives). B2/B3 test sensitivity.

### 4.3 Phase C: full sweep (~4h, contingent on Phase A/B success)

| Dim | Values |
|---|---|
| β | {0.3, 0.5, 0.75, 1.0} |
| α | {0.05, 0.1, 0.2} |
| drop | {neutral, shuffle} |

Best config from A/B feeds into MMBench-DEV-EN (~250) for statistical power.

---

## 5. Instrumentation

Add these to debug records (when `MMADA_CV_RETURN_DEBUG=1`):

```python
debug_records.append({
    "step": step_idx, "batch": b, "position": p,
    "x0_token": int(x0[b, p]),
    "base_argmax": int(logits[b, p].argmax()),
    "argmax_changed": bool(x0[b, p] != logits[b, p].argmax()),  # KEY metric
    "raw_conf_base": float(F.softmax(logits[b, p], dim=-1).max()),
    "raw_conf_effective": float(F.softmax(effective_logits[b, p], dim=-1).max()),
    "v_valid_size": int(plausible[b, p].sum()),                 # APC bites
    "visual_gain_at_x0": float(logits[b, p, x0] - drop_logits[b, p, x0])
                          if drop_logits is not None else 0.0,
})
```

Diagnostic thresholds after Phase A:
- **`argmax_changed` rate**: healthy ≈ 5-15% overall, 30-50% on blended positions.
- **`v_valid_size` distribution**: median ≈ 20-100 tokens (out of vocab_size ≈ 134k). If median < 5, APC too tight. If > 1000, APC not biting.
- **`raw_conf_effective` vs `raw_conf_base`**: differences show where CD is reshaping the distribution.

---

## 6. Handling of v2's Four Bugs Under v3.1

| Bug | v2 status | v3.1 status |
|---|---|---|
| **B1** (log vs prob scale) | Fixed in v2 Fix 1 but caused new problems | Fixed by using `softmax(effective_logits)` directly — natively [0,1] |
| **B2** (double gumbel argmax) | Latent | Fixed — one argmax inside `_pick_transfer` on `effective_logits` |
| **B3** (mask_id not neutral) | Fixed in v2 Fix 3 | Kept (`image_drop='neutral'` used) |
| **B4** (only reorders, no token change) | Root architectural issue | **Fixed** — argmax on `effective_logits` genuinely can pick different tokens |

Additional new failure modes CD literature identifies:

| Mode | v3 (no APC) | v3.1 (APC) |
|---|---|---|
| Nonsense-token false positive (base 0.001 / drop 0.0001 → high CD score) | ❌ can trigger | ✅ blocked by APC |
| Easy-token false negative (high base prob diluted by CD noise) | ⚠️ possible | ✅ V_valid keeps only expert-top → forces staying with easy tokens |

---

## 7. Expected Outcomes (Revised Priors)

Based on:
- O'Brien 2023 achieved +8pp on GSM8K with same recipe on LLaMA (1B amateur, 65B expert)
- DoLa achieved +21pp on TruthfulQA MC2 by using same-model layer contrast
- Our attention analysis (v2 §3): only 32% of LLaVABench failures are attention-related, cap on rescue coverage
- MMaDA is diffusion-based, not autoregressive; CD literature is autoregressive → some transfer risk

**Realistic prior for LLaVABench overall**:
- Optimistic: baseline +2.0pp (from ~44 to ~46)
- Expected: baseline +0.5 to +1.5pp
- Pessimistic: break-even (no catastrophic regression like Fix 1)

**Realistic prior for degenerate subgroup** (baseline score ≤ 3):
- Optimistic: mean score +2.5 (some samples 1→5+)
- Expected: mean score +1.0 to +2.0
- Pessimistic: mean score flat

---

## 8. Non-Goals

- Rescue **Case B** (confident language-prior errors) — fundamental limitation of CD.
- Beat SOTA — this is a validation of the mechanism, not a leaderboard chase.
- Modify DCD threshold gate — v3.1 works WITHIN DCD's confidence gate (using softmax of effective_logits).
- Cross-benchmark generalization for now — LLaVABench first; MMBench in Phase C.

---

## 9. Actionable Checklist

Priority-ordered, each item is ~15 min - 1h:

1. **[45min]** Add `_apply_cd_style` and `_apply_naive_contrastive` helpers to `models/mmada_decode.py`. Add `cv_alpha`, `cv_mode` to `MMaDADecodeConfig`. Update 4 CV decode loops to dispatch on `cv_mode`. Preserve v2 Fix 1 code path as `cv_mode='legacy_score'`.
2. **[15min]** Add `MMADA_CV_ALPHA` and `MMADA_CV_MODE` env-var reads to `mmada.py`.
3. **[15min]** CPU-only unit tests: `_apply_cd_style` on hand-crafted logits. Verify APC masking and blending numerics.
4. **[15min]** Backward compatibility check: `cv_mode='legacy_score'` reproduces v2 Fix 1 NPZ dumps (identical numerics).
5. **[10min]** GPU 5-sample sanity: mixed indices (degenerate + good + borderline), no repetition, no crash.
6. **[1.5h]** Phase A sweep: 3 configs × 60 LLaVABench.
7. **[15min]** Extended summarizer to slot in v3.1 configs; compute per-baseline-group deltas and `argmax_changed` stats.
8. **Decision point**: if Phase A shows degenerate lift and no crash, proceed to Phase B ablations (§4.2). Else diagnose via debug metrics (§5).

**Total to "have Phase A results in hand": ~4 hours** including the 1.5h inference + 30min GPT scoring + 30min analysis.

---

## 10. Related Work Cited in Design

1. **Li et al. 2022** — Contrastive Decoding: Open-ended Text Generation as Optimization. arXiv:2210.15097.
   - Original CD, introduces α-mask (APC) and log-prob-difference score.
2. **O'Brien & Lewis 2023** — Contrastive Decoding Improves Reasoning in Large Language Models. arXiv:2309.09117.
   - Simplified logit-space formulation `(1+β)·s_e - β·s_a`; default α=0.1, β=0.5.
   - Shows CD helps reasoning tasks (HellaSwag, GSM8K), not just open-ended.
3. **Chuang et al. 2023** — DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models. arXiv:2309.03883.
   - Same APC recipe, but contrasts layers within one model instead of two models.
   - Explains APC's two roles (false-positive and false-negative protection).
   - Future direction: could we do intra-MMaDA-layer contrast to save one forward?
4. **Sanchez et al. 2023** — Stay on topic with Classifier-Free Guidance. arXiv:2306.17806.
   - CFG for language models; same math as CD, framed as generative-model guidance.

---

## 11. What's Different From The Original CV-DCD Proposal

Our original CV-DCD (Chinese proposal doc) had `cv_score = raw_conf + λ·visual_gain`
applied as a re-ranking on `raw_conf`. This is fundamentally different from v3.1:

| Property | Original CV-DCD | v3.1 CD-style |
|---|---|---|
| Where the visual signal enters | after argmax, as a score modification | before argmax, as a logit shift |
| Can change committed token? | No | **Yes** |
| Related literature | Ad-hoc | Contrastive Decoding family |
| Handles Case A (uncertain LM error) | ❌ | ✅ |
| Handles Case B (confident LM error) | ❌ | ❌ (inherent to CD) |
| DCD compatible | ⚠️ (v2 showed scale issues) | ✅ (softmax stays in [0,1]) |
| Fights language priors | ❌ (only defers them) | ✅ (can swap the argmax) |

The **spirit** of the original proposal ("use image ablation to detect language-prior errors") is preserved; the **mechanism** is upgraded to what the literature has shown to work.

---

*End of v3.1 design.*
