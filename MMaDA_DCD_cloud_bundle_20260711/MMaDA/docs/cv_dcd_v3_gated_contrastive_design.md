# CV-DCD v3 Design: Gated Contrastive Decoding

> **Purpose**: Redesign CV-DCD after v2 diagnostics falsified the original mechanism.
> Successor to `cv_dcd_report_v1.md` (design) and `cv_dcd_report_v2.md` (findings).
> Generated: 2026-07-06

---

## 0. TL;DR

**Problem** (from v2 §5 diagnosis): the original CV-DCD only modifies commit **timing** via a re-scored confidence; it cannot modify the committed **token**, because `argmax` runs on `base_logits` and the CV score is applied downstream. When base_logits pick a wrong language-prior token, CV-DCD is powerless.

**New mechanism**: apply the visual causal signal **at the logit level, before argmax**:

$$
\text{blended} = \text{logits} + \lambda \cdot (\text{logits} - \text{drop\_logits})
$$

Then re-argmax. This can genuinely swap `"Grapefruit"` → `"apple"` when image evidence disagrees with language prior.

**Guarding against over-boost on already-correct samples**: apply blending only where the base distribution is not confident (per-position `softmax(logits).max() < gate`, e.g., gate = 0.9). Confident positions bypass blending → good samples untouched.

**Expected outcome** (calibrated by v2 attention analysis):
- Degenerate subgroup (≈32% of LLaVABench): partial rescue — **+1.5 to +3.0 mean score on the target subgroup**, mainly for "uncertain language-prior" errors (Case A in §7).
- Good subgroup (≈33%): near-zero change (gate closed).
- Overall LLaVABench: **+1.0 to +2.5pp vs DCD baseline (44.1)**, i.e., target ~45-47pp.
- Not a "beat SOTA" ambition — a proof-of-concept that the CV-DCD idea can work in a properly-designed form.

---

## 1. What v2 Findings Rule In and Rule Out

| v2 finding | Implication for v3 |
|---|---|
| 32% of failures show under-attention (Cohen d=-2.47) | Target group exists → some intervention worth doing |
| 68% show no attention deficit | Uniform-λ intervention is wrong; need a gate |
| Fix 1 (probability-space cv_score) crashed sweep | Cannot boost low-`raw_conf` tokens over threshold |
| B4 (only reorders commits, cannot change token) | Any real fix MUST act on logits before argmax |
| `mask` drop is OOD; `neutral` VQ-encoded gray is in-distribution | Use `image_drop='neutral'` from Fix 3 |
| Sweep sensitivity `shuffle > mask` | Confirms drop-quality matters; `neutral` should replace both |

The four in-code bugs from v2 are addressed as follows:

| v2 bug | v3 resolution |
|---|---|
| **B1**: cv_score/threshold scale mismatch | Eliminated. `raw_conf` used by DCD threshold is now `softmax(blended)[argmax]`, natively in [0,1]. |
| **B2**: double gumbel-noise mismatch | Eliminated. Only one argmax draw (inside `_pick_transfer`), on the blended logits. |
| **B3**: `mask_id` is not a neutral image | Kept fix. `image_drop='neutral'` uses VQ codes of a gray image (from Fix 3). |
| **B4**: commit-order-only architecture | **Directly addressed by v3**. Blending happens before argmax. |

---

## 2. Core Mechanism

### 2.1 Two-step decoding pass at each DCD iteration

```
Step 1: forward BOTH the real input and the image-dropped input
        → logits, drop_logits    (2 model forwards, or 1 with dual-cache trick)

Step 2: per-position gate — is the base distribution already confident?
        base_top1 = softmax(logits).max(dim=-1)              [B, T]
        is_uncertain = base_top1 < config.cv_gate            [B, T]

Step 3: build blended logits
        blended_all = logits + λ · (logits − drop_logits)    [B, T, V]
        blended = where(is_uncertain, blended_all, logits)   [B, T, V]

Step 4: standard DCD pick_transfer, but from the blended distribution
        x0 = argmax(blended)                                  [B, T]
        raw_conf = softmax(blended)[x0]                       [B, T]
        transfer_index = raw_conf >= 0.9                      [B, T]
        x[transfer_index] = x0[transfer_index]
```

### 2.2 Why this preserves DCD's "commit when confident" invariant

The `raw_conf` used by the threshold gate is computed from **the same distribution** that generated `x0` (`blended`). This is different from Fix 1 in v2, where `cv_score` was a modified quantity applied on top of a raw_conf from a different distribution.

Consequence: any token committed by v3 satisfies `blended_softmax[x0] >= 0.9`. This is an honest per-position confidence in the blended model, not a boost.

### 2.3 Why the gate protects good samples

A "good" sample position typically has base_top1 ≈ 0.95+ (model laser-focused on a specific token). For any such position, `is_uncertain = False` → `blended = logits` → same decode as DCD.

A "degenerate risk" position has base_top1 in the 0.5-0.85 zone (model unsure). Only these positions get blending, which is where CV can actually add value.

---

## 3. Mathematical Notes

### 3.1 Equivalence with Classifier-Free Guidance (CFG)

$$
\text{blended} = \text{logits} + \lambda \cdot (\text{logits} - \text{drop\_logits}) = (1+\lambda) \cdot \text{logits} - \lambda \cdot \text{drop\_logits}
$$

This is the **CFG guidance formula** at $w = 1 + \lambda$. When $\lambda = 0$ we recover the base model; $\lambda = 1$ is CFG scale = 2 (a common setting); higher λ is more aggressive.

### 3.2 Relationship to PMI decoding

$$
\log P_{\text{blended}}(y) - Z = (1+\lambda) \log P_{\text{base}}(y) - \lambda \log P_{\text{drop}}(y) = \log P_{\text{base}}(y) + \lambda \cdot [\log P_{\text{base}}(y) - \log P_{\text{drop}}(y)]
$$

The bracketed term is exactly the point-wise mutual information (PMI) between token $y$ and image $I$:
$$
\text{PMI}(y; I | \text{context}) \approx \log P(y | I, \text{ctx}) - \log P(y | I_{\text{null}}, \text{ctx})
$$

**So v3 = CFG-augmented decoding = PMI-based decoding**. Both formulations are equivalent to what Li et al. (2022) call "Contrastive Decoding".

### 3.3 λ tuning is different from v1/v2

v1/v2's `cv_score = raw_conf * exp(λ · gain)`: `λ = 0.5` means moderate confidence-space boost/veto.

v3's `blended = logits + λ · Δ`: `λ = 0.5` means CFG guidance scale = 1.5 (relatively conservative).

Rough correspondence based on CFG literature: `λ_v3 ∈ [0.1, 0.5]` is typical; `1.0` is aggressive; `>1.5` risks over-boost. Recommend sweeping `λ ∈ {0.1, 0.3, 0.5}` for v3.

### 3.4 Gate threshold `cv_gate`

- **0.5** (aggressive): most positions get blended. High rescue coverage but high risk of damaging good samples.
- **0.7** (medium): only truly-uncertain positions blend. Balanced.
- **0.9** (conservative, recommended): only DCD-deferral-triggering positions blend. Preserves nearly all DCD good behaviors.

At `cv_gate = 0.9`:
- Positions committed by DCD anyway (raw_conf ≥ 0.9) → never blended, they commit as usual.
- Positions DCD would defer (raw_conf < 0.9) → blended, might argmax differently, may pass or fail threshold on blended distribution.

**This is the semantically-cleanest gate**: "only intervene where DCD would defer anyway".

---

## 4. Implementation

### 4.1 File-level changes

| File | Change |
|---|---|
| `models/mmada_decode.py` | Add `_apply_gated_contrastive()`; simplify 4 CV decode loops. |
| `models/mmada_decode.py::MMaDADecodeConfig` | Add `cv_gate: float = 0.9`. |
| `evaluation/VLMEvalKit/vlmeval/vlm/mmada/mmada.py` | Add `cv_gate` env-var read (`MMADA_CV_GATE`) and pass to config. |
| `evaluation/VLMEvalKit/scripts/run_cvdcd_v3.sh` | New script for v3 sweep. |
| `evaluation/VLMEvalKit/attention_analysis/summarize_cv_dcd_phase1.py` | Extend to summarize v3 outputs. |

Old code paths (`_cv_scores_from_paired_logits`, `_resolve_cv_confidence`) stay in place for backward compatibility with old runs but become **dead code** for v3 (guarded by `cv_mode` field, see §4.5).

### 4.2 New function

```python
def _apply_gated_contrastive(
    logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
    config: MMaDADecodeConfig,
) -> torch.Tensor:
    """Blend base and image-dropped logits where the base is uncertain.

    On positions where softmax(logits).max() >= config.cv_gate, return logits
    unchanged (the base model is already confident, do not perturb).

    On positions where softmax(logits).max() < config.cv_gate, apply the
    contrastive shift `logits + lambda * (logits - drop_logits)`.
    This is equivalent to classifier-free guidance at scale (1 + lambda)
    or to PMI-guided decoding.

    When drop_logits is None (cv_stride > 1 skip step) or lambda == 0,
    the function is an identity.
    """
    if drop_logits is None or config.causal_lambda == 0.0:
        return logits
    with torch.no_grad():
        base_probs = F.softmax(logits, dim=-1)
        base_top1 = base_probs.amax(dim=-1, keepdim=True)     # [B, T, 1]
        is_uncertain = base_top1 < config.cv_gate             # [B, T, 1] bool
    blended = logits + config.causal_lambda * (logits - drop_logits)
    return torch.where(is_uncertain, blended, logits)
```

### 4.3 Replacement pattern for each of the 4 CV decode loops

**Before** (repeated in `dcd_decode_text_cv`, `dcd_decode_text_cv_prefix_cache`, `dcd_decode_text_cv_dual_cache`, and the mid-block iterations of the two cache variants):

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

**After**:

```python
effective_logits = _apply_gated_contrastive(logits, drop_logits, config)
x0, transfer_index = _pick_transfer(
    effective_logits, config, mask_index, x,
    debug_records=debug_records, step_idx=step_idx,
)
```

Notes:
- The intermediate `x0_cand` is gone. Only one argmax happens now (inside `_pick_transfer`).
- The `confidence_override` parameter is not used → `_pick_transfer` computes `raw_conf` internally from `effective_logits`, which is exactly what we want.
- `debug_records` now needs a minor update to log `base_top1` and `blended_top1` diagnostics (rather than `cv_score`).

### 4.4 Config additions

```python
@dataclass
class MMaDADecodeConfig:
    ...
    # v3 gated contrastive parameters
    cv_gate: float = 0.9        # blend only where softmax(base).max() < cv_gate
    cv_mode: str = "gated_contrastive"   # {gated_contrastive, contrastive_naive, legacy_score, off}
    ...
```

- `gated_contrastive`: v3 default, gate at cv_gate
- `contrastive_naive`: always blend (no gate), for ablation
- `legacy_score`: fall back to v1/Fix1 formula (for reproducibility of v2 report)
- `off`: identity, equivalent to base DCD

### 4.5 Backward-compatibility dispatch

Inside each CV loop:

```python
if config.cv_mode == "gated_contrastive":
    effective_logits = _apply_gated_contrastive(logits, drop_logits, config)
    x0, transfer_index = _pick_transfer(effective_logits, config, mask_index, x, ...)
elif config.cv_mode == "contrastive_naive":
    effective_logits = _apply_naive_contrastive(logits, drop_logits, config)
    x0, transfer_index = _pick_transfer(effective_logits, config, mask_index, x, ...)
elif config.cv_mode == "legacy_score":
    # v1 / Fix1 path -- for regression tests only
    logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
    x0_cand = torch.argmax(logits_with_noise, dim=-1)
    cv_conf = _resolve_cv_confidence(logits, drop_logits, x0_cand, config, step_idx)
    x0, transfer_index = _pick_transfer(logits, config, mask_index, x,
                                          confidence_override=cv_conf, ...)
else:  # "off" or unknown
    x0, transfer_index = _pick_transfer(logits, config, mask_index, x, ...)
```

### 4.6 Environment variable knobs

To keep the sweep script flexible, add these env-var reads in `mmada.py`:

```python
self.cv_gate = float(os.getenv('MMADA_CV_GATE', 0.9))
self.cv_mode = os.getenv('MMADA_CV_MODE', 'gated_contrastive')
```

---

## 5. Hyperparameter Sweep Plan

### 5.1 Phase A — validate the mechanism works (small, ~1.5h)

```
6 configs: λ ∈ {0.1, 0.3, 0.5}  ×  image_drop ∈ {neutral, shuffle}
gate = 0.9
mode = gated_contrastive
cv_stride = 1

Dataset: LLaVABench full 60 samples
Judge:  GPT-4o-mini (existing setup)
```

**Success criterion (Phase A)**:
- Non-catastrophic: overall no worse than baseline − 1pp on any config.
- Mechanism-active: **degenerate subgroup mean score** improves ≥ +1.0 on at least one config.
- Ideal: overall +0.5pp on at least one config.

If Phase A shows degenerate lift AND overall break-even, proceed to Phase B. Otherwise diagnose (see §7).

### 5.2 Phase B — hyperparameter tuning (~4h)

Contingent on Phase A success:

```
9 configs: λ ∈ {best-from-A, 0.4, 0.6}  ×  gate ∈ {0.7, 0.8, 0.9}  ×  drop = neutral
Dataset: LLaVABench + MMBench_DEV_EN (or ScienceQA_VAL, whichever runs faster)
```

### 5.3 Phase C — ablations to confirm mechanism attribution

```
- v3 with cv_mode=contrastive_naive       ← ablate the gate
- v3 with cv_mode=legacy_score            ← reproduce v2 regression as sanity
- v3 with image_drop=mask (same λ, gate)  ← check whether Fix 3 (neutral drop) matters
```

Expected findings:
- `contrastive_naive` should show larger degenerate lift but larger good-sample damage → validates the gate as a protector.
- `legacy_score` should reproduce v2 regression → confirms Fix 1 was a red herring.
- `image_drop=mask` should be worse than `neutral` → confirms Fix 3.

---

## 6. Success Criteria & Failure Diagnostics

### 6.1 Positive outcomes (in decreasing order of significance)

1. **Overall +1.0pp**: v3 clearly beats DCD → publishable positive result.
2. **Overall break-even + degenerate +2.0**: v3 hits the target subgroup without hurting others → mechanism validated, tuning needed.
3. **Overall break-even + no crash**: no repetition or grammatical breaks (unlike Fix 1) → engineering victory but no science claim.

### 6.2 Negative outcomes and next actions

| Outcome | Diagnosis | Next |
|---|---|---|
| Overall drops (like v2 Fix 1) | Gate is too permissive OR drop signal unreliable | Raise gate to 0.95, or switch to attention-based drop |
| Overall break-even, degenerate flat | λ too small OR degenerate errors are of the "confident language prior" type (Case B in §7) | Raise λ to 0.7-1.0; separately try `contrastive_naive` |
| Overall +0.3-0.5, degenerate slight lift | Small win, need statistical significance | Extend to MMBench (250+) for power |
| High repetition rate | Blended distribution has multi-modal argmax → non-determinism | Add temperature=0 forcing, or top-k=1 constraint |

### 6.3 Instrumentation to add

In `_pick_transfer` when `config.return_debug=True`:

```python
debug_records.append({
    "step": step_idx, "batch": b, "position": p,
    "x0_token": int(x0[b, p]),
    "raw_conf_base": float(base_probs[b, p].max()),         # NEW
    "raw_conf_blended": float(blended_probs[b, p].max()),   # NEW
    "was_blended": bool(is_uncertain[b, p]),                 # NEW
    "argmax_changed_by_blend": bool(base_argmax[b, p] != x0[b, p]),  # NEW
})
```

The `argmax_changed_by_blend` counter is the most direct diagnostic: it measures how often the mechanism actually swapped tokens. A healthy run should have:
- ~30-50% of blended positions have argmax_changed=True (blending is doing something)
- ~5-15% of ALL positions have argmax_changed=True (most tokens are confident anyway)

If argmax_changed ≈ 0%, blending is producing minor logit adjustments only → λ too small or drop signal too weak.
If argmax_changed ≈ 100% at blended positions, drop signal is dominating → λ too large.

---

## 7. Detailed Failure-Mode Analysis

### 7.1 Case A — successful rescue (real: LLaVABench #12)

Scenario: Uncertain language-prior error. `base_top1 ≈ 0.55-0.85`, argmax = wrong token, image evidence points to correct token.

```
base logits:  {"It":   3.5, "The": 2.5, "This": 2.0}   → argmax "It", raw_conf 0.65
drop logits:  {"It":   4.0, "The": 1.5, "This": 2.0}   → Δ("It")=−0.5, Δ("The")=+1.0

gate check:   0.65 < 0.9 → blend
blended λ=1.0: {"It":  3.0, "The": 3.5, "This": 2.0}   → argmax "The"

x0 changes to "The"; blended_raw_conf ≈ 0.53 → defer (correct)
Next iteration fills in "Lion King" via subsequent blendings.
```

Real result: `"It"` (score 1) → `"The Lion King"` (score 7).

### 7.2 Case B — confident language prior (real: LLaVABench #48 approx)

Scenario: base is confidently wrong. `base_top1 > 0.9`, gate closes.

```
base logits:  {"My": 5.0, "The": 2.0, "Your": 1.5}    → argmax "My", raw_conf 0.94
drop logits:  {"My": 4.8, "The": 3.0, "Your": 1.6}    → Δ("My")=+0.2, Δ("The")=−1.0

gate check:   0.94 > 0.9 → NO blend
argmax stays "My" → wrong
```

**Not rescuable by gated contrastive.** Mitigation:
- Lower gate to 0.7 → this position now blends, but now many good positions also blend.
- Add auxiliary "image_signal_strong" trigger: also blend if `max(drop_logits) - min(drop_logits)` differs strongly from base's spread, indicating image contributes real info. (Not in v3 scope; noted for future.)

### 7.3 Case C — misleading drop signal

Scenario: `neutral` VQ codes are not truly information-free. Model reacts to gray-image encoding as if it were content.

```
Real problem:  Neutral gray image tokens ≈ constant across positions.
               drop_logits ≈ "how would model respond if input were 'a photo of a gray blob'"
               This IS an in-distribution query but not "no image".
```

**Signature**: `Δ = base - drop` is small everywhere → blending has minimal effect → v3 degenerates to base DCD.

**Mitigation** (out of scope for v3, noted): switch drop to attention-based masking (set attention weights to visual tokens to −inf during drop forward). This is a bigger change to the model call.

### 7.4 Case D — blend flips a correct token

Scenario: base is right (`base_top1 = 0.8`), drop is misleadingly suggestive of a wrong token.

```
base logits:  {"and": 4.0, "or": 3.5, "with": 3.0}    → argmax "and", raw_conf 0.60
drop logits:  {"and": 4.2, "or": 2.5, "with": 3.5}    → Δ("or")=+1.0, Δ("with")=−0.5

gate check:   0.60 < 0.9 → blend
blended λ=1.0: {"and": 3.8, "or": 4.5, "with": 2.5}   → argmax "or"  ← flipped correct → wrong
```

**Mitigation**: This is a genuine trade-off. The gate is not sufficient to catch these; only stronger `neutral` drop or lower λ helps.

**Empirical calibration**: run Phase A first with `debug_records`; look at `argmax_changed_by_blend` counter for positions where the sample is in the "good" baseline group. If flip rate is > 10% in good samples, gate/λ too aggressive.

---

## 8. Non-Goals for v3

- **Not** trying to beat SOTA on LLaVABench.
- **Not** attempting to fix confident-language-prior errors (Case B) — that needs a different intervention.
- **Not** touching image generation (only text decoding).
- **Not** changing the drop strategy (`neutral`, `shuffle`, `mask` all supported via existing Fix 3 code).

---

## 9. Concrete Implementation Steps

Ordered checklist (roughly ½ day of coding + 1 day of experiments):

1. **Code** (~1h):
   - [ ] Add `cv_gate`, `cv_mode` fields to `MMaDADecodeConfig` with defaults.
   - [ ] Add `_apply_gated_contrastive()` and `_apply_naive_contrastive()` helpers in `mmada_decode.py`.
   - [ ] Refactor `dcd_decode_text_cv`, `dcd_decode_text_cv_prefix_cache`, `dcd_decode_text_cv_dual_cache` to dispatch on `config.cv_mode`.
   - [ ] Preserve old `cv_score` code path under `cv_mode='legacy_score'` for reproducibility.
   - [ ] Wire `MMADA_CV_GATE` and `MMADA_CV_MODE` env vars in `mmada.py`.

2. **Smoke tests** (~15min):
   - [ ] CPU-only unit test: `_apply_gated_contrastive` with hand-crafted `logits` + `drop_logits`; check gate masking works.
   - [ ] Backward-compat test: `cv_mode='legacy_score'` produces same NPZ dumps as v1/Fix1 code (regression check).

3. **GPU 3-sample sanity** (~5min):
   - [ ] `run_cvdcd_v3.sh` with `MMADA_INDICES=0,5,12,25,54` (mix of degenerate + good baseline samples).
   - [ ] Inspect outputs manually: no repetition, sentences complete, plausible answers.

4. **Phase A sweep** (~1.5h): 6 configs, GPT scored via existing `score_cv_dcd_sweep.py` or auto-triggered by `run.py`.

5. **Analysis** (~30min):
   - [ ] `summarize_cv_dcd_phase1.py` extended to slot in v3 configs.
   - [ ] Compute per-baseline-group deltas (degenerate / mid / good).
   - [ ] Extract `argmax_changed_by_blend` from debug NPZs.

6. **Decision point**: proceed to Phase B (§5.2) if success criteria met (§6.1 outcome ≥ 2), otherwise diagnose via §6.2 table.

---

## 10. Open Design Questions

1. **Should the gate use `base_top1` or `base_top1 − base_top2` (margin)?**
   - `base_top1` captures "how sure of THE choice"
   - `top1 - top2` captures "how sure THIS choice is better than the runner-up"
   - Second is more principled for our case (we want to blend when the runner-up is competitive), but requires a second sort. Recommend starting with `base_top1` and revisiting if flip rate looks off.

2. **Should λ decay across DCD steps?**
   - Early steps: sparse context → language prior dominates → higher λ might help.
   - Late steps: rich context → base logits mostly correct → lower λ preserves.
   - Not in v3 scope, but a natural v4 extension.

3. **Should we cache blended logits across steps within a block?**
   - The dual-cache path may allow reusing the drop_logits contribution when only a few positions change per step, saving compute.
   - Not in v3; can be a perf optimization later.

4. **Should we do PMI-based drop instead of VQ-neutral drop?**
   - Alternative to `neutral` VQ: pass an attention_bias that zeros out visual tokens, forcing the model to "reason without image".
   - Cleaner semantically ("no image") but requires new model call plumbing.
   - Not in v3; noted as a §7.3 mitigation direction.

---

## 11. Traceable Artifacts (Existing)

- v2 attention analysis (all 60 LLaVABench samples): `attention_analysis/full60_per_sample_corrected.json`
- v2 NPZ debug dumps: `attention_analysis/cv_debug_dumps/*.npz`
- v2 mini-sweep results (Fix 1 crash): `outputs/cvdcd_sweep/cvdcd_min3_fix1_neutral_20260705_224750/`
- v2 report: `docs/cv_dcd_report_v2.md`

v3 artifacts (to be created after §9 checklist):
- Code diff patch: to be committed together as `cv_dcd_v3_gated_contrastive_impl.diff`
- New sweep dir: `outputs/cvdcd_sweep/cvdcd_v3_gated_*_TIMESTAMP/`
- Phase A summary CSV: `attention_analysis/cv_dcd_v3_phase_a_summary.csv`
- Final report: `docs/cv_dcd_report_v3.md` (following the v2 template)

---

## 12. Quick Reference: What Changes vs v2 Fix 1

| Aspect | v2 Fix 1 (`raw_conf * exp(λ·gain)`) | v3 (gated contrastive) |
|---|---|---|
| Where CV signal applies | On `confidence` value, AFTER argmax | On `logits`, BEFORE argmax |
| Can change committed token | ❌ No | ✅ Yes (via argmax on blended) |
| Preserves DCD confidence gate | ❌ No (bypasses via boosting) | ✅ Yes (uses `softmax(blended)`) |
| Damages good samples | ❌ Yes (~20% repetition) | ✅ No (gate closes on confident positions) |
| Targeted at degenerate | ✅ Yes | ✅ Yes |
| Related literature | Ad-hoc mix | Contrastive Decoding (Li et al. 2022), CFG |
| λ regime | 0.25-1.0 (mixed effect) | 0.1-0.5 (typical CFG range) |
| New hyperparameter | None | `cv_gate` (default 0.9) |

---

*End of v3 design.*
