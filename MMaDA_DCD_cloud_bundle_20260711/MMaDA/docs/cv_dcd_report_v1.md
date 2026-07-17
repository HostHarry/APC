# CV-DCD for MMaDA — Experiments Report v1

> Causal-Visual Deferred Commitment (CV-DCD): training-free decoding that calibrates
> token commitment with image-ablation causal evidence.
> Generated: 2026-07-05

## 1. Implementation Status

| Component | Path | Status |
|-----------|------|--------|
| CV-DCD decode kernels | `models/mmada_decode.py` | Done |
| Model dispatch | `models/modeling_mmada.py` | Done |
| VLMEvalKit glue | `vlmeval/vlm/mmada/mmada.py` | Done |
| Config entry | `vlmeval/config.py` → `MMaDA-MixCoT-CV-DCD` | Done |
| Smoke tests | `tests/test_cv_dcd_smoke.py` | Pass |
| LLaVABench sweep script | `scripts/run_cvdcd_llavabench.sh` | Ready |
| Debug dump (NPZ) | `attention_analysis/cv_debug_io.py` | Done |
| Phase-1 summarizer | `attention_analysis/summarize_cv_dcd_phase1.py` | Done |
| Phase-2 calibration | `attention_analysis/calibration_analysis.py` | Done |
| Token grounding judge | `attention_analysis/token_grounding_judge.py` | Done |

### Core formula

```
base_logp = log p(y_i | I, Q, x_t)
drop_logp = log p(y_i | I_drop, Q, x_t)
visual_gain = clamp(base_logp - drop_logp, -C, +C)
cv_score = base_logp + lambda * visual_gain
```

When `causal_lambda=0`, CV-DCD reduces to standard DCD (verified by smoke test).

### Usage

```bash
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_LAMBDA=0.5
export MMADA_CV_DROP=mask
bash scripts/run_cvdcd_debug.sh          # 5 samples
bash scripts/run_cvdcd_llavabench.sh     # full 9-config sweep
```

---

## 2. Hypothesis Validation (Pre-CV-DCD Evidence)

From existing DCD-DualCache LLaVABench 60 + attention analysis (`report.html`):

| Finding | Evidence | Supports CV-DCD? |
|---------|----------|----------------|
| Language-prior errors | Grapefruit/mangosteen, cable car/Space Needle, apples/mangosteen | **Yes** — high-conf, low visual gain |
| Degeneration (short output) | 16/60 samples, len≤10, conv 52.9% degen | Partial — needs min-tokens guard |
| Low image attention in degen | left_half_frac=0.510 vs 0.549 correct | **Yes** — visual unsupported commits |
| Long hallucination | 14/60, Top-5% concentration high but wrong | **Yes** — attention proxy fails; causal ablation needed |

**Conclusion**: CV-DCD targets long-hallucination and language-prior failures. Degeneration is a separate DCD-boundary issue.

---

## 3. Phase-1 Main Experiment

### 3.1 Baseline (completed)

Source: `outputs/MMaDA-MixCoT-DCD-DualCache/T20260613_G/`

| Split | N | Mean Score | Pass≥6 | Degen% | Long-Hall% | Mean Len |
|-------|---|------------|--------|--------|------------|----------|
| overall | 60 | 3.82 | 33.3% | 26.7% | 23.3% | 384.7 |
| complex | 28 | 4.21 | 46.4% | 25.0% | 17.9% | 518.2 |
| conv | 17 | 3.29 | 17.6% | 52.9% | 11.8% | 52.6 |
| detail | 15 | 3.67 | 26.7% | 0.0% | 46.7% | 511.9 |

### 3.2 CV-DCD sweep (pending GPU run)

Run: `bash scripts/run_cvdcd_llavabench.sh`

| Config | lambda | drop | Expected target |
|--------|--------|------|-----------------|
| lam0.25_mask | 0.25 | mask | conservative calibration |
| lam0.5_mask | 0.5 | default |
| lam1.0_mask | 1.0 | mask | aggressive visual veto |
| lam0.5_shuffle | 0.5 | shuffle | non-causal control |
| lam0.5_random_mask | 0.5 | random_mask | non-causal control |

After sweep, summarize:

```bash
python attention_analysis/summarize_cv_dcd_phase1.py \
  --sweep-dir outputs/cvdcd_sweep/<RUN_ID> \
  --baseline outputs/MMaDA-MixCoT-DCD-DualCache/T20260613_G \
  --output attention_analysis/cv_dcd_phase1_summary.csv
```

### 3.3 Success criteria

- Pass rate (≥6) improves ≥5pp on overall vs baseline
- Long-hallucination rate drops ≥5pp on detail/complex
- shuffle/random_mask ablations do NOT beat mask (causal evidence)

---

## 4. Phase-2 Calibration Diagnostics

### 4.1 Pipeline

1. Enable debug dump: `export MMADA_CV_RETURN_DEBUG=1`
2. Run inference → `attention_analysis/cv_debug_dumps/*.npz`
3. Token grounding labels: `python attention_analysis/token_grounding_judge.py --xlsx <result.xlsx>`
4. Calibration: `python attention_analysis/calibration_analysis.py --npz-dir <dir> --grounding-json <labels.json>`

### 4.2 Baseline grounding labels (heuristic, 325 tokens)

Generated from DCD predictions without GPT API (`--no-gpt`). Re-run with `OPENAI_API_KEY` for GPT-4o-mini judge.

### 4.3 Demo calibration (synthetic NPZ)

See `attention_analysis/calibration_out/calibration_diag.json`:
- raw_conf ECE: 0.267
- cv_score ECE: 0.203
- delta_ECE: +0.063 (CV better calibrated on demo data)

Plots: `attention_analysis/calibration_out/calibration_curves.png`, `grounding_vs_conf_scatter.png`

---

## 5. Ablation Matrix

| Ablation | Variable | Purpose |
|----------|----------|---------|
| lambda sweep | 0.25 / 0.5 / 1.0 | causal weight sensitivity |
| drop strategy | mask / shuffle / random_mask | causal vs non-causal |
| cv_stride | 1 / 2 / 4 | speed-quality tradeoff |
| causal_clip | 2 / 4 / 6 | gain saturation |

---

## 6. Known Limitations (MVP)

1. Image-token span mask only (no patch-level grounding)
2. No reasoning-debt or reversible commitment
3. DualCache drop path: 2x forward per step when lambda>0
4. `visual_token_start/end` auto-inferred from `<|soi|>/<|eoi|>`; manual override via `decode_config`

---

## 7. Next Steps

1. **Run GPU sweep**: `bash scripts/run_cvdcd_llavabench.sh`
2. **GPT-4 judge**: re-run `token_grounding_judge.py` with API key
3. **Compare per-sample**: diff CV-DCD vs DCD on indices {4,5,6,19,28} (language-prior failures)
4. **HallusionBench**: extend sweep to hallucination-focused benchmark
5. **Paper draft**: §3.1 baseline + §3.2 sweep results → ICLR fast-track

---

## Appendix: File Map

```
models/mmada_decode.py          # CV-DCD kernels + dispatch_cv_dcd_decode_text
models/modeling_mmada.py        # mmu_generate cv_dcd branch
vlmeval/vlm/mmada/mmada.py      # env vars + debug NPZ save
vlmeval/config.py               # MMaDA-MixCoT-CV-DCD
tests/test_cv_dcd_smoke.py      # regression tests
scripts/run_cvdcd_*.sh          # experiment runners
attention_analysis/
  cv_dcd_phase1_summary.csv     # metrics table
  cv_debug_io.py                # NPZ I/O
  token_grounding_judge.py      # Phase-2 labels
  calibration_analysis.py       # ECE/AUC plots
  summarize_cv_dcd_phase1.py    # aggregate results
```
