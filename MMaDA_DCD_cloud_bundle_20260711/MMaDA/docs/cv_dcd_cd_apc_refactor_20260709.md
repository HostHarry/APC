# CV-DCD `cd_apc` Refactor — Test Plan (2026-07-09)

## Motivation

The user provided a modified `mmada_decode.py` that unifies the "decoupled
selection + confidence + gate" pattern (previously only in `cd_apc_v32`) into
the default `cd_apc` / `cd_naive` modes. This resolves an implicit bug where
`cd_apc` was using `softmax(blended)[x0]` as commit confidence — inflated by
APC masking → premature commits of CD flips → damage.

**Key change**: `cd_apc` / `cd_naive` now:
1. Uses `x0_base` when `cv_gate_tau > 0` **and** `base_conf ≥ tau` (protect
   easy tokens from CD flips — flips are irreversible in masked diffusion).
2. Uses `cv_conf_source ∈ {base, min_base_blended, blended}` for commit
   ranking (default flipped to `min_base_blended`).
3. Falls back to plain base pick when `drop_logits is None` or
   `causal_lambda == 0.0` — guarantees byte-parity with plain DCD in these
   corner cases.

This makes `cd_apc` production-viable again, whereas before it was known to
hurt on visual benchmarks.

## What we've already verified

- ✓ All 74 unit tests pass after the refactor
  (`tests/test_cv_common.py`, `test_cv_v32.py`, `test_defer_only.py`,
   `test_cd_style_smoke.py`, `test_cv_dcd_smoke.py`).
- ✓ Byte-level parity smoke on synthetic tensors:
  - `cd_apc, λ=0` ≡ plain DCD.
  - `cd_apc, λ>0, drop_logits=None (cv_stride skip)` ≡ plain DCD.
  - `cd_naive, λ=0` ≡ plain DCD.
  - `cd_apc active (λ=0.5)` diverges from plain DCD (16/32 token diffs).
  - `cd_apc` with `cv_conf_source ∈ {base, blended, min_base_blended}` share
    the same `x0` — only commit ranking differs.

## Testable hypotheses

Given the 7-dataset证伪 of `defer_only`, we frame `cd_apc` similarly:

- **H1 (E0 parity, GPU)**: `cd_apc + λ=0` on real GPU produces byte-identical
  predictions to plain DCD (B0) on any dataset.
- **H2 (safety)**: New default `cd_apc` (`λ=0.5, α=0.1, min_base_blended,
  τ=0`) is no worse than plain DCD on visual benchmarks.
- **H3 (protection helps)**: Adding `cv_gate_tau > 0` reduces damage from CD
  flips. Sweep `τ ∈ {0.5, 0.9, 0.95, 0.99}`.
- **H4 (net upside on visual grounding)**: Some (`λ, α, τ`) combo actually
  improves POPE / MME / MMBench / CHAIR over plain DCD.

Prior belief: H1 & H2 should hold. H3 likely (mostly protective, so
somewhat closer to plain). H4 is the interesting question.

## Test phases

### Phase 1 — GPU E0 parity smoke (~5-8 min)

Verify **on real GPU** that `cd_apc, λ=0` is byte-identical to plain DCD.

- Dataset: LLaVABench (8 samples, fast).
- Configs:
  - `B0_plain_dcd` — `decode_strategy=dcd, cv_mode=off`
  - `E0_cd_apc_l0`  — `cv_mode=cd_apc, λ=0.0`
- Success: **8/8 predictions byte-identical** between B0 and E0.

Command:
```bash
MMADA_SAMPLE_N=8 bash scripts/run_cd_apc_e0_parity.sh
```

### Phase 2 — LLaVABench smoke on `cd_apc` variants (~30 min)

Verify (a) H1 on real GPU, (b) that variants actually diverge from B0, and
(c) get an early feel for GPT-scored ranking.

- Dataset: LLaVABench, 20 samples.
- Configs (5 total):
  - `B0_plain_dcd`
  - `E0_cd_apc_l0`         (λ=0, must equal B0)
  - `E1_cd_apc_default`    (λ=0.5, α=0.1, min_base_blended, τ=0)
  - `E2_cd_apc_gate090`    (λ=0.5, α=0.1, min_base_blended, τ=0.9)
  - `E3_cd_apc_gate095`    (λ=0.5, α=0.1, min_base_blended, τ=0.95)

Metrics: byte-diff vs B0 (# samples that changed), GPT-scored mean/std,
NFE per sample.

Command:
```bash
MMADA_SAMPLE_N=20 bash scripts/run_cd_apc_llavabench_smoke.sh
```

### Phase 3 — Deterministic Y/N + MCQ benchmarks (~4-5 h)

Test on POPE / MME / MMBench_DEV_EN (all deterministic scoring, low noise).

- 200 samples per dataset (same subsets as previous defer_only sweep,
  `hf_to_vlmeval_tsv.py --shuffle-seed 42`).
- Configs (from Phase 2 top-2 + baselines):
  - `B0_plain_dcd`
  - `E0_cd_apc_l0`         (parity anchor)
  - `E1_cd_apc_default`    (λ=0.5, α=0.1, min_base_blended, τ=0)
  - `E2_cd_apc_gate090`    (λ=0.5, α=0.1, min_base_blended, τ=0.9)
  - `E3_cd_apc_gate095`    (λ=0.5, α=0.1, min_base_blended, τ=0.95)

Command:
```bash
bash scripts/run_cd_apc_visual_benchmarks.sh
```

### Phase 4 — CHAIR long-form caption hallucination (~5-6 h)

- 200 val2017 images, `Please describe this image in detail.` prompt.
- Best 1-2 configs from Phase 3 + baseline.
  - `B0_plain_dcd`
  - Winner of Phase 3 (likely `E2_cd_apc_gate090`)
  - Optionally: `E1_cd_apc_default` for ungated comparison.

Command:
```bash
bash scripts/run_cd_apc_chair.sh
```

### Phase 5 (contingent) — Widen sweep if Phase 3-4 shows positive signal

If any config beats plain DCD by > noise on Phase 3-4, expand:
- λ ∈ {0.25, 0.5, 1.0, 2.0}
- α ∈ {0.01, 0.1, 0.3}
- τ ∈ {0, 0.5, 0.9, 0.95, 0.99}

Otherwise, mark `cd_apc` as also-null and consolidate the paper conclusion:
_"Neither logit-level CD nor confidence-level defer helps MMaDA — the base
DCD is already well-calibrated to visual grounding on 7 datasets."_

## Environment / knob mapping

| Knob                    | Env var                      | Meaning                                                                                     |
|-------------------------|------------------------------|---------------------------------------------------------------------------------------------|
| `cv_mode`               | `MMADA_CV_MODE`              | `off`(=plain) / `cd_apc` / `cd_naive` / `defer_only`                                        |
| `causal_lambda`         | `MMADA_CV_LAMBDA`            | CD blend strength β in `(1+β)base - β·drop`                                                 |
| `cv_alpha`              | `MMADA_CV_ALPHA`             | APC plausibility threshold — smaller = stricter (keeps more low-prob tokens plausible)      |
| `cv_conf_source`        | `MMADA_CV_CONF_SOURCE`       | `base` / `min_base_blended` / `blended` — which softmax feeds the DCD threshold             |
| `cv_gate_tau`           | `MMADA_CV_GATE_TAU`          | If `>0`, positions with `base_conf >= τ` bypass CD (keep base argmax + base conf)           |
| `image_drop_strategy`   | `MMADA_CV_DROP`              | `mask` / `shuffle` / `neutral` / `text_only` — how to build `x_drop`                        |

## Time budget

| Phase | Time (H100/RTX 4090) | Notes                                              |
|-------|----------------------|----------------------------------------------------|
| 1     | 5-8 min              | 8 samples × 2 configs                              |
| 2     | ~30 min              | 20 samples × 5 configs on LLaVABench               |
| 3     | ~4-5 h               | 200 × 3 datasets × 5 configs                       |
| 4     | ~5-6 h               | 200 × 3 configs, paired forward on long captions   |
| 5     | contingent           | up to +6 h if signal warrants                      |

**Total**: ~10-12 hours GPU (Phase 1-4).

## Success criteria

- **Must**: Phase 1 passes E0 parity (no regression from refactor).
- **Nice to have (H3)**: Some τ > 0 config matches or beats plain DCD on POPE
  overall F1 within ± 0.5 pp.
- **Breakthrough (H4)**: Some config beats plain DCD by ≥ 1 pp on **any**
  deterministic benchmark, and doesn't regress on the rest.

## Notes

- Scripts are new (`scripts/run_cd_apc_*.sh`), do **not** overwrite existing
  `defer_only` scripts.
- The 200-sample TSVs already exist for POPE / MME / MMBench_DEV_EN (built
  earlier with `hf_to_vlmeval_tsv.py --shuffle-seed 42`); no data prep needed.
- CHAIR TSV already exists (`LMUData/CHAIR.tsv`, 200 val2017 images).
