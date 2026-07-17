# Code Bundle — MANIFEST

Self-contained snapshot of the **CV-DCD Direction B (defer_only)** implementation,
its VLMEvalKit patches, run scripts and unit tests as of 2026-07-08. All files here
are direct copies of the source-of-truth files in the parent repo (paths given below).

## Layout

```
code/
├── MANIFEST.md                     # this file
├── models/
│   ├── mmada_decode.py             # main DCD/CV-DCD decode entry point
│   ├── cv_v32_apc.py               # legacy v3.2 CD+APC (deprecated; kept for history)
│   ├── cv_common/                  # shared utilities used by both v4 directions
│   │   ├── __init__.py
│   │   ├── image_drop.py           # image-drop strategies (mask/shuffle/random/text_only/neutral)
│   │   ├── log_prob.py             # memory-efficient logp_of_tokens
│   │   ├── paired_forward.py       # helper to run base + drop forwards together
│   │   └── types.py                # shared dataclasses / enums
│   └── defer_only/                 # v4 Direction B (defer-only CV)
│       ├── __init__.py
│       ├── dispatcher.py           # orchestrates: base argmax + veto + λ blend
│       └── veto.py                 # hard / mult / min / soft veto formulas
├── vlmeval_patches/                # modifications to VLMEvalKit (drop-in replacements)
│   ├── vlm__mmada__mmada.py        # MMaDA wrapper (POPE prompt, env-var config surface)
│   ├── dataset__image_yorn.py      # POPE/MME MD5 checks bypassed
│   ├── dataset__image_mcq.py       # MMBench_DEV_EN MD5 check bypassed
│   ├── dataset__image_base.py      # MMADA_SKIP_LOCALIZE env var (sandbox mp.Pool workaround)
│   ├── dataset__image_caption.py   # + CHAIRDataset (long-form hallucination bench)
│   ├── dataset____init__.py        # registers CHAIRDataset in IMAGE_DATASET
│   ├── dataset_utils__chair.py     # canonical CHAIR port (Rohrbach 2018 → Py3)
│   └── chair_data/synonyms.txt     # verbatim from LisaAnne/Hallucination
├── scripts/
│   ├── hf_to_vlmeval_tsv.py        # HF parquet -> VLMEval TSV converter
│   ├── build_chair_tsv.py          # MSCOCO val2017 -> CHAIR.tsv converter
│   ├── compare_e0_parity.py        # byte-level pred diff + score delta analyzer
│   ├── run_defer_only_smoke.sh, *_v2.sh, *_v3.sh, *_v4.sh
│   ├── run_defer_only_phase_d_focused.sh
│   ├── run_defer_only_matrix_v5.sh
│   ├── run_defer_only_e0_parity.sh
│   ├── run_defer_only_deterministic_benchmarks.sh
│   ├── run_defer_only_visual_benchmarks.sh   # POPE/MME/MMBench_DEV_EN sweep
│   └── run_defer_only_chair.sh               # NEW: CHAIR long-form hallucination sweep
└── tests/
    └── test_defer_only.py          # 32 unit tests; run on CPU
```

## Per-file responsibilities

### `models/mmada_decode.py` (~874 lines)

Original path: `models/mmada_decode.py`. Contains:
- `MMaDADecodeConfig` dataclass (all CV-DCD knobs: `causal_lambda`, `cv_mode`,
  `defer_veto_type`, `defer_tau`, `defer_gain_type`, `image_drop_strategy`, ...).
- `_pick_transfer`, `_confidence_from_logits`: plain-DCD confidence & scheduling.
- `dcd_decode_text_dual_cache`: plain DCD path (no CV intervention).
- `dcd_decode_text_cv_dual_cache`: CV-DCD path. Dispatches to `defer_only` when
  `cv_mode == 'defer_only'`, else legacy CD paths.
- Key invariant (verified by unit tests): `defer_only(causal_lambda=0)` produces
  bit-identical confidence and transfer decisions as `dcd_decode_text_dual_cache`.

### `models/defer_only/`

- **`dispatcher.py`**: the "argmax + veto + λ blend" orchestrator.
  - `base_conf` uses `F.softmax(logits.to(fp64))` to match `_confidence_from_logits`
    bit-for-bit (fixes the 2026-07-07 E0-parity bug).
  - `eff_conf = (1 - λ) * base_conf + λ * veto_conf`; λ is intervention strength.
  - When `λ == 0` this MUST equal plain DCD (byte parity).
- **`veto.py`**:
  - `hard`  : `eff = base_conf if gain ≥ τ else 0`
  - `mult`  : `eff = base_conf * clamp(1 + gain/|τ|, 0, 1)`  (initial design)
  - `min`   : `eff = min(base_conf, ratio_from_gain(gain, τ))`
  - `soft`  : `eff = base_conf * exp(-β * clamp(τ - gain, min=0))`  (smooth)
  - Also defines `VetoType` enum and `compute_visual_gain(logit|logprob, x0)`.

### `models/cv_common/`

- **`image_drop.py`**: build drop-image token ids given a strategy
  (`mask` / `shuffle` / `random_mask` / `text_only` / `neutral`).
- **`log_prob.py`**: `logp_of_tokens(logits, tokens)` — memory-efficient
  gather+logsumexp so we don't need `to(fp64)` on the full vocabulary tensor
  (avoids OOM on large batches).
- **`paired_forward.py`**: `paired_forward_logits(model, x, drop_x)` — runs
  base and drop-image forwards, returns their logits.
- **`types.py`**: shared dataclasses (e.g. `CVDebugRecord`).

### `models/cv_v32_apc.py` (deprecated)

Legacy v3.2 CD (contrastive decoding) + APC (adaptive plausibility constraint)
path. Superseded by v4 (Direction B here + Direction A design in `report/cv_dcd_v4_design.md`).
Kept purely for reproducibility of the v3.2 phase-B sweep, not used in v5 matrix
or later experiments.

### `vlmeval_patches/`

Files are named with the flattened path so it's obvious where to drop them back.
When integrating into a fresh VLMEvalKit clone:
- `vlm__mmada__mmada.py`   → `vlmeval/vlm/mmada/mmada.py`
- `dataset__image_yorn.py` → `vlmeval/dataset/image_yorn.py`
- `dataset__image_mcq.py`  → `vlmeval/dataset/image_mcq.py`
- `dataset__image_base.py` → `vlmeval/dataset/image_base.py`

**Key changes** (relative to upstream VLMEvalKit at the fork point):
- `mmada.py`
  - Reads all CV-DCD knobs from `MMADA_*` env vars (env-driven experiment sweeps).
  - Adds POPE to the "Please answer yes or no." Y/N prompt branch.
  - Adds `MMADA_INDICES` env for subsampling and `MMADA_CV_RETURN_DEBUG` for NPZ dumps.
- `image_yorn.py`: removed `MME` and `POPE` MD5 entries (we use local
  TSVs converted from HF parquets by `hf_to_vlmeval_tsv.py`; upstream MD5
  triggers a re-download attempt to opencompass.openxlab.space that fails
  behind our sandbox).
- `image_mcq.py`: removed `MMBench_DEV_EN` MD5 entry (same reason).
- `image_base.py`: honour `MMADA_SKIP_LOCALIZE=1` to bypass the LOCALIZE
  multiprocessing.Pool step (needs semaphores that the sandbox blocks; only
  triggered on TSVs >1 GiB).
- `image_caption.py`: added ``CHAIRDataset`` (``TYPE = 'Caption'``). ``evaluate()``
  loads MSCOCO ``instances_val2017.json`` (via ``CHAIR_COCO_ANN`` env var) and
  writes ``*_chair_score.csv`` + ``*_chair_details.jsonl``.
- `dataset/__init__.py`: registered ``CHAIRDataset`` in ``IMAGE_DATASET``.
- `dataset/utils/chair.py`: **new**. Faithful Python-3 port of
  `LisaAnne/Hallucination/utils/chair.py` (Rohrbach 2018). Substitutes
  `pattern.en.singularize` (broken on py3.9+) with
  `nltk.stem.WordNetLemmatizer`; `nltk.word_tokenize` retained. Ships the
  canonical `synonyms.txt` verbatim in `chair_data/`. Provides both canonical
  GT (instance-masks ∪ caption-derived objects, default) and a `CHAIR_STRICT_GT=1`
  ablation (instance-masks only).
  Requires NLTK data (`punkt punkt_tab wordnet omw-1.4`) — install to
  workspace-local `nltk_data/` via
  `python -m nltk.downloader -d $VLMEVAL_ROOT/nltk_data punkt punkt_tab wordnet omw-1.4`
  (chair.py auto-inserts this path into `nltk.data.path`).

### `scripts/`

- **`hf_to_vlmeval_tsv.py`**: Downloads-free parquet-to-TSV converter.
  Handles POPE (Y/N), MME (Y/N with 14 categories), MMBench_DEV_EN (MCQ).
  Supports `--shuffle-seed` so `MMADA_INDICES=0..N-1` gives a
  category-stratified subset.
- **`compare_e0_parity.py`**: For a pair of run directories, compares raw
  prediction text byte-by-byte and reports per-sample score deltas. Used to
  verify `defer_only λ=0 == plain DCD` byte-identically (60/60 hit on
  LLaVABench, 199/199 on ScienceQA_VAL).
- **Run scripts (`run_defer_only_*.sh`)** — each drives one experiment phase:
  - `smoke.sh` … `smoke_v4.sh`: incremental smoke tests over gain-type / drop
    strategy / τ sweeps (Phase B).
  - `phase_d_focused.sh`: 60-sample re-run of best config with strict same-env
    baseline for the LLaVABench sanity check.
  - `matrix_v5.sh`: 60-sample (λ, τ) 7-config matrix that established the
    top-performers (E1/E4/E6).
  - `e0_parity.sh`: strict head-to-head B0 vs E0 vs E4 vs E6 on LLaVABench.
  - `deterministic_benchmarks.sh`: 200-sample sweep on ScienceQA_VAL and
    MathVision_MINI (rule-scored, low noise).
  - `visual_benchmarks.sh`: 200-sample sweep on POPE / MME / MMBench_DEV_EN
    (visual/hallucination benchmarks).

### `tests/test_defer_only.py`

CPU unit tests. Notable cases:
- `test_hard_veto_semantics`: canonical `gain ≥ τ` gate.
- `test_lambda_zero_base_conf_bit_identical_to_plain_dcd`: fp64 parity.
- `test_lambda_zero_stress_bit_identical_plain_dcd`: 100-position stress.
- `test_lambda_zero_with_temperature_bit_identical_plain_dcd`: ensures
  temperature multiplication does not break the parity.
- Assortment of `mult`, `min`, `soft` veto edge cases.

Run:
```bash
cd /home/user/dcd/MMada_DCD/MMaDA
/home/user/anaconda3/envs/mmada/bin/python -m pytest tests/test_defer_only.py -q
# expected: 32 passed
```

## Reproducing an experiment

The scripts assume the parent MMaDA project layout. To use this bundle in a
fresh checkout:

1. Drop `models/cv_common/`, `models/defer_only/`, `models/mmada_decode.py`
   and `models/cv_v32_apc.py` into the parent `models/` directory.
2. Copy `vlmeval_patches/*` back to their canonical VLMEvalKit paths (see
   the mapping above).
3. Put `scripts/*` into `evaluation/VLMEvalKit/scripts/`.
4. Ensure `LMUData/` has the TSVs (or use `scripts/hf_to_vlmeval_tsv.py` to
   convert HF parquets — no opencompass mirror needed).
5. Run one of the shell scripts. Each is idempotent (VLMEval skips already-
   computed samples if `--reuse` is passed).
