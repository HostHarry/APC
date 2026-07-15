# History-VCHD Cloud Bundle (2026-07-14)

Minimal self-contained bundle for running the **full History-VCHD** decoder
(VCHD dual-gate + Sparse History + CCAW) on the MMMU and MMBench benchmarks
in a cloud GPU environment. **No model weights, no dataset TSVs, no logs, no
attention dumps** are shipped — they are fetched or built on the target
machine.

## What's inside

```
history_vchd_cloud_20260714/
├── README.md                              # this file
├── .rsync-exclude                         # exclude list used to build the bundle
├── docs/
│   └── CD_APC_Adaptive_Decoding_Implementation_Guide_v3.md
├── scripts/
│   ├── setup_env.sh                       # install deps + editable VLMEvalKit
│   ├── prepare_datasets.sh                # HF parquet → LMUData TSV
│   └── run_history_vchd.sh                # main sweep entry point
└── MMaDA/
    ├── LICENSE, README.md, requirements.txt
    ├── models/                            # LLaDA + MMaDA + magvitv2 + mmada_decode
    │   ├── cv_common/                     # (needed by mmu_generate dispatch)
    │   └── defer_only/                    # (needed by mmu_generate dispatch)
    ├── decoding/                          # ⭐ VCHD core (new implementation)
    ├── training/                          # only prompting_utils + utils (imports from mmada.py)
    ├── tests/
    │   ├── test_vchd.py                   # dual-gate + adapter tests
    │   ├── test_vchd_history_ccaw.py      # sparse history + CCAW tests
    │   └── test_vchd_cache.py             # KV cache tests
    ├── docs/
    │   ├── vchd_cross_check_20260714.md   # hidden-issue audit vs MMMU/MMBench
    │   ├── experimental_results_20260713.md
    │   └── cv_dcd_latest_program_summary.md
    └── evaluation/VLMEvalKit/
        ├── run.py                         # main eval driver
        ├── setup.py, requirements.txt
        ├── vlmeval/                       # VLMEvalKit core (with mmada wrapper)
        ├── scripts/                       # includes hf_to_vlmeval_tsv.py, run_vchd_overnight_20260714.sh
        └── tools/
```

Total bundle size: ~7 MB. Everything above sits *before* pip install
(dependencies) and *before* first HF download (weights + datasets).

## Prerequisites on the cloud

* Linux + CUDA-capable GPU (A100 40 / 80 GB tested; anything ≥ 24 GB should
  fit MMaDA-8B at BF16 inference).
* NVIDIA driver + CUDA 12.1 wheels (`setup_env.sh` pins `torch==2.4.1+cu121`);
  edit the script if your image ships a different CUDA major.
* Python 3.10 / 3.11 (as required by transformers 4.46 and torch 2.4).
* Internet egress to HuggingFace (weights + datasets are pulled at run time).
* Disk budget:
  - ~16 GB for `Gen-Verse/MMaDA-8B-MixCoT` weights
  - ~1 GB for `showlab/magvitv2` VQ tokenizer
  - ~10 GB for `lmms-lab/MMBench_EN` + `lmms-lab/MMMU` parquets
  - ~5 GB for the converted TSVs
  - a few GB for per-sample VCHD ledger JSONs (if history_ccaw profile)

## Quick start (three commands)

```bash
# 0. Activate the target environment (conda or venv), then:
bash scripts/setup_env.sh          # pip install everything, sanity-check imports
bash scripts/prepare_datasets.sh   # ~15 min: download + convert MMMU / MMBench
bash scripts/run_history_vchd.sh   # ~10 h on 1× A100: History-VCHD × 2 datasets
```

The runner selects the dedicated `MMaDA-MixCoT-VCHD` registry entry. It
automatically uses `/root/autodl-tmp/MMaDA-8B-MixCoT` and
`/root/autodl-tmp/magvitv2` when those local directories exist; otherwise it
falls back to the Hugging Face IDs. Override this explicitly with
`MMADA_MODEL_PATH`, `MMADA_TOKENIZER_PATH`, and `MMADA_VQ_MODEL_PATH`.

Outputs land under:

```
MMaDA/evaluation/VLMEvalKit/outputs/<RUN_ID>/<DATASET>/…/*.xlsx
MMaDA/evaluation/VLMEvalKit/logs/<RUN_ID>/<DATASET>.log
MMaDA/evaluation/VLMEvalKit/logs/<RUN_ID>/<DATASET>_reports/vchd_report_<index>.json
```

The `_reports/` directory contains one ledger JSON per sample, including
`model_evaluations`, `threshold_commits / fallback_commits`,
`history_observations`, `ccaw_final_mask_capacity`, `mean_history_stability`
and the per-step trace (if `MMADA_VCHD_COLLECT_TRACE=1`).

## Custom sweeps

`run_history_vchd.sh` is thin — every knob is env-driven and can be
overridden. Examples:

```bash
# Only MMMU, larger commit window, tighter contrast gate
DATASETS="MMMU_DEV_VAL_FULL" \
MMADA_VCHD_MASK_CAPACITY=32 MMADA_VCHD_MAX_COMMIT=32 \
MMADA_VCHD_TAU_CONTRAST=0.95 \
RUN_ID=vchd_tight_ccaw \
bash scripts/run_history_vchd.sh

# Reproduce the overnight 7-run apples-to-apples sweep (original + b0 + e1 +
# vchd + history_ccaw × 2 datasets)
bash MMaDA/evaluation/VLMEvalKit/scripts/run_vchd_overnight_20260714.sh
```

Documented knobs (see `MMaDA/decoding/config.py` for defaults + validation):

| Env var | Default | Meaning |
|---|---:|---|
| `MMADA_VCHD_TAU_BASE` | 0.10 | base gate (raw visual prob on contrast token) |
| `MMADA_VCHD_TAU_CONTRAST` | 0.90 | contrast gate (`C·[1−ρ(1−T)]`) |
| `MMADA_VCHD_MASK_CAPACITY` | 16 | initial window size |
| `MMADA_VCHD_MAX_COMMIT` | 16 | max tokens committed per iteration |
| `MMADA_VCHD_FALLBACK_TO_RAW` | 0 | when window fallbacks, use raw argmax instead of CD-APC top-1 |
| `MMADA_VCHD_HISTORY` | 1 | enable sparse-history reliability correction |
| `MMADA_VCHD_HISTORY_TOP_V` | 8 | sparse history top-V truncation |
| `MMADA_VCHD_HISTORY_EMA_DECAY` | 0.7 | history EMA on old distribution |
| `MMADA_VCHD_CCAW` | 1 | enable CCAW dynamic window expansion |
| `MMADA_VCHD_CCAW_MAX_CAPACITY` | 64 | upper bound for CCAW window growth |
| `MMADA_VCHD_CCAW_PRESSURE_DECAY` | 0.8 | CCAW pressure EMA |
| `MMADA_VCHD_CCAW_EXPAND_STEP` | 8 | window growth step |
| `MMADA_VCHD_CCAW_SHRINK_STEP` | 4 | window shrink step |
| `MMADA_CV_LAMBDA` | 0.5 | ⚠ CD-APC α (naming inherited from CV-DCD) |
| `MMADA_CV_ALPHA` | 0.1 | ⚠ APC β threshold (naming inherited from CV-DCD) |
| `MMADA_VCHD_FORCE_MATH_SDPA` | 1 | force MATH backend for numerically identical paired forward |
| `MMADA_VCHD_COLLECT_TRACE` | 1 | dump per-iteration trace into ledger JSON |
| `MMADA_VCHD_RETURN_REPORT` | 1 | keep the summary report in memory + JSON |

⚠ The two variables prefixed `MMADA_CV_*` are re-used for VCHD α / β because
`mmada.py` maps them via `alpha=self.cv_causal_lambda, beta=self.cv_alpha`.
See H7 in `docs/vchd_cross_check_20260714.md` for the naming caveat.

## Verifying imports without a GPU

`setup_env.sh` already runs a CPU-only sanity check. To reproduce manually:

```bash
PYTHONPATH=MMaDA python - <<'PY'
from decoding import VCHDDecodeConfig, visual_contrast_decode
cfg = VCHDDecodeConfig(history_enabled=True, ccaw_enabled=True); cfg.validate()
print("VCHD import + config OK")
PY
```

For the full deterministic test-suite (still CPU-only, no weights):

```bash
cd MMaDA
python tests/test_vchd.py
python tests/test_vchd_history_ccaw.py
python tests/test_vchd_cache.py
```

## Known caveats before you draw conclusions from the numbers

Before comparing against `docs/experimental_results_20260713.md`, please read
`docs/vchd_cross_check_20260714.md` — it enumerates 15 hidden issues
(H1–H15). The most important ones you probably want to fix *before* the run:

1. **H1** – `post_process_response` regex hard-caps at `[A-E]`; MMMU has
   samples whose answer is F/H/I → they will always be wrong. Trivial one-
   line fix in `MMaDA/evaluation/VLMEvalKit/vlmeval/vlm/mmada/mmada.py`
   line 843 (see full patch in `docs/vchd_cross_check_20260714.md`).
2. **H2** – VCHD only stops on `eos_token_id=126081` (`<|eot|>`), but the
   MMaDA chat template ends on `<|eot_id|>=126348`. Open-question accuracy
   is 1.6% instead of the expected ≥3%. Extend the EOS set in
   `MMaDA/decoding/decoder.py::_truncate_full_sequence_at_eos` and in
   `MMaDA/decoding/vocabulary.py::build_valid_text_vocab`.
3. **H3** – Chat-template specials (`<|start_header_id|>`, `<|end_header_id|>`,
   `<|eot_id|>`, `[iPAD]`, `<|r2i|>`) are not in `forbidden_ids` and can be
   emitted as answer tokens by CD-APC. Add them in `mmada.py::__init__`.
4. **H4** – `mmada.py` forces `temperature=0.0` in the VCHD branch, whereas
   the CV-DCD baselines were swept at 0.8. If you want an apples-to-apples
   number, add a matching T=0 sweep for `b0` / `e1` profiles too (or drop
   the forced-zero for VCHD).

Fixing H1+H2+H3 typically raises MMMU overall by ~1 pp with zero change to
the decoder logic.

## What is NOT in this bundle (fetched at run time)

- Model weights: `Gen-Verse/MMaDA-8B-MixCoT` (via `transformers`
  auto-download; ~16 GB safetensors).
- VQ tokenizer: `showlab/magvitv2` (via `MAGVITv2.from_pretrained`;
  fallback default `multimodalart/MAGVIT2` if the partial in
  `vlmeval/config.py` is overridden).
- MMBench / MMMU parquets: `lmms-lab/MMBench_EN`, `lmms-lab/MMMU`
  (via `huggingface_hub.snapshot_download` inside `prepare_datasets.sh`).
- Any `LMUData/*.tsv`, `outputs/`, `logs/`, `attention_*/` from the local
  development machine (excluded via `.rsync-exclude`).

## License

Inherits Apache-2.0 from MMaDA (see `MMaDA/LICENSE`) and from VLMEvalKit
(see `MMaDA/evaluation/VLMEvalKit/LICENSE`).
