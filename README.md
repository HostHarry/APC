# APC: MMaDA decoding experiments on VLind-Bench

This branch contains the code that is actually executed by the current
MMaDA/VCHD experiments. Model weights, benchmark data, predictions, traces,
and older source snapshots are intentionally kept outside Git.

## Repository layout

```text
.
├── MMaDA_DCD_cloud_bundle_20260711/
│   └── MMaDA/
│       ├── decoding/                 # VCHD, persistent focus trajectory, CCAW
│       ├── models/                   # MMaDA model and decode dispatch
│       ├── evaluation/VLMEvalKit/    # MMaDA evaluation wrapper
│       ├── tests/                    # deterministic CPU regressions
│       └── docs/
└── VLind-Bench/
    └── eval/
        ├── mmada_eval.py             # VLind A/B/C/D experiment runner
        └── score_pipeline.py         # official-style metric aggregation
```

`MMaDA_DCD_cloud_bundle_20260711/MMaDA` is the only authoritative model and
decoder tree. `VLind-Bench/eval/mmada_eval.py` is orchestration glue: it loads
VLind records, builds prompts, selects profiles, invokes the bundled
VLMEvalKit wrapper, and writes resumable predictions.

The active VCHD call chain is:

```text
VLind-Bench/eval/mmada_eval.py
  -> evaluation/VLMEvalKit/vlmeval/vlm/mmada/mmada.py
  -> models/modeling_mmada.py::mmu_generate
  -> decoding/decoder.py::visual_contrast_decode
```

## External runtime assets

By default the runner reads large assets from `/root/autodl-tmp`:

```text
/root/autodl-tmp/MMaDA-8B-MixCoT
/root/autodl-tmp/magvitv2
/root/autodl-tmp/datasets/VLind-Bench/VLind-Bench Dataset
```

Set `MMADA_EXTERNAL_ROOT` to use another parent directory, or pass
`--model-path`, `--tokenizer-path`, `--vq-model-path`, `--data-path`, and the
two image-directory arguments explicitly. The MMaDA source root is resolved
relative to this repository and can still be overridden with `--mmada-root`.

## Running VLind

From the repository root:

```bash
python VLind-Bench/eval/mmada_eval.py \
  --strategy vchd \
  --vchd-profile focus_longtail \
  --model-identifier mmada_vchd_focus_longtail \
  --global-ids "<comma-separated VLind global IDs>" \
  --max-new-tokens 128 \
  --steps 128 \
  --block-length 64 \
  --temperature 0.8 \
  --vchd-save-reports \
  --vchd-collect-trace \
  --resume
```

Omit `--global-ids` for the complete valid set. Predictions default to
`VLind-Bench/outputs/`, which is ignored by Git.

Score a completed prediction file with:

```bash
python VLind-Bench/eval/score_pipeline.py \
  --data_path VLind-Bench/outputs/data_mmada_vchd_focus_longtail.json \
  --model_identifier mmada_vchd_focus_longtail
```

## Persistent focus trajectory (focus-dwell + focus-longtail)

The `focus_dwell` profile snapshots the top-V unresolved MASK positions each
iteration into a bounded ring buffer and only marginalizes over positions
whose per-position *dwell counter* reaches the configured depth—no explicit
frame intersection is ever formed. The `focus_longtail` profile keeps the
same eligibility primitive and adds a shifted discrete Log-logistic survival
kernel over CD-APC lags, gated by exposure, visual relevance, and
current-vs-dwell conflict. Setting `--vchd-focus-longtail-mix-ceiling 0`
(or a fully unreliable visual channel) is token-by-token identical to
`focus_dwell`.

See
`MMaDA_DCD_cloud_bundle_20260711/MMaDA/docs/vchd_focus_longtail.md`
for formulas, defaults, and diagnostics.

## CPU regression tests

```bash
cd MMaDA_DCD_cloud_bundle_20260711/MMaDA
python -m py_compile decoding/*.py \
  evaluation/VLMEvalKit/vlmeval/vlm/mmada/mmada.py
python -m pytest \
  tests/test_vchd.py \
  tests/test_vchd_history_ccaw.py \
  tests/test_vchd_cache.py
```

No model weights or GPU are required for these deterministic tests.
