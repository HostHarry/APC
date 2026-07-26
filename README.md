# 43133-lavida

Local LaViDa Thinking Diffusion (SWD / PSP / PSP+VRG) snapshot from the autodl workspace.
Code lives under `LaViDa/` (weights and eval logs are not uploaded).
Includes the Prefix-LM `attention_bias` safeguard in `modeling_llada.py`.

# APC: MMaDA decoding experiments on VLind-Bench

This branch contains the code that is actually executed by the current
MMaDA/VCHD experiments. Model weights, benchmark data, predictions, traces,
and older source snapshots are intentionally kept outside Git.

## Repository layout

```text
.
├── MMaDA_DCD_cloud_bundle_20260711/
│   └── MMaDA/
│       ├── decoding/                 # VCHD, CCD, long-tail history, CCAW
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
  --vchd-profile adaptive_temporal \
  --model-identifier mmada_vchd_loglogistic \
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
  --data_path VLind-Bench/outputs/data_mmada_vchd_loglogistic.json \
  --model_identifier mmada_vchd_loglogistic
```

## Log-logistic trajectory history

The `adaptive_temporal` profile uses CCD's top-V position intersection and
full-vocabulary marginalization. Exposure-calibrated visual and trajectory
conflict mix a shifted discrete Log-logistic survival kernel into CCD's
three-round rectangular kernel. If activation is zero—or
`--vchd-adaptive-temporal-tail-mix-max 0`—selection and token behavior are
exactly CCD.

See
`MMaDA_DCD_cloud_bundle_20260711/MMaDA/docs/vchd_loglogistic_history.md`
for formulas and diagnostics.

## SWD / PSP / VRG reproduction

The original-remasking path supports three training-free controls:

- SWD: `MMADA_THINKING_SWD=1` and
  `MMADA_THINKING_SWD_LAMBDA=5.0`
- PSP: `MMADA_THINKING_PSP=1` and
  `MMADA_THINKING_PSP_GAMMA=0.5`
- VRG (normally combined with PSP): `MMADA_THINKING_VRG=1` and
  `MMADA_THINKING_VRG_SCALE=0.5`

Set `MMADA_DECODE_STRATEGY=original` when invoking VLMEvalKit directly.
The reproduction runners set the strategy and method-specific variables:

```bash
bash VLind-Bench/scripts/run_llavabench_thinking_infer.sh
```

The MMBench runner defaults to `MMBench_DEV_EN_2C` and the methods
`swd psp psp_vrg`. Prepare its two-cycle manifest from the ordinary
MMBench development TSV:

```bash
export LMUData=/path/to/LMUData
python VLind-Bench/scripts/make_mmbench_two_cycle.py \
  "$LMUData/MMBench_DEV_EN.tsv" \
  "$LMUData/MMBench_DEV_EN_2C.tsv"
bash VLind-Bench/scripts/run_mmbench_thinking_chain.sh
```

If `MMBench_DEV_EN_2C.tsv` is absent but `MMBench_DEV_EN.tsv` exists, the
runner safely performs this generation step itself; if both are absent, it
stops and prints the exact generation command. The registered two-cycle
dataset shares `LMUData/images/MMBench` with `MMBench_DEV_EN`; no images are
copied into the repository.

The scripts resolve source code from this checkout. External weights and data
default to `/root/autodl-tmp`; override that parent with
`MMADA_EXTERNAL_ROOT`, or set `MMADA_MODEL_PATH`, `MMADA_TOKENIZER_PATH`,
`MMADA_VQ_MODEL_PATH`, `MMADA_LMUDATA_SOURCE`, and `LMUData` individually.
Use `MMADA_OUTPUT_ROOT` to relocate generated outputs. This repository does
not include model weights, benchmark data, logs, predictions, or results.

## CPU regression tests

```bash
cd MMaDA_DCD_cloud_bundle_20260711/MMaDA
python -m py_compile decoding/*.py \
  evaluation/VLMEvalKit/vlmeval/vlm/mmada/mmada.py
python -m pytest \
  tests/test_thinking_swd.py \
  tests/test_mmbench_two_cycle.py \
  tests/test_vchd.py \
  tests/test_vchd_history_ccaw.py \
  tests/test_vchd_cache.py
```

No model weights or GPU are required for these deterministic tests.
