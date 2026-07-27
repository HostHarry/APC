# LaViDa MMMU results (43133-lavida)

Model: `lavida-llada-hd-reason` (Thinking Diffusion).
Schedule: L/T/B = `128/128/64`, `step_per_block=64`.
Task: `mmmu_dev_val_full` via lmms-eval.

| Mode | val | dev | Source run |
|------|-----|-----|------------|
| SWD | 39.67 | 36.00 | `lavida_thinking_3bench_no_original_20260724` (infer 2026-07-24 23:05–23:39, ~3.77 s/it) |
| PSP | 38.67 | 40.67 | `lavida_thinking_3bench_no_original_20260724` (infer 2026-07-24 23:39–07-25 00:13, ~3.81 s/it) |
| PSP+VRG | 39.78 | 38.00 | `lavida_psp_vrg_rerun_20260727_055355` (infer 2026-07-27 06:00–06:48, ~5.28 s/it; VRG attention_bias fixed) |

Each mode directory contains:
- `*_results.json` — lmms-eval metrics
- `*_samples_mmmu_dev.jsonl` / `*_samples_mmmu_val.jsonl` — predictions

**Not included:** invalidated `attentionfix` / pre-fix PSP+VRG runs, and `*.pre_boxed_fix.*` backups.
