# M3CoT full (LaViDa dual run 2026-07-28)

| Mode | Acc | Stderr | N |
|---|---:|---:|---:|
| `vchd_prefix_cache` | 26.06% | ±0.91 | 2318 |
| `vchd_ccaw_prefix_cache` | 25.58% | ±0.91 | 2318 |

## Decode knobs

- `max_new_tokens=512`
- `prefix_lm=True`, `vchd__prefix_prompt_cache=True`
- `vchd__mask_capacity=16`, `alpha=0.25`, `beta=0.1`, `tau_base=0.1`, `tau_contrast=0.9`
- CCAW mode only: `ccaw_enabled=true`, `ccaw_mode=inverse_window`, `ccaw_max_mask_capacity=64`
- Extraction: official `judge_answer` via `extract_m3cot_answer`
