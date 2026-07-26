# LaViDa VCHD / CCAW Decoding

This document describes the LaViDa port of MMaDA's VCHD (visual-access contrastive masked diffusion decoding) and CCAW (capacity-adaptive commit window).

## What changed

- New package: [`llava/decoding/`](../llava/decoding/)
- Generation switch: `decode_strategy="original" | "vchd"`
- Supported backends:
  - LaViDa-LLaDA (`llava_llada`)
  - LaViDa-Dream (`llava_dream`)
- Default path is unchanged: omitting `decode_strategy` keeps the original decoder.

## Key constraints (v1)

- `batch_size=1`
- requires visual tokens (`IMAGE_TOKEN_INDEX` after multimodal expansion)
- `prefix_lm=False` for VCHD
- `cache_type="none"` only (dual KV cache is rejected)
- ablated branch blocks non-visual queries from attending to visual keys

## CLI / lmms-eval knobs

Flattened VCHD fields use the `vchd__` prefix:

```bash
--gen_kwargs decode_strategy=vchd,prefix_lm=False,max_new_tokens=64,vchd__ccaw_enabled=true,vchd__ccaw_mode=inverse_window,vchd__tau_base=0.1,vchd__tau_contrast=0.9
```

Common fields:

| Field | Meaning |
|-------|---------|
| `vchd__tau_base` | base-confidence gate |
| `vchd__tau_contrast` | contrast-confidence gate |
| `vchd__enable_g_gate` | enable CD-APC visual-gain gate |
| `vchd__ccaw_enabled` | enable adaptive window |
| `vchd__ccaw_mode` | `legacy` / `inverse_window` / `hard_block` |
| `vchd__mask_capacity` | base window capacity |
| `vchd__ccaw_max_mask_capacity` | max adaptive capacity |
| `vchd__history_enabled` | sparse history reliability |

## Smoke scripts

```bash
# LLaDA
MODE=original bash eval/run_vchd_llada.sh /path/to/lavida-llada-hd
MODE=vchd bash eval/run_vchd_llada.sh /path/to/lavida-llada-hd
MODE=vchd_ccaw bash eval/run_vchd_llada.sh /path/to/lavida-llada-hd

# Dream
MODE=original bash eval/run_vchd_dream.sh /path/to/lavida-dream-hd
MODE=vchd bash eval/run_vchd_dream.sh /path/to/lavida-dream-hd
MODE=vchd_ccaw bash eval/run_vchd_dream.sh /path/to/lavida-dream-hd
```

Set `LIMIT=0` for a full split, or keep the default `LIMIT=1` for smoke tests.

## Python API

```python
output = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    decode_strategy="vchd",
    max_new_tokens=64,
    prefix_lm=False,
    tokenizer=tokenizer,
    vchd__ccaw_enabled=True,
    vchd__ccaw_mode="inverse_window",
)
```

When `vchd__return_report=true`, the latest report is also stored on `model._last_vchd_report`.

## Unit tests

```bash
cd LaViDa
PYTHONPATH=. python -m pytest tests/decoding -q
```

## Notes on Dream registration

The lmms-eval Dream adapter is registered as `llava_dream`. Older copies incorrectly reused the `llava_llada` registration name; `eval/run_dream.sh` and the VCHD Dream smoke script both expect `llava_dream`.
