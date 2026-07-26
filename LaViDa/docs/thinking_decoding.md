# LaViDa SWD / PSP / VRG Decoding

This port adds training-free score modulation to the original
LaViDa-LLaDA low-confidence decoder. The default decoder is unchanged.

## Methods

- **SWD**: multiplies token confidence by
  `exp(-lambda * KL(p_previous || p_current))`. Only response-token
  distributions are cached.
- **PSP**: penalizes later response positions more strongly during early
  denoising steps.
- **VRG**: performs visual and visual-access-ablated forwards and applies
  `L_ablated + (scale + 1) * (L_visual - L_ablated)`.
- **PSP+VRG**: applies VRG to logits and PSP to the resulting token confidence.

Defaults match the existing MMaDA experiments:

- `thinking__swd_lambda=5.0`
- `thinking__psp_gamma=0.5`
- `thinking__vrg_scale=0.5`

## lmms-eval

```bash
CKPT=lavida-ckpts/lavida-llada-hd-reason

MODE=swd LIMIT=1 bash eval/run_thinking_llada.sh "$CKPT"
MODE=psp LIMIT=1 bash eval/run_thinking_llada.sh "$CKPT"
MODE=vrg LIMIT=1 bash eval/run_thinking_llada.sh "$CKPT"
MODE=psp_vrg LIMIT=1 bash eval/run_thinking_llada.sh "$CKPT"
```

Single synthetic-image smoke (loads the checkpoint once):

```bash
PYTHONPATH=. python scripts/smoke_thinking_llada.py --ckpt "$CKPT"
```

Direct generation arguments:

```text
decode_strategy=swd,prefix_lm=True,thinking__swd_lambda=5.0
decode_strategy=psp,prefix_lm=True,thinking__psp_gamma=0.5
decode_strategy=vrg,prefix_lm=True,thinking__vrg_scale=0.5
decode_strategy=psp_vrg,prefix_lm=True,thinking__psp_gamma=0.5,thinking__vrg_scale=0.5
```

VRG builds separate visual and visual-access-ablated prefix KV caches in one
paired prefill. During every denoising step, the ablated branch is still
blocked from attending to visual keys. VRG therefore uses roughly twice the
branch compute and KV-cache memory of PSP or the original decoder.

## Python API

```python
output = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    decode_strategy="psp_vrg",
    prefix_lm=True,
    max_new_tokens=128,
    block_length=64,
    step_per_block=64,
    thinking__psp_gamma=0.5,
    thinking__vrg_scale=0.5,
)
```

The supported strategy names are `original`, `swd`, `psp`, `vrg`,
`psp_vrg`, and `thinking`. The generic `thinking` strategy accepts any
combination of `thinking__psp_enabled`, `thinking__vrg_enabled`, and
`thinking__swd_enabled`.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/decoding/test_thinking_decoding.py -q
```
