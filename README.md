# 43133-vcd

Dual-model **Visual Contrastive Decoding (VCD, CVPR 2024)** snapshot from the autodl workspace.

Negative branch = Gaussian-noised image (`add_diffusion_noise`, default `noise_step=500`), then
`(1+α)·logits_v − α·logits_v'` with APC (paper α=1.0, β=0.1).

## Layout

```text
.
├── LaViDa/                                      # LaViDa VCD
│   ├── llava/decoding/vcd_noise.py
│   ├── llava/decoding/vcd_decoder.py
│   ├── eval/run_vcd_five_benchmarks.sh
│   └── tests/decoding/test_vcd.py
├── MMaDA_DCD_cloud_bundle_20260711/MMaDA/       # MMaDA VCD
│   ├── models/cv_common/vcd_noise.py
│   ├── models/mmada_decode.py                   # decode_strategy='vcd'
│   └── evaluation/VLMEvalKit/.../MMaDA-MixCoT-VCD
└── VLind-Bench/
    ├── eval/mmada_eval.py                       # --strategy vcd
    ├── eval/mmada_m3cot_eval.py                 # LightChen233 M3CoT scorer
    └── scripts/run_mmada_vcd_matrix.sh
```

## Run (high level)

```bash
# LaViDa five benches (vcd_prefix_cache)
bash LaViDa/eval/run_vcd_five_benchmarks.sh

# MMaDA matrix (VLind + VLMEvalKit + official M3CoT)
bash VLind-Bench/scripts/run_mmada_vcd_matrix.sh
```

Weights, datasets, and eval logs are not uploaded.

Also includes MMMU answer extraction fix: unparseable MC → `""` (no `random.choice`).
