# MMMU boxed-first offline rescore

Rules: last `\boxed{}` (incl. unclosed) → answer-is → `(A)` → bare letter → option text; unparseable = wrong (no random).

| Run | Split | n | Official all | Boxed all | Δ | Official MCQ | Boxed MCQ |
|---|---|---:|---:|---:|---:|---:|---:|
| mmmu_dev_val_full/original | dev | 150 | 24.67% | 36.00% | +11.33 | 26.24% | 38.30% |
| mmmu_dev_val_full/original | val | 900 | 31.78% | 35.78% | +4.00 | 33.29% | 37.54% |
| mmmu_dev_val_full/original | weighted | 1050 | 30.76% | 35.81% | +5.05 | 32.29% | 37.65% |
| mmmu_dev_val_full/vchd_ccaw_prefix_cache | dev | 150 | 24.00% | 39.33% | +15.33 | 25.53% | 41.84% |
| mmmu_dev_val_full/vchd_ccaw_prefix_cache | val | 900 | 29.44% | 34.33% | +4.89 | 30.58% | 35.77% |
| mmmu_dev_val_full/vchd_ccaw_prefix_cache | weighted | 1050 | 28.67% | 35.05% | +6.38 | 29.86% | 36.64% |
| mmmu/vchd_prefix_cache | dev | 150 | 22.00% | 38.67% | +16.67 | 23.40% | 41.13% |
| mmmu/vchd_prefix_cache | val | 900 | 31.78% | 35.44% | +3.66 | 33.18% | 37.07% |
| mmmu/vchd_prefix_cache | weighted | 1050 | 30.38% | 35.90% | +5.52 | 31.78% | 37.65% |

## Verdict: 43133 reported MMMU (36-40%) fully decomposed (2026-07-27 21:45)

Their uploaded samples (hostharry/apc @ ea0541b, `eval/reported_results/mmmu/`) rescored
with THIS repo's strict boxed-first script (no random fallback), weighted dev+val:

| run | official (their utils) | strict boxed-first | inflation |
|---|---|---|---|
| theirs SWD      | 39.14% | **33.52%** | +5.62pp |
| theirs PSP      | 38.95% | **35.90%** | +3.05pp |
| theirs PSP+VRG  | 39.52% | **35.81%** | +3.71pp |
| ours original   | 30.76% (no boxed parse) | **35.81%** | — |
| ours VCHD+CCAW  | 28.66% | **35.05%** | — |
| ours VCHD bidir | 30.38% | **35.90%** | — |

Root causes of the apparent 8-16pp gap:
1. Their `eval/lmms_eval/tasks/mmmu/utils.py` ships a boxed-first patch (L397-410)
   that our tree lacked -> their "official" numbers were already boxed-aware (+4-9pp real recovery).
2. Their official numbers keep the upstream random.choice fallback. Unparseable (truncated
   CoT at 128 tok) = 17-25% of val (SWD 208/847, PSP 147/847, PSP+VRG 164/847);
   random adds ~= share x 25% ~= +3-6pp of pure noise. Matches observed inflation per mode.
3. Genuine method effect on MMMU at this protocol: none. All six runs land at 33.5-35.9%;
   ours original (35.81) ties their best (PSP 35.90); SWD is actually worst (33.52).

Gen protocol confirmed identical both sides: max_new_tokens=128, until ['\n\n'],
prefix_lm=True, block=64, steps/block=64, same lavida-llada-reason weights.

MC no-regression for VCHD: CONFIRMED on MMMU (35.05-35.90 vs original 35.81, within noise, n=1050).
