# CV-DCD 当前状态

**最后更新**：2026-07-07  
**当前分支**：v4 defer-only + intervention-strength λ + E0 parity fix

本文档只包含**当前有效**的方案与结果。历史迭代（v3.1 gate、v3.2 CD-APC、logit-boost 变体）不再维护，见 `dcd_to_cv_dcd_full_report.md`。

---

## 1. 项目结构

```
MMaDA/
├── models/
│   ├── mmada_decode.py                       # DCD 主入口，_pick_transfer_cv 分派
│   │
│   ├── cv_common/                            # 所有 CV 方向共享工具
│   │   ├── image_drop.py                     # 图像 ablation：mask/shuffle/text_only/…
│   │   ├── log_prob.py                       # log-probability 工具
│   │   ├── paired_forward.py                 # base + drop 双前向
│   │   └── types.py
│   │
│   └── defer_only/                           # ★ Direction B：当前主力
│       ├── dispatcher.py                     # pick_transfer_defer_only()
│       └── veto.py                           # hard / mult / min / soft veto
│
├── tests/
│   └── test_defer_only.py                    # 32 CPU 单测（含 3 个 bit-parity）
│
├── evaluation/VLMEvalKit/
│   ├── scripts/
│   │   ├── run_defer_only_matrix_v5.sh       # 已跑：(λ,τ) 矩阵
│   │   ├── run_defer_only_phase_d_focused.sh # 已跑：E6 focused 60 样本
│   │   ├── run_defer_only_e0_parity.sh       # 待跑：4-config head-to-head
│   │   └── compare_e0_parity.py              # byte-parity 对比工具
│   └── outputs/cvdcd_sweep/                  # 所有 sweep 结果
│
└── docs/
    ├── cv_dcd_status_current.md              # 本文档
    ├── cv_dcd_v4_design.md                   # 详细设计
    ├── cv_dcd_v4_plan.md                     # 里程碑规划
    └── (被抛弃的方向：cv_dcd_v3.1/v3/full_report 等)
```

被抛弃的路径（保留仅供 API 兼容，不再维护）：`cv_v32_*`、`cd_apc`、`cd_naive`、`legacy_score`。

---

## 2. 核心 idea：defer-only + intervention strength

### 2.1 一句话陈述

> **视觉信号只用来推迟提交，永不修改 argmax。**  
> commit 顺序由 base_logits 的 argmax 决定（与 plain DCD 一致），视觉 gain 只在 gain < τ 时降低该位置的 confidence，让它在后续 diffusion 轮次被重新评估。

### 2.2 决策管线

```
1. x0 = argmax(base_logits)                      # 永不修改
2. base_conf = softmax(base_logits.fp64)[x0]     # 与 plain DCD bit-identical
3. if drop_logits available AND λ > 0:
     gain    = clip(base_logit[x0] - drop_logit[x0], ±causal_clip)
     veto_c  = apply_veto(base_conf, gain, veto_type, τ, β)
     eff_c   = (1 - λ) * base_c + λ * veto_c     # ★ λ = 干预强度（线性 blend）
   else:
     eff_c   = base_c
4. transfer_index = threshold_select(eff_c, decode_param=0.9)   # DCD 原生
```

### 2.3 两个正交超参

| 超参 | 含义 | 语义 |
|---|---|---|
| **τ** (`defer_tau`) | gain 阈值 | 控制干预**频率**（哪些位置被降 conf）|
| **λ** (`causal_lambda ∈ [0, 1]`) | intervention strength | 控制干预**强度**（降多少）|

关键边界：
- `λ = 0` → 严格 = plain DCD（sanity）
- `λ = 1` → 完全 veto（v1–v4 旧语义）
- `λ ∈ (0, 1)` → base_conf 与 veto_conf 的线性 blend

### 2.4 Veto 变体

| 变体 | 公式 | 状态 |
|---|---|---|
| **hard** | `eff = base if gain ≥ τ else 0` | **主力**（v5 矩阵采用）|
| soft | `eff = base * exp(-β · max(τ - gain, 0))` | 平滑替代，未大规模验证 |
| mult / min | `sigmoid(β·(gain-τ))` 系 | 有零点病态，仅供 ablation |

### 2.5 关键 invariants（32 单测保证）

1. `x0[mask_positions] == argmax(base_logits)[mask_positions]`（含 Gumbel 时也成立）
2. `eff_conf ≤ base_conf` 逐位（永不 boost）
3. `λ = 0` → dispatcher 输出与 `_pick_transfer` **bit-identical**（新增 3 个 strict-parity 测试）

---

## 3. 已完成的实验

### 3.1 v5 矩阵（60 样本 LLaVABench）

**固定**：`hard` veto、`text_only` drop、`logit` gain、`decode_param=0.9`、`temperature=0.8`。

| tag | λ | τ | Relative | VLM | GPT4 | conv | complex | detail |
|---|---|---|---|---|---|---|---|---|
| E0_sanity | 0 | – | 28.6 | 25.3 | 88.7 | 25.6 | 31.1 | 27.7 |
| E1 | 0.25 | −2.0 | 30.7 | 27.0 | 87.8 | 27.5 | 35.8 | 25.9 |
| E2 | 0.25 | −3.0 | 30.4 | 26.8 | 88.3 | 27.5 | 34.8 | 26.3 |
| E3 | 0.5 | −2.0 | 30.0 | 26.3 | 87.7 | 26.9 | 34.9 | 25.4 |
| **E4** | 0.5 | **−3.0** | **30.7** | **27.2** | 88.5 | 28.1 | 34.9 | 26.5 |
| E5 | 1.0 | −2.0 | 29.9 | 26.5 | 88.5 | 27.5 | 34.5 | 25.0 |
| **E6** | 1.0 | **−3.0** | **30.9** | **27.2** | 87.8 | 27.5 | 34.6 | **28.6** |

### 3.2 v5 结论

1. **E6 (λ=1.0, τ=−3.0)** 与 **E4 (λ=0.5, τ=−3.0)** 并列最优（VLM +1.9 vs E0）。
2. **E6 唯一保住 detail 类别**（28.6，其它 config 都低于 E0 的 27.7）。
3. 窄 τ（−3.0）优于宽 τ（−2.0）：只在极负 gain 位置介入更稳。
4. λ 效应弱于 τ；**频率控制**（τ）比**强度**（λ）重要。

### 3.3 今日诊断：E0 与 plain DCD 不 bit-identical

**症状**：v5 矩阵的 E0（λ=0）本应作为 plain DCD sanity，但：
- 与历史 plain DCD baseline (2026-06-13) 有 **39/60** 文本差异（21/60 完全相同）
- VLM Score 25.3 vs 历史 36.7（差 11 分）

**根因分解**：

| 因素 | 影响 | 修复状态 |
|---|---|---|
| **D1（代码）**：defer_only 用 `logsumexp(bfloat16)→fp64` 计算 base_conf，plain DCD 用 `softmax(fp64)`。末几位 bit 差（~2⁻⁷）足以在 threshold=0.9 边界翻越。 | 小（触发少数 flip）| **已修** |
| **D2（环境）**：GPT scorer 版本更新 + `temperature=0.8` 采样漂移 | **大**（占主要分差）| 无法修，只能同环境重跑作 baseline |

**修复动作**（已实施，仍待 GPU 验证）：
1. `models/defer_only/dispatcher.py:218-224` 改用 `F.softmax(fp64)` → gather，与 `_confidence_from_logits` 严格一致
2. 新增 3 个 strict-parity 单测：`test_lambda_zero_base_conf_bit_identical_to_plain_dcd`、`test_lambda_zero_stress_bit_identical_plain_dcd`、`test_lambda_zero_with_temperature_bit_identical_plain_dcd`
3. **32/32 单测通过**
4. Byte-parity 对比工具 `scripts/compare_e0_parity.py`

---

## 4. 唯一待跑：E0-parity head-to-head

**脚本**：`scripts/run_defer_only_e0_parity.sh`（~2.5h GPU on H100）

| tag | strategy | cv_mode | λ | τ | 目的 |
|---|---|---|---|---|---|
| `B0_plain_dcd` | `dcd` | `off` | 0 | – | **真** plain DCD（`dcd_decode_text_dual_cache`）|
| `E0_defer_l0` | `cv_dcd` | `defer_only` | 0 | – | Fix 后应与 B0 **byte-identical** |
| `E4_defer_l0.5_t-3.0` | `cv_dcd` | `defer_only` | 0.5 | −3.0 | v5 最优候选 |
| `E6_defer_l1.0_t-3.0` | `cv_dcd` | `defer_only` | 1.0 | −3.0 | v5 detail 保留最优 |

**通过标准**：
1. **B0 vs E0**：60/60 byte-identical（证 fix 生效）
2. **B0 vs E4/E6**：同环境下的相对提升（替换不可靠的历史 baseline）
3. **E6 保 detail**：延续 v5 结论

**运行命令**：

```bash
cd /home/user/dcd/MMada_DCD/MMaDA/evaluation/VLMEvalKit
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
bash scripts/run_defer_only_e0_parity.sh 2>&1 | tee outputs/e0_parity_$(date +%Y%m%d_%H%M%S).log
```

跑完后：

```bash
RUN_DIR=$(ls -dt outputs/cvdcd_sweep/defer_only_e0_parity_* | head -1)
for tag in E0_defer_l0 E4_defer_l0.5_t-3.0 E6_defer_l1.0_t-3.0; do
  /home/user/anaconda3/envs/mmada/bin/python scripts/compare_e0_parity.py \
      --base "${RUN_DIR}/B0_plain_dcd" \
      --alt  "${RUN_DIR}/${tag}"
done
```

---

## 5. 关键设计决策一览（不再讨论的问题）

| 问题 | 结论 | 依据 |
|---|---|---|
| 是否修改 argmax？ | **不**。visual signal 只降 conf，不改 token。 | v3.2 CD-APC argmax 修改损伤 LM 先验（重复抑制），Phase B 已证 |
| 用 `logit` 还是 `logprob` gain？ | **logit**（`base_logit[x0] − drop_logit[x0]`）。 | v4 早期 logprob gain 因 LSE 抵消退化为 ~0；logit 分布 ±1.5 更适合 τ 门控 |
| 图像 ablation 选哪种？ | **text_only**（drop 整段 image tokens）。 | Smoke v3/v4 显示 text_only gain 分布更稳定 |
| λ 在管线里的位置？ | **intervention strength**（线性 blend），非开关。 | v5 前 λ 只做 !=0 判定；v5 允许 λ∈(0,1) 分离强度与频率 |
| Veto 变体？ | **hard**。 | soft 未大规模验证；mult/min 有零点病态 |
| Decoding cost（NFE）？ | 未额外优化。`cv_stride` 在 dual-cache 实现里是 no-op（drop_logits 已算但被丢弃）。 | 保留为已知优化空间 |
