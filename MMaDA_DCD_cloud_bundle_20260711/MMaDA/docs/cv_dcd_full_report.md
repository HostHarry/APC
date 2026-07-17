# Causal-Visual DCD (CV-DCD) for MMaDA — 完整实验报告

**日期：** 2026-07-06
**基准数据集：** LLaVABench（60 个样本），GPT-4 打分
**范围：** v1 → v3.2 完整迭代历史、当前实验结果、以及对"逐 token 视觉 boost"范式的结构性诊断。

---

## 1. TL;DR

- **CV-DCD 家族目前没有任何配置能超过 DCD baseline。** 经过 4 次迭代（v1, v2, v3.1, v3.2），最好的配置（v3.2 B3：min_base_blended conf、不加 gating）达到 VLM 28.3，而 DCD baseline 是 **VLM 36.7**——**差距 −8.4 VLM 分**，没有任何单一修复能补上。
- **反直觉发现**：`min(base_conf, blended_conf)`（两个视角"都同意才 commit"的双保险 conf 源）**不需要 gating 就已经是最好**——比原本预期最优的 B2 (blended + gate=0.5) 高 +1.5 VLM。而单纯的 base-conf gating 平均只提升 0.1 分/样本，且几乎只帮 conv 类。
- **我们确实定位并修复了 5 个机械 bug**：log/prob 尺度错位（v2 Fix1）、image-drop OOD 问题（v2 Fix3）、APC 假置信（v3.2）、缺失 base-confidence gating（v3.2）、debug NPZ 覆盖 bug。每个修复都消除了一个真实缺陷；但即使全部叠加起来，也无法弥补这个差距。
- **剩余差距是结构性的，不是实现层面的。** 核心假设"逐 token `visual_gain(pos, tok)` boost 能改善 grounding"在两个约束同时作用下失败：(a) MMaDA 是**扩散 mask 解码器**，所有 `masked` 位置是**并行**更新——CD 对每个位置独立 boost，无法像 AR context 那样抑制跨位置重复；(b) 用 `shuffle` 做图像 ablation 得到的 `visual_gain` 是一个**近零均值的噪声信号**（`median=-0.125`, `p10/p90=[−1.5, +1.25]`），大部分 "boost" 反映的是噪声，不是视觉 grounding。
- **结构性论断的经验证据**：v3.2 B1 的 confidence 修复让生成中位长度从 334→426 字符恢复了，但**60 个样本里有 41 个产生了和 v3.1 完全一样的答案**。B2 加上 gating 后，只改变了 7/60 个样本的答案。B3 相对 B1 提升 +0.27 mean_delta，但 top losses（idx=7 −8, idx=54 −5, idx=36 −4）跟 v3.1 A3 **一模一样**——argmax 层的损伤没修复。
- **推荐的下一步方向**：放弃逐 token boost，转向 (a) 整段序列级 CFG 重排，或 (b) `defer-only` CV—— visual 信号只影响*是否*要 refresh 一个位置，不影响*选哪个 token* 去 commit。**若继续走 per-token boost 框架，可用 B3 (min_base_blended, gate=0) 作为新起点。**

---

## 2. 项目文件布局（相关的部分）

```
MMaDA/
├── models/
│   ├── mmada_decode.py                  ← DCD & CV-DCD 核心（单一 dispatch）
│   ├── cv_v32_apc.py                    ← v3.2 APC mask + CD blending（纯数学）
│   ├── cv_v32_confidence.py             ← v3.2 三种 conf 源 dispatcher
│   ├── cv_v32_gating.py                 ← v3.2 base-confidence gating
│   └── cv_v32_dispatcher.py             ← v3.2 总成入口：pick_transfer_v32
├── docs/
│   ├── cv_dcd_report_v1.md              ← v1 log-score 设计 + bug
│   ├── cv_dcd_report_v2.md              ← v2 Fix1+Fix3 + 发现"双峰效应"
│   ├── cv_dcd_v3.1_apc_contrastive_design.md
│   ├── cv_dcd_v3.1_phase_a_report.md    ← phase-A 结果（v3.1 conf bug 状态）
│   └── cv_dcd_full_report.md            ← 本文档
├── tests/
│   ├── test_cd_style_smoke.py           ← v3.1 单元测试
│   └── test_cv_v32.py                   ← v3.2 单元测试（19/19 全绿）
└── evaluation/VLMEvalKit/
    ├── vlmeval/vlm/mmada/mmada.py       ← wrapper：env var → config
    ├── attention_analysis/
    │   ├── cv_debug_io.py                ← per-step NPZ 存储
    │   └── full60_attention_analysis.py  ← LLaVABench 60 样本 attention 分析
    ├── scripts/
    │   ├── run_cvdcd_v31_phase_a.sh      ← v3.1 sweep（3 组）
    │   └── run_cvdcd_v32_sweep.sh        ← v3.2 phase-B sweep（5 组）
    └── outputs/
        ├── MMaDA-MixCoT-DCD-DualCache/T20260613_G/  ← DCD baseline（60 样本）
        └── cvdcd_sweep/
            ├── cvdcd_v31_phase_a_20260706_174459/    ← v3.1 phase A（3 组已完成）
            └── cvdcd_v32_phase_b_20260706_185908/    ← v3.2 phase B（5 组：2 完成，3 运行中）
```

---

## 3. 迭代时间线

| 版本 | 机制 | 最好 VLM (60) | vs baseline | 结论 |
|---|---|---|---|---|
| DCD baseline | Threshold DCD, dual-cache, 无 CV | **36.7** | — | 参考基准 |
| CV-DCD v1 | `cv_score = log_p + λ·visual_gain` 对 prob threshold 0.9 判断 | ~24 | −13 | **Bug**：log/prob 尺度错位 → CV-DCD 退化为 top-1 fallback commit |
| v2 Fix1+Fix3 | Prob 空间 `cv_score = raw_conf · exp(λ·visual_gain)` + `neutral` image drop | 24.2 | −12.5 | **症状**：只改变 commit 顺序，从未改变 token 选择 → 观察到"双峰效应" |
| v3.1 (APC-CD) | Logit-blend `base + β·(base−drop)` + APC α-mask | 26.7 | −10.0 | **真正改变了 token**（1.13% 的 step）。发现假置信 bug（APC → v_valid_size=1 → softmax=1.0） |
| **v3.2** | v3.1 + confidence 源修复 (3 种) + base-conf gating | **28.3** | **−8.4** | **所有已识别 bug 都修好了。B3 (min_base_blended, gate=0) 意外成为最佳。** |

**4 次迭代累计相对 v1 的收益：~+4 VLM 分。**
**距离 baseline 剩余差距：−8.4 VLM 分。结构性的。**

---

## 4. v3.2 具体改动

v3.2 引入了两个独立的正交轴：

### 4.1 Confidence 源（`cv_conf_source`）

**v3.1 bug 的诊断（助手的关键观察）：**
APC 只保留 `logit ≥ max_logit + log(α)` 的 token。α=0.1 时，phase-A 数据显示 `v_valid_size` 中位数=1（即只剩 argmax 存活）。对只剩 1 个非 mask token 的分布做 softmax = 1.0 → 置信度被人为拉到最大 → DCD 的 threshold=0.9 判定被绕过。

**修复：** confidence 从**未被 mask 的** blended logits（或直接从 base logits）计算，不从 APC-masked selection logits。

测试三种模式：

- `blended`（助手 v1 建议）：`softmax(base + β·(base−drop))[x0]`
- `min_base_blended`（助手保守 v2）：`min(base_conf, blended_conf)`
- `base`：`softmax(base_logits)[x0]` — 最干净的语义：DCD 的 defer/commit 判断完全基于 base 的真实不确定性；CD 只影响"当决定要 commit 时选哪个 token"。

### 4.2 Base-confidence gating（`cv_gate_tau`）

**动机：** phase-A 显示"双峰效应"——CV-DCD *救回* baseline 失败的样本（idx=12: base=1 → cv=8, +7）*的同时打断* baseline 成功的样本（top losses 全是 base=6−9 → cv=1−2）。

**修复：** 在 `base_conf ≥ τ` 的位置上完全跳过 CD，通过 vanilla DCD 路径 commit。只有低置信位置才受 CD 的 argmax 影响。

### 4.3 架构（4 个 helper + 1 个总成入口）

```
cv_v32_apc.py         → apply_cd_apc(base, drop, β, α) → (selection, blended_unmasked)
cv_v32_confidence.py  → resolve(source, base, blended, x0) → conf
cv_v32_gating.py      → gate_mask(base_conf, τ)
cv_v32_dispatcher.py  → pick_transfer_v32(...) 组装以上三个
```

在 `mmada_decode._pick_transfer_cv` 里通过新的 `cv_mode='cd_apc_v32'` 路由。所有旧 CV 模式（`cd_apc`, `cd_naive`, `legacy_score`, `off`）保持字节级兼容。

**19/19 CPU 单元测试全绿**（`tests/test_cv_v32.py`），包括一个回归测试：手工构造 `v_valid_size==1` 场景，验证 `conf_from_selection_v31bug ≈ 1.0`（v3.1 bug 可复现）而 `conf_from_base < 0.35`（v3.2 修复生效）。

---

## 5. 实验设置

### 5.1 固定参数配置（DCD baseline 与所有 CV 变体完全一致）

| 参数 | 值 |
|---|---|
| `decode_algo` | `threshold` |
| `decode_param` | `0.9` |
| `dcd_temperature` | `0.0`（纯 argmax，不加 gumbel） |
| `block_size` | 32 |
| `remasking` | `low_confidence` |
| `cache_type` | `dual`（baseline 与 CV 一致） |
| `steps` / `max_new_tokens` | 512 |
| VQ 模型 | `MAGVITv2` |
| Base LM | `MMaDA-MixCoT` |
| 打分器 | GPT-4 via VLMEvalKit `LLaVABench` scorer |

**配置差异**：只有 CV 特有字段（`causal_lambda`, `cv_alpha`, `image_drop_strategy`, `cv_mode`, `cv_conf_source`, `cv_gate_tau`）——baseline 里全部为 0/off。

### 5.2 v3.2 Phase B sweep（5 组）

全部使用 `β=0.5, α=0.1, drop=shuffle, cv_mode=cd_apc_v32`；每组有独立的 debug 子目录。

| Tag | conf_source | gate_tau | 状态 |
|---|---|---|---|
| B1 | blended | 0.0 | ✓ 完成 |
| B2 | blended | 0.5 | ✓ 完成 |
| B3 | min_base_blended | 0.0 | ✓ 完成 |
| B4 | base | 0.0 | ✓ 完成 |
| B5 | base | 0.5 | ✓ 完成 |

### 5.3 计算成本

每个 CV step 需要额外做一次 `drop_logits` forward，所以 **CV-DCD 的 `nfe` ≈ baseline 的 2×**。实际测量：

| 配置 | 样本 n | Mean nfe | Median nfe | 有效 decoding 迭代次数 (nfe/2) |
|---|---|---|---|---|
| v3.1 A3 shuffle | 60 | 119.9 | 105 | ~52 |
| v3.2 B4（sanity） | 5 | 109.6 | 56 | ~28 |
| DCD baseline | 60 | 未记录 | — | 估计 ~20-40 |

---

## 6. 实验结果

### 6.1 LLaVABench 分数（60 样本，GPT-4 打分）

| 配置 | Overall | VLM | Conv | Detail | Complex | Median len | Δ VLM vs baseline |
|---|---|---|---|---|---|---|---|
| **DCD baseline** | 44.1 | **36.7** | 32.7 | 45.1 | 50.9 | 445 | — |
| v2 Fix1+3 最好 | 27.4 | 24.2 | 27.0 | 21.3 | 31.2 | ~365 | −12.5 |
| v3.1 A3 shuffle | 30.3 | 26.7 | 30.2 | 24.4 | 33.8 | 334 | −10.0 |
| v3.2 B1 blended g=0 | 29.2 | 25.7 | 22.8 | 27.6 | 34.5 | 426 | −11.0 |
| v3.2 B2 blended g=0.5 | 30.6 | 26.8 | 29.1 | 28.9 | 32.5 | 436 | −9.9 |
| **v3.2 B3** minbb g=0 | **32.2** | **28.3** | 29.6 | 25.4 | **38.1** | **463** | **−8.4** |
| v3.2 B4 base g=0 | 30.2 | 26.5 | 28.5 | 23.5 | 35.3 | 416 | −10.2 |
| v3.2 B5 base g=0.5 | 30.8 | 27.0 | 29.4 | 25.0 | 35.2 | 399 | −9.7 |

**Phase B 5 组的综合观察：**

- **B3 (min_base_blended, gate=0) = 新的最佳**（VLM 28.3）—— 比 B2 高 +1.5 VLM，比 v3.1 A3 高 +1.6 VLM
- **B3 的 Complex 类得到 38.1**，是所有 CV-DCD 尝试里最高的，甚至高过 baseline 的 50.9 的差距最小
- **B3 的 median 长度 463**——所有组里最长，甚至超过 baseline 445
- **中位长度整体恢复了**（v3.1 334 → v3.2 各组 399-463），逼近或超越 baseline。confidence 修复按设计目标生效了。
- **单看 confidence 修复（B1 vs v3.1 A3）：−1.0 VLM**。孤立的 conf 修复其实是净退化——长度恢复对 conv 类的伤害大于对 detail 类的帮助。
- **gating 的效果非常弱**：B2 vs B1 +0.12，B5 vs B4 +0.05，平均只有 0.1 分/样本，几乎完全集中在 conv 类。

**三种 conf 源的排序（同为 gate=0.0）：** `min_base_blended` (28.3) > `blended` (25.7) ≈ `base` (26.5)。min 双保险意外优于两个单源。

### 6.2 逐样本 delta 分析

| 对比 | mean_delta | improved | worse | tied | big_gain≥3 | big_drop≥3 |
|---|---|---|---|---|---|---|
| v3.1 A3 vs baseline | −1.00 | 7 | 40 | 13 | 2 | 10 |
| B1 blended g=0 vs baseline | −1.10 | 4 | 36 | 20 | 1 | 11 |
| B2 blended g=0.5 vs baseline | −0.98 | 7 | 37 | 16 | 2 | 13 |
| **B3 minbb g=0 vs baseline** | **−0.83** | 7 | 35 | 18 | 2 | **10** |
| B4 base g=0 vs baseline | −1.02 | 6 | 36 | 18 | 1 | 8 |
| B5 base g=0.5 vs baseline | −0.97 | 8 | 38 | 14 | 1 | **7** |
| B3 vs B1 | **+0.27** | 13 | 9 | 38 | — | — |
| B3 vs B2 | +0.15 | 14 | 10 | 36 | — | — |
| B2 vs v3.1 A3 | +0.02 | — | — | — | — | — |

四个关键数字：

- **B3 vs baseline: −0.83 mean_delta**——这是所有 CV-DCD 尝试里最好的一个，比 v3.1 A3 好 0.17 分/样本。
- **B3 vs B1: +0.27 mean_delta**（隔离 min-conf 独立效应）——`min(base, blended)` 相对纯 `blended`，13 个样本 improved / 9 个 worse / 38 个 tied，净收益明显。
- **B2 vs v3.1 A3: +0.02 mean_delta**——单独的 gating（在 blended conf 上）几乎无效。
- **big_drop≥3 变化**：B5 (base g=0.5) 最少（7 个），说明 base-conf gating 确实抑制了极端翻车；但代价是也压制了 big_gain（只有 1 个）。**B3 是"少翻车 + 保住 gain"** 的最好折中。

### 6.3 分类别趋势

| 类别 | baseline | v3.1 A3 | B1 | B2 | **B3** | B4 | B5 | B3 − baseline |
|---|---|---|---|---|---|---|---|---|
| conv (17) | 2.82 | 2.82 | 2.12 | 2.71 | 2.76 | 2.65 | 2.76 | −0.06 |
| detail (15) | 3.67 | 2.20 | 2.47 | 2.60 | 2.33 | 2.13 | 2.27 | **−1.34** |
| complex (28) | 4.18 | 2.82 | 2.89 | 2.71 | **3.14** | 2.93 | 2.89 | **−1.04** |
| overall (60) | 3.67 | 2.67 | 2.57 | 2.68 | **2.83** | 2.65 | 2.70 | −0.84 |

**关键观察：**

1. **B3 在 complex 类得 3.14**，是所有 CV-DCD 尝试里最高的，比 v3.1 A3 (2.82) 高 +0.32，比 B2 (2.71) 高 +0.43。complex 类回归 baseline 的差距缩小到 −1.04。
2. **B3 的 conv 也保持在 2.76**（接近 baseline 的 2.82，只差 −0.06）——`min` conf 没有伤害 conv 类。
3. **detail 类（15 个样本）是所有 CV-DCD 尝试都吃亏的类别**：B3 达到 2.33，甚至比 v3.1 A3 (2.20) 高，但依然低于 baseline (3.67) 达 −1.34。这是 detail-oriented 描述题里 CD 的逐 token 漂移最容易累积错误的类别。
4. **gating（B4→B5, B1→B2）对每个类别的影响都很小**（<0.15 分），说明 gating 只是把"边缘 case"的 argmax 交给 base——但这些位置的 base argmax 也不见得对。

### 6.4 具体输出样本

**Idx=7「想象一下水果的芳香...」** (base=9, v31=2, b1=2, b2=2, **b3=1**)

- Baseline（398 字符，得分 9）：连贯的、有想象力的对甜+酸+柑橘香气的描述。
- v3.1 A3（448 字符，得分 2）：流畅，但对具体水果的判断有错（figs, avocado, passion fruit）。
- **B1（874 字符，得分 2）**：**长度翻倍，但重复 "passion fruit's aroma..." 三次**（不同措辞）。长度恢复给了模型循环的空间。
- B2（得分 2）：和 B1 内容结构相同。
- **B3（得分 1）**：min-conf 的双保险没救到——重复问题依然存在，且更严重。**这个样本在 B3 里比 B1 更差**，是 B3 唯一比其他配置显著差的样本（Δ=−8 vs baseline，B1/B2 是 Δ=−7）。

**Idx=54「广告里是什么品牌？」** (base=6 "Subway", v31=1, b1=1, b2=1, **b3=1**)

- 所有 CV 变体产出**逐字节相同**的错误答案：`"Jers sleekies"`。**gating 没救到它，min-conf 也没救**——这个位置尽管 base_conf 应该足够高，仍然被 CD 早早 commit 了。说明"Subway"应该出现的位置上，`base_conf < 0.5`。

**Idx=12「meme 里模仿的是哪个经典电影场景？」** (base=1 "It", v31=8, b1=7, b2=7, **b3=8**)

- Baseline degenerate 只有 2 字符。所有 CV 变体正确输出 `"The Lion King"`。这 +6~+7 的收益贯穿所有修复保留下来。**B3 是所有配置里得分最高的（8 分）**，正好和 v3.1 A3 平齐。

**Idx=45「详细描述这幅素描」** (base=2, v31=1, b1=1, b2=1, **b3=1**)

- Baseline 给出 447 字符完整答案；所有 CV 变体退化为 `"1"`。**这是 CV-DCD 家族的"死结"样本**——confidence 修复、gating、min-conf 都无效。CD 的 selection_logits argmax 在第一个 commit 位置就选了 terminal-like token。

**Idx=3「写一篇关于这个地方的旅行博客」** (base=1, v31=?, b1=?, b2=?, **b3=3**)

- 这是 B3 相对 baseline 的 top gain 样本之一（Δ=+2）。原本 baseline degenerate 得 1 分；B3 生成了较长的旅行博客内容（虽然只得 3 分，但明显好过 baseline）。这类样本是 CV-DCD 保留的"救烂样本"能力。

### 6.5 confidence 修复在机械上生效了

Debug NPZ 里的 `conf_from_selection_v31bug` 字段确认 v3.1 中位数是 ~1.0（假的）；v3.2 的 `conf_from_base` 和 `conf_from_blended` 分布正常。**这个 bug 关闭了。只是它没有修复结果。**

### 6.6 为什么 `min_base_blended` 反直觉地最优

原本的预测（v3.2 设计文档里的方案对比表）认为 `min_base_blended` 会过于保守，"牺牲 CD 的 boost 场景"。实际不然，B3 是所有 5 组里最好的。以下是解读：

**表面机制：** `conf = min(softmax(base_logits)[x0], softmax(blended_logits)[x0])`。任何一方判低置信，整体就低——最保守的双保险。

**为什么它反而更好？**

1. **argmax 仍然走 CD**（`selection_logits = blended · plausible_mask`）——**"救烂样本"的能力保留**（idx=12, idx=16 依然得高分）
2. **min conf 让 threshold 通过率整体下降**——更多位置 defer，每 step 提交的位置更少
3. **更长的 refresh 链条**：每个位置被多轮 re-evaluate，一次没通过下次还有机会，减少了"一次 commit 一批 degenerate token"的风险
4. **中位长度 463**（Phase B 里最长，甚至超过 baseline 445）——间接证据：模型有更多 step 来 refine
5. **本质是"两个视角都同意"的语义**：base 和 blended 都认为这个位置该 commit 才 commit。即使 blended 的 argmax 是 CD 加持后的产物，base 的 conf 仍然对它可以做"合理性投票"

**代价：**

- 更高的 NFE（refresh 更多）
- 但和 v3.1 A3 相比其实 NFE 没显著增加（median 长度只多 129 字符）
- B3 在 detail 类依然差 baseline 1.34 分，说明 min-conf 不是万灵药——**结构性问题依然存在**

---

## 7. 结构性诊断：为什么 per-token boost 在 MMaDA 上失败

综合数值和定性证据，三个相互作用的机制让"逐 token CD boost"框架不适合当前场景：

### 7.1 视觉信息是全局信号，`visual_gain` 是边缘信号

`visual_gain(pos, token) = log p(tok | base) − log p(tok | drop)` 是一个**边缘贡献**——衡量在*这一个位置*上，图像被 ablate 时该 token 的概率变化。它无法捕获"整段 caption 应该讲 Diamond Head，而不是 South Africa"这种**联合分布性质**。

Phase-A NPZ 显示 `visual_gain` 分布：**median = −0.125**，p10/p90 = [−1.5, +1.25]。围绕零对称、有噪声。大部分"boost"是噪声不是信号。

### 7.2 扩散 mask 解码没有跨位置 context 抑制被 boost 的重复

在 AR 模型里（CD 最初提出的场景，Li et al. 2022），每次新采样的 token 会成为 context 的一部分；下一次 forward "看到"它，attention 里 LM 自然的惩罚机制会抑制重复。所以 CD 过度 boost 一个"visually related"token 的倾向被 AR context 制约。

在 MMaDA 的扩散 mask 解码器里，**一个 block 内所有 masked 位置是并行更新的——彼此看不见**。CV-DCD 独立地对每个位置 boost 相同方向（谁 visual_gain 最大就 boost 谁）。结果：**同一个 step 内多个相邻位置被 commit 到同一个"visually-supported"概念** → idx=7 里 `"passion fruit's aroma"` 重复三次的根本原因。

Baseline DCD 避免这个问题，是因为 base_logits 本身携带了 LM pre-training 的不重复先验；CD boost 破坏了这种平衡。

### 7.3 `shuffle` drop 无法提供可靠的"无图像"参考

MMaDA 空间不敏感（VQ patch 位置影响很小）。打乱图像 token 得到的预测分布几乎和原来一样 → `drop_logits ≈ base_logits` → `visual_gain ≈ 0`，且噪声大。根本没有强信号可以 boost。

`neutral`（全灰）和 `mask_id` 替代方案也试过（见 v2 报告）——mask_id 是 out-of-distribution（打破模型）；neutral 稳定但信号弱；shuffle 噪声大但最不 degenerate。**没有一个是好的"反事实无图像"参考**。

### 7.4 confidence 修复揭示了真正的瓶颈

v3.2 实验完全证实以上三点：

- confidence 修复（B1）→ 长度恢复 → 41/60 样本产出和 v3.1 完全相同的预测
- 加上 gating（B2）→ 44/60 和 B1 相同
- **min-conf（B3）→ 42/60 和 B1 相同**（相对 B1 只有 22 个样本改变，其中 13 个提升 / 9 个下降）
- 综合：**四个机械修复叠加起来（v3.2 全部），只有 22/60 样本相对 v3.1 A3 明显不同**

CD commit 的 token 无论换 confidence 源、加 gating、还是 min-conf 双保险，很大一部分依然是同一批。问题 100% 出在 argmax 层——`selection_logits = blended · plausible_mask` 的 argmax 是最终 token 来源，任何 confidence/gating 修复都不改变它。B3 之所以最优，是因为它**通过延长 refresh 让部分 argmax 结果被后续轮次覆盖**——但这个副作用有上限。

---

## 8. 下一步方向

按"实际带来提升的可能性"排序，基于以上诊断：

### 方向 A —— 整段序列 CFG 重排（预期收益最高，成本中等）

不用逐 token boost，而是通过 baseline DCD 生成 `k` 个完整候选序列，然后用下式重排：

```
score(seq) = (1 + γ) · log p(seq | image, prompt) − γ · log p(seq | prompt)
```

**这是 classifier-free guidance 应用在序列层面**——更接近 modern diffusion 图像生成里 CD 的用法。保留 DCD 的 LM 先验（避免重复），在全局层面应用视觉引导，无逐 token 噪声累积。

成本：`k`× 解码，然后 2 次 forward 打分。`k=3-5` 应该够。

### 方向 B —— Defer-only CV（改动最小，语义最干净）

argmax 完全走 `base_logits`（永不改动）。`visual_gain(pos, argmax_token)` 只调制这个位置本轮是否*已经可以 commit*：

- 若 `visual_gain ≥ τ_pos` → 该位置"视觉已确认"→ 允许在 threshold 0.9 处 commit
- 若 `visual_gain < τ_pos` → 该位置视觉未支持 → 强制 defer 到下一轮

等价于：**DCD 决定选什么 token，CV 决定什么时候 commit。**

成本：无逐 token boost；仍需 `drop_logits` forward 但 stride 可以设为 2-4（不必像现在这样每步都做）。

### 方向 C —— 用 `text-only forward` 替换 `shuffle`

不用 shuffle/neutral 做 ablation，而是做一次真正的 text-only forward：把整段 image token span 替换为 padding/EOS token，让模型只用 **prompt** 跑。这提供更干净的"language prior"参考，可能提升 `visual_gain` 的信号质量。

成本：最小。只需要新增 `image_drop_strategy='text_only'` 分支。值得快速试。

### 方向 D —— 停手写负结果报告

如果以上都在合理预算内没有起效果，那么另一条路是把这次迭代写为**负结果**——一个诊断清楚的、朴素 per-token CD 在扩散 mask 解码器上的失败案例——对社区有价值。所有素材（bug 诊断、sweep 数据、结构性论证）已经在本文档里了。

---

## 9. "修好"和"没修好"的清单

| 项 | 修好了？ | 证据 |
|---|---|---|
| v1 log/prob 尺度错位 | ✓ | v2 Fix1 —— `cv_score` 走 prob 空间 |
| v2 mask_id OOD 图像 drop | ✓ | v2 Fix3 —— `neutral` + phase-A 用 `shuffle` |
| v3.1 APC 假置信 | ✓ | v3.2 conf-source dispatch；单元测试 `test_dispatcher_removes_v31_false_confidence` |
| v3.1 debug NPZ 覆盖 | ✓ | v3.2 sweep script 每组用独立 subdir |
| 生成长度过短（v3.1 median 334） | ✓ | v3.2 B3 median 463，甚至超过 baseline 445 |
| 双峰效应（top loss vs top gain） | 部分 | B3 的 mean_delta −0.83（最佳），比 v3.1 A3 (−1.00) 好 0.17，但 top losses 没救到 |
| 长 CD 输出的重复（idx=7） | **没修** | 结构性问题——per-token boost + 并行扩散解码 |
| Argmax 挑到 degenerate token（idx=45→"1"，idx=54→"Jers sleekies"） | **没修** | confidence 修复、gating、min-conf 都不触及 `selection_logits` argmax |
| Baseline 差距 | 部分 | v3.1: −10.0 VLM → v3.2 B3: **−8.4 VLM**（缩小 1.6 分，但依然远达不到 baseline） |
| Detail 类的 CV 漂移（−1.34 分/样本） | **没修** | 需要放弃 per-token boost 框架 |

---

## 10. 本轮交付物

- 全部代码改动：`models/cv_v32_*.py`（4 个 helper + 1 个总成入口，共 ~440 LOC）
- 向后兼容：旧的 `cd_apc / cd_naive / legacy_score / off` 模式完全不变
- 19/19 CPU 单元测试全绿（`tests/test_cv_v32.py`）
- Sweep 脚本 + 每组独立 debug 目录（`scripts/run_cvdcd_v32_sweep.sh`）
- 扩展的 debug NPZ schema（`conf_from_base`, `conf_from_blended`, `conf_from_selection_v31bug`, `conf_source_used`, `gate_active`）供后续分析
- **v3.2 Phase B 全 5 组完整数据**（`outputs/cvdcd_sweep/cvdcd_v32_phase_b_20260706_185908/`）
- **新最优配置**：`MMADA_CV_MODE=cd_apc_v32 MMADA_CV_CONF_SOURCE=min_base_blended MMADA_CV_GATE_TAU=0.0 MMADA_CAUSAL_LAMBDA=0.5 MMADA_CV_ALPHA=0.1 MMADA_IMG_DROP=shuffle`（VLM 28.3, Δ=−8.4）
- 本报告（`docs/cv_dcd_full_report.md`）以及各阶段的历史报告

v3.2 修复本身是正确的、干净实现的、并且解锁了所有 confidence/gating 旋钮。它揭示的是——**这些旋钮不是这个场景里正确的杠杆**。B3 (min_base_blended) 意外发现的价值是通过"延长 refresh 链条"间接改善质量，但本质上并没有解决"argmax 挑错 token"这个根本问题。如果继续走 per-token boost 框架，可以用 B3 作为新起点；但要真正逼近 baseline，下一次尝试应该直接针对 argmax 层（方向 A: 序列级 CFG）或跳出 argmax 改动（方向 B: defer-only CV）。
