# CV-DCD v3.1 (APC-Style Contrastive Decoding) — Phase A Report

**Run:** `cvdcd_v31_phase_a_20260706_174459`
**Data:** LLaVABench (60 samples), GPT-4 judge
**Sweep:** 3 configs (β = 0.5 / 1.0, drop = neutral / shuffle), APC α = 0.1, `cv_mode=cd_apc`
**Date:** 2026-07-06

---

## 1. TL;DR

- **A3 (β=0.5, shuffle) = 30.3 overall / VLM 26.7** — 相对 v2 Fix1+3 最好组 (`27.4/24.2`) 提升 **+2.5 VLM**，方向正确。
- **仍然显著落后 DCD baseline (44.1 / VLM 36.7)**，**−10.0 VLM 分**。
- **A2 β=1.0 明显 collapse**（VLM 18.7，median 长度 218 vs baseline 445），说明当前 CD 融合 β 敏感度很高。
- **调试数据揭示核心问题：APC α=0.1 过严 → v_valid_size median = 1 → 只有 1.13% steps 真正换 token；且换的地方大多"换错方向"，让 baseline 原本正确的样本退化。**
- **Per-sample 分布出现"双峰效应"**：A3 vs baseline 有 7 个样本改善（含 2 个 +6/+7 大改善），却有 40 个样本退化（10 个 ≥3 分大跌）。**这为 "gated CD"（只在低置信/低 base 分数 step 启用）方案提供了强证据。**

---

## 2. Phase A 分组打分

| 配置 | Overall | VLM | Conv | Detail | Complex | GPT4 判官 | vs DCD baseline VLM |
|---|---|---|---|---|---|---|---|
| **DCD baseline** | 44.1 | **36.7** | 32.7 | 45.1 | 50.9 | 83.2 | — |
| v2 Fix1+3 best (λ 变体) | 27.4 | 24.2 | 27.0 | 21.3 | 31.2 | 88.2 | −12.5 |
| **v3.1 A1** β=0.5 neutral | 29.3 | 25.8 | 29.1 | 25.0 | 31.9 | 88.2 | −10.9 |
| **v3.1 A2** β=1.0 neutral | 21.4 | 18.7 | 16.9 | 20.9 | 24.8 | 87.3 | −18.0 |
| **v3.1 A3** β=0.5 shuffle | **30.3** | **26.7** | 30.2 | 24.4 | 33.8 | 88.0 | −10.0 |

**观察：**
- A1 vs A3 相同 β 不同 drop：shuffle > neutral（+0.9 VLM），与 v2 阶段的观察一致。
- β 0.5 → 1.0 (neutral)：VLM 25.8 → 18.7（−7 分），**β 敏感、上限被 collapse 卡住**。
- CV 组 GPT4 判官打分（GPT-4 打给参考答案的分数）都 ~88，baseline 是 83.2；这是 relative-score 分母，所以 CV 组的相对分数被自然拉低，但 VLM 绝对分数仍显著低于 baseline。

---

## 3. 生成质量对比

**长度分布（60 样本合计）：**

| 配置 | mean_len | median_len | <40字符占比 |
|---|---|---|---|
| DCD baseline | 384.7 | 445 | 22/60 |
| A1 β=0.5 neutral | 325.8 | 365 | 26/60 |
| A2 β=1.0 neutral | 324.1 | **218** | 27/60 |
| A3 β=0.5 shuffle | 336.8 | 334 | 25/60 |

**Degenerate 案例（Index 45, "Describe this sketch in detail."）：**
- baseline：`"The sketch is a hand-drawn sketch of a person, likely a woman, with a neutral expression..."` （完整段落）
- **A1/A2/A3 均输出：`"1"`** — 单字符 degenerate，且三组一致。这说明当前 APC-CD 在少数关键 step 上"换错 token"就足以摧毁整个答案。

**Index 20, "What's the best weather... visiting this place?":**
- baseline：`"Summer"` （正确）
- **A1 / A2：`"No"`** — 从描述性回答退化为极短的 yes/no
- A3：`"Summer"` （保留正确）

---

## 4. Debug 诊断（仅 A3，其它两组的 debug 被覆盖）

**60 样本合计 15,360 CV pick steps 的分布：**

| 指标 | 值 | 解读 |
|---|---|---|
| Total CV pick steps | 15,360 | 60 样本 × 每样本 256 step |
| **Argmax changed by APC-CD** | **174 (1.13%)** | 只有 1.13% 的 step 实际换了 token |
| n_zero samples (完全没换) | 20/60 | 三分之一样本 CV 完全没有生效 |
| **v_valid_size (APC 后候选数) median** | **1.00** | median 只留 1 个候选 → 强制走 base argmax |
| v_valid_size p90 | 2.00 | 90% 情况只留 1-2 个 token |
| raw_conf_base median | 1.000 | 大部分 step baseline 本来极高置信 |
| raw_conf_effective median | 1.000 | CD 融合后仍然极高置信 |
| **visual_gain_at_x0 median** | **−0.125** | shuffle 图像下 x0 概率反而略降（说明当前 base argmax 已经是 visual-supported） |
| visual_gain p10 / p90 | −1.50 / +1.25 | 大部分 step 视觉扰动方向弱且不稳定 |

**按 category 拆分 argmax_changed 率：**

| Category | mean_change_rate | n |
|---|---|---|
| conv | 0.37% | 17 |
| detail | 1.80% | 15 |
| complex | 1.24% | 28 |

Detail 类样本换 token 最多但也是 degenerate 最严重的地方（Index 45 → "1"），说明 detail 长文本正是 CV 打断得最多的地方。

---

## 5. Per-Sample Score Delta 分布（相对 baseline）

| 配置 | mean_delta | improved | worse | tied | ≥3 分大跌 | ≥3 分大涨 |
|---|---|---|---|---|---|---|
| A1_b05_neutral | −1.08 | 7 | **37** | 16 | 12 | 2 |
| A2_b10_neutral | −1.80 | 1 | **43** | 16 | 17 | 0 |
| A3_b05_shuffle | −1.00 | 7 | **40** | 13 | 10 | 2 |

**A3 top-5 losses：** 都是长文本 & 需要 visual grounding 的 detail/complex 问题
- idx=7 (complex) base=9 → cv=2，−7：`"describe fragrance of the fruits"` 类问题被打崩
- idx=1 (detail) base=7 → cv=2，−5：`"Describe this photo in detail"`
- idx=54 (conv) base=6 → cv=1，−5：`"What brand is featured in this advertisement"`

**A3 top-5 gains：** 恰好是 baseline 表现极差的 conv 样本
- **idx=12 (conv) base=1 → cv=8，+7**：`"Which iconic movie scene is being parodied in the meme"` — CV 成功拉回 baseline 完全失败的样本
- **idx=16 (conv) base=2 → cv=8，+6**：`"Do you know who paint this?"` — 类似
- idx=41 (conv) base=6 → cv=8，+2

**关键观察 —— 双峰效应：**

CV-DCD 有能力在 **baseline 本来就烂（score ≤ 2）** 的样本上大幅拉分（idx=12/16 均 +6~7 分），但同时**在 baseline 本来就 OK（score ≥ 6）的样本上会打断**（top losses 全是 baseline 6~9 分的样本）。

**净收益负** 是因为"本来 OK 的样本"数量远多于"本来烂的样本"，加上单次伤害幅度也大（−4 到 −7 分）。

---

## 6. 根本原因诊断

### 6.1 APC α=0.1 过严

- v_valid_size median=1 意味着"绝大多数 step APC 只留下 base argmax 自己"→ 融合公式 `base + β * visual_gain` 只在极少数 step 有 candidate 参与竞争，且这种"竞争一定发生在低置信步骤"（那里 max_prob 较低，α*max_prob 阈值也低，多个 token 才能通过 APC）。
- 结果：CD **只在最不稳定的 step 上被激活**，方向敏感、噪声大。

### 6.2 shuffle 未提供稳定的 "no visual" 对照

`visual_gain` 的分布 median = **−0.125** 且 p10/p90 = **−1.5 / +1.25**，说明"打乱图像"对 x0 概率的影响并不系统偏向"降低"—— shuffle 之后模型经常仍然给 baseline argmax 相似或更高的 logit（因为 shuffle 保留了 patch 分布只是打乱位置，MMaDA 的空间敏感度可能弱）。这意味着 `visual_gain` 作为对比信号本身就**方向不明**。

### 6.3 融合逻辑独立于 base confidence

当前 v3.1 的 CD 在每一个 CV step 都会试图融合，**不管 base 本来置信度多高**。结果就是：
- 高置信 step：APC 通常只留 1 个 token → CD 什么都不做（占 98.9%）
- 低置信 step：APC 留 2+ token → CD 生效，**但方向随 visual_gain 噪声**

这就直接导致 §5 的 "双峰效应"：**能救的样本 base 本来就置信度低**（那些 CV 有机会介入的 step），**打崩的样本是 CD 换出了一个更差的 token**。

---

## 7. 与之前迭代对比

| 迭代 | 机制 | 主要问题 |
|---|---|---|
| v1 (log-score) | `cv_score = log_p + λ · visual_gain`，与 threshold 0.9 混用 | log/prob 尺度错位 → 退化为 top-1 commit |
| v2 Fix1+3 | 概率空间 `cv_score = raw_conf · exp(λ · visual_gain)` + neutral drop | 只改 commit 顺序，不改 token 选择 → 打断节奏但无 grounding 收益 |
| **v3.1 (APC-CD)** | logit-level `contrast = base + β · visual_gain`，APC α=0.1 遮罩 | **APC 过严 + visual_gain 方向不稳 + 无 gating → 双峰效应，净收益负** |

**方向上 v3.1 是唯一能真正改变 token 的实现**（v1/v2 只改顺序不改 token），也是唯一能在少数样本上产生 **+6/+7 大改善** 的实现。**问题在于代价 (10 个 −3~−7 分退化)** 远超收益 (2 个 +6~+7 分改善)。

---

## 8. 下一步候选（未决定）

不确定哪条最有前景，等你决定。三条独立路径都能验证一个具体假设：

### Option A：放宽 APC（最省时，直接 verify 假设 6.1）
- 扫 α ∈ {0.3, 0.5, 0.7} × β ∈ {0.3, 0.5}，drop 固定 shuffle
- 假设：α 放宽后 CD 生效面提升，`argmax_changed` 从 1.13% 升到 5-15%，可能拉回损失也可能加剧
- 风险：仍无 gating，双峰效应可能加剧
- 实验成本：3 × 60 样本 = ~35 分钟 + 打分

### Option B：加入 base-confidence gating（Verify 假设 6.3）
- 只在 `base_conf < τ`（τ ∈ {0.7, 0.8, 0.9}）的 step 启用 CD，其它步骤走 baseline
- 假设：直接消除"打断本来正确样本"的部分，保留"救烂样本"的部分
- 实现代价：`_pick_transfer_cv` 加一个 conf gate（~30 行代码）
- 实验成本：2-3 × 60 样本

### Option C：换 ablation 策略（Verify 假设 6.2）
- shuffle → **spatial pixel shuffle**（不是 VQ token shuffle，而是 pixel 打乱后再 encode），或 **completely random VQ**
- 或走 CFG 路径：`base(text+image) − γ · base(text_only)`（不用第二次 forward 图像 tokens，用空图像）
- 目的：让 `visual_gain` 有更稳定的方向

### Option D：写复盘停手
- 把当前发现整理为最终报告，停止 v3.1 相关工作
- 转向：或复用 baseline，或换其它 method（e.g. visual token importance 训练监督）

---

## 9. 附录

### 9.1 生成产出位置
- Excel/CSV：`evaluation/VLMEvalKit/outputs/cvdcd_sweep/cvdcd_v31_phase_a_20260706_174459/A{1,2,3}_.../MMaDA-MixCoT-CV-DCD/T20260706_G/`
- Debug NPZ（只 A3 保留，前两组被覆盖）：`evaluation/VLMEvalKit/attention_analysis/cv_debug_v31/cvdcd_v31_phase_a_20260706_174459/`
- Sweep log：`/tmp/cvdcd_v31/cvdcd_v31_phase_a_20260706_174230.log`（前置试跑）

### 9.2 已知遗漏
- **Debug NPZ 覆盖问题**：sweep 脚本 3 组共用一个 `MMADA_CV_DEBUG_DIR`，导致只有最后一组 A3 的 debug 保留。如需 A1/A2 debug，需改脚本给每个 tag 单独目录后重跑。
- **未做 error/degenerate 类型再分类**：v2 阶段用 attention 特征把失败分为 `degenerate` / `error_long`，v3.1 未按此拆分统计。
