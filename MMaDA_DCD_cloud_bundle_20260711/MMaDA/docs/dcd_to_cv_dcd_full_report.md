# 从 DCD 到 CV-DCD / Defer-only CV 的完整迭代报告

**日期：** 2026-07-07  
**模型：** MMaDA-MixCoT  
**主要数据集：** LLaVABench  
**目标：** 总结从原始 DCD baseline 开始，所有 Causal-Visual DCD 方向的设计修改、实现修复、实验结果与当前结论。

---

## 0. 总览结论

从原始 DCD 到目前，我们主要尝试了三条路线：

1. **CV score 路线**：保持 base argmax 不变，只用视觉 gain 改 commit confidence。
2. **Token-level CD 路线**：在 argmax 前直接修改 logits，让视觉信号影响 token 选择。
3. **Defer-only 路线**：argmax 永远来自 base logits，视觉信号只决定是否延迟 commit。

最终结论如下：

| 阶段 | 代表机制 | 最好结果 | 结论 |
|---|---|---:|---|
| DCD baseline | `raw_conf >= 0.9` commit | full-60 VLM 36.7 | 当前强基线 |
| v1 CV score | `logp + lambda * gain` | 低于 baseline | log/prob 尺度错位 |
| v2 CV score | `raw_conf * exp(lambda * gain)` | 约 VLM 24.2 | 会把低置信 token 提前 commit |
| v3/v3.1 CD | APC + contrastive logits | VLM 26.7 | 能改 token，但有 APC 假置信 bug |
| v3.2 CD | 修复 confidence source | VLM 28.3 | 机械 bug 修完仍低于 baseline |
| v4 defer-only v2 | `soft/hard + text_only/logit` | 5-sample mean 3.8 | 数学修复成功，但 `tau=0` 干预过多 |
| v4 defer-only v3 | 负 `tau` 控制 defer rate | 5-sample best mean 3.8 | 干预率降下来了，但没有超过 baseline |

最重要的结论是：

**目前没有任何 CV-DCD 配置超过原始 DCD baseline。**  
我们确实修复了多个真实实现问题，但修完之后仍然低于 baseline，说明主要瓶颈不是单个 bug，而是 token 级视觉干预和 MMaDA 并行 mask decoding 的结构不匹配。

---

## 1. 原始 DCD baseline

### 1.1 基本机制

原始 DCD（Deferred Commitment Decoding）的核心逻辑是：

```text
x0 = argmax(base_logits)
raw_conf = softmax(base_logits)[x0]

if raw_conf >= threshold:
    commit x0
else:
    defer / refresh
```

当前实验中 threshold 使用：

```text
decode_algo  = threshold
decode_param = 0.9
temperature = 0.0
cache_type   = dual
```

DCD 的关键思想不是简单地取 top-1，而是：

> 如果当前位置置信度不足，就不要在上下文还没成型时急着提交，等下一轮 diffusion refresh 后再判断。

这个机制对 MMaDA 很重要，因为 MMaDA 是并行 mask decoder。许多 token 是在大量上下文仍为 mask 的情况下同时预测的。如果低置信 token 被过早 commit，后续 token 会沿着错误上下文继续生成。

### 1.2 Baseline 数字

当前讨论里有两个 baseline 来源，需要区分：

| baseline 来源 | 用途 | 结果 |
|---|---|---|
| `outputs/MMaDA-MixCoT-DCD-DualCache/T20260613_G` | full-60 历史 DCD 评测 | VLM 36.7 |
| `attention_analysis/llavabench_scored.json` | v3/v3.2/v4 分析使用的 canonical baseline | 60 样本 mean 约 3.82 |

最新 5-sample sanity 使用的 canonical baseline 是：

```text
indices = [3, 7, 12, 45, 54]

baseline scores:
3  -> 1
7  -> 7
12 -> 1
45 -> 1
54 -> 10

mean = 4.0
```

后续 v4 的 `lambda=0` sanity 都应优先对齐这个 canonical baseline，而不是另一次 GPT-4 打分的历史 xlsx 文件。

---

## 2. v1：最初的 Causal-Visual DCD

### 2.1 设计动机

最初想法是给 DCD 加入视觉因果证据。对每个候选 token 计算：

```text
visual_gain = log p(token | image) - log p(token | dropped_image)
```

如果一个 token 在有图像时概率更高，在去掉/扰乱图像后概率降低，就认为它更受视觉支持。

### 2.2 v1 公式

早期实现使用类似：

```text
cv_score = base_logp + lambda * visual_gain
```

然后用 `cv_score` 和 DCD 的 threshold `0.9` 比较。

### 2.3 问题

这个设计有一个根本 bug：**尺度错位**。

`base_logp` 和 `visual_gain` 是 log-space 数值，通常小于等于 0；而 DCD threshold `0.9` 是 probability-space confidence。把 log-space score 拿去和 `0.9` 比，会导致 commit 判定完全失真。

结果是：

- 大量 token 无法正常通过 threshold；
- 解码退化为 fallback/top-k commit；
- DCD 原本的 confidence gate 被破坏。

### 2.4 结论

v1 没有证明视觉信号无效，只证明了原始公式不能直接和 DCD threshold 结合。

---

## 3. v2：修正到 probability space

### 3.1 公式修复

v2 把 CV score 改成：

```text
cv_score = raw_conf * exp(lambda * visual_gain)
```

其中：

```text
raw_conf = softmax(base_logits)[x0]
```

这样 `cv_score` 回到了 probability-like scale，能和 threshold `0.9` 比较。

### 3.2 同步修复的其他问题

v2 还处理了 image drop 的问题。

早期 `mask` drop 会把 image span 替换成 text mask id，这对图像 token 是 out-of-distribution。后来加入或尝试了：

| drop strategy | 含义 | 评价 |
|---|---|---|
| `mask` | image token 替换为 mask id | OOD，不稳定 |
| `shuffle` | 打乱 image token | 常用，但 gain 噪声大 |
| `neutral` | 灰图 VQ code | 更合理，但效果仍有限 |
| `text_only` | image span 替换成 pad/eos | v4 中更稳定 |

### 3.3 新问题：提前 commit

v2 修复了尺度，但暴露出更深的问题：

```text
raw_conf < 0.9
visual_gain 很大
=> cv_score >= 0.9
=> 提前 commit
```

这和 DCD 的初衷相反。DCD 认为 `raw_conf < 0.9` 时上下文还不够可靠，应该延迟；CV score 却可能把这些 token 提前推过 threshold。

后果是：

- 低置信 token 在上下文仍残缺时被提交；
- 周围还有大量 mask；
- 后续 token 基于错误上下文继续生成；
- 容易出现重复、语法断裂、短答或幻觉。

### 3.4 实验结果

v2 最好大约：

```text
VLM ~= 24.2
DCD baseline = 36.7
差距约 -12.5
```

### 3.5 结论

v2 说明：

1. probability-scale 修复是必要的；
2. 但仅调 commit score 不够；
3. 如果视觉信号能把低置信 token 提前 commit，就会破坏 DCD 的 deferred commitment 机制。

---

## 4. v3 / v3.1：转向 token-level Contrastive Decoding

### 4.1 设计动机

v2 只改变 commit 时机，不改变选哪个 token。于是 v3 转向在 argmax 前修改 logits：

```text
blended_logits = base_logits + lambda * (base_logits - drop_logits)
```

等价写法：

```text
blended_logits = (1 + lambda) * base_logits - lambda * drop_logits
```

这和 CFG / Contrastive Decoding 的形式相同。

### 4.2 v3.1 APC

为了避免任意 token 被视觉差分拉上来，v3.1 加入 APC（Adaptive Plausibility Constraint）：

```text
valid(token) = base_logit(token) >= max_base_logit + log(alpha)
```

只在 plausible token 集合里使用 contrastive logits。

### 4.3 结果

v3.1 最好配置大约：

```text
VLM = 26.7
vs DCD baseline = -10.0
```

相对 v2 有提升，但仍明显低于 baseline。

### 4.4 发现的关键 bug：APC 假置信

v3.1 中 confidence 是从 APC-masked selection logits 上算的。问题是：

```text
APC 后只剩 1 个 token
softmax(masked_logits) = 1.0
```

也就是说，只要 plausible set size 是 1，confidence 就会被人为变成 1.0。DCD threshold `0.9` 会被绕过。

这不是模型真的自信，而是 masked softmax 的数学假象。

### 4.5 结论

v3.1 说明 token-level CD 确实能改变 token 选择，但也会引入新的 confidence bug，并且对好样本的损伤很明显。

---

## 5. v3.2：修复 APC confidence bug

### 5.1 核心修复

v3.2 把 token selection 和 confidence calculation 分开：

```text
selection_logits = APC-masked blended logits
blended_logits   = unmasked blended logits

x0 = argmax(selection_logits)
confidence = softmax(unmasked logits source)[x0]
```

这避免了 `v_valid_size=1 -> confidence=1.0` 的假置信。

### 5.2 新增配置

v3.2 加了：

```text
cv_mode = cd_apc_v32
cv_conf_source = blended | min_base_blended | base
cv_gate_tau = 0.0 or 0.5
```

三种 confidence source：

| `cv_conf_source` | 定义 | 语义 |
|---|---|---|
| `blended` | `softmax(blended_logits)[x0]` | 让视觉影响 confidence |
| `min_base_blended` | `min(base_conf, blended_conf)` | base 和视觉都同意才 commit |
| `base` | `softmax(base_logits)[x0]` | DCD commit 完全保持 base 语义 |

`cv_gate_tau` 的想法是：如果 base confidence 已经很高，则跳过 CD，避免破坏好样本。

### 5.3 实现模块

v3.2 拆出了四个 helper：

```text
models/cv_v32_apc.py
models/cv_v32_confidence.py
models/cv_v32_gating.py
models/cv_v32_dispatcher.py
```

并在 `mmada_decode._pick_transfer_cv()` 中 dispatch。

### 5.4 单测

`tests/test_cv_v32.py` 覆盖：

- APC mask；
- selection logits 与 confidence logits 分离；
- `v_valid_size=1` 假置信回归；
- gating；
- 三种 confidence source。

测试通过，说明 v3.2 的机械 bug 已修复。

### 5.5 Full-60 结果

v3.2 Phase B 结果：

| 配置 | conf source | gate | VLM | 结论 |
|---|---|---:|---:|---|
| B1 | blended | 0.0 | 25.7 | confidence 修复后仍弱 |
| B2 | blended | 0.5 | 26.8 | gating 小幅帮助 |
| B3 | min_base_blended | 0.0 | **28.3** | v3.2 最佳 |
| B4 | base | 0.0 | 26.5 | 语义干净但不强 |
| B5 | base | 0.5 | 27.0 | 极端翻车稍少 |

DCD baseline：

```text
VLM = 36.7
```

所以最佳 v3.2：

```text
28.3 - 36.7 = -8.4
```

### 5.6 结论

v3.2 很关键，因为它说明：

> 已知的主要实现 bug 修复后，CV-DCD 仍然远低于 baseline。

因此问题不是单个实现错误，而是 token-level logit boost 在 MMaDA 这种并行 mask decoder 上结构性不适配。

---

## 6. 结构性诊断：为什么 token-level 视觉 boost 不行

### 6.1 并行 mask decoding 的限制

自回归模型中，前面的 token 一旦生成，会成为后面 token 的上下文。但 MMaDA 的 mask decoding 是并行的：

```text
多个 masked positions 在同一步同时预测
```

因此逐 token 的视觉 boost 不能像 AR decoding 那样自然地通过上下文传播约束。每个位置都独立地受到 `visual_gain(pos, token)` 影响，容易导致：

- 重复；
- 局部 token 合理但整体句子不连贯；
- detail 问题上累积漂移；
- 好样本被打断。

### 6.2 visual_gain 噪声

尤其是 `shuffle` drop 下：

```text
drop image 并不是纯语言先验
而是一个被扰乱的 image-conditioned distribution
```

所以 `base_logits - drop_logits` 不一定代表“视觉因果贡献”，可能只是分布扰动。

### 6.3 经验现象

多轮实验反复出现同一模式：

```text
少数坏样本被救
更多好样本被破坏
```

例如：

- `#12` 有时能从 1 分被救到 7/8；
- 但 `#7`、`#54` 这类 baseline 高分样本经常被打到 1/2。

这说明视觉信号不是完全没用，而是 token-level 干预太不稳定。

---

## 7. v4 设计：放弃 token-level boost

v4 提出了两条方向：

| 方向 | 名称 | 核心思想 |
|---|---|---|
| A | Sequence-level CFG rerank | 先完整生成候选，再用视觉 CFG 分数重排 |
| B | Defer-only CV | 不改 argmax，只改是否 commit |

当前主要实现和实验的是方向 B。

---

## 8. v4 Defer-only CV

### 8.1 设计原则

Defer-only 的不变量是：

```text
x0 = argmax(base_logits)
```

视觉信号不允许改变 token，只能降低 confidence：

```text
eff_conf <= base_conf
```

也就是说：

```text
视觉只能 veto / defer
视觉不能 boost / 改 token
```

### 8.2 初始 veto 类型

实现了：

| veto | 公式 | 问题 |
|---|---|---|
| `hard` | `eff = base_conf if gain >= tau else 0` | 太硬 |
| `mult` | `eff = base_conf * sigmoid(beta*(gain-tau))` | `gain=0` 时砍半 |
| `min` | `eff = min(base_conf, sigmoid(beta*gain))` | 同样有零点问题 |

### 8.3 发现的问题：零点砍半

如果 `gain` 接近 0：

```text
sigmoid(0) = 0.5
```

那么：

```text
eff_conf = base_conf * 0.5
```

这不是“只惩罚视觉反证”，而是全局降低 confidence。会大幅减少每步 commit 数，使 DCD 的并行效率下降。

### 8.4 修复：soft veto

新增 `soft`：

```text
eff_conf = base_conf * exp(-beta * max(tau - gain, 0))
```

性质：

```text
gain >= tau  => eff_conf = base_conf
gain < tau   => eff_conf < base_conf
```

这修复了 `gain=0` 默认砍半的问题。

### 8.5 修复：gain type

新增：

```text
defer_gain_type = logit | logprob
```

两种 gain：

```text
logit:
gain = base_logit[x0] - drop_logit[x0]

logprob:
gain = log_softmax(base)[x0] - log_softmax(drop)[x0]
```

实验发现：

- `shuffle + logprob` 的 gain 基本塌到 0；
- `logit` gain 有足够展开；
- `text_only + logit` 更稳定。

### 8.6 修复：`lambda=0` dual-cache

还发现一个重要一致性问题：

`cv_dcd` 的 dual-cache 路径在 `causal_lambda=0` 时，首步仍然做了 paired forward：

```text
model([x, x_drop])
```

这会使 `lambda=0` 不严格等价 DCD。

修复后：

```text
causal_lambda == 0
=> 不构造 x_drop
=> 不做 paired forward
=> 单路 forward
```

并新增 CPU 回归测试，确认 `cv_dual_cache + lambda=0` 不再触发 batch=2 paired forward。

### 8.7 单测

`tests/test_defer_only.py` 当前覆盖：

- `gain_type=logit/logprob`；
- `soft/hard/mult/min` veto；
- `eff_conf <= base_conf`；
- `lambda=0` 等价；
- `drop_logits=None` 等价；
- `argmax` 永远来自 base；
- v3.2 路径不被破坏；
- `cv_dual_cache + lambda=0` 单路 forward 回归。

结果：

```text
24/24 passed
```

---

## 9. Defer-only smoke v2：数学修复后的第一轮结果

Run:

```text
defer_only_smoke_v2_20260707_013757
indices = [3, 7, 12, 45, 54]
```

### 9.1 配置与结果

| Tag | 配置 | mean | 分数 |
|---|---|---:|---|
| S1 | `soft + shuffle + logit + tau=0` | 2.6 | `{3:2, 7:1, 12:8, 45:1, 54:1}` |
| S2 | `soft + text_only + logit + tau=0` | 3.8 | `{3:1, 7:6, 12:1, 45:1, 54:10}` |
| S3 | `hard + shuffle + logit + tau=0` | 1.0 | 全 1 |
| S4 | `hard + text_only + logit + tau=0` | 3.8 | `{3:1, 7:6, 12:1, 45:1, 54:10}` |
| S5 | `soft + shuffle + logprob` | 1.0 | 全部很差 |
| S6 | `lambda=0 sanity` | 4.0 | canonical baseline 完全一致 |

### 9.2 关键观察

1. `lambda=0` sanity 通过。
2. `logit` gain 修复成功。
3. `shuffle + logprob` 基本无效。
4. `text_only + logit` 明显比 `shuffle` 稳定。
5. 但 `tau=0` 干预太多。

### 9.3 gain 分布

v2 debug 统计：

```text
shuffle/logit:
gain < 0.0   = 27.5%
gain < -0.5  = 15.9%
gain < -1.0  = 8.7%
gain < -1.5  = 3.9%

text_only/logit:
gain < 0.0   = 30.0%
gain < -0.5  = 21.2%
gain < -1.0  = 14.0%
gain < -1.5  = 9.6%
gain < -2.0  = 5.9%
```

结论：

```text
问题不是 gain 没信号
问题是 tau=0 太宽松
```

如果 `tau=0`，所有轻微负 gain 都会触发 veto，导致 27-30% 的 committed positions 被压 confidence，破坏 DCD 的并行 commit 节奏。

---

## 10. 为什么应调 tau 而不是 beta

soft veto：

```text
eff_conf = base_conf * exp(-beta * max(tau - gain, 0))
```

DCD commit 条件：

```text
eff_conf >= 0.9
```

当 `gain < tau` 时：

```text
base_conf * exp(-beta * (tau - gain)) >= 0.9
```

等价于：

```text
gain >= tau - log(base_conf / 0.9) / beta
```

如果典型：

```text
base_conf = 0.95
log(0.95 / 0.9) ~= 0.054
```

则 `tau=0` 时：

| beta | 实际 defer 边界 |
|---:|---:|
| 1.0 | `gain < -0.054` |
| 0.5 | `gain < -0.108` |
| 0.25 | `gain < -0.216` |

这几个边界都仍然接近 0，仍会吃掉大量轻微负 gain。

所以：

```text
beta 控制 penalty 深度
tau 控制 intervention threshold
```

当前问题是 intervention rate 过高，因此主旋钮应该是 `tau`。

---

## 11. Defer-only smoke v3：负 tau 结果

Run:

```text
defer_only_smoke_v3_20260707_021722
indices = [3, 7, 12, 45, 54]
```

### 11.1 配置与结果

| Tag | 配置 | mean | 分数 |
|---|---|---:|---|
| S1 | `hard + shuffle + tau=-1.0` | 1.2 | `{3:1, 7:2, 12:1, 45:1, 54:1}` |
| S2 | `soft + shuffle + tau=-1.0` | 2.2 | `{3:1, 7:7, 12:1, 45:1, 54:1}` |
| S3 | `soft + shuffle + tau=-1.5` | 2.0 | `{3:1, 7:6, 12:1, 45:1, 54:1}` |
| S4 | `hard + text_only + tau=-1.5` | **3.8** | `{3:1, 7:6, 12:1, 45:1, 54:10}` |
| S5 | `soft + text_only + tau=-1.5` | 3.0 | `{3:1, 7:2, 12:1, 45:1, 54:10}` |
| S6 | `soft + text_only + tau=-2.0` | 3.0 | `{3:1, 7:2, 12:1, 45:1, 54:10}` |
| S7 | `lambda=0 sanity` | 4.0 | `{3:1, 7:7, 12:1, 45:1, 54:10}` |

### 11.2 干预率

debug 中 `defer_active`：

| Tag | defer_active |
|---|---:|
| S1 `hard shuffle tau=-1.0` | 6.48% |
| S2 `soft shuffle tau=-1.0` | 5.55% |
| S3 `soft shuffle tau=-1.5` | 2.19% |
| S4 `hard text tau=-1.5` | 7.81% |
| S5 `soft text tau=-1.5` | 7.42% |
| S6 `soft text tau=-2.0` | 5.00% |
| S7 `lambda=0` | 0.00% |

这验证了：

```text
负 tau 确实能把干预率从 27-30% 降到 5-8%
```

### 11.3 但效果没有提升

关键样本：

```text
#7 baseline=7:
S2=7 保住
S3=6 小跌
S4=6 小跌
S5/S6=2 明显下降

#54 baseline=10:
text_only 三组保住 10
shuffle 三组全掉到 1

#12 baseline=1:
所有 v3 配置都没救回来
```

### 11.4 nfe / 解码效率

虽然 debug 里的 `defer_active` 降下来了，但 nfe 仍明显高于 baseline sanity。

meta 中记录的 nfe 大致显示：

```text
S7 lambda=0 sanity: avg nfe 约 50
S2 soft shuffle tau=-1.0: avg nfe 约 124
S4 hard text tau=-1.5: avg nfe 约 166
S5 soft text tau=-1.5: avg nfe 约 163
```

这说明当前 debug 的 `defer_active` 只记录最终 committed positions 的统计，不能完全代表所有候选位置上被 veto 导致未提交的真实影响。真实解码效率仍然被拖慢。

### 11.5 v3 结论

负 `tau` 的方向验证了一半：

```text
干预率确实降下来了
但质量没有超过 baseline
```

当前最好：

```text
hard + text_only + logit + tau=-1.5
mean = 3.8
```

它接近 baseline，但没有超过 baseline，也没救回 `#12`。

---

## 12. 当前总判断

### 12.1 已经确认正确的部分

1. v1 的 log/prob 尺度错位已定位。
2. v2 的提前 commit 问题已定位。
3. v3.1 的 APC 假置信 bug 已定位并在 v3.2 修复。
4. `lambda=0` CV dual-cache 不等价 DCD 的问题已修复。
5. `mult/min` 的 `gain=0` 砍半问题已修复。
6. `logit` gain 比 `logprob` gain 更适合当前工程目标。
7. `text_only` drop 比 `shuffle` 更稳定。
8. `tau` 是控制 intervention rate 的主旋钮。

### 12.2 仍未解决的问题

1. Defer-only 的 token-level veto 仍然没有找到超过 baseline 的 sweet spot。
2. 即使 `defer_active` 降到 5-8%，nfe 仍显著增加。
3. `shuffle` 方向持续不稳定，尤其破坏 `#54`。
4. `text_only` 虽稳定，但目前只是接近 baseline，不能带来净提升。
5. `#12` 的 rescue 在 v2 出现过，但负 tau v3 中消失，说明 rescue 依赖更强干预，而强干预又会破坏好样本。

### 12.3 当前最佳候选

```text
hard + text_only + logit + tau=-1.5
```

原因：

- 5-sample mean=3.8，最接近 baseline；
- `#54` 保持 10；
- `#7` 只从 7 降到 6；
- 干预率约 7.8%；
- 比 soft text variants 更稳。

但它仍不是成功方案，因为：

- mean 低于 `lambda=0` baseline 4.0；
- 没救 `#12`；
- nfe 明显增加。

---

## 13. 下一步建议

### 13.1 不建议直接 full sweep

当前没有一个 5-sample 配置超过 baseline。直接 full sweep 成本高，成功概率低。

### 13.2 若继续 defer-only，建议只做更小范围

围绕当前最佳方向：

```text
hard + text_only + logit
```

继续试更保守的 tau：

```text
hard text_only tau=-2.0
hard text_only tau=-2.5
hard text_only tau=-3.0
lambda=0 sanity
```

目标：

```text
#7 >= 7
#54 = 10
#12 是否有机会 > 1
avg nfe 接近 lambda=0 sanity
```

如果更保守的 tau 仍不能提升，则 defer-only 方向基本可以判定没有有效 sweet spot。

### 13.3 更推荐转向 Sequence-level CFG rerank

原因：

1. token-level boost 已经失败；
2. token-level defer-only 也接近失败；
3. sequence-level rerank 不破坏 DCD 的 commit 过程；
4. 视觉信号在完整答案级别判断，可能更适合 LLaVABench 这种整体质量评分。

推荐下一阶段：

```text
K 个 DCD candidates
计算 sequence-level visual CFG score
选择最高分候选
```

这条路线保留 DCD 的强语言结构，又让视觉信号在更稳定的序列层面发挥作用。

---

## 14. 一句话总结

从 DCD 到 CV-DCD 的所有迭代说明：

> 视觉信号本身不是完全没用；真正的问题是把视觉信号用于逐 token 的 commit 或 argmax 干预时，会破坏 MMaDA 并行 mask decoding 的节奏和语言结构。当前 defer-only 已经修正数学问题，但仍未超过 baseline。下一步应优先转向 sequence-level rerank，或者只对 `hard + text_only + logit` 做极小范围的保守 tau 验证。
