# CV-DCD 最新程序结构与创新点

> 更新时间：2026-07-11  
> 代码根目录：`/home/user/dcd/MMada_DCD/MMaDA`  
> 本文仅描述当前程序结构、算法和接口。

## 1. 程序定位

当前程序在 MMaDA 并行 mask decoding 上实现统一的 DCD/CV-DCD 推理解码框架。

主要能力：

1. 使用 DCD 控制 token 的不可逆提交时机；
2. 对原图输入和图像消融输入执行 paired forward；
3. 从两套 logits 中构造视觉对比信号；
4. 支持 `cd_apc`、`cd_naive`、`defer_only`、`legacy_score` 和 `off`；
5. 统一复用 block decoding、cache 更新与 token transfer；
6. 支持 λ=0 identity、逐步 debug 和 NFE 统计；
7. 全部功能位于推理阶段，不需要重新训练模型。

---

## 2. 目录结构

```text
MMaDA/
├── models/
│   ├── modeling_mmada.py
│   ├── mmada_decode.py
│   ├── cv_common/
│   │   ├── image_drop.py
│   │   ├── paired_forward.py
│   │   ├── log_prob.py
│   │   └── types.py
│   └── defer_only/
│       ├── dispatcher.py
│       ├── veto.py
│       └── __init__.py
├── tests/
│   ├── test_cv_common.py
│   ├── test_cv_dcd_smoke.py
│   ├── test_cd_style_smoke.py
│   └── test_defer_only.py
└── evaluation/VLMEvalKit/
    ├── run.py
    ├── vlmeval/vlm/mmada/mmada.py
    ├── scripts/
    └── attention_analysis/cv_debug_io.py
```

### 2.1 `models/modeling_mmada.py`

负责 MMaDA 生成入口：

- 接收 `decode_strategy`；
- 补全 `MMaDADecodeConfig`；
- 推断视觉 token span；
- 分发 plain DCD 或 CV-DCD；
- 接收 token 和 debug 信息。

### 2.2 `models/mmada_decode.py`

当前解码核心，负责：

- `MMaDADecodeConfig`；
- plain DCD token ranking 和 transfer；
- non-cache、prefix-cache、dual-cache kernel；
- CD/APC logits；
- selection-confidence 解耦；
- CV mode dispatcher；
- λ=0 identity；
- debug records 和 NFE。

### 2.3 `models/cv_common/`

共享视觉对比组件：

| 文件 | 职责 |
|---|---|
| `image_drop.py` | 构造视觉消融输入 |
| `paired_forward.py` | 计算 base/drop logits 与 cache |
| `log_prob.py` | 计算指定 token 的 log-prob |
| `types.py` | 调试和候选数据结构 |

### 2.4 `models/defer_only/`

实现视觉 veto：

- token 始终来自 base argmax；
- visual gain 只调节提交置信度；
- 支持 `hard/soft/mult/min`；
- 支持 `logit/logprob` gain；
- 支持 λ 干预强度插值。

### 2.5 VLMEvalKit wrapper

`evaluation/VLMEvalKit/vlmeval/vlm/mmada/mmada.py` 负责：

- 加载 tokenizer、MAGVIT2 和 MMaDA；
- 图像预处理与 VQ 编码；
- 从环境变量构造解码配置；
- 设置 neutral/text-only 消融 token；
- 调用 `mmu_generate()`；
- 保存 CV debug 文件。

---

## 3. 调用链

### 3.1 Plain DCD

```text
run.py
  └── MMaDA.generate_inner()
      └── MMaDA.generate_mmada()
          └── MMadaModelLM.mmu_generate()
              └── dispatch_dcd_decode_text()
                  ├── dcd_decode_text()
                  ├── dcd_decode_text_prefix_cache()
                  └── dcd_decode_text_dual_cache()
                      └── _pick_transfer()
```

### 3.2 CV-DCD

```text
run.py
  └── MMaDA.generate_inner()
      └── MMaDA.generate_mmada()
          └── MMadaModelLM.mmu_generate()
              └── dispatch_cv_dcd_decode_text()
                  ├── dcd_decode_text_cv()
                  ├── dcd_decode_text_cv_prefix_cache()
                  └── dcd_decode_text_cv_dual_cache()
                      ├── build_dropped_image()
                      ├── paired_forward_logits()
                      └── _pick_transfer_cv()
                          ├── cd_apc
                          ├── cd_naive
                          ├── defer_only
                          ├── legacy_score
                          └── off
```

当前 VLMEvalKit CV-DCD 使用 dual cache。

---

## 4. 解码配置

配置类：

```text
models/mmada_decode.py::MMaDADecodeConfig
```

### 4.1 DCD 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `window_type` | `sliding` | decode window |
| `initial_window_length` | `32` | 初始窗口长度 |
| `block_size` | `32` | block 长度 |
| `decode_algo` | `threshold` | 提交算法 |
| `decode_param` | `0.9` | threshold/factor 参数 |
| `temperature` | `0.0` | Gumbel 温度 |
| `remasking` | `low_confidence` | ranking 方式 |
| `mask_id` | `126336` | 文本 mask id |
| `cache_type` | `none` | `none/prefix/dual` |
| `refresh_count` | `1` | factor 刷新参数 |
| `return_debug` | `False` | 返回 debug |

### 4.2 CV 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `visual_token_start` | `2` | 视觉 span 起点 |
| `visual_token_end` | `1026` | 视觉 span 终点 |
| `causal_lambda` | `0.5` | CD 或 veto 强度 |
| `causal_clip` | `4.0` | gain 截断 |
| `cv_stride` | `1` | drop signal 使用间隔 |
| `image_drop_strategy` | `mask` | 视觉消融方式 |
| `cv_alpha` | `0.1` | APC 阈值 |
| `cv_mode` | `cd_apc` | CV 模式 |
| `cv_conf_source` | `min_base_blended` | 提交置信度来源 |
| `cv_gate_tau` | `0.0` | 高置信 bypass gate |

### 4.3 Defer-only 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `defer_veto_type` | `soft` | veto 函数 |
| `defer_tau` | `0.0` | gain threshold |
| `defer_beta` | `1.0` | penalty 陡峭程度 |
| `defer_gain_type` | `logit` | gain 类型 |
| `text_only_fill_id` | runtime | text-only filler |

---

## 5. Plain DCD

模型为待生成位置输出：

\[
s_e\in\mathbb{R}^{B\times L\times V}
\]

候选 token：

\[
x_0=\arg\max_j s_e^{(j)}
\]

置信度：

\[
c_{\text{base}}
=
\operatorname{softmax}(s_e)[x_0]
\]

程序使用 float64 计算 confidence：

```python
probs = F.softmax(logits.to(torch.float64), dim=-1)
confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
```

在 threshold 模式下：

\[
\operatorname{commit}(x_0)
\iff
c_{\text{base}}\geq\theta
\]

默认 \(\theta=0.9\)。

生成区间被切分为多个 block。每个 block 在 cache 上迭代，直到其中不再包含 mask。

---

## 6. 视觉消融与 paired forward

CV-DCD 同时维护：

- `x`：原始视觉输入；
- `x_drop`：视觉 span 被消融的输入。

对应 logits：

\[
s_e=f(x)
\]

\[
s_a=f(x_{\text{drop}})
\]

### 6.1 消融策略

| 策略 | 行为 |
|---|---|
| `mask` | 视觉 span 填充 `mask_id` |
| `shuffle` | 固定随机排列视觉 tokens |
| `random_mask` | 随机替换部分视觉 tokens |
| `mean_token` | 使用视觉 token id 均值 |
| `neutral` | 使用灰图 VQ tokens |
| `text_only` | 使用 pad/eos token |

`shuffle` permutation 在一次 decode 中保持不变。

`neutral_image_tokens` 由 wrapper 预编码灰图得到。

`text_only_fill_id` 优先取 `pad_token_id`，其次取 `eos_token_id`，也可由环境变量指定。

### 6.2 Dual cache

base 与 drop 分支分别维护：

```text
past_key_values
past_key_values_drop
```

初始化时可以沿 batch 维拼接 base/drop 输入；后续 block 更新分别推进两套 cache。

---

## 7. CD-APC

### 7.1 Contrastive logits

\[
s_{\text{blend}}
=
s_e+\lambda(s_e-s_a)
\]

即：

\[
s_{\text{blend}}
=(1+\lambda)s_e-\lambda s_a
\]

### 7.2 APC

候选集合：

\[
V_{\text{valid}}
=
\left\{
j\mid
s_e^{(j)}
\geq
\max_k s_e^{(k)}+\log\alpha
\right\}
\]

集合外 token 被设为 dtype 最小值。

CD token：

\[
x_{\text{CD}}
=
\arg\max s_{\text{effective}}
\]

### 7.3 Selection 与 confidence 解耦

`_cd_pick_with_conf()` 分开处理：

- selection logits：选择 token；
- commit confidence：控制提交时机。

`cv_conf_source`：

#### `base`

\[
c=\operatorname{softmax}(s_e)[x_{\text{CD}}]
\]

#### `blended`

\[
c=\operatorname{softmax}(s_{\text{blend}})[x_{\text{CD}}]
\]

#### `min_base_blended`

\[
c=
\min
\left(
\operatorname{softmax}(s_e)[x_{\text{CD}}],
\operatorname{softmax}(s_{\text{blend}})[x_{\text{CD}}]
\right)
\]

当前默认使用 `min_base_blended`。

### 7.4 高置信 gate

base token：

\[
x_{\text{base}}=\arg\max s_e
\]

当 `cv_gate_tau>0` 且：

\[
\operatorname{softmax}(s_e)[x_{\text{base}}]
\geq
\tau_{\text{gate}}
\]

该位置使用 base token 和 base confidence；否则使用 CD token。

---

## 8. CD-Naive

`cd_naive` 使用同一 contrastive 公式，但不使用 APC mask。

它与 `cd_apc` 共享：

- `cv_conf_source`
- `cv_gate_tau`
- DCD transfer
- cache kernel
- debug 输出

该模式用于隔离 APC 对候选空间的作用。

---

## 9. Defer-only

Defer-only 始终使用：

\[
x_0=\arg\max s_e
\]

视觉信号只调整提交置信度。

### 9.1 Gain

Logit gain：

\[
g_{\text{logit}}=s_e[x_0]-s_a[x_0]
\]

Log-prob gain：

\[
g_{\text{logprob}}
=
\log p_e(x_0)-\log p_a(x_0)
\]

gain 被截断到 \([-C,C]\)。

### 9.2 Veto

Hard：

\[
c_{\text{veto}}
=
\begin{cases}
c_{\text{base}},&g\geq\tau\\
0,&g<\tau
\end{cases}
\]

Soft：

\[
c_{\text{veto}}
=
c_{\text{base}}
\exp[-\beta\max(\tau-g,0)]
\]

Mult：

\[
c_{\text{veto}}
=
c_{\text{base}}\sigma(\beta(g-\tau))
\]

Min：

\[
c_{\text{veto}}
=
\min(c_{\text{base}},\sigma(\beta g))
\]

最终 confidence：

\[
c_{\text{eff}}
=(1-\lambda)c_{\text{base}}+\lambda c_{\text{veto}}
\]

`c_eff` 进入共享 DCD transfer selector。

---

## 10. Mode Dispatcher

统一入口：

```text
models/mmada_decode.py::_pick_transfer_cv()
```

| `cv_mode` | 执行路径 |
|---|---|
| `cd_apc` | `_cd_pick_with_conf(..., apc=True)` |
| `cd_naive` | `_cd_pick_with_conf(..., apc=False)` |
| `defer_only` | `pick_transfer_defer_only()` |
| `legacy_score` | `_resolve_cv_confidence()` |
| `off` | `_pick_transfer()` |

所有模式最终输出：

- `x0`
- `transfer_index`

block decoding、cache 更新和 token 写回不随模式变化。

---

## 11. λ=0 Identity

dual-cache kernel 使用：

```python
use_drop = config.causal_lambda > 0.0
```

当 λ=0：

1. 不构造 `x_drop`；
2. 不执行 paired forward；
3. `drop_logits=None`；
4. 使用 base cache；
5. selector 回退 base pick；
6. NFE 与 plain DCD 路径一致。

identity 同时覆盖 token、confidence、cache 和计算路径。

---

## 12. Cache、Stride 与 NFE

支持：

- `none`
- `prefix`
- `dual`

NFE 规则：

```text
base forward      → nfe += 1
base/drop forward → nfe += 2
```

`cv_stride` 控制某一步是否把 `drop_logits` 交给 CV selector。

当前 dual-cache inner loop 仍会更新 drop branch，因此 `cv_stride` 控制的是 signal 使用频率，不会完整跳过该步的 drop 计算。

---

## 13. 环境变量

### 13.1 模式

| 环境变量 | 作用 |
|---|---|
| `MMADA_DECODE_STRATEGY` | 解码策略 |
| `MMADA_CACHE_TYPE` | cache 类型 |
| `MMADA_CV_MODE` | CV 模式 |

### 13.2 CD/APC

| 环境变量 | 参数 |
|---|---|
| `MMADA_CV_LAMBDA` | `causal_lambda` |
| `MMADA_CV_CLIP` | `causal_clip` |
| `MMADA_CV_STRIDE` | `cv_stride` |
| `MMADA_CV_DROP` | `image_drop_strategy` |
| `MMADA_CV_ALPHA` | `cv_alpha` |
| `MMADA_CV_CONF_SOURCE` | `cv_conf_source` |
| `MMADA_CV_GATE_TAU` | `cv_gate_tau` |

### 13.3 Defer-only

| 环境变量 | 参数 |
|---|---|
| `MMADA_DEFER_VETO` | `defer_veto_type` |
| `MMADA_DEFER_TAU` | `defer_tau` |
| `MMADA_DEFER_BETA` | `defer_beta` |
| `MMADA_DEFER_GAIN_TYPE` | `defer_gain_type` |
| `MMADA_TEXT_ONLY_FILL_ID` | `text_only_fill_id` |

### 13.4 Debug

| 环境变量 | 作用 |
|---|---|
| `MMADA_CV_RETURN_DEBUG` | 返回 CV debug |
| `MMADA_CV_DEBUG_DIR` | debug 输出目录 |
| `MMADA_RUN_ID` | 运行标识 |
| `MMADA_INDICES` | 指定样本索引 |
| `MMADA_CURRENT_INDEX` | 当前样本索引 |

---

## 14. Debug 数据

逐步记录包括：

- step 和 batch index；
- committed positions；
- selected token；
- base/effective confidence；
- visual gain；
- base/drop token logits；
- veto 参数；
- λ；
- NFE。

metadata 包括：

```text
dataset
run_id
image
prompt
causal_lambda
cv_alpha
cv_mode
cv_conf_source
cv_gate_tau
image_drop
defer_veto_type
defer_tau
defer_beta
defer_gain_type
nfe
```

---

## 15. 测试结构

| 测试文件 | 覆盖范围 |
|---|---|
| `test_cv_common.py` | image drop、paired forward、log-prob |
| `test_cv_dcd_smoke.py` | CV dispatcher、cache、identity |
| `test_cd_style_smoke.py` | APC、CD、confidence、gate |
| `test_defer_only.py` | gain、veto、λ、transfer、identity |

---

## 16. 运行示例

### Plain DCD

```bash
export MMADA_DECODE_STRATEGY=dcd
export MMADA_CACHE_TYPE=dual

python run.py \
  --data LLaVABench \
  --model MMaDA-MixCoT-CV-DCD
```

### CD-APC

```bash
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=cd_apc
export MMADA_CV_LAMBDA=0.5
export MMADA_CV_ALPHA=0.1
export MMADA_CV_DROP=shuffle
export MMADA_CV_CONF_SOURCE=min_base_blended
export MMADA_CV_GATE_TAU=0.0

python run.py \
  --data LLaVABench \
  --model MMaDA-MixCoT-CV-DCD
```

### Defer-only

```bash
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=defer_only
export MMADA_CV_LAMBDA=1.0
export MMADA_CV_DROP=text_only
export MMADA_DEFER_VETO=hard
export MMADA_DEFER_TAU=-3.0
export MMADA_DEFER_GAIN_TYPE=logit
```

---

## 17. 当前程序创新点

### 17.1 统一 DCD/CV-DCD 解码内核

不同视觉策略共享 block、cache 和 transfer 主循环，策略模块只需提供 token 与有效置信度。

### 17.2 Selection 与 Commitment 解耦

程序将“选择哪个 token”和“何时提交 token”定义为两个独立问题，使 contrastive selection 可以与保守 DCD confidence 同时使用。

### 17.3 APC 与 confidence 独立

APC 只约束候选集合；提交 confidence 可来自 base、blended 或二者最小值。

### 17.4 高置信 base gate

`cv_gate_tau` 允许已确定位置绕过视觉对比选择，保护不可逆提交。

### 17.5 约束式 Defer-only

视觉信号不改变 base argmax，只通过 τ、β、λ 控制提交置信度。

### 17.6 多种视觉消融条件

统一支持 mask、shuffle、random-mask、mean-token、neutral 和 text-only，并维护稳定的 runtime 状态。

### 17.7 严格 identity

λ=0 时跳过 drop 输入和 paired forward，使关闭 CV 后的 token、confidence、cache 和 NFE 都回到 plain DCD。

### 17.8 可插拔策略接口

新增策略可接入 `_pick_transfer_cv()`，不需要复制完整生成循环或修改模型权重。

### 17.9 完整可观测性

程序统一记录 gain、confidence、commit、策略参数和 NFE，支持 token-level 诊断。

---

## 18. 当前程序边界

1. active CV 通常需要 base/drop 两套 forward；
2. `cv_stride` 当前不会完全跳过 dual-cache drop 计算；
3. image drop 是人工构造的对照条件；
4. token-level gain 只描述当前 step；
5. token 提交后不会重新 mask；
6. `causal_lambda` 在 CD 和 defer-only 中具有不同语义；
7. sequence-level candidate rerank 不在当前调用链中。

---

## 19. 关键源码

| 功能 | 文件 |
|---|---|
| 解码配置与 kernel | `models/mmada_decode.py` |
| 生成分发 | `models/modeling_mmada.py` |
| 图像消融 | `models/cv_common/image_drop.py` |
| Paired forward | `models/cv_common/paired_forward.py` |
| Log-prob | `models/cv_common/log_prob.py` |
| Defer dispatcher | `models/defer_only/dispatcher.py` |
| Veto | `models/defer_only/veto.py` |
| VLMEvalKit wrapper | `evaluation/VLMEvalKit/vlmeval/vlm/mmada/mmada.py` |
| Debug 保存 | `evaluation/VLMEvalKit/attention_analysis/cv_debug_io.py` |

---

## 20. 总结

当前程序的主流程为：

```text
base/drop paired forward
  → visual contrast
  → strategy-specific token/confidence
  → shared DCD transfer
  → dual-cache block decoding
```

核心设计包括：

1. 统一 DCD 与 CV-DCD 主循环；
2. 分离 token selection 和 commitment confidence；
3. 使用 APC 限制 contrastive candidate；
4. 使用 gate 保护高置信 base token；
5. 使用 defer-only 实现不改变 argmax 的视觉 veto；
6. 使用 λ=0 identity 保证策略可完全关闭；
7. 使用共享消融、debug 和测试组件保证可扩展性。
