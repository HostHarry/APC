# CV-DCD v4 执行计划

**基础文档：** `docs/cv_dcd_v4_design.md`
**起始版本：** v3.2 B3（VLM 28.3，`cv_mode=cd_apc_v32, cv_conf_source=min_base_blended, gate=0`）
**目标：** 落地方向 B（defer-only CV）和方向 A（CFG rerank），确定 CV-DCD 家族的可用上限
**时间预算：** 4-5 个工作日 + ~20 GPU 小时

---

## 0. 总览

| 阶段 | 内容 | 工作量 | GPU 时间 | 前置 |
|---|---|---|---|---|
| **M0** | 抽 `cv_common/`，v3.2 回归 | 半天 | 0.5h | 无 |
| **M1** | 方向 B 实现 (`defer_only/`) | 半天 | 0h | M0 |
| **M2** | 方向 B smoke test (5 样本) | 半天 | 0.5h | M1 |
| **M3** | **方向 B Phase D sweep** (60 样本 × 5 组) | 半天 | 3-4h | M2 |
| **M3.5** | **Checkpoint** | 半天 | 0h | M3 |
| **M4** | 方向 A 实现 (`cfg_rerank/`) | 1 天 | 0h | M0 |
| **M5** | 方向 A smoke test (5 样本 × 2 组) | 半天 | 2h | M4 |
| **M6** | 方向 A Phase C sweep (60 样本 × 6 组) | 1 天 | 10-15h | M5 |
| **M7** | 综合分析 + 更新 `cv_dcd_full_report.md` | 半天 | 0h | M3, M6 |

**关键决策点：M3.5**  
方向 B 结果决定要不要做方向 A。三种可能：
1. **B 最好 VLM > 37**：真正超过 baseline，优先深挖 B（缩短 M4-M6），或直接结束项目。
2. **B 最好 VLM 32-37**：接近 baseline，做 A 有价值，正常推进 M4-M6。
3. **B 最好 VLM < 32**：defer-only 也救不了 gain 信号 —— **暂停方向 A**，直接进入 M7 写负结果。

---

## 1. M0 — 抽出 `cv_common/`（Day 0, 半天）

### 1.1 目标

把 `mmada_decode.py` 里的 3 个纯函数搬到独立模块，两个 v4 方向共享。**行为完全不变**，v3.2 单元测试和 5 样本 sanity 必须**bit-exact 通过**。

### 1.2 任务清单

| # | 任务 | 具体内容 | 验收 |
|---|---|---|---|
| M0.1 | 新建目录 | `mkdir models/cv_common/`，加空 `__init__.py` | 目录存在 |
| M0.2 | 实现 `image_drop.py` | 把 `_build_dropped_image` 完整搬入，函数名改为 `build_dropped_image`。新增 `"text_only"` 策略：用 pad_token_id 填充 image span | 单测 6/6 pass |
| M0.3 | 实现 `paired_forward.py` | 把 `_paired_forward_logits` 完整搬入 | 单测 3/3 pass |
| M0.4 | 实现 `log_prob.py` | `logp_of_tokens`（等价现 `_logp_of_x0`）+ `stepwise_logp_from_records` + `aggregate_sequence_ll` 三个新函数 | 单测 4/4 pass |
| M0.5 | 实现 `types.py` | `@dataclass StepRecord`, `@dataclass Candidate` | 导入无错 |
| M0.6 | thin wrapper 化 mmada_decode.py | 原 `_build_dropped_image` / `_paired_forward_logits` / `_logp_of_x0` 变成 `return cv_common.xxx(...)` 一行调用；给未来若删除留 shim | v3.2 全部 19 单测 pass |
| M0.7 | 写 `tests/test_cv_common.py` | 覆盖 build_dropped_image (含 text_only 新策略) / paired_forward / logp / stepwise LL 的语义正确性 | 13/13 tests pass |
| M0.8 | 5 样本 bit-exact sanity | 用 v3.2 B3 配置跑 5 样本，对比重构前后 predictions | 60/60 tokens per sample 逐字节相同 |

### 1.3 验收标准（Definition of Done）

- [ ] `tests/test_cv_common.py` 全绿
- [ ] `tests/test_cv_v32.py` 全绿（v3.2 不受影响）
- [ ] 5 样本 sanity：`diff prev_result.xlsx new_result.xlsx` 为空
- [ ] `git status` 只显示 4 个 cv_common 文件 + `mmada_decode.py` 内 3 处小改动

### 1.4 风险

- **风险 R0.1**：thin wrapper 化时 signature 有变化 → 破坏 v3.2 单测
  - **缓解**：保持原函数名和参数名不变，只改函数体为 `return cv_common.xxx(*args, **kwargs)`
- **风险 R0.2**：`text_only` 策略需要 `pad_token_id`，config 里没有
  - **缓解**：`text_only` 用 `config.mask_id` 之外的备选：优先 `config.text_only_fill_id`（新加，默认 None），fallback 到 tokenizer 提供的 pad_id（wrapper 层填）；如果都拿不到就 raise 明确错误

---

## 2. M1 — 方向 B 实现 `defer_only/`（Day 0-1，半天）

### 2.1 目标

三种 veto（hard / mult / min）全部实现，加上 dispatcher；`_pick_transfer_cv` 新增 `defer_only` 分支。**argmax 永不改**——这是最强的正确性约束。

### 2.2 任务清单

| # | 任务 | 具体内容 | 验收 |
|---|---|---|---|
| M1.1 | `defer_only/veto.py` | 5 个纯函数：`compute_visual_gain`, `apply_veto_hard`, `apply_veto_mult`, `apply_veto_min`, `apply_veto`（dispatcher） | 单测 8/8 pass |
| M1.2 | `defer_only/dispatcher.py` | `pick_transfer_defer_only`（~90 LOC）；debug 记录函数 `_record_defer_debug` | 单测 4/4 pass |
| M1.3 | `mmada_decode.py` 集成 | `_pick_transfer_cv` 加 `if mode == "defer_only":` 分支；`MMaDADecodeConfig` 加 3 个字段（`defer_veto_type`, `defer_tau`, `defer_beta`） | v3.2 全测通过 + defer_only 单测通过 |
| M1.4 | wrapper 集成 | `mmada.py` 加 3 个 env vars：`MMADA_DEFER_VETO`, `MMADA_DEFER_TAU`, `MMADA_DEFER_BETA`；接入 config 构造 | wrapper 启动时无异常，打印 defer 参数 |
| M1.5 | debug NPZ schema 扩展 | `attention_analysis/cv_debug_io.py` 新字段：`defer_veto_type`, `defer_gain`, `defer_base_conf`, `defer_eff_conf`, `defer_active` | NPZ 加载能看到新字段 |
| M1.6 | 单测 `tests/test_defer_only.py` | 详见 § 2.3 | 13/13 pass |

### 2.3 单元测试清单（详细）

| # | 测试名 | 断言 |
|---|---|---|
| 1 | `test_gain_zero_when_logits_equal` | `base_logits == drop_logits` → `gain == 0` |
| 2 | `test_gain_clip_respected` | 极端 logits 差 → `abs(gain) <= causal_clip` |
| 3 | `test_hard_veto_semantics` | `gain=[-1,0,1,2], tau=0.5` → `eff_conf=[0,0,base,base]` |
| 4 | `test_mult_veto_monotonic` | 固定 base_conf，gain 单调↑ → eff_conf 单调↑ |
| 5 | `test_min_veto_never_boosts` | 任意 base_conf, gain → `eff_conf <= base_conf` |
| 6 | `test_veto_dispatch_by_type` | `apply_veto(veto_type='hard')` 等价直接调 `apply_veto_hard` |
| 7 | `test_pick_transfer_argmax_from_base_only` | `x0` 必须 == `argmax(base_logits)`（关键正确性） |
| 8 | `test_pick_transfer_no_drop_equals_baseline` | `drop_logits=None` → 输出与 baseline `_pick_transfer` **完全一致** |
| 9 | `test_pick_transfer_debug_populated` | `return_debug=True` → debug_records 有 `defer_active`, `defer_gain` 字段 |
| 10 | `test_config_defaults` | `MMaDADecodeConfig()` 时 `defer_veto_type='mult'`, `defer_tau=0.0`, `defer_beta=1.0` |
| 11 | `test_env_var_override` | `os.environ['MMADA_DEFER_VETO']='hard'` → config `defer_veto_type == 'hard'` |
| 12 | `test_v32_still_works` | v3.2 B3 配置 5 步小模型能跑通 |
| 13 | `test_defer_only_end_to_end` | 3 层小模型跑 defer_only 出结果 |

### 2.4 验收标准

- [ ] 13/13 tests pass
- [ ] v3.2 全部 19 tests 依然 pass
- [ ] `import models.defer_only.dispatcher; pick_transfer_defer_only` 可导入
- [ ] `MMADA_CV_MODE=defer_only` 时 wrapper 正确构造 config

### 2.5 风险

- **风险 R1.1**：`_select_transfer` 复用问题——现在它嵌在 `_pick_transfer` 里
  - **缓解**：先在 `mmada_decode.py` 里把 threshold/factor selection 抽成一个 `pick_transfer_by_confidence(confidence, mask_index, config)` 独立函数，defer_only 和 v3.2 都能调用。这个抽取不影响 v3.2 行为。
- **风险 R1.2**：`argmax_from_base_only` 断言可能因为 Gumbel noise 失败
  - **缓解**：defer_only 里 `temperature=0` 时 Gumbel 是 identity，断言在 temperature=0 场景保证成立。文档里明确 `temperature > 0` 时 argmax 可能有 noise，但**不来自 CD 修改**。

---

## 3. M2 — 方向 B smoke test（Day 1，半天）

### 3.1 目标

用 5 个 LLaVABench 样本快速验证 defer_only 三种 veto 都能跑通、生成结果不崩、和 baseline DCD 有可观察差异。

### 3.2 sweep 配置

| Tag | veto | τ | β | λ | drop | 期望 |
|---|---|---|---|---|---|---|
| S1 | mult | 0.0 | 1.0 | 0.5 | shuffle | 主候选，平滑 veto |
| S2 | hard | 0.0 | — | 0.5 | shuffle | 硬阈值，看 defer 增加 |
| S3 | min | — | 1.0 | 0.5 | shuffle | 双保险 |
| S4 | mult | 0.0 | 1.0 | 0.0 | shuffle | **λ=0 sanity: 应等价 baseline DCD** |

### 3.3 验收标准

- [ ] 4 个 tag 都跑完 5 样本不崩溃
- [ ] S4 的 5 个预测**逐字节等于**同 seed 的 baseline DCD（关键 sanity）
- [ ] S1/S2/S3 之间有可观察差异（不是所有预测都相同）
- [ ] `defer_active` 字段在 debug NPZ 里被填充，取值合理（不是全 0 也不是全 1）

### 3.4 GPU 时间

约 30 分钟（4 组 × 5 样本 × ~1.5min/sample）。

### 3.5 命令

```bash
bash scripts/run_defer_only_smoke.sh   # 新建，5 样本 4 组
```

脚本模板参考 `scripts/run_cvdcd_v32_sweep.sh`。

---

## 4. M3 — 方向 B 完整 sweep Phase D（Day 1-2，半天工作 + 3-4h GPU）

### 4.1 目标

方向 B 的核心实验。5 组配置 × 60 LLaVABench 样本 × GPT-4 打分，得出 defer-only 家族的完整性能画像。

### 4.2 sweep 配置（Phase D）

| Tag | veto | τ | β | λ | drop | 假设 |
|---|---|---|---|---|---|---|
| **D0** | mult | 0.0 | 1.0 | **0.0** | shuffle | Sanity 内对照：应 ≈ baseline (36.7) |
| D1 | hard | 0.0 | — | 0.5 | shuffle | Hard veto |
| **D2** | mult | 0.0 | 1.0 | 0.5 | shuffle | **默认，主候选** |
| D3 | mult | 0.5 | 1.0 | 0.5 | shuffle | 更严的 τ |
| D4 | min | — | 1.0 | 0.5 | shuffle | 双保险 |
| D5 | mult | 0.0 | 1.0 | 0.5 | **text_only** | 更干净的 uncond 参考 |

**共 6 组**（原设计 5 组，加 D0 sanity 和 D5 text_only）。

### 4.3 验收标准

- [ ] 6/6 tags 全部跑完
- [ ] D0 VLM ≈ 36.7（±0.5，内部一致性）
- [ ] 至少 1 个 tag VLM > 28.3（即超过 v3.2 B3 上限）
- [ ] GPT-4 打分成功率 > 95%（4o-mini 打分稳定）
- [ ] 每组 debug NPZ 保存到独立子目录（`outputs/cvdcd_sweep/cvdcd_v4_phase_d_<timestamp>/D<n>_<tag>/`）

### 4.4 快速分析脚本

M3 结束后自动跑：

```bash
python attention_analysis/summarize_cv_dcd_v4.py \
    --run-dir outputs/cvdcd_sweep/cvdcd_v4_phase_d_<timestamp>/ \
    --baseline-xlsx outputs/MMaDA-MixCoT-DCD-DualCache/T20260613_G/*_openai_result.xlsx
```

输出：per-tag CSV, per-sample delta 表, top-5 gains/losses（跟 Phase B 分析一致的 schema）。

### 4.5 GPU 时间

约 3-4 小时（6 组 × 60 样本 × ~30s/sample）。

---

## 5. M3.5 — 关键决策点（Day 2，半天）

### 5.1 决策矩阵

根据 M3 结果决定 M4-M6 走向：

| M3 最好 VLM | 判断 | 下一步 |
|---|---|---|
| > 37.0 | defer-only 超过 baseline | **停止 M4-M6，直接 M7 写胜利报告**。可选：更多 defer-only 参数细调 |
| 33.0-37.0 | defer-only 接近 baseline，但未超过 | **正常推进 M4-M6**，A 有可能锦上添花 |
| 30.0-33.0 | defer-only 好于 B3 (28.3) 但不接近 baseline | **正常推进 M4-M6**，A 是主要希望 |
| 28.3-30.0 | 与 B3 持平；defer 机制没有 headroom | **推进 M4-M6，但降低期望**——A 也可能只能到这里 |
| < 28.3 | defer-only 比 B3 还差 | **暂停 M4-M6，直接 M7 写负结果**。gain 信号被证伪 |

### 5.2 M3.5 交付物

- [ ] Phase D 结果表（合入 `cv_dcd_full_report.md` § 6）
- [ ] 决策 memo：一句话结论 + 三行推理（在这里贴给用户 review）
- [ ] 更新 `docs/cv_dcd_v4_plan.md` 后续 milestone 的时间预估

---

## 6. M4 — 方向 A 实现 `cfg_rerank/`（Day 2-3，1 天）

### 6.1 目标

Sequence-level CFG rerank 完整实现。4 个模块：生成、uncond LL、rerank、pipeline。

### 6.2 任务清单

| # | 任务 | LOC | 验收 |
|---|---|---|---|
| M4.1 | `cfg_rerank/generator.py` | ~120 | 单测：3 个 seed 产出 3 个不同 seq；step_records 数量 == commit 步数 |
| M4.2 | `cfg_rerank/uncond_ll.py` | ~80 | 单测：drop=identity 时 `logp_cond == logp_uncond`（浮点内） |
| M4.3 | `cfg_rerank/rerank.py` | ~40 | 单测：cfg_score 公式正确；rerank 选大 gain |
| M4.4 | `cfg_rerank/pipeline.py` | ~80 | 集成入口 `cfg_rerank_decode` 可调 |
| M4.5 | `mmada_decode.py` config 扩展 | ~10 | 4 个新字段：`rerank_k, rerank_gamma, rerank_seeds, rerank_ll_reduction` |
| M4.6 | wrapper 集成 | ~20 | `MMADA_DECODE_STRATEGY=cfg_rerank` 分支 + 4 个 env vars |
| M4.7 | `tests/test_cfg_rerank.py` | ~120 | 见 § 6.3 |

### 6.3 单元测试清单

| # | 测试名 | 断言 |
|---|---|---|
| 1 | `test_step_record_dataclass` | 字段类型、可 pickle |
| 2 | `test_stepwise_ll_sums_correctly` | 手工 3 步 → aggregate 结果 == 手算和 |
| 3 | `test_ll_length_normalization` | reduction='mean' 与 'sum' 差异 == 1/n |
| 4 | `test_reconstruct_x_at_step` | 3 步 candidate → `_reconstruct_x_at_step(t=2)` 等于 apply step 0-1 |
| 5 | `test_uncond_ll_equals_cond_when_drop_is_identity` | drop = noop → logp_cond == logp_uncond |
| 6 | `test_cfg_score_math` | `(1+γ)·c - γ·u == c + γ·(c-u)` |
| 7 | `test_rerank_picks_larger_gain` | 2 candidates, γ=1 → 选 gain 更大者 |
| 8 | `test_generator_seeds_differ` | 3 seeds on toy model → 3 different seqs |
| 9 | `test_pipeline_end_to_end` | toy model 跑通完整 pipeline |
| 10 | `test_config_defaults` | 默认 `rerank_k=4, gamma=1.0` |
| 11 | `test_env_var_parsing` | `MMADA_RERANK_SEEDS='7,42,123,999'` → list of 4 ints |
| 12 | `test_v32_still_works` | v3.2 全兼容 |

### 6.4 验收标准

- [ ] 12/12 tests pass
- [ ] v3.2 全部单测 pass
- [ ] `import models.cfg_rerank.pipeline; cfg_rerank_decode` 可导入
- [ ] `MMADA_DECODE_STRATEGY=cfg_rerank` 时 wrapper 打印所有 rerank 参数

### 6.5 风险

- **风险 R4.1**：`_reconstruct_x_at_step` 逻辑复杂容易错
  - **缓解**：先在测试里手工构造 3 步 candidate，逐位置验证；用 assert 而不是随机测试
- **风险 R4.2**：KV cache 复杂——candidate 生成走 dual_cache，uncond replay 需要自己维护 KV cache
  - **缓解**：**初版不复用 KV cache**，每步 forward from scratch。**接受 uncond replay 的 NFE ≈ candidate 生成 NFE**（8× baseline 总预算不变）。M6 sweep 后再看优化。

---

## 7. M5 — 方向 A smoke test（Day 3，半天）

### 7.1 目标

跑 5 样本 × 2 组 sanity，验证 CFG rerank pipeline 端到端可执行、候选真的不同、rerank 真的在选。

### 7.2 sweep 配置

| Tag | K | γ | drop | 检查 |
|---|---|---|---|---|
| SA1 | 4 | 1.0 | shuffle | 主候选 |
| SA2 | 4 | **0.0** | shuffle | **γ=0 sanity: 等价随机选 seed 0** |

### 7.3 验收标准

- [ ] 2 组 × 5 样本全跑完
- [ ] 每个样本的 4 个候选**不完全相同**（至少 2 个 seq 有 >20% token 不同）
- [ ] SA2 (γ=0) 的 5 个预测约等于 seed 0 单跑（best_idx 主要选 0）
- [ ] SA1 有可观察的 best_idx 分布（不是永远选同一个 seed）
- [ ] `debug_dict` 结构完整：n_candidates=4, cfg_scores 是有限浮点

### 7.4 GPU 时间

约 2 小时（2 组 × 5 样本 × 8× baseline NFE ≈ 15min/sample）。

---

## 8. M6 — 方向 A 完整 sweep Phase C（Day 3-4，1 天工作 + 10-15h GPU）

### 8.1 目标

CFG rerank 家族的完整性能画像。7 组配置 × 60 样本。

### 8.2 sweep 配置（Phase C）

| Tag | K | γ | drop | LL_reduce | 假设 |
|---|---|---|---|---|---|
| **C0** | 4 | **0.0** | shuffle | sum | Sanity：γ=0 应 ≈ baseline single seed |
| C1 | 4 | 0.5 | shuffle | sum | 弱 guidance |
| **C2** | 4 | 1.0 | shuffle | sum | **默认，主候选** |
| C3 | 4 | 2.0 | shuffle | sum | 强 guidance |
| C4 | 4 | 1.0 | text_only | sum | 更干净 uncond |
| C5 | 4 | 1.0 | shuffle | **mean** | Length-normalized LL |
| C6 | 6 | 1.0 | shuffle | sum | 增大候选池 |

### 8.3 验收标准

- [ ] 7/7 tags 全跑完
- [ ] C0 VLM ≈ single-seed baseline（±1）
- [ ] 至少 1 个 tag VLM > max(28.3, M3_best)
- [ ] 用与 Phase D 相同的分析脚本产出 per-sample delta 表

### 8.4 GPU 时间

约 10-15 小时（7 组 × 60 样本 × 8× baseline NFE ≈ 4min/sample）。

**可以 M6 挂夜里跑，M7 白天分析。**

### 8.5 分阶段发布策略

M6 是长跑，建议：
1. **优先跑 C0, C2**（sanity + 主候选，~5h）→ 早期判断
2. 如果 C2 VLM < 30，**暂停其他 5 组**，直接进入 M7 分析
3. 如果 C2 VLM > 32，跑完剩下 5 组

---

## 9. M7 — 综合分析（Day 4-5，半天）

### 9.1 交付物

1. **`cv_dcd_full_report.md` § 11 更新**：
   - Phase D + Phase C 完整结果表（跟 Phase B 表格 schema 一致）
   - 每方向的 top-5 gain/loss 样本分析
   - 方向 A vs 方向 B 的对比表
   - 更新 § 9 "修好/没修好清单"

2. **`docs/cv_dcd_v4_results.md`**（新文件）：v4 独立结果报告，可作为项目最终交付

3. **决策 memo**：写在 v4_results.md § 结论 —— 三种可能结论之一：
   - **A. CV-DCD 家族可用**：defer-only 或 CFG rerank 超过 baseline
   - **B. CV-DCD 家族接近但不超过**：给出实用配置
   - **C. CV-DCD 家族被证伪**：明确 "gain 信号在此设置下不足以驱动改进"，给出未来方向（e.g., train-time visual grounding）

### 9.2 分析脚本任务

- 合并 Phase B/D/C 结果到一张总表
- 计算每个类别（conv/detail/complex）的 delta
- 抽样 3-5 个 top gain 和 top loss，对照 baseline / B3 / D_best / C_best 的具体输出

---

## 10. 资源与依赖

### 10.1 GPU 需求

| 阶段 | 累积 GPU 时间 |
|---|---|
| M0-M2 | 1h |
| M3 (Phase D) | 4-5h |
| M4-M5 | 2-3h |
| M6 (Phase C) | 10-15h |
| **总计** | **17-24h GPU** |

单张 H100 / A100 上，**分 2-3 天完成，每天占用 <10h**。

### 10.2 依赖包

- 无新依赖。全部用现有环境（`mmada` conda env）。
- `text_only` 需要 tokenizer 提供 `pad_token_id`——现有 MMaDA tokenizer 已有。

### 10.3 阻塞事项

无。所有前置（v3.2 完成、debug NPZ 工具就绪、GPT-4 打分链路稳定）已具备。

---

## 11. 时间线（相对起始日）

```
Day 0  [======] M0 (cv_common)   [====] M1 (defer_only 代码)
Day 1  [==] M2 (B smoke)         [========] M3 (Phase D sweep 白天跑)
Day 2  [==] M3.5 (决策)          [============] M4 (rerank 代码)
Day 3  [====] M5 (A smoke)       [======] M6 前半 (C0/C2)
Day 4  [==============] M6 后半 (C1/C3/C4/C5/C6，可能挂夜里)
Day 5  [==========] M7 (分析 + 更新报告)
```

如果 M3.5 决定停止方向 A，Day 3 直接进入 M7，**总时间压缩到 3 天**。

---

## 12. 立即可以启动的 4 件事

按优先级排列，你可以任选一件（或多件并行）：

1. **M0.1-M0.8**（半天，无 GPU）：抽 cv_common + 5 样本回归。**零风险**，做完 v3.2 更整洁。
2. **M1.1-M1.6**（半天，无 GPU）：写方向 B 代码 + 单测。CPU 就够。
3. **审阅 v4 设计**：如果对 § 3.2 stepwise LL 或 § 4.1 三种 veto 公式有异议，现在改成本最低。
4. **准备 GPU 排期**：确认接下来 3-5 天能占用 GPU 多少小时，决定要不要跑全量 Phase C。

我的推荐：**先 M0 → M1 → M2，做到方向 B smoke 通过再看 M3 sweep**。这样 Day 1 结束时已经能看到 defer-only 的初步信号，无需等 M3 全量。
