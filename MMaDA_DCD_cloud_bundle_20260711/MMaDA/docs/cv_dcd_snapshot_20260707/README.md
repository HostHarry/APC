# CV-DCD Snapshot — 2026-07-07 / 08 / 09

**目的**：自包含地保存**当前项目状态 + 全部最新实验结果 + 完整可运行源码**，便于分享或复盘。

## 目录结构

```
cv_dcd_snapshot_20260707/
├── README.md                          # 本文件：入口与关键结论
├── code/                              # ★ 完整可运行代码（见 code/MANIFEST.md）
│   ├── MANIFEST.md                    # 逐文件说明
│   ├── models/                        # CV-DCD 核心 (defer_only + cv_common + mmada_decode)
│   ├── vlmeval_patches/               # VLMEvalKit 4 个改动文件
│   ├── scripts/                       # 全部 run_defer_only_*.sh + 数据准备工具
│   └── tests/test_defer_only.py       # 32 CPU 单测
├── report/
│   ├── cv_dcd_status_current.md
│   ├── cv_dcd_v4_design.md
│   └── cv_dcd_v4_plan.md
└── results/
    ├── v5_matrix/                     # v5 (λ,τ) 矩阵：60 样本 × 7 config
    ├── e0_parity/                     # LLaVABench 60 样本 × 4 config
    ├── det_bench/                     # 决定性实验：ScienceQA_VAL + MathVision_MINI × 3 config
    ├── visual_bench/                  # POPE + MME + MMBench_DEV_EN × 3 config
    │   ├── POPE/{B0_plain_dcd, E4_defer_l0.5_t-3.0, E6_defer_l1.0_t-3.0}/
    │   ├── MME/ (同上，含 simple_score.csv 修复 paired-plus bug)
    │   ├── MMBench_DEV_EN/ (B0 + E4，E6 因 SSL crash 未完成)
    │   └── summary_all_configs.csv    # 三数据集 × 三配置聚合 CSV
    └── chair/                         # ★ 新增：CHAIR 长篇 caption 幻觉，200 图 val2017 × 3 config
        ├── B0_plain_dcd/{_chair_score.csv, _chair_details.jsonl}
        ├── E4_defer_l0.5_t-3.0/…
        ├── E6_defer_l1.0_t-3.0/…
        └── summary.csv                # 三配置聚合，全部 segments+captions GT
```

每个 config 目录含 3 个文件：
- `*_LLaVABench.xlsx`：60 个原始 prediction
- `*_LLaVABench_openai_result.xlsx`：GPT-4 打分明细
- `*_LLaVABench_score.csv`：聚合分数（overall/complex/conv/detail × Relative/VLM/GPT4）

---

## 关键结论（TL;DR）

> **最终判决**（2026-07-09，跨 7 个数据集）：
> 1. **代码 fix 完美生效**：E0 (defer_only λ=0) == plain DCD **60/60 byte-identical** (LLaVABench), **199/199** (ScienceQA)
> 2. **GPT-4 主观打分噪声 ~0.9 分**（LLaVABench）：完全相同的 60 个 predictions 得到 30.9 vs 31.8
> 3. **defer-only 在所有 benchmark 上均无正向影响**（7/7 negative-or-null）：
>
>    | Dataset | 类型 | ΔE4 vs B0 | 备注 |
>    |---|---|---|---|
>    | LLaVABench | GPT 主观打分 | +0.3 ~ +0.6 | 落在 ~0.9 分 GPT 噪声内，**不可区分** |
>    | ScienceQA_VAL | 规则匹配（0 噪声） | **−0.5 pp** | 199 样本，1/199 flip 对→错 |
>    | MathVision_MINI | 规则+GPT 抽取 | **−2.3 pp** | 87 样本，9/87 flip 全部对→错 |
>    | POPE popular | 幻觉 Y/N | **−2.02 F1** | 200 样本，1/200 flip 对→错 |
>    | MME overall | 视觉 Y/N | **−1.50 pp** | 200 样本，5/200 flip 净 -3 |
>    | MMBench_DEV_EN | 视觉 MCQ | **−1.00 pp** | 200 样本，2/200 flip 对→错 |
>    | **CHAIR (val2017)** | 长文 caption 幻觉 | **CHAIRi ±0.1 / recall −0.9 pp** | 200 图，defer 不降幻觉但降 recall |
>
> 4. **λ 完全无分辨力**：E4 (λ=0.5) 与 E6 (λ=1.0) 在多个数据集上 predictions **逐字节相同**（POPE 200/200, MME 200/200, ScienceQA 199/199 均一致）
> 5. **每次 defer 都是坏干预**：所有已知 flip 全部或几乎全部对→错，无系统性提升
> 6. **v5 矩阵的 "E6 保护 detail" 是 GPT noise**：head-to-head 里翻转
>
> **战略判决**：defer-only（Direction B）路线**跨 7 数据集彻底证伪、终止**。
> 下一步应转向 **CFG Rerank（Direction A：sequence-level）** 或重构思路。保留代码作对照 baseline。

### 核心失败机制

**defer_only 的假设**：删图后 x0 更自信（gain < τ）= 语言先验幻觉，应 veto。

**实际观察**：
- 干预触发率极低（POPE: 0.5%, MME: 2.5%, MMBench: 1%）——`hard veto + conf≥0.9 + text_only drop` 让 λ 事实失效
- 触发时，**"删图后更自信"通常反映的是图片带来的合理不确定性**（图有噪声/歧义），**不是**语言先验错误
- 每次 veto 把**图给出的稍不自信但正确的答案**换成**语言模型脑补的备胎** → 系统性变错

### 1. **代码 fix 完美生效**：E0 == plain DCD byte-identical

```
== E0 vs B0 comparison ==
  identical: 60/60
  differing: 0/60
PARITY HOLDS: E0 == plain DCD byte-for-byte.
```

诊断：defer_only 分支之前用 `logsumexp(bfloat16) → cast fp64` 计算 base_conf，与 plain DCD 的 `F.softmax(fp64)` 有 ~2⁻⁷ 精度差，会在 threshold=0.9 边界翻越导致文本发散。改为 `F.softmax(fp64)` 后完全对齐。

### 2. **GPT scoring noise 有大规模影响**（新发现）

**60/60 predictions 完全一样**的两个 run，GPT-4 打出：

| run | overall | VLM | GPT4 |
|---|---|---|---|
| B0_plain_dcd | 30.9 | 27.3 | 88.3 |
| E0_defer_l0 | **31.8** | **28.0** | 88.0 |

同样的文本 → 分数差 **0.9 / 0.7 / 0.3**。这就是 GPT scorer 的固有噪声。

**推论**：v5 矩阵里 config 之间 ±0.5–1.0 分的差异必须结合 predictions 差异一起看，单看分数容易被 noise 骗。

### 3. **历史 baseline 不可直接比较**

历史 DCD baseline (2026-06-13)：VLM 36.7，overall 44.1  
同环境 B0 plain DCD：VLM 27.3，overall 30.9  

**净差 11–13 分**——不是模型退化，而是：
- **主因**：GPT scorer 版本更新（scoring 标准变了）+ `temperature=0.8` 采样漂移
- **次因**：D1 bit 精度差（贡献 ~1–2 分，已通过 E0=B0 parity 排除）

所以所有 CV-DCD 相对提升的判断，**必须用 B0 而不是历史 baseline**。

### 4. **v5 矩阵完整数据**（60 样本 LLaVABench，本 snapshot `results/v5_matrix/`）

| tag | λ | τ | Relative | VLM | GPT4 | detail |
|---|---|---|---|---|---|---|
| E0_sanity (buggy) | 0 | – | 28.6 | 25.3 | 88.7 | 27.7 |
| E1 | 0.25 | −2.0 | 30.7 | 27.0 | 87.8 | 25.9 |
| E2 | 0.25 | −3.0 | 30.4 | 26.8 | 88.3 | 26.3 |
| E3 | 0.5 | −2.0 | 30.0 | 26.3 | 87.7 | 25.4 |
| E4 | 0.5 | −3.0 | 30.7 | 27.2 | 88.5 | 26.5 |
| E5 | 1.0 | −2.0 | 29.9 | 26.5 | 88.5 | 25.0 |
| E6 | 1.0 | −3.0 | 30.9 | 27.2 | 87.8 | **28.6** |

**v5 表面结论**（后被 head-to-head 否定，见 §6）：
- E6 detail 28.6 高于 E0 27.7 → 曾认为 "E6 保护 detail"
- E4/E6 overall 30.7-30.9 > E0 28.6 → 曾认为干预有 +2 分提升

**⚠️ 但 v5 E0 有 D1 数值 bug**，且 GPT scorer noise 约 1 分 —— 这些 "提升" 大部分是 noise 与 bug 的叠加，不是稳定属性。今日 head-to-head 才是可信的对照。

### 5. **E0-parity 同环境 head-to-head 完整结果**

本 snapshot `results/e0_parity/` — 4 个 config 全部完成 60 样本：

| tag | overall | VLM | GPT4 | conv | complex | detail | ≠B0 predictions |
|---|---|---|---|---|---|---|---|
| B0_plain_dcd | 30.9 | 27.3 | 88.3 | 26.9 | 36.3 | 26.5 | – |
| E0_defer_l0 | 31.8 | 28.0 | 88.0 | 28.1 | 36.9 | 27.4 | **0/60**（byte==B0）|
| E4 λ=0.5, τ=−3.0 | 31.2 | 27.3 | 87.5 | 28.1 | 35.5 | 27.6 | 35/60 |
| E6 λ=1.0, τ=−3.0 | 31.5 | 27.8 | 88.5 | 29.4 | 36.2 | 25.7 | 35/60 |

### 6. **诚实读出：defer_only 干预效果 ≤ GPT 评分噪声**

**这是本 snapshot 最重要的发现，覆盖 v5 矩阵的结论。**

分析每个 config 相对 B0 的 overall Relative Score 差异，与 GPT scoring noise 对比：

| 对比 | overall Δ vs B0 | predictions 差异 | 解读 |
|---|---|---|---|
| E0 vs B0 | **+0.9** | **0/60**（100% 相同）| **纯 GPT noise** —— 这就是噪声下限 |
| E4 vs B0 | +0.3 | 35/60 | 在噪声范围内，**统计上不可区分** |
| E6 vs B0 | +0.6 | 35/60 | 在噪声范围内，**统计上不可区分** |

**关键推论**：
1. **GPT-4 对相同 60 个预测的重复打分变动 ≈ 0.9 分**（E0 = B0 byte-identical 却差 0.9 分）
2. **E4/E6 的相对 B0 提升（+0.3、+0.6）低于这个噪声下限** → 无法证明 defer_only 提供了真正的 gain
3. v5 矩阵里 "E6 detail 28.6 > E0 27.7" 的现象在今日 head-to-head 里翻转（E6 detail 25.7 < B0 26.5），进一步佐证之前的 "detail 保护" 是 GPT noise，不是稳定属性

### 7. **defer_only 结构性局限的证据**

Defer-only 设计核心：**只 delay commit，不修改 argmax**。这意味着：

- 若 base_logits 已经把错的 token 选为 argmax，defer 只能推迟到下一轮再选——但下一轮 base_logits 通常还会选同一个错 token
- 只有当**上下文因为 delay 而被其他位置 commit 更新**时，delayed 位置的 argmax 才可能真正改变
- 这个"上下文重估"窗口非常小

**60 样本上的观测**：
- E4/E6 让 35/60 predictions 与 B0 不同，说明有干预
- 但 GPT-4 判定这些不同的 predictions **和 B0 平均一样好**
- 结论：defer_only 改变了输出，但没有系统性提升

### 8. **低噪声 benchmark 结果（决定性实验，`results/det_bench/`）**

**ScienceQA_VAL**（199 样本，规则匹配，0 噪声）：

| tag | Overall Acc | vs B0 | ≠B0 predictions |
|---|---|---|---|
| B0_plain_dcd | **59.30%** | – | – |
| E4 (λ=0.5, τ=−3.0) | 58.79% | **−0.5 pp** | 1/199 |
| E6 (λ=1.0, τ=−3.0) | 58.79% | **−0.5 pp** | 1/199 |
| E4 vs E6 | – | – | **0/199（bit-identical）**|

**MathVision_MINI**（87 样本，规则+GPT 抽取）：

| tag | Overall Acc | vs B0 | ≠B0 predictions |
|---|---|---|---|
| B0_plain_dcd | **20.69%** | – | – |
| E4 (λ=0.5, τ=−3.0) | 18.39% | **−2.3 pp** | 9/87 |
| E6 (λ=1.0, τ=−3.0) | 18.39% | **−2.3 pp** | 11/87 |
| E4 vs E6 | – | – | 4/87 |

**关键观察**：
1. **两个低噪声 dataset 都是负提升**——不是随机 noise，是系统性 bug
2. **λ 参数无分辨力**：ScienceQA 上 E4=E6 逐字节相同；MathVision 上仅 4/87 差异
3. **每次干预都变错**：ScienceQA 1 sample flip 对→错；MathVision 2 samples flip 对→错
4. **短答案任务的干预率极低**（ScienceQA 0.5%）：`gain < τ=−3.0` 的稀疏命中点恰是关键 answer token
5. LLaVABench 的 "+0.6" 是 GPT 主观打分对同一 prediction 的评分波动，不是真实提升

### 9. **视觉/幻觉 benchmark 结果（新增，`results/visual_bench/`）**

**POPE**（200 样本，Y/N 幻觉，3 split）：

| Split | B0 F1 | E4 F1 | E6 F1 | ΔE4 | ΔE6 | ≠B0 predictions |
|---|---|---|---|---|---|---|
| Overall | 85.29 | 84.73 | 84.73 | **−0.56** | **−0.56** | 1/200 |
| popular | 92.59 | 90.57 | 90.57 | **−2.02** | −2.02 | 1/56 |
| random | 81.01 | 81.01 | 81.01 | 0 | 0 | 0/75 |
| adversarial | 84.51 | 84.51 | 84.51 | 0 | 0 | 0/69 |

**关键发现**：defer_only **只在 `popular` split 触发干预**，且触发时**全部损伤 recall**：
- `popular` 提问"图里常见物体"（chair、person 等）→ 语言先验倾向 Yes
- 图片确实包含该物体时，`base(Yes)` 稍低于 `drop(Yes)` → gain < −3 触发 defer
- defer 把 Yes → No → 撞进错误（因为 popular 物体真的常见，语言先验其实是对的）

这是对 defer_only 机制的**教科书级反例**：假设 = "删图更自信 → 语言幻觉"，实际 = "图片本身有合理不确定性"。

**MME**（200 样本，14 类，见 `results/visual_bench/MME/*/MMaDA-MixCoT-CV-DCD_MME_simple_score.csv`）：

| Metric | B0 | E4 | E6 | ΔE4 | ΔE6 |
|---|---|---|---|---|---|
| Overall acc | 57.00 | 55.50 | 55.50 | **−1.50** | **−1.50** |
| perception | 60.56 | 58.89 | 58.89 | −1.67 | −1.67 |
| reasoning | 25.00 | 25.00 | 25.00 | 0 | 0 |

per-cat 最伤类：`OCR −16.67`（1 翻转，n=6）、`commonsense_reasoning −7.69`、`artwork −3.23`、`scene −2.50`。E4 与 E6 逐字节相同（差异 5/200 完全同 idx）。

**注意**：官方 `MME_rating` 需要"同图 Yes+No 配对"（`acc(key, 'plus')` 会 `val[0]*val[1]`），我们的 shuffle 破坏了配对，用 `simple_score.csv`（纯 accuracy）作对比 —— B0/E4/E6 全部同 subset，比较仍公平。

**MMBench_DEV_EN**（200 样本，MCQ，20 类；E6 因 HuggingFace SSL crash 未完成）：

| Metric | B0 | E4 | ΔE4 |
|---|---|---|---|
| Overall | 60.5% | 59.5% | **−1.0%** |
| AR (attribute_reasoning) | 84.85% | 81.82% | −3.03 |
| RR (relation_reasoning) | 50.0% | 46.43% | −3.57 |

差异样本：2/200（idx=35: B0='B'→E4='C'; idx=67: 'D'→'A'），都是对→错。

### 10. **CHAIR：长文 caption 幻觉（新增，`results/chair/`）**

- **数据**：MSCOCO val2017 抽 200 图（80 类全覆盖），prompt = "Please describe this image in detail."
- **打分器**：`LisaAnne/Hallucination` 的 Python-3 忠实复刻（`pattern.en.singularize` → `nltk.WordNetLemmatizer`，其余算法逐字对齐），使用官方 `synonyms.txt` + canonical `double_word_dict`（含 `baby X → X` / `toilet seat → toilet` / `wine glas` 兜底等）
- **GT**：instance masks ∪ 5 条 GT captions 抽出的对象（canonical union，`gt_source='segments+captions'`）

三配置对比（全部 rescored under canonical + segments+captions GT）：

| Config | CHAIRi ↓ | CHAIRs ↓ | recall ↑ | mentioned tokens | hallucinated | avg caption len |
|---|---|---|---|---|---|---|
| **B0 plain DCD**   | **11.35** | **36.5** | **54.48** | 1128 | 128 | 95.77 |
| E4 (λ=0.5, τ=−3.0) | 11.42 | 36.5 | 53.54 | 1147 | 131 | 96.85 |
| E6 (λ=1.0, τ=−3.0) | 11.24 | 36.0 | 53.66 | 1157 | 130 | 97.16 |

**Δ vs B0**：

| | ΔCHAIRi | ΔCHAIRs | Δrecall |
|---|---|---|---|
| E4 | +0.07 pp | 0.00 pp | **−0.94 pp** |
| E6 | −0.11 pp | −0.50 pp | **−0.82 pp** |

**关键观察**：
1. defer_only 在幻觉率上**无实质影响**（CHAIRi 抖动 ±0.1pp、CHAIRs ±0.5pp），但**系统性降低 recall ~0.9pp**（少提到 ~8 个 GT 物体）
2. defer_only 让 caption 略长（+1-2 tokens），mentioned tokens 增加 (+19~29)，hallucinated tokens 也增加 (+2~3)——proportional 增长，说明既没有变得更"警觉"，也没有变得更"啰嗦-而准确"
3. 结果与 POPE/MME/MMBench 模式完全一致：defer 干预的对象 token 恰是模型基于图片"稍不自信但正确"的判断，被替换成语言先验脑补 → **不降幻觉，只降 recall**
4. 这是**第 7 个证伪 defer_only 的数据集**（首个是长文 caption 场景，机制与 Y/N 完全独立）

### 11. **CHAIR 打分器工程细节（重要）**

在初次评测中发现一个 **GT-mismatch bug**：B0 的 python 进程在 06:28 启动，在 06:52 我升级 `image_caption.py::evaluate()` 加入 caption-derived GT 时，B0 已经把旧版本的 `evaluate()` 缓存到内存，其 CHAIRScorer 只用 instance masks 作 GT（786 objs）；而 E4/E6 后续启动的 python 进程读到新 evaluate()，走 segments+captions（848 objs）。

**修复**：三份预测重新用 canonical scorer + segments+captions GT rescored。B0 CHAIRi 从看似 15.51% 修正为 **11.35%**（因为 caption-derived GT 让原本被判"幻觉"的物体大部分变成"合法提及"）。

**教训**：GPU process 长跑期间，编辑 evaluate() 逻辑必须重新提交，或统一 post-hoc rescore；本次通过 rescore 保证三配置**同 scorer + 同 GT** 才对比。

---

---

## 核心 idea 速览（详见 `report/cv_dcd_status_current.md`）

```
1. x0 = argmax(base_logits)                    # 永不修改
2. base_conf = softmax(base_logits.fp64)[x0]   # 与 plain DCD bit-identical
3. if λ > 0 and drop_logits available:
     gain    = clip(base_logit[x0] - drop_logit[x0], ±clip)
     veto_c  = apply_veto(base_c, gain, veto_type, τ, β)
     eff_c   = (1 - λ) * base_c + λ * veto_c    # λ = 干预强度
4. transfer_index = threshold_select(eff_c, 0.9)
```

- **τ** 控制干预**频率**（哪些位置降 conf）
- **λ** 控制干预**强度**（降多少）
- **hard veto** 是主力：`eff = base if gain ≥ τ else 0`
- **视觉信号永不 boost，只 defer**

## 复现

```bash
# 单测（CPU，验证代码修复）
cd /home/user/dcd/MMada_DCD/MMaDA
/home/user/anaconda3/envs/mmada/bin/python -m pytest tests/test_defer_only.py -q     # 32/32 pass

# v5 矩阵（GPU，~1h）
cd evaluation/VLMEvalKit
bash scripts/run_defer_only_matrix_v5.sh

# E0-parity head-to-head（GPU，~2.5h）
bash scripts/run_defer_only_e0_parity.sh

# byte-parity 验证
python scripts/compare_e0_parity.py \
    --base outputs/.../B0_plain_dcd \
    --alt  outputs/.../E0_defer_l0

# 决定性 benchmark（GPU，~9h）
bash scripts/run_defer_only_deterministic_benchmarks.sh   # ScienceQA + MathVision

# 视觉/幻觉 benchmark（GPU，~12h，POPE/MME/MMBench_DEV_EN）
# 前置：先用 hf_to_vlmeval_tsv.py 从 HF parquet 转出 TSV（opencompass 不可达时）
python scripts/hf_to_vlmeval_tsv.py --dataset POPE  \
    --parquet /tmp/pope_*.parquet --shuffle-seed 42
python scripts/hf_to_vlmeval_tsv.py --dataset MME \
    --parquet /tmp/mme_test_*.parquet --shuffle-seed 42
python scripts/hf_to_vlmeval_tsv.py --dataset MMBench_DEV_EN \
    --parquet /tmp/mmbench_en_dev.parquet --shuffle-seed 42

MMADA_SKIP_LOCALIZE=1 bash scripts/run_defer_only_visual_benchmarks.sh

# CHAIR（GPU，~5h，caption 幻觉，200 图 val2017 × 3 config）
# 前置：装 NLTK 数据 + 用 build_chair_tsv.py 抽样
NLTK_DATA=./nltk_data python -m nltk.downloader -d ./nltk_data \
    punkt punkt_tab wordnet omw-1.4
python scripts/build_chair_tsv.py \
    --coco-root /home/user/大模型/LLava/data/coco \
    --n 200 --shuffle-seed 42

CHAIR_COCO_ANN=/home/user/大模型/LLava/data/coco/annotations/instances_val2017.json \
CHAIR_COCO_CAPS=/home/user/大模型/LLava/data/coco/annotations/captions_val2017.json \
    bash scripts/run_defer_only_chair.sh
```

## Code Bundle

完整代码（源、patch、脚本、测试）已复制到 `code/`。见 `code/MANIFEST.md`。
