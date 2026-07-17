# CV-DCD v4 设计：Sequence-CFG-Rerank + Defer-only CV

**日期：** 2026-07-06
**基础版本：** v3.2 B3 (`cv_mode='cd_apc_v32'`, `cv_conf_source='min_base_blended'`, VLM 28.3)
**范围：** 两条独立的、可并行推进的方向：**方向 A**（整段 CFG 重排）和 **方向 B**（defer-only CV）
**目标：** 干净的模块化实现；不改动 v3.2 现有代码；两个方向共享底层 utility，各自独立子模块

---

## 0. 为什么要 v4

v3.2 报告 (§ 7) 里的结构性诊断已经定性证明：**per-token logit boost 在扩散 mask 解码器上有原理性缺陷。**核心问题是 CD 不改变 argmax 层就无法触及真正的错误 token，但改变了 argmax 又会破坏 LM 的重复抑制先验（并行 masked 位置无法互相看见）。

v4 的两条路都**绕开 per-token logit boost**：

- **方向 A（CFG rerank）**：让 DCD baseline 完整生成 K 个候选，再用 CFG 打分选。视觉信号只在**序列级**起作用，不在每个位置上单独 boost。
- **方向 B（Defer-only CV）**：argmax 走纯 base_logits（永不改），视觉信号只调制 confidence——决定**要不要 commit**，不决定**commit 什么**。

两个方向在设计上完全解耦，代码上共享底层组件。可以并行实现、并行评测、并行 ablation。

---

## 1. 顶层文件结构

```
MMaDA/
├── models/
│   ├── mmada_decode.py                  ← 现状；不改核心逻辑，只新增 dispatch 分支
│   ├── cv_v32_*.py                      ← v3.2 现有 4 文件；完全不动
│   │
│   ├── cv_common/                       ← 新增：两个方向都用的底层 utility
│   │   ├── __init__.py
│   │   ├── image_drop.py                ← 从 mmada_decode.py 抽出 _build_dropped_image
│   │   ├── paired_forward.py            ← 从 mmada_decode.py 抽出 _paired_forward_logits
│   │   ├── log_prob.py                  ← _logp_of_x0 + 新增 stepwise log-p 计算
│   │   └── types.py                     ← 共享 dataclass: StepRecord, Candidate
│   │
│   ├── cfg_rerank/                      ← 新增：方向 A
│   │   ├── __init__.py
│   │   ├── generator.py                 ← K 个候选生成（带 stepwise LL hook）
│   │   ├── uncond_ll.py                 ← 重播 commit 顺序算 uncond log-p
│   │   ├── rerank.py                    ← CFG 打分 + argmax
│   │   └── pipeline.py                  ← 总成入口 cfg_rerank_decode
│   │
│   └── defer_only/                      ← 新增：方向 B
│       ├── __init__.py
│       ├── veto.py                      ← 三种 veto 机制（hard/mult/min）
│       └── dispatcher.py                ← pick_transfer_defer_only
│
├── evaluation/VLMEvalKit/vlmeval/vlm/mmada/
│   └── mmada.py                         ← 添加新 decode_strategy 和环境变量
│
├── tests/
│   ├── test_cv_common.py                ← image_drop / paired_forward / log_prob 单测
│   ├── test_cfg_rerank.py               ← generator / uncond_ll / rerank 单测
│   └── test_defer_only.py               ← veto / dispatcher 单测
│
├── scripts/
│   ├── run_cfg_rerank_sweep.sh          ← 方向 A sweep（K × γ × drop_strategy）
│   └── run_defer_only_sweep.sh          ← 方向 B sweep（veto_type × τ × β）
│
└── docs/
    ├── cv_dcd_full_report.md            ← 现有；v4 结束后追加 § 11 结果
    └── cv_dcd_v4_design.md              ← 本文档
```

**代码总量预算：** ~800 LOC 新代码（cv_common ~200，cfg_rerank ~300，defer_only ~150，测试 ~150）。

---

## 2. 复用组件 `models/cv_common/`

从 `mmada_decode.py` 抽出的**纯函数**，两个方向都要用。原文件的实现保留为 thin wrapper 引用 cv_common，向后兼容 v3.2 不受影响。

### 2.1 `cv_common/image_drop.py`（~60 LOC）

```python
def build_dropped_image(
    x: torch.Tensor,
    config: MMaDADecodeConfig,
) -> torch.Tensor:
    """
    Return x_drop where the visual token span [visual_token_start:visual_token_end)
    is ablated according to config.image_drop_strategy.
    
    Supported strategies (unchanged from v3.2):
      - 'mask'        : replace visual tokens with mask_id (OOD; not recommended)
      - 'shuffle'     : permute visual tokens (default)
      - 'random_mask' : half-random-half-original
      - 'mean_token'  : replace with mean token id
      - 'neutral'     : replace with pre-encoded VQ codes of gray image
      - 'text_only'   : NEW - replace visual span with padding/EOS token
    
    'text_only' is a new strategy for v4: it removes the entire image span
    contribution rather than shuffling it, giving a cleaner "language prior"
    signal. See § 6.1 for discussion.
    """
```

**新增 `text_only` 策略** — 用 EOS / padding token 替换整个 image span。相比 `shuffle`，这更严格地把"图像贡献"清零；相比 `mask_id`，避免了 OOD 问题（EOS/pad 是训练时见过的 token）。

**兼容性：** 原 `_build_dropped_image` 保留在 mmada_decode.py 但改成 `return image_drop.build_dropped_image(x, config)`。v3.2 代码路径零改动。

### 2.2 `cv_common/paired_forward.py`（~80 LOC）

```python
def paired_forward_logits(
    model,
    x: torch.Tensor,
    x_drop: torch.Tensor,
    attention_bias: Optional[torch.Tensor],
    past_key_values: Optional[Any] = None,
    use_cache: bool = False,
    replace_position: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
    """
    Run model on (x, x_drop) simultaneously (concat batch) or sequentially
    (dual cache). Returns (logits, drop_logits, past_kv, past_kv_drop).
    Zero behavioral change from v3.2's _paired_forward_logits.
    """
```

**行为完全不变**，只是从 mmada_decode.py 提到独立文件。

### 2.3 `cv_common/log_prob.py`（~50 LOC）

```python
def logp_of_tokens(
    logits: torch.Tensor,      # [B, L, V]
    tokens: torch.Tensor,      # [B, L]
) -> torch.Tensor:
    """
    Return log p(tokens[b, l] | logits[b, l]) for each position.
    Equivalent to v3.2's _logp_of_x0.
    Returns tensor of shape [B, L] in float64.
    """


def stepwise_logp_from_records(
    step_records: List['StepRecord'],
) -> torch.Tensor:
    """
    Concatenate per-step committed log-p into a single [n_committed] tensor.
    Preserves commit order; useful for aggregating full-sequence pseudo-LL.
    """


def aggregate_sequence_ll(
    step_records: List['StepRecord'],
    reduction: str = "sum",  # 'sum' | 'mean' | 'length_normalized'
) -> float:
    """
    Reduce stepwise log-p to a single scalar for rerank scoring.
    'length_normalized' divides by n_committed to remove length bias.
    """
```

**新增：** `stepwise_logp_from_records` 和 `aggregate_sequence_ll` 是方向 A 用的。

### 2.4 `cv_common/types.py`（~30 LOC）

```python
@dataclass
class StepRecord:
    """Per-step commit info recorded during DCD decoding.
    
    Used by:
      - cfg_rerank/generator.py: hook writes it during generation
      - cfg_rerank/uncond_ll.py: reader replays it under x_drop
    """
    step_idx: int
    committed_positions: List[int]     # positions committed at this step
    committed_tokens: List[int]        # tokens that were committed
    logp_cond: torch.Tensor            # log p(commit | x, image) [n_commit]
    # NOTE: logp_uncond is filled later by uncond_ll.py (initially None)
    logp_uncond: Optional[torch.Tensor] = None


@dataclass
class Candidate:
    """One decoded sequence + its scoring metadata."""
    seq: torch.Tensor                  # [1, L] full decoded token ids
    step_records: List[StepRecord]
    seed: int
    total_logp_cond: float = 0.0
    total_logp_uncond: float = 0.0
    cfg_score: float = 0.0
```

---

## 3. 方向 A：Sequence-level CFG Rerank

### 3.1 数学与假设

**核心想法：** DCD baseline 已经能生成通顺、grounded 的答案（有 22/60 样本 baseline 得分 ≥5）。问题在于随机性——有些 seed 生成的答案 grounded，有些不 grounded。**如果我们能识别出哪个候选更 grounded，就选它。**

**打分公式（Classifier-Free Guidance 应用到序列级）：**

$$
\text{score}(y) = (1 + \gamma) \cdot \log p(y \mid I, x) - \gamma \cdot \log p(y \mid \varnothing, x)
$$

$$
= \log p(y \mid I, x) + \gamma \cdot \underbrace{\left[\log p(y \mid I, x) - \log p(y \mid \varnothing, x)\right]}_{\text{visual gain of the whole sequence}}
$$

其中：
- $y$ = 候选文本序列
- $I$ = 图像
- $x$ = 用户 prompt
- $\varnothing$ = 无图像（用 image_drop_strategy 处理）
- $\gamma$ = guidance strength（0 = 纯 LM，>0 = 越大越倾向 grounded）

**扩散模型 $\log p(y)$ 怎么算？**（这是你之前选择的 stepwise 方案）

我们**不算** $\log p(y)$ 的严格边缘分布（DDPM ELBO 太贵）。而是**复用 DCD 生成时的 commit 顺序**：

1. 生成 $y$ 时，DCD 每一步在当前 $x^{(t)}$ 下 forward 一次，得到 $\log p(x_0[\text{commit}] \mid x^{(t)}, I)$。把这些累加起来就是 $\log p(y \mid I, x)$ 的 stepwise 近似。
2. 算 unconditional 时，**用同一个 commit 顺序**，但每一步的 forward 换成 $x_{\text{drop}}^{(t)}$（image ablation），得到 $\log p(x_0[\text{commit}] \mid x_{\text{drop}}^{(t)}, I=\varnothing)$。累加得到 $\log p(y \mid \varnothing, x)$ 的 stepwise 近似。

**关键性质：** 因为两个 LL 都用**同一个 commit 顺序**（同一个"扩散路径"），差值 $\log p_\text{cond} - \log p_\text{uncond}$ 就是**这个序列因为图像存在而获得的额外似然**。这正是我们想要的 visual gain。

### 3.2 数据流

```
input: (image, prompt, K seeds, γ)
  │
  ▼
For each seed in [s_1, ..., s_K]:
  │   ┌─── run baseline DCD with temperature > 0 ────────────┐
  │   │   at each commit step, hook records:                  │
  │   │     - step_idx                                        │
  │   │     - committed positions & tokens                    │
  │   │     - logp_cond = log p(committed | x^(t), I)         │
  │   └────────────────────────────────────────────────────────┘
  │   → Candidate(seq, step_records)
  │
  ▼
For each candidate:
  │   ┌─── replay step_records under x_drop ─────────────────┐
  │   │   at each step t:                                    │
  │   │     x_drop^(t) = build_dropped_image(x^(t))          │
  │   │     logits_drop = model(x_drop^(t))                  │
  │   │     logp_uncond[t] = logp_of_tokens(                 │
  │   │        logits_drop[committed_positions],             │
  │   │        committed_tokens)                             │
  │   └────────────────────────────────────────────────────────┘
  │   → candidate.total_logp_uncond
  │
  ▼
For each candidate:
  │   cfg_score = (1+γ)·total_logp_cond − γ·total_logp_uncond
  │
  ▼
best = argmax(candidates, by cfg_score)
return best.seq
```

**NFE 预算：**
- 单个 baseline DCD ≈ $N$ NFE（LLaVABench length=256, block=32, 约 25 NFE）
- $K$ 个候选生成：$K \cdot N$
- $K$ 个 uncond replay：$K \cdot N$（每个 step 一次 drop forward）
- **总计：** $2 \cdot K \cdot N$，即 baseline 的 $2K$ 倍
- $K=4$ 时约 **8× baseline NFE**

### 3.3 模块划分与接口

#### 3.3.1 `cfg_rerank/generator.py`（~120 LOC）

```python
from typing import Callable, List, Optional
import torch
from ..cv_common.types import Candidate, StepRecord

def generate_candidates(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: 'MMaDADecodeConfig',
    seeds: List[int],
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
) -> List[Candidate]:
    """
    Generate K candidates via baseline DCD, one per seed.
    
    Contract:
      - Each seed produces one Candidate with populated step_records.
      - config.temperature must be > 0 for candidates to differ.
      - Uses the currently-selected cache_type (none/prefix/dual) internally.
      - Does NOT compute drop_logits (that's uncond_ll.py's job).
    """


def _run_dcd_with_hook(
    model, tokens, decode_start, decode_end,
    config, seed, attention_bias, prompt_index,
) -> Candidate:
    """
    Internal: one DCD run with a step-level hook that records logp_cond.
    Structured to mirror dcd_decode_text_dual_cache() but with hook injection.
    """


def _record_step(
    logits: torch.Tensor,
    x0: torch.Tensor,
    transfer_index: torch.Tensor,
    step_idx: int,
) -> StepRecord:
    """Extract StepRecord from a single commit step."""
```

**关键实现细节：**
- **多样性来源：** 每个 seed 前调用 `torch.manual_seed(seed)`，DCD 的 Gumbel noise 自动带上差异。
- **Hook 位置：** 在 `_pick_transfer` 返回 `(x0, transfer_index)` 之后立即调用 `_record_step`，用当前 forward 的 logits 计算 logp。
- **不复用 `_pick_transfer_cv`：** 候选生成走**纯 baseline DCD**，不掺 CV 逻辑。这样 rerank 才是纯"事后打分"，语义干净。

#### 3.3.2 `cfg_rerank/uncond_ll.py`（~80 LOC）

```python
def compute_uncond_ll(
    model,
    tokens: torch.Tensor,
    candidate: Candidate,
    decode_start: int,
    decode_end: int,
    config: 'MMaDADecodeConfig',
    attention_bias: Optional[torch.Tensor] = None,
) -> Candidate:
    """
    Given a candidate produced by generator.generate_candidates(), replay
    its commit trajectory under x_drop and fill candidate.step_records[i].logp_uncond.
    
    Modifies candidate in place; also returns it for chaining.
    
    Contract:
      - Uses config.image_drop_strategy for ablation.
      - NFE = number of DCD steps in the candidate's generation.
      - Does NOT re-run DCD; only forward-once per step with x_drop.
    """


def _replay_step_uncond(
    model,
    x_drop: torch.Tensor,
    step_record: StepRecord,
    attention_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    One-step uncond forward:
      logits_drop = model(x_drop)
      logp_uncond = logp_of_tokens(
          logits_drop[step_record.committed_positions],
          step_record.committed_tokens)
    """


def _reconstruct_x_at_step(
    initial_tokens: torch.Tensor,
    candidate: Candidate,
    up_to_step: int,
    decode_start: int,
) -> torch.Tensor:
    """
    Build x^(t) at step t by starting from initial_tokens (all-mask in decode
    span) and applying candidate.step_records[0..up_to_step-1] in order.
    Then build_dropped_image on top of this.
    """
```

**关键实现细节：**
- **必须重建 $x^{(t)}$：** step $t$ 时的部分-decoded 状态。因为 candidate 只存了 final seq 和 step_records，重建需要 replay 前 $t-1$ 步的 commit。
- **NFE 优化：** 可以尝试 cache 复用（KV cache），但初版不做，先追求正确性。
- **顺序 identity check：** 每一步 replay 后可以断言 $\log p_\text{drop}$ 的 argmax **不必等于** committed token（因为图像被移除后 argmax 会变），但 committed token 的 rank 应该合理。这可以做 debug 断言。

#### 3.3.3 `cfg_rerank/rerank.py`（~40 LOC）

```python
def compute_cfg_score(
    candidate: Candidate,
    gamma: float,
    ll_reduction: str = "sum",
) -> float:
    """
    score = (1 + gamma) * logp_cond - gamma * logp_uncond
          = logp_cond + gamma * (logp_cond - logp_uncond)
    
    ll_reduction: 'sum' | 'mean' | 'length_normalized'
      - 'sum'                : Σ_t logp_t (default; length-biased)
      - 'mean'               : Σ_t logp_t / n_committed
      - 'length_normalized'  : Σ_t logp_t / n_committed^0.7 (partial correction)
    """


def pick_best_candidate(candidates: List[Candidate]) -> Tuple[Candidate, int]:
    """Return (best_candidate, best_idx) by cfg_score."""


def rerank_all(
    candidates: List[Candidate],
    gamma: float,
    ll_reduction: str = "sum",
) -> List[Candidate]:
    """Populate candidate.cfg_score for all, then return sorted list."""
```

**length bias：** 关键讨论——sum 会偏向短序列（每个 token 贡献都是负数）。默认用 `'sum'`（对 DCD 输出的相对长度差异容忍），但保留 `'mean'` 和 `'length_normalized'` 供 ablation。

#### 3.3.4 `cfg_rerank/pipeline.py`（~80 LOC）

```python
@torch.no_grad()
def cfg_rerank_decode(
    model,
    tokens: torch.Tensor,
    decode_start: int,
    decode_end: int,
    config: 'MMaDADecodeConfig',
    attention_bias: Optional[torch.Tensor] = None,
    prompt_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Top-level entry point for CFG rerank.
    
    Pipeline:
      1. candidates = generate_candidates(model, ..., seeds=config.rerank_seeds)
      2. for c in candidates: compute_uncond_ll(model, ..., c)
      3. for c in candidates: c.cfg_score = compute_cfg_score(c, config.rerank_gamma)
      4. best, best_idx = pick_best_candidate(candidates)
      5. return best.seq   (also debug info if config.return_debug)
    
    Config fields consumed:
      - rerank_k, rerank_gamma, rerank_seeds
      - image_drop_strategy (for uncond_ll)
      - temperature (must be > 0)
    """


def _default_seeds(k: int, base_seed: int = 0) -> List[int]:
    """Return [base_seed, base_seed+1, ..., base_seed+k-1]."""
```

**返回值：** 与其他 `dcd_decode_text_*` 保持一致 —— 默认返回 `x`；`config.debug=True` 返回 `(x, nfe)`；`config.return_debug=True` 返回 `(x, debug_dict)` 其中 debug_dict 包含所有候选的 cfg_score、step_records 摘要。

### 3.4 Config 扩展

在 `MMaDADecodeConfig` 里新增字段（不改现有字段，不动 v3.2）：

```python
# --- CFG Rerank (v4 direction A) ---
rerank_k: int = 4
rerank_gamma: float = 1.0
rerank_seeds: Optional[List[int]] = None   # None → [0, 1, ..., k-1]
rerank_ll_reduction: str = "sum"           # sum | mean | length_normalized
# reuses: image_drop_strategy, temperature
```

### 3.5 环境变量与 wrapper 集成

`mmada.py` 添加：

```python
# 新 decode_strategy
elif self.decode_strategy == 'cfg_rerank':
    self.dcd_config = MMaDADecodeConfig(
        # ... base DCD fields as usual ...
        temperature=float(os.getenv('MMADA_TEMPERATURE', dcd_temperature)),
        image_drop_strategy=os.getenv('MMADA_CV_DROP', 'shuffle'),
        rerank_k=int(os.getenv('MMADA_RERANK_K', 4)),
        rerank_gamma=float(os.getenv('MMADA_RERANK_GAMMA', 1.0)),
        rerank_seeds=_parse_int_list(os.getenv('MMADA_RERANK_SEEDS')),
        rerank_ll_reduction=os.getenv('MMADA_RERANK_LL_REDUCE', 'sum'),
    )
```

新增环境变量：
- `MMADA_DECODE_STRATEGY=cfg_rerank`
- `MMADA_RERANK_K` （默认 4）
- `MMADA_RERANK_GAMMA`（默认 1.0）
- `MMADA_RERANK_SEEDS` （默认自动 `0,1,2,3`，可传 `"7,42,123,999"`）
- `MMADA_RERANK_LL_REDUCE` （默认 `sum`）

**Dispatch：** `mmada.py` 里 `_generate_text_with_dcd` 增加一条 branch：

```python
if config.rerank_k > 0 and self.decode_strategy == 'cfg_rerank':
    from ..models.cfg_rerank.pipeline import cfg_rerank_decode
    return cfg_rerank_decode(...)
```

### 3.6 单元测试 `tests/test_cfg_rerank.py`（~80 LOC）

在 CPU 上、假模型上测：

1. **`test_stepwise_ll_sums_correctly`** — 手工构造 3 步的 StepRecord，验证 `aggregate_sequence_ll` 结果 == 手算和。
2. **`test_uncond_ll_reconstructs_x_at_step`** — 3 步的 candidate，验证 `_reconstruct_x_at_step(t=2)` 等于把 step_records[0..1] 应用到 initial_tokens。
3. **`test_cfg_score_math`** — 验证 `(1+γ)·cond − γ·uncond == cond + γ·(cond − uncond)`（浮点精度内）。
4. **`test_rerank_picks_larger_gain`** — 两个候选，一个 cond=10 uncond=5，另一个 cond=8 uncond=7。γ=1 时应该选第一个（gain=5 vs gain=1）。
5. **`test_generator_seeds_produce_different_seqs`** — 用 pytest monkeypatch 的假模型验证不同 seed 走不同 argmax。

### 3.7 实验计划（方向 A）

**Phase A1：可行性 smoke test（5 样本，~15 分钟）**
- 单一配置 `K=4, γ=1.0, drop=shuffle`，跑 5 个样本，检查：
  - 4 个候选是否 mutually distinct？（相同就是 seed 不生效）
  - cfg_score 是否合理 ranged（-2000 ~ -500，取决于 length）？
  - best_idx 分布：不是永远选 0 号或永远选 K-1 号？

**Phase A2：核心 sweep（60 样本，8× NFE，~2 小时）**

| Tag | K | γ | drop | 假设 |
|---|---|---|---|---|
| A1 | 4 | 0.0 | shuffle | Sanity: γ=0 应约等于 baseline seed=0（无视觉信号）|
| A2 | 4 | 0.5 | shuffle | 弱 guidance |
| A3 | 4 | 1.0 | shuffle | 中等 guidance（预期最佳）|
| A4 | 4 | 2.0 | shuffle | 强 guidance |
| A5 | 4 | 1.0 | text_only | 更干净的 uncond；预期比 shuffle 好 |
| A6 | 4 | 1.0 | neutral | Neutral image ablation |
| A7 | 6 | 1.0 | shuffle | 增大候选池，看是否单调改善 |

**期望信号：**
- 至少一个配置 VLM > 28.3（超过 B3 就有价值）
- γ=0 与 baseline temperature=0.8 单跑相差 <1 分（内部 sanity）
- 更大 K 单调改善（如果不改善说明多样性瓶颈）

---

## 4. 方向 B：Defer-only CV

### 4.1 数学与假设

**核心想法：** 保留 DCD 的完整语义——argmax 由 base_logits 决定（跟 baseline 完全一样），CV 只调制**是否要 commit**。视觉信号只当"否决权"用，永不主动 boost 任何 token。

**为什么这样能起作用？** DCD baseline 已经能生成正确 token（v3.2 观察：60 样本 tied=18，argmax 层能给出好结果的样本很多）。问题是 DCD 有时在低置信位置 commit 错误 token（e.g. idx=45 "1"）。如果 CV 能识别出"这个位置视觉支持不足，别急着 commit"，就能让扩散多迭代几轮，让上下文帮助解开歧义。

**三种 veto 变体（全部实现，用 env var 切换 ablation）：**

**Variant 1 — Hard veto**
$$
\text{eff\_conf}(pos) = \begin{cases}
\text{base\_conf}(pos) & \text{if } \text{gain}(pos) \geq \tau \\
0 & \text{otherwise}
\end{cases}
$$

**Variant 2 — Multiplicative penalty**
$$
\text{eff\_conf}(pos) = \text{base\_conf}(pos) \cdot \sigma\left(\beta \cdot (\text{gain}(pos) - \tau)\right)
$$

其中 $\sigma$ 是 sigmoid。`gain >> τ` 时 sigmoid → 1（不惩罚）；`gain << τ` 时 sigmoid → 0（重压）。$\beta$ 控制陡峭度。

**Variant 3 — Min form**
$$
\text{eff\_conf}(pos) = \min\left(\text{base\_conf}(pos), \sigma\left(\beta \cdot \text{gain}(pos)\right)\right)
$$

思想上和 v3.2 B3 的 `min_base_blended` 类似——两个信号都要过关。但这里第二个信号是**纯 visual gain**，不是 blended logits。

**注意：** 三种 veto 都**不改 argmax**——`x0 = argmax(base_logits)`。只改 conf，进而影响 `_select_transfer` 里 threshold 决定是否 commit。

### 4.2 数据流

```
input: (base_logits, drop_logits, mask_index, current_tokens, config)
  │
  ▼
x0 = argmax(base_logits)  ──── 完全走 base；不 CD，不 boost
  │
  ▼
base_conf = softmax(base_logits)[x0]   ── 完全走 base
  │
  ▼
if drop_logits is not None:
    gain = clamp(logp(x0|image) − logp(x0|no image), [−clip, +clip])
    eff_conf = apply_veto(base_conf, gain, veto_type, τ, β)
else:
    eff_conf = base_conf   ── stride > 1 的 step 完全等于 baseline DCD
  │
  ▼
feed (x0, eff_conf) into _select_transfer(...)  ── DCD 原逻辑不动
  │
  ▼
return (x0, transfer_index)
```

**关键差异 vs v3.2 CV 家族：**
- `selection_logits = base_logits`（不掺 drop_logits）
- `blended_logits` 概念不存在
- 只有一个 conf 源（base_conf 或它的 veto 版），不用 dispatch
- Gating 不需要（veto 本身就是 gating——gain 高即"gate open"）

### 4.3 模块划分与接口

#### 4.3.1 `defer_only/veto.py`（~100 LOC）

```python
from enum import Enum
import torch

class VetoType(str, Enum):
    HARD = "hard"
    MULT = "mult"
    MIN  = "min"


def compute_visual_gain(
    base_logits: torch.Tensor,      # [B, L, V]
    drop_logits: torch.Tensor,      # [B, L, V]
    x0: torch.Tensor,               # [B, L] base argmax
    causal_clip: float,
) -> torch.Tensor:
    """
    gain[b,l] = clamp(
        log_softmax(base_logits[b,l])[x0[b,l]] −
        log_softmax(drop_logits[b,l])[x0[b,l]],
        min=-causal_clip, max=+causal_clip
    )
    Returns [B, L] float64.
    """


def apply_veto_hard(
    base_conf: torch.Tensor,
    gain: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """conf[b,l] = base_conf[b,l] if gain[b,l] >= tau else 0.0"""


def apply_veto_mult(
    base_conf: torch.Tensor,
    gain: torch.Tensor,
    tau: float,
    beta: float,
) -> torch.Tensor:
    """conf[b,l] = base_conf[b,l] * sigmoid(beta * (gain[b,l] - tau))"""


def apply_veto_min(
    base_conf: torch.Tensor,
    gain: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """conf[b,l] = min(base_conf[b,l], sigmoid(beta * gain[b,l]))"""


def apply_veto(
    base_conf: torch.Tensor,
    gain: torch.Tensor,
    veto_type: VetoType,
    tau: float,
    beta: float,
) -> torch.Tensor:
    """
    Dispatcher. All three variants are pure functions of (base_conf, gain).
    Returns [B, L] with values in [0, 1].
    """
```

**接口设计原则：**
- 全部纯函数、无副作用、tensor in tensor out
- 输入尺寸完全对齐（`base_conf` 和 `gain` 都是 `[B, L]`）
- 无 `config` 参数，只接收具体 float——便于单元测试和复用

#### 4.3.2 `defer_only/dispatcher.py`（~90 LOC）

```python
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from .veto import compute_visual_gain, apply_veto, VetoType

@torch.no_grad()
def pick_transfer_defer_only(
    base_logits: torch.Tensor,
    drop_logits: Optional[torch.Tensor],
    config: 'MMaDADecodeConfig',
    mask_index: torch.Tensor,
    current_tokens: torch.Tensor,
    debug_records: Optional[List[Dict[str, Any]]] = None,
    step_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Defer-only CV:
      1. x0 = argmax(base_logits)   (with optional Gumbel via temperature)
      2. base_conf = softmax(base_logits)[x0]
      3. If drop_logits is not None:
           gain = compute_visual_gain(base_logits, drop_logits, x0, config.causal_clip)
           eff_conf = apply_veto(base_conf, gain, config.defer_veto_type,
                                 config.defer_tau, config.defer_beta)
         Else:
           eff_conf = base_conf
      4. transfer_index = _select_transfer(eff_conf, mask_index, config)
      5. Optional debug: record gain, base_conf, eff_conf, veto_active.
    
    Returns (x0, transfer_index) — same signature as _pick_transfer.
    """


def _select_transfer(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    config: 'MMaDADecodeConfig',
) -> torch.Tensor:
    """
    Reuse the DCD threshold/factor selection logic. Extract it from
    _pick_transfer() in mmada_decode.py (already exists inline).
    
    NOTE: We prefer to import from mmada_decode.py rather than duplicate.
    Add a public `pick_transfer_by_confidence(...)` there if needed.
    """


def _record_defer_debug(
    debug_records: List[Dict[str, Any]],
    step_idx: int,
    batch_idx: int,
    chosen_positions: torch.Tensor,
    base_conf: torch.Tensor,
    gain: Optional[torch.Tensor],
    eff_conf: torch.Tensor,
    veto_type: str,
    x0: torch.Tensor,
) -> None:
    """Per-committed-position debug entry: base_conf, gain, eff_conf, veto_active."""
```

**集成到 mmada_decode.py：**

在 `_pick_transfer_cv` 里加一条分支：

```python
if mode == "defer_only":
    from .defer_only.dispatcher import pick_transfer_defer_only
    return pick_transfer_defer_only(
        logits, drop_logits, config, mask_index, current_tokens,
        debug_records=debug_records,
        step_idx=step_idx,
    )
```

现有的 `dcd_decode_text_cv_dual_cache` 等入口零改动——它们已经调用 `_pick_transfer_cv` 做 dispatch。

### 4.4 Config 扩展

```python
# --- Defer-only CV (v4 direction B) ---
# Activated by cv_mode == 'defer_only'
defer_veto_type: str = "mult"    # 'hard' | 'mult' | 'min'
defer_tau: float = 0.0           # gain threshold
defer_beta: float = 1.0          # sigmoid steepness (for mult/min)
# reuses: causal_lambda (interpreted as "on/off"; 0 means skip drop forward)
#         causal_clip, cv_stride, image_drop_strategy
```

**说明：**
- `cv_mode='defer_only'` 是新增的 dispatch key
- `causal_lambda` 在 defer_only 里语义变了——不再是 boost 系数，而是"是否启用 defer" 的开关（0 = 完全等价 baseline DCD）。这有点丑，可以考虑用 `defer_enable: bool` 更明确。**建议保留 `causal_lambda`，因为它已经控制"是否算 drop forward"，语义一致。**

### 4.5 环境变量与 wrapper

`mmada.py` 添加：

```python
self.defer_veto_type = os.getenv('MMADA_DEFER_VETO', 'mult')
self.defer_tau = float(os.getenv('MMADA_DEFER_TAU', 0.0))
self.defer_beta = float(os.getenv('MMADA_DEFER_BETA', 1.0))

# in dcd_config construction:
defer_veto_type=self.defer_veto_type,
defer_tau=self.defer_tau,
defer_beta=self.defer_beta,
```

**激活方式：**

```bash
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=defer_only
export MMADA_DEFER_VETO=mult       # 或 hard, min
export MMADA_DEFER_TAU=0.0
export MMADA_DEFER_BETA=1.0
export MMADA_CV_LAMBDA=0.5          # >0 to enable drop forward
export MMADA_CV_DROP=shuffle
```

### 4.6 单元测试 `tests/test_defer_only.py`（~80 LOC）

1. **`test_gain_zero_when_logits_equal`** — `base==drop` → gain=0，任何 veto 应该退化为 `eff_conf = base_conf` 或 `eff_conf = min(base_conf, 0.5)`（min 变体的 sigmoid(0)=0.5）。
2. **`test_hard_veto_semantics`** — 手工 `gain = [-1, 0, 1, 2]`, `tau=0.5`。期望 `eff_conf = [0, 0, base_conf, base_conf]`。
3. **`test_mult_veto_monotonic`** — 固定 `base_conf=0.9`，`gain` 从 -2 到 +2 单调递增，`eff_conf` 应该单调递增（sigmoid 单调）。
4. **`test_min_veto_never_boosts`** — 任意输入下 `eff_conf ≤ base_conf`。
5. **`test_pick_transfer_defer_only_argmax_unchanged`** — 传入 `base_logits` argmax 为某个 token，`x0` 输出必须与之相等（**关键：不能被 CD 改变**）。
6. **`test_pick_transfer_defer_only_no_drop_equals_baseline`** — 当 `drop_logits=None` 时，行为必须与 `_pick_transfer(base_logits, ...)` 完全一致。这是"stride>1 步应该等于 baseline"的强保证。

### 4.7 实验计划（方向 B）

**Phase B1：Sanity（5 样本，20 分钟）**
- `veto=mult, τ=0, β=1`, `λ=0.5`, drop=shuffle
- 检查生成结果不崩、length ≈ baseline（因为 argmax 完全走 base）

**Phase B2：Veto 类型对比（60 样本，1× NFE，~30 分钟每组）**

| Tag | veto | τ | β | 假设 |
|---|---|---|---|---|
| D1 | hard | 0 | — | 最激进；期望 length 更长（更多 defer）|
| D2 | mult | 0 | 1 | 中等；期望是 3 者最好 |
| D3 | mult | 0 | 2 | 更陡的 sigmoid |
| D4 | min | — | 1 | 双保险；类似 B3 minbb 语义 |
| D5 | mult | 0.5 | 1 | 更严格的 τ |

**关键对照：**
- **D0 (skip if λ=0)**: 应等价于 baseline DCD（36.7 VLM）——这是**内部一致性 sanity**
- **D1-D5 vs baseline**: 期望至少一个配置 **VLM > 36.0**（比 baseline 差 <1 分即"没伤到"，是这条路的最低价值证明）；**理想 VLM > 37**（真正超过）

**理由：** defer-only 保留了 baseline 的所有正确决策，只是让扩散多迭代来解决歧义。如果这样都不能超过 baseline，说明 gain 信号对 confidence 的调制没用——那 gain 就真的没有可用的信号，方向 A 也悬。

**这是 v4 里最便宜、最能定性证伪整个 CV 假设的实验。**

---

## 5. 共同风险 & 缓解

### 5.1 image_drop_strategy 的信号质量

v3.2 已知：`shuffle` 是 median gain ≈ 0 的噪声信号。v4 会同时试 `text_only`（新增）+ `neutral`。**建议默认 `text_only`，`shuffle` 保留作 ablation。**

**text_only 具体实现（`cv_common/image_drop.py` 新增）：**
```python
elif strategy == "text_only":
    # Replace entire visual span with pad_token_id (or eos_token_id).
    # This is stronger than shuffle: removes all visual contribution
    # rather than just spatial order. Safer than mask_id (which is
    # OOD for the diffusion model).
    pad_id = config.text_only_fill_id  # 新增字段，默认 tokenizer pad_token_id
    x_drop[:, s:e] = pad_id
```

需要 wrapper 在 config 构造时填 `text_only_fill_id = self.tokenizer.pad_token_id`（或 `eos_token_id`）。

### 5.2 NFE 预算

- **方向 B: 1× baseline NFE**（跟 CV-DCD v3.2 一样，一次 paired forward）—— 便宜。
- **方向 A: 8× baseline NFE** —— 昂贵。**建议先跑 5 样本 smoke test，再决定要不要跑 60 样本。**

### 5.3 v3.2 与 v4 的隔离

**硬性约束：** v4 代码**不改** `models/cv_v32_*.py` 和 `models/mmada_decode.py` 的现有逻辑，只**添加**：
- 新分支到 `_pick_transfer_cv`（defer_only）
- 新 dispatch 到 `dcd_decode_text_dual_cache`（cfg_rerank）—— 通过 wrapper 层直接调 `cfg_rerank_decode`，绕开 `_pick_transfer_cv`
- 新 config 字段（默认值保证 v3.2 行为不变）

v3.2 的所有已通过测试和 sweep 数据保持有效，v4 失败不影响 v3.2。

### 5.4 stepwise LL 的正确性

`compute_uncond_ll` 里 `_reconstruct_x_at_step` 逻辑复杂，容易写错。**测试策略：**
- 单元测试用 3 步的假 candidate 手算验证
- 集成测试：把同一个 candidate 用 image forward 得 logp_cond，用 image drop=identity（即不 drop）forward 得 logp_uncond。这时应该 `logp_cond == logp_uncond`（在浮点精度内）。这是"drop=noop 时 uncond=cond"的强 sanity 断言。

### 5.5 seed 与 CUDA 非确定性

`torch.manual_seed(seed)` 不保证 CUDA kernel（如 `topk` / attention）完全确定。但对 rerank 只需要"候选之间有差异"即可，不需要严格可复现。**Phase A1 smoke test 会验证候选真的不同。**

---

## 6. 实施顺序与里程碑

| Step | 内容 | 预计工作量 | 依赖 |
|---|---|---|---|
| **M0** | 抽出 `cv_common/` 三个文件；跑通 v3.2 全部单元测试 | 半天 | 无 |
| **M1** | 实现方向 B `defer_only/veto.py` + `dispatcher.py` + 单测 | 半天 | M0 |
| **M2** | 方向 B 集成 wrapper + smoke test（5 sample） | 半天 | M1 |
| **M3** | 方向 B Phase B2 sweep（5 组 × 60 样本） | 3-4 小时 GPU | M2 |
| **M4** | 实现方向 A `cfg_rerank/` 四个文件 + 单测 | 1 天 | M0 |
| **M5** | 方向 A 集成 wrapper + smoke test（5 sample） | 半天 | M4 |
| **M6** | 方向 A Phase A2 sweep（7 组 × 60 样本，8× NFE） | 10-16 小时 GPU | M5 |
| **M7** | 综合分析 + 更新 cv_dcd_full_report.md | 半天 | M3, M6 |

**关键 checkpoint：**
- **M3 结束**：如果方向 B 最好配置 VLM < 35（即比 baseline 差 >2 分），**暂停方向 A**——说明 gain 信号根本没用，rerank 也救不了。
- **M3 结束**：如果方向 B 最好配置 VLM > 37（真正超过 baseline），可以先冷冻 A，专攻 B 的更细粒度调参。

---

## 7. 与现有代码的关系

### 7.1 v3.2 兼容性矩阵

| 修改项 | v3.2 行为 | v4 之后 |
|---|---|---|
| `_build_dropped_image` in mmada_decode.py | 定义 | 变成 `models.cv_common.image_drop.build_dropped_image` 的 thin wrapper |
| `_paired_forward_logits` in mmada_decode.py | 定义 | 变成 `models.cv_common.paired_forward.paired_forward_logits` 的 thin wrapper |
| `_logp_of_x0` in mmada_decode.py | 定义 | 变成 `models.cv_common.log_prob.logp_of_tokens` 的 thin wrapper |
| `_pick_transfer_cv` dispatch | 4 分支 (cd_apc/cd_apc_v32/cd_naive/legacy_score/off) | +1 分支 (defer_only) |
| `dcd_decode_text_cv*` 函数 | 3 个 (base/prefix/dual cache) | 完全不动 |
| `MMaDADecodeConfig` 字段 | 现有 | +6 新字段（defer_only×3 + rerank×3），全部有默认值 |
| `mmada.py` wrapper | 现有 | +新 env vars, +新 `decode_strategy='cfg_rerank'` 分支 |

**验证：** 抽出 cv_common 后，跑一遍 v3.2 的完整单元测试（`tests/test_cv_v32.py` 全部 19 项）+ 5 样本 sanity，必须**完全一致的输出**（可以对比 v3.2 B3 sanity 结果做 bit-exact 检查）。

### 7.2 环境变量总表

| 变量 | 方向 A | 方向 B | 说明 |
|---|---|---|---|
| `MMADA_DECODE_STRATEGY` | `cfg_rerank` | `cv_dcd` | Top-level dispatch |
| `MMADA_CV_MODE` | — | `defer_only` | v3.2 兼容; 方向 B 用 |
| `MMADA_CV_LAMBDA` | — | 需 >0 | 方向 B 用来 gate drop forward |
| `MMADA_CV_DROP` | ✓ | ✓ | image_drop_strategy |
| `MMADA_RERANK_K` | ✓ | — | 候选数 |
| `MMADA_RERANK_GAMMA` | ✓ | — | CFG 系数 |
| `MMADA_RERANK_SEEDS` | ✓ | — | 种子列表 |
| `MMADA_RERANK_LL_REDUCE` | ✓ | — | LL 聚合方式 |
| `MMADA_DEFER_VETO` | — | ✓ | hard/mult/min |
| `MMADA_DEFER_TAU` | — | ✓ | gain 阈值 |
| `MMADA_DEFER_BETA` | — | ✓ | sigmoid 斜率 |
| `MMADA_TEMPERATURE` | ✓ | — | 候选多样性来源 |

### 7.3 输出格式

**方向 B**：与 v3.2 一致——只是 `_pick_transfer_cv` 的新分支。所有 `dcd_decode_text_cv_dual_cache` 等入口的返回签名不变。

**方向 A**：新增 `cfg_rerank_decode` 顶层入口，签名与 `dcd_decode_text_dual_cache` 一致：`(x)` 或 `(x, nfe)` 或 `(x, debug_dict)`。debug_dict 结构：

```python
{
    "nfe": int,
    "n_candidates": int,
    "best_idx": int,
    "candidates": [
        {
            "seed": int,
            "seq_len": int,
            "n_committed": int,
            "total_logp_cond": float,
            "total_logp_uncond": float,
            "cfg_score": float,
            "step_records": [...compact summary...],
        },
        ...
    ],
    "chosen_seq_tokens": List[int],
}
```

---

## 8. 开放问题

以下问题不影响本次实现，但影响未来演进方向。留下笔记备后续讨论：

1. **方向 A 的候选生成是否应该本身就用 CV-DCD (B3)？**
   - 当前设计：候选生成走**纯 baseline DCD**，rerank 是纯"事后打分"。语义最干净。
   - 替代：候选生成走 v3.2 B3（min_base_blended），rerank 再挑最好的。语义混合，但可能更好——B3 已经能"救烂样本"，rerank 再选。
   - **建议：** M6 sweep 里可以加一组 `generator=v3.2_B3` 做 ablation。

2. **stepwise LL 是否需要考虑 x_drop^(t) 的合法性？**
   - 我们从 candidate 的 step_records 重建 `x^(t)`，然后 `build_dropped_image(x^(t))`。
   - 但 `x^(t)` 包含已 commit 的**文本** token 和未 commit 的 mask。drop 时只改 image span——**文本 token 保留**。这符合"只把图像 counterfactual 掉"的语义。
   - ✅ 无问题。

3. **方向 B 的 `causal_lambda` 语义模糊**
   - 现在 `causal_lambda` 既控制"是否算 drop"，又控制 CD boost 强度。方向 B 只用前者。
   - **建议：** 长期可以拆成 `enable_drop_forward: bool` 和 `causal_lambda: float`。短期先复用。

4. **KV cache 复用是否可能？**
   - 方向 A 的 uncond replay 每一步都 forward 一次 `model(x_drop^(t))`——**不能复用 candidate 生成时的 KV cache**（那是 image 版本）。
   - 但 uncond replay 内部的多次 forward，可以自己维护 KV cache。收益可能是 30-50% 加速。
   - **建议：** 初版不做，等 Phase A2 跑通再优化。

---

## 9. 总结

**方向 A（CFG Rerank）**
- 数学干净：CFG 语义在序列级
- 代码模块化：generator / uncond_ll / rerank / pipeline 四个纯函数
- NFE 昂贵：8× baseline
- 预期收益最大（真正利用视觉信号在全序列上打分）

**方向 B（Defer-only CV）**
- 数学最保守：argmax 完全走 base
- 代码最简单：一个 veto 函数 + 一个 dispatcher
- NFE 便宜：1× baseline（同 v3.2）
- 是 v4 里**最能证伪 CV 假设的实验**——如果 defer-only 都超不过 baseline，"gain 信号有用"这个假设可以基本 close 掉

**推荐实施顺序：M0 → M1 → M2 → M3（方向 B 完整跑完再看要不要做 A）**

代码总量 ~800 LOC 新代码，配套 ~150 LOC 单元测试。v3.2 现有代码 0 修改（除了抽 cv_common 出去的 thin wrapper 化）。
