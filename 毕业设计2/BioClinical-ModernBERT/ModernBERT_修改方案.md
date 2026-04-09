# 基于 BioClinical ModernBERT 的修改方案（面向 CH-WLS 开题课题）

## 1. 修改目标

本项目不是重新训练一个全新的医学大模型，而是**在 ModernBERT / BioClinical ModernBERT 作为表征编码器的基础上，重构末端预测层**，使模型能够完成以下目标：

1. 将传统的 MLP 分类/回归头替换为**结构化最小二乘输出头**。
2. 同时建模：
   - **预测均值**
   - **Aleatoric 不确定性（异方差）**
   - **Epistemic 不确定性（Bayesian Last Layer）**
3. 在输出阶段加入：
   - **Ridge / WLS / 稳健重尾似然**
   - **β-NLL 或截断梯度 NLL**
   - **Conformal Prediction / CQR**
4. 最终形成适用于医学 EHR/临床文本预测任务的 **CH-WLS 闭环**。

---

## 2. 为什么基于 BioClinical ModernBERT 改，而不是直接改别的模型

BioClinical ModernBERT 是一个 **encoder-only** 的现代化 BERT 变体，适合做：

- 文本 / EHR 的固定长度或长上下文编码
- 特征提取后接结构化下游头
- 冻结 backbone，仅训练轻量输出头

它相比老的 ClinicalBERT 更适合本课题，因为：

- 支持长上下文
- 本质上仍然是标准 encoder，便于接你的统计输出头
- 与“先冻结编码器、后改最后一层”的研究路线高度一致

因此，**你真正要改的不是 ModernBERT 主干注意力结构，而是它的输出端组织方式**。

---

## 3. 总体原则：少改 backbone，多改 head

### 3.1 第一阶段原则

第一阶段不要直接改 ModernBERT 的预训练结构，不动：

- attention 机制
- token embedding
- RoPE / Flash Attention 相关实现
- 预训练 MLM 逻辑

第一阶段只做三件事：

1. 加载 BioClinical ModernBERT 作为 encoder
2. 提取 patient/document feature
3. 把默认下游预测头替换成你自己的结构化输出头

### 3.2 第二阶段原则

在 baseline 跑通后，再做非常轻量的参数高效微调，例如：

- 仅解冻最后 1–2 个 transformer block
- 或引入 LoRA / 其他 PEFT 方案

### 3.3 不推荐的做法

不建议一开始就：

- 重写 ModernBERT 预训练逻辑
- 深改 HuggingFace transformers 源码
- 直接在原仓库内部把训练框架整体推倒重来

因为你的创新点不在 backbone，而在**可微最小二乘输出层 + 不确定性校准闭环**。

---

## 4. 你的课题对应的核心改动

## 4.1 改动一：把默认分类/回归头替换为结构化输出头

原始 ModernBERT 的常见下游方式是：

```text
input text -> ModernBERT encoder -> pooled embedding -> linear / MLP head -> prediction
```

你需要改成：

```text
input text
-> BioClinical ModernBERT encoder
-> pooled embedding z
-> mean branch:      mu(z)
-> variance branch:  log_var(z)
-> structured LS layer (Ridge / WLS / robust)
-> prediction + uncertainty
```

这一步是全课题最核心的修改。

---

## 4.2 改动二：加入异方差建模分支

需要让模型不仅输出预测值，还输出与输入相关的噪声尺度。

建议增加：

- `mean_head(z)`：输出均值 `mu`
- `var_head(z)`：输出 `log_sigma2`

然后定义：

```math
w_i = 1 / (sigma_i^2 + eps)
```

并将其用于 WLS 目标：

```math
\hat{\beta} = \arg\min_\beta \sum_i w_i (y_i - x_i^T \beta)^2 + \lambda \|\beta\|_2^2
```

这样模型可以对高噪声患者自动降低权重。

---

## 4.3 改动三：把求解器写成可微层

你的输出层不能只写成一个普通线性层，而应该把 **Ridge / WLS 求解器**写成模块。

建议单独实现：

- `ridge_solver.py`
- `wls_solver.py`
- `linalg_utils.py`

在里面完成：

1. 组装正规化矩阵
2. 避免显式求逆
3. 使用 `torch.linalg.solve` / QR / Cholesky
4. 监控条件数

### 数值原则

不要直接写：

```python
beta = torch.inverse(XT_W_X) @ XT_W_y
```

应该优先写成：

```python
beta = torch.linalg.solve(A, b)
```

并准备两个稳定版本：

- QR 版本：更稳
- Cholesky 版本：更快

当特征维数远大于样本数时，再加入 Woodbury 加速版本。

---

## 4.4 改动四：替换异方差损失函数

直接使用普通高斯 NLL 容易出现均值分支与方差分支的梯度耦合问题。

因此建议增加：

- `gaussian_nll.py`
- `beta_nll.py`
- `student_t_nll.py`

优先级建议：

1. baseline：Gaussian NLL
2. 主实验：β-NLL / truncated-gradient NLL
3. 稳健实验：Student-t NLL

这样可以减少“方差学坏、均值被拖偏”的现象。

---

## 4.5 改动五：加入 Bayesian Last Layer（BLL）

在 CH-WLS 头训练完成后，需要在最后一层上做贝叶斯近似，而不是对整个 ModernBERT 做全贝叶斯化。

实现思路：

1. 冻结 encoder
2. 提取训练集特征矩阵 `X`
3. 对最后一层权重设高斯先验
4. 计算后验精度矩阵
5. 对测试样本输出 epistemic uncertainty

建议新建：

- `uncertainty/bll.py`
- `uncertainty/epistemic.py`
- `uncertainty/aleatoric.py`

BLL 的目标不是提升拟合精度，而是：

- 衡量模型对陌生样本的认知不确定性
- 在跨院区/跨时间漂移时发出预警

---

## 4.6 改动六：加入 Conformal Prediction / CQR

当 CH-WLS + BLL 跑通后，再在独立校准集上做 conformal calibration。

建议新增：

- `conformal/split_cp.py`
- `conformal/cqr.py`
- `conformal/metrics_interval.py`

实现目标：

1. 基于校准集计算非一致性分数
2. 构造 90% / 95% 预测区间
3. 输出：
   - PICP
   - MPIW
   - 区间覆盖率
   - 区间宽度

你的最终系统不只输出点预测，而是：

- 均值预测
- 总不确定性
- 可信区间
- 必要时拒识/报警

---

## 5. 推荐的代码改法：尽量采用“外包一层”的方式

## 5.1 最推荐结构

建议不要在仓库里把 `ModernBERTModel` 本体直接改坏，而是写一个包装模型：

```python
class CHWLSModernBERT(nn.Module):
    def __init__(self, encoder, ...):
        self.encoder = encoder
        self.mean_head = ...
        self.var_head = ...
        self.solver = ...
```

这样做的好处：

- 不破坏原模型加载方式
- 方便 baseline 与主方法切换
- 方便做 ablation
- 方便替换不同 encoder（BioClinical ModernBERT、Clinical ModernBERT、甚至别的 encoder）

---

## 5.2 你需要改 / 加的文件

### 在原仓库基础上建议新增

```text
models/
├── encoder_backbones.py
├── chwls_model.py
├── heads_baseline.py
├── heads_chwls.py
├── heads_quantile.py

losses/
├── gaussian_nll.py
├── beta_nll.py
├── student_t_nll.py

solvers/
├── ridge_solver.py
├── wls_solver.py
├── woodbury.py
├── linalg_utils.py

uncertainty/
├── bll.py
├── aleatoric.py
├── epistemic.py

conformal/
├── split_cp.py
├── cqr.py
├── metrics_interval.py

train/
├── extract_features.py
├── train_baseline.py
├── train_chwls.py
├── fit_bll.py
├── calibrate_cp.py

configs/
├── mimic_bioclinical_modernbert.yaml
├── eicu_eval.yaml
└── chexpert_transfer.yaml
```

### 需要轻改的已有文件

#### `main.py`
用途：
- 保留原来的任务入口
- 新增参数开关，例如：
  - `--head baseline_mlp`
  - `--head ridge`
  - `--head chwls`
  - `--head chwls_bll`
- 增加 `--freeze_encoder`、`--extract_only`、`--use_beta_nll` 等选项

#### `src/dataloader.py`
用途：
- 新增你自己的临床任务
- 支持 MIMIC-IV / eICU
- 支持 patient-level split
- 支持 train / calibration / test 三段划分

#### `scripts/`
用途：
- 增加一键运行脚本
- 区分 baseline、CH-WLS、BLL、CP 四阶段实验

---

## 6. 推荐训练流程

## 6.1 第一阶段：冻结 encoder，只提特征

目标：最稳地跑通主线。

步骤：

1. 加载 `thomas-sounack/BioClinical-ModernBERT-base` 或 `large`
2. 冻结全部 encoder 参数
3. 提取 pooled embedding
4. 保存到本地：
   - `X_train.pt`
   - `X_val.pt`
   - `X_calib.pt`
   - `X_test.pt`

这一阶段之后，大部分实验都可以在特征矩阵上完成，计算成本会下降很多。

---

## 6.2 第二阶段：训练 baseline

建议先做：

- Linear head
- MLP head
- Ridge regression head

目的：

- 拿到基础 RMSE / MAE / AUROC / AUPRC
- 验证特征是否有效
- 为 CH-WLS 做对照

---

## 6.3 第三阶段：训练 CH-WLS 头

训练内容：

- `mean_head`
- `var_head`
- Ridge/WLS solver
- β-NLL 或 Student-t NLL

需要记录：

- loss 曲线
- 预测方差分布
- 条件数变化
- 是否出现梯度爆炸/塌陷

---

## 6.4 第四阶段：拟合 BLL

在固定 CH-WLS 后：

1. 提取训练特征
2. 估计后验协方差
3. 推理时输出 epistemic
4. 在 eICU 外部测试集中观察 epistemic 是否抬升

---

## 6.5 第五阶段：做 conformal calibration

最后用校准集完成：

- split conformal
- CQR
- 区间评估

最终输出：

- 点预测值
- aleatoric uncertainty
- epistemic uncertainty
- total uncertainty
- confidence interval

---

## 7. 数学接口建议

## 7.1 encoder 输出

建议统一成：

```python
z = encoder_output.last_hidden_state[:, 0]   # [CLS] 方案
```

或使用 pooling：

```python
z = attention_masked_mean_pooling(last_hidden_state)
```

如果任务以长病历全文为主，建议比较：

- `[CLS]`
- mean pooling
- attention pooling

通常第一版先用 mean pooling 更稳。

---

## 7.2 CH-WLS 输出接口

建议统一 forward 返回：

```python
{
    "pred": pred,
    "mu": mu,
    "log_var": log_var,
    "aleatoric": sigma2,
    "weights": w,
    "beta": beta,
}
```

这样后面做 BLL 和 conformal 会很方便。

---

## 8. 不要改错方向

以下内容不应作为第一阶段重点：

### 8.1 不要重点改
- tokenizer 词表
- ModernBERT 预训练 mask 策略
- 长上下文注意力实现
- Flash Attention 内部 CUDA 细节
- 继续预训练配置

### 8.2 应该重点改
- pooled feature 的取法
- 下游头结构
- 求解器实现
- 损失函数稳定性
- 不确定性分解
- conformal 校准流程

---

## 9. 建议的最小可运行版本（MVP）

第一周到第三周，你最应该完成的是：

### MVP-1：特征抽取
- 载入 BioClinical ModernBERT
- 冻结 encoder
- 对 MIMIC-IV 文本生成 embedding

### MVP-2：baseline
- Ridge / MLP 跑通
- 输出 RMSE / AUC

### MVP-3：CH-WLS
- 加入 mean + variance 双分支
- 加入 WLS solver
- 用 β-NLL 替代普通高斯 NLL

只要这三步能跑通，你的课题主线就已经成立。

---

## 10. 一句话总结

**对 ModernBERT 的正确修改方向，不是“重写 ModernBERT”，而是“把它从通用 encoder 变成 CH-WLS 医学可信预测系统的前端表征器”。**

也就是说：

- **主干尽量少动**
- **输出层彻底重构**
- **训练流程分阶段解耦**
- **先表征、后统计推断、再保形校准**

这才是最符合你开题报告、也最容易在毕设周期内落地的修改路线。
