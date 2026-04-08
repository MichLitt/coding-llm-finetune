# CodeTune v3：代码生成 SFT + DPO Pipeline 计划

> **目标**：在 Qwen3.5-4B-Instruct 上完成 SFT → DPO 完整 post-training pipeline，针对 Coder Agent 的真实 failure patterns 构造 targeted 训练数据，用无污染基准量化效果。
>
> **硬件**：本地 RTX 5070 8GB（训练 + 评估全流程） + Colab A100（备用，数据量更大时）
>
> **核心原则**：
> 1. **Baseline 必须公平**：用 `-Instruct` 版对比 `-Instruct` 版，不拿 base model 充数
> 2. **评估集必须干净**：训练数据不碰 HumanEval / MBPP，留给评估
> 3. **少而精**：3-4K 高质量 SFT 数据，不以数量掩盖质量问题

---

## 一、模型选型

| 维度 | 说明 |
|------|------|
| **Base model** | `unsloth/Qwen3.5-4B-Instruct` |
| **Baseline 评估** | 同一模型 zero-shot（公平对比） |
| **训练方式** | SFT（bf16 LoRA） → DPO（execution-driven pairs） |
| **框架** | Unsloth + TRL |

**为什么选 4B-Instruct 而不是 9B：**

- RTX 5070 8GB 可直接跑 4-bit LoRA 训练（~5GB），不再强依赖 Colab
- A100 上 SFT 3-4K 数据约 30-40 分钟/次，可以多跑 ablation
- `-Instruct` 版作为起点，SFT 只需激活/修正特定能力，不需要从零建指令遵循

---

## 二、数据防污染设计（最高优先级）

```
训练数据来源          评估基准
─────────────         ─────────────────────
Magicoder             HumanEval   ← 完全干净
EvolCodeAlpaca        HumanEval+  ← 完全干净
Targeted (合成)       Targeted subset ← 自建，无重叠
─────────────
HumanEval    ✗ 禁止
MBPP         ✗ 禁止（用于 DPO pairs 生成，不进 SFT）
```

**执行规则**：

- `prepare_sft_data.py` 默认 sources 只有 `magicoder,evol`，HumanEval 从代码中移除
- DPO pairs 来源：MBPP（自带 test cases，与 HumanEval 评估集无重叠）
- 最终报告主指标：**HumanEval pass@1**（干净后才有意义）+ **targeted subset pass rate**

---

## 三、Phase 0：数据准备

### 3.1 SFT 数据目标

| 来源 | 数量 | 筛选策略 |
|------|------|---------|
| Magicoder-OSS-Instruct | ~2,000 | Python only，token 长度 20-512/10-1024，AST 可解析 |
| EvolCodeAlpaca | ~1,500 | Python only，同上过滤 |
| Targeted failure data | ~200 | 针对 8 个已知 failure cases 合成，每 case 25 变体 |
| **合计** | **~3,700** | MinHash 去重（threshold=0.85） |

> 3-4K 精选数据对 LoRA SFT 已足够。不以数量为目标。

### 3.2 Targeted data 生成

针对 Coder Agent v6 的 8 个真实 failure cases（HE/54, 55, 107, 108, 110, 128, 141, 147），每个 case 提取错误模式，用 MiniMax API 生成 25 个变体：

```python
# generate_targeted_data.py
# 每个 case 的 pattern 描述要精确到具体错误，而不是泛化的错误类型
# 例如 HE/54：不是"字符集合比较"，而是"sorted() 和 set() 的语义差异"
# 生成后必须：1) AST 可解析 2) 逻辑正确（人工抽查 10%）
```

### 3.3 污染检查脚本（新增）

数据准备最后一步必须运行：

```python
# scripts/check_contamination.py
# 用 Jaccard n-gram 相似度 + 已知 HumanEval 函数名匹配（无需额外模型）
# 输出：重叠条数、重叠比例，超过 0 条则 exit(1) 阻止训练
```

---

## 四、Phase 1：SFT 训练（本地 RTX 5070）

### 4.1 配置

```yaml
# configs/sft_config.yaml
model:
  name: "unsloth/Qwen3.5-4B-Instruct"
  load_in_4bit: true          # RTX 5070 8GB
  max_seq_length: 2048

lora:
  r: 32
  lora_alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  use_gradient_checkpointing: "unsloth"

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8   # effective batch = 16
  num_train_epochs: 3
  learning_rate: 2e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  bf16: true                        # RTX 5070 支持 bf16
  optim: "adamw_8bit"
```

### 4.2 Ablation 实验（利用本地快速迭代）

| 实验名 | 对比点 | 目的 |
|--------|--------|------|
| `sft_generic` | magicoder + evol only | baseline SFT |
| `sft_targeted` | + targeted 200条 | 验证 targeted data 价值（核心实验） |
| `sft_targeted_r16` | LoRA r=16 vs r=32 | rank 敏感性 |
| `sft_targeted_ep2` | 2 epoch vs 3 epoch | 过拟合检查 |

每组实验约 30-40 分钟（本地）。

### 4.3 进入 DPO 的门槛

- [ ] val loss 稳定下降，无明显过拟合
- [ ] HumanEval pass@1 > baseline（`sft_targeted` > `Qwen3.5-4B-Instruct` zero-shot）
- [ ] targeted subset 较 baseline 有提升
- [ ] 生成代码 AST 可解析率 > 95%

---

## 五、Phase 2：DPO 训练

### 5.1 Pairs 来源：MBPP（不用 HumanEval）

MBPP 有 374 道题（test split）自带 test cases，与 HumanEval 评估集无重叠。

```
DPO pairs 生成流程：
1. 用 sft_targeted 模型对每道 MBPP 题生成 8 个候选（temperature 0.2/0.7/1.0/1.2）
2. 每个候选跑 MBPP unit tests → pass / fail
3. 同一题有 pass 和 fail 时构造 pair：
   chosen = 通过测试的候选（最短）
   rejected = 未通过的候选（随机选一个）
4. 目标：500+ 有效 pairs（预计有效率 40-60%，需约 250 道题）
```

### 5.2 针对持续失败 cases 手工构造 pairs

HE/108（intersection）和 HE/128（tri）在 v2 所有版本中均失败，需手工构造：

```python
# 每个 case 构造 10-20 对 targeted pairs
# chosen：正确实现（手写或 GPT-4 生成后验证）
# rejected：模型的错误输出（从 sft_targeted 推理结果中取）
```

### 5.3 配置

```yaml
# configs/dpo_config.yaml
lora:
  r: 16           # DPO 用更小 rank，防过拟合
  lora_alpha: 32

training:
  num_train_epochs: 1    # 1 epoch 防止小数据集 reward hacking
  learning_rate: 5e-5
  beta: 0.1       # KL penalty，先用默认值
  loss_type: "sigmoid"
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### 5.4 DPO 退化诊断

| 症状 | 根因 | 对策 |
|------|------|------|
| pass@1 整体下降 | pairs 数量不足 / 质量差 | 扩大 pairs，排除噪声 pair |
| targeted 改善但整体退步 | alignment tax | 增大 beta（0.1 → 0.3） |
| 生成格式混乱 | SFT 与 DPO chat template 不一致 | 检查 template 对齐 |

---

## 六、Phase 3：评估

### 6.1 评估矩阵

| 模型 | HumanEval | HumanEval+ | Targeted (8题) |
|------|-----------|-----------|----------------|
| Qwen3.5-4B-Instruct（baseline） | ✓ | ✓ | ✓ |
| sft_generic | ✓ | ✓ | ✓ |
| sft_targeted | ✓ | ✓ | ✓ |
| dpo_v1 | ✓ | ✓ | ✓ |

> MBPP 用于生成 DPO pairs（训练集），不再作为评估基准。

HumanEval 现在是干净的（训练数据里没有它），pass@1 数字可信。
HumanEval+ 作为更严格的交叉验证。

### 6.2 主要结论维度

1. **整体能力提升**：baseline → sft_targeted HumanEval pass@1 delta
2. **Targeted data 价值**：sft_generic vs sft_targeted 在 targeted subset 上的对比
3. **DPO 修复能力**：HE/108、HE/128 是否被修复
4. **Alignment tax**：dpo_v1 整体 pass@1 是否 ≥ sft_targeted

---

## 七、时间线

| 阶段 | 任务 | 环境 | 预计时长 |
|------|------|------|---------|
| Day 1 | 清理 repo，修复 `prepare_sft_data.py`，写 `check_contamination.py` | 本地 | 半天 |
| Day 2 | 跑数据准备，验证污染检查通过，生成 targeted data | 本地 | 半天 |
| Day 3 | `sft_generic` + `sft_targeted` 训练，初步评估 | 本地 | 1天 |
| Day 4 | SFT ablation（rank 实验），选出最优 SFT | 本地 | 半天 |
| Day 5 | 生成 MBPP DPO pairs（目标 500+） | 本地 | 半天 |
| Day 6 | DPO v1 训练 + 评估 | 本地 | 半天 |
| Day 7 | DPO v2（若 v1 退步：扩 pairs / 调 beta）| 本地 | 半天 |
| Day 8 | HumanEval + HumanEval+ 全量评估，写最终报告 | 本地 | 1天 |

**总计约 8 天**（v2 原计划 14 天，大部分时间浪费在等待 Colab）

---

## 八、文件结构（清理后）

```
.
├── configs/
│   ├── sft_config.yaml           # 更新为 4B 配置
│   ├── dpo_config.yaml           # 更新
│   └── ablation_matrix.yaml      # 更新实验矩阵
├── scripts/
│   ├── prepare_sft_data.py       # 移除 HumanEval source
│   ├── check_contamination.py    # 新增：训练前污染检查
│   ├── generate_targeted_data.py # 复用
│   ├── generate_dpo_pairs.py     # 改用 MBPP 为 prompt 来源
│   ├── sft_train.py              # 更新模型名
│   ├── dpo_train.py              # 复用
│   ├── run_humaneval.py          # 复用（现在结果可信）
│   └── run_targeted_eval.py      # 复用
├── report/
│   ├── CodeTune_v3_Project_Plan.md   # 本文件
│   └── archive/                       # v2 的旧报告，留作参考
├── notebooks/                         # Colab 备用
└── README.md                          # 更新结果表格
```

---

## 九、v2 → v3 关键变化对照

| 维度 | v2（问题版） | v3（修正版） |
|------|-------------|-------------|
| Base model | Qwen3.5-9B（base，非 instruct） | **Qwen3.5-4B-Instruct** |
| SFT 数据 | 86K（含 HumanEval 答案） | **3-4K（不含任何 benchmark）** |
| Baseline | Base model + chat template（偏低） | **Instruct 版 zero-shot（公平）** |
| DPO pairs 来源 | HumanEval（评估集泄露）| **MBPP（与评估集无重叠）** |
| DPO pairs 数量 | ~147（不足） | **500+** |
| 主要评估指标 | HumanEval（污染）| **HumanEval（现在干净）+ HumanEval+** |
| 训练环境 | 强依赖 Colab | **本地 RTX 5070 为主** |
| 污染检查 | 无 | **`check_contamination.py`，CI 级别强制** |
