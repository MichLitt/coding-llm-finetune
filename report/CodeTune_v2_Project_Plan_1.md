# CodeTune v2: Code Generation SFT + DPO Fine-Tuning Project Plan

> **目标**：在 Qwen2.5-Coder-7B 上完成 SFT + DPO 完整 post-training pipeline，提升代码生成能力，用 HumanEval/MBPP/LiveCodeBench 量化验证效果，并与 Coder Agent 项目形成闭环。
>
> **硬件**：本地 RTX 5070 Laptop 8GB（数据准备 + 4bit 推理验证） + Google Colab Pro A100 80GB（训练）
>
> **核心原则**：少而精，每一步都做扎实。宁可 scope 小但结果 solid，不要摊太大但每项都浅。

---

## 一、项目概览

| 维度 | 内容 |
|------|------|
| **Base Model（首选）** | Qwen2.5-Coder-7B-Instruct（专用 code 模型，社区成熟，LoRA 经验丰富） |
| **Base Model（备选）** | Qwen3.5-9B（更新更强，但兼容性风险略高；遇到问题退回首选） |
| **训练方式** | Phase 1: SFT（bf16 LoRA）→ Phase 2: DPO（简洁偏好信号） |
| **训练框架** | Unsloth + TRL（SFTTrainer / DPOTrainer） |
| **评估基准** | HumanEval, HumanEval+, MBPP, LiveCodeBench, 自定义 targeted subset |
| **预估周期** | 14 天 |
| **预估成本** | Colab Pro 学生版（已有），基本零额外成本 |

### 为什么选 Qwen2.5-Coder-7B 而不是 Qwen3.5-9B

| 维度 | Qwen2.5-Coder-7B | Qwen3.5-9B |
|------|-------------------|-------------|
| **代码专精度** | 专门为代码优化，base HumanEval 更高 | 通用模型，代码不是主要优化方向 |
| **社区成熟度** | 大量 fine-tuning 教程和经验 | 发布仅一周，踩坑经验少 |
| **架构稳定性** | 标准 Transformer | Gated DeltaNet hybrid attention，工具兼容性未完全验证 |
| **fine-tuning 增量空间** | base 已经强，targeted data 可以进一步提升特定弱项 | base 在代码上未必最优，提升来源不够清晰 |
| **显存需求** | bf16 LoRA ~18GB，A100 轻松 | bf16 LoRA ~22GB，A100 可以但余量更小 |
| **面试叙事** | "我选了专用 code 模型，因为..." 体现技术判断力 | "我选了最新模型" 但面试官可能追问为什么不用专用模型 |

> **策略**：默认用 Qwen2.5-Coder-7B 推进。如果 Day 4 pipeline 验证阶段一切顺利且有余力，可以同时跑一组 Qwen3.5-9B 作为额外对比实验。

---

## 二、项目结构

```
codetune/
├── data/
│   ├── raw/                        # 原始数据集下载
│   ├── sft/                        # SFT 训练数据
│   │   ├── train.jsonl             # ~3,000 条
│   │   └── val.jsonl               # ~300 条
│   ├── dpo/                        # DPO preference pairs
│   │   ├── train.jsonl             # ~1,500 对
│   │   └── val.jsonl               # ~150 对
│   └── scripts/
│       ├── prepare_sft_data.py     # 数据清洗主脚本
│       ├── generate_dpo_pairs.py   # preference pair 构造
│       └── data_stats.py           # 数据集统计分析
├── training/
│   ├── sft_train.py
│   ├── dpo_train.py
│   └── configs/
│       ├── sft_config.yaml
│       └── dpo_config.yaml
├── evaluation/
│   ├── run_humaneval.py
│   ├── run_mbpp.py
│   ├── run_targeted_eval.py        # 针对 failure pattern 的评估
│   └── analysis.py                 # 结果分析 & 可视化
├── notebooks/
│   ├── 01_data_preparation.ipynb   # 本地
│   ├── 02_sft_training.ipynb       # Colab
│   ├── 03_dpo_training.ipynb       # Colab
│   └── 04_evaluation.ipynb         # 本地
├── results/
│   ├── figures/                    # 可视化图表
│   └── tables/                     # 结果表格
├── README.md
└── requirements.txt
```

---

## 三、Phase 0: 数据准备（Day 1-3）

### 3.1 核心原则：少而精

你的课程资料和 GPT 的 review 都指出同一件事：SFT 阶段，**数据质量远比数量重要**。1K-10K 条精选数据就已经可以很好地激活能力，关键在于质量和覆盖面。

因此我们的数据目标是 **3,000-4,000 条高质量 Python code instruction**，分两个来源：

| 数据来源 | 条数 | 用途 |
|----------|------|------|
| **通用 code instruction**（从开源数据集筛选） | ~2,500 | 覆盖常见编程任务类型 |
| **Targeted failure data**（从 Coder Agent 提炼） | ~500-1,000 | 针对性修复特定错误模式 |

### 3.2 通用数据：来源与筛选

从以下开源数据集中筛选：

| 数据集 | 特点 | 筛选策略 |
|--------|------|----------|
| **Code-Alpaca** | 简单 instruction，覆盖面广 | 取 Python 子集，过滤掉过于简单的（< 3 行 output） |
| **Magicoder-OSS-Instruct** | 基于真实开源代码生成，质量较高 | 优先选取，这是最高质量的来源 |
| **Evol-Instruct-Code** | 多难度梯度 | 只取 medium/hard 难度 |

### 3.3 数据清洗 Pipeline

```python
# prepare_sft_data.py

import ast
import json
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

def clean_sample(sample):
    instruction = sample["instruction"]
    output = sample["output"]

    # 1. 语言过滤：只保留 Python
    if not is_python(output):
        return None

    # 2. 长度过滤
    inst_tokens = len(tokenizer.encode(instruction))
    out_tokens = len(tokenizer.encode(output))
    if not (10 <= inst_tokens <= 400):
        return None
    if not (20 <= out_tokens <= 1500):
        return None

    # 3. 语法验证：必须能 parse
    code = extract_code_block(output)
    try:
        ast.parse(code)
    except SyntaxError:
        return None

    # 4. 基础可执行性：尝试 import 检查（非必须通过，只排除明显错误）
    if has_obvious_import_error(code):
        return None

    return {
        "instruction": instruction.strip(),
        "output": output.strip(),
        "source": sample.get("source", "unknown"),
        "inst_tokens": inst_tokens,
        "out_tokens": out_tokens,
    }

def deduplicate(samples, threshold=0.85):
    """基于 instruction 的 MinHash 去重"""
    # 使用 datasketch 库做近似去重
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_samples = []
    for i, s in enumerate(samples):
        mh = compute_minhash(s["instruction"])
        if not lsh.query(mh):
            lsh.insert(str(i), mh)
            unique_samples.append(s)
    return unique_samples
```

### 3.4 Targeted Failure Data（核心差异化亮点）

这是整个项目最有讲故事价值的部分。从你的 Coder Agent 项目中提取 failure patterns，针对性构造训练数据：

**Step 1: 从 Coder Agent 中提取 failure taxonomy**

```python
# 假设你在 Coder Agent 项目中已经分类了错误类型
# 这里列出最常见的 5 类 failure patterns
failure_patterns = {
    "off_by_one": {
        "description": "循环边界、索引、range 的 ±1 错误",
        "example_task": "返回数组中第 k 大的元素",
        "target_count": 150,
    },
    "edge_case_missing": {
        "description": "空输入、单元素、负数、零值等边界条件未处理",
        "example_task": "实现二分查找，要求处理空数组",
        "target_count": 150,
    },
    "type_handling": {
        "description": "类型转换错误、None 检查遗漏、混合类型处理",
        "example_task": "合并两个可能包含 None 的列表",
        "target_count": 100,
    },
    "recursion_base_case": {
        "description": "递归终止条件遗漏或错误",
        "example_task": "用递归实现树的序列化",
        "target_count": 100,
    },
    "algorithm_correctness": {
        "description": "算法逻辑本身有误，如贪心不适用的场景用了贪心",
        "example_task": "最小硬币找零问题",
        "target_count": 100,
    },
}
```

**Step 2: 用 GPT-4o-mini 批量生成，人工 review**

```python
# 为每个 failure pattern 生成 instruction-output pairs
# 关键：output 必须正确处理该 pattern 对应的错误

def generate_targeted_data(pattern_name, pattern_info, n=150):
    system_prompt = f"""你是一个编程教学专家。请生成 Python 编程题目和标准答案。

要求：
1. 题目必须涉及 "{pattern_info['description']}" 这类常见错误
2. 标准答案必须正确处理这个边界情况
3. 在答案的注释中明确标注关键的边界处理逻辑
4. 难度范围：LeetCode Easy 到 Medium
5. 只使用 Python 标准库

返回格式：
{{"instruction": "题目描述", "output": "完整 Python 代码"}}"""

    results = []
    for i in range(n):
        response = call_gpt4o_mini(system_prompt)
        parsed = json.loads(response)
        # 验证代码可执行
        if validate_code(parsed["output"]):
            parsed["pattern"] = pattern_name
            results.append(parsed)

    return results
```

**Step 3: 人工 review checklist**

对每条生成的数据，检查以下几点：
- 代码是否真的正确处理了目标 failure pattern
- 代码风格是否一致（docstring、type hints、命名规范）
- 是否和 HumanEval/MBPP 的题目有重叠（**必须排除，防止数据污染**）
- 难度是否合理（不要全是 trivial 的）

### 3.5 数据格式

统一使用 Qwen2.5-Coder 的 chat template：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert Python programmer. Write clean, efficient, and correct code. Always handle edge cases and include brief comments for complex logic."
    },
    {
      "role": "user",
      "content": "Write a function that finds the kth largest element in an unsorted array. Handle edge cases where k is out of bounds or the array is empty."
    },
    {
      "role": "assistant",
      "content": "def find_kth_largest(nums: list[int], k: int) -> int | None:\n    \"\"\"Find the kth largest element in an unsorted array.\n    \n    Args:\n        nums: List of integers\n        k: The kth position (1-indexed)\n    \n    Returns:\n        The kth largest element, or None if k is invalid\n    \"\"\"\n    # Edge case: empty array or invalid k\n    if not nums or k < 1 or k > len(nums):\n        return None\n    \n    # Use quickselect for O(n) average case\n    import random\n    \n    def quickselect(left, right, target):\n        pivot_idx = random.randint(left, right)\n        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]\n        pivot = nums[right]\n        \n        store = left\n        for i in range(left, right):\n            if nums[i] > pivot:  # Descending for kth largest\n                nums[store], nums[i] = nums[i], nums[store]\n                store += 1\n        \n        nums[store], nums[right] = nums[right], nums[store]\n        \n        if store == target:\n            return nums[store]\n        elif store < target:\n            return quickselect(store + 1, right, target)\n        else:\n            return quickselect(left, store - 1, target)\n    \n    return quickselect(0, len(nums) - 1, k - 1)  # k-1: convert to 0-indexed"
    }
  ]
}
```

### 3.6 数据统计与质量验证

在数据准备完成后，输出一份数据报告：

```python
# data_stats.py
def generate_data_report(dataset):
    report = {
        "total_samples": len(dataset),
        "source_distribution": Counter(d["source"] for d in dataset),
        "token_stats": {
            "instruction": describe(inst_tokens),  # mean, median, std, min, max
            "output": describe(out_tokens),
        },
        "task_type_distribution": classify_task_types(dataset),
        "targeted_pattern_coverage": {
            pattern: count for pattern, count in pattern_counts.items()
        },
        "syntax_valid_rate": syntax_valid / total,
        "dedup_rate": original_count / deduped_count,
    }
    return report
```

---

## 四、Phase 1: SFT 训练（Day 4-7）

### 4.1 Day 4: Pipeline 验证（本地）

在正式用 A100 训练之前，先在本地用小模型验证整个 pipeline 跑通：

```python
# 用 Qwen2.5-Coder-0.5B 在本地验证
# 目的不是训练效果，而是确认：
# 1. 数据加载正确
# 2. chat template 格式正确
# 3. tokenizer 工作正常
# 4. 训练循环无报错
# 5. checkpoint 保存和加载正常
# 6. 推理 pipeline 正常

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,  # 本地 8GB，用 4bit
)

# 用 100 条数据跑 10 steps，确认没有报错
# 检查 loss 是否在下降
# 检查生成的代码格式是否正确
```

**这一步非常重要**。很多人直接上 Colab 开始训，结果数据格式有问题、template 对不上，浪费了几个小时的 GPU 时间。

### 4.2 Day 5-6: 正式 SFT 训练（Colab A100）

```python
# sft_train.py — 在 Colab 上运行

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# ============ 模型加载 ============
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=False,       # A100 80GB，用 bf16 LoRA，质量更好
    dtype=torch.bfloat16,
)

# ============ LoRA 配置 ============
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,            # alpha = 2 * r
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=2048,
)

# ============ 数据加载 ============
dataset = load_dataset("json", data_files={
    "train": "data/sft/train.jsonl",
    "val": "data/sft/val.jsonl",
})

def format_chat(example):
    """将 messages 格式转为模型需要的 text 格式"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat)

# ============ 训练 ============
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    args=SFTConfig(
        # Batch
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,     # effective batch size = 16
        # Schedule
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Precision
        bf16=True,
        optim="adamw_8bit",
        # Logging & Saving
        output_dir="outputs/sft",
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        save_total_limit=3,               # 只保留最近 3 个 checkpoint
        # Data
        max_seq_length=2048,
        dataset_text_field="text",
        # Reproducibility
        seed=42,
    ),
)

# 训练前先跑一次 eval 作为 baseline
trainer.evaluate()

# 开始训练
trainer.train()

# 保存最终模型
model.save_pretrained("outputs/sft/final")
tokenizer.save_pretrained("outputs/sft/final")
```

### 4.3 Day 7: SFT Ablation（Colab A100）

只做 **2-3 组最有价值的消融实验**，不贪多：

| 实验 | 对比 | 目的 |
|------|------|------|
| **数据组成** | generic only vs generic + targeted | 验证 targeted data 的价值（最重要的实验） |
| **LoRA Rank** | r=16 vs r=32 | 确认 rank 选择是否合理 |
| **训练轮数** | 1 epoch vs 3 epochs vs 5 epochs | 找到过拟合拐点 |

**为什么"generic vs generic+targeted"是最重要的实验**：

这个实验直接验证了你的核心叙事——从 Coder Agent 发现问题 → 构造 targeted data → 训练改善。如果 generic+targeted 明显好于 generic only（尤其在 targeted failure subset 上），这就是整个项目最有力的证据。

```python
# ablation 实验管理
ablation_configs = [
    {"name": "generic_only",    "data": "data/sft/train_generic.jsonl", "r": 32, "epochs": 3},
    {"name": "generic_targeted","data": "data/sft/train.jsonl",         "r": 32, "epochs": 3},
    {"name": "rank_16",         "data": "data/sft/train.jsonl",         "r": 16, "epochs": 3},
    {"name": "epoch_1",         "data": "data/sft/train.jsonl",         "r": 32, "epochs": 1},
    {"name": "epoch_5",         "data": "data/sft/train.jsonl",         "r": 32, "epochs": 5},
]

# 每个实验跑完立即在 val set 上评估，记录 loss
# 最终选 val loss 最低的配置作为 best SFT model 进入 DPO 阶段
```

### 4.4 SFT 阶段的检查点

在进入 DPO 之前，确认 SFT 模型满足以下条件：

- [ ] val loss 稳定下降且未明显过拟合
- [ ] 生成的代码格式正确（能 parse、风格一致）
- [ ] 在 HumanEval 上 pass@1 ≥ base model（至少不退化）
- [ ] 在 targeted failure subset 上有初步改善

**如果 SFT 模型反而比 base model 差**，说明数据有问题，不要急着上 DPO，先回去排查数据质量。

---

## 五、Phase 2: DPO 训练（Day 8-10）

### 5.1 核心原则：最干净的偏好信号

DPO 的质量完全取决于 preference pair 的质量。第一版只用**最简单、最客观的偏好定义**：

```
偏好优先级（严格顺序）：
1. 通过所有 unit tests  >  未通过
2. 通过更多 tests      >  通过更少 tests
3. 都通过时，代码更短    >  代码更长（弱信号，作为 tiebreaker）
```

**不做**复杂的多维度加权评分。不做可读性评分、不做效率评分、不做 robustness 评分。这些东西定义不稳定，会污染偏好信号。

### 5.2 Day 8: Preference Pair 构造

```python
# generate_dpo_pairs.py

import json
import subprocess
import tempfile
from pathlib import Path

def generate_candidates(model, tokenizer, prompt, n=8, temperature=0.8):
    """用 SFT 模型对同一个 prompt 生成多个候选"""
    candidates = []
    for _ in range(n):
        output = model.generate(
            prompt,
            temperature=temperature,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
        )
        candidates.append(output)
    return candidates

def run_tests(code, test_cases):
    """在沙箱中执行代码并运行测试"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code + "\n\n" + test_cases)
        f.flush()
        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                text=True,
                timeout=10,  # 10 秒超时
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "stdout": "", "stderr": "timeout"}

def construct_preference_pairs(prompts_with_tests, sft_model, tokenizer):
    """构造 DPO preference pairs"""
    pairs = []
    stats = {"total_prompts": 0, "valid_pairs": 0, "all_pass": 0, "all_fail": 0}

    for item in prompts_with_tests:
        prompt = item["prompt"]
        test_cases = item["test_cases"]
        stats["total_prompts"] += 1

        # 生成 8 个候选
        candidates = generate_candidates(sft_model, tokenizer, prompt, n=8)

        # 对每个候选运行测试
        results = []
        for code in candidates:
            test_result = run_tests(code, test_cases)
            results.append({
                "code": code,
                "passed": test_result["passed"],
                "code_length": len(code),
            })

        # 分组
        passed = [r for r in results if r["passed"]]
        failed = [r for r in results if not r["passed"]]

        # 只有同时存在 pass 和 fail 时才构造 pair
        if passed and failed:
            # chosen: 通过测试 + 最短代码（作为弱 tiebreaker）
            chosen = min(passed, key=lambda x: x["code_length"])
            # rejected: 随机选一个未通过的
            rejected = failed[0]
            pairs.append({
                "prompt": prompt,
                "chosen": chosen["code"],
                "rejected": rejected["code"],
            })
            stats["valid_pairs"] += 1
        elif passed and not failed:
            stats["all_pass"] += 1
        else:
            stats["all_fail"] += 1

    print(f"Stats: {stats}")
    print(f"Valid pair rate: {stats['valid_pairs']}/{stats['total_prompts']}")
    return pairs
```

### 5.3 DPO Prompt 来源

| 来源 | 特点 | 预估有效 pair 数 |
|------|------|-----------------|
| **HumanEval prompts**（164 题） | 自带 test cases，直接可用 | ~80-120 pairs |
| **MBPP prompts**（500 题） | 自带 test cases | ~200-350 pairs |
| **自构造 targeted prompts** | 从 Coder Agent failure cases 中提取，附带自写 test cases | ~200-400 pairs |
| **EvalPlus 扩展测试** | HumanEval+ 的额外 test cases，更严格 | 额外筛选信号 |

**目标总计：~1,000-1,500 对有效 preference pairs**

> 注意：不是所有 prompt 都能产出有效 pair。如果 SFT 模型太强（全部通过）或太弱（全部失败），就没有有效的偏好信号。预计有效率在 40-60% 左右。

### 5.4 关于数据污染

**极其重要**：如果你用 HumanEval/MBPP 的 prompt 来构造 DPO 数据，那这些 prompt 就不能再作为评估集使用。有两种处理方式：

**方案 A（推荐）**：用 HumanEval/MBPP 的 prompt 构造 DPO 数据，但评估时用 **HumanEval+（扩展测试集）** 和 **LiveCodeBench（完全独立的 benchmark）** 来报告最终结果。

**方案 B**：将 HumanEval/MBPP 按 70/30 切分，70% 用于 DPO 数据构造，30% 保留为 held-out 评估集。

> 面试时如果被问到数据污染问题，你能清楚地解释你的处理方式，这本身就是加分项。

### 5.5 Day 9-10: DPO 训练

```python
# dpo_train.py — 在 Colab 上运行

from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import torch

# ============ 加载 SFT 模型（作为 policy 起点和 reference） ============
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/sft/final",      # 加载 SFT 后的模型
    max_seq_length=2048,
    dtype=torch.bfloat16,
)

# 添加新的 LoRA adapter 用于 DPO
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                                 # DPO 用更小的 rank，防止过拟合
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)

# ============ 加载 DPO 数据 ============
dpo_dataset = load_dataset("json", data_files={
    "train": "data/dpo/train.jsonl",
    "val": "data/dpo/val.jsonl",
})

def format_dpo(example):
    """格式化为 DPOTrainer 需要的格式"""
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dpo_dataset = dpo_dataset.map(format_dpo)

# ============ DPO 训练 ============
trainer = DPOTrainer(
    model=model,
    ref_model=None,           # Unsloth 会自动处理 reference model
    args=DPOConfig(
        # Batch
        per_device_train_batch_size=1,    # DPO 需要同时处理 chosen+rejected，显存占用更大
        gradient_accumulation_steps=16,   # effective batch size = 16
        # Schedule
        num_train_epochs=1,               # DPO 通常只需 1 epoch
        learning_rate=5e-5,               # 比 SFT 更小的学习率
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # DPO specific
        beta=0.1,                         # KL penalty 系数
        loss_type="sigmoid",              # 标准 DPO loss
        # Precision
        bf16=True,
        optim="adamw_8bit",
        # Logging & Saving
        output_dir="outputs/dpo",
        logging_steps=5,
        eval_steps=20,
        save_steps=50,
        save_total_limit=2,
        # Reproducibility
        seed=42,
    ),
    train_dataset=dpo_dataset["train"],
    eval_dataset=dpo_dataset["val"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("outputs/dpo/final")
tokenizer.save_pretrained("outputs/dpo/final")
```

### 5.6 DPO 阶段不做大范围 ablation

和 SFT 不同，DPO 阶段**只调一个关键参数**：beta。

| beta 值 | 含义 | 预期行为 |
|---------|------|----------|
| 0.05 | 激进优化偏好 | 可能过度偏离 reference，导致退化 |
| **0.1** | 标准值 | 大多数场景的合理起点 |
| 0.3 | 保守 | 改善幅度小但更安全 |

如果 beta=0.1 的结果已经有明显改善，就不需要继续调了。如果结果不理想（退化或无变化），再尝试 0.05 和 0.3。

### 5.7 DPO 阶段的检查点

- [ ] DPO loss 在下降
- [ ] chosen 和 rejected 的 reward margin 在增大
- [ ] 生成质量没有明显退化（格式正常、不出现重复/乱码）
- [ ] 在 val set 上 chosen win rate > 50%（理想 > 70%）

**如果 DPO 导致模型退化**（比 SFT-only 差），可能的原因和对策：

| 症状 | 可能原因 | 对策 |
|------|----------|------|
| 生成重复/乱码 | beta 太小，过度偏离 | 增大 beta 到 0.3 |
| pass@1 下降 | preference pair 质量差 | 回去检查 pair，排除噪声 |
| 格式混乱 | DPO 数据和 SFT 格式不一致 | 检查 chat template 是否对齐 |
| 完全没改善 | 数据太少或偏好信号太弱 | 增加数据量或只保留最 clean 的 pair |

---

## 六、Phase 3: 评估（Day 11-13）

### 6.1 评估框架

评估 **4 个模型**的表现：

| 模型 | 描述 |
|------|------|
| **Model A** | Qwen2.5-Coder-7B-Instruct（原始 base model） |
| **Model B** | + SFT（generic data only） |
| **Model C** | + SFT（generic + targeted data） |
| **Model D** | + SFT + DPO（完整 pipeline） |

### 6.2 评估基准

| 基准 | 指标 | 工具 | 是否有数据污染风险 |
|------|------|------|-------------------|
| **HumanEval** | pass@1, pass@5 | bigcode-evaluation-harness | 如果用了 DPO 构造数据则有风险 |
| **HumanEval+** | pass@1 | EvalPlus | 低风险（扩展测试集） |
| **MBPP** | pass@1 | bigcode-evaluation-harness | 同 HumanEval |
| **LiveCodeBench** | pass@1 | 官方 repo | **无风险（推荐作为主要报告指标）** |
| **Targeted Failure Subset** | pass rate per category | 自建 | 无风险 |

### 6.3 评估脚本

```python
# run_humaneval.py — 在本地用 4bit 推理运行

from unsloth import FastLanguageModel
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
import torch

def evaluate_model(model_path, output_file, n_samples=1):
    """评估单个模型在 HumanEval 上的表现"""
    # 加载模型（4bit 量化，本地 8GB 可跑）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    problems = read_problems()
    samples = []

    for task_id, problem in problems.items():
        prompt = problem["prompt"]
        for _ in range(n_samples):
            completion = generate_completion(model, tokenizer, prompt)
            samples.append({
                "task_id": task_id,
                "completion": completion,
            })

    write_jsonl(output_file, samples)
    results = evaluate_functional_correctness(output_file)
    return results

# 评估所有模型
models = {
    "base":         "Qwen/Qwen2.5-Coder-7B-Instruct",
    "sft_generic":  "outputs/sft_generic/final",
    "sft_targeted": "outputs/sft/final",
    "sft_dpo":      "outputs/dpo/final",
}

all_results = {}
for name, path in models.items():
    all_results[name] = evaluate_model(path, f"results/{name}_humaneval.jsonl")
    print(f"{name}: {all_results[name]}")
```

### 6.4 Targeted Failure Evaluation（核心差异化）

这是最能体现你项目价值的评估维度。为每个 failure pattern 准备 20-30 道专门的测试题：

```python
# run_targeted_eval.py

targeted_test_suite = {
    "off_by_one": [
        {
            "prompt": "Write a function that returns elements from index i to j (inclusive) of a list.",
            "test": "assert get_range([1,2,3,4,5], 1, 3) == [2,3,4]",
        },
        # ... 20-30 道题
    ],
    "edge_case_missing": [
        {
            "prompt": "Write a binary search function that works on empty arrays and returns -1.",
            "test": "assert binary_search([], 5) == -1\nassert binary_search([1], 1) == 0",
        },
        # ... 20-30 道题
    ],
    # ... 其他 pattern
}

def evaluate_targeted(model_path, test_suite):
    """按 failure category 评估模型表现"""
    results = {}
    for pattern, tests in test_suite.items():
        passed = 0
        for test in tests:
            code = generate_completion(model, tokenizer, test["prompt"])
            if run_test(code, test["test"]):
                passed += 1
        results[pattern] = {
            "pass_rate": passed / len(tests),
            "passed": passed,
            "total": len(tests),
        }
    return results

# 对比结果
for model_name, model_path in models.items():
    results = evaluate_targeted(model_path, targeted_test_suite)
    print(f"\n=== {model_name} ===")
    for pattern, r in results.items():
        print(f"  {pattern}: {r['pass_rate']:.1%} ({r['passed']}/{r['total']})")
```

### 6.5 Failure Analysis

不只是报数字，还要分析**失败案例的分布变化**：

```python
# analysis.py

def failure_analysis(base_results, sft_results, dpo_results):
    """分析 SFT 和 DPO 分别修复了哪些类型的错误"""

    # 分类每个失败案例的错误类型
    error_categories = [
        "syntax_error",         # 语法错误
        "runtime_error",        # 运行时错误（NameError, TypeError 等）
        "wrong_output",         # 输出不正确
        "timeout",              # 超时（可能是死循环）
        "edge_case_miss",       # 边界条件未处理
    ]

    for category in error_categories:
        base_count = count_errors(base_results, category)
        sft_count = count_errors(sft_results, category)
        dpo_count = count_errors(dpo_results, category)
        print(f"{category}:")
        print(f"  Base → SFT: {base_count} → {sft_count} ({improvement(base_count, sft_count)})")
        print(f"  SFT  → DPO: {sft_count} → {dpo_count} ({improvement(sft_count, dpo_count)})")
```

### 6.6 结果呈现

准备以下图表（用 matplotlib 生成，放入 README 和面试 slides）：

1. **Benchmark 对比柱状图**：4 个模型在 HumanEval / MBPP / LiveCodeBench 上的 pass@1
2. **Targeted failure heatmap**：每个 failure pattern × 每个模型的 pass rate 热力图
3. **Training loss curve**：SFT 和 DPO 的 loss 收敛曲线
4. **Error distribution 变化图**：Sankey 图或堆叠柱状图，展示错误类型从 base → SFT → DPO 的变化
5. **Ablation 对比表**：generic vs targeted、不同 rank 的效果

---

## 七、Phase 4: 工程化与文档（Day 14）

### 7.1 README 结构

```markdown
# CodeTune: Targeted Post-Training for Code Generation

## TL;DR
Fine-tuned Qwen2.5-Coder-7B with SFT + DPO, using failure-pattern-driven
training data derived from a coding agent's error analysis. Achieved X% pass@1
on HumanEval (↑Y%) and Z% on LiveCodeBench (↑W%), with particularly strong
improvements on edge-case handling (+V%).

## Key Results
[结果表格]

## Method
[Pipeline 架构图：Data Curation → SFT → DPO → Evaluation]

## Key Findings
1. Targeted training data 针对性修复了 X 类错误
2. DPO 在 Y 方面进一步提升了 Z%
3. [其他发现]

## Ablation Studies
[generic vs targeted 对比]

## Reproduction
[完整步骤]

## Limitations
[诚实地列出局限]
```

### 7.2 模型发布

- 将 LoRA adapter 上传到 Hugging Face Hub
- 写 model card：base model、训练数据描述、超参、benchmark 结果
- 提供推理示例代码

### 7.3 与 Coder Agent 的联动验证

如果时间允许（Day 14 或之后），做一个简单但有力的联动实验：

```python
# 在 Coder Agent 中替换 base model 为 fine-tuned model
# 对比 agent 在同一组任务上的表现变化

agent_tasks = load_coder_agent_test_suite()
base_agent_results = run_agent(base_model, agent_tasks)
ft_agent_results = run_agent(finetuned_model, agent_tasks)

print(f"Base model agent pass rate: {base_agent_results['pass_rate']:.1%}")
print(f"Fine-tuned agent pass rate: {ft_agent_results['pass_rate']:.1%}")
```

> 即使只是一个小规模的对比实验，也能完成"discover → train → deploy → verify"的完整闭环叙事。

---

## 八、时间线

| 日期 | 阶段 | 具体任务 | 环境 | 产出 |
|------|------|----------|------|------|
| **Day 1** | 数据准备 | 下载数据集，搭建清洗 pipeline | 本地 | 清洗脚本 |
| **Day 2** | 数据准备 | 通用数据清洗、过滤、去重 | 本地 | ~2,500 条通用数据 |
| **Day 3** | 数据准备 | 构造 targeted data + 数据报告 | 本地 | ~500-1,000 条 targeted 数据 |
| **Day 4** | SFT 验证 | 用 0.5B 模型本地验证 pipeline | 本地 | 确认 pipeline 无 bug |
| **Day 5** | SFT 训练 | 7B 模型正式 SFT 训练 | Colab A100 | SFT checkpoint |
| **Day 6** | SFT 训练 | 继续训练 + 初步评估 | Colab A100 | SFT 最终模型 |
| **Day 7** | SFT Ablation | generic vs targeted + rank 对比 | Colab A100 | ablation 结果 |
| **Day 8** | DPO 数据 | 用 SFT 模型生成候选，构造 pairs | Colab A100 | ~1,000-1,500 pairs |
| **Day 9** | DPO 训练 | DPO 训练 | Colab A100 | DPO checkpoint |
| **Day 10** | DPO 验证 | DPO 评估 + beta 调整（如需要） | Colab A100 | DPO 最终模型 |
| **Day 11** | 评估 | HumanEval, MBPP, LiveCodeBench | 本地 4bit | benchmark 结果 |
| **Day 12** | 评估 | Targeted eval + failure analysis | 本地 | 分类错误分析 |
| **Day 13** | 分析 | 可视化、结果整理、图表生成 | 本地 | figures & tables |
| **Day 14** | 工程化 | README、model card、HF Hub 上传 | 本地 | 完整 repo |

---

## 九、与 Coder Agent 项目的联动叙事

```
Coder Agent（inference-time）              CodeTune（training-time）
        │                                          │
        │  1. 运行 agent，发现反复出错的 patterns     │
        │     (off-by-one, edge cases, ...)         │
        ├──────────────────────────────────────────►│
        │                                          │  2. 针对 failure patterns
        │                                          │     构造 targeted SFT 数据
        │                                          │
        │                                          │  3. SFT + DPO 训练
        │                                          │
        │  4. 用 fine-tuned 模型替换 base model      │
        │◄──────────────────────────────────────────┤
        │                                          │
        │  5. 验证 agent 在同类任务上表现改善          │
        ├──────────────────────────────────────────►│  6. 量化改善幅度
        │                                          │
        ▼                                          ▼
   完整闭环：discover → curate → train → deploy → verify
```

**面试叙事模板**：

> "在做 Coder Agent 项目的过程中，我发现 base model 在 off-by-one errors 和 edge case handling 上反复犯错。于是在 CodeTune 项目中，我针对这些 failure patterns 构造了 targeted training data，在 Qwen2.5-Coder-7B 上做了 SFT + DPO。结果显示，targeted SFT 相比 generic-only SFT 在 edge case 类任务上 pass rate 提升了 X%，加上 DPO 后在 HumanEval+ 上 pass@1 从 A% 提升到 B%。最后我把 fine-tuned model 接回 Coder Agent，验证 agent 的整体任务完成率从 C% 提升到 D%。"

---

## 十、面试常见问题准备

### 模型与数据

**Q: 为什么选 Qwen2.5-Coder-7B 而不是 CodeLlama 或 DeepSeek-Coder？**
> Qwen2.5-Coder-7B 在同量级 code model 中 benchmark 表现领先，Apache 2.0 开源，社区 LoRA fine-tuning 支持成熟。CodeLlama 相对较老，DeepSeek-Coder-V2 是 MoE 架构，QLoRA 支持不如 dense model 稳定。

**Q: 为什么不做 full fine-tuning？**
> 9B 模型 full FT 需要 ~72GB VRAM（bf16），A100 80GB 也装不下。LoRA 只训练 ~0.5-1% 的参数，效果接近 full FT，而且 adapter 只有几十 MB，便于保存、切换和共享。

**Q: 数据量为什么只有 3K-4K？**
> 对于 LoRA SFT，数据质量远比数量重要。研究表明 1K-10K 条精选数据就能有效激活能力。我的实验也通过 2K vs 4K 的 ablation 验证了收益递减点。关键在于数据的质量控制：AST 可解析、去重、难度分布合理、无 benchmark 污染。

**Q: 怎么防止数据污染？**
> 两层防护：第一，SFT 数据中排除与 HumanEval/MBPP 题目相似的样本（用 embedding similarity 检测）；第二，评估时主要依赖 LiveCodeBench，这是一个和训练数据完全独立的 benchmark。

### SFT 与 DPO

**Q: DPO 相比 PPO 的优劣？**
> DPO 优势：不需要单独训练 reward model，实现更简单，训练更稳定。代码任务特别适合 DPO，因为 unit test 提供了客观的偏好信号，不需要人工标注。PPO 在开放域对话等偏好更主观的场景可能更灵活，但实现复杂度高且容易出现 reward hacking。

**Q: DPO 的 beta 怎么选？**
> beta 控制 policy 偏离 reference model 的程度。beta=0.1 是标准起点。太小（0.01）会导致过度优化偏好，可能生成格式混乱的代码；太大（1.0）会过于保守，改善不明显。我通过在 val set 上对比 0.05/0.1/0.3 确定了最优值。

**Q: 为什么 DPO 只用 pass/fail 作为偏好信号？**
> 代码质量的其他维度（可读性、效率、robustness）很难自动化地稳定定义。不稳定的偏好信号会污染 DPO 训练，导致模型学到噪声而不是真正的偏好。pass/fail 通过 unit test 客观判定，是最干净的信号。后续版本可以引入更多维度，但第一版要确保 baseline 干净。

**Q: SFT 之后 HumanEval 没有提升怎么办？**
> 首先检查数据质量：是否有格式错误、是否与 base model 的 chat template 对齐、是否有数据污染。其次检查超参：学习率是否太高导致遗忘。如果数据和超参都没问题，可能是 base model 在 HumanEval 上已经很强，improvement headroom 有限——这时 targeted failure subset 上的提升更有说服力。

### 工程与部署

**Q: 如果要把这个部署到生产环境？**
> 模型量化（AWQ/GPTQ 4bit）降低推理成本 → vLLM 部署支持 batch inference → A/B test 验证线上效果 → 监控 latency、pass rate、用户反馈。LoRA adapter 的好处是可以热切换：不同任务加载不同 adapter，无需重新部署 base model。

**Q: LoRA adapter 怎么做版本管理？**
> 每个 adapter 只有几十 MB，可以用 Git LFS 或 Hugging Face Hub 做版本管理。记录每个版本的训练数据、超参和 benchmark 结果，方便回溯。

---

## 十一、简历 Bullet Point 建议

完成项目后，简历上可以这样写（根据实际结果填入数字）：

> **CodeTune — Code Generation Post-Training Pipeline**
>
> - Built end-to-end SFT + DPO pipeline on Qwen2.5-Coder-7B with bf16 LoRA; improved LiveCodeBench pass@1 from X% to Y% (+Z%) using failure-pattern-driven training data derived from coding agent error analysis.
>
> - Designed automated DPO preference pair construction via unit test execution; conducted ablation showing targeted training data improved edge-case pass rate by W% over generic-only SFT baseline.

---

## 十二、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Qwen2.5-Coder-7B LoRA 训练报错 | 低 | 高 | 社区成熟，大量已验证案例；如遇问题查 Unsloth GitHub issues |
| SFT 后 HumanEval 无提升 | 中 | 中 | base model 已经很强；转而突出 targeted subset 提升 |
| DPO 导致模型退化 | 中 | 中 | 增大 beta、减少训练步数、检查 pair 质量 |
| Colab 断连 | 高 | 低 | 每 50 steps 保存 checkpoint，脚本支持 resume from checkpoint |
| DPO 有效 pair 数不够 | 中 | 中 | 降低 temperature 增加多样性、增加采样次数、补充更多 prompt |
| 本地 8GB 跑不动 7B 推理 | 低 | 中 | 4-bit GGUF 量化约 4-5GB，8GB 够用；或回 Colab 跑评估 |
| 时间不够完成全部 | 中 | 中 | 优先级：SFT > 评估 > DPO > ablation > 工程化 |

### 优先级排序（如果时间不够）

如果 14 天内无法完成所有内容，按以下优先级砍：

1. **必须完成**：SFT 训练 + HumanEval/MBPP 评估 + targeted failure eval
2. **高优先级**：DPO 训练 + 对比评估
3. **中优先级**：generic vs targeted ablation
4. **低优先级**：其他 ablation（rank、epochs）
5. **可选**：Coder Agent 联动验证、HF Hub 发布

即使只完成到第 2 级，你也已经有了一个 SFT+DPO 的完整项目，足以在简历和面试中讲清楚。
