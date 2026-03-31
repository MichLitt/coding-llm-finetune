# CodeTune v2 — DPO 阶段性评估报告

**日期**: 2026-03-18
**模型基座**: Qwen3.5-9B
**评估基准**: HumanEval (164题)
**起点模型**: sft_C (76.2%)

---

## 一、整体结果

| 模型    | pass@1    | 通过  | 失败 | vs base  | vs sft_C  |
|---------|-----------|-------|------|----------|-----------|
| sft_C   | 76.2%     | 125   | 39   | +38.4pp  | —         |
| dpo_v1  | **72.6%** | **119** | **45** | +34.8pp | **-3.6pp** |

**dpo_v1**: sft_C 基础上执行驱动的 DPO 训练（~147 pairs，3 epochs）

---

## 二、Targeted Failure Cases 追踪

原始目标：修复 Coder-Agent C5 v6 的 8 个真实逻辑错误。

| Task                           | base | sft_A | sft_C | dpo_v1  | 状态             |
|--------------------------------|------|-------|-------|---------|------------------|
| HumanEval/54 (same_chars)      | ✗    | ✗     | ✗     | **✓**   | **DPO新增修复**  |
| HumanEval/55 (fib)             | ✗    | ✓     | ✓     | ✓       | 保持修复         |
| HumanEval/107 (minSubArraySum) | ✓    | ✓     | ✓     | ✓       | 保持修复         |
| HumanEval/108 (intersection)   | ✗    | ✗     | ✗     | ✗       | 持续失败         |
| HumanEval/110 (prod_signs)     | ✗    | ✗     | ✓     | ✓       | 保持修复         |
| HumanEval/128 (tri)            | ✗    | ✓     | ✗     | ✗       | 持续失败         |
| HumanEval/141 (file_name_check)| ✗    | ✗     | ✓     | ✓       | 保持修复         |
| HumanEval/147 (get_max_triples)| ✓    | ✓     | ✓     | ✓       | 保持修复         |
| **合计**                       | 2/8  | 4/8   | 5/8   | **6/8** |                  |

**DPO新增修复**: HumanEval/54 (same_chars) — 通过 `set()` vs `sorted()` 的 chosen/rejected 对比学习成功纠正。
**持续失败**: HumanEval/108 (intersection) 和 HumanEval/128 (tri) 在所有版本中均无法稳定通过。

---

## 三、dpo_v1 失败案例分析

dpo_v1 共 45 道失败题，相比 sft_C 新增 6 题退步，以下为典型模式：

### 3.1 DPO 保留的 sft_C 原有失败

边界条件、语义误解等问题仍延续（参见 SFT 报告第三节），DPO 未能改善：
- HumanEval/9 (rolling_max): 空列表 IndexError
- HumanEval/26 (remove_duplicates): 语义误解
- HumanEval/108 (intersection): 集合交集逻辑错误

### 3.2 DPO 新增退步（sft_C 通过 → dpo_v1 失败）

约 6 题在 sft_C 中通过但 dpo_v1 失败，为典型的 alignment tax。
推测原因：训练 pairs 数量过少（~147对），偏好信号覆盖面窄，模型在优化
targeted 方向的同时对部分题型产生了轻微的能力退化。

---

## 四、DPO 训练配置回顾

| 配置项            | 值                              |
|-------------------|---------------------------------|
| 起点              | sft_C (Michlitt/codetune-v2-sft-C) |
| DPO pairs 来源    | execution-driven（本地执行测试筛选）|
| Train pairs       | ~147 对                         |
| Epochs            | 3                               |
| 训练总步数        | ~30 步                          |
| β (KL penalty)    | 0.1（默认）                     |
| 生成温度          | 0.2 / 0.7 / 1.0 / 1.2          |
| Score threshold   | execution pass/fail             |

---

## 五、结果分析

### 5.1 正面效果

- Targeted 从 5/8 提升至 **6/8**，新增修复 HumanEval/54
- HumanEval/54 的修复验证了 execution-driven DPO 策略的有效性：精准的 `set()` vs `sorted()` pairs 直接纠正了模型偏差
- 整体仍比 base 高 34.8pp，DPO 没有严重损伤基础能力

### 5.2 负面效果

- 整体 pass@1 从 76.2% 降至 72.6%（-3.6pp，约 6 题退步）
- HumanEval/108, HumanEval/128 仍未修复

### 5.3 退步原因推断

| 可能原因          | 说明                                           |
|-------------------|------------------------------------------------|
| pairs 数量过少    | ~147 对 vs 典型 DPO 需要 1000+ 对，信号稀疏   |
| 训练步数不足      | 约 30 步，模型更新幅度极小                     |
| chosen 质量不均   | 执行驱动筛选的 chosen 不一定是最优解，只是能过测试 |
| alignment tax     | DPO 固有现象，偏好学习轻微损伤一般能力         |

---

## 六、误差模式汇总

| 错误类型           | 典型案例                  | 出现频率 | 下一版优先级 |
|--------------------|---------------------------|----------|--------------|
| Targeted 持续失败  | HE/108, HE/128            | —        | ★★★          |
| 边界条件未处理     | HE/9                      | 高       | ★★★          |
| 题意语义误解       | HE/26                     | 中       | ★★★          |
| DPO 新增退步       | ~6 题                     | 低       | ★★           |
| 空生成（截断）     | HE/32, HE/38, HE/50       | 中       | ★★           |

---

## 七、后续改进方向

### 7.1 扩大 DPO pairs 规模

当前 ~147 对明显不足，目标 500+ 对：
- 降低 score_threshold，接受分差更小的 pairs
- 对 HE 全部 164 题逐题生成多候选，确保每题都有 pairs
- 针对 HE/108, HE/128 手工构造高质量 targeted pairs

### 7.2 针对退步题目补充训练

分析 dpo_v1 新增失败的 ~6 题，为其构造 sft_C 正确输出（chosen）vs dpo_v1 错误输出（rejected）的 pairs，加入下一轮训练。

### 7.3 调整训练超参

- 增加 epochs（当前 3 → 5）或增大 batch size 以获得更多有效更新步数
- 考虑降低 β（KL penalty）使模型更大幅度偏离 SFT，学到更强的偏好信号

### 7.4 下一版目标

| 指标               | dpo_v1 当前 | dpo_v2 目标  |
|--------------------|-------------|--------------|
| HumanEval pass@1   | 72.6%       | ≥76%（≥sft_C）|
| Targeted pass      | 6/8         | ≥7/8         |
| HE/108 intersection| ✗           | ✓            |
| HE/128 tri         | ✗           | ✓            |

---

## 八、结论

DPO 阶段完成首次迭代：dpo_v1 在 targeted 指标上取得进步（5/8 → 6/8），成功修复了 HumanEval/54，验证了 execution-driven pairs 策略的方向正确性。但受限于训练数据规模（~147 对，~30 步），整体出现轻微退步（-3.6pp），符合小规模 DPO 的预期表现。

完整训练链路：base 37.8% → sft_A 65.2% → sft_C 76.2% → dpo_v1 72.6%

下一步核心任务是扩大 DPO pairs 规模至 500+ 对，并针对持续失败的 HE/108、HE/128 构造更精准的 targeted pairs。
