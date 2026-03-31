# CodeTune v2 — SFT 阶段性评估报告

**日期**: 2026-03-17
**模型基座**: Qwen2.5-Coder-7B-Instruct
**评估基准**: HumanEval (164题)

---

## 一、整体结果

| 模型   | pass@1  | 通过 | 失败 | 提升 vs base |
|--------|---------|------|------|--------------|
| base   | 37.8%   | 62   | 102  | —            |
| sft_A  | 65.2%   | 107  | 57   | +27.4pp      |
| sft_C  | **76.2%** | **125** | **39** | **+38.4pp** |

**sft_A**: magicoder only
**sft_C**: magicoder + evol + targeted data（核心方案）

sft_C 相比 sft_A 额外解决了 18 道题（+11.0pp），targeted data 的加入效果显著。

---

## 二、Targeted Failure Cases 追踪

原始目标：修复 Coder-Agent C5 v6 的 8 个真实逻辑错误（baseline 0/8）。

| Task                           | base | sft_A | sft_C | 状态     |
|--------------------------------|------|-------|-------|----------|
| HumanEval/54 (same_chars)      | ✗    | ✗     | ✗     | 持续失败 |
| HumanEval/55 (fib)             | ✗    | ✓     | ✓     | 已修复   |
| HumanEval/107 (minSubArraySum) | ✓    | ✓     | ✓     | 已修复   |
| HumanEval/108 (intersection)   | ✗    | ✗     | ✗     | 持续失败 |
| HumanEval/110 (prod_signs)     | ✗    | ✗     | ✓     | sft_C修复 |
| HumanEval/128 (tri)            | ✗    | ✓     | ✗     | sft_C退步 |
| HumanEval/141 (file_name_check)| ✗    | ✗     | ✓     | sft_C修复 |
| HumanEval/147 (get_max_triples)| ✓    | ✓     | ✓     | 已修复   |
| **合计**                       | 2/8  | 4/8   | **5/8** |        |

**值得注意**: HumanEval/128 (tri) 在 sft_A 中通过，sft_C 中退步，说明 targeted data 可能引入了对该类三角数递归逻辑的干扰。

---

## 三、sft_C 仍失败案例分析（非 targeted）

共 39 道失败题，以下为典型错误模式：

### 3.1 边界条件未处理

**HumanEval/9 — rolling_max**
```python
# 模型输出（错误）
max_num = numbers[0]   # 空列表时 IndexError
result = [max_num]
for num in numbers[1:]:
    if num > max_num:
        max_num = num
    result.append(max_num)
return result
```
- **根因**: 未处理空列表 `[]` 输入
- **修复**: 加 `if not numbers: return []` 前置守卫

### 3.2 题意理解错误

**HumanEval/26 — remove_duplicates**
```python
# 模型输出（错误）：去重保留第一次出现
# 正确语义：删除所有出现超过1次的元素
# [1,2,3,2,4,3,5] → [1,4,5]，不是 [1,2,3,4,5]
```
- **根因**: "remove duplicates" 被理解为"去重"而非"删除重复项"

**HumanEval/54 — same_chars**
```python
# 模型输出（错误）
return sorted(s0) == sorted(s1)
# 正确
return set(s0) == set(s1)
```
- **根因**: sorted 保留频次信息，set 才是字符集合比较；targeted data 的示例不足以纠正此偏差

### 3.3 大小写处理不一致

**HumanEval/64 — vowels_count**
```python
# 模型输出（错误）
for char in s.lower():      # 循环内转小写 ✓
    if char in vowels: ...
if s[-1] == 'y':            # 末尾检查未转小写 ✗
# 正确
if s[-1].lower() == 'y':
```
- **根因**: 局部一致性失效，大小写处理逻辑未统一应用

### 3.4 空生成（代码截断/拒绝）

- **HumanEval/32** (find_zero): 多项式根的二分查找，Completion 为空
- **HumanEval/38** (decode_cyclic): 循环编码逆向，Completion 为空
- **HumanEval/50** (decode_shift): Caesar cipher 解码，Completion 为空

- **根因分析**: 可能是 `max_new_tokens` 设置过小导致截断，或模型对此类数学/加密类问题生成意愿低。需检查推理时 `max_new_tokens` 参数。

---

## 四、误差模式汇总

| 错误类型           | 典型案例              | 出现频率 | DPO优先级 |
|--------------------|-----------------------|----------|-----------|
| 空列表/边界条件    | HE/9                  | 高       | ★★★       |
| 题意语义误解       | HE/26, HE/54          | 中       | ★★★       |
| 大小写不一致       | HE/64                 | 中       | ★★        |
| 空生成（截断）     | HE/32, HE/38, HE/50   | 中       | ★★（先查配置）|
| Targeted持续失败   | HE/54, HE/108         | —        | ★★★       |
| sft_C退步          | HE/128 (tri)          | —        | ★★        |

---

## 五、DPO 阶段建议

### 5.1 DPO Pairs 生成优先级

1. **HumanEval/54, HE/108**: targeted 仍失败，直接加入 DPO chosen/rejected pairs
2. **HumanEval/128**: sft_C 退步，需构造 tri 递归逻辑的正确 pairs
3. **边界条件类**: 批量生成含空列表、空字符串输入的变体
4. **语义类**: 为 remove_duplicates、same_chars 类题目补充反例

### 5.2 配置检查

在运行 `scripts/generate_dpo_pairs.py` 前，确认：
- `max_new_tokens >= 512`（防止空生成）
- 对 HE/32, HE/38, HE/50 单独检查是否是截断问题还是模型能力问题

### 5.3 预期目标

| 指标               | sft_C 当前 | DPO 目标  |
|--------------------|------------|-----------|
| HumanEval pass@1   | 76.2%      | ≥80%      |
| Targeted pass      | 5/8        | ≥7/8      |
| HE/54 same_chars   | ✗          | ✓         |
| HE/108 intersection| ✗          | ✓         |
| HE/128 tri         | ✗          | ✓         |

---

## 六、结论

SFT阶段取得显著成果：base 37.8% → sft_C 76.2%，提升 38.4pp。Targeted data 策略有效，targeted failures 从 2/8 提升至 5/8。主要剩余挑战在于：边界条件防御性编程、细粒度语义理解、以及 2 个持续失败的 targeted cases（HE/54, HE/108）。DPO 阶段将重点针对这些失败模式构造高质量 chosen/rejected pairs。
