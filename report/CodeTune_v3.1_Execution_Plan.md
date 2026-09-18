# CodeTune v3.1 执行计划：修复评测 → 基线 → SFT → DPO

> 状态：阶段 0 已在分支 `fix/v3.1-eval-harness` 上实现（2026-09-18），阶段 1–5 待 Colab 执行。与原草案的差异见文末“实现记录”。
> 取代：`CodeTune_v3_Project_Plan.md` 中的 targeted 数据部分（3.2 节、5.2 节）以及 `sft_targeted*` 系列消融实验
> 算力：本机 Apple M5 只做 CPU 工作；GPU 工作全部在 Colab A100/H100 上完成
> 协作约束：所有改动都在 `coding-llm-finetune` 子仓库的独立分支上进行，不碰主仓库（Codex 正在主仓库工作），详见第 9 节

---

## 0. 目标与非目标

**目标**

1. 在 `Qwen/Qwen3.5-4B`（后训练版）上完成 基线 → SFT → DPO 三组对照实验。评测必须干净，差异要能归因。
2. 核心卖点是**执行反馈 DPO**：偏好对由单元测试执行结果构造，并分析 alignment tax。
3. 不论结果是提升、持平还是退化，都产出一份带置信区间和分类失败分析的报告。

**非目标**

- 不针对某几道 HumanEval 题做定向修复（targeted 数据整体取消，理由见第 1 节 P3、P4）。
- 不追求刷榜数字。结果是负面的也照实写进报告。
- GRPO 只作为可选加分项，放在第 5 阶段，不阻塞主线。

---

## 1. 审计结论：当前代码的问题清单

严重程度：🔴 会让结果作废　🟠 会明显扭曲结果　🟡 影响效率或复现

| ID | 严重度 | 问题 | 位置 |
|---|---|---|---|
| P1 | 🔴 | 评测不抽取代码，直接执行 `prompt + 原始聊天输出`，代码块标记和重复定义都会被判为失败 | `scripts/run_humaneval.py` `_run_tests`；`run_targeted_eval.py` 同样 |
| P2 | 🔴 | thinking 模式默认开启：不传参数时，生成提示以 `<think>\n` 结尾。而 SFT 训练文本里助手回复被渲染成空的 `<think>\n\n</think>\n\n`（非思考格式）。基线和 SFT 模型在不同模式下被比较；另外 `max_new_tokens=512` 会截断思考内容 | 所有 `apply_chat_template(..., add_generation_prompt=True)` 调用处 |
| P3 | 🔴 | targeted 数据泄露答案：`key_insight` 写的就是题解，脚本还用 HumanEval 的 `check()` 验证生成数据 | `scripts/generate_targeted_data.py` |
| P4 | 🔴 | 8 道题里有 4 道 task_id 标错：/107 实际是 even_odd_palindrome，/108 是 count_nums，/110 是 exchange，/128 是 prod_signs。v2 报告里"HE/108 intersection 持续失败"的结论测的其实是另一道题。另外，这 8 道失败题来自 Coder Agent，而不是 Qwen3.5-4B | `generate_targeted_data.py`、`run_targeted_eval.py`、`generate_dpo_pairs.py`、v3 计划、`report/archive/*` |
| P5 | 🔴 | 生成 DPO 偏好对时同样存在 P1 和 P2，通过与否主要由输出格式决定，偏好对学到的是格式 | `scripts/generate_dpo_pairs.py` |
| P6 | 🟠 | MBPP 的 prompt 里没有函数名或测试样例，模型给函数起别的名字就会被判失败（假阴性） | `generate_dpo_pairs.py` `_generate_candidates` |
| P7 | 🟠 | DPO 训练用的 prompt 是纯字符串，TRL 不会套聊天模板，训练格式和推理格式不一致 | `generate_dpo_pairs.py` 输出、`dpo_train.py` |
| P8 | 🟠 | 采样温度最高到 1.8，rejected 多半是乱码或截断输出，负例太容易区分；没有长度控制 | `generate_dpo_pairs.py` `--temperatures` |
| P9 | 🟠 | 每题只产出 1 对，250 题最多约 100–150 对。文档写的"test split 374 题"也是错的：test split 有 500 题，374 是 train split | 同上 |
| P10 | 🟠 | 消融矩阵覆盖了 `lora_r=32`，但 `lora_alpha` 仍是 128，alpha/r 从 2 变成 4 | `configs/ablation_matrix.yaml` 与 `sft_config_colab_a100.yaml` |
| P11 | 🟠 | lr 2e-4、3 个 epoch 对已经后训练过的模型偏激进，容易遗忘原有能力 | `sft_config_colab_*.yaml` |
| P12 | 🟡 | A100 配置下大约只训练 170 步，但 `eval_steps=250`、`save_steps=500`，训练中既不评测也不存检查点，`load_best_model_at_end` 形同虚设；结果只在最后同步回 Drive | `sft_config_colab_a100.yaml`、notebook |
| P13 | 🟡 | 评测和生成偏好对时写死了 4-bit 加载，和 bf16 训练不一致 | `run_humaneval.py`、`generate_dpo_pairs.py` |
| P14 | 🟡 | notebook 安装 Unsloth 的 git 最新版并升级 torch，环境不可复现；notebook 里没有评测步骤 | `notebooks/colab_train_a100_h100.ipynb` |
| P15 | 🟡 | 本机 `.venv` 是 Python 3.14，`datasets` 会报 dill 错误 | `.python-version` |
| P16 | 🟡 | README 和计划文档写的是 "Qwen3.5-4B-Instruct"，HF 上不存在这个模型 | `README.md`、v3 计划 |
| P17 | 🟠 | DPO 从 SFT 适配器起步时，先把 SFT 合并进基座再挂新的 LoRA，保存下来的只有 DPO 增量；直接用「基座 + DPO 适配器」评测会漏掉 SFT 权重 | `dpo_train.py`、评测加载 |
| P18 | 🟡 | `datasets>=4` 不再支持 `trust_remote_code`，Evol-CodeAlpaca 加载直接报错 | `prepare_sft_data.py` |

---

## 2. 关键设计决策

**D1. 取消 targeted 数据。**
删除 `generate_targeted_data.py`、`run_targeted_eval.py`、`sft_targeted*` 实验，以及计划 5.2 节的手工偏好对。

理由：P3 和 P4。即使去掉泄露，这批"失败模式"也不是被训练模型自己的失败。可以用"每个模型自己由对变错、由错变对的题目"做分析来替代。

**D2. 统一使用非思考模式。**
推理、偏好对生成、DPO 训练的 prompt 都通过同一个函数构造，固定传 `enable_thinking=False`。SFT 训练文本本来就会渲染出空的 think 块，与此一致（加单元测试锁定这一点）。

**D3. 唯一的 prompt 与抽取入口。**
新增 `scripts/codegen_common.py`，所有脚本都只从这里导入：

- `build_messages(task)`：构造 system 和 user 消息
- `render_prompt(tokenizer, messages)`：套模板，固定 `enable_thinking=False`
- `extract_code(raw, entry_point)`：抽取代码
- `classify_result(...)`：给执行结果分类

**D4. 评测集划分**

| 用途 | 数据集 | 说明 |
|---|---|---|
| 主评测 | HumanEval（164 题）+ HumanEval+ | 用 evalplus 执行 |
| 第二评测 | MBPP+（evalplus 版） | 不参与任何训练 |
| DPO 偏好对来源 | MBPP full（974 题）**减去所有 MBPP+ 的 task_id** | 约 600 题，以实际计算为准；按 task_id 硬排除 |
| SFT 数据 | Magicoder-OSS-Instruct + Evol-CodeAlpaca | 通过污染检查 |

**D5. 评测精度统一用 bf16**，不做 4-bit 量化。所有模型用同一套生成参数：贪心解码，`max_new_tokens=1024`。

**D6. 统计方法**

- 报告 pass@1 时附带配对 bootstrap 95% 置信区间（重采样 10,000 次）。
- 两个模型比较时报告 McNemar 精确检验的 p 值，并列出由对变错、由错变对的题目。
- 可选：以温度 0.2 每题采样 10 次，用无偏估计计算 pass@1，作为稳健性检查。

**D7. 失败分类**（写进每道题的结果里）

| 类别 | 判定条件 |
|---|---|
| `pass` | 通过全部测试 |
| `no_code` | 抽取不到代码 |
| `truncated` | 生成达到 `max_new_tokens` 上限且没有 EOS |
| `syntax_error` | `ast.parse` 失败 |
| `missing_entry_point` | 代码里没有定义目标函数 |
| `runtime_error` | 执行时抛出异常（断言失败除外） |
| `wrong_answer` | `AssertionError` |
| `timeout` | 执行超时 |

---

## 3. 分阶段计划

### 阶段 0：修复与重构（只在本机做，CPU 即可，1.5–2 天）

先在子仓库建分支：`git switch -c fix/v3.1-eval-harness`（基于 `origin/main`）。

#### 0.1 环境

- 本机固定用 Python 3.12：修改 `.python-version`，然后执行 `uv sync`。CI 本来就是 3.12。
- `pyproject.toml` 的 dev 依赖中加入 `evalplus` 和 `pytest`。

#### 0.2 新增 `scripts/codegen_common.py`（D3）

- `render_prompt` 调用模板时固定 `enable_thinking=False`。
- `extract_code` 的逻辑（最终实现见文末“实现记录”）：
  1. 去掉 `<think>…</think>` 块；
  2. 如果有 ```` ```python ```` 代码块，取其中包含 `def {entry_point}` 的那个，否则取最后一个；
  3. 如果没有代码块，就把整段输出当作代码；
  4. 抽出的代码如果已经定义了 `entry_point`，就单独执行，同时带上 prompt 里的 import 头；否则拼在 prompt 后面再执行。
  - 实现时优先复用 `evalplus.sanitize`，只在外层补充上述判断。
- `classify_result`：按 D7 分类；截断通过生成的 token 数和 EOS 是否出现来判断。

#### 0.3 重写评测：`scripts/run_eval.py`（替代 `run_humaneval.py`）

- 参数：`--model`、`--label`、`--dataset {humaneval,mbpp}`、`--backend {hf,vllm}`、`--max-new-tokens 1024`
- 输出目录 `results/eval/<label>/`，包含：
  - `raw.jsonl`：原始输出、生成 token 数、是否结束
  - `samples.jsonl`：抽取后的代码，格式符合 evalplus
  - `per_problem.jsonl`：base 和 plus 测试各自的状态，以及失败类别
  - `summary.json`：pass@1、失败类别分布、平均输出长度、截断率
- 执行交给 `evalplus.evaluate`；失败类别由我们自己的分类函数在 base 测试上补充。
- 模型加载：LoRA adapter 先合并到 bf16 基座再推理；vLLM 后端直接加载合并后的模型目录。

#### 0.4 重写偏好对生成：`scripts/generate_dpo_pairs.py`

- **题库**：MBPP full 减去 MBPP+ 的 task_id，把排除的数量写进日志。
- **prompt**：题目描述 + "Your code should pass this test:" + 第一条 `assert`（MBPP 的标准做法），解决 P6。
- **采样**：每题 8 个候选，温度取 {0.2, 0.4, 0.6, 0.8, 1.0}，`top_p=0.95`；用 vLLM 批量生成。
- **判定**：所有候选都经过 `extract_code` 和 `classify_result`。
  - chosen 取 `pass` 的候选；
  - rejected 只能是**难负例**：`wrong_answer` 或 `runtime_error`，并且定义了目标函数；
  - `no_code`、`syntax_error`、`truncated`、`timeout` 一律不能作为 rejected。
- **每题最多 2 对**：同一个 chosen 不重复使用；优先选长度比在 [0.67, 1.5] 以内的组合，超出的丢弃。
- **存储内容**：chosen 和 rejected 存模型的**原始回复文本**（去掉 think 块后），保持同策略的格式；抽取出的代码只用来判定。
- **数据格式**（解决 P7）：`prompt` 存用 `render_prompt` 预先套好模板的字符串，`chosen`、`rejected` 存回复文本。加单元测试：DPO 的 prompt 字符串必须与推理时的 prompt 逐字相同。
- **切分**：按 task_id 以 90/10 切分训练集和验证集。
- **统计输出** `dpo_pairs_stats.json`：产出率、rejected 的类别分布、长度比的分位数、chosen 与 rejected 的平均 token 数。
- **目标**：300 对以上。不够的话把每题候选数提到 12。

#### 0.5 DPO 训练：`scripts/dpo_train.py`

- 确认 TRL 版本对字符串 prompt 不会重复套模板，也不会漏加 EOS。用 3 条样本打印 tokenize 后的结果，做一次人工检查。
- 训练日志额外记录：`rewards/margins`、`rewards/accuracies`、`logps/chosen`、`logps/rejected`。如果 `logps/chosen` 持续下降，要在报告里写明。
- 支持 `--sft-checkpoint base`，也就是直接从基线模型开始做 DPO（SFT 退化时使用）。

#### 0.6 SFT 与配置

- `ablation_matrix.yaml` 改为同时指定 `lora_r`、`lora_alpha`、`learning_rate`、`num_train_epochs`，由矩阵统一控制，A100 配置只作为唯一基准（解决 P10）：

  ```yaml
  sft_generic_aggr:   {data_sources: [magicoder, evol], lora_r: 32, lora_alpha: 64, learning_rate: 2.0e-4, num_train_epochs: 3}
  sft_generic_cons:   {data_sources: [magicoder, evol], lora_r: 16, lora_alpha: 32, learning_rate: 5.0e-5, num_train_epochs: 1}
  ```

- `sft_train.py` 读取矩阵里的 `lora_alpha` 和 `learning_rate`，最终生效的超参写进 `meta.json`。
- A100 和 H100 配置：`eval_steps: 25`、`save_steps: 25`、`save_total_limit: 3`；`output_dir` 可以通过环境变量指向 Drive（解决 P12）。
- 加单元测试：用本地缓存的 Qwen3.5 tokenizer 渲染一条 SFT 样本，断言助手回复部分以 `<think>\n\n</think>\n\n` 开头，与 D2 一致。

#### 0.7 清理 targeted 相关内容（D1）

- 删除 `generate_targeted_data.py`、`run_targeted_eval.py`，以及 `generate_dpo_pairs.py` 里的 `TARGETED_TASK_IDS`。
- `check_contamination.py` 里保留 HumanEval 函数名黑名单；新增与 MBPP+ 题目的 n-gram 检查。
- 在 `report/archive/` 下新增 `ERRATA.md`，说明 P4 的 ID 错误，以及 v2 报告里逐题归因的结论不成立。归档报告正文不改。

#### 0.8 notebook 改造：`notebooks/colab_train_a100_h100.ipynb`

- **锁定版本**：Unsloth 用固定版本号，不再装 git 最新版；torch、trl、peft、vllm、evalplus 都写死版本，版本号在第一次成功运行后确定。
- **运行目录**：每次运行在 Drive 上写 `runs/<日期>_<实验名>/`，里面包含：
  - `manifest.json`：git commit、配置文件内容、GPU 型号、CUDA 版本
  - `pip_freeze.txt`
  - 日志
- **断点续训**：检查点直接写到 Drive；`SFT_RESUME_FROM` 支持自动找最新的检查点。
- **新增评测 cell**：`EVAL_TARGETS = {"base": ..., "sft_x": ..., "dpo_y": ...}`，对每个模型跑 HumanEval 和 MBPP+；每跑完一个就同步回 Drive。
- **运行开关**：`RUN_EVAL_BASE`、`RUN_SFT`、`RUN_EVAL_SFT`、`RUN_DPO_PAIR_GEN`、`RUN_DPO`、`RUN_EVAL_DPO`。

#### 0.9 测试与 CI

在 `tests/` 下添加离线测试，都不需要 GPU：

- `extract_code`：覆盖至少 10 种真实形态的输出，包括代码块、无代码块、重复定义、think 残留、多个代码块、只有函数体。
- `classify_result`：D7 中每个类别至少一个例子。
- 模板一致性：D2 以及 0.4 里的逐字相同断言。模板测试依赖 tokenizer，CI 中如果拿不到就跳过。
- DPO 偏好对过滤逻辑：用人工构造的候选验证难负例规则和长度比规则。

CI 增加一步 `uv run pytest -q`。

#### 0.10 文档

- README 和 v3 计划里的 "Instruct" 改为 `Qwen/Qwen3.5-4B（后训练版）`。
- v3 计划开头加一段"已被 v3.1 取代"的说明，并链接到本文件。

**阶段 0 验收条件**

- [ ] `uv run pytest` 全部通过，CI 通过
- [ ] 代码中搜不到 `TARGETED_TASK_IDS`、`generate_targeted_data` 等 targeted 相关引用
- [ ] 分支推送到 GitHub，notebook 里 `REPO_BRANCH` 可以指向该分支
- [ ] SFT 数据通过 `prepare_sft_data.py` 和 `check_contamination.py`（结果为 0 条命中），并上传到 Drive

---

### 阶段 1：基线评测（Colab 会话 1，约 0.5 天）

1. 用 `Qwen/Qwen3.5-4B`、bf16 跑 HumanEval、HumanEval+、MBPP+。
2. **评测可信度门槛**，不满足就回到阶段 0 修复：
   - `truncated` 不超过 1%；
   - `no_code` 与 `syntax_error` 合计不超过 2%；
   - 人工抽查 10 道失败题，确认是真的做错了，而不是抽取出了问题。
3. 可选：和官方或社区公布的 Qwen3.5-4B HumanEval 分数对照。差距过大（超过 10pp）要查明原因。

**产出**：`results/eval/base/`，以及基线的失败分类表。

---

### 阶段 2：SFT（Colab 会话 2，1–2 天）

1. 训练 `sft_generic_aggr` 和 `sft_generic_cons` 两组。
2. 每组训练完立刻用阶段 1 同样的参数评测。
3. 看训练曲线：训练 loss 和验证 loss；验证 loss 如果早早开始上升，记录下来。
4. 分别和基线做配对比较（D6）。

**决策规则**

- 至少一组 SFT 的 pass@1 显著不低于基线，并且没有新增格式类失败：最好的那组作为 DPO 起点。
- 两组都退化：DPO 从基线模型开始，SFT 退化本身作为报告的一个发现（对已经后训练过的模型再做通用 SFT 的代价）。

---

### 阶段 3：DPO（Colab 会话 3，1–2 天）

1. 用 DPO 起点模型生成偏好对（0.4），用 vLLM 生成，预计不到 30 分钟。
2. **偏好对门槛**，满足后才开始训练：
   - 至少 300 对；
   - chosen 与 rejected 的长度比中位数在 [0.8, 1.25] 之间；
   - rejected 中 `wrong_answer` 占比至少 50%。
3. 训练 `dpo_b01`（β=0.1）和 `dpo_b03`（β=0.3），其他超参相同：1 个 epoch，lr 5e-6 到 5e-5。第一轮先用 5e-6，reward margin 增长太慢再调高。
4. 训练过程中监控：reward margin 和 accuracy、`logps/chosen` 的变化趋势，以及验证集上生成的平均长度。
5. 训练完做评测和配对比较。对比组有：DPO 与起点模型、DPO 与基线。在 MBPP+ 上的变化用来衡量迁移效果；HumanEval 上 pass@1 下降的部分就是 alignment tax。

---

### 阶段 4：分析与报告（本机，约 1 天）

新增 `scripts/compare_runs.py`，输入若干个 `results/eval/<label>/`，输出：

- 汇总表：模型 × 数据集的 pass@1，附 95% 置信区间
- 两两配对比较：差值、置信区间、McNemar 检验 p 值、由对变错和由错变对的题目列表
- 各模型的失败类别分布，以及输出长度分布

报告写入 `report/CodeTune_v3.1_Report.md`，结构：

1. 设置：模型、数据、评测协议，以及做过的修复（把"评测修复前后基线分数的变化"作为一个发现单独写）
2. 主结果表
3. SFT 分析
4. DPO 分析：偏好对统计、训练动态、alignment tax
5. 局限性：单一种子、单次贪心采样、数据规模

---

### 阶段 5（可选）：GRPO

- 前提：阶段 3 完成，而且还有剩余算力。
- 用 TRL 的 `GRPOTrainer`，奖励定为 MBPP 单元测试的通过比例（复用 0.4 的题库和执行器），每组生成 4–8 个样本，训练 100–200 步。
- 和 DPO 同样的评测与比较方式。做出来之后简历上才可以写 RL。

---

## 4. 实验矩阵

| 标签 | 起点 | 数据 | 关键超参 | 评测集 |
|---|---|---|---|---|
| `base` | Qwen3.5-4B | — | — | HE、HE+、MBPP+ |
| `sft_generic_aggr` | base | Magicoder + Evol | r32 / α64 / lr 2e-4 / 3 ep | 同上 |
| `sft_generic_cons` | base | Magicoder + Evol | r16 / α32 / lr 5e-5 / 1 ep | 同上 |
| `dpo_b01` | 最好的 SFT 或 base | MBPP 偏好对（已去除 MBPP+ 题目） | β 0.1 / r16 / 1 ep | 同上 |
| `dpo_b03` | 同上 | 同上 | β 0.3 | 同上 |
| `grpo_*`（可选） | 同上 | MBPP（已去除 MBPP+ 题目） | 待定 | 同上 |

所有评测统一使用：bf16、非思考模式、贪心解码、`max_new_tokens=1024`、相同的 system prompt、相同的抽取逻辑。

---

## 5. Colab 运维规范

- 每个 Colab 会话只完成一个阶段；会话开始先跑环境检查 cell，把 manifest 写到 Drive。
- 所有长任务的输出直接写到 Drive，或者每完成一个子步骤就同步一次，不要等到最后。
- 检查点每 25 步存一次；断线后通过 `SFT_RESUME_FROM=auto` 续训。
- API key 只在会话里临时输入，不写进 notebook（沿用现有规定）。
- 本机只做结果汇总：从 Drive 下载 `results/eval/*` 到本地，不下载模型权重。

---

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| 评测修好后基线已经很高（例如 HumanEval 超过 85%），提升空间很小 | 以 HumanEval+ 和 MBPP+ 为主要指标；把"对强后训练模型做 SFT/DPO 的收益和代价"写成报告的主线 |
| SFT 两组都退化 | 按阶段 2 的决策规则，DPO 直接从基线开始 |
| 偏好对不足 300 对 | 每题候选数提到 12；调整温度组合；如果还是不够，如实报告实际数量 |
| DPO 让 `logps/chosen` 下降、输出变长 | 用 β=0.3 那组、降低学习率；报告里给出长度分析 |
| Unsloth、vLLM 与 Qwen3.5 版本不兼容 | 回退到 HF generate 做评测（慢但稳定）；版本锁定在第一次跑通的组合 |
| Colab 断线 | 检查点和结果都在 Drive 上，按第 5 节续跑 |

---

## 7. 时间线（估计）

| 天 | 内容 | 地点 |
|---|---|---|
| 第 1–2 天 | 阶段 0（0.1–0.10），CI 通过，推送分支，上传数据 | 本机 |
| 第 3 天 | 阶段 1 基线评测，检查评测可信度门槛 | Colab |
| 第 4–5 天 | 阶段 2 SFT 两组，训练和评测 | Colab |
| 第 6–7 天 | 阶段 3 生成偏好对，DPO 两组，训练和评测 | Colab |
| 第 8 天 | 阶段 4 分析和报告，更新简历 | 本机 |
| 之后 | 阶段 5 GRPO（可选） | Colab |

---

## 8. 做完后的简历写法（按实际结果二选一）

- **有提升时**：构建执行反馈 DPO 流水线：以 MBPP 单元测试构造难负例偏好对，并与 MBPP+ 评测集去重。在 Qwen3.5-4B 上做 SFT/DPO 对照实验，HumanEval+ 提升 X pp（95% 置信区间 [a, b]），同时量化 alignment tax。
- **持平或退化时**：修复评测流水线中的代码抽取和 thinking 模式问题，基线分数由 A% 修正为 B%。系统比较了 SFT 与执行反馈 DPO 在已后训练模型上的收益和代价，并按失败类型分析退化来源。

---

## 9. 与 Codex 的协作约束

- 所有改动只在 `coding-llm-finetune` 子仓库的 `fix/v3.1-eval-harness` 分支上进行，不修改主仓库的任何文件，包括 `docs/` 和子模块指针。
- 子仓库目前处于 detached HEAD 状态（1402dd8，也就是 origin/main）。先基于 `origin/main` 建分支再开始工作。
- 合入方式：在子仓库发 PR 合到 `main`。主仓库的子模块指针等 Codex 当前任务提交完成后，再单独提交更新。
- 阶段 1 到 3 在 Colab 上运行，只读取 GitHub 分支和 Drive，不影响本机任何仓库。

---

## 10. 实现记录（阶段 0 完成后补充）

与上文草案相比，实现时做了这些调整，均已在代码和测试中落实：

| 项 | 草案 | 实现 | 原因 |
|---|---|---|---|
| 代码抽取 | 优先复用 `evalplus.sanitize` | 自写 `codegen_common.extract_code`（fenced 块选择 + `ast` 去掉顶层示例调用/`__main__`） | 需要可离线单测、并区分 `no_code`/`missing_entry_point`；已在 HumanEval 全部 164 道参考解上验证：仅函数体、完整定义两种回复形态都是 164/164 |
| 执行器 | 全部交给 evalplus | HumanEval base 测试由自己的沙箱执行器跑，plus 测试和 MBPP+ 由 evalplus 跑，并互相交叉校验（`executor_disagreements`） | evalplus 只给 pass/fail/timeout，分不出 `wrong_answer` 与 `runtime_error`；evalplus 在 macOS 上因 `RLIMIT_AS` 无法运行，只能在 Linux（Colab/CI）上跑 |
| 后端 | `{hf, vllm}` | `{unsloth, hf, vllm}`，默认 `unsloth` | Qwen3.5-4B 是多模态混合注意力架构，训练脚本一直用 Unsloth 加载；`vllm` 标记为实验性，未验证 |
| 截断判定 | 生成达到上限且无 EOS 即 `truncated` | 只有当失败属于格式类（`no_code`/`syntax_error`/`missing_entry_point`）且触顶时才记为 `truncated`；触顶但代码可运行仍记 `wrong_answer`。基线门槛改用 `hit_token_limit_rate` | 避免把“答错但话多”误记为截断 |
| embedding 相似度检查 | 在 `check_contamination.py` 增加 | 未做；改为 5-gram 包含度（containment）检查，对象是 HumanEval 题面、HumanEval 参考解和 MBPP+ 题面，并接入 `prepare_sft_data.py` 作为过滤步骤 | 取消 targeted 数据后 LLM 改写变体的风险不存在；公开语料用 n-gram 包含度即可覆盖，且不引入 sentence-transformers 依赖 |
| DPO 题库 | MBPP full 去掉 MBPP+，约 600 题 | 596 题，再剔除参考解自己都过不了测试的 4 题，剩 **592 题**，存入 `configs/mbpp_dpo_pool_ids.json` | 测试有误的题会把正确候选判成失败 |
| DPO 适配器 | — | `experiment_meta.json` 记录起点 SFT 适配器；`model_io.adapter_chain` 评测时按顺序重新合并 | 见 P17 |
| SFT 数据规模 | — | `prepare_sft_data.py --magicoder-n 2000 --evol-n 1500`（取代 `--max-samples`） | 与原 v3 计划的数据规模一致，组成明确 |
| DPO 学习率 | 5e-6 起步 | 三份 DPO 配置统一改为 5e-6，日志每 2 步、评测每 10 步 | 偏好对只有几百对，步数很少 |
| 新增 | — | `train_utils.py`（输出目录/断点续训/格式自检/训练动态汇总）、`compare_runs.py`（阶段 4）、`tests/`（70 项离线测试）、CI 中 Linux 上的 evalplus 集成测试（advisory） | — |

**尚未验证的部分**（本机是 CPU-only Mac，无法验证）：Unsloth/TRL/vLLM 在 Colab 上的实际行为、`run_eval.py` 的 evalplus 调用（CI 的 Linux 任务会先覆盖一部分）、Qwen3.5 上 `train_on_responses_only` 的实际掩码效果。第一次 Colab 会话应先跑 `--subset` 小样本冒烟，再跑全量。
