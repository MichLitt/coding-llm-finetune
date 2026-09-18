# CodeTune v3.1: SFT + DPO Post-Training Pipeline

A supervised fine-tuning (SFT) and direct preference optimization (DPO) pipeline for code generation on
Qwen3.5-4B, built around a **leak-free, failure-classifying evaluation harness** and **execution-feedback preference pairs**.

**Base model**: `Qwen/Qwen3.5-4B` (the post-trained release; loaded as `unsloth/Qwen3.5-4B`. There is no "-Instruct" variant)
**Hardware**: Colab A100/H100 for all GPU work; a CPU-only Mac for data prep, tests and analysis
**Key design**: no benchmark data in any training set; one prompt, one extraction routine and one failure taxonomy shared by evaluation and DPO-pair generation

Plan: [report/CodeTune_v3.1_Execution_Plan.md](report/CodeTune_v3.1_Execution_Plan.md) (supersedes the targeted-data parts of [CodeTune_v3_Project_Plan.md](report/CodeTune_v3_Project_Plan.md)).

---

## Results

> No experiment results yet. Phase 0 (harness, data, tests) is implemented; baseline, SFT and DPO runs are pending (Colab).

| Model | HumanEval pass@1 | HumanEval+ | MBPP+ | vs baseline (paired 95% CI) |
|-------|------------------|-----------|-------|------------------------------|
| Qwen3.5-4B (baseline) | TBD | TBD | TBD | — |
| sft_generic_aggr | TBD | TBD | TBD | TBD |
| sft_generic_cons | TBD | TBD | TBD | TBD |
| dpo_b01 / dpo_b03 | TBD | TBD | TBD | TBD |

---

## Why v3.1 (history)

* **v2** trained on all 164 HumanEval canonical solutions and then evaluated on HumanEval, and compared against a base model. Its numbers are not meaningful.
* **v3** removed the leak but the audit for v3.1 found further problems (see [report/archive/ERRATA.md](report/archive/ERRATA.md) and the plan, section 1):
  the evaluation executed raw chat output with no code extraction; thinking mode was inconsistent between eval and SFT text;
  the "targeted" data embedded HumanEval solutions in its generator prompts and targeted failures of a *different* model,
  and four of the eight targeted task ids pointed at the wrong HumanEval problems; DPO pairs used garbage negatives (temperature up to 1.8).
* **v3.1** fixes the harness, drops targeted data, and adds failure classification and paired statistics.

---

## Project Structure

```
.
├── configs/
│   ├── sft_config*.yaml / dpo_config*.yaml   # Colab A100/H100 profiles (batch size, precision, checkpoint cadence)
│   ├── ablation_matrix.yaml                  # SFT experiments: owns r / alpha / lr / epochs
│   ├── mbpp_plus_task_ids.json               # 378 held-out MBPP+ ids (never used for training data)
│   └── mbpp_dpo_pool_ids.json                # 592 MBPP tasks eligible for DPO pairs
├── scripts/
│   ├── codegen_common.py        # prompt, code extraction, sandbox execution, failure taxonomy (shared)
│   ├── model_io.py              # bf16 model/adapter loading + batched generation (unsloth | hf | vllm)
│   ├── run_eval.py              # HumanEval(+) / MBPP+ evaluation, per-problem status, trust gate
│   ├── compare_runs.py          # paired bootstrap CI, McNemar, fixed/broken problems (CPU)
│   ├── prepare_sft_data.py      # Magicoder + Evol-CodeAlpaca, filtered and contamination-checked
│   ├── check_contamination.py   # pre-training gate (name, n-gram containment vs HumanEval/MBPP+)
│   ├── sft_train.py / dpo_train.py
│   ├── generate_dpo_pairs.py    # execution-feedback pairs: hard negatives, length-matched, split by task
│   ├── train_utils.py, data_stats.py, validate_pipeline.py
├── tests/                       # offline unit tests (no GPU)
├── notebooks/colab_train_a100_h100.ipynb   # one stage per Colab session
└── report/                      # plans, results, archive/ (v2 reports + ERRATA)
```

---

## Setup (local, CPU)

```bash
cp .env.example .env            # HF_TOKEN, WANDB_API_KEY (optional)
uv sync --extra dev             # Python 3.12; pytest, jinja2
uv sync --extra dev --extra eval   # + evalplus (HumanEval+/MBPP+). evalplus only *runs* on Linux (Colab/CI)
uv run pytest -q
```

GPU stages run on Colab; training extras (Unsloth, TRL, PEFT) are installed by the notebook.

---

## Pipeline

### Phase 0 — data (local)

```bash
uv run python scripts/prepare_sft_data.py --magicoder-n 2000 --evol-n 1500
uv run python scripts/check_contamination.py      # hard gate; must print PASSED
```

Upload `data/processed/sft_*.jsonl` to Drive (`DATA_SOURCE_DIR` in the notebook).

### Phase 1 — baseline (Colab session 1)

`run_eval.py` prints a **trust gate**: token-limit hits ≤ 1%, no-code + syntax errors ≤ 2%, our executor agrees with evalplus.
Do not compare any numbers until it passes.

```bash
python scripts/run_eval.py --model unsloth/Qwen3.5-4B --label base --dataset humaneval
python scripts/run_eval.py --model unsloth/Qwen3.5-4B --label base --dataset mbpp
```

### Phase 2 — SFT (Colab session 2)

```bash
python scripts/sft_train.py --config-path configs/sft_config_colab_a100.yaml --exp-id sft_generic_aggr --resume-from auto
python scripts/sft_train.py --config-path configs/sft_config_colab_a100.yaml --exp-id sft_generic_cons --resume-from auto
```

Set `CODETUNE_OUTPUT_ROOT` to a Drive folder so checkpoints survive disconnects (the notebook does this).

### Phase 3 — DPO (Colab session 3)

```bash
python scripts/generate_dpo_pairs.py --sft-checkpoint base          # or an SFT adapter dir
python scripts/dpo_train.py --config-path configs/dpo_config_colab_a100.yaml --sft-checkpoint base --exp-id dpo_b01 --beta 0.1
python scripts/dpo_train.py --config-path configs/dpo_config_colab_a100.yaml --sft-checkpoint base --exp-id dpo_b03 --beta 0.3
```

`generate_dpo_pairs.py` prints a stage-3 gate (≥ 300 pairs, median length ratio 0.8–1.25, wrong-answer ≥ 50% of rejected).

### Phase 4 — analysis (local)

```bash
uv run python scripts/compare_runs.py --labels base,sft_generic_cons,dpo_b01,dpo_b03 --baseline base
```

---

## Key Design Decisions

* **One extraction path.** `codegen_common.extract_code` strips think blocks, picks the fenced block that defines the entry point,
  and drops demo calls / `__main__` blocks. Verified to reproduce 164/164 on HumanEval reference solutions (body-only and full-definition replies).
* **Non-thinking mode everywhere.** `render_prompt` always passes `enable_thinking=False`; the SFT training text extends exactly that prompt (unit-tested against the real tokenizer).
* **Failures are classified**: `pass, no_code, truncated, syntax_error, missing_entry_point, runtime_error, wrong_answer, timeout`.
* **DPO pairs are hard-negative only** (`wrong_answer` / `runtime_error`), sampled at T ∈ [0.2, 1.0], length-matched (0.67–1.5), rendered with the inference prompt, split by task id.
* **Held-out sets**: HumanEval(+) and MBPP+ never touch training data; DPO pairs come from MBPP minus every MBPP+ id.
* **A DPO adapter trained on an SFT adapter stores only the DPO delta**; `model_io.adapter_chain` re-applies the SFT adapter first (recorded in `experiment_meta.json`).

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace Hub |
| `WANDB_API_KEY` | Training monitoring (wandb is disabled automatically when unset) |
| `CODETUNE_OUTPUT_ROOT` | Root for checkpoints/eval outputs (e.g. a Drive folder) |

---

## Current Status (2026-09-18)

- [x] v3 audit: evaluation, thinking mode, targeted-data leak, task-id errors, DPO pair quality
- [x] Shared harness (`codegen_common`, `run_eval`, `model_io`) with offline tests
- [x] Execution-feedback DPO pair generation rewritten; MBPP+ held out
- [x] SFT matrix / hyper-parameter resolution fixed; checkpoint cadence and Drive output
- [x] Contamination gate extended (n-gram containment vs HumanEval and MBPP+) and wired into data prep
- [x] Colab notebook rewritten (eval cells, resume, run manifest)
- [ ] Push branch, upload data to Drive (manual)
- [ ] Baseline evaluation + trust gate (Colab)
- [ ] SFT runs and evaluation (Colab)
- [ ] Preference pairs, DPO, evaluation (Colab)
- [ ] Paired analysis and report
