# AGENTS.md

When this repository is checked out inside `agent-systems-portfolio`, also read
`../AGENTS.md` and `../docs/engineering/ENGINEERING_GUIDE.md`. This file remains
the complete local entry point for a standalone clone. The project is the
post-training extension and is not part of the completed Agent/RAG/EvalOps
closure.

## Commands

- Install: `uv sync --extra dev && cp .env.example .env` (Python 3.12; add `--extra eval` for evalplus, `--extra train` for training extras — install Unsloth separately)
- Offline tests: `uv run pytest -q`
- Prepare data: `uv run python scripts/prepare_sft_data.py --magicoder-n 2000 --evol-n 1500`
- Contamination gate: `uv run python scripts/check_contamination.py`
- Evaluate (GPU): `python scripts/run_eval.py --model <path|id> --label <name> --dataset humaneval|mbpp`
- SFT (GPU): `python scripts/sft_train.py --config-path configs/sft_config_colab_a100.yaml --exp-id sft_generic_aggr|sft_generic_cons`
- DPO pairs (GPU): `python scripts/generate_dpo_pairs.py --sft-checkpoint base|<adapter dir>`
- DPO (GPU): `python scripts/dpo_train.py --config-path configs/dpo_config_colab_a100.yaml --sft-checkpoint base|<adapter dir> --exp-id dpo_b01 --beta 0.1`
- Compare runs (CPU): `uv run python scripts/compare_runs.py --labels base,... --baseline base`

Plan of record: `report/CodeTune_v3.1_Execution_Plan.md`.

## Current Status

Harness, data pipeline, scripts and tests are implemented (v3.1 phase 0). Baseline, SFT, DPO
training and final evaluation have NOT been run. Do not replace README TBDs with results unless
matching artifacts exist.

## Methodology Invariants

- HumanEval(+) and MBPP+ are held-out evaluation data and must never enter SFT or DPO training.
- DPO pairs come from MBPP minus every MBPP+ task id (`configs/mbpp_dpo_pool_ids.json`), never the final benchmark.
- Baseline and trained models use the same base model, the same prompt (`codegen_common`), the same extraction and the same generation settings (bf16, non-thinking, greedy, 1024 new tokens).
- Pass the baseline trust gate (`run_eval.py` output) before comparing any numbers.
- Contamination check is a hard pre-training gate.
- Report baseline, SFT (both settings) and DPO (both betas) separately, with paired confidence intervals.
- Preserve raw configs, seeds, environment/hardware metadata (notebook run manifest) and evaluation artifacts.
- Targeted (Coder-Agent failure) data is retired: it leaked HumanEval solutions and targeted a different model.

## Environment and Artifacts

- `CODETUNE_OUTPUT_ROOT` — optional root for checkpoints and eval outputs (Drive on Colab)
- `HF_TOKEN` — model/dataset access
- `WANDB_API_KEY` — optional tracking
- Never commit keys, checkpoints, weights, generated training corpora, or W&B state.

## EvalOps Boundary

`finetune/v1` remains unsupported in EvalOps. Implementing it requires a real
versioned producer report, consumer schema/adapter, contract tests, and updates
to the root engineering guide. Do not mark it active based on a stub.

## Required Handoff

- Run contamination checks before any training claim.
- Attach results to concrete model/config/evaluation artifacts.
- Update README status and a versioned report after each completed phase.
- `git diff --check` passes.
