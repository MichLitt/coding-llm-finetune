# AGENTS.md

When this repository is checked out inside `agent-systems-portfolio`, also read
`../AGENTS.md` and `../docs/engineering/ENGINEERING_GUIDE.md`. This file remains
the complete local entry point for a standalone clone. The project is the
post-training extension and is not part of the completed Agent/RAG/EvalOps
closure.

## Commands

- Install pipeline dependencies: `uv sync && cp .env.example .env`
- Install training extras: `uv sync --extra train` (install Unsloth separately as documented in the README)
- Prepare data: `uv run python scripts/prepare_sft_data.py --sources magicoder,evol`
- Contamination gate: `uv run python scripts/check_contamination.py`
- SFT: `uv run python scripts/sft_train.py --exp-id sft_targeted`
- DPO pairs: `uv run python scripts/generate_dpo_pairs.py --sft-checkpoint results/sft_targeted/final --num-problems 250`
- DPO: `uv run python scripts/dpo_train.py --sft-checkpoint results/sft_targeted/final --exp-id dpo_v1`

## Current Status

Pipeline/config preparation is complete, but v3 data generation, training,
checkpoints, and final evaluation are not complete. Do not replace README TBDs
with results unless matching artifacts exist.

## Methodology Invariants

- HumanEval is held-out evaluation data and must never enter SFT or DPO training.
- DPO pairs use MBPP or independently verified failure cases, not the final benchmark.
- Baseline and trained model must use the same base/instruct family and evaluation harness.
- Contamination check is a hard pre-training gate.
- Report zero-shot baseline, SFT generic, SFT targeted, and DPO separately.
- Preserve raw configs, seeds, environment/hardware metadata, and evaluation artifacts.

## Environment and Artifacts

- `MINIMAX_API_KEY` — optional targeted-data generation
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
