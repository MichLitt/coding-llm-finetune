# CodeTune v3: SFT + DPO Post-Training Pipeline

A supervised fine-tuning (SFT) and direct preference optimization (DPO) pipeline for code generation, targeting known failure patterns in a real-world coding agent.

**Base model**: Qwen3.5-4B-Instruct
**Hardware**: Local RTX 5070 8GB (full training + evaluation)
**Key design**: No benchmark data in training set — HumanEval is held out as a clean evaluation target

---

## Results

> v3 training in progress. See [report/CodeTune_v3_Project_Plan.md](report/CodeTune_v3_Project_Plan.md) for the current plan.

| Model | HumanEval pass@1 | Targeted (8 cases) | vs Baseline |
|-------|------------------|--------------------|-------------|
| Qwen3.5-4B-Instruct (baseline) | TBD | TBD | — |
| sft_generic | TBD | TBD | TBD |
| sft_targeted | TBD | TBD | TBD |
| dpo_v1 | TBD | TBD | TBD |

The 8 targeted failure cases are real logic errors from a separate Coder Agent project (HumanEval/54, 55, 107, 108, 110, 128, 141, 147).

---

## v2 → v3 Changes

v2 had two critical methodology errors that invalidated its results:

1. **HumanEval data leakage**: v2 included all 164 HumanEval canonical solutions in SFT training data, then evaluated on the same problems. The 37.8% → 76.2% improvement was not meaningful.
2. **Unfair baseline**: v2 used `Qwen3.5-9B` (base, non-instruct) as the baseline, evaluated via chat template — structurally disadvantaged compared to the SFT model.

v3 fixes both: HumanEval is removed from training data entirely, and the baseline is the same `Qwen3.5-4B-Instruct` model evaluated zero-shot.

See [report/archive/](report/archive/) for v2 reports.

---

## Project Structure

```
.
├── configs/
│   ├── sft_config.yaml          # SFT hyperparameters (LoRA r=32, 4B model)
│   ├── dpo_config.yaml          # DPO hyperparameters (LoRA r=16, beta=0.1)
│   └── ablation_matrix.yaml     # SFT experiment variants
├── scripts/
│   ├── prepare_sft_data.py      # Download + clean + deduplicate (no HumanEval)
│   ├── check_contamination.py   # Pre-training benchmark contamination check
│   ├── generate_targeted_data.py # Targeted failure-pattern data via MiniMax API
│   ├── data_stats.py            # Dataset statistics
│   ├── sft_train.py             # SFT training
│   ├── generate_dpo_pairs.py    # DPO pairs from MBPP (execution-driven)
│   ├── dpo_train.py             # DPO training
│   ├── run_humaneval.py         # HumanEval benchmark evaluation
│   └── run_targeted_eval.py     # Targeted failure case comparison
├── report/
│   ├── CodeTune_v3_Project_Plan.md
│   └── archive/                 # v2 reports (methodology issues documented)
├── .env.example
└── pyproject.toml
```

---

## Setup

```bash
cp .env.example .env
# Edit .env: MINIMAX_API_KEY, HF_TOKEN, WANDB_API_KEY

uv sync
uv sync --extra train   # for training dependencies
```

---

## Pipeline

### Phase 0: Data Preparation

```bash
# Magicoder + EvolCodeAlpaca only — HumanEval excluded by design
uv run python scripts/prepare_sft_data.py --sources magicoder,evol

# Generate targeted data for 8 known failure patterns
uv run python scripts/generate_targeted_data.py

# Verify no benchmark contamination before training
uv run python scripts/check_contamination.py
```

Final dataset: ~3,700 samples (2K magicoder + 1.5K evol + 200 targeted)

### Phase 1: SFT Training (Local RTX 5070)

```bash
uv run python scripts/sft_train.py --experiment sft_targeted
uv run python scripts/sft_train.py --experiment sft_generic   # ablation
```

Key config: LoRA r=32, alpha=64, 3 epochs, lr=2e-4, 4-bit quantization

### Phase 2: DPO Training (Local)

```bash
# Generate pairs from MBPP (not HumanEval — keeping eval set clean)
uv run python scripts/generate_dpo_pairs.py --source mbpp --target-pairs 500

uv run python scripts/dpo_train.py --sft-checkpoint results/sft_targeted/final
```

Key config: LoRA r=16, beta=0.1, execution-driven pairs (MBPP unit tests)

### Phase 3: Evaluation

```bash
# HumanEval (clean — not in training data)
uv run python scripts/run_humaneval.py --model results/sft_targeted/final --label sft_targeted

# Targeted failure case comparison
uv run python scripts/run_targeted_eval.py
```

---

## Key Design Decisions

**Why remove HumanEval from training data?** v2 included HumanEval canonical solutions in SFT training and then evaluated on the same problems — direct data leakage. v3 keeps HumanEval as a held-out clean benchmark.

**Why MBPP for DPO pairs?** MBPP prompts have unit tests for execution-driven pair construction, and are independent from the HumanEval evaluation set. This keeps both the DPO training signal and the final evaluation clean.

**Why 4B instead of 9B?** RTX 5070 8GB can run 4-bit LoRA training for 4B (~5GB VRAM), making full local iteration possible — no Colab dependency. Faster iteration enables the 500+ DPO pairs that v2 never achieved.

**Why Qwen3.5-4B-Instruct as baseline?** Comparing an instruction-tuned SFT model against a base model via chat template (as v2 did) is unfair — the improvement reflects instruction following, not code ability. v3 uses the same instruct model as a zero-shot baseline.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `MINIMAX_API_KEY` | MiniMax API for targeted data generation |
| `MINIMAX_BASE_URL` | MiniMax endpoint (OpenAI-compatible) |
| `HF_TOKEN` | HuggingFace Hub |
| `WANDB_API_KEY` | Training monitoring |

---

## Current Status (2026-04-07)

- [x] v2 methodology issues identified and documented
- [x] v3 plan written
- [x] Fix `prepare_sft_data.py` (HumanEval source removed)
- [x] Write `check_contamination.py` (Jaccard n-gram + function name matching)
- [x] Update configs for 4B model (sft_config.yaml, dpo_config.yaml, ablation_matrix.yaml)
- [x] Update `generate_dpo_pairs.py` to use MBPP (execution-driven pairs)
- [x] Update `sft_train.py` with `train_on_responses_only`
- [x] Update `dpo_train.py` with 4-bit + adapter detection
- [x] Update `validate_pipeline.py` to use Qwen3.5-4B-Instruct
- [ ] Generate SFT data (Phase 0)
- [ ] Run SFT training experiments (Phase 1)
- [ ] Generate MBPP DPO pairs (Phase 2)
- [ ] Run DPO training (Phase 2)
- [ ] Run HumanEval + targeted evaluation (Phase 3)
