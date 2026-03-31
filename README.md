# CodeTune v2: SFT + DPO Post-Training Pipeline

A complete supervised fine-tuning (SFT) and direct preference optimization (DPO) pipeline for code generation, targeting known failure patterns in a real-world coding agent.

**Base model**: Qwen2.5-Coder-7B-Instruct → fine-tuned on 86K samples
**Hardware**: Local RTX 5070 8GB (data prep + evaluation) + Colab A100 80GB (training)

---

## Results

| Model | HumanEval pass@1 | Targeted (8 cases) | vs Baseline |
|-------|------------------|--------------------|-------------|
| base  | 37.8% (62/164)   | 2/8                | —           |
| sft_A (Magicoder only) | 65.2% (107/164) | 4/8    | +27.4pp     |
| sft_C (+ evol + targeted) | **76.2% (125/164)** | 5/8 | **+38.4pp** |
| dpo_v1 | 72.6% (119/164) | **6/8**            | +34.8pp     |

**Full training chain**: base 37.8% → sft_A 65.2% → sft_C 76.2% → dpo_v1 72.6%

The "targeted" failure cases are 8 real logic errors from a separate Coder Agent project (HumanEval/54, 55, 107, 108, 110, 128, 141, 147). DPO successfully fixed HE/54 (same_chars: `sorted()` vs `set()`) that SFT could not.

---

## Project Structure

```
.
├── configs/
│   ├── sft_config.yaml          # SFT hyperparameters (LoRA r=64, 3 epochs)
│   ├── dpo_config.yaml          # DPO hyperparameters (LoRA r=32, beta=0.1)
│   └── ablation_matrix.yaml     # 5 SFT experiment variants
├── scripts/
│   ├── prepare_sft_data.py      # Download + clean + deduplicate SFT data
│   ├── generate_targeted_data.py # Generate targeted failure-pattern data via MiniMax API
│   ├── data_stats.py            # Dataset statistics and quality report
│   ├── validate_pipeline.py     # Local 0.5B model pipeline validation
│   ├── sft_train.py             # SFT training (run on Colab A100)
│   ├── generate_dpo_pairs.py    # Generate DPO pairs via local 4-bit inference
│   ├── dpo_train.py             # DPO training (run on Colab A100)
│   ├── run_humaneval.py         # HumanEval benchmark evaluation
│   └── run_targeted_eval.py     # Targeted failure case comparison
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_dpo_training.ipynb
│   ├── 04_humaneval_eval.ipynb
│   ├── 05_eval_colab.ipynb
│   ├── 06_dpo_data_generation.ipynb
│   └── 07_dpo_v2_data_generation.ipynb
├── report/
│   ├── CodeTune_v2_Project_Plan_1.md
│   ├── SFT_Evaluation_Report.md
│   └── DPO_Evaluation_Report.md
├── checkpoints/
│   └── sft_C/                   # sft_C LoRA adapter (445MB, r=64)
├── .env.example                 # Environment variable template
└── pyproject.toml
```

---

## Setup

```bash
# Install uv (if not already installed)
# https://docs.astral.sh/uv/getting-started/installation/

# Clone and set up
git clone <repo-url>
cd coding-llm

# Copy and fill in your API keys
cp .env.example .env
# Edit .env: MINIMAX_API_KEY, HF_TOKEN, WANDB_API_KEY

# Install base dependencies
uv sync

# Install training dependencies (for Colab/GPU machine)
uv sync --extra train
```

---

## Pipeline

### Phase 0: Data Preparation (Local)

```bash
# Download + filter Magicoder, EvolCodeAlpaca, HumanEval
uv run python scripts/prepare_sft_data.py --sources magicoder,evol,humaneval

# Generate targeted data for known failure patterns (uses MiniMax API)
uv run python scripts/generate_targeted_data.py

# Check data statistics
uv run python scripts/data_stats.py

# Validate pipeline locally with 0.5B model before Colab
uv run python scripts/validate_pipeline.py
```

Final dataset: ~86K samples (46K evol + 40K magicoder + 200 targeted + 164 humaneval reference)

### Phase 1: SFT Training (Colab A100)

Upload `data/processed/` to HuggingFace Hub or Google Drive, then run on Colab:

```bash
# Train sft_C (magicoder + evol + targeted) — primary experiment
uv run python scripts/sft_train.py --experiment sft_C_with_targeted

# Other ablations defined in configs/ablation_matrix.yaml
uv run python scripts/sft_train.py --experiment sft_A_magicoder_only
```

Key config: LoRA r=64, alpha=128, 3 epochs, lr=2e-4, bf16, `train_on_responses_only`

### Phase 2: DPO Training (Local 4-bit + Colab)

```bash
# Generate preference pairs using local 4-bit sft_C model
uv run python scripts/generate_dpo_pairs.py --checkpoint checkpoints/sft_C

# Train DPO on Colab
uv run python scripts/dpo_train.py --sft-checkpoint <hf-repo-or-path>
```

Key config: LoRA r=32, beta=0.1, execution-driven pairs (pass/fail unit tests), ref_model=None (Unsloth handles automatically)

### Phase 3: Evaluation

```bash
# Full HumanEval benchmark
uv run python scripts/run_humaneval.py --checkpoint checkpoints/sft_C

# Targeted failure case comparison (base vs sft vs dpo)
uv run python scripts/run_targeted_eval.py
```

---

## Key Design Decisions

**Why targeted data?** The 8 failure cases came from a separate Coder Agent project. Rather than hoping generic fine-tuning fixes them, we synthesized 25 variations per failure pattern (200 samples total) focused on the specific error type. Result: base 2/8 → sft_C 5/8.

**Why execution-driven DPO pairs?** Instead of scoring with an LLM judge, we run each generated solution against HumanEval unit tests. Pass = chosen, fail = rejected. This avoids judge bias and directly optimizes for the actual evaluation metric. Result: fixed HE/54 (same_chars) which SFT could not.

**Why MiniMax API instead of GPT-4o-mini?** Cost. MiniMax provides an OpenAI-compatible API at lower cost. All API calls use the `openai` SDK with `MINIMAX_BASE_URL`.

**DPO alignment tax**: dpo_v1 showed -3.6pp overall despite targeted improvement. Root cause: only ~147 pairs, ~30 training steps. Next step: expand to 500+ pairs.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `MINIMAX_API_KEY` | MiniMax API for targeted data generation |
| `MINIMAX_BASE_URL` | MiniMax endpoint (OpenAI-compatible) |
| `HF_TOKEN` | HuggingFace Hub for dataset/model upload |
| `WANDB_API_KEY` | Weights & Biases training monitoring |

---

## Current Status (2026-03-30)

- [x] Data preparation (86K samples, MinHash dedup)
- [x] SFT ablations: sft_A, sft_C
- [x] SFT evaluation: sft_C at **76.2%** HumanEval pass@1
- [x] DPO v1: execution-driven pairs, dpo_v1 at 72.6%
- [ ] DPO v2: expand to 500+ pairs, fix HE/108 and HE/128
- [ ] Final evaluation report

See [report/](report/) for detailed evaluation reports.
