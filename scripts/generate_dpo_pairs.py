"""Generate DPO preference pairs via MBPP execution-driven scoring.

Prompt source  : MBPP test split (374 problems with built-in unit tests)
                 HumanEval is NOT used — it is held out as the evaluation benchmark
Scoring method : Unit test execution (pass = chosen, fail = rejected)
                 No LLM judge — avoids judge bias, directly optimizes eval metric

Pair construction:
  - For each MBPP problem, generate N candidates at mixed temperatures
  - Run MBPP assert-based unit tests on each candidate
  - Same problem must have ≥1 pass and ≥1 fail to yield a valid pair
  - chosen  = shortest passing candidate
  - rejected = random failing candidate

Target: 500+ valid pairs from ~250 MBPP problems (40-60% yield expected)

Usage:
    uv run python scripts/generate_dpo_pairs.py \\
        --sft-checkpoint results/sft_checkpoints/sft_targeted/final
    uv run python scripts/generate_dpo_pairs.py \\
        --sft-checkpoint <path> --num-problems 300 --n-candidates 8
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import click
from tqdm import tqdm

ROOT = Path(__file__).parent.parent

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write clean, efficient, and correct code. "
    "Always handle edge cases and include brief comments for complex logic."
)

TARGETED_TASK_IDS = [
    "HumanEval/54", "HumanEval/55", "HumanEval/107", "HumanEval/108",
    "HumanEval/110", "HumanEval/128", "HumanEval/141", "HumanEval/147",
]


# ---------------------------------------------------------------------------
# Execution-based scoring
# ---------------------------------------------------------------------------

def _run_tests(code: str, test_list: list[str], setup_code: str, timeout: int) -> bool:
    """Execute candidate code + MBPP assert tests. Returns True if all pass."""
    full_code = ""
    if setup_code:
        full_code += setup_code.strip() + "\n\n"
    full_code += code.strip() + "\n\n"
    full_code += "\n".join(test_list) + "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_candidates(
    model,
    tokenizer,
    problem_text: str,
    temperatures: list[float],
    max_new_tokens: int,
) -> list[str]:
    """Generate one candidate per temperature."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Write a Python function to solve the following:\n\n{problem_text}"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=768).to(model.device)

    candidates = []
    for temp in temperatures:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temp > 0),
                temperature=temp if temp > 0 else None,
                top_p=0.95 if temp > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        candidates.append(generated)

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--sft-checkpoint", required=True,
              help="Path to SFT LoRA adapter or full model checkpoint.")
@click.option("--num-problems", default=250, show_default=True,
              help="Number of MBPP problems to sample.")
@click.option("--n-candidates", default=8, show_default=True,
              help="Candidates to generate per problem.")
@click.option("--temperatures", default="0.2,0.5,0.8,1.0,1.2,1.4,1.6,1.8", show_default=True,
              help="Comma-separated temperatures (one candidate each; truncated to n-candidates).")
@click.option("--max-new-tokens", default=512, show_default=True)
@click.option("--timeout", default=10, show_default=True,
              help="Per-problem test execution timeout in seconds.")
@click.option("--val-ratio", default=0.1, show_default=True)
@click.option("--seed", default=42, show_default=True)
def main(
    sft_checkpoint: str,
    num_problems: int,
    n_candidates: int,
    temperatures: str,
    max_new_tokens: int,
    timeout: int,
    val_ratio: float,
    seed: int,
):
    random.seed(seed)
    temps = [float(t) for t in temperatures.split(",")][:n_candidates]

    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip install unsloth[colab-new] -q")
        sys.exit(1)

    # ---- Load MBPP ----
    print("Loading MBPP test split …")
    from datasets import load_dataset
    mbpp_ds = load_dataset("google-research-datasets/mbpp", "full", split="test", trust_remote_code=True)
    problems = list(mbpp_ds)
    random.shuffle(problems)
    problems = problems[:num_problems]
    print(f"Sampled {len(problems)} MBPP problems")

    # ---- Load SFT model ----
    ckpt_path = Path(sft_checkpoint)
    base_model = "unsloth/Qwen3.5-4B-Instruct"

    if ckpt_path.exists() and not (ckpt_path / "config.json").exists():
        print(f"Loading base model {base_model} + LoRA adapter {ckpt_path} (4-bit) …")
        from peft import PeftModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=1536, load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, str(ckpt_path))
    else:
        print(f"Loading model {sft_checkpoint} (4-bit) …")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_checkpoint, max_seq_length=1536, load_in_4bit=True,
        )
    FastLanguageModel.for_inference(model)

    # ---- Generate pairs ----
    pairs: list[dict] = []
    stats = {"total": 0, "valid_pairs": 0, "all_pass": 0, "all_fail": 0, "skipped": 0}

    for prob in tqdm(problems, desc="Generating pairs", unit="problem"):
        task_id    = str(prob.get("task_id", ""))
        text       = prob.get("text", "").strip()
        test_list  = prob.get("test_list", [])
        setup_code = prob.get("test_setup_code", "") or ""

        if not text or not test_list:
            stats["skipped"] += 1
            continue

        stats["total"] += 1
        candidates = _generate_candidates(model, tokenizer, text, temps, max_new_tokens)

        passed, failed = [], []
        for code in candidates:
            if _run_tests(code, test_list, setup_code, timeout):
                passed.append(code)
            else:
                failed.append(code)

        if passed and failed:
            chosen   = random.choice(passed)          # random passing solution (avoid length bias)
            rejected = random.choice(failed)
            pairs.append({
                "prompt": f"Write a Python function to solve the following:\n\n{text}",
                "chosen": chosen,
                "rejected": rejected,
                "source": "mbpp",
                "task_id": task_id,
            })
            stats["valid_pairs"] += 1
        elif passed:
            stats["all_pass"] += 1
        else:
            stats["all_fail"] += 1

    # ---- Save ----
    random.shuffle(pairs)
    val_n       = max(1, int(len(pairs) * val_ratio))
    val_pairs   = pairs[:val_n]
    train_pairs = pairs[val_n:]

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dpo_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in train_pairs) + "\n",
        encoding="utf-8",
    )
    (out_dir / "dpo_pairs_val.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in val_pairs) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'='*50}")
    print(f"Problems processed : {stats['total']}")
    print(f"Valid pairs        : {stats['valid_pairs']}  ({stats['valid_pairs']/max(stats['total'],1):.0%} yield)")
    print(f"All passed         : {stats['all_pass']}")
    print(f"All failed         : {stats['all_fail']}")
    print(f"Skipped            : {stats['skipped']}")
    print(f"Train pairs        : {len(train_pairs)}")
    print(f"Val pairs          : {len(val_pairs)}")
    print(f"Saved to           : {out_dir}/dpo_pairs.jsonl")
    print(f"{'='*50}\n")

    if len(train_pairs) < 200:
        print("WARNING: fewer than 200 train pairs. Consider --num-problems 350 or lowering --n-candidates threshold.")


if __name__ == "__main__":
    main()
