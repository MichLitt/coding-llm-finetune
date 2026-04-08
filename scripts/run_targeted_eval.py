"""Targeted evaluation on the 8 Coder-Agent failure cases.

Compares base model / SFT / DPO on exactly the problems where the
Coder Agent previously failed, and prints a per-task comparison table.

Usage:
    uv run python scripts/run_targeted_eval.py \
        --models "base=unsloth/Qwen3.5-4B-Instruct,sft=results/sft_checkpoints/sft_targeted/final,dpo=results/dpo_checkpoints/dpo_v1/final"

    # Or compare previously computed eval results (no model loading):
    uv run python scripts/run_targeted_eval.py --from-results
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import click
from datasets import load_dataset

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results/eval"

# The 8 tasks where Coder Agent failed
TARGETED_TASK_IDS = {
    "HumanEval/54":  "same_chars",
    "HumanEval/55":  "fib",
    "HumanEval/107": "minSubArraySum",
    "HumanEval/108": "intersection",
    "HumanEval/110": "prod_signs",
    "HumanEval/128": "tri",
    "HumanEval/141": "file_name_check",
    "HumanEval/147": "get_max_triples",
}

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given function. "
    "Write only the function body — no extra explanation."
)


def _load_model(model_path: str):
    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed.")
        sys.exit(1)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=1536, load_in_4bit=True,
    )
    from unsloth import FastLanguageModel as FLM
    FLM.for_inference(model)
    return model, tokenizer


def _generate(model, tokenizer, prompt: str) -> str:
    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Complete this Python function:\n\n{prompt}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _run_test(prompt: str, completion: str, test_code: str, entry_point: str) -> bool:
    full = prompt + completion + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(tmp).unlink(missing_ok=True)


def _eval_model_on_targeted(model_path: str, he_problems: dict) -> dict[str, bool]:
    """Evaluate a model on the 8 targeted tasks. Returns {task_id: passed}."""
    model, tokenizer = _load_model(model_path)
    results = {}
    for task_id, fn_name in TARGETED_TASK_IDS.items():
        prob = he_problems.get(task_id)
        if not prob:
            results[task_id] = None
            continue
        completion = _generate(model, tokenizer, prob["prompt"])
        passed = _run_test(prob["prompt"], completion, prob["test"], prob["entry_point"])
        results[task_id] = passed
        marker = "PASS" if passed else "FAIL"
        print(f"  {task_id} ({fn_name}): {marker}")
    return results


def _load_from_results(label: str) -> dict[str, bool] | None:
    """Load targeted results from a previously saved humaneval_{label}.json."""
    path = RESULTS_DIR / f"humaneval_{label}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    per = data.get("per_problem", {})
    return {tid: per[tid]["passed"] for tid in TARGETED_TASK_IDS if tid in per}


@click.command()
@click.option("--models", default=None,
              help="Comma-separated label=model_path pairs, e.g. 'base=Qwen/...,sft=results/...'")
@click.option("--from-results", is_flag=True,
              help="Load results from previously saved humaneval_*.json files instead of running models.")
@click.option("--result-labels", default="base,sft_generic,sft_targeted,dpo_v1", show_default=True,
              help="Labels to load when using --from-results.")
def main(models: str | None, from_results: bool, result_labels: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load HumanEval problems ----
    print("Loading HumanEval dataset …")
    he_ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    he_problems = {p["task_id"]: p for p in he_ds}

    all_results: dict[str, dict[str, bool]] = {}

    if from_results:
        for label in result_labels.split(","):
            label = label.strip()
            r = _load_from_results(label)
            if r is None:
                print(f"WARNING: results/eval/humaneval_{label}.json not found, skipping.")
            else:
                all_results[label] = r
    elif models:
        for entry in models.split(","):
            entry = entry.strip()
            if "=" not in entry:
                print(f"WARNING: invalid entry '{entry}', expected label=path")
                continue
            label, model_path = entry.split("=", 1)
            print(f"\nEvaluating [{label}] on 8 targeted tasks …")
            results = _eval_model_on_targeted(model_path, he_problems)
            all_results[label] = results
    else:
        print("ERROR: provide --models or --from-results")
        sys.exit(1)

    if not all_results:
        print("No results to display.")
        return

    # ---- Print comparison table ----
    labels = list(all_results.keys())
    col_w = 8

    print(f"\n{'='*70}")
    print("Targeted Failure Evaluation — 8 Coder-Agent Failure Cases")
    print(f"{'='*70}")

    # Header
    header = f"{'Task ID':<20} {'Function':<20}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    # Rows
    for task_id, fn_name in TARGETED_TASK_IDS.items():
        row = f"{task_id:<20} {fn_name:<20}"
        for label in labels:
            val = all_results[label].get(task_id)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{'PASS' if val else 'FAIL':>{col_w}}"
        print(row)

    # Totals
    print("-" * len(header))
    total_row = f"{'TOTAL':.<40}"
    for label in labels:
        passed = sum(1 for v in all_results[label].values() if v)
        total_row += f"{f'{passed}/8':>{col_w}}"
    print(total_row)
    print(f"{'='*70}")

    # ---- Save summary ----
    summary = {
        "labels": labels,
        "tasks": TARGETED_TASK_IDS,
        "results": {
            label: {tid: all_results[label].get(tid) for tid in TARGETED_TASK_IDS}
            for label in labels
        },
    }
    out_path = RESULTS_DIR / "targeted_eval_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
