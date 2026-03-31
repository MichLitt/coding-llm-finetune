"""HumanEval evaluation script — runs locally with 4-bit quantized model.

Evaluates one or more model checkpoints on HumanEval (164 problems)
and saves per-problem pass/fail results.

Usage:
    uv run python scripts/run_humaneval.py --model unsloth/Qwen3.5-9B --label base
    uv run python scripts/run_humaneval.py --model results/sft_checkpoints/sft_C_with_targeted/final --label sft_C
    uv run python scripts/run_humaneval.py --model results/dpo_checkpoints/dpo_v1/final --label dpo_v1
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results/eval"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given function. "
    "Write only the function body — no extra explanation, no test code."
)


BASE_MODEL = "unsloth/Qwen3.5-9B"


def _is_adapter_only(model_path: str) -> bool:
    """Check if path contains only a LoRA adapter (no config.json)."""
    from pathlib import Path as _Path
    import json as _json
    p = _Path(model_path)
    if p.exists():
        return not (p / "config.json").exists()
    # HF repo: check for adapter_config.json via hub
    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(model_path))
        return "adapter_config.json" in files and "config.json" not in files
    except Exception:
        return False


def _load_model(model_path: str):
    """Load model with 4-bit quantization for local inference."""
    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. pip install unsloth[colab-new] -q")
        sys.exit(1)

    if _is_adapter_only(model_path):
        # LoRA adapter only — load base model first, then merge adapter
        print(f"Detected LoRA adapter, loading base model {BASE_MODEL} + adapter …")
        from peft import PeftModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=1536,
            load_in_4bit=True,
            dtype=None,
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=1536,
            load_in_4bit=True,
            dtype=None,
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a single completion for the HumanEval prompt."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Complete this Python function:\n\n{prompt}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return completion


def _run_tests(prompt: str, completion: str, test_code: str, entry_point: str, timeout: int) -> dict:
    """Execute solution + tests in a subprocess sandbox."""
    full_code = prompt + completion + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        return {
            "passed": passed,
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:500],
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"passed": False, "stdout": "", "stderr": str(e)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@click.command()
@click.option("--model", required=True, help="Model path or HuggingFace model ID.")
@click.option("--label", required=True, help="Short name for this run (used in output filename).")
@click.option("--max-new-tokens", default=512, show_default=True)
@click.option("--timeout", default=10, show_default=True, help="Per-problem execution timeout (seconds).")
@click.option("--subset", default=None, help="Comma-separated HumanEval problem IDs to evaluate (e.g. 54,55). Default: all.")
def main(model: str, label: str, max_new_tokens: int, timeout: int, subset: str | None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load HumanEval ----
    print("Loading HumanEval dataset …")
    he_ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    problems = list(he_ds)

    if subset:
        ids = {int(x) for x in subset.split(",")}
        problems = [p for p in problems if int(p["task_id"].split("/")[-1]) in ids]
        print(f"Evaluating subset: {[p['task_id'] for p in problems]}")

    # ---- Load model ----
    print(f"Loading model: {model}")
    lm, tokenizer = _load_model(model)

    # ---- Evaluate ----
    results = []
    passed_count = 0

    for prob in tqdm(problems, desc=f"Evaluating [{label}]", unit="problem"):
        task_id     = prob["task_id"]
        prompt      = prob["prompt"]
        test_code   = prob["test"]
        entry_point = prob["entry_point"]

        completion  = _generate_completion(lm, tokenizer, prompt, max_new_tokens)
        test_result = _run_tests(prompt, completion, test_code, entry_point, timeout)

        passed = test_result["passed"]
        if passed:
            passed_count += 1

        results.append({
            "task_id":    task_id,
            "passed":     passed,
            "completion": completion[:800],
            "stderr":     test_result.get("stderr", "")[:300],
        })

    # ---- Summary ----
    total = len(results)
    pass_at_1 = passed_count / total if total else 0

    summary = {
        "label":      label,
        "model":      model,
        "total":      total,
        "passed":     passed_count,
        "pass_at_1":  pass_at_1,
        "failed_ids": [r["task_id"] for r in results if not r["passed"]],
        "per_problem": {r["task_id"]: {"passed": r["passed"], "stderr": r["stderr"]} for r in results},
    }

    out_path = RESULTS_DIR / f"humaneval_{label}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"Model: {label}")
    print(f"pass@1: {passed_count}/{total} = {pass_at_1:.1%}")
    print(f"Failed: {summary['failed_ids']}")
    print(f"Saved:  {out_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
