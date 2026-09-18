"""Unified evaluation harness (replaces run_humaneval.py).

Every model (base / SFT / DPO) goes through the SAME path: bf16, non-thinking
prompt, greedy decoding, one code-extraction routine, one failure taxonomy.

Usage (Colab/GPU):
    python scripts/run_eval.py --model unsloth/Qwen3.5-4B --label base --dataset humaneval
    python scripts/run_eval.py --model results/sft_checkpoints/sft_x/final --label sft_x --dataset mbpp

Outputs in ``results/eval/<label>/<dataset>/``:
    raw.jsonl          raw model output, token count, finished flag
    samples.jsonl      extracted programs in evalplus format
    per_problem.jsonl  status per task (failure taxonomy) + evalplus base/plus status
    summary.json       pass@1, failure distribution, truncation rate, output length
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

import codegen_common as cc  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results/eval"


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def load_tasks(dataset: str, subset: list[str] | None = None) -> list[dict]:
    """Return tasks as dicts with ``task_id``, ``entry_point``, ``messages`` (+ HumanEval fields)."""
    tasks: list[dict] = []
    if dataset == "humaneval":
        from datasets import load_dataset

        for row in load_dataset("openai/openai_humaneval", split="test"):
            tasks.append({
                "task_id": row["task_id"],
                "entry_point": row["entry_point"],
                "prompt": row["prompt"],
                "test": row["test"],
                "messages": cc.humaneval_messages(row["prompt"]),
            })
    elif dataset == "mbpp":
        from evalplus.data import get_mbpp_plus

        for task_id, row in get_mbpp_plus().items():
            text, first_test = cc.split_mbpp_plus_prompt(row["prompt"])
            tasks.append({
                "task_id": task_id,
                "entry_point": row["entry_point"],
                "messages": cc.mbpp_messages(text, first_test),
            })
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if subset:
        wanted = set(subset)
        tasks = [t for t in tasks if t["task_id"] in wanted or t["task_id"].split("/")[-1] in wanted]
    return tasks


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def judge_generation(dataset: str, task: dict, text: str, finished: bool, timeout: float) -> dict:
    """Judge one generation. For MBPP+, tests run later inside evalplus.

    Returns a per-problem record; ``status`` is None while an MBPP+ candidate
    still awaits evalplus.
    """
    hit_limit = not finished
    if dataset == "humaneval":
        verdict = cc.judge_humaneval(text, task["prompt"], task["test"], task["entry_point"],
                                     hit_token_limit=hit_limit, timeout=timeout)
        return {"status": verdict.status, "solution": verdict.solution, "detail": verdict.detail}

    extraction = cc.extract_code(text, task["entry_point"])
    failure = None
    if not extraction.code:
        failure = cc.NO_CODE
    elif not extraction.parsed_ok:
        failure = cc.SYNTAX_ERROR
    elif not extraction.defines_entry:
        failure = cc.MISSING_ENTRY_POINT
    status = cc.finalize_status(failure, hit_limit) if failure else None
    return {"status": status, "solution": extraction.code, "detail": failure or ""}


def run_evalplus(dataset: str, samples_path: Path) -> dict[str, dict]:
    """Run evalplus and return ``{task_id: {"base": status, "plus": status}}``.

    evalplus applies OS resource limits that only work on Linux (Colab).
    """
    subprocess.run([sys.executable, "-m", "evalplus.evaluate", dataset, "--samples", str(samples_path)],
                   check=True)
    results_path = samples_path.with_name(samples_path.stem + "_eval_results.json")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return {
        task_id: {"base": entries[0]["base_status"], "plus": entries[0]["plus_status"]}
        for task_id, entries in data["eval"].items()
    }


def apply_evalplus(dataset: str, records: list[dict], ep: dict[str, dict]) -> None:
    """Merge evalplus statuses into records (mutates)."""
    for rec in records:
        got = ep.get(rec["task_id"])
        if got is None:
            continue
        rec["evalplus_base"], rec["evalplus_plus"] = got["base"], got["plus"]
        if dataset == "mbpp" and rec["status"] is None:
            # MBPP+ primary metric = base AND plus tests.
            if got["plus"] == "pass" and got["base"] == "pass":
                rec["status"] = cc.PASS
            elif "timeout" in (got["base"], got["plus"]):
                rec["status"] = cc.TIMEOUT
            else:
                rec["status"] = cc.WRONG_ANSWER  # evalplus does not separate assertion vs exception


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(dataset: str, records: list[dict]) -> dict:
    n = len(records)
    statuses = Counter(rec["status"] or "unjudged" for rec in records)
    passed = statuses[cc.PASS]
    summary = {
        "dataset": dataset,
        "total": n,
        "passed": passed,
        "pass_at_1": passed / n if n else 0.0,
        "status_counts": {s: statuses.get(s, 0) for s in (*cc.ALL_STATUSES, "unjudged") if statuses.get(s, 0)},
        "hit_token_limit_rate": sum(rec["hit_token_limit"] for rec in records) / n if n else 0.0,
        "format_failure_rate": (
            sum(statuses[s] for s in (cc.NO_CODE, cc.SYNTAX_ERROR, cc.MISSING_ENTRY_POINT, cc.TRUNCATED)) / n
            if n else 0.0
        ),
        "mean_output_tokens": sum(rec["n_tokens"] for rec in records) / n if n else 0.0,
        "failed_ids": [rec["task_id"] for rec in records if rec["status"] != cc.PASS],
    }
    plus = [rec for rec in records if "evalplus_plus" in rec]
    if plus:
        summary["evalplus_base_pass_at_1"] = sum(r["evalplus_base"] == "pass" for r in plus) / n
        summary["evalplus_plus_pass_at_1"] = sum(r["evalplus_plus"] == "pass" for r in plus) / n
        if dataset == "humaneval":
            # Cross-check: our executor and evalplus must agree on base tests.
            summary["executor_disagreements"] = [
                r["task_id"] for r in plus if (r["status"] == cc.PASS) != (r["evalplus_base"] == "pass")
            ]
    return summary


def eval_gate(summary: dict, max_hit_limit: float = 0.01, max_no_code_syntax: float = 0.02) -> dict[str, bool]:
    """Baseline trust gate (plan stage 1): the harness itself must not create failures.

    All entries must be True before baseline numbers are compared with anything.
    """
    n = summary["total"] or 1
    counts = summary["status_counts"]
    no_code_syntax = (counts.get(cc.NO_CODE, 0) + counts.get(cc.SYNTAX_ERROR, 0)) / n
    return {
        f"hit_token_limit_rate<={max_hit_limit:.0%}": summary["hit_token_limit_rate"] <= max_hit_limit,
        f"no_code+syntax_error<={max_no_code_syntax:.0%}": no_code_syntax <= max_no_code_syntax,
        "executors_agree": not summary.get("executor_disagreements"),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def evaluate(generator, tokenizer, dataset: str, tasks: list[dict], max_new_tokens: int,
             timeout: float) -> tuple[list[dict], list[dict]]:
    """Generate and judge. Returns (records, raw_rows)."""
    prompts = [cc.render_prompt(tokenizer, task["messages"]) for task in tasks]
    outs = generator.generate(prompts, max_new_tokens=max_new_tokens, temperature=0.0)
    records, raw_rows = [], []
    for task, out in zip(tasks, outs):
        rec = {"task_id": task["task_id"], "n_tokens": out.n_tokens, "hit_token_limit": not out.finished}
        rec.update(judge_generation(dataset, task, out.text, out.finished, timeout))
        records.append(rec)
        raw_rows.append({"task_id": task["task_id"], "raw": out.text,
                         "n_tokens": out.n_tokens, "finished": out.finished})
    return records, raw_rows


def write_outputs(out_dir: Path, records: list[dict], raw_rows: list[dict], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict]) -> None:
        (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8")

    dump("raw.jsonl", raw_rows)
    dump("samples.jsonl", [{"task_id": r["task_id"], "solution": r["solution"] or "# no code extracted\n"}
                           for r in records])
    dump("per_problem.jsonl", records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


@click.command()
@click.option("--model", required=True, help="HF repo id, model dir, or LoRA adapter dir.")
@click.option("--label", required=True, help="Run name (output directory).")
@click.option("--dataset", type=click.Choice(["humaneval", "mbpp"]), default="humaneval", show_default=True)
@click.option("--backend", type=click.Choice(["unsloth", "hf", "vllm"]), default="unsloth", show_default=True)
@click.option("--max-new-tokens", default=1024, show_default=True)
@click.option("--timeout", default=10.0, show_default=True, help="Per-problem execution timeout (s).")
@click.option("--batch-size", default=16, show_default=True)
@click.option("--subset", default=None, help="Comma-separated task ids or numbers (debugging).")
@click.option("--skip-evalplus", is_flag=True, help="Skip evalplus (HumanEval+/MBPP+ tests). Local debugging only.")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
def main(model, label, dataset, backend, max_new_tokens, timeout, batch_size, subset, skip_evalplus, results_dir):
    from model_io import load_generator

    tasks = load_tasks(dataset, subset.split(",") if subset else None)
    print(f"[{label}] {dataset}: {len(tasks)} tasks")
    generator = load_generator(model, backend=backend, batch_size=batch_size)
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None:  # vLLM: use the HF tokenizer only for chat templating
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model if not Path(model, "adapter_config.json").exists()
                                                  else "unsloth/Qwen3.5-4B")

    records, raw_rows = evaluate(generator, tokenizer, dataset, tasks, max_new_tokens, timeout)
    out_dir = Path(results_dir) / label / dataset
    write_outputs(out_dir, records, raw_rows, {})

    if not skip_evalplus:
        apply_evalplus(dataset, records, run_evalplus(dataset, out_dir / "samples.jsonl"))
    summary = summarize(dataset, records)
    summary.update({"label": label, "model": model, "backend": backend, "max_new_tokens": max_new_tokens,
                    "decoding": "greedy", "thinking": False, "precision": "bf16", "git_commit": _git_commit()})
    write_outputs(out_dir, records, raw_rows, summary)

    print(f"\n{'=' * 60}\n{label} / {dataset}: pass@1 = {summary['passed']}/{summary['total']} "
          f"= {summary['pass_at_1']:.1%}")
    print(f"status counts       : {summary['status_counts']}")
    print(f"hit token limit     : {summary['hit_token_limit_rate']:.1%}")
    print(f"format failure rate : {summary['format_failure_rate']:.1%}")
    print(f"trust gate          : {eval_gate(summary)}")
    if "evalplus_plus_pass_at_1" in summary:
        print(f"evalplus base/plus  : {summary['evalplus_base_pass_at_1']:.1%} / {summary['evalplus_plus_pass_at_1']:.1%}")
    print(f"saved to {out_dir}\n{'=' * 60}")


if __name__ == "__main__":
    main()
