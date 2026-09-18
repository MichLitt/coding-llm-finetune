"""Generate DPO preference pairs from execution feedback on MBPP.

Pool          : MBPP (all 4 splits, 974 tasks) MINUS every MBPP+ task id (kept as a
                held-out second benchmark) and minus tasks whose reference solution
                fails its own tests -> configs/mbpp_dpo_pool_ids.json (592 tasks).
                HumanEval is never touched.
Candidates    : N samples per task at temperatures in [0.2, 1.0] (no garbage
                high-temperature samples), non-thinking prompt, 1024 new tokens.
Judging       : the shared extraction + sandbox in codegen_common (same code path as
                evaluation): pass / wrong_answer / runtime_error / format failures.
chosen        : a candidate that PASSES and finished cleanly.
rejected      : a HARD negative only - parses, defines the function, runs, but is
                wrong (wrong_answer / runtime_error). Truncated, unparsable, no-code
                and timeout candidates are never used as negatives, otherwise DPO
                mostly learns format/length instead of correctness.
Length control: pairs whose rejected/chosen token ratio falls outside
                [--min-length-ratio, --max-length-ratio] are dropped.
Format        : ``prompt`` is the chat-template-rendered generation prompt (identical
                to the one used at inference, non-thinking), ``chosen``/``rejected``
                are the raw assistant replies.
Split         : train/val are split by task_id.

Usage (GPU):
    python scripts/generate_dpo_pairs.py --sft-checkpoint base
    python scripts/generate_dpo_pairs.py --sft-checkpoint results/sft_checkpoints/sft_x/final
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

import codegen_common as cc  # noqa: E402

ROOT = Path(__file__).parent.parent
POOL_IDS_PATH = ROOT / "configs/mbpp_dpo_pool_ids.json"
MBPP_PLUS_IDS_PATH = ROOT / "configs/mbpp_plus_task_ids.json"


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

def load_pool(num_problems: int = 0, seed: int = 42) -> list[dict]:
    """MBPP tasks eligible for preference-pair generation (never a MBPP+ task)."""
    from datasets import load_dataset

    pool_ids = set(json.loads(POOL_IDS_PATH.read_text(encoding="utf-8"))["task_ids"])
    plus_ids = set(json.loads(MBPP_PLUS_IDS_PATH.read_text(encoding="utf-8"))["task_ids"])
    assert not pool_ids & plus_ids, "DPO pool overlaps MBPP+ (held-out benchmark)"

    tasks = []
    for split in ("train", "validation", "test", "prompt"):
        for row in load_dataset("google-research-datasets/mbpp", "full", split=split):
            if row["task_id"] not in pool_ids:
                continue
            setup = row.get("test_setup_code") or ""
            entry = cc.mbpp_entry_point(row["test_list"], setup)
            if entry is None:
                continue
            tasks.append({
                "task_id": row["task_id"],
                "text": row["text"],
                "test_list": list(row["test_list"]),
                "setup_code": setup,
                "entry_point": entry,
                "messages": cc.mbpp_messages(row["text"], row["test_list"][0], entry),
            })
    tasks.sort(key=lambda t: t["task_id"])
    random.Random(seed).shuffle(tasks)
    return tasks[:num_problems] if num_problems else tasks


# ---------------------------------------------------------------------------
# Pair construction (pure functions, unit-tested)
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def build_pairs_for_task(
    prompt: str,
    task_id: int | str,
    candidates: list[dict],
    max_pairs: int = 2,
    min_ratio: float = 0.67,
    max_ratio: float = 1.5,
) -> tuple[list[dict], dict]:
    """Turn judged candidates of one task into <= ``max_pairs`` preference pairs.

    Each candidate: ``{"text", "n_tokens", "finished", "status", "temperature"}``.
    Returns ``(pairs, info)``; ``info`` explains why no pair was produced.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for cand in candidates:
        key = _norm(cand["text"])
        if key and key not in seen:
            seen.add(key)
            unique.append(cand)

    chosen_pool = [c for c in unique if c["status"] == cc.PASS and c["finished"]]
    reject_pool = [c for c in unique if c["status"] in cc.HARD_NEGATIVE_STATUSES and c["finished"]]
    info = {"n_pass": len(chosen_pool), "n_hard_negative": len(reject_pool), "reason": None}
    if not chosen_pool:
        info["reason"] = "no_passing_candidate"
        return [], info
    if not reject_pool:
        info["reason"] = "no_hard_negative"
        return [], info

    combos = []
    for chosen in chosen_pool:
        for rejected in reject_pool:
            ratio = max(rejected["n_tokens"], 1) / max(chosen["n_tokens"], 1)
            if min_ratio <= ratio <= max_ratio:
                combos.append((abs(math.log(ratio)), ratio, chosen, rejected))
    if not combos:
        info["reason"] = "length_ratio_out_of_range"
        return [], info

    combos.sort(key=lambda item: item[0])  # closest length match first
    pairs, used_chosen, used_rejected = [], set(), set()
    for _, ratio, chosen, rejected in combos:
        if id(chosen) in used_chosen or id(rejected) in used_rejected:
            continue
        used_chosen.add(id(chosen))
        used_rejected.add(id(rejected))
        pairs.append({
            "prompt": prompt,
            "chosen": chosen["text"].strip(),
            "rejected": rejected["text"].strip(),
            "source": "mbpp",
            "task_id": task_id,
            "chosen_tokens": chosen["n_tokens"],
            "rejected_tokens": rejected["n_tokens"],
            "length_ratio": round(ratio, 4),
            "rejected_status": rejected["status"],
            "chosen_temperature": chosen["temperature"],
            "rejected_temperature": rejected["temperature"],
        })
        if len(pairs) >= max_pairs:
            break
    return pairs, info


def split_by_task(pairs: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split so that no task appears in both train and validation."""
    task_ids = sorted({p["task_id"] for p in pairs})
    random.Random(seed).shuffle(task_ids)
    n_val = max(1, int(len(task_ids) * val_ratio)) if len(task_ids) > 1 else 0
    val_ids = set(task_ids[:n_val])
    return ([p for p in pairs if p["task_id"] not in val_ids],
            [p for p in pairs if p["task_id"] in val_ids])


def _quantiles(values: list[float], qs=(0.1, 0.5, 0.9)) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {f"p{int(q * 100)}": round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 4) for q in qs}


def summarize_pairs(pairs: list[dict], task_infos: list[dict], all_status: Counter) -> dict:
    ratios = [p["length_ratio"] for p in pairs]
    reasons = Counter(info["reason"] for info in task_infos if info["reason"])
    return {
        "tasks": len(task_infos),
        "tasks_with_pairs": len({p["task_id"] for p in pairs}),
        "pairs": len(pairs),
        "skip_reasons": dict(reasons),
        "candidate_status_counts": dict(all_status),
        "rejected_status_counts": dict(Counter(p["rejected_status"] for p in pairs)),
        "length_ratio_quantiles": _quantiles(ratios),
        "mean_chosen_tokens": sum(p["chosen_tokens"] for p in pairs) / len(pairs) if pairs else 0,
        "mean_rejected_tokens": sum(p["rejected_tokens"] for p in pairs) / len(pairs) if pairs else 0,
    }


def check_gate(stats: dict, min_pairs: int = 300) -> dict[str, bool]:
    """Stage-3 gate from the execution plan; all must be True before DPO training."""
    median = stats.get("length_ratio_quantiles", {}).get("p50")
    rejected = stats.get("rejected_status_counts", {})
    total_rejected = sum(rejected.values())
    return {
        f"at_least_{min_pairs}_train_pairs": stats.get("train_pairs", stats.get("pairs", 0)) >= min_pairs,
        "median_length_ratio_in_0.8_1.25": median is not None and 0.8 <= median <= 1.25,
        "wrong_answer_at_least_half_of_rejected":
            total_rejected > 0 and rejected.get(cc.WRONG_ANSWER, 0) / total_rejected >= 0.5,
    }


# ---------------------------------------------------------------------------
# Generation + judging
# ---------------------------------------------------------------------------

def judge_candidates(tasks: list[dict], per_task_outs: dict, timeout: float, workers: int) -> dict:
    """Judge ``per_task_outs[task_id] = [(GenOut, temperature), ...]`` in parallel."""
    jobs = [(task, out, temp) for task in tasks for out, temp in per_task_outs[task["task_id"]]]

    def run(job):
        task, out, temp = job
        verdict = cc.judge_mbpp(out.text, task["entry_point"], task["test_list"], task["setup_code"],
                                hit_token_limit=not out.finished, timeout=timeout)
        return task["task_id"], {"text": cc.strip_think(out.text), "n_tokens": out.n_tokens,
                                 "finished": out.finished, "status": verdict.status, "temperature": temp}

    judged: dict = {task["task_id"]: [] for task in tasks}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task_id, cand in pool.map(run, jobs):
            judged[task_id].append(cand)
    return judged


@click.command()
@click.option("--sft-checkpoint", required=True,
              help="Model to sample from: 'base', a HF id, a model dir or a LoRA adapter dir.")
@click.option("--num-problems", default=0, show_default=True, help="0 = the whole pool.")
@click.option("--n-candidates", default=8, show_default=True)
@click.option("--temperatures", default="0.2,0.4,0.6,0.8,1.0", show_default=True,
              help="Cycled over candidates. Must stay within [0.2, 1.0].")
@click.option("--max-new-tokens", default=1024, show_default=True)
@click.option("--max-pairs-per-task", default=2, show_default=True)
@click.option("--min-length-ratio", default=0.67, show_default=True)
@click.option("--max-length-ratio", default=1.5, show_default=True)
@click.option("--timeout", default=10.0, show_default=True)
@click.option("--val-ratio", default=0.1, show_default=True)
@click.option("--min-pairs", default=300, show_default=True, help="Warn below this many pairs.")
@click.option("--backend", type=click.Choice(["unsloth", "hf", "vllm"]), default="unsloth", show_default=True)
@click.option("--batch-size", default=16, show_default=True)
@click.option("--workers", default=8, show_default=True)
@click.option("--output-dir", default="data/processed", show_default=True)
@click.option("--seed", default=42, show_default=True)
def main(sft_checkpoint, num_problems, n_candidates, temperatures, max_new_tokens, max_pairs_per_task,
         min_length_ratio, max_length_ratio, timeout, val_ratio, min_pairs, backend, batch_size,
         workers, output_dir, seed):
    temps = [float(t) for t in temperatures.split(",")]
    if not all(0.2 <= t <= 1.0 for t in temps):
        raise click.BadParameter("temperatures must lie in [0.2, 1.0]; hotter samples are garbage negatives")

    from model_io import load_generator

    tasks = load_pool(num_problems, seed)
    print(f"Pool: {len(tasks)} MBPP tasks (MBPP+ excluded)")

    generator = load_generator(sft_checkpoint, backend=backend, batch_size=batch_size)
    tokenizer = generator.tokenizer
    prompts = [cc.render_prompt(tokenizer, t["messages"]) for t in tasks]

    per_task_outs: dict = {t["task_id"]: [] for t in tasks}
    for k in range(n_candidates):
        temp = temps[k % len(temps)]
        print(f"Sampling round {k + 1}/{n_candidates} at T={temp} ...")
        for task, out in zip(tasks, generator.generate(prompts, max_new_tokens=max_new_tokens,
                                                       temperature=temp, seed=seed + k)):
            per_task_outs[task["task_id"]].append((out, temp))

    print("Executing candidates ...")
    judged = judge_candidates(tasks, per_task_outs, timeout, workers)

    pairs, task_infos, all_status = [], [], Counter()
    for task, prompt in zip(tasks, prompts):
        cands = judged[task["task_id"]]
        all_status.update(c["status"] for c in cands)
        task_pairs, info = build_pairs_for_task(prompt, task["task_id"], cands, max_pairs_per_task,
                                                min_length_ratio, max_length_ratio)
        pairs.extend(task_pairs)
        task_infos.append(info)

    train, val = split_by_task(pairs, val_ratio, seed)
    out_dir = ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("dpo_pairs.jsonl", train), ("dpo_pairs_val.jsonl", val)):
        (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8")
    stats = summarize_pairs(pairs, task_infos, all_status)
    stats.update({"train_pairs": len(train), "val_pairs": len(val), "sampled_from": sft_checkpoint,
                  "temperatures": temps, "n_candidates": n_candidates})
    (out_dir / "dpo_pairs_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    gate = check_gate(stats, min_pairs)
    print("Stage-3 gate:", json.dumps(gate))
    if not all(gate.values()):
        print("WARNING: preference-pair gate NOT met; do not start DPO training "
              "(try --n-candidates 12 or inspect dpo_pairs_stats.json).")


if __name__ == "__main__":
    main()
