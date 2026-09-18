"""Light-weight helpers shared by sft_train.py and dpo_train.py (no torch import)."""

from __future__ import annotations

import os
import re
from pathlib import Path

NON_THINKING_SUFFIX = "<think>\n\n</think>\n\n"
OUTPUT_ROOT_ENV = "CODETUNE_OUTPUT_ROOT"


def resolve_output_dir(repo_root: Path, configured: str, exp_id: str) -> Path:
    """``<root>/<configured>/<exp_id>`` where root is ``$CODETUNE_OUTPUT_ROOT`` (e.g. a Drive
    folder, so checkpoints survive Colab disconnects) or the repository."""
    root = Path(os.environ.get(OUTPUT_ROOT_ENV) or repo_root)
    return root / configured / exp_id


def latest_checkpoint(out_dir: Path) -> Path | None:
    """Newest ``checkpoint-<step>`` directory under ``out_dir`` (None if there is none)."""
    best: tuple[int, Path] | None = None
    if out_dir.is_dir():
        for child in out_dir.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", child.name)
            if match and child.is_dir() and (best is None or int(match.group(1)) > best[0]):
                best = (int(match.group(1)), child)
    return best[1] if best else None


def resolve_resume(out_dir: Path, resume_from: str | None) -> str | None:
    """``auto`` -> newest checkpoint in ``out_dir`` (None when starting fresh)."""
    if resume_from == "auto":
        found = latest_checkpoint(out_dir)
        print(f"Resume: {found}" if found else "Resume: no checkpoint found, starting fresh")
        return str(found) if found else None
    return resume_from or None


def check_dpo_format(rows: list[dict], n: int = 3) -> list[str]:
    """Return a list of problems (empty == OK) for the first ``n`` DPO rows.

    Guards the train/inference mismatch: prompts must be the chat-template-rendered,
    non-thinking generation prompts and completions must be bare replies.
    """
    problems: list[str] = []
    for i, row in enumerate(rows[:n]):
        prompt = row["prompt"]
        if not isinstance(prompt, str):
            problems.append(f"row {i}: prompt must be a rendered string, got {type(prompt).__name__}")
            continue
        if not prompt.startswith("<|im_start|>system"):
            problems.append(f"row {i}: prompt does not start with the system turn (template not applied?)")
        if not prompt.endswith("<|im_start|>assistant\n" + NON_THINKING_SUFFIX):
            problems.append(f"row {i}: prompt does not end with the non-thinking assistant header")
        for key in ("chosen", "rejected"):
            text = row[key]
            if text.lstrip().startswith("<think>"):
                problems.append(f"row {i}: {key} starts with a think block")
            if "<|im_end|>" in text or "<|im_start|>" in text:
                problems.append(f"row {i}: {key} contains chat control tokens (would duplicate EOS)")
            if not text.strip():
                problems.append(f"row {i}: {key} is empty")
    return problems


_DYNAMICS_KEYS = ("rewards/margins", "rewards/accuracies", "logps/chosen", "logps/rejected",
                  "rewards/chosen", "rewards/rejected")


def summarize_dynamics(log_history: list[dict], window: int = 3) -> dict:
    """Compare the first and last ``window`` training logs of the DPO reward metrics."""
    logs = [entry for entry in log_history if "rewards/margins" in entry or "logps/chosen" in entry]
    summary: dict = {"n_logs": len(logs)}
    if len(logs) < 2:
        return summary
    head, tail = logs[:window], logs[-window:]

    def mean(rows: list[dict], key: str):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else None

    for key in _DYNAMICS_KEYS:
        first, last = mean(head, key), mean(tail, key)
        if first is not None and last is not None:
            summary[key] = {"first": first, "last": last}
    chosen = summary.get("logps/chosen")
    summary["chosen_logp_decreased"] = bool(chosen and chosen["last"] < chosen["first"])
    return summary


def resolve_sft_hparams(base_cfg: dict, exp_cfg: dict, *, data_sources: list[str] | None = None,
                        lora_r: int | None = None, lora_alpha: int | None = None,
                        learning_rate: float | None = None, epochs: int | None = None) -> dict:
    """Effective SFT hyper-parameters. Precedence: CLI > experiment matrix > base config.

    The matrix owns r / alpha / lr / epochs. When r changes without an explicit alpha,
    alpha is reset to 2*r; a stale base-config alpha must never leak through (that
    silently doubled alpha/r in the first version of the ablations).
    """
    base_lora, base_train = base_cfg.get("lora", {}), base_cfg.get("training", {})
    r = lora_r or exp_cfg.get("lora_r") or base_lora.get("r", 32)
    alpha = lora_alpha or exp_cfg.get("lora_alpha")
    if alpha is None:
        alpha = base_lora.get("lora_alpha", 2 * r) if r == base_lora.get("r") else 2 * r
    return {
        "lora_r": r,
        "lora_alpha": alpha,
        "learning_rate": learning_rate or exp_cfg.get("learning_rate") or base_train.get("learning_rate", 2e-4),
        "num_train_epochs": epochs or exp_cfg.get("num_train_epochs") or base_train.get("num_train_epochs", 3),
        "data_sources": data_sources or exp_cfg.get("data_sources") or ["magicoder", "evol"],
    }
