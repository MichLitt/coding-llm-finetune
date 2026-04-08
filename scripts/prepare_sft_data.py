"""SFT data preparation pipeline.

Downloads Magicoder-OSS-Instruct and Evol-CodeAlpaca, filters to
high-quality Python instruction-response pairs, deduplicates with
MinHash LSH, and writes train/val JSONL files.

Usage:
    uv run python scripts/prepare_sft_data.py
    uv run python scripts/prepare_sft_data.py --max-samples 5000 --output-dir data/processed
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import click
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write clean, efficient, and correct code. "
    "Always handle edge cases and include brief comments for complex logic."
)

# ---- language / code helpers ------------------------------------------------

_PY_KEYWORDS = re.compile(
    r"\b(def |class |import |from |return |yield |lambda |with |async |await )\b"
)
_NON_PY_HINTS = re.compile(
    r"\b(public static|void |System\.out|console\.log|#include|std::|var |let |const )\b"
)
_CODE_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_CODE_BLOCK = re.compile(r"```\w*\s*(.*?)```", re.DOTALL)


def is_python(text: str) -> bool:
    """Return True if text contains Python code (not C/JS/Java/etc.)."""
    # Explicit language tag wins
    if "```python" in text.lower():
        return True
    if re.search(r"```(javascript|java|c\+\+|cpp|c#|typescript|go|rust|ruby)", text, re.I):
        return False
    # Heuristic: count Python indicators vs non-Python indicators
    py_hits = len(_PY_KEYWORDS.findall(text))
    non_py_hits = len(_NON_PY_HINTS.findall(text))
    return py_hits >= 2 and non_py_hits == 0


def extract_code_block(text: str) -> str:
    """Extract the first ```python ... ``` block; fall back to any code block."""
    m = _CODE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    m = _ANY_CODE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    # Last resort: find a `def ` block
    lines = text.splitlines()
    in_block = False
    block_lines: list[str] = []
    for line in lines:
        if line.startswith("def ") or line.startswith("class "):
            in_block = True
        if in_block:
            block_lines.append(line)
    return "\n".join(block_lines).strip()


def has_obvious_issue(code: str) -> bool:
    """Return True if code has obvious issues (syntax error, trivial body, bad import)."""
    if not code.strip():
        return True
    # Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    # Trivial body: only Pass nodes and nothing else
    func_defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if func_defs:
        for fn in func_defs:
            body_stmts = [s for s in fn.body if not isinstance(s, (ast.Pass, ast.Expr))]
            if not body_stmts:
                # Check if Expr is a non-docstring
                exprs = [s for s in fn.body if isinstance(s, ast.Expr)]
                if not exprs:
                    return True
    # Known bad import patterns
    _BAD_IMPORTS = {"nonexistent_lib", "fake_module", "undefined_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BAD_IMPORTS:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _BAD_IMPORTS:
                return True
    return False


# ---- MinHash helpers ---------------------------------------------------------

def _trigrams(text: str) -> list[str]:
    text = text.lower()
    return [text[i:i+3] for i in range(len(text) - 2)]


def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for gram in _trigrams(text):
        m.update(gram.encode("utf8"))
    return m


# ---- formatting --------------------------------------------------------------

def to_messages(instruction: str, output: str, source: str, task_id: str | None = None) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": output.strip()},
        ],
        "source": source,
        "task_id": task_id,
    }


# ---- per-dataset loaders -----------------------------------------------------

def _iter_magicoder(tokenizer, min_p: int, max_p: int, min_r: int, max_r: int) -> Iterator[dict]:
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", trust_remote_code=True)
    for row in tqdm(ds, desc="Magicoder", unit="row"):
        instruction = row.get("problem") or row.get("instruction") or ""
        output = row.get("solution") or row.get("output") or ""
        if not instruction or not output:
            continue
        if not is_python(output):
            continue
        p_tok = len(tokenizer.encode(instruction))
        r_tok = len(tokenizer.encode(output))
        if not (min_p <= p_tok <= max_p):
            continue
        if not (min_r <= r_tok <= max_r):
            continue
        code = extract_code_block(output) or output
        if has_obvious_issue(code):
            continue
        yield to_messages(instruction, output, source="magicoder")


def _iter_evol(tokenizer, min_p: int, max_p: int, min_r: int, max_r: int) -> Iterator[dict]:
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", trust_remote_code=True)
    for row in tqdm(ds, desc="EvolCodeAlpaca", unit="row"):
        instruction = row.get("instruction") or ""
        output = row.get("output") or ""
        if not instruction or not output:
            continue
        if not is_python(output):
            continue
        p_tok = len(tokenizer.encode(instruction))
        r_tok = len(tokenizer.encode(output))
        if not (min_p <= p_tok <= max_p):
            continue
        if not (min_r <= r_tok <= max_r):
            continue
        code = extract_code_block(output) or output
        if has_obvious_issue(code):
            continue
        yield to_messages(instruction, output, source="evol")



# ---- deduplication -----------------------------------------------------------

def deduplicate(samples: list[dict], threshold: float = 0.85, num_perm: int = 128) -> list[dict]:
    """MinHash LSH deduplication on instruction text."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique: list[dict] = []
    # Targeted samples are never deduplicated (small, curated set)
    priority = [s for s in samples if s["source"] == "targeted"]
    general  = [s for s in samples if s["source"] != "targeted"]

    # Insert priority samples first (no dedup among them)
    for i, s in enumerate(priority):
        key = f"p_{i}"
        mh = compute_minhash(s["messages"][1]["content"])
        try:
            lsh.insert(key, mh)
        except ValueError:
            pass  # duplicate key (shouldn't happen with enumerated keys)
        unique.append(s)

    # Dedup general samples against existing set
    for i, s in enumerate(tqdm(general, desc="Deduplicating", unit="sample")):
        mh = compute_minhash(s["messages"][1]["content"])
        results = lsh.query(mh)
        if results:
            continue  # near-duplicate found
        key = f"g_{i}"
        lsh.insert(key, mh)
        unique.append(s)

    return unique


# ---- main --------------------------------------------------------------------

@click.command()
@click.option("--output-dir", default="data/processed", show_default=True,
              help="Directory to write train/val JSONL files.")
@click.option("--max-samples", default=0, type=int,
              help="Cap total samples (0 = no cap, for quick testing use e.g. 500).")
@click.option("--val-ratio", default=0.05, show_default=True,
              help="Fraction of data held out as validation set.")
@click.option("--sources", default="magicoder,evol",
              show_default=True, help="Comma-separated list of data sources to include.")
@click.option("--seed", default=42, show_default=True)
def main(output_dir: str, max_samples: int, val_ratio: float, sources: str, seed: int):
    import random
    random.seed(seed)

    out = ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer (Qwen3.5-9B) …")
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Qwen3.5-9B", trust_remote_code=True
    )

    # Token length filters
    MIN_P, MAX_P = 20, 512
    MIN_R, MAX_R = 10, 1024

    source_list = [s.strip() for s in sources.split(",")]
    all_samples: list[dict] = []

    if "magicoder" in source_list:
        all_samples.extend(_iter_magicoder(tokenizer, MIN_P, MAX_P, MIN_R, MAX_R))

    if "evol" in source_list:
        all_samples.extend(_iter_evol(tokenizer, MIN_P, MAX_P, MIN_R, MAX_R))

    # Load targeted data if present
    targeted_path = ROOT / "data/processed/targeted/targeted_filtered.jsonl"
    if targeted_path.exists():
        targeted = [json.loads(l) for l in targeted_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Loaded {len(targeted)} targeted samples from {targeted_path}")
        all_samples.extend(targeted)

    print(f"\nBefore dedup: {len(all_samples)} samples")
    source_counts = Counter(s["source"] for s in all_samples)
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    # Deduplication
    all_samples = deduplicate(all_samples)
    print(f"After dedup:  {len(all_samples)} samples")

    # Optional cap
    if max_samples and len(all_samples) > max_samples:
        # Always keep targeted samples
        priority = [s for s in all_samples if s["source"] == "targeted"]
        general  = [s for s in all_samples if s["source"] != "targeted"]
        random.shuffle(general)
        cap_general = max(0, max_samples - len(priority))
        all_samples = priority + general[:cap_general]
        print(f"After cap ({max_samples}): {len(all_samples)} samples")

    # Shuffle and split
    random.shuffle(all_samples)
    val_n = max(1, int(len(all_samples) * val_ratio))
    val_samples = all_samples[:val_n]
    train_samples = all_samples[val_n:]

    # Write JSONL
    train_path = out / "sft_train.jsonl"
    val_path   = out / "sft_val.jsonl"

    train_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in train_samples) + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in val_samples) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {len(train_samples):,} train samples → {train_path}")
    print(f"Wrote {len(val_samples):,}   val samples → {val_path}")
    print("Done.")


if __name__ == "__main__":
    main()
