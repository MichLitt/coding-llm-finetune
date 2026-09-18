"""SFT data preparation pipeline.

Downloads Magicoder-OSS-Instruct and Evol-CodeAlpaca, filters to
high-quality Python instruction-response pairs, deduplicates with
MinHash LSH, and writes train/val JSONL files.

Usage:
    uv run python scripts/prepare_sft_data.py
    uv run python scripts/prepare_sft_data.py --magicoder-n 2000 --evol-n 1500 --output-dir data/processed
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


# ---- HumanEval contamination filter -----------------------------------------

# Compound / unique HumanEval entry-point names.  Samples whose code defines
# one of these functions (via ``def name(``) are filtered to prevent benchmark
# contamination.  Generic single-word names (is_prime, remove_duplicates, …)
# are excluded to avoid false positives on ordinary CS exercises.
_HE_FUNC_NAMES: set[str] = {
    "has_close_elements", "separate_paren_groups", "truncate_number",
    "below_zero", "mean_absolute_deviation", "intersperse", "parse_nested_parens",
    "filter_by_substring", "sum_product", "rolling_max", "make_palindrome",
    "string_xor", "greatest_common_divisor", "all_prefixes",
    "string_sequence", "count_distinct_characters", "parse_music",
    "how_many_times", "sort_numbers", "find_closest_elements", "rescale_to_unit",
    "filter_integers", "fibfib", "vowels_count",
    "circular_shift", "digitSum", "fruit_distribution", "pluck",
    "strange_sort_list", "triangle_area", "will_it_fly", "smallest_change",
    "total_match", "is_multiply_prime", "is_simple_power", "iscube",
    "hex_key", "decimal_to_binary", "is_happy", "numerical_letter_grade",
    "prime_length", "starts_one_ends", "anti_shuffle", "get_row",
    "sort_array", "next_smallest", "is_bored", "any_int",
    "skjkasdkd", "check_dict_case", "count_up_to",
    "count_upper", "closest_integer", "make_a_pile", "words_string",
    "choose_num", "rounded_avg", "unique_digits", "by_length", "even_odd_count",
    "int_to_mini_roman", "right_angle_triangle", "find_max", "do_algebra",
    "string_to_md5", "generate_integers",
    # HumanEval entry points that look generic as free text but are still exact
    # benchmark function names when they appear as ``def name(`` in code
    "same_chars", "fib", "minSubArraySum", "intersection",
    "prod_signs", "tri", "file_name_check", "get_max_triples",
}

_HE_DEF_PATTERNS: set[str] = {f"def {n}(" for n in _HE_FUNC_NAMES}


def has_he_func_def(code: str) -> bool:
    """Return True if *code* defines a known HumanEval entry-point function."""
    return any(p in code for p in _HE_DEF_PATTERNS)


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
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
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
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
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
    for i, s in enumerate(tqdm(samples, desc="Deduplicating", unit="sample")):
        mh = compute_minhash(s["messages"][1]["content"])
        if lsh.query(mh):
            continue  # near-duplicate found
        lsh.insert(f"s_{i}", mh)
        unique.append(s)
    return unique


def rendered_length(tokenizer, sample: dict) -> int:
    """Token length of the full chat-template text the trainer will see (system + user + reply)."""
    text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
    return len(tokenizer(text).input_ids)


def cap_per_source(samples: list[dict], caps: dict[str, int], seed: int = 42) -> list[dict]:
    """Randomly keep at most ``caps[source]`` samples per source (0/absent = keep all)."""
    import random

    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for sample in samples:
        by_source.setdefault(sample["source"], []).append(sample)
    kept: list[dict] = []
    for source, rows in sorted(by_source.items()):
        cap = caps.get(source, 0)
        if cap and len(rows) > cap:
            rows = rng.sample(rows, cap)
        kept.extend(rows)
    return kept


# ---- main --------------------------------------------------------------------

@click.command()
@click.option("--output-dir", default="data/processed", show_default=True,
              help="Directory to write train/val JSONL files.")
@click.option("--magicoder-n", default=2000, show_default=True, type=int,
              help="Samples kept from Magicoder after filtering/dedup (0 = all).")
@click.option("--evol-n", default=1500, show_default=True, type=int,
              help="Samples kept from Evol-CodeAlpaca after filtering/dedup (0 = all).")
@click.option("--max-total-tokens", default=1024, show_default=True, type=int,
              help="Drop samples whose rendered chat text exceeds this (must equal max_seq_length; 0 = off).")
@click.option("--val-ratio", default=0.05, show_default=True,
              help="Fraction of data held out as validation set.")
@click.option("--sources", default="magicoder,evol",
              show_default=True, help="Comma-separated list of data sources to include.")
@click.option("--seed", default=42, show_default=True)
def main(output_dir: str, magicoder_n: int, evol_n: int, max_total_tokens: int, val_ratio: float,
         sources: str, seed: int):
    import random
    random.seed(seed)

    out = ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer (Qwen3.5-4B) …")
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Qwen3.5-4B"
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

    print(f"\nBefore dedup: {len(all_samples)} samples")
    source_counts = Counter(s["source"] for s in all_samples)
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    # Deduplication
    all_samples = deduplicate(all_samples)
    n_after_dedup = len(all_samples)
    print(f"After dedup:  {len(all_samples)} samples")

    # HumanEval contamination filter
    pre_filter = len(all_samples)
    all_samples = [s for s in all_samples if not has_he_func_def(s["messages"][2]["content"])]
    n_contaminated = pre_filter - len(all_samples)
    if n_contaminated:
        print(f"Removed {n_contaminated} samples with HumanEval function definitions (contamination filter)")

    # Drop anything the pre-training gate would flag (same checks as check_contamination.py),
    # so the prepared data passes the gate by construction. Done before the cap so the
    # per-source counts are still reached after removal.
    from check_contamination import find_contaminated, load_benchmarks

    statement_index, solution_index = load_benchmarks()
    if statement_index is None or solution_index is None:
        print("WARNING: benchmark data unavailable; n-gram contamination filter incomplete.")
    flagged = find_contaminated(all_samples, statement_index, solution_index)
    if flagged:
        reasons = Counter(check for hits in flagged.values() for check, _ in hits)
        print(f"Removed {len(flagged)} samples flagged by contamination checks: {dict(reasons)}")
        all_samples = [s for i, s in enumerate(all_samples) if i not in flagged]

    # Length filter on the *rendered* text: samples longer than max_seq_length would be
    # truncated mid-reply (no EOS) during training.
    n_before_len = len(all_samples)
    if max_total_tokens:
        all_samples = [s for s in tqdm(all_samples, desc="Length filter", unit="sample")
                       if rendered_length(tokenizer, s) <= max_total_tokens]
        print(f"Removed {n_before_len - len(all_samples)} samples over {max_total_tokens} rendered tokens")
    n_after_len = len(all_samples)

    # Per-source cap (keeps the SFT set small and its composition explicit)
    all_samples = cap_per_source(all_samples, {"magicoder": magicoder_n, "evol": evol_n}, seed)
    print(f"After per-source cap: {len(all_samples)} samples "
          f"{dict(Counter(s['source'] for s in all_samples))}")

    # Shuffle and split
    random.shuffle(all_samples)
    val_n = max(1, int(len(all_samples) * val_ratio))
    val_samples = all_samples[:val_n]
    train_samples = all_samples[val_n:]

    # Write JSONL
    train_path = out / "sft_train.jsonl"
    val_path   = out / "sft_val.jsonl"

    def _safe_dumps(obj: dict) -> str:
        s = json.dumps(obj, ensure_ascii=False)
        return s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    train_path.write_text(
        "\n".join(_safe_dumps(s) for s in train_samples) + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(_safe_dumps(s) for s in val_samples) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "seed": seed, "sources": source_list, "magicoder_n": magicoder_n, "evol_n": evol_n,
        "max_total_tokens": max_total_tokens, "val_ratio": val_ratio,
        "counts": {
            "loaded": dict(source_counts), "after_dedup": n_after_dedup,
            "removed_by_def_filter": n_contaminated, "removed_by_contamination_checks": len(flagged),
            "removed_by_length_filter": n_before_len - n_after_len,
            "train": len(train_samples), "val": len(val_samples),
            "train_by_source": dict(Counter(s["source"] for s in train_samples)),
        },
    }
    (out / "sft_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {len(train_samples):,} train samples → {train_path}")
    print(f"Wrote {len(val_samples):,}   val samples → {val_path}")
    print("Done.")


if __name__ == "__main__":
    main()
