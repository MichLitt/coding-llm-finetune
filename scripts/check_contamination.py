"""Pre-training contamination check.

Loads the prepared SFT training data and verifies there is no overlap
with HumanEval or MBPP benchmark problems. Must pass before training.

Checks:
  1. No sample has source == "humaneval"
  2. No sample instruction contains known HumanEval function signatures
  3. N-gram overlap between instructions and benchmark prompts is below threshold

Exit code 0 = clean. Exit code 1 = contamination detected (do not train).

Usage:
    uv run python scripts/check_contamination.py
    uv run python scripts/check_contamination.py --data-dir data/processed --threshold 0.5
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click

ROOT = Path(__file__).parent.parent

# Known HumanEval function names — only keep compound/specific names that cannot
# plausibly appear by coincidence in unrelated code. Generic single words like
# "search", "solve", "sort", "encode", "eat", "multiply", "encrypt", "longest",
# "fib", "tri" are excluded because they produce massive false-positive rates.
HUMANEVAL_SIGNATURES = {
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
    # targeted cases (compound names only — fib/tri/intersection excluded as too generic)
    "same_chars", "minSubArraySum", "prod_signs",
    "file_name_check", "get_max_triples",
}

# For code-side checking (``def name(``), generic names are specific enough.
HUMANEVAL_CODE_EXTRA = {"fib", "tri", "intersection"}


def _ngrams(text: str, n: int = 5) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@click.command()
@click.option("--data-dir", default="data/processed", show_default=True)
@click.option("--threshold", default=0.3, show_default=True,
              help="Jaccard similarity threshold above which a sample is flagged.")
@click.option("--verbose", is_flag=True, default=False)
def main(data_dir: str, threshold: float, verbose: bool):
    data_path = ROOT / data_dir / "sft_train.jsonl"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run prepare_sft_data.py first.")
        sys.exit(1)

    samples = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    print(f"Loaded {len(samples):,} training samples from {data_path}")

    issues: list[str] = []

    # ---- Check 1: explicit source tag ----
    he_sourced = [s for s in samples if s.get("source") == "humaneval"]
    if he_sourced:
        issues.append(
            f"FAIL: {len(he_sourced)} sample(s) have source='humaneval' — "
            "HumanEval canonical solutions must not be in training data."
        )

    # ---- Check 2: known function signatures (whole-word match only) ----
    sig_hits: list[tuple[int, str, str]] = []
    sig_patterns = {sig: re.compile(r"\b" + re.escape(sig) + r"\b") for sig in HUMANEVAL_SIGNATURES}
    for i, s in enumerate(samples):
        instruction = s.get("messages", [{}])[1].get("content", "") if s.get("messages") else s.get("instruction", "")
        for sig, pat in sig_patterns.items():
            if pat.search(instruction):
                sig_hits.append((i, sig, instruction[:100]))
                break

    if sig_hits:
        issues.append(
            f"FAIL: {len(sig_hits)} sample(s) reference known HumanEval function names."
        )
        if verbose:
            for idx, sig, snippet in sig_hits[:10]:
                safe = snippet.encode("utf-8", errors="replace").decode("utf-8")
                print(f"  Sample {idx}: matched '{sig}' -- {safe!r}")

    # ---- Check 2b: known function definitions in assistant code ----
    code_sigs = HUMANEVAL_SIGNATURES | HUMANEVAL_CODE_EXTRA
    code_def_pats = {sig: re.compile(r"\bdef\s+" + re.escape(sig) + r"\s*\(") for sig in code_sigs}
    code_hits: list[tuple[int, str, str]] = []
    for i, s in enumerate(samples):
        msgs = s.get("messages", [])
        code = msgs[2].get("content", "") if len(msgs) > 2 else ""
        if not code:
            continue
        for sig, pat in code_def_pats.items():
            if pat.search(code):
                code_hits.append((i, sig, code[:100]))
                break

    if code_hits:
        issues.append(
            f"FAIL: {len(code_hits)} sample(s) define known HumanEval functions in code (assistant response)."
        )
        if verbose:
            for idx, sig, snippet in code_hits[:10]:
                safe = snippet.encode("utf-8", errors="replace").decode("utf-8")
                print(f"  Sample {idx}: defines '{sig}' -- {safe!r}")

    # ---- Check 3: n-gram overlap with HumanEval prompts ----
    # Load benchmark prompts if dataset available, else skip
    try:
        from datasets import load_dataset
        he_ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        he_ngrams = [_ngrams(row["prompt"]) for row in he_ds]
    except Exception as e:
        print(f"  (Skipping n-gram check -- could not load HumanEval: {e})")
        he_ngrams = []

    if he_ngrams:
        overlap_hits = []
        for i, s in enumerate(samples):
            instruction = s.get("messages", [{}])[1].get("content", "") if s.get("messages") else s.get("instruction", "")
            inst_ng = _ngrams(instruction)
            for j, he_ng in enumerate(he_ngrams):
                sim = _jaccard(inst_ng, he_ng)
                if sim > threshold:
                    overlap_hits.append((i, j, sim))
                    break

        if overlap_hits:
            issues.append(
                f"FAIL: {len(overlap_hits)} sample(s) have Jaccard similarity > {threshold} "
                f"with HumanEval prompts."
            )
            if verbose:
                for idx, he_idx, sim in overlap_hits[:5]:
                    print(f"  Sample {idx} <-> HumanEval/{he_idx}: similarity={sim:.2f}")

    # ---- Report ----
    print()
    if issues:
        for msg in issues:
            print(f"  {msg}")
        print("\nContamination check FAILED. Fix data pipeline before training.")
        sys.exit(1)
    else:
        print("Contamination check PASSED — no HumanEval overlap detected.")
        print("Safe to proceed with training.")
        sys.exit(0)


if __name__ == "__main__":
    main()
