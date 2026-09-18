"""Pre-training contamination check (hard gate for SFT data).

Loads the prepared SFT training data and verifies there is no overlap with the
held-out benchmarks: HumanEval (evaluation) and MBPP+ (second held-out benchmark).

Checks:
  1. No sample has source == "humaneval"
  2. No instruction mentions a known HumanEval function name
  2b. No assistant reply defines a known HumanEval function
  3. 5-gram overlap between an instruction and any HumanEval prompt / MBPP+ task
     (Jaccard, and containment: how much of the benchmark statement the instruction contains)
  4. 5-gram containment of a HumanEval reference solution inside an assistant reply

Exit code 0 = clean. Exit code 1 = contamination detected (do not train).

Usage:
    uv run python scripts/check_contamination.py
    uv run python scripts/check_contamination.py --data-dir data/processed --threshold 0.3
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
    # more HumanEval entry points (compound names only — fib/tri/intersection excluded as too generic)
    "same_chars", "minSubArraySum", "prod_signs",
    "file_name_check", "get_max_triples",
}

# For code-side checking (``def name(``), generic names are specific enough.
HUMANEVAL_CODE_EXTRA = {"fib", "tri", "intersection"}

# Containment thresholds (fraction of the benchmark text's 5-grams found in the sample).
CONTAINMENT_THRESHOLD = 0.5
SOLUTION_CONTAINMENT_THRESHOLD = 0.6


def _ngrams(text: str, n: int = 5) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NgramIndex:
    """Inverted 5-gram index over benchmark texts.

    ``scan(text)`` returns, per benchmark entry, the fraction of that entry's n-grams
    contained in ``text`` (containment) - robust to a short benchmark statement being
    embedded in a much longer instruction, where plain Jaccard is diluted.
    """

    def __init__(self, entries: dict[str, str], n: int = 5, min_ngrams: int = 6):
        self.n = n
        self.grams: dict[str, set[str]] = {}
        self.postings: dict[str, list[str]] = {}
        for name, text in entries.items():
            grams = _ngrams(text, n)
            if len(grams) < min_ngrams:  # too short to be meaningful (avoids false positives)
                continue
            self.grams[name] = grams
            for g in grams:
                self.postings.setdefault(g, []).append(name)

    def scan(self, text: str) -> dict[str, tuple[float, float]]:
        """``{name: (containment_in_text, jaccard)}`` for entries sharing any n-gram."""
        sample = _ngrams(text, self.n)
        counts: dict[str, int] = {}
        for g in sample:
            for name in self.postings.get(g, ()):
                counts[name] = counts.get(name, 0) + 1
        return {
            name: (hits / len(self.grams[name]), hits / (len(sample) + len(self.grams[name]) - hits))
            for name, hits in counts.items()
        }


def worst_match(index: NgramIndex, text: str) -> tuple[str, float, float] | None:
    """Best (highest containment) benchmark match for ``text`` or None."""
    scores = index.scan(text)
    if not scores:
        return None
    name = max(scores, key=lambda k: scores[k][0])
    return name, scores[name][0], scores[name][1]


def load_benchmarks() -> tuple[NgramIndex | None, NgramIndex | None]:
    """(problem-statement index, HumanEval reference-solution index); None when unavailable."""
    statements: dict[str, str] = {}
    solutions: dict[str, str] = {}
    try:
        from datasets import load_dataset

        for row in load_dataset("openai/openai_humaneval", split="test"):
            statements[row["task_id"]] = row["prompt"]
            solutions[row["task_id"]] = row["canonical_solution"]
    except Exception as e:
        print(f"  (Skipping HumanEval n-gram checks -- could not load HumanEval: {e})")
    try:
        from evalplus.data import get_mbpp_plus

        for task_id, row in get_mbpp_plus().items():
            statements[task_id] = row["prompt"]
    except Exception as e:
        print(f"  (Skipping MBPP+ n-gram check -- could not load MBPP+: {e})")
    return (NgramIndex(statements) if statements else None,
            NgramIndex(solutions) if solutions else None)


def _instruction(sample: dict) -> str:
    msgs = sample.get("messages")
    return msgs[1].get("content", "") if msgs and len(msgs) > 1 else sample.get("instruction", "")


def _reply(sample: dict) -> str:
    msgs = sample.get("messages", [])
    return msgs[2].get("content", "") if len(msgs) > 2 else ""


_SIG_PATTERNS = {sig: re.compile(r"\b" + re.escape(sig) + r"\b") for sig in HUMANEVAL_SIGNATURES}
_CODE_DEF_PATTERNS = {
    sig: re.compile(r"\bdef\s+" + re.escape(sig) + r"\s*\(") for sig in HUMANEVAL_SIGNATURES | HUMANEVAL_CODE_EXTRA
}


def find_contaminated(
    samples: list[dict],
    statement_index: NgramIndex | None,
    solution_index: NgramIndex | None,
    threshold: float = 0.3,
) -> dict[int, list[tuple[str, str]]]:
    """Run every check; returns ``{sample_index: [(check, detail), ...]}`` for flagged samples.

    Shared by this script (the pre-training gate) and prepare_sft_data.py (which drops
    flagged samples so the prepared data passes the gate by construction).
    Checks: ``source_tag``, ``instruction_name``, ``code_definition``, ``statement_overlap``,
    ``solution_overlap``.
    """
    flagged: dict[int, list[tuple[str, str]]] = {}

    def flag(i: int, check: str, detail: str) -> None:
        flagged.setdefault(i, []).append((check, detail))

    for i, s in enumerate(samples):
        instruction, reply = _instruction(s), _reply(s)
        if s.get("source") == "humaneval":
            flag(i, "source_tag", "source == 'humaneval'")
        for sig, pat in _SIG_PATTERNS.items():
            if pat.search(instruction):
                flag(i, "instruction_name", sig)
                break
        for sig, pat in _CODE_DEF_PATTERNS.items():
            if reply and pat.search(reply):
                flag(i, "code_definition", sig)
                break
        if statement_index is not None:
            match = worst_match(statement_index, instruction)
            if match and (match[1] >= CONTAINMENT_THRESHOLD or match[2] > threshold):
                flag(i, "statement_overlap", f"{match[0]} containment={match[1]:.2f} jaccard={match[2]:.2f}")
        if solution_index is not None and reply:
            match = worst_match(solution_index, reply)
            if match and match[1] >= SOLUTION_CONTAINMENT_THRESHOLD:
                flag(i, "solution_overlap", f"{match[0]} containment={match[1]:.2f}")
    return flagged


CHECK_TITLES = {
    "source_tag": "sample(s) tagged source='humaneval'",
    "instruction_name": "instruction(s) reference a known HumanEval function name",
    "code_definition": "assistant reply(ies) define a known HumanEval function",
    "statement_overlap": "instruction(s) overlap a HumanEval/MBPP+ problem statement "
                         f"(containment >= {CONTAINMENT_THRESHOLD} or Jaccard > threshold)",
    "solution_overlap": "assistant reply(ies) contain a HumanEval reference solution "
                        f"(containment >= {SOLUTION_CONTAINMENT_THRESHOLD})",
}


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

    statement_index, solution_index = load_benchmarks()
    flagged = find_contaminated(samples, statement_index, solution_index, threshold)

    by_check: dict[str, list[tuple[int, str]]] = {}
    for idx, hits in flagged.items():
        for check, detail in hits:
            by_check.setdefault(check, []).append((idx, detail))

    print()
    if by_check:
        for check, rows in by_check.items():
            print(f"  FAIL: {len(rows)} {CHECK_TITLES[check]}.")
            if verbose:
                for idx, detail in rows[:10]:
                    safe = _instruction(samples[idx])[:80].encode("utf-8", errors="replace").decode("utf-8")
                    print(f"    sample {idx}: {detail} -- {safe!r}")
        print("\nContamination check FAILED. Fix data pipeline before training.")
        sys.exit(1)
    print("Contamination check PASSED — no HumanEval/MBPP+ overlap detected.")
    print("Safe to proceed with training.")
    sys.exit(0)


if __name__ == "__main__":
    main()
