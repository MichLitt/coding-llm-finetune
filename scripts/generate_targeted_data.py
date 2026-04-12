"""Generate targeted SFT data for Coder-Agent failure patterns.

Uses MiniMax API (OpenAI-compatible) to generate Python coding problems
and correct solutions for the 8 HumanEval tasks where the agent failed.

NOTE: Targeted data is conceptually aligned with HumanEval failure patterns
(same error concepts, different function names/contexts). This is intentional —
the goal is to teach the model specific error-fix patterns. It is NOT direct
benchmark contamination (no HumanEval prompts or solutions are used), but should
be disclosed in evaluation reports.

Features:
  - Resume from partial runs (append to targeted_raw.jsonl)
  - Syntax validation via compile()
  - Optional functional validation against HumanEval check() functions
  - Progress bar with per-pattern counts

Usage:
    uv run python scripts/generate_targeted_data.py
    uv run python scripts/generate_targeted_data.py --target-per-pattern 30 --batch-size 3
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import click
import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).parent.parent
RAW_PATH    = ROOT / "data/processed/targeted/targeted_raw.jsonl"
OUT_PATH    = ROOT / "data/processed/targeted/targeted_filtered.jsonl"
HE_DATA_PATH = Path(
    os.environ.get("HE_DATA_PATH", str(ROOT / "data/raw/humaneval_data.jsonl"))
)

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write clean, efficient, and correct code. "
    "Always handle edge cases and include brief comments for complex logic."
)

# ---------------------------------------------------------------------------
# Failure patterns from Coder Agent HumanEval C5 v6 results
# ---------------------------------------------------------------------------

FAILURE_PATTERNS: dict[str, dict] = {
    "HumanEval_54": {
        "task_id": "HumanEval/54",
        "concept": "string character set comparison",
        "weakness": "checking if two strings contain exactly the same character set (order independent)",
        "key_insight": "use set() comparison, e.g. set(s1) == set(s2), not sorted() with wrong logic",
        "example": "same_chars('abcd', 'dcba') == True, same_chars('abc', 'abd') == False",
    },
    "HumanEval_55": {
        "task_id": "HumanEval/55",
        "concept": "fibonacci sequence (1-indexed)",
        "weakness": "correct indexing: fib(1)=1, fib(2)=1, fib(10)=55",
        "key_insight": "avoid off-by-one: fib(n) = fib(n-1) + fib(n-2); base cases fib(1)=fib(2)=1",
        "example": "fib(10) == 55, fib(1) == 1, fib(2) == 1",
    },
    "HumanEval_107": {
        "task_id": "HumanEval/107",
        "concept": "minimum sum contiguous subarray (Kadane variant)",
        "weakness": "finding minimum sum subarray, handling all-positive arrays",
        "key_insight": "track current_min = min(current_min + x, x); global_min starts at arr[0]",
        "example": "minSubArraySum([2,3,4,1,2,4]) == 1, minSubArraySum([-1,-2,-3]) == -6",
    },
    "HumanEval_108": {
        "task_id": "HumanEval/108",
        "concept": "interval intersection length and primality check",
        "weakness": "computing overlap of two intervals, then testing if overlap length is prime",
        "key_insight": "overlap = max(0, min(b1,b2) - max(a1,a2)); then is_prime(overlap) where prime >= 2",
        "example": "intersection((1,2),(2,3)) == 'NO', intersection((-1,1),(0,4)) == 'NO', intersection((-3,-1),(-5,5)) == 'YES'",
    },
    "HumanEval_110": {
        "task_id": "HumanEval/110",
        "concept": "product of signs with absolute value sum",
        "weakness": "computing sign product (pos=+1, neg=-1, zero=0) * sum of absolute values",
        "key_insight": "zero in array -> return 0; sign_product = (-1)^(count of negatives); result = sign_product * sum(abs)",
        "example": "prod_signs([1,2,2,-4]) == -9, prod_signs([0,1]) == 0, prod_signs([]) == None",
    },
    "HumanEval_128": {
        "task_id": "HumanEval/128",
        "concept": "Tribonacci-like sequence with even/odd index rule",
        "weakness": "sequences where even/odd index determines different recurrence formulas",
        "key_insight": "tri(1)=3; tri(n): if n even -> n/2+1; if n odd -> tri(n-1)+tri(n-2)+tri(n+1)",
        "example": "tri(3) == [1,3,2.0,8.0], even n: n/2+1, odd n: sum of neighbors",
    },
    "HumanEval_141": {
        "task_id": "HumanEval/141",
        "concept": "multi-condition filename validation",
        "weakness": "validating filename with multiple simultaneous conditions",
        "key_insight": "exactly one dot; extension in {txt,exe,dll}; name part starts with letter; at most 3 digits total",
        "example": "file_name_check('example.txt') == 'Yes', file_name_check('1example.dll') == 'No'",
    },
    "HumanEval_147": {
        "task_id": "HumanEval/147",
        "concept": "array triple combination with modular arithmetic",
        "weakness": "counting triples (i,j,k) where a[i]+a[j]+a[k] divisible by 3, with a[i]=i+1",
        "key_insight": "a = [i+1 for i in range(n)]; count triples i<j<k where (a[i]+a[j]+a[k])%3==0",
        "example": "get_max_triples(5) == 1 (only (1,5,3) sums to 9 which is div by 3... check actual)",
    },
}


# ---------------------------------------------------------------------------
# API backends: GLM (OpenAI SDK) + MiniMax M2.7 (Anthropic SDK)
# ---------------------------------------------------------------------------

GLM_DEFAULT_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")
MINIMAX_DEFAULT_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")


def _make_glm_client() -> OpenAI:
    api_key = os.environ.get("GLM_API_KEY")
    base_url = os.environ.get("GLM_BASE_URL")
    if not api_key:
        raise RuntimeError("GLM_API_KEY not set in .env")
    return OpenAI(api_key=api_key, base_url=base_url)


def _make_minimax_client() -> anthropic.Anthropic:
    api_key = os.environ.get("MINIMAX_API_KEY")
    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
    # Anthropic SDK appends /v1/messages, strip trailing /v1 to avoid double path
    base_url = re.sub(r"/v1/?$", "", base_url)
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=api_key, base_url=base_url)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _call_glm(client: OpenAI, messages: list[dict], temperature: float = 0.7) -> str:
    resp = client.chat.completions.create(
        model=GLM_DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _call_minimax(client: anthropic.Anthropic, messages: list[dict], temperature: float = 0.7) -> str:
    # Anthropic SDK: system is a separate parameter
    system = ""
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            chat_messages.append(msg)
    resp = client.messages.create(
        model=MINIMAX_DEFAULT_MODEL,
        max_tokens=2048,
        system=system,
        messages=chat_messages,
        temperature=temperature,
    )
    # M2.7 returns [ThinkingBlock, TextBlock] — extract the text block
    for block in resp.content:
        if block.type == "text":
            return block.text or ""
    return ""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_PROBLEM_RE = re.compile(r"\bPROBLEM\b\s*\d*\s*[:\-]?\s*(.+?)(?=\bSOLUTION\b|$)", re.DOTALL | re.IGNORECASE)
_SOLUTION_CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SOLUTION_BLOCK_RE = re.compile(r"\bSOLUTION\b\s*\d*\s*[:\-]?\s*(.+?)(?=\bPROBLEM\b|$)", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>/<thinking> blocks from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _parse_pairs(text: str) -> list[tuple[str, str]]:
    """Parse PROBLEM/SOLUTION pairs from API response."""
    text = _strip_thinking(text)
    pairs: list[tuple[str, str]] = []

    # Split on numbered sections: "1. PROBLEM:", "1) PROBLEM:", etc.
    chunks = re.split(r"\n(?=\d+[\.\)]\s*PROBLEM\b)", text, flags=re.IGNORECASE)
    if len(chunks) <= 1:
        # Try unnumbered PROBLEM markers
        chunks = re.split(r"\n(?=PROBLEM\b\s*\d*\s*[:\-])", text, flags=re.IGNORECASE)
    if len(chunks) <= 1:
        chunks = [text]

    for chunk in chunks:
        problem_m = _PROBLEM_RE.search(chunk)
        code_m = _SOLUTION_CODE_RE.search(chunk)

        if problem_m and code_m:
            problem = problem_m.group(1).strip()
            # Clean up: remove everything after SOLUTION marker
            problem = re.split(r"\bSOLUTION\b", problem, flags=re.I)[0].strip()
            code = code_m.group(1).strip()
            if problem and code:
                pairs.append((problem, code))

    return pairs


def _validate_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _validate_with_humaneval(code: str, task_id: str, he_data: dict) -> bool | None:
    """Run HumanEval check() against the generated code. Returns None if test not available."""
    entry = he_data.get(task_id)
    if not entry:
        return None
    test_code = entry.get("test", "")
    entry_point = entry.get("entry_point", "")
    if not test_code or not entry_point:
        return None

    full_code = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        exec(compile(full_code, "<string>", "exec"), {})
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def _build_prompt(pattern_key: str, pattern: dict, batch_size: int) -> list[dict]:
    user_msg = (
        f"Generate {batch_size} different Python coding problems and correct solutions related to:\n\n"
        f"Concept: {pattern['concept']}\n"
        f"Common mistake to avoid: {pattern['weakness']}\n"
        f"Key insight for correct solution: {pattern['key_insight']}\n"
        f"Example behavior: {pattern['example']}\n\n"
        "Requirements:\n"
        "- Each problem must have a clear function signature with a docstring\n"
        "- Solution must correctly handle the concept and ALL edge cases\n"
        "- Difficulty: LeetCode Easy to Medium\n"
        "- Only use Python standard library\n"
        "- Vary the problem framing (different variable names, contexts)\n\n"
        "Format each problem exactly as:\n"
        "PROBLEM: <clear description of what the function should do>\n"
        "SOLUTION:\n"
        "```python\n"
        "<complete, correct Python function with docstring>\n"
        "```\n"
    )
    return [
        {"role": "system", "content": "You are a Python programming teacher. Generate coding exercises with correct solutions."},
        {"role": "user", "content": user_msg},
    ]


def _load_existing(raw_path: Path) -> dict[str, int]:
    """Return {pattern_key: count} of already-generated samples."""
    counts: dict[str, int] = {}
    if not raw_path.exists():
        return counts
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            pk = obj.get("pattern_key", "")
            counts[pk] = counts.get(pk, 0) + 1
        except json.JSONDecodeError:
            pass
    return counts


def _load_humaneval_data() -> dict:
    """Load HumanEval data indexed by task_id (HumanEval/N format)."""
    data: dict = {}
    if not HE_DATA_PATH.exists():
        return data
    for line in HE_DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            data[row["task_id"]] = row
        except (json.JSONDecodeError, KeyError):
            pass
    return data


@click.command()
@click.option("--target-per-pattern", default=25, show_default=True,
              help="Target number of valid samples per failure pattern (25 × 8 patterns = 200 total).")
@click.option("--batch-size", default=8, show_default=True,
              help="Number of problems to request per API call.")
@click.option("--backends", default="glm,minimax", show_default=True,
              help="Comma-separated backends to use. Alternates between them for diversity.")
@click.option("--temperature", default=0.7, show_default=True)
@click.option("--validate-he/--no-validate-he", default=False,
              help="Run HumanEval check() on generated code for the 8 target tasks.")
def main(target_per_pattern: int, batch_size: int, backends: str, temperature: float, validate_he: bool):
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Initialize backends
    backend_list = [b.strip() for b in backends.split(",") if b.strip()]
    clients: dict[str, tuple] = {}  # name -> (client, call_fn)
    for name in backend_list:
        if name == "glm":
            clients[name] = (_make_glm_client(), _call_glm)
        elif name == "minimax":
            clients[name] = (_make_minimax_client(), _call_minimax)
        else:
            raise click.BadParameter(f"Unknown backend: {name}. Use 'glm' or 'minimax'.")
    backend_names = list(clients.keys())
    print(f"Backends: {', '.join(f'{n} ({type(c).__name__})' for n, (c, _) in clients.items())}\n")

    he_data = _load_humaneval_data() if validate_he else {}
    existing = _load_existing(RAW_PATH)

    print(f"Target: {target_per_pattern} samples per pattern × {len(FAILURE_PATTERNS)} patterns = "
          f"{target_per_pattern * len(FAILURE_PATTERNS)} total\n")
    for pk, cnt in existing.items():
        print(f"  Already have {cnt} samples for {pk}")

    raw_fp = RAW_PATH.open("a", encoding="utf-8")
    call_idx = 0  # round-robin counter for backend alternation

    try:
        for pattern_key, pattern in FAILURE_PATTERNS.items():
            already = existing.get(pattern_key, 0)
            needed  = target_per_pattern - already
            if needed <= 0:
                print(f"\n[{pattern_key}] Already complete ({already}/{target_per_pattern}), skipping.")
                continue

            print(f"\n[{pattern_key}] Generating {needed} more (have {already}/{target_per_pattern}) …")
            generated = 0
            attempts  = 0
            max_attempts = needed * 4  # allow some waste

            with tqdm(total=needed, desc=pattern_key, unit="sample") as pbar:
                while generated < needed and attempts < max_attempts:
                    attempts += 1
                    # Round-robin between backends
                    bname = backend_names[call_idx % len(backend_names)]
                    client, call_fn = clients[bname]
                    call_idx += 1

                    cur_batch = min(batch_size, needed - generated)
                    messages  = _build_prompt(pattern_key, pattern, cur_batch)
                    try:
                        raw_text = call_fn(client, messages, temperature=temperature)
                    except Exception as e:
                        print(f"  [{bname}] API error: {e}", file=sys.stderr)
                        time.sleep(5)
                        continue

                    pairs = _parse_pairs(raw_text)
                    if not pairs:
                        time.sleep(1)
                        continue
                    for problem, code in pairs:
                        if not _validate_syntax(code):
                            continue
                        if validate_he and pattern.get("task_id"):
                            result = _validate_with_humaneval(code, pattern["task_id"], he_data)
                            if result is False:
                                continue

                        record = {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": problem},
                                {"role": "assistant", "content": code},
                            ],
                            "source": "targeted",
                            "pattern_key": pattern_key,
                            "task_id": None,
                            "backend": bname,
                        }
                        raw_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                        raw_fp.flush()
                        generated += 1
                        pbar.update(1)
                        if generated >= needed:
                            break

            if generated < needed:
                print(f"  WARNING: only generated {generated}/{needed} for {pattern_key}")

    finally:
        raw_fp.close()

    # --- Filter and write final output ---
    print("\nFiltering raw → filtered …")
    all_raw = [
        json.loads(l)
        for l in RAW_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    # Re-validate all (in case of previous partial runs with bad data)
    filtered = []
    for rec in all_raw:
        code = rec["messages"][2]["content"]
        if _validate_syntax(code):
            filtered.append(rec)

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in filtered) + "\n",
        encoding="utf-8",
    )

    by_pattern: dict[str, int] = {}
    for r in filtered:
        pk = r.get("pattern_key", "unknown")
        by_pattern[pk] = by_pattern.get(pk, 0) + 1

    print(f"\nFinal targeted dataset: {len(filtered)} valid samples")
    for pk, cnt in sorted(by_pattern.items()):
        print(f"  {pk}: {cnt}")
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
