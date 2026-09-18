"""Shared prompt, extraction, execution and failure-classification helpers.

Every script that generates or judges code (evaluation, DPO pair generation,
training-format checks) imports from here so that all of them use ONE prompt
format, ONE code-extraction routine and ONE failure taxonomy.

Design rules (see report/CodeTune_v3.1_Execution_Plan.md, D2/D3/D7):
  * Prompts are always rendered with ``enable_thinking=False``.
  * Raw chat output is never executed directly: fenced code is extracted,
    ``<think>`` blocks are stripped and stray top-level statements (prints,
    demo calls, ``if __name__`` blocks) are dropped.
  * Failures are classified, not just counted.

This module must stay importable without torch/transformers so the unit tests
run in CI.
"""

from __future__ import annotations

import ast
import builtins
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

# The system prompt is identical to the one used to build the SFT corpus
# (scripts/prepare_sft_data.py) so train and eval share a format.
SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write clean, efficient, and correct code. "
    "Always handle edge cases and include brief comments for complex logic."
)

# ---------------------------------------------------------------------------
# Failure taxonomy (plan D7)
# ---------------------------------------------------------------------------

PASS = "pass"
NO_CODE = "no_code"
TRUNCATED = "truncated"
SYNTAX_ERROR = "syntax_error"
MISSING_ENTRY_POINT = "missing_entry_point"
RUNTIME_ERROR = "runtime_error"
WRONG_ANSWER = "wrong_answer"
TIMEOUT = "timeout"

ALL_STATUSES = (
    PASS, NO_CODE, TRUNCATED, SYNTAX_ERROR,
    MISSING_ENTRY_POINT, RUNTIME_ERROR, WRONG_ANSWER, TIMEOUT,
)

# A failure is a "hard negative" for DPO only if the model produced runnable
# code that is merely wrong. Format failures teach length/format, not logic.
HARD_NEGATIVE_STATUSES = (WRONG_ANSWER, RUNTIME_ERROR)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. "
    "Reply with the complete function in a single ```python code block."
)
_MBPP_INSTRUCTION = (
    "Write a Python function to solve the task below. "
    "Reply with the complete solution in a single ```python code block."
)


def humaneval_messages(prompt: str) -> list[dict]:
    """Chat messages for a HumanEval-style task (``prompt`` is the code stub)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{_HUMANEVAL_INSTRUCTION}\n\n{prompt.strip()}"},
    ]


def mbpp_messages(text: str, first_test: str) -> list[dict]:
    """Chat messages for an MBPP-style task.

    The first assert is included so the model learns the required function name
    and signature (standard MBPP practice).
    """
    body = (
        f"{_MBPP_INSTRUCTION}\n\n{text.strip()}\n\n"
        f"Your code should pass this test:\n{first_test.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]


def split_mbpp_plus_prompt(prompt: str) -> tuple[str, str]:
    """Split an evalplus MBPP+ prompt into (description, first assert).

    evalplus prompts look like ``\"\"\"\\n<text>\\n<assert ...>\\n\"\"\"\\n``.
    """
    body = prompt.strip()
    if body.startswith('"""'):
        body = body[3:]
    if body.endswith('"""'):
        body = body[:-3]
    lines = body.strip().splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("assert"):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
    return body.strip(), ""


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render ``messages`` for generation, always in non-thinking mode."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_ONLY = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
# ```lang\n ... ``` ; an unterminated final fence (truncated output) runs to EOF.
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_PYTHON_LANGS = {"", "python", "py", "python3", "py3"}


def strip_think(text: str) -> str:
    """Remove ``<think>`` blocks (closed, or open-ended when truncated)."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN_ONLY.sub("", text)
    return text.replace("</think>", "").strip()


def defines_function(code: str, name: str) -> bool:
    return bool(re.search(rf"^[ \t]*(?:async[ \t]+)?def[ \t]+{re.escape(name)}[ \t]*\(", code, re.MULTILINE))


_TOP_LEVEL_DEF = re.compile(r"^(?:async[ \t]+)?(?:def|class)[ \t]+\w+", re.MULTILINE)


def _code_blocks(text: str) -> list[str]:
    blocks = []
    for match in _FENCE.finditer(text):
        lang = match.group(1).lower()
        body = match.group(2).rstrip()
        if lang in _PYTHON_LANGS and body.strip():
            blocks.append(body)
    return blocks


def keep_definitions_only(code: str) -> tuple[str, bool]:
    """Drop top-level statements that are not definitions.

    Removes demo calls, prints, asserts, ``if __name__ == "__main__"`` blocks and
    assignments that call functions defined in the same snippet. Returns
    ``(code, parsed_ok)``; unparsable code is returned unchanged with False.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False

    lines = code.splitlines()
    defined = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    def is_demo_assign(node: ast.stmt) -> bool:
        value = getattr(node, "value", None)
        if value is None:
            return False
        return any(
            isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in defined
            for sub in ast.walk(value)
        )

    kept: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pass
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if is_demo_assign(node):
                continue
        else:
            continue
        start = node.lineno
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, *(d.lineno for d in decorators))
        kept.append("\n".join(lines[start - 1: node.end_lineno]))
    return "\n\n".join(kept), True


@dataclass
class Extraction:
    code: str            # sanitized code ("" when nothing usable was found)
    found_block: bool    # a fenced code block was present
    parsed_ok: bool      # the extracted code parses as Python
    defines_entry: bool  # the code defines the entry-point function


def extract_code(raw: str, entry_point: str) -> Extraction:
    """Pull the solution code out of a raw chat reply."""
    text = strip_think(raw)
    if not text:
        return Extraction("", False, False, False)

    blocks = _code_blocks(text)
    found_block = bool(blocks)
    if blocks:
        with_entry = [b for b in blocks if defines_function(b, entry_point)]
        with_def = [b for b in blocks if _TOP_LEVEL_DEF.search(b)]
        code = (with_entry or with_def or blocks)[0]
    else:
        # No fence: accept the whole reply only if it looks like code.
        code = text if (_TOP_LEVEL_DEF.search(text) or defines_function(text, entry_point)) else ""
        if not code:
            return Extraction("", False, False, False)

    defines_entry = defines_function(code, entry_point)
    parsed_ok = True
    if defines_entry:
        code, parsed_ok = keep_definitions_only(code)
    else:
        try:
            ast.parse(code)
        except SyntaxError:
            parsed_ok = False
    return Extraction(code, found_block, parsed_ok, defines_entry)


def assemble_humaneval(prompt: str, extraction: Extraction, entry_point: str) -> tuple[str, str | None]:
    """Build the program to test for a HumanEval task.

    Returns ``(source, failure)``; ``failure`` is a taxonomy status when the
    reply cannot be turned into a candidate program at all.

    Full definition  -> ``prompt`` (imports and helpers) + definition.  The
                        stub in the prompt is a valid docstring-only function
                        that the later definition overrides.
    Body only        -> appended straight to the stub (indented when needed).
    """
    if not extraction.code:
        return prompt, NO_CODE
    code = extraction.code
    if extraction.defines_entry:
        return prompt.rstrip("\n") + "\n\n" + code + "\n", None
    if _TOP_LEVEL_DEF.search(code):
        return prompt + code, MISSING_ENTRY_POINT
    for candidate in (prompt.rstrip("\n") + "\n" + code, prompt.rstrip("\n") + "\n" + textwrap.indent(code, "    ")):
        try:
            # compile(), not ast.parse(): "return outside function" only fails here.
            compile(candidate, "<candidate>", "exec")
            return candidate + "\n", None
        except SyntaxError:
            continue
    return prompt + code, SYNTAX_ERROR


# ---------------------------------------------------------------------------
# MBPP helpers
# ---------------------------------------------------------------------------

_BUILTIN_NAMES = set(dir(builtins))


def mbpp_entry_point(test_list: list[str], setup_code: str = "") -> str | None:
    """Infer the function under test from MBPP assert statements.

    Picks the first called name that is neither a builtin nor a helper defined in
    the task's setup code (e.g. ``set(similar_elements(...))`` -> similar_elements).
    """
    helpers: set[str] = set()
    if setup_code:
        try:
            helpers = {
                n.name for n in ast.parse(setup_code).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
        except SyntaxError:
            pass
    for test in test_list:
        try:
            tree = ast.parse(test.strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name not in _BUILTIN_NAMES and name not in helpers:
                    return name
    return None


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    status: str   # PASS / WRONG_ANSWER / RUNTIME_ERROR / SYNTAX_ERROR / TIMEOUT
    detail: str = ""


def run_program(source: str, timeout: float = 10.0) -> ExecResult:
    """Run ``source`` in a fresh interpreter and classify the outcome.

    Isolation is best-effort (separate process group, temp cwd, no stdin,
    bounded output). Only run this on machines meant for executing model code.
    """
    with tempfile.TemporaryDirectory(prefix="codetune_run_") as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "candidate.py"
        script.write_text(source, encoding="utf-8")
        out_path, err_path = tmp_path / "stdout.txt", tmp_path / "stderr.txt"
        with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(script)],
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait()
                return ExecResult(TIMEOUT, f"timed out after {timeout}s")
        stderr = err_path.read_bytes()[-2000:].decode("utf-8", errors="replace")

    if proc.returncode == 0:
        return ExecResult(PASS)
    last = next((ln for ln in reversed(stderr.strip().splitlines()) if ln.strip()), "")
    if last.startswith("AssertionError"):
        return ExecResult(WRONG_ANSWER, last)
    if last.startswith(("SyntaxError", "IndentationError", "TabError")):
        return ExecResult(SYNTAX_ERROR, last)
    return ExecResult(RUNTIME_ERROR, last or f"exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


@dataclass
class Judgement:
    status: str
    solution: str      # program that was (or would have been) tested
    detail: str = ""


def finalize_status(status: str, hit_token_limit: bool) -> str:
    """Relabel format failures as ``truncated`` when generation hit the token cap."""
    if hit_token_limit and status in (NO_CODE, SYNTAX_ERROR, MISSING_ENTRY_POINT):
        return TRUNCATED
    return status


def judge_humaneval(raw: str, prompt: str, test: str, entry_point: str,
                    hit_token_limit: bool = False, timeout: float = 10.0) -> Judgement:
    extraction = extract_code(raw, entry_point)
    source, failure = assemble_humaneval(prompt, extraction, entry_point)
    # A body-only reply is legitimately unparsable on its own; only full
    # definitions must parse standalone.
    if failure is None and extraction.defines_entry and not extraction.parsed_ok:
        failure = SYNTAX_ERROR
    if failure is not None:
        return Judgement(finalize_status(failure, hit_token_limit), source, failure)
    program = source + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    result = run_program(program, timeout)
    return Judgement(finalize_status(result.status, hit_token_limit), source, result.detail)


def judge_mbpp(raw: str, entry_point: str, test_list: list[str], setup_code: str = "",
               hit_token_limit: bool = False, timeout: float = 10.0) -> Judgement:
    extraction = extract_code(raw, entry_point)
    if not extraction.code:
        return Judgement(finalize_status(NO_CODE, hit_token_limit), "", NO_CODE)
    if not extraction.parsed_ok:
        return Judgement(finalize_status(SYNTAX_ERROR, hit_token_limit), extraction.code, SYNTAX_ERROR)
    if not extraction.defines_entry:
        return Judgement(finalize_status(MISSING_ENTRY_POINT, hit_token_limit), extraction.code, MISSING_ENTRY_POINT)
    program = ""
    if setup_code:
        program += setup_code.strip() + "\n\n"
    program += extraction.code + "\n\n" + "\n".join(test_list) + "\n"
    result = run_program(program, timeout)
    return Judgement(finalize_status(result.status, hit_token_limit), extraction.code, result.detail)
