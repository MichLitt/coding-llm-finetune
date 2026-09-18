"""Offline tests for extraction, assembly, execution and failure classification."""

import pytest

import codegen_common as cc

HE_PROMPT = '''from typing import List


def helper(x: int) -> int:
    return x * 2


def add_twice(a: int, b: int) -> int:
    """Return helper(a) + helper(b).
    >>> add_twice(1, 2)
    6
    """
'''
HE_TEST = """
def check(candidate):
    assert candidate(1, 2) == 6
    assert candidate(0, 0) == 0
"""
EP = "add_twice"
GOOD_DEF = "def add_twice(a: int, b: int) -> int:\n    return helper(a) + helper(b)"


def judge(raw, hit=False):
    return cc.judge_humaneval(raw, HE_PROMPT, HE_TEST, EP, hit_token_limit=hit, timeout=5)


# ---- extract_code: 12 realistic reply shapes ------------------------------

def test_plain_python_fence():
    out = cc.extract_code(f"```python\n{GOOD_DEF}\n```", EP)
    assert out.defines_entry and out.parsed_ok and out.found_block


def test_reply_with_explanation_around_fence():
    raw = f"Here is the solution:\n\n```python\n{GOOD_DEF}\n```\n\nThis works because ..."
    assert cc.extract_code(raw, EP).defines_entry


def test_untagged_fence():
    assert cc.extract_code(f"```\n{GOOD_DEF}\n```", EP).defines_entry


def test_think_block_is_stripped():
    raw = f"<think>\nmaybe def add_twice(a, b): return 0\n</think>\n\n```python\n{GOOD_DEF}\n```"
    out = cc.extract_code(raw, EP)
    assert "return 0" not in out.code and "helper(a)" in out.code


def test_unterminated_think_yields_nothing():
    out = cc.extract_code("<think>\nlet me think about def add_twice(", EP)
    assert out.code == ""


def test_multiple_blocks_prefers_definition_of_entry_point():
    raw = "```bash\npip install x\n```\n```python\nprint(1)\n```\n" f"```python\n{GOOD_DEF}\n```"
    out = cc.extract_code(raw, EP)
    assert out.defines_entry and "print" not in out.code


def test_first_defining_block_wins_over_usage_example():
    raw = f"```python\n{GOOD_DEF}\n```\nExample:\n```python\ndef add_twice(a, b):\n    return 0\nprint(add_twice(1,2))\n```"
    assert "helper(a)" in cc.extract_code(raw, EP).code


def test_unterminated_final_fence_is_still_extracted():
    out = cc.extract_code(f"```python\n{GOOD_DEF}\n", EP)
    assert out.defines_entry and out.parsed_ok


def test_no_fence_but_raw_code():
    out = cc.extract_code(GOOD_DEF, EP)
    assert out.defines_entry and not out.found_block


def test_prose_only_reply_is_no_code():
    assert cc.extract_code("I would use a helper to double each value.", EP).code == ""


def test_demo_calls_and_main_guard_are_dropped():
    raw = f"```python\n{GOOD_DEF}\n\nprint(add_twice(1, 2))\nresult = add_twice(3, 4)\nassert result == 14\n\nif __name__ == '__main__':\n    input()\n```"
    code = cc.extract_code(raw, EP).code
    assert "print" not in code and "input" not in code and "result" not in code and "assert" not in code
    assert cc.judge_humaneval(raw, HE_PROMPT, HE_TEST, EP, timeout=5).status == cc.PASS


def test_decorators_and_imports_survive_sanitizing():
    raw = "```python\nimport functools\n\n@functools.lru_cache(None)\ndef add_twice(a, b):\n    return helper(a) + helper(b)\n```"
    code = cc.extract_code(raw, EP).code
    assert "@functools.lru_cache" in code and "import functools" in code


# ---- assembly / judging ----------------------------------------------------

def test_full_definition_keeps_prompt_helpers():
    assert judge(f"```python\n{GOOD_DEF}\n```").status == cc.PASS


def test_body_only_reply_completes_the_stub():
    assert judge("```python\n    return helper(a) + helper(b)\n```").status == cc.PASS


def test_unindented_body_is_indented():
    assert judge("```python\nreturn helper(a) + helper(b)\n```").status == cc.PASS


def test_wrong_answer():
    assert judge("```python\ndef add_twice(a, b):\n    return a + b\n```").status == cc.WRONG_ANSWER


def test_runtime_error():
    assert judge("```python\ndef add_twice(a, b):\n    return undefined_name\n```").status == cc.RUNTIME_ERROR


def test_syntax_error():
    assert judge("```python\ndef add_twice(a, b)\n    return 1\n```").status == cc.SYNTAX_ERROR


def test_missing_entry_point():
    assert judge("```python\ndef something_else(a, b):\n    return 1\n```").status == cc.MISSING_ENTRY_POINT


def test_no_code():
    assert judge("Sorry, I cannot help with that.").status == cc.NO_CODE


def test_timeout():
    raw = "```python\ndef add_twice(a, b):\n    while True:\n        pass\n```"
    assert cc.judge_humaneval(raw, HE_PROMPT, HE_TEST, EP, timeout=1).status == cc.TIMEOUT


def test_truncation_relabels_format_failures_only():
    cut = "```python\ndef add_twice(a, b):\n    return helper(a) +"
    assert judge(cut, hit=True).status == cc.TRUNCATED
    assert judge(cut, hit=False).status == cc.SYNTAX_ERROR
    wrong = "```python\ndef add_twice(a, b):\n    return a + b\n```"
    assert judge(wrong, hit=True).status == cc.WRONG_ANSWER  # runnable-but-wrong stays wrong


def test_every_status_is_in_taxonomy():
    assert set(cc.ALL_STATUSES) >= {cc.PASS, cc.WRONG_ANSWER, cc.TIMEOUT, cc.TRUNCATED}


# ---- MBPP -----------------------------------------------------------------

MBPP_TESTS = ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)"]


def test_mbpp_entry_point_skips_builtins():
    assert cc.mbpp_entry_point(["assert set(similar_elements((1,2),(2,3))) == set((2,))"]) == "similar_elements"
    assert cc.mbpp_entry_point(["assert sorted(f(3)) == [1]"], "") == "f"
    assert cc.mbpp_entry_point(["assert len('a') == 1"]) is None


def test_mbpp_entry_point_skips_setup_helpers():
    setup = "def helper(x):\n    return x"
    assert cc.mbpp_entry_point(["assert helper(target(2)) == 2"], setup) == "target"


def test_judge_mbpp_pass_and_fail():
    ok = "```python\ndef similar_elements(a, b):\n    return tuple(sorted(set(a) & set(b)))\n```"
    bad = "```python\ndef similar_elements(a, b):\n    return ()\n```"
    renamed = "```python\ndef common(a, b):\n    return (4, 5)\n```"
    assert cc.judge_mbpp(ok, "similar_elements", MBPP_TESTS, timeout=5).status == cc.PASS
    assert cc.judge_mbpp(bad, "similar_elements", MBPP_TESTS, timeout=5).status == cc.WRONG_ANSWER
    assert cc.judge_mbpp(renamed, "similar_elements", MBPP_TESTS, timeout=5).status == cc.MISSING_ENTRY_POINT


def test_split_mbpp_plus_prompt():
    prompt = '"""\nWrite a function to find shared elements.\nassert set(f((1,2),(2,3))) == set((2,))\n"""\n'
    text, test = cc.split_mbpp_plus_prompt(prompt)
    assert text == "Write a function to find shared elements."
    assert test.startswith("assert set(f(")


# ---- prompt rendering ------------------------------------------------------

class _FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "rendered"


def test_render_prompt_disables_thinking():
    tok = _FakeTokenizer()
    assert cc.render_prompt(tok, cc.humaneval_messages("def f():\n    pass")) == "rendered"
    assert tok.kwargs["enable_thinking"] is False
    assert tok.kwargs["add_generation_prompt"] is True
    assert tok.kwargs["tokenize"] is False


def test_messages_use_shared_system_prompt():
    for msgs in (cc.humaneval_messages("x"), cc.mbpp_messages("t", "assert f()", "f")):
        assert msgs[0] == {"role": "system", "content": cc.SYSTEM_PROMPT}
        assert msgs[1]["role"] == "user"
    content = cc.mbpp_messages("t", "assert f()", "Find_Min_Length")[1]["content"]
    assert "assert f()" in content
    assert content.rstrip().endswith("The function must be named `Find_Min_Length`.")
