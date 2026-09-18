"""Harness tests with a fake generator (no GPU, no datasets download)."""

import json

import codegen_common as cc
import run_eval
from model_io import GenOut, is_adapter_dir

PROMPT = 'def add(a, b):\n    """Add two numbers."""\n'
TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[-1]["content"]


class FakeGenerator:
    def __init__(self, outs):
        self.outs = outs
        self.seen = None

    def generate(self, prompts, max_new_tokens=1024, temperature=0.0, **_):
        self.seen = (prompts, max_new_tokens, temperature)
        return self.outs


def he_tasks(n):
    return [{"task_id": f"HumanEval/{i}", "entry_point": "add", "prompt": PROMPT, "test": TEST,
             "messages": cc.humaneval_messages(PROMPT)} for i in range(n)]


def test_humaneval_end_to_end_with_all_failure_kinds(tmp_path):
    outs = [
        GenOut("```python\ndef add(a, b):\n    return a + b\n```", 20, True),      # pass
        GenOut("```python\ndef add(a, b):\n    return a - b\n```", 20, True),      # wrong_answer
        GenOut("```python\ndef add(a, b):\n    return a +", 1024, False),          # truncated
        GenOut("I will add the numbers.", 8, True),                                # no_code
        GenOut("```python\ndef add(a, b)\n    return 1\n```", 12, True),           # syntax_error
    ]
    gen = FakeGenerator(outs)
    records, raw_rows = run_eval.evaluate(gen, FakeTokenizer(), "humaneval", he_tasks(5), 1024, timeout=5)
    assert [r["status"] for r in records] == [
        cc.PASS, cc.WRONG_ANSWER, cc.TRUNCATED, cc.NO_CODE, cc.SYNTAX_ERROR]
    assert gen.seen[1] == 1024 and gen.seen[2] == 0.0  # greedy, 1024 tokens

    summary = run_eval.summarize("humaneval", records)
    assert summary["passed"] == 1 and summary["total"] == 5
    assert summary["hit_token_limit_rate"] == 0.2
    assert summary["format_failure_rate"] == 0.6
    assert summary["status_counts"][cc.WRONG_ANSWER] == 1

    run_eval.write_outputs(tmp_path, records, raw_rows, summary)
    for name in ("raw.jsonl", "samples.jsonl", "per_problem.jsonl", "summary.json"):
        assert (tmp_path / name).exists()
    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 5 and all(s["solution"] for s in samples)


def test_evalplus_cross_check_flags_disagreements():
    records = [
        {"task_id": "HumanEval/0", "status": cc.PASS, "n_tokens": 1, "hit_token_limit": False},
        {"task_id": "HumanEval/1", "status": cc.PASS, "n_tokens": 1, "hit_token_limit": False},
    ]
    run_eval.apply_evalplus("humaneval", records, {
        "HumanEval/0": {"base": "pass", "plus": "fail"},
        "HumanEval/1": {"base": "fail", "plus": "fail"},   # disagrees with our executor
    })
    summary = run_eval.summarize("humaneval", records)
    assert summary["evalplus_base_pass_at_1"] == 0.5
    assert summary["evalplus_plus_pass_at_1"] == 0.0
    assert summary["executor_disagreements"] == ["HumanEval/1"]


def test_mbpp_defers_tests_to_evalplus_but_flags_format_failures():
    task = {"task_id": "Mbpp/2", "entry_point": "f", "messages": []}
    ok = run_eval.judge_generation("mbpp", task, "```python\ndef f(x):\n    return x\n```", True, 5)
    assert ok["status"] is None and "def f" in ok["solution"]
    missing = run_eval.judge_generation("mbpp", task, "```python\ndef g(x):\n    return x\n```", True, 5)
    assert missing["status"] == cc.MISSING_ENTRY_POINT
    cut = run_eval.judge_generation("mbpp", task, "```python\ndef f(x):\n    return x +", False, 5)
    assert cut["status"] == cc.TRUNCATED

    records = [{"task_id": "Mbpp/2", "status": None, "n_tokens": 1, "hit_token_limit": False},
               {"task_id": "Mbpp/3", "status": None, "n_tokens": 1, "hit_token_limit": False},
               {"task_id": "Mbpp/4", "status": None, "n_tokens": 1, "hit_token_limit": False}]
    run_eval.apply_evalplus("mbpp", records, {
        "Mbpp/2": {"base": "pass", "plus": "pass"},
        "Mbpp/3": {"base": "pass", "plus": "fail"},
        "Mbpp/4": {"base": "timeout", "plus": "timeout"},
    })
    assert [r["status"] for r in records] == [cc.PASS, cc.WRONG_ANSWER, cc.TIMEOUT]


def test_is_adapter_dir(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    assert is_adapter_dir(str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    assert not is_adapter_dir(str(tmp_path))
    assert not is_adapter_dir("unsloth/Qwen3.5-4B")


def test_eval_gate_flags_harness_induced_failures():
    good = {"total": 100, "hit_token_limit_rate": 0.0, "status_counts": {"pass": 80, "wrong_answer": 20}}
    assert all(run_eval.eval_gate(good).values())
    truncated = dict(good, hit_token_limit_rate=0.08)
    assert not all(run_eval.eval_gate(truncated).values())
    junky = dict(good, status_counts={"pass": 80, "no_code": 5, "syntax_error": 5, "wrong_answer": 10})
    assert not all(run_eval.eval_gate(junky).values())
    disagree = dict(good, executor_disagreements=["HumanEval/1"])
    assert not run_eval.eval_gate(disagree)["executors_agree"]
