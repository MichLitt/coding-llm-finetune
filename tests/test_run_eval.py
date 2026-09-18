"""Harness tests with a fake generator (no GPU, no datasets download)."""

import json
from pathlib import Path

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


# ---- evalplus staleness guard and --rejudge ------------------------------------------------

def _fake_evalplus(monkeypatch, write=None):
    """Replace the evalplus subprocess. ``write(results_path)`` simulates evalplus output."""
    calls = []

    def fake_run(cmd, check, stdin=None):
        samples = Path(cmd[cmd.index("--samples") + 1])
        results = samples.with_name(samples.stem + "_eval_results.json")
        calls.append(results.exists())          # was a stale file still present when evalplus started?
        if write:
            write(results)

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)
    return calls


def _results(entries):
    return {"eval": {t: [{"task_id": t, "solution": sol, "base_status": b, "plus_status": p}]
                     for t, (sol, b, p) in entries.items()}}


def test_run_evalplus_deletes_stale_results_and_returns_fresh_ones(tmp_path, monkeypatch):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("")
    stale = tmp_path / "samples_eval_results.json"
    stale.write_text(json.dumps(_results({"HumanEval/0": ("old code", "pass", "pass")})))

    def write(results_path):
        results_path.write_text(json.dumps(_results({"HumanEval/0": ("new code\n", "fail", "fail")})))

    calls = _fake_evalplus(monkeypatch, write)
    out = run_eval.run_evalplus("humaneval", samples, {"HumanEval/0": "new code"})
    assert calls == [False], "stale results must be removed before evalplus runs"
    assert out == {"HumanEval/0": {"base": "fail", "plus": "fail"}}


def test_run_evalplus_rejects_results_for_different_code(tmp_path, monkeypatch):
    import pytest
    samples = tmp_path / "samples.jsonl"
    samples.write_text("")
    _fake_evalplus(monkeypatch, lambda p: p.write_text(json.dumps(_results({"HumanEval/0": ("old code", "pass", "pass")}))))
    with pytest.raises(RuntimeError, match="do not match"):
        run_eval.run_evalplus("humaneval", samples, {"HumanEval/0": "new code"})


def test_run_evalplus_rejects_missing_tasks_and_missing_file(tmp_path, monkeypatch):
    import pytest
    samples = tmp_path / "samples.jsonl"
    samples.write_text("")
    _fake_evalplus(monkeypatch, lambda p: p.write_text(json.dumps(_results({"HumanEval/0": ("a", "pass", "pass")}))))
    with pytest.raises(RuntimeError, match="1 missing"):
        run_eval.run_evalplus("humaneval", samples, {"HumanEval/0": "a", "HumanEval/1": "b"})
    _fake_evalplus(monkeypatch, None)
    (tmp_path / "samples_eval_results.json").unlink(missing_ok=True)
    with pytest.raises(RuntimeError, match="did not write"):
        run_eval.run_evalplus("humaneval", samples, None)


def test_rejudge_roundtrip_reproduces_statuses_without_a_model(tmp_path):
    outs = [GenOut("```python\ndef add(a, b):\n    return a + b\n```", 20, True),
            GenOut("```python\ndef add(a, b):\n    return a +", 1024, False)]
    tasks = he_tasks(2)
    records, raw_rows = run_eval.judge_all("humaneval", tasks, outs, timeout=5)
    run_eval.write_outputs(tmp_path, records, raw_rows, {})
    reloaded = run_eval.load_raw_outputs(tmp_path, tasks)
    assert [(o.text, o.n_tokens, o.finished) for o in reloaded] == [(o.text, o.n_tokens, o.finished) for o in outs]
    again, _ = run_eval.judge_all("humaneval", tasks, reloaded, timeout=5)
    assert [r["status"] for r in again] == [r["status"] for r in records] == [cc.PASS, cc.TRUNCATED]


def test_rejudge_fails_loudly_when_raw_is_incomplete(tmp_path):
    import pytest
    (tmp_path / "raw.jsonl").write_text(json.dumps({"task_id": "HumanEval/0", "raw": "x", "n_tokens": 1, "finished": True}) + "\n")
    with pytest.raises(RuntimeError, match="lacks 1 tasks"):
        run_eval.load_raw_outputs(tmp_path, he_tasks(2))
