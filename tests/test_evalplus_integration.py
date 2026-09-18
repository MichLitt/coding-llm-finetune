"""Advisory Linux-only test of the real evalplus path (skipped unless RUN_EVALPLUS_INTEGRATION=1).

evalplus applies RLIMIT_AS, which macOS rejects, so this runs in CI on ubuntu. It checks that
reference solutions pass HumanEval+/MBPP+ through run_eval.run_evalplus and that our own
executor agrees with evalplus on the HumanEval base tests.
"""

import json
import os

import pytest

import codegen_common as cc
import run_eval

pytestmark = pytest.mark.skipif(os.environ.get("RUN_EVALPLUS_INTEGRATION") != "1",
                                reason="set RUN_EVALPLUS_INTEGRATION=1 (Linux + evalplus) to run")


def write_samples(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_humaneval_reference_solutions_and_executor_agreement(tmp_path):
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()
    ids = list(problems)[:12]
    rows = [{"task_id": t, "solution": problems[t]["prompt"] + problems[t]["canonical_solution"]} for t in ids]
    rows[-1] = {"task_id": ids[-1], "solution": problems[ids[-1]]["prompt"] + "    return None\n"}  # wrong on purpose
    samples = tmp_path / "samples.jsonl"
    write_samples(samples, rows)

    results = run_eval.run_evalplus("humaneval", samples)
    assert all(results[t]["base"] == "pass" and results[t]["plus"] == "pass" for t in ids[:-1])
    assert results[ids[-1]]["base"] != "pass"

    # our executor and evalplus must agree on the base tests
    from datasets import load_dataset
    he = {r["task_id"]: r for r in load_dataset("openai/openai_humaneval", split="test")}
    for row in rows:
        task = he[row["task_id"]]
        raw = "```python\n" + row["solution"] + "\n```"
        ours = cc.judge_humaneval(raw, task["prompt"], task["test"], task["entry_point"], timeout=20)
        assert (ours.status == cc.PASS) == (results[row["task_id"]]["base"] == "pass"), row["task_id"]


def test_mbpp_reference_solutions(tmp_path):
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    ids = list(problems)[:10]
    samples = tmp_path / "samples.jsonl"
    write_samples(samples, [{"task_id": t, "solution": problems[t]["prompt"].split('"""')[0] + problems[t]["canonical_solution"]}
                            for t in ids])
    results = run_eval.run_evalplus("mbpp", samples)
    assert all(results[t]["base"] == "pass" and results[t]["plus"] == "pass" for t in ids)
