import json

from click.testing import CliRunner

import inspect_failures as inf


def test_repetition_detector_flags_loops_not_normal_code():
    loop = "\n".join(["    result.append(x)"] * 30 + ["return result"])
    normal = "def f(x):\n    y = x + 1\n    z = y * 2\n    if z > 3:\n        return z\n    return y\n" * 1
    assert inf.repetition_stats(loop)["looping"]
    assert not inf.repetition_stats(normal)["looping"]
    assert inf.repetition_stats("")["lines"] == 0


def test_defined_names():
    assert inf.defined_names("import x\ndef a():\n    pass\nclass B:\n    def c(self): pass\n") == ["a", "B", "c"]


def test_cli_reports_statuses_and_probe_command(tmp_path):
    per = [
        {"task_id": "Mbpp/2", "status": "pass", "n_tokens": 50, "hit_token_limit": False, "solution": "def f(): pass"},
        {"task_id": "Mbpp/3", "status": "truncated", "n_tokens": 1024, "hit_token_limit": True, "solution": "def g(): pass"},
        {"task_id": "Mbpp/4", "status": "missing_entry_point", "n_tokens": 80, "hit_token_limit": False,
         "solution": "def other(): pass", "detail": "missing_entry_point"},
        {"task_id": "Mbpp/5", "status": "pass", "n_tokens": 1024, "hit_token_limit": True, "solution": "def h(): pass"},
    ]
    raw = [{"task_id": p["task_id"], "raw": "```python\n" + ("x = 1\n" * 40 if p["task_id"] == "Mbpp/3" else "def f(): pass\n") + "```"}
           for p in per]
    (tmp_path / "per_problem.jsonl").write_text("".join(json.dumps(p) + "\n" for p in per))
    (tmp_path / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    result = CliRunner().invoke(inf.main, ["--run-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "hit token limit: 2" in out and "of which passed anyway: 1" in out
    assert "missing_entry_point: 1 problems" in out and "defs=['other']" in out
    assert "truncated: 1 problems (1 look like repetition loops)" in out
    assert "--subset 3,5" in out
