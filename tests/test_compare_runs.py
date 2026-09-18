import json

import numpy as np
from click.testing import CliRunner

import compare_runs as cr


def rec(passed, status=None, plus=None, tokens=100, hit=False):
    status = status or ("pass" if passed else "wrong_answer")
    return {"status": status, "n_tokens": tokens, "hit_token_limit": hit,
            "evalplus_plus": plus or ("pass" if passed else "fail")}


def run_of(bits, **kw):
    return {f"HumanEval/{i}": rec(bool(b), **kw) for i, b in enumerate(bits)}


def test_mcnemar_exact_matches_known_values():
    assert cr.mcnemar_exact(0, 0) == 1.0
    assert cr.mcnemar_exact(5, 5) == 1.0
    assert abs(cr.mcnemar_exact(0, 10) - 2 / 1024) < 1e-12          # 2 * P(X<=0), n=10
    assert abs(cr.mcnemar_exact(1, 9) - 2 * 11 / 1024) < 1e-12      # 2 * P(X<=1)
    assert cr.mcnemar_exact(3, 7) == cr.mcnemar_exact(7, 3)


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_n():
    small = np.array([1.0] * 5 + [0.0] * 5)
    big = np.array([1.0] * 500 + [0.0] * 500)
    lo_s, hi_s = cr.bootstrap_ci(small, 2000, seed=1)
    lo_b, hi_b = cr.bootstrap_ci(big, 2000, seed=1)
    assert lo_s < 0.5 < hi_s and lo_b < 0.5 < hi_b
    assert (hi_b - lo_b) < (hi_s - lo_s)


def test_compare_pair_reports_flips_and_ci():
    base = run_of([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    model = run_of([1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
    out = cr.compare_pair(base, model, n_boot=2000)
    assert out["fixed"] == ["HumanEval/2", "HumanEval/3", "HumanEval/8"]
    assert out["broken"] == ["HumanEval/1"]
    assert abs(out["delta"] - 0.2) < 1e-9
    lo, hi = out["delta_ci95"]
    assert lo <= out["delta"] <= hi
    assert 0.0 <= out["mcnemar_p"] <= 1.0


def test_identical_runs_have_zero_delta_and_p_one():
    base = run_of([1, 0, 1, 0])
    out = cr.compare_pair(base, dict(base), n_boot=500)
    assert out["delta"] == 0 and out["mcnemar_p"] == 1.0 and out["fixed"] == out["broken"] == []


def test_plus_metric_uses_evalplus_status():
    a = {"HumanEval/0": rec(True, plus="fail")}
    assert cr.passed(a["HumanEval/0"], "primary") and not cr.passed(a["HumanEval/0"], "plus")


def test_report_and_cli_roundtrip(tmp_path):
    for label, bits in (("base", [1, 0, 1, 0, 1, 0]), ("dpo", [1, 1, 1, 0, 1, 0])):
        d = tmp_path / label / "humaneval"
        d.mkdir(parents=True)
        (d / "per_problem.jsonl").write_text(
            "".join(json.dumps({"task_id": t, **r}) + "\n" for t, r in run_of(bits).items()))
    out = tmp_path / "report.md"
    result = CliRunner().invoke(cr.main, ["--labels", "base,dpo", "--datasets", "humaneval",
                                          "--results-dir", str(tmp_path), "--n-boot", "500", "--out", str(out)])
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "| dpo | +16.7" in text and "fixed HumanEval/1" in text
    assert "wrong_answer" in text
