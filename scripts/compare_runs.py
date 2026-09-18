"""Paired comparison of evaluation runs (runs locally, CPU only).

Reads ``results/eval/<label>/<dataset>/per_problem.jsonl`` written by run_eval.py and reports:
  * pass@1 per model with a bootstrap 95% CI (resampling problems)
  * paired differences vs a reference run: delta, paired-bootstrap 95% CI and an exact
    McNemar p-value, plus the exact problems that flipped (fixed / broken)
  * failure-category distribution and mean output length per model

With 164 problems one flipped problem is 0.6pp, so differences without an interval
are not interpretable.

Usage:
    python scripts/compare_runs.py --labels base,sft_generic_cons,dpo_b01 --baseline base \
        --results-dir results/eval --out report/results_v3.1.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import codegen_common as cc  # noqa: E402

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(results_dir: Path, label: str, dataset: str) -> dict[str, dict]:
    path = Path(results_dir) / label / dataset / "per_problem.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")
    return {rec["task_id"]: rec for rec in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def passed(rec: dict, metric: str) -> bool:
    if metric == "plus":
        return rec.get("evalplus_plus") == "pass"
    return rec.get("status") == cc.PASS


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean(b) - mean(a), resampling problems jointly."""
    return bootstrap_ci(b.astype(float) - a.astype(float), n_boot, seed)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare_pair(a_recs: dict, b_recs: dict, metric: str = "primary", n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of run A (reference) and run B on their common problems."""
    ids = sorted(set(a_recs) & set(b_recs))
    a = np.array([passed(a_recs[t], metric) for t in ids])
    b = np.array([passed(b_recs[t], metric) for t in ids])
    fixed = [t for t, x, y in zip(ids, a, b) if not x and y]
    broken = [t for t, x, y in zip(ids, a, b) if x and not y]
    lo, hi = paired_bootstrap_ci(a, b, n_boot, seed)
    return {
        "n": len(ids),
        "a_pass": float(a.mean()) if len(ids) else 0.0,
        "b_pass": float(b.mean()) if len(ids) else 0.0,
        "delta": float(b.mean() - a.mean()) if len(ids) else 0.0,
        "delta_ci95": (lo, hi),
        "mcnemar_p": mcnemar_exact(len(broken), len(fixed)),
        "fixed": fixed,
        "broken": broken,
    }


def model_row(records: dict, metric: str, n_boot: int, seed: int) -> dict:
    vals = np.array([passed(r, metric) for r in records.values()], dtype=float)
    lo, hi = bootstrap_ci(vals, n_boot, seed)
    statuses: dict[str, int] = {}
    for rec in records.values():
        statuses[rec.get("status") or "unjudged"] = statuses.get(rec.get("status") or "unjudged", 0) + 1
    return {
        "n": len(vals), "pass": float(vals.mean()), "ci95": (lo, hi), "statuses": statuses,
        "hit_limit": float(np.mean([r.get("hit_token_limit", False) for r in records.values()])),
        "mean_tokens": float(np.mean([r.get("n_tokens", 0) for r in records.values()])),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_report(runs: dict[str, dict[str, dict]], baseline: str, dataset: str, metric: str,
                 n_boot: int = 10000, seed: int = 0) -> str:
    """``runs[label] = {task_id: record}`` for one dataset -> Markdown."""
    lines = [f"## {dataset} ({'evalplus plus tests' if metric == 'plus' else 'primary tests'})", ""]
    lines += ["| model | n | pass@1 | 95% CI | hit token limit | mean output tokens |", "|---|---|---|---|---|---|"]
    rows = {label: model_row(recs, metric, n_boot, seed) for label, recs in runs.items()}
    for label, r in rows.items():
        lines.append(f"| {label} | {r['n']} | {_pct(r['pass'])} | [{_pct(r['ci95'][0])}, {_pct(r['ci95'][1])}] "
                     f"| {_pct(r['hit_limit'])} | {r['mean_tokens']:.0f} |")

    lines += ["", f"### Paired comparison vs `{baseline}`", "",
              "| model | delta (pp) | 95% CI (pp) | McNemar p | fixed | broken |", "|---|---|---|---|---|---|"]
    details = []
    for label, recs in runs.items():
        if label == baseline:
            continue
        cmp = compare_pair(runs[baseline], recs, metric, n_boot, seed)
        lines.append(f"| {label} | {100 * cmp['delta']:+.1f} | [{100 * cmp['delta_ci95'][0]:+.1f}, "
                     f"{100 * cmp['delta_ci95'][1]:+.1f}] | {cmp['mcnemar_p']:.3f} | {len(cmp['fixed'])} | {len(cmp['broken'])} |")
        details.append((label, cmp))
    for label, cmp in details:
        lines += ["", f"**{label}**: fixed {', '.join(cmp['fixed']) or '-'}; broken {', '.join(cmp['broken']) or '-'}"]

    statuses = [s for s in (*cc.ALL_STATUSES, "unjudged") if any(s in r["statuses"] for r in rows.values())]
    lines += ["", "### Failure categories (problem counts)", "", "| model | " + " | ".join(statuses) + " |",
              "|---|" + "---|" * len(statuses)]
    for label, r in rows.items():
        lines.append(f"| {label} | " + " | ".join(str(r["statuses"].get(s, 0)) for s in statuses) + " |")
    return "\n".join(lines) + "\n"


@click.command()
@click.option("--labels", required=True, help="Comma-separated run labels; the baseline must be included.")
@click.option("--baseline", default="base", show_default=True)
@click.option("--datasets", default="humaneval,mbpp", show_default=True)
@click.option("--metric", type=click.Choice(["primary", "plus"]), default="primary", show_default=True,
              help="primary: our executor (HumanEval) / evalplus base+plus (MBPP+); plus: evalplus plus-tests pass.")
@click.option("--results-dir", default="results/eval", show_default=True)
@click.option("--n-boot", default=10000, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", default=None, help="Write the Markdown report here (default: print only).")
def main(labels, baseline, datasets, metric, results_dir, n_boot, seed, out):
    label_list = [l.strip() for l in labels.split(",") if l.strip()]
    if baseline not in label_list:
        raise click.BadParameter(f"baseline '{baseline}' must be one of --labels")
    root = Path(results_dir)
    root = root if root.is_absolute() else ROOT / root
    parts = ["# CodeTune v3.1 - evaluation comparison", ""]
    for dataset in [d.strip() for d in datasets.split(",") if d.strip()]:
        runs = {label: load_records(root, label, dataset) for label in label_list}
        sizes = {label: len(recs) for label, recs in runs.items()}
        if len(set(sizes.values())) != 1:
            print(f"WARNING: runs differ in problem count on {dataset}: {sizes} (comparing common problems only)")
        parts.append(build_report(runs, baseline, dataset, metric, n_boot, seed))
    report = "\n".join(parts)
    print(report)
    if out:
        Path(out).write_text(report, encoding="utf-8")
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
