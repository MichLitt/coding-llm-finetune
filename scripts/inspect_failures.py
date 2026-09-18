"""Inspect the failures of one evaluation run (CPU only; reads run_eval.py outputs).

Answers "why did the trust gate fail?": for every problem whose status is in --status it prints
the token count, whether generation hit the limit, a repetition-loop score and the head/tail of
the raw reply. It also prints the ids ready to paste into ``run_eval.py --subset`` for a
larger-limit probe (a reply that finishes and passes with 2048 tokens means the limit was too
low; one that loops again is model behaviour, not a harness defect).

Usage:
    python scripts/inspect_failures.py --run-dir <results>/eval/base/mbpp --status missing_entry_point,truncated
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import click

DEFAULT_STATUSES = "truncated,missing_entry_point,timeout,runtime_error,no_code,syntax_error"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repetition_stats(text: str) -> dict:
    """Crude loop detector: share of non-empty lines that repeat and the most repeated line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {"lines": 0, "dup_ratio": 0.0, "top_repeat": 0, "looping": False}
    counts = Counter(lines)
    dup = sum(c - 1 for c in counts.values())
    top = counts.most_common(1)[0][1]
    ratio = dup / len(lines)
    return {"lines": len(lines), "dup_ratio": round(ratio, 2), "top_repeat": top,
            "looping": len(lines) >= 10 and (ratio >= 0.4 or top >= 8)}


def defined_names(code: str) -> list[str]:
    return re.findall(r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)", code, re.MULTILINE)


def select(per_problem: list[dict], raw: dict[str, dict], statuses: set[str]) -> list[dict]:
    rows = []
    for rec in per_problem:
        hit = rec.get("hit_token_limit", False)
        if rec.get("status") in statuses or ("hit_limit" in statuses and hit):
            row = dict(rec)
            row["raw"] = raw.get(rec["task_id"], {}).get("raw", "")
            row["repetition"] = repetition_stats(row["raw"])
            rows.append(row)
    return rows


@click.command()
@click.option("--run-dir", required=True, help="results/eval/<label>/<dataset> directory.")
@click.option("--status", default=DEFAULT_STATUSES, show_default=True,
              help="Comma-separated statuses to show; add 'hit_limit' to include every reply that hit the token cap.")
@click.option("--max-show", default=8, show_default=True, help="Detailed examples per status.")
@click.option("--chars", default=350, show_default=True, help="Head/tail characters of each raw reply.")
def main(run_dir: str, status: str, max_show: int, chars: int):
    root = Path(run_dir)
    per_problem = load_jsonl(root / "per_problem.jsonl")
    raw = {r["task_id"]: r for r in load_jsonl(root / "raw.jsonl")}
    statuses = {s.strip() for s in status.split(",") if s.strip()}
    rows = select(per_problem, raw, statuses)

    print(f"{root}: {len(per_problem)} problems, {len(rows)} selected")
    print("status counts:", dict(Counter(r["status"] for r in per_problem)))
    hit = [r for r in per_problem if r.get("hit_token_limit")]
    print(f"hit token limit: {len(hit)} -> {[r['task_id'] for r in hit]}")
    print("  of which passed anyway:", sum(r["status"] == "pass" for r in hit))

    by_status: dict[str, list[dict]] = {}
    for row in rows:
        by_status.setdefault(row["status"], []).append(row)
    for st, group in by_status.items():
        loops = sum(r["repetition"]["looping"] for r in group)
        print(f"\n{'=' * 78}\n{st}: {len(group)} problems ({loops} look like repetition loops)")
        print("ids:", ",".join(r["task_id"].split("/")[-1] for r in group))
        for row in group[:max_show]:
            rep = row["repetition"]
            print(f"\n--- {row['task_id']} | tokens={row['n_tokens']} | hit_limit={row.get('hit_token_limit')} "
                  f"| detail={row.get('detail', '')!r} | defs={defined_names(row.get('solution') or '')} "
                  f"| loop={rep['looping']} (dup {rep['dup_ratio']}, top x{rep['top_repeat']})")
            text = row["raw"]
            print("HEAD:", text[:chars].replace("\n", "\\n"))
            if len(text) > chars * 2:
                print("TAIL:", text[-chars:].replace("\n", "\\n"))

    if hit:
        ids = ",".join(r["task_id"].split("/")[-1] for r in hit)
        print(f"\nProbe with a larger limit:\n  python scripts/run_eval.py --model <model> --label <label>_probe2048 "
              f"--dataset <dataset> --max-new-tokens 2048 --subset {ids}")


if __name__ == "__main__":
    main()
