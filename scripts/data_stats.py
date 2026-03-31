"""Data quality report for SFT training data.

Reads sft_train.jsonl / sft_val.jsonl and prints:
  - Sample counts per source
  - Token length distributions (p50 / p95 / p99)
  - Deduplication rate (via filename)
  - Targeted data coverage per pattern
  - Sample previews (3 per source)

Usage:
    uv run python scripts/data_stats.py
    uv run python scripts/data_stats.py --data-dir data/processed
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import click
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def _token_stats(samples: list[dict], tokenizer) -> dict:
    prompt_tokens: list[int] = []
    response_tokens: list[int] = []
    for s in samples:
        msgs = s.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompt_tokens.append(len(tokenizer.encode(user_msg)))
        response_tokens.append(len(tokenizer.encode(asst_msg)))
    return {
        "prompt": {
            "mean":   statistics.mean(prompt_tokens)   if prompt_tokens else 0,
            "median": statistics.median(prompt_tokens) if prompt_tokens else 0,
            "p95":    _percentile(prompt_tokens, 95),
            "p99":    _percentile(prompt_tokens, 99),
            "min":    min(prompt_tokens, default=0),
            "max":    max(prompt_tokens, default=0),
        },
        "response": {
            "mean":   statistics.mean(response_tokens)   if response_tokens else 0,
            "median": statistics.median(response_tokens) if response_tokens else 0,
            "p95":    _percentile(response_tokens, 95),
            "p99":    _percentile(response_tokens, 99),
            "min":    min(response_tokens, default=0),
            "max":    max(response_tokens, default=0),
        },
    }


def _print_bar(label: str, count: int, total: int, width: int = 30) -> None:
    frac = count / total if total else 0
    bar  = "█" * int(frac * width)
    print(f"  {label:<25} {count:>6}  ({frac:.1%}) {bar}")


@click.command()
@click.option("--data-dir", default="data/processed", show_default=True)
@click.option("--tokenizer-name", default="unsloth/Qwen3.5-9B", show_default=True)
@click.option("--preview-n", default=2, show_default=True,
              help="Number of sample previews to show per source.")
def main(data_dir: str, tokenizer_name: str, preview_n: int):
    data_path = ROOT / data_dir
    train_samples = _load_jsonl(data_path / "sft_train.jsonl")
    val_samples   = _load_jsonl(data_path / "sft_val.jsonl")
    all_samples   = train_samples + val_samples

    if not all_samples:
        print(f"No data found in {data_path}. Run prepare_sft_data.py first.")
        return

    print(f"\n{'='*60}")
    print("SFT Data Quality Report")
    print(f"{'='*60}")

    # ---- Counts ----
    total = len(all_samples)
    print(f"\nTotal samples : {total:,}  (train={len(train_samples):,} / val={len(val_samples):,})")

    source_counts = Counter(s.get("source", "unknown") for s in all_samples)
    print(f"\nSource distribution:")
    for src, cnt in source_counts.most_common():
        _print_bar(src, cnt, total)

    # ---- Targeted pattern coverage ----
    targeted = [s for s in all_samples if s.get("source") == "targeted"]
    if targeted:
        pattern_counts = Counter(s.get("pattern_key", "unknown") for s in targeted)
        print(f"\nTargeted failure coverage ({len(targeted)} samples):")
        for pk, cnt in sorted(pattern_counts.items()):
            _print_bar(pk, cnt, len(targeted))

    # ---- Token stats ----
    print(f"\nLoading tokenizer ({tokenizer_name}) for token stats …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        stats = _token_stats(all_samples, tokenizer)

        print("\nToken length statistics:")
        print(f"  {'Field':<12} {'mean':>8} {'median':>8} {'p95':>8} {'p99':>8} {'min':>6} {'max':>6}")
        print(f"  {'-'*58}")
        for field in ("prompt", "response"):
            s = stats[field]
            print(
                f"  {field:<12} {s['mean']:>8.1f} {s['median']:>8.1f} "
                f"{s['p95']:>8.0f} {s['p99']:>8.0f} {s['min']:>6} {s['max']:>6}"
            )
    except Exception as e:
        print(f"  (Skipping token stats — tokenizer load failed: {e})")

    # ---- Sample previews ----
    print(f"\nSample previews ({preview_n} per source):")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for s in all_samples:
        by_source[s.get("source", "unknown")].append(s)

    for src, samples in sorted(by_source.items()):
        print(f"\n  --- {src} ({len(samples)} samples) ---")
        for s in samples[:preview_n]:
            msgs  = s.get("messages", [])
            user  = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst  = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            print(f"  [user]  {user[:120].strip()!r}")
            print(f"  [asst]  {asst[:120].strip()!r}")
            if s.get("task_id"):
                print(f"  task_id={s['task_id']}")
            print()

    # ---- Save stats JSON ----
    stats_dir = ROOT / "data/stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_out = {
        "total": total,
        "train": len(train_samples),
        "val":   len(val_samples),
        "by_source": dict(source_counts),
        "targeted_by_pattern": dict(Counter(s.get("pattern_key", "") for s in targeted)),
    }
    (stats_dir / "sft_stats.json").write_text(
        json.dumps(stats_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Stats saved to {stats_dir / 'sft_stats.json'}")


if __name__ == "__main__":
    main()
