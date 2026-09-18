"""Preference-pair construction rules (hard negatives, length control, task split)."""

from collections import Counter

import codegen_common as cc
import generate_dpo_pairs as gd


def cand(text, status, tokens=100, finished=True, temp=0.6):
    return {"text": text, "n_tokens": tokens, "finished": finished, "status": status, "temperature": temp}


def build(cands, **kw):
    return gd.build_pairs_for_task("PROMPT", 1, cands, **kw)


def test_basic_pair_uses_pass_and_hard_negative():
    pairs, info = build([cand("good", cc.PASS), cand("bad", cc.WRONG_ANSWER)])
    assert len(pairs) == 1 and info["reason"] is None
    assert pairs[0]["chosen"] == "good" and pairs[0]["rejected"] == "bad"
    assert pairs[0]["prompt"] == "PROMPT" and pairs[0]["rejected_status"] == cc.WRONG_ANSWER


def test_format_failures_are_never_negatives():
    for status in (cc.TRUNCATED, cc.SYNTAX_ERROR, cc.NO_CODE, cc.TIMEOUT, cc.MISSING_ENTRY_POINT):
        pairs, info = build([cand("good", cc.PASS), cand("junk", status)])
        assert pairs == [] and info["reason"] == "no_hard_negative", status


def test_runtime_error_counts_as_hard_negative():
    pairs, _ = build([cand("good", cc.PASS), cand("crash", cc.RUNTIME_ERROR)])
    assert len(pairs) == 1


def test_unfinished_candidates_are_excluded_on_both_sides():
    assert build([cand("good", cc.PASS, finished=False), cand("bad", cc.WRONG_ANSWER)])[1]["reason"] == "no_passing_candidate"
    assert build([cand("good", cc.PASS), cand("bad", cc.WRONG_ANSWER, finished=False)])[1]["reason"] == "no_hard_negative"


def test_length_ratio_filter_drops_lopsided_pairs():
    long_bad = cand("bad", cc.WRONG_ANSWER, tokens=400)
    pairs, info = build([cand("good", cc.PASS, tokens=100), long_bad])
    assert pairs == [] and info["reason"] == "length_ratio_out_of_range"
    short_bad = cand("bad", cc.WRONG_ANSWER, tokens=30)
    assert build([cand("good", cc.PASS, tokens=100), short_bad])[0] == []
    ok_bad = cand("bad", cc.WRONG_ANSWER, tokens=120)
    pairs, _ = build([cand("good", cc.PASS, tokens=100), ok_bad])
    assert pairs[0]["length_ratio"] == 1.2


def test_prefers_closest_length_and_never_reuses_a_candidate():
    cands = [
        cand("good_a", cc.PASS, tokens=100), cand("good_b", cc.PASS, tokens=200),
        cand("bad_a", cc.WRONG_ANSWER, tokens=105), cand("bad_b", cc.WRONG_ANSWER, tokens=190),
        cand("bad_c", cc.WRONG_ANSWER, tokens=150),
    ]
    pairs, _ = build(cands, max_pairs=5)
    assert len(pairs) == 2
    assert {(p["chosen"], p["rejected"]) for p in pairs} == {("good_a", "bad_a"), ("good_b", "bad_b")}
    assert len({p["chosen"] for p in pairs}) == 2 and len({p["rejected"] for p in pairs}) == 2


def test_max_pairs_per_task_is_respected():
    cands = [cand(f"g{i}", cc.PASS) for i in range(4)] + [cand(f"b{i}", cc.WRONG_ANSWER) for i in range(4)]
    assert len(build(cands, max_pairs=2)[0]) == 2
    assert len(build(cands, max_pairs=1)[0]) == 1


def test_duplicate_candidates_are_collapsed():
    pairs, _ = build([cand("good", cc.PASS), cand("good  ", cc.PASS), cand("bad", cc.WRONG_ANSWER)], max_pairs=3)
    assert len(pairs) == 1


def test_split_by_task_has_no_leakage():
    pairs = [{"task_id": t, "chosen": "c", "rejected": "r"} for t in range(20) for _ in range(2)]
    train, val = gd.split_by_task(pairs, 0.1, seed=1)
    assert len(train) + len(val) == len(pairs)
    assert not {p["task_id"] for p in train} & {p["task_id"] for p in val}
    assert val


def test_gate_reports_each_criterion():
    stats = {"train_pairs": 320, "length_ratio_quantiles": {"p50": 1.05},
             "rejected_status_counts": {cc.WRONG_ANSWER: 70, cc.RUNTIME_ERROR: 30}}
    assert all(gd.check_gate(stats).values())
    bad = dict(stats, train_pairs=100, length_ratio_quantiles={"p50": 1.6},
               rejected_status_counts={cc.WRONG_ANSWER: 10, cc.RUNTIME_ERROR: 90})
    assert not any(gd.check_gate(bad).values())


def test_summary_counts_reasons_and_statuses():
    pairs, info = build([cand("good", cc.PASS), cand("bad", cc.WRONG_ANSWER)])
    _, none_info = build([cand("only bad", cc.WRONG_ANSWER)])
    stats = gd.summarize_pairs(pairs, [info, none_info], Counter({cc.PASS: 1}))
    assert stats["pairs"] == 1 and stats["tasks"] == 2
    assert stats["skip_reasons"] == {"no_passing_candidate": 1}


def test_pool_files_are_disjoint_from_mbpp_plus():
    import json
    pool = set(json.loads(gd.POOL_IDS_PATH.read_text())["task_ids"])
    plus = set(json.loads(gd.MBPP_PLUS_IDS_PATH.read_text())["task_ids"])
    assert len(pool) > 500 and len(plus) == 378
    assert not pool & plus
