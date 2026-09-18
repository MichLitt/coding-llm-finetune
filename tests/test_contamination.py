import json

from click.testing import CliRunner

import check_contamination as chk

STATEMENT = ("Write a function to find the shared elements from the given two lists and return them "
             "as a sorted tuple without duplicates using only builtin operations")
SOLUTION = "def similar_elements(a, b):\n    return tuple(sorted(set(a) & set(b) & set(a + b) & set(b + a)))\n"


def index():
    return chk.NgramIndex({"Mbpp/2": STATEMENT}), chk.NgramIndex({"HumanEval/0": SOLUTION * 2}, min_ngrams=3)


def sample(instruction, reply="```python\nx = 1\n```", source="magicoder"):
    return {"source": source, "messages": [{"role": "system", "content": "s"},
                                           {"role": "user", "content": instruction},
                                           {"role": "assistant", "content": reply}]}


def run(tmp_path, monkeypatch, samples, indexes=None):
    (tmp_path / "sft_train.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    monkeypatch.setattr(chk, "load_benchmarks", lambda: indexes or (None, None))
    return CliRunner().invoke(chk.main, ["--data-dir", str(tmp_path)])


def test_containment_catches_embedded_statement_that_jaccard_misses():
    stmt_index, _ = index()
    long_instruction = "Please help me. " + STATEMENT + " " + " ".join(f"filler{i}" for i in range(300))
    name, containment, jaccard = chk.worst_match(stmt_index, long_instruction)
    assert name == "Mbpp/2" and containment > 0.9 and jaccard < 0.3


def test_unrelated_text_has_no_match():
    stmt_index, _ = index()
    assert chk.worst_match(stmt_index, "Implement a binary search tree with insert and delete operations") is None


def test_short_benchmark_entries_are_ignored():
    assert chk.NgramIndex({"tiny": "add two numbers"}).grams == {}


def test_cli_fails_on_embedded_benchmark_statement(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch, [sample("Hi! " + STATEMENT)], index())
    assert result.exit_code == 1 and "problem statement" in result.output


def test_cli_fails_when_reply_contains_reference_solution(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch, [sample("Sort things", reply="```python\n" + SOLUTION * 2 + "```")], index())
    assert result.exit_code == 1 and "reference solution" in result.output


def test_cli_fails_on_humaneval_function_name_and_source_tag(tmp_path, monkeypatch):
    assert run(tmp_path, monkeypatch, [sample("Implement has_close_elements for me")]).exit_code == 1
    assert run(tmp_path, monkeypatch, [sample("Sort things", source="humaneval")]).exit_code == 1
    reply = "```python\ndef file_name_check(name):\n    return 'Yes'\n```"
    assert run(tmp_path, monkeypatch, [sample("Validate a file name", reply=reply)]).exit_code == 1


def test_cli_passes_clean_data(tmp_path, monkeypatch):
    clean = [sample("Implement a stack class with push and pop"), sample("Parse a CSV file into dictionaries")]
    result = run(tmp_path, monkeypatch, clean, index())
    assert result.exit_code == 0, result.output


def test_find_contaminated_reports_every_check():
    stmt_index, sol_index = index()
    samples = [
        sample("Sort things", source="humaneval"),                                      # 0 source tag
        sample("Please implement has_close_elements"),                                  # 1 name
        sample("Validate", reply="```python\ndef file_name_check(x):\n    pass\n```"),  # 2 code def
        sample("Hi! " + STATEMENT),                                                     # 3 statement
        sample("Sort", reply="```python\n" + SOLUTION * 2 + "```"),                     # 4 solution
        sample("Implement a stack class with push and pop"),                            # 5 clean
    ]
    flagged = chk.find_contaminated(samples, stmt_index, sol_index)
    assert set(flagged) == {0, 1, 2, 3, 4}
    assert [c for c, _ in flagged[0]] == ["source_tag"]
    assert flagged[1][0][0] == "instruction_name" and flagged[3][0][0] == "statement_overlap"
    assert flagged[4][0][0] == "solution_overlap"


def test_prepare_data_cap_is_per_source_and_deterministic():
    import prepare_sft_data as prep
    rows = [{"source": "magicoder", "messages": [{}, {"content": str(i)}, {}]} for i in range(50)] + \
           [{"source": "evol", "messages": [{}, {"content": str(i)}, {}]} for i in range(30)]
    kept = prep.cap_per_source(rows, {"magicoder": 10, "evol": 0}, seed=3)
    assert sum(r["source"] == "magicoder" for r in kept) == 10
    assert sum(r["source"] == "evol" for r in kept) == 30          # 0 = keep all
    assert kept == prep.cap_per_source(rows, {"magicoder": 10, "evol": 0}, seed=3)


def test_rendered_length_counts_template_overhead():
    import prepare_sft_data as prep

    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is False
            return " ".join(m["content"] for m in messages) + " <|im_end|>"

        def __call__(self, text):
            class Out:
                input_ids = text.split()
            return Out()

    sample = {"messages": [{"content": "sys prompt"}, {"content": "a b c"}, {"content": "d e"}]}
    assert prep.rendered_length(Tok(), sample) == 2 + 3 + 2 + 1
