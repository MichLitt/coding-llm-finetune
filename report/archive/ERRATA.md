# Errata for the archived v2 reports

Written during the v3.1 audit (2026-09). The archived reports are left untouched; read them with these corrections.

## 1. Four "targeted" HumanEval task ids pointed at the wrong problems

`SFT_Evaluation_Report.md` and `DPO_Evaluation_Report.md` attribute per-task results to
"HumanEval/N (function_name)". Checked against `openai/openai_humaneval`, four ids were wrong:

| Id used in the reports | Function the report named | What that id actually is | Correct id of the named function |
|---|---|---|---|
| HumanEval/107 | minSubArraySum | `even_odd_palindrome` | HumanEval/114 |
| HumanEval/108 | intersection | `count_nums` | HumanEval/127 |
| HumanEval/110 | prod_signs | `exchange` | HumanEval/128 |
| HumanEval/128 | tri | `prod_signs` | HumanEval/130 |

(HumanEval/54, /55, /141 and /147 are correct.) The evaluation script looked tasks up by id, so the
tabulated pass/fail values for these four rows describe the *other* problems. Conclusions such as
"HE/108 (intersection) and HE/128 (tri) never pass in any version" are therefore not supported.

## 2. The 8 "failure cases" belong to a different model

The eight tasks were failures of a separate Coder Agent, not of Qwen3.5. They are not a valid
description of this model's weaknesses, and training data derived from them (with solution hints in
the generator prompts) would have leaked benchmark answers. v3.1 drops targeted data entirely.

## 3. Evaluation harness caveats for the v2 numbers

Checked against `notebooks/archive/04_humaneval_eval.ipynb` and `05_eval_colab.ipynb`:

* Completions were post-processed with regexes (strip fences and `<think>` blocks, cut an echoed
  function header, re-indent). This is fragile: it rewrites replies heuristically instead of
  extracting a definition, and there was no failure taxonomy, so truncation, syntax errors and
  wrong answers were indistinguishable.
* `enable_thinking` was never passed, so the chat template's default (thinking on) applied while
  generation was capped at 512 new tokens, which truncates thinking replies. The SFT text, in
  contrast, is rendered by the template without a reasoning block. Baseline and fine-tuned models
  may therefore have been evaluated under different modes.

(The current v3 pre-audit script `scripts/run_humaneval.py`, removed in v3.1, went further and executed
the raw chat reply with no extraction at all.)

The v3.1 harness (`scripts/codegen_common.py`, `scripts/run_eval.py`) uses one prompt, one extraction
routine, non-thinking mode, 1024 new tokens, and a per-problem failure taxonomy. v2 pass@1 values
should not be compared with v3.1 values.
