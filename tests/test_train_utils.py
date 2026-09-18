import json

import train_utils as tu
from model_io import adapter_chain

GOOD_PROMPT = ("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def row(**kw):
    base = {"prompt": GOOD_PROMPT, "chosen": "```python\ndef f():\n    return 1\n```", "rejected": "```python\ndef f():\n    return 2\n```"}
    base.update(kw)
    return base


def test_dpo_format_accepts_rendered_non_thinking_rows():
    assert tu.check_dpo_format([row()]) == []


def test_dpo_format_rejects_plain_string_prompts():
    problems = tu.check_dpo_format([row(prompt="Write a Python function to solve the following:\n\nX")])
    assert any("system turn" in p for p in problems) and any("non-thinking" in p for p in problems)


def test_dpo_format_rejects_thinking_prompt_and_control_tokens():
    thinking = GOOD_PROMPT.replace("<think>\n\n</think>\n\n", "<think>\n")
    assert tu.check_dpo_format([row(prompt=thinking)])
    assert tu.check_dpo_format([row(chosen="x<|im_end|>")])
    assert tu.check_dpo_format([row(rejected="<think>hmm</think> code")])
    assert tu.check_dpo_format([row(rejected="  ")])


def test_prompt_matches_inference_prompt_with_real_tokenizer():
    """The SFT training text must extend the exact non-thinking inference prompt."""
    import pytest
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except Exception as exc:  # offline CI without the hub cache
        pytest.skip(f"tokenizer unavailable: {exc}")
    import codegen_common as cc
    msgs = cc.mbpp_messages("Write f.", "assert f(1) == 1", "f")
    prompt = cc.render_prompt(tok, msgs)
    assert prompt.endswith("<|im_start|>assistant\n" + tu.NON_THINKING_SUFFIX)
    assert tu.check_dpo_format([{"prompt": prompt, "chosen": "a", "rejected": "b"}]) == []
    reply = "```python\ndef f(x):\n    return x\n```"
    sft_text = tok.apply_chat_template(msgs + [{"role": "assistant", "content": reply}], tokenize=False)
    assert sft_text.startswith(prompt), "SFT and inference formats diverge"
    assert tok.eos_token == "<|im_end|>"


def test_dynamics_detects_likelihood_displacement():
    logs = [{"logps/chosen": -50.0, "rewards/margins": 0.0}, {"logps/chosen": -60.0, "rewards/margins": 1.0},
            {"logps/chosen": -80.0, "rewards/margins": 3.0}, {"step": 5}]
    out = tu.summarize_dynamics(logs, window=1)
    assert out["chosen_logp_decreased"] is True
    assert out["rewards/margins"] == {"first": 0.0, "last": 3.0}
    assert tu.summarize_dynamics([])["n_logs"] == 0


def test_latest_checkpoint_and_auto_resume(tmp_path):
    assert tu.resolve_resume(tmp_path, "auto") is None
    for step in (25, 100, 50):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    assert tu.latest_checkpoint(tmp_path).name == "checkpoint-100"
    assert tu.resolve_resume(tmp_path, "auto").endswith("checkpoint-100")
    assert tu.resolve_resume(tmp_path, "some/path") == "some/path"
    assert tu.resolve_resume(tmp_path, None) is None


def test_output_dir_honours_env_root(tmp_path, monkeypatch):
    monkeypatch.delenv(tu.OUTPUT_ROOT_ENV, raising=False)
    assert tu.resolve_output_dir(tmp_path, "results/x", "e1") == tmp_path / "results/x/e1"
    monkeypatch.setenv(tu.OUTPUT_ROOT_ENV, str(tmp_path / "drive"))
    assert tu.resolve_output_dir(tmp_path, "results/x", "e1") == tmp_path / "drive/results/x/e1"


def test_adapter_chain_reapplies_sft_before_dpo(tmp_path):
    sft, dpo = tmp_path / "sft", tmp_path / "dpo"
    for d in (sft, dpo):
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
    (dpo / "experiment_meta.json").write_text(json.dumps({"sft_checkpoint": str(sft)}))
    assert adapter_chain(str(dpo)) == [str(sft), str(dpo)]
    (dpo / "experiment_meta.json").write_text(json.dumps({"sft_checkpoint": "base"}))
    assert adapter_chain(str(dpo)) == [str(dpo)]
    assert adapter_chain(str(sft)) == [str(sft)]


def test_hparams_come_from_matrix_and_keep_alpha_consistent():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    matrix = yaml.safe_load((root / "configs/ablation_matrix.yaml").read_text())["experiments"]
    base = yaml.safe_load((root / "configs/sft_config_colab_a100.yaml").read_text())
    assert set(matrix) == {"sft_generic_aggr", "sft_generic_cons"}
    for name, exp in matrix.items():
        hp = tu.resolve_sft_hparams(base, exp)
        assert hp["lora_alpha"] == 2 * hp["lora_r"], name
        assert "targeted" not in hp["data_sources"]
    aggr = tu.resolve_sft_hparams(base, matrix["sft_generic_aggr"])
    cons = tu.resolve_sft_hparams(base, matrix["sft_generic_cons"])
    assert (aggr["learning_rate"], aggr["num_train_epochs"]) == (2e-4, 3)
    assert (cons["learning_rate"], cons["num_train_epochs"]) == (5e-5, 1)


def test_changing_r_resets_stale_alpha():
    base = {"lora": {"r": 64, "lora_alpha": 128}, "training": {"learning_rate": 2e-4, "num_train_epochs": 3}}
    assert tu.resolve_sft_hparams(base, {"lora_r": 32})["lora_alpha"] == 64      # old bug: stayed 128
    assert tu.resolve_sft_hparams(base, {})["lora_alpha"] == 128                  # r unchanged -> keep
    assert tu.resolve_sft_hparams(base, {"lora_r": 32}, lora_alpha=16)["lora_alpha"] == 16
    assert tu.resolve_sft_hparams(base, {"lora_r": 16}, learning_rate=1e-5, epochs=2)["learning_rate"] == 1e-5


def test_colab_profiles_are_identical_in_training_settings():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "configs"
    a = yaml.safe_load((root / "sft_config_colab_a100.yaml").read_text())
    h = yaml.safe_load((root / "sft_config_colab_h100.yaml").read_text())
    assert a == h
    assert a["checkpointing"]["save_steps"] <= 50 and a["checkpointing"]["eval_steps"] <= 50
