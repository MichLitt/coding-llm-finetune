"""SFT training script for local 8 GB or Colab A100/H100 runs.

Loads Qwen3.5-4B via Unsloth, attaches LoRA, trains with
TRL SFTTrainer, and saves the adapter.

Usage:
    python scripts/sft_train.py --exp-id sft_generic_aggr
    python scripts/sft_train.py --config-path configs/sft_config_colab_a100.yaml --exp-id sft_generic_cons
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from train_utils import resolve_output_dir, resolve_resume, resolve_sft_hparams  # noqa: E402

load_dotenv()

ROOT = Path(__file__).parent.parent
DEFAULT_SFT_CFG_PATH = ROOT / "configs/sft_config.yaml"
ABLATION_CFG_PATH = ROOT / "configs/ablation_matrix.yaml"


def _patch_dill_for_py314() -> None:
    """Disable datasets caching to avoid dill incompatibilities on Python 3.14."""
    try:
        import datasets as _ds

        _ds.disable_caching()
    except Exception:
        pass


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}\nRun scripts/prepare_sft_data.py first.")

    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _filter_by_sources(samples: list[dict], sources: list[str]) -> list[dict]:
    if not sources:
        return samples
    return [sample for sample in samples if sample.get("source", "") in sources]


def _resolve_cfg_path(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"SFT config not found: {path}")
    return path


def _resolve_torch_dtype(torch_module, dtype_name: str):
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype in config: {dtype_name}") from exc


def _build_sft_config(config_cls, kwargs: dict):
    """Drop kwargs unsupported by the installed TRL version."""
    valid = set(inspect.signature(config_cls.__init__).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in valid}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"SFTConfig compatibility: ignoring unsupported args {dropped}")
    return config_cls(**filtered)


@click.command()
@click.option("--exp-id", default="sft_generic_aggr", show_default=True,
              help="Experiment ID from configs/ablation_matrix.yaml")
@click.option("--config-path", default=str(DEFAULT_SFT_CFG_PATH), show_default=True,
              help="Path to SFT YAML config.")
@click.option("--data-sources", default=None,
              help="Comma-separated data sources to include (overrides ablation matrix).")
@click.option("--lora-r", default=None, type=int, help="LoRA rank (overrides matrix).")
@click.option("--lora-alpha", default=None, type=int, help="LoRA alpha (overrides matrix; default 2*r).")
@click.option("--learning-rate", default=None, type=float, help="Learning rate (overrides matrix).")
@click.option("--epochs", default=None, type=int, help="Number of training epochs (overrides matrix).")
@click.option("--resume-from", default=None,
              help="Checkpoint path to resume from, or 'auto' for the newest checkpoint in the output dir.")
def main(
    exp_id: str,
    config_path: str,
    data_sources: str | None,
    lora_r: int | None,
    lora_alpha: int | None,
    learning_rate: float | None,
    epochs: int | None,
    resume_from: str | None,
) -> None:
    try:
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install unsloth[colab-new] trl peft accelerate wandb bitsandbytes -q")
        sys.exit(1)

    _patch_dill_for_py314()

    cfg_path = _resolve_cfg_path(config_path)
    base_cfg = _load_yaml(cfg_path)
    ablation = _load_yaml(ABLATION_CFG_PATH).get("experiments", {})
    if exp_id not in ablation:
        print(f"ERROR: unknown experiment '{exp_id}'. Known: {sorted(ablation)}")
        sys.exit(1)
    exp_cfg = ablation[exp_id]

    hp = resolve_sft_hparams(
        base_cfg, exp_cfg,
        data_sources=[item.strip() for item in data_sources.split(",") if item.strip()] if data_sources else None,
        lora_r=lora_r, lora_alpha=lora_alpha, learning_rate=learning_rate, epochs=epochs,
    )
    sources, r, n_epochs = hp["data_sources"], hp["lora_r"], hp["num_train_epochs"]

    model_cfg = base_cfg["model"]
    train_cfg = base_cfg["training"]
    ckpt_cfg = base_cfg["checkpointing"]
    lora_cfg = base_cfg["lora"]
    data_cfg = base_cfg["data"]

    model_name = model_cfg["name"]
    max_seq_len = model_cfg["max_seq_length"]
    dtype = _resolve_torch_dtype(torch, model_cfg.get("dtype", "bfloat16"))

    out_dir = resolve_output_dir(ROOT, ckpt_cfg["output_dir"], exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_from = resolve_resume(out_dir, resume_from)

    print(f"\n{'=' * 60}")
    print(f"SFT Training | {exp_id}")
    print(f"{'=' * 60}")
    print(f"Config path : {cfg_path}")
    print(f"Model       : {model_name}")
    print(f"LoRA r/alpha: {r}/{hp['lora_alpha']}")
    print(f"Learning rate: {hp['learning_rate']}")
    print(f"Epochs      : {n_epochs}")
    print(f"Sources     : {sources}")
    print(f"Output dir  : {out_dir}")
    print()

    train_path = ROOT / data_cfg["train_file"]
    val_path = ROOT / data_cfg["val_file"]
    train_raw = _filter_by_sources(_load_jsonl(train_path), sources)
    val_raw = _filter_by_sources(_load_jsonl(val_path), sources)
    print(f"Train samples: {len(train_raw):,}  Val samples: {len(val_raw):,}")

    print(f"\nLoading {model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        dtype=dtype,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=lora_cfg["target_modules"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
    )

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_train_params / 1e6:.2f}M")

    def to_text(sample: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_ds = Dataset.from_list([to_text(sample) for sample in train_raw])
    val_ds = Dataset.from_list([to_text(sample) for sample in val_raw])

    report_to = ckpt_cfg.get("report_to", "none")
    if report_to == "wandb" and not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set; disabling wandb logging.")
        report_to = "none"

    sft_config_kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "num_train_epochs": n_epochs,
        "learning_rate": hp["learning_rate"],
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "weight_decay": train_cfg["weight_decay"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "bf16": train_cfg.get("bf16", False),
        "tf32": train_cfg.get("tf32", True),
        "optim": train_cfg["optim"],
        "dataloader_num_workers": train_cfg.get("dataloader_num_workers", 0),
        "group_by_length": train_cfg.get("group_by_length", False),
        "seed": train_cfg["seed"],
        "eval_strategy": "steps",
        "logging_steps": ckpt_cfg["logging_steps"],
        "eval_steps": ckpt_cfg["eval_steps"],
        "save_steps": ckpt_cfg["save_steps"],
        "save_total_limit": ckpt_cfg["save_total_limit"],
        "load_best_model_at_end": ckpt_cfg["load_best_model_at_end"],
        "metric_for_best_model": ckpt_cfg["metric_for_best_model"],
        "report_to": report_to,
        "dataset_text_field": data_cfg.get("dataset_text_field", "text"),
        "max_seq_length": max_seq_len,
        "run_name": exp_id,
    }

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=_build_sft_config(SFTConfig, sft_config_kwargs),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print(f"\nStarting training ({n_epochs} epoch(s)) ...")
    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    meta = {
        "exp_id": exp_id,
        "config_path": str(cfg_path),
        "model_name": model_name,
        "hyperparameters": hp,
        "lora_r": r,
        "lora_alpha": hp["lora_alpha"],
        "learning_rate": hp["learning_rate"],
        "num_epochs": n_epochs,
        "data_sources": sources,
        "train_samples": len(train_raw),
        "val_samples": len(val_raw),
    }
    (final_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nModel saved to {final_dir}")
    print(f"Training complete for experiment: {exp_id}")


if __name__ == "__main__":
    main()
