"""SFT training script — run on Colab A100 40GB.

Loads Qwen2.5-Coder-7B-Instruct via Unsloth, attaches LoRA, trains
with TRL SFTTrainer, and saves the adapter.

Usage (Colab):
    python scripts/sft_train.py --exp-id sft_C_with_targeted
    python scripts/sft_train.py --exp-id sft_A_magicoder_only --data-sources magicoder

The --exp-id argument selects defaults from configs/ablation_matrix.yaml;
individual flags override those defaults.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
SFT_CFG_PATH     = ROOT / "configs/sft_config.yaml"
ABLATION_CFG_PATH = ROOT / "configs/ablation_matrix.yaml"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write clean, efficient, and correct code. "
    "Always handle edge cases and include brief comments for complex logic."
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}\nRun scripts/prepare_sft_data.py first.")
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def _filter_by_sources(samples: list[dict], sources: list[str]) -> list[dict]:
    if not sources:
        return samples
    return [s for s in samples if s.get("source", "") in sources]


@click.command()
@click.option("--exp-id", default="sft_C_with_targeted", show_default=True,
              help="Experiment ID from configs/ablation_matrix.yaml")
@click.option("--data-sources", default=None,
              help="Comma-separated data sources to include (overrides ablation matrix).")
@click.option("--lora-r", default=None, type=int,
              help="LoRA rank (overrides config).")
@click.option("--epochs", default=None, type=int,
              help="Number of training epochs (overrides config).")
@click.option("--resume-from", default=None,
              help="Path to checkpoint to resume from.")
def main(exp_id: str, data_sources: str | None, lora_r: int | None, epochs: int | None, resume_from: str | None):
    try:
        import torch
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}")
        print("Install with: pip install unsloth[colab-new] trl peft accelerate wandb bitsandbytes -q")
        sys.exit(1)

    # ---- Load configs ----
    base_cfg    = _load_yaml(SFT_CFG_PATH)
    ablation    = _load_yaml(ABLATION_CFG_PATH).get("experiments", {})
    exp_cfg     = ablation.get(exp_id, {})

    # Merge: base < exp_cfg < CLI overrides
    sources = (
        data_sources.split(",") if data_sources
        else exp_cfg.get("data_sources", ["magicoder", "evol", "targeted"])
    )
    r = lora_r or exp_cfg.get("lora_r", base_cfg.get("lora", {}).get("r", 64))
    n_epochs = epochs or exp_cfg.get("num_train_epochs", base_cfg.get("training", {}).get("num_train_epochs", 3))

    model_name = base_cfg["model"]["name"]
    max_seq_len = base_cfg["model"]["max_seq_length"]
    train_cfg  = base_cfg["training"]
    ckpt_cfg   = base_cfg["checkpointing"]
    lora_cfg   = base_cfg["lora"]

    out_dir = ROOT / ckpt_cfg["output_dir"] / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SFT Training — {exp_id}")
    print(f"{'='*60}")
    print(f"Model      : {model_name}")
    print(f"LoRA rank  : {r}")
    print(f"Epochs     : {n_epochs}")
    print(f"Sources    : {sources}")
    print(f"Output dir : {out_dir}")
    print()

    # ---- Load data ----
    train_path = ROOT / base_cfg["data"]["train_file"]
    val_path   = ROOT / base_cfg["data"]["val_file"]
    train_raw = _filter_by_sources(_load_jsonl(train_path), sources)
    val_raw   = _filter_by_sources(_load_jsonl(val_path),   sources)

    print(f"Train samples: {len(train_raw):,}  Val samples: {len(val_raw):,}")

    # ---- Load model ----
    print(f"\nLoading {model_name} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=base_cfg["model"]["load_in_4bit"],
        dtype=getattr(torch, base_cfg["model"]["dtype"].replace("bfloat16", "bfloat16")),
    )

    # ---- LoRA ----
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=lora_cfg["target_modules"],
        lora_alpha=r * 2,            # alpha = 2 × r
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
    )

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_train_params/1e6:.2f}M")

    # ---- Format datasets ----
    def format_fn(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list(train_raw).map(format_fn, remove_columns=["messages", "source", "task_id"])
    val_ds   = Dataset.from_list(val_raw).map(format_fn, remove_columns=["messages", "source", "task_id"])

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            num_train_epochs=n_epochs,
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            max_grad_norm=train_cfg["max_grad_norm"],
            bf16=train_cfg["bf16"],
            tf32=train_cfg.get("tf32", True),
            optim=train_cfg["optim"],
            dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
            seed=train_cfg["seed"],
            eval_strategy="steps",
            logging_steps=ckpt_cfg["logging_steps"],
            eval_steps=ckpt_cfg["eval_steps"],
            save_steps=ckpt_cfg["save_steps"],
            save_total_limit=ckpt_cfg["save_total_limit"],
            load_best_model_at_end=ckpt_cfg["load_best_model_at_end"],
            metric_for_best_model=ckpt_cfg["metric_for_best_model"],
            report_to=ckpt_cfg.get("report_to", "none"),
            dataset_text_field="text",
            max_seq_length=max_seq_len,
            run_name=exp_id,
        ),
    )

    # Eval before training (baseline reference)
    print("\nRunning pre-training eval …")
    trainer.evaluate()

    # ---- Train ----
    print(f"\nStarting training ({n_epochs} epoch(s)) …")
    trainer.train(resume_from_checkpoint=resume_from)

    # ---- Save ----
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save experiment metadata
    meta = {
        "exp_id": exp_id,
        "model_name": model_name,
        "lora_r": r,
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
