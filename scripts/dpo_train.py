"""DPO training script for local 8 GB or Colab A100/H100 runs.

Loads an SFT adapter or full checkpoint, attaches a fresh LoRA adapter, trains
with TRL DPOTrainer, and saves the resulting adapter.

Usage:
    python scripts/dpo_train.py --sft-checkpoint results/sft_checkpoints/sft_targeted/final
    python scripts/dpo_train.py --config-path configs/dpo_config_colab_a100.yaml --sft-checkpoint ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DEFAULT_DPO_CFG_PATH = ROOT / "configs/dpo_config.yaml"
DEFAULT_BASE_MODEL = "unsloth/Qwen3.5-4B"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run generate_dpo_pairs.py first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_cfg_path(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"DPO config not found: {path}")
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


@click.command()
@click.option("--sft-checkpoint", required=True,
              help="Path to SFT final adapter directory.")
@click.option("--config-path", default=str(DEFAULT_DPO_CFG_PATH), show_default=True,
              help="Path to DPO YAML config.")
@click.option("--beta", default=None, type=float,
              help="DPO beta (KL penalty). Overrides config.")
@click.option("--exp-id", default="dpo_v1", show_default=True)
def main(sft_checkpoint: str, config_path: str, beta: float | None, exp_id: str) -> None:
    try:
        import torch
        from unsloth import FastLanguageModel, PatchDPOTrainer
        from datasets import Dataset
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer

        PatchDPOTrainer()
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install: pip install unsloth[colab-new] trl peft accelerate wandb bitsandbytes -q")
        sys.exit(1)

    cfg_path = _resolve_cfg_path(config_path)
    cfg = _load_yaml(cfg_path)
    model_cfg = cfg.get("model", {})
    dpo_cfg = cfg.get("dpo", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpointing", {})
    data_cfg = cfg.get("data", {})

    base_model_name = model_cfg.get("name", DEFAULT_BASE_MODEL)
    dtype = _resolve_torch_dtype(torch, model_cfg.get("dtype", "bfloat16"))
    effective_beta = beta if beta is not None else dpo_cfg.get("beta", 0.1)
    load_in_4bit = model_cfg.get("load_in_4bit", True)
    out_dir = ROOT / ckpt_cfg.get("output_dir", "results/dpo_checkpoints") / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(sft_checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: SFT checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"DPO Training | {exp_id}")
    print(f"{'=' * 60}")
    print(f"Config path     : {cfg_path}")
    print(f"SFT checkpoint  : {ckpt_path}")
    print(f"Base model      : {base_model_name}")
    print(f"load_in_4bit    : {load_in_4bit}")
    print(f"beta            : {effective_beta}")
    print(f"Output dir      : {out_dir}")
    print()

    train_raw = _load_jsonl(ROOT / data_cfg["train_file"])
    val_raw = _load_jsonl(ROOT / data_cfg["val_file"])
    print(f"DPO train pairs: {len(train_raw):,}  val pairs: {len(val_raw):,}")

    print(f"\nLoading SFT model from {ckpt_path} ...")
    is_adapter_only = not (ckpt_path / "config.json").exists()
    if is_adapter_only:
        print(f"Detected LoRA adapter. Loading base model {base_model_name} first.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=model_cfg.get("max_seq_length", 2048),
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )
        model = PeftModel.from_pretrained(model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(ckpt_path),
            max_seq_length=model_cfg.get("max_seq_length", 2048),
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )

    r = lora_cfg.get("r", 16)
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=lora_cfg.get("lora_alpha", r * 2),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=lora_cfg.get("random_state", 42),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable DPO parameters: {n_params / 1e6:.2f}M")

    train_ds = Dataset.from_list(train_raw)
    val_ds = Dataset.from_list(val_raw)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=DPOConfig(
            output_dir=str(out_dir),
            beta=effective_beta,
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
            max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
            max_length=dpo_cfg.get("max_length", 1536),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
            num_train_epochs=train_cfg.get("num_train_epochs", 1),
            learning_rate=train_cfg.get("learning_rate", 5e-5),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
            bf16=train_cfg.get("bf16", True),
            tf32=train_cfg.get("tf32", True),
            optim=train_cfg.get("optim", "adamw_8bit"),
            dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
            seed=train_cfg.get("seed", 42),
            logging_steps=ckpt_cfg.get("logging_steps", 5),
            eval_strategy="steps",
            eval_steps=ckpt_cfg.get("eval_steps", 20),
            save_steps=ckpt_cfg.get("save_steps", 50),
            save_total_limit=ckpt_cfg.get("save_total_limit", 2),
            report_to=ckpt_cfg.get("report_to", "none"),
            run_name=exp_id,
        ),
    )

    print("\nStarting DPO training ...")
    print("(Watch the reward margin: chosen should stay above rejected.)")
    trainer.train()

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    meta = {
        "exp_id": exp_id,
        "config_path": str(cfg_path),
        "sft_checkpoint": str(sft_checkpoint),
        "base_model": base_model_name,
        "beta": effective_beta,
        "lora_r": r,
        "train_pairs": len(train_raw),
        "val_pairs": len(val_raw),
    }
    (final_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDPO model saved to {final_dir}")
    print("\nRunning final eval ...")
    trainer.evaluate()
    print("Done.")


if __name__ == "__main__":
    main()
