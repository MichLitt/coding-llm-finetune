"""DPO training script — run on Colab A100 40GB.

Loads the best SFT checkpoint, adds a new LoRA adapter,
and trains with TRL DPOTrainer using binary preference pairs.

Usage (Colab):
    python scripts/dpo_train.py --sft-checkpoint results/sft_checkpoints/sft_C_with_targeted/final
    python scripts/dpo_train.py --sft-checkpoint <path> --beta 0.05
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
DPO_CFG_PATH = ROOT / "configs/dpo_config.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run generate_dpo_pairs.py first.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@click.command()
@click.option("--sft-checkpoint", required=True,
              help="Path to SFT final adapter directory.")
@click.option("--beta", default=None, type=float,
              help="DPO beta (KL penalty). Overrides config. Try 0.05, 0.1, 0.3.")
@click.option("--exp-id", default="dpo_v1", show_default=True)
def main(sft_checkpoint: str, beta: float | None, exp_id: str):
    try:
        import torch
        from unsloth import FastLanguageModel, PatchDPOTrainer
        PatchDPOTrainer()  # Must patch before importing DPOTrainer
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}")
        print("Install: pip install unsloth[colab-new] trl peft accelerate wandb bitsandbytes -q")
        sys.exit(1)

    cfg = _load_yaml(DPO_CFG_PATH)
    dpo_cfg   = cfg.get("dpo", {})
    lora_cfg  = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    ckpt_cfg  = cfg.get("checkpointing", {})

    effective_beta = beta if beta is not None else dpo_cfg.get("beta", 0.1)
    out_dir = ROOT / ckpt_cfg.get("output_dir", "results/dpo_checkpoints") / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(sft_checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: SFT checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DPO Training — {exp_id}")
    print(f"{'='*60}")
    print(f"SFT checkpoint : {ckpt_path}")
    print(f"beta           : {effective_beta}")
    print(f"Output dir     : {out_dir}")
    print()

    # ---- Load data ----
    train_raw = _load_jsonl(ROOT / cfg["data"]["train_file"])
    val_raw   = _load_jsonl(ROOT / cfg["data"]["val_file"])
    print(f"DPO train pairs: {len(train_raw):,}  val pairs: {len(val_raw):,}")

    # ---- Load SFT model ----
    print(f"\nLoading SFT model from {ckpt_path} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ckpt_path),
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=False,           # A100: bf16 full precision
        dtype=torch.bfloat16,
    )

    # Add fresh LoRA for DPO (smaller rank to avoid catastrophic forgetting)
    r = lora_cfg.get("r", 32)
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_alpha=r * 2,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none",
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=lora_cfg.get("random_state", 42),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable DPO parameters: {n_params/1e6:.2f}M")

    # ---- Format datasets ----
    # DPOTrainer expects: prompt, chosen, rejected
    def _fmt(examples):
        return {
            "prompt":   examples["prompt"],
            "chosen":   examples["chosen"],
            "rejected": examples["rejected"],
        }

    train_ds = Dataset.from_list(train_raw).map(_fmt, batched=False)
    val_ds   = Dataset.from_list(val_raw).map(_fmt, batched=False)

    # ---- DPO Trainer ----
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # Unsloth + LoRA: auto-create frozen reference
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
            seed=train_cfg.get("seed", 42),
            logging_steps=ckpt_cfg.get("logging_steps", 5),
            eval_steps=ckpt_cfg.get("eval_steps", 20),
            save_steps=ckpt_cfg.get("save_steps", 50),
            save_total_limit=ckpt_cfg.get("save_total_limit", 2),
            report_to=ckpt_cfg.get("report_to", "none"),
            run_name=exp_id,
        ),
    )

    # Monitor reward margin during training
    print("\nStarting DPO training …")
    print("(Watch: chosen_rewards should stay > rejected_rewards, margin should grow)")
    trainer.train()

    # ---- Save ----
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    meta = {
        "exp_id": exp_id,
        "sft_checkpoint": str(sft_checkpoint),
        "beta": effective_beta,
        "lora_r": r,
        "train_pairs": len(train_raw),
        "val_pairs": len(val_raw),
    }
    (final_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDPO model saved to {final_dir}")

    # Quick sanity: eval on val set
    print("\nRunning final eval …")
    trainer.evaluate()
    print("Done.")


if __name__ == "__main__":
    main()
