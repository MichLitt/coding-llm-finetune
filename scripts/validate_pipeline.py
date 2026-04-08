"""Local pipeline validation using Qwen3.5-4B-Instruct (4-bit quantized).

Checks that the full SFT training loop works without errors before
running full-scale training on the complete dataset.

Requirements (install separately — not in base uv env):
    pip install unsloth[colab-new] trl peft accelerate bitsandbytes

Usage:
    uv run python scripts/validate_pipeline.py
    uv run python scripts/validate_pipeline.py --data-dir data/processed --num-samples 50
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import click

ROOT = Path(__file__).parent.parent

VALIDATION_CONFIG = {
    "model_name": "unsloth/Qwen3.5-4B-Instruct",
    "load_in_4bit": True,
    "max_seq_length": 1024,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "output_dir": "results/validate_checkpoint",
}


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return ok


def _load_samples(data_dir: Path, n: int) -> list[dict]:
    train_path = data_dir / "sft_train.jsonl"
    if not train_path.exists():
        return []
    samples = []
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(json.loads(line))
        if len(samples) >= n:
            break
    return samples


@click.command()
@click.option("--data-dir", default="data/processed", show_default=True)
@click.option("--num-samples", default=100, show_default=True,
              help="Number of training samples to use for validation.")
@click.option("--num-val-samples", default=20, show_default=True)
def main(data_dir: str, num_samples: int, num_val_samples: int):
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. Run: uv sync --extra eval")
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError:
        print("ERROR: unsloth/trl not installed.")
        print("Install with: pip install unsloth[colab-new] trl peft accelerate bitsandbytes")
        sys.exit(1)

    data_path = ROOT / data_dir
    all_train = _load_samples(data_path, num_samples)

    val_path = data_path / "sft_val.jsonl"
    if val_path.exists():
        val_samples = [
            json.loads(l) for l in val_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ][:num_val_samples]
    else:
        val_samples = all_train[:num_val_samples]
        all_train   = all_train[num_val_samples:]

    print(f"\n{'='*60}")
    print("Pipeline Validation — Qwen3.5-4B-Instruct (4-bit)")
    print(f"{'='*60}")
    print(f"Train samples : {len(all_train)}")
    print(f"Val   samples : {len(val_samples)}")
    print()

    checks_passed = 0
    total_checks  = 8

    # ---- Check 1: Model load ----
    print("(1/8) Loading model with 4-bit quantization …")
    try:
        cfg = VALIDATION_CONFIG
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model_name"],
            max_seq_length=cfg["max_seq_length"],
            load_in_4bit=cfg["load_in_4bit"],
        )
        checks_passed += _check("Model loaded (4-bit)", True,
                                f"{cfg['model_name']}")
    except Exception as e:
        _check("Model loaded (4-bit)", False, str(e))
        print("\nCannot continue — model load failed.")
        sys.exit(1)

    # ---- Check 2: LoRA attach ----
    print("(2/8) Attaching LoRA adapter …")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            target_modules=cfg["target_modules"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        checks_passed += _check("LoRA attached", True,
                                f"{n_params/1e6:.2f}M trainable params")
    except Exception as e:
        _check("LoRA attached", False, str(e))
        sys.exit(1)

    # ---- Check 3: Data formatting ----
    print("(3/8) Formatting samples with chat template …")
    try:
        def format_sample(example):
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}

        train_ds = Dataset.from_list(all_train).map(format_sample)
        val_ds   = Dataset.from_list(val_samples).map(format_sample)

        sample_text = train_ds[0]["text"]
        has_assistant = "<|im_start|>assistant" in sample_text or "assistant" in sample_text
        checks_passed += _check("Chat template formatting", has_assistant,
                                f"Sample length={len(sample_text)} chars")
    except Exception as e:
        _check("Chat template formatting", False, str(e))
        sys.exit(1)

    # ---- Check 4-6: Training loop ----
    print("(4/8) Running forward pass …")
    print("(5/8) Running backward pass …")
    print("(6/8) Running full training epoch …")
    try:
        out_dir = ROOT / cfg["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=SFTConfig(
                output_dir=str(out_dir),
                per_device_train_batch_size=cfg["per_device_train_batch_size"],
                gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                num_train_epochs=cfg["num_train_epochs"],
                learning_rate=cfg["learning_rate"],
                bf16=torch.cuda.is_bf16_supported(),
                fp16=not torch.cuda.is_bf16_supported(),
                logging_steps=5,
                eval_steps=20,
                save_steps=50,
                save_total_limit=1,
                dataset_text_field="text",
                max_seq_length=cfg["max_seq_length"],
                seed=42,
                report_to="none",
            ),
        )

        # Only compute loss on assistant response tokens
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

        train_result = trainer.train()
        train_loss = train_result.training_loss
        ok_forward  = not math.isnan(train_loss) and not math.isinf(train_loss)
        ok_backward = train_result.global_step > 0

        checks_passed += _check("Forward pass (loss not NaN/Inf)", ok_forward,
                                f"train_loss={train_loss:.4f}")
        checks_passed += _check("Backward pass (steps completed)", ok_backward,
                                f"steps={train_result.global_step}")
        checks_passed += _check("Full epoch completed", True,
                                f"loss={train_loss:.4f}")
    except Exception as e:
        _check("Forward pass (loss not NaN/Inf)", False, str(e))
        _check("Backward pass (steps completed)", False, "")
        _check("Full epoch completed", False, "")
        sys.exit(1)

    # ---- Check 7: Checkpoint save ----
    print("(7/8) Saving checkpoint …")
    try:
        model.save_pretrained(str(out_dir / "lora_adapter"))
        tokenizer.save_pretrained(str(out_dir / "lora_adapter"))
        adapter_exists = (out_dir / "lora_adapter" / "adapter_model.safetensors").exists() or \
                         (out_dir / "lora_adapter" / "adapter_model.bin").exists() or \
                         any((out_dir / "lora_adapter").iterdir())
        checks_passed += _check("Checkpoint saved", adapter_exists,
                                str(out_dir / "lora_adapter"))
    except Exception as e:
        _check("Checkpoint saved", False, str(e))

    # ---- Check 8: Inference ----
    print("(8/8) Running sample inference …")
    try:
        FastLanguageModel.for_inference(model)
        test_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are an expert Python programmer."},
                {"role": "user", "content": "Write a function to add two numbers."},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        has_code = "def " in generated or "return" in generated or "+" in generated
        checks_passed += _check("Inference output contains code", has_code,
                                repr(generated[:80]))
    except Exception as e:
        _check("Inference output contains code", False, str(e))

    # ---- VRAM report ----
    if torch.cuda.is_available():
        used_gb  = torch.cuda.max_memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nPeak VRAM: {used_gb:.2f} GB / {total_gb:.2f} GB")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Result: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("ALL CHECKS PASSED — ready to run full training")
    else:
        print(f"FAILED {total_checks - checks_passed} checks — fix issues before training")
    print(f"{'='*60}\n")

    sys.exit(0 if checks_passed == total_checks else 1)


if __name__ == "__main__":
    main()
