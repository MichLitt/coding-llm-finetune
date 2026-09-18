"""Model loading and text generation backends (Colab/GPU side).

All heavy imports are lazy so the module can be imported (and its pure helpers
unit-tested) on a CPU-only machine.

Every backend exposes the same interface::

    gen = load_generator(model_path, backend="unsloth")
    outs = gen.generate(prompts, max_new_tokens=1024, temperature=0.0)  # list[GenOut]

Models are always run in bf16 (never 4-bit) so evaluation matches training
precision (plan D5). ``model_path`` may be a HF repo id, a full model directory
or a LoRA adapter directory; adapters are attached to ``base_model``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_MODEL = "unsloth/Qwen3.5-4B"  # mirror of Qwen/Qwen3.5-4B (post-trained release)
MAX_SEQ_LENGTH = 4096


@dataclass
class GenOut:
    text: str
    n_tokens: int
    finished: bool  # an end-of-turn/EOS token was generated (False => hit max_new_tokens)


def is_adapter_dir(model_path: str) -> bool:
    """A LoRA adapter directory has adapter_config.json and no full config.json."""
    path = Path(model_path)
    return path.is_dir() and (path / "adapter_config.json").exists() and not (path / "config.json").exists()


def adapter_base_model(model_path: str, default: str = DEFAULT_BASE_MODEL) -> str:
    cfg = Path(model_path) / "adapter_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get("base_model_name_or_path") or default
    return default


def adapter_chain(model_path: str) -> list[str]:
    """Adapters to apply, in order, to rebuild ``model_path``.

    A DPO adapter trained on top of an SFT adapter only contains the DPO delta (the
    SFT weights were merged into the frozen base before the new LoRA was attached),
    so evaluation must re-apply the SFT adapter first. The link is recorded in
    ``experiment_meta.json`` by dpo_train.py.
    """
    meta_path = Path(model_path) / "experiment_meta.json"
    chain: list[str] = []
    if meta_path.exists():
        parent = json.loads(meta_path.read_text(encoding="utf-8")).get("sft_checkpoint")
        if parent and parent != "base" and is_adapter_dir(parent):
            chain = adapter_chain(parent)
    return chain + [model_path]


def _text_tokenizer(tokenizer):
    """Multimodal checkpoints return a Processor; we only need its tokenizer."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def _eos_ids(tokenizer, model=None) -> set[int]:
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            ids.add(tid)
    gen_cfg = getattr(model, "generation_config", None)
    extra = getattr(gen_cfg, "eos_token_id", None)
    if isinstance(extra, int):
        ids.add(extra)
    elif isinstance(extra, (list, tuple)):
        ids.update(int(x) for x in extra)
    return ids


class HFGenerator:
    """Batched generation with transformers (works for Unsloth-loaded models too)."""

    def __init__(self, model, tokenizer, batch_size: int = 16):
        self.model = model
        self.tokenizer = _text_tokenizer(tokenizer)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.eos_ids = _eos_ids(self.tokenizer, model)

    def generate(self, prompts: list[str], max_new_tokens: int = 1024,
                 temperature: float = 0.0, top_p: float = 0.95, seed: int = 0) -> list[GenOut]:
        import torch

        torch.manual_seed(seed)
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        results: list[GenOut | None] = [None] * len(prompts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            batch = [prompts[i] for i in idx]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                 max_length=MAX_SEQ_LENGTH - max_new_tokens).to(self.model.device)
            kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)
            if temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kwargs.update(do_sample=False, temperature=None, top_p=None)
            with torch.no_grad():
                out = self.model.generate(**enc, **kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            for row, i in zip(new_tokens, idx):
                ids = row.tolist()
                cut, finished = len(ids), False
                for pos, tid in enumerate(ids):
                    if tid in self.eos_ids:
                        cut, finished = pos, True
                        break
                text = self.tokenizer.decode(ids[:cut], skip_special_tokens=True)
                results[i] = GenOut(text=text, n_tokens=cut + (1 if finished else 0), finished=finished)
        return results  # type: ignore[return-value]


class VLLMGenerator:
    """Experimental: vLLM backend (not verified against Qwen3.5 in CI)."""

    def __init__(self, model_path: str, base_model: str):
        from vllm import LLM

        self.adapter = model_path if is_adapter_dir(model_path) else None
        self.llm = LLM(
            model=base_model if self.adapter else model_path,
            dtype="bfloat16",
            max_model_len=MAX_SEQ_LENGTH,
            gpu_memory_utilization=0.85,
            enable_lora=bool(self.adapter),
            max_lora_rank=64,
        )

    def generate(self, prompts: list[str], max_new_tokens: int = 1024,
                 temperature: float = 0.0, top_p: float = 0.95, seed: int = 0) -> list[GenOut]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p if temperature > 0 else 1.0,
            max_tokens=max_new_tokens,
            seed=seed,
        )
        kwargs = {}
        if self.adapter:
            from vllm.lora.request import LoRARequest

            kwargs["lora_request"] = LoRARequest("adapter", 1, self.adapter)
        outputs = self.llm.generate(prompts, params, **kwargs)
        results = []
        for item in outputs:
            comp = item.outputs[0]
            results.append(GenOut(text=comp.text, n_tokens=len(comp.token_ids),
                                  finished=comp.finish_reason == "stop"))
        return results


def load_generator(model_path: str, backend: str = "unsloth", base_model: str | None = None,
                   batch_size: int = 16):
    """Load ``model_path`` (repo id, model dir, or LoRA adapter dir) in bf16."""
    if model_path == "base":
        model_path = DEFAULT_BASE_MODEL
    base = base_model or (adapter_base_model(model_path) if is_adapter_dir(model_path) else DEFAULT_BASE_MODEL)

    if backend == "vllm":
        return VLLMGenerator(model_path, base)

    import torch

    adapter = is_adapter_dir(model_path)
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base if adapter else model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )
    elif backend == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base if adapter else model_path)
        model = AutoModelForCausalLM.from_pretrained(
            base if adapter else model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if adapter:
        from peft import PeftModel

        for adapter_path in adapter_chain(model_path):
            print(f"Applying adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            try:
                model = model.merge_and_unload()
            except Exception as exc:  # keep the unmerged adapter if merging is unsupported
                print(f"WARNING: could not merge adapter ({exc}); running with adapter attached")
    if backend == "unsloth":
        FastLanguageModel.for_inference(model)
    model.eval()
    return HFGenerator(model, tokenizer, batch_size=batch_size)
