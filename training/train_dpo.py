#!/usr/bin/env python3
"""Preference-train the Ethos adapter to penalise invented caller details.

DPO starts from a supervised model, not from the base model: it adjusts a model
that already does the task, pushing probability away from the rejected response
and toward the chosen one. Run train_lora.py first and point --sft-adapter here.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Hyperparameters:
    beta: float = 0.1
    learning_rate: float = 5e-6
    epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    max_prompt_length: int = 896
    warmup_ratio: float = 0.1


def load_pairs(path: Path, tokenizer) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)
            rows.append({
                "prompt": tokenizer.apply_chat_template(
                    pair["messages"], tokenize=False, add_generation_prompt=True
                ),
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ethos_preferences_v1.jsonl"))
    parser.add_argument("--sft-adapter", type=Path, default=Path("checkpoints/ethos-qwen-lora"),
                        help="the supervised adapter to start from (model v2)")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ethos-qwen-dpo"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--beta", type=float, default=None,
                        help="how hard to push away from rejected responses; higher stays closer to the SFT model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="per-device batch size; keep at 1 on MPS — DPO holds policy and "
                             "reference logits at once, so it needs more room than SFT did")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hyperparameters = Hyperparameters()
    if args.beta:
        hyperparameters.beta = args.beta
    if args.epochs:
        hyperparameters.epochs = args.epochs
    if args.batch_size:
        hyperparameters.per_device_batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    if not args.sft_adapter.exists():
        parser.error(f"{args.sft_adapter} not found — train the supervised model first "
                     f"(python3 training/train_lora.py)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Same reasoning as train_lora.py: bf16 only pays off on CUDA, and MPS is more reliable
    # in fp32. `dtype` rather than the deprecated `torch_dtype`.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    model = PeftModel.from_pretrained(model, str(args.sft_adapter), is_trainable=True)

    dataset = load_pairs(args.data, tokenizer)
    print(f"{len(dataset)} preference pairs")

    config = DPOConfig(
        output_dir=str(args.output),
        beta=hyperparameters.beta,
        learning_rate=hyperparameters.learning_rate,
        num_train_epochs=hyperparameters.epochs,
        per_device_train_batch_size=hyperparameters.per_device_batch_size,
        gradient_accumulation_steps=hyperparameters.gradient_accumulation_steps,
        max_length=hyperparameters.max_length,
        max_prompt_length=hyperparameters.max_prompt_length,
        warmup_ratio=hyperparameters.warmup_ratio,
        logging_steps=10,
        gradient_checkpointing=True,
        save_strategy="epoch",
        bf16=(device == "cuda"),
        seed=args.seed,
        report_to=[],
    )

    # No explicit reference model: with a PEFT model, DPOTrainer uses the adapter
    # disabled as the reference, which halves memory.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    (args.output / "hyperparameters.json").write_text(
        json.dumps(vars(hyperparameters) | {
            "base_model": args.base_model,
            "sft_adapter": str(args.sft_adapter),
            "preference_pairs": len(dataset),
        }, indent=2)
    )
    print(f"adapter saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
