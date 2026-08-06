#!/usr/bin/env python3
"""LoRA fine-tune of Qwen2.5-1.5B-Instruct on the Ethos booking dataset.

Trains on assistant turns only: the prompt tokens are masked out of the loss so
the model learns to produce the next assistant turn, not to reproduce the caller.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

# transformers pulls in its TensorFlow integration on import, which fails against
# Keras 3. Nothing here uses TF.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
IGNORE_INDEX = -100


@dataclass
class Hyperparameters:
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_sequence_length: int = 1024
    lr_scheduler: str = "cosine"

    # q/k/v/o plus the MLP projections. Attention-only (drop the last three)
    # trains faster and fits smaller GPUs at some cost in instruction fidelity.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_tokenize_fn(tokenizer, max_length: int):
    """Render each example with the chat template and mask everything but the response."""

    def tokenize(example: dict) -> dict:
        prompt = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        full = prompt + example["response"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False, truncation=True,
                             max_length=max_length)["input_ids"]

        labels = list(full_ids)
        for position in range(min(len(prompt_ids), len(labels))):
            labels[position] = IGNORE_INDEX

        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    return tokenize


def split_by_conversation(examples: list[dict], eval_fraction: float, seed: int) -> tuple[list, list]:
    """Hold out whole conversations, never individual turns from a seen conversation."""
    import random

    conversation_ids = sorted({e["conversation_id"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(conversation_ids)
    cutoff = max(1, int(len(conversation_ids) * eval_fraction))
    held_out = set(conversation_ids[:cutoff])

    train = [e for e in examples if e["conversation_id"] not in held_out]
    evaluation = [e for e in examples if e["conversation_id"] in held_out]
    return train, evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ethos_booking.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ethos-qwen-lora"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None,
                        help="continue a run from a checkpoint directory. The checkpoint holds "
                             "optimizer and scheduler state as well as weights, so the run picks "
                             "up mid-schedule rather than restarting the warmup and decay.")
    args = parser.parse_args()

    hyperparameters = Hyperparameters()
    if args.epochs:
        hyperparameters.epochs = args.epochs
    if args.learning_rate:
        hyperparameters.learning_rate = args.learning_rate
    if args.batch_size:
        hyperparameters.per_device_batch_size = args.batch_size

    device = select_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples(args.data)
    train_examples, eval_examples = split_by_conversation(
        examples, args.eval_fraction, args.seed
    )
    print(f"{len(examples)} examples -> {len(train_examples)} train / {len(eval_examples)} eval "
          f"(split by conversation, no turn leakage)")

    tokenize = build_tokenize_fn(tokenizer, hyperparameters.max_sequence_length)
    train_dataset = Dataset.from_list(train_examples).map(
        tokenize, remove_columns=Dataset.from_list(train_examples).column_names
    )
    eval_dataset = Dataset.from_list(eval_examples).map(
        tokenize, remove_columns=Dataset.from_list(eval_examples).column_names
    )

    # bf16 only pays off on CUDA. MPS trains more reliably in fp32, and a 1.5B model
    # in fp32 still fits comfortably in unified memory.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=hyperparameters.lora_rank,
        lora_alpha=hyperparameters.lora_alpha,
        lora_dropout=hyperparameters.lora_dropout,
        target_modules=list(hyperparameters.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Recompute activations during backward instead of storing all of them. On a
    # 16GB unified-memory machine, storing full activations for a 1.5B model at
    # seq_len=1024 swaps hard enough to make training impractically slow.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    training_arguments = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=hyperparameters.epochs,
        per_device_train_batch_size=hyperparameters.per_device_batch_size,
        gradient_accumulation_steps=hyperparameters.gradient_accumulation_steps,
        learning_rate=hyperparameters.learning_rate,
        lr_scheduler_type=hyperparameters.lr_scheduler,
        warmup_ratio=hyperparameters.warmup_ratio,
        weight_decay=hyperparameters.weight_decay,
        logging_steps=10,
        # Eval forwards full-vocab logits (152k) on top of memory the training loop
        # hasn't released yet — on this 16GB MPS machine that tips it over the cap
        # right at the epoch boundary. Skip in-loop eval here; run it separately
        # against the saved checkpoint instead.
        eval_strategy="epoch" if (eval_examples and device != "mps") else "no",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=(device == "cuda"),
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_examples else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX),
    )

    resume = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    if resume:
        if not args.resume_from_checkpoint.exists():
            parser.error(f"{args.resume_from_checkpoint} not found")
        # Everything else — epochs, batch size, learning rate — must match the original run,
        # or the scheduler resumes onto a curve that was never planned.
        print(f"resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    metrics_path = args.output / "hyperparameters.json"
    metrics_path.write_text(json.dumps(vars(hyperparameters) | {
        "base_model": args.base_model,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
    }, indent=2, default=list))
    print(f"adapter saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
