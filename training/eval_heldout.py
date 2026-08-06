#!/usr/bin/env python3
"""Measure an adapter against conversations it was never trained on.

Training loss cannot tell learning from memorising: both make it fall. The gap between
training loss and held-out loss can. A model that has learned the task scores about the
same on unseen conversations as on seen ones; a model that has memorised the training set
scores far worse on unseen ones, and the gap widens with every extra epoch.

The split is reproduced exactly as train_lora.py made it — same seed, same fraction,
whole conversations held out rather than individual turns — so the examples scored here
are the ones the adapter genuinely never saw.

    python3 training/eval_heldout.py \\
        --adapter v4:checkpoints/ethos-v4 \\
        --adapter v6:checkpoints/ethos-v6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import (BASE_MODEL, IGNORE_INDEX, Hyperparameters, build_tokenize_fn,
                        load_examples, split_by_conversation)


def mean_loss(model, tokenizer, examples: list[dict], device: str, max_length: int,
              label: str = "") -> float:
    """Average per-token loss over the assistant turns, which is what training optimised."""
    tokenize = build_tokenize_fn(tokenizer, max_length)
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for index, example in enumerate(examples, 1):
            if label and index % 25 == 0:
                print(f"    {label}: {index}/{len(examples)}", flush=True)
            encoded = tokenize(example)
            input_ids = torch.tensor([encoded["input_ids"]], device=device)
            labels = torch.tensor([encoded["labels"]], device=device)
            supervised = int((labels != IGNORE_INDEX).sum())
            if not supervised:
                continue
            loss = model(input_ids=input_ids, labels=labels).loss
            # Weight by supervised tokens, so a long response counts for more than a short
            # one — an unweighted mean would let two-word turns dominate the number.
            total_loss += float(loss) * supervised
            total_tokens += supervised

    return total_loss / total_tokens if total_tokens else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", action="append", required=True,
                        help="name:path — repeat per adapter; an empty path is the base model")
    parser.add_argument("--data", type=Path, default=Path("data/ethos_booking_v2.jsonl"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sample", type=int, default=120,
                        help="how many training examples to score for the comparison")
    parser.add_argument("--out", type=Path, default=Path("data/heldout_eval.json"))
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples(args.data)
    train_examples, held_out = split_by_conversation(examples, args.eval_fraction, args.seed)
    # flush on every print: redirected output block-buffers otherwise, and a long scoring
    # run then shows nothing at all until it exits, which is indistinguishable from a hang.
    print(f"{len(held_out)} held-out examples, {len(train_examples)} training "
          f"(scoring {min(args.train_sample, len(train_examples))} of them)\n", flush=True)

    max_length = Hyperparameters().max_sequence_length
    seen = train_examples[:args.train_sample]
    results = {}

    for spec in args.adapter:
        name, _, path = spec.partition(":")
        adapter = Path(path) if path else None
        if adapter is not None and not adapter.exists():
            print(f"skipping {name}: {adapter} not found", file=sys.stderr)
            continue

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
        model.to(device).eval()

        print(f"  scoring {name}…", flush=True)
        train_loss = mean_loss(model, tokenizer, seen, device, max_length, f"{name} train")
        held_loss = mean_loss(model, tokenizer, held_out, device, max_length, f"{name} held-out")
        results[name] = {
            "train_loss": round(train_loss, 4),
            "heldout_loss": round(held_loss, 4),
            "gap": round(held_loss - train_loss, 4),
        }
        print(f"{name:>6}  train {train_loss:.3f}  held-out {held_loss:.3f}  "
              f"gap {held_loss - train_loss:+.3f}", flush=True)
        del model

    print()
    if results:
        best = min(results.items(), key=lambda item: item[1]["heldout_loss"])
        print(f"Lowest held-out loss: {best[0]} ({best[1]['heldout_loss']:.3f}) — this is the "
              f"one that generalises best, whatever the training losses say.")
        widest = max(results.items(), key=lambda item: item[1]["gap"])
        if widest[1]["gap"] > 0.15:
            print(f"Widest gap: {widest[0]} (+{widest[1]['gap']:.3f}). A gap that grows with "
                  f"epochs while held-out loss stops falling is memorisation.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
