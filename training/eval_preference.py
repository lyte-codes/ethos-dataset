#!/usr/bin/env python3
"""Score a model on preference pairs it was never trained on.

Held-out loss is the wrong instrument for a preference-trained model. It measures how
likely the chosen response is in absolute terms, and DPO does not optimise that — it
optimises the *margin* between chosen and rejected. A run can widen that margin while
making both responses less likely, which shows up as a worse held-out loss and a better
model. Judging DPO by likelihood mostly measures how far it moved away from likelihood.

What this measures instead:

- **accuracy** — how often the model prefers the chosen response to the rejected one.
  This is the question the preference data actually asks. 50% is a coin flip; the base
  model tends to sit a little above it because the corrupted detail is often slightly
  less fluent.
- **margin** — the mean log-probability gap between chosen and rejected, per token. How
  strongly it prefers, not just how often. A model can be right 90% of the time by a
  hair, which will not survive a real call.
- **accuracy by corruption** — phone digits, weekday, clock time, an unearned name.
  These are different failures, and a model that fixes the easy ones while still
  inventing phone numbers has not fixed the one that matters.

The pairs are rebuilt from the held-out conversations, so nothing scored here appeared in
supervised or preference training.

    python3 training/eval_preference.py --adapter v4:checkpoints/ethos-v4 \\
                                        --adapter v5:checkpoints/ethos-v5
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import BASE_MODEL, load_examples, split_by_conversation


def response_logprob(model, tokenizer, messages: list[dict], response: str,
                     device: str, max_length: int) -> float:
    """Mean log-probability per token of `response`, given the conversation before it."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + response, add_special_tokens=False,
                         truncation=True, max_length=max_length)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return float("nan")

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    # Per token, so a longer response is not penalised for being longer. The pairs differ
    # by a few tokens, but not always by the same number of them.
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    chosen = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
    response_part = chosen[len(prompt_ids) - 1:]
    return float(response_part.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", action="append", required=True,
                        help="name:path — repeat per model; an empty path is the base model")
    parser.add_argument("--data", type=Path, default=Path("data/ethos_booking_v2.jsonl"))
    parser.add_argument("--pairs", type=Path, default=Path("data/ethos_preferences_v2.jsonl"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--out", type=Path, default=Path("data/preference_eval.json"))
    args = parser.parse_args()

    # Only pairs from held-out conversations. A pair whose conversation was trained on tells
    # you what the model memorised, not what it prefers.
    _, held_out = split_by_conversation(load_examples(args.data), args.eval_fraction, args.seed)
    held_ids = {e["conversation_id"] for e in held_out}
    pairs = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    pairs = [p for p in pairs if p.get("conversation_id") in held_ids]
    if not pairs:
        print("no preference pairs from held-out conversations", file=sys.stderr)
        return 1
    print(f"{len(pairs)} held-out preference pairs from {len(held_ids)} conversations\n", flush=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    for spec in args.adapter:
        name, _, path = spec.partition(":")
        adapter = Path(path) if path else None
        if adapter is not None and not adapter.exists():
            print(f"skipping {name}: {adapter} not found", file=sys.stderr)
            continue

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
        model.to(device).eval()
        print(f"  scoring {name}…", flush=True)

        correct, margins = 0, []
        by_kind = defaultdict(lambda: [0, 0])
        for index, pair in enumerate(pairs, 1):
            if index % 25 == 0:
                print(f"    {name}: {index}/{len(pairs)}", flush=True)
            good = response_logprob(model, tokenizer, pair["messages"], pair["chosen"],
                                    device, args.max_length)
            bad = response_logprob(model, tokenizer, pair["messages"], pair["rejected"],
                                   device, args.max_length)
            if good != good or bad != bad:  # NaN
                continue
            kind = pair.get("corruption", "unknown")
            by_kind[kind][1] += 1
            if good > bad:
                correct += 1
                by_kind[kind][0] += 1
            margins.append(good - bad)

        total = len(margins)
        results[name] = {
            "pairs": total,
            "accuracy": round(correct / total, 4) if total else None,
            "mean_margin": round(sum(margins) / total, 4) if total else None,
            "by_corruption": {k: round(v[0] / v[1], 3) for k, v in sorted(by_kind.items()) if v[1]},
        }
        print(f"{name:>6}  prefers the right answer {correct}/{total} "
              f"({correct / total:.1%})  mean margin {sum(margins) / total:+.4f}", flush=True)
        for kind, rate in results[name]["by_corruption"].items():
            print(f"         {kind:16} {rate:.0%}", flush=True)

        args.out.write_text(json.dumps(results, indent=2))
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    print()
    if len(results) > 1:
        best = max(results.items(), key=lambda i: i[1]["accuracy"] or 0)
        print(f"Prefers the right answer most often: {best[0]} ({best[1]['accuracy']:.1%})")
        phone = {n: r["by_corruption"].get("phone") for n, r in results.items()
                 if r["by_corruption"].get("phone") is not None}
        if phone:
            print("Phone digits specifically — the costliest detail to get wrong: "
                  + ", ".join(f"{n} {r:.0%}" for n, r in phone.items()))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
