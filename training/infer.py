#!/usr/bin/env python3
"""Interactive check of a trained Ethos adapter: play the caller, watch the assistant."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a phone booking assistant for {business_name}. {business_context}\n"
    "Be concise, confirm details clearly, and ask one question at a time. "
    "Collect the service type, the date and time, the duration when it is relevant, "
    "and the caller's name and phone number. "
    "Read the booking back to the caller before confirming it."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("checkpoints/ethos-qwen-lora"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--business-name", default="Northgate Services")
    parser.add_argument("--business-context",
                        default="A local services business that books appointments by phone, "
                                "weekdays 8am to 6pm and Saturday mornings.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--base-only", action="store_true",
                        help="skip the adapter, to compare against the untuned base model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32 if device == "cpu" else torch.bfloat16
    )
    if not args.base_only:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to(device).eval()

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
        business_name=args.business_name, business_context=args.business_context)}]
    print(f"device: {device} | {'base model only' if args.base_only else args.adapter}")
    print("Type as the caller. Ctrl-C to quit.\n")

    while True:
        try:
            caller = input("caller> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not caller:
            continue

        messages.append({"role": "user", "content": caller})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
        print(f"assistant> {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
