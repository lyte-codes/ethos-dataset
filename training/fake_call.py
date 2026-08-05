#!/usr/bin/env python3
"""Run a fake phone call end to end: the model under test answers, a local
Ollama model plays the caller.

The caller is given a fact sheet — a name, a phone number, a service and a
preferred day and time — and told to reveal each detail only when asked. That
makes the transcript diagnostic as well as illustrative: every concrete detail
the assistant reads back can be checked against what the caller actually said,
which is exactly the failure preference training is meant to fix.

    python3 training/fake_call.py --base-only -n 5        # model v1, the untuned baseline
    python3 training/fake_call.py --adapter checkpoints/ethos-v2 -n 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a phone booking assistant for {business_name}. {business_context}\n"
    "Be concise, confirm details clearly, and ask one question at a time. "
    "Collect the service type, the date and time, the duration when it is relevant, "
    "and the caller's name and phone number. "
    "Read the booking back to the caller before confirming it."
)

CALLER_SYSTEM_PROMPT = """You are playing a member of the public phoning {business_name} to book an appointment. You are NOT the assistant — you are the customer.

Your details, which you must stick to exactly:
- Your name: {name}
- Your phone number: {phone}
- The service you want: {service}
- When you want it: {when}

How you speak: {persona_description}

Rules:
- Say only what you would say out loud on the phone. No narration, no stage directions, no labels.
- One short turn at a time — a sentence or two, like real speech.
- Give a detail ONLY when the assistant asks for it. Never volunteer your phone number or name unprompted.
- Never invent details beyond the four above. If asked something not covered, improvise something small and plausible and stay consistent.
- When the assistant has read your booking back to you and you are satisfied, say a brief goodbye and then write [HANG UP] on its own at the end.
- If the assistant is going in circles after several tries, say goodbye and write [HANG UP].
"""

PERSONAS = {
    "terse": "Answers in as few words as possible, often one or two words. Never volunteers anything.",
    "polite": "Warm and cooperative, thanks the assistant often, answers clearly.",
    "impatient": "In a hurry, pushes back on being asked for details, wants it booked and done.",
    "confused": "Unsure what you need, mixes up days of the week, asks for things to be repeated.",
    "rambling": "Talks around the point and buries the answer inside a long sentence.",
}

CALLERS = [
    {"name": "Marcus Whitfield", "phone": "07700 900412", "service": "boiler repair",
     "when": "Tuesday morning", "persona": "polite"},
    {"name": "Priya Raman", "phone": "07700 900318", "service": "kitchen fitting",
     "when": "Thursday afternoon", "persona": "terse"},
    {"name": "Denise Okoro", "phone": "07700 900577", "service": "roof inspection",
     "when": "Saturday morning", "persona": "impatient"},
    {"name": "Alan Beattie", "phone": "07700 900243", "service": "boiler installation",
     "when": "next Wednesday around 2pm", "persona": "confused"},
    {"name": "Sophie Lindqvist", "phone": "07700 900861", "service": "annual assessment",
     "when": "Friday, late morning", "persona": "rambling"},
]

OPENERS = {
    "terse": "Hello. Need to book something.",
    "polite": "Oh hello there — I was hoping to book an appointment, if that's alright?",
    "impatient": "Hi, yeah, I need to get something booked in, quickly if possible.",
    "confused": "Hello? Is this the right number for booking? I'm not entirely sure who I need.",
    "rambling": "Hi there, sorry — I've been meaning to ring all week and it kept slipping my mind, but I need to get something sorted.",
}


@dataclass
class CallerProfile:
    name: str
    phone: str
    service: str
    when: str
    persona: str


class OllamaError(RuntimeError):
    pass


def ollama_chat(messages: list[dict], model: str, base_url: str,
                temperature: float, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": messages,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise OllamaError(f"ollama HTTP {exc.code}: {exc.read().decode()[:200]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OllamaError(
            f"could not reach ollama at {base_url} ({exc}). Start it with: ollama serve"
        ) from exc
    return body.get("message", {}).get("content", "").strip()


def clean_caller_turn(raw: str) -> tuple[str, bool]:
    """Strip anything the caller model narrated, and detect the hang-up marker."""
    hung_up = "[HANG UP]" in raw.upper()
    text = re.sub(r"\[hang up\]", "", raw, flags=re.IGNORECASE)
    # Models sometimes prefix a speaker label despite being told not to.
    text = re.sub(r"^\s*(caller|customer|me|you)\s*[:>-]\s*", "", text, flags=re.IGNORECASE)
    # Drop stage directions on their own lines.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not (line.startswith("*") and line.endswith("*"))]
    return " ".join(lines).strip(), hung_up


def load_assistant(base_model: str, adapter: Path | None, device_preference: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_preference == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = device_preference

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.to(device).eval()
    return tokenizer, model, device


def assistant_reply(tokenizer, model, device, messages: list[dict],
                    max_new_tokens: int, temperature: float) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_call(profile: CallerProfile, tokenizer, model, device, args) -> dict:
    system = SYSTEM_PROMPT.format(
        business_name=args.business_name, business_context=args.business_context
    )
    assistant_messages = [{"role": "system", "content": system}]
    caller_messages = [{
        "role": "system",
        "content": CALLER_SYSTEM_PROMPT.format(
            business_name=args.business_name,
            name=profile.name,
            phone=profile.phone,
            service=profile.service,
            when=profile.when,
            persona_description=PERSONAS[profile.persona],
        ),
    }]

    turns: list[dict] = []
    caller_text = OPENERS[profile.persona]

    for _ in range(args.max_turns):
        turns.append({"role": "caller", "text": caller_text})
        assistant_messages.append({"role": "user", "content": caller_text})
        caller_messages.append({"role": "assistant", "content": caller_text})

        reply = assistant_reply(
            tokenizer, model, device, assistant_messages,
            args.max_new_tokens, args.temperature,
        )
        if not reply:
            reply = "(no response)"
        turns.append({"role": "assistant", "text": reply})
        assistant_messages.append({"role": "assistant", "content": reply})
        caller_messages.append({"role": "user", "content": reply})

        raw = ollama_chat(caller_messages, args.caller_model, args.base_url, args.caller_temperature)
        caller_text, hung_up = clean_caller_turn(raw)
        if hung_up:
            if caller_text:
                turns.append({"role": "caller", "text": caller_text})
            break
        if not caller_text:
            break

    return {"caller": asdict(profile), "turns": turns, "turn_count": len(turns)}


def audit(call: dict) -> dict:
    """Check the assistant's spoken details against what the caller actually gave."""
    profile = call["caller"]
    said_by_assistant = " ".join(t["text"] for t in call["turns"] if t["role"] == "assistant")
    said_by_caller = " ".join(t["text"] for t in call["turns"] if t["role"] == "caller")

    digits = lambda text: re.sub(r"\D", "", text)
    true_phone = digits(profile["phone"])
    quoted = re.findall(r"(?:\d[\s\-().]*){6,}\d", said_by_assistant)
    quoted_digits = [digits(q) for q in quoted]

    surname = profile["name"].split()[-1].lower()
    forename = profile["name"].split()[0].lower()

    return {
        "read_back_phone": any(d == true_phone for d in quoted_digits),
        "wrong_phone_digits": [q for q, d in zip(quoted, quoted_digits)
                               if d and d != true_phone and len(d) >= 6],
        "used_caller_name": forename in said_by_assistant.lower() or surname in said_by_assistant.lower(),
        "name_before_caller_gave_it": (
            (forename in said_by_assistant.lower() or surname in said_by_assistant.lower())
            and forename not in said_by_caller.lower() and surname not in said_by_caller.lower()
        ),
        "quoted_a_price": bool(re.search(r"[£$€]\s?\d", said_by_assistant)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, default=None,
                        help="LoRA adapter to load; omit (or use --base-only) for the untuned baseline")
    parser.add_argument("--base-only", action="store_true", help="model v1 — no adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--out", type=Path, default=Path("data/fake_calls.json"))
    parser.add_argument("-n", "--calls", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=16,
                        help="hard stop, counting both sides")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--caller-model", default="llama3.2:3b")
    parser.add_argument("--caller-temperature", type=float, default=0.8)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--business-name", default="Northgate Services")
    parser.add_argument("--business-context",
                        default="A local services business that books appointments by phone, "
                                "weekdays 8am to 6pm and Saturday mornings.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adapter = None if args.base_only else args.adapter
    if adapter is not None and not adapter.exists():
        parser.error(f"{adapter} not found")

    # Fail before loading 1.5B of weights if the caller model is unreachable.
    try:
        ollama_chat([{"role": "user", "content": "ping"}], args.caller_model, args.base_url, 0.0, timeout=30)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    random.seed(args.seed)
    profiles = [CallerProfile(**{k: v for k, v in caller.items()})
                for caller in CALLERS[:args.calls]]
    while len(profiles) < args.calls:
        profiles.append(CallerProfile(**CALLERS[len(profiles) % len(CALLERS)]))

    label = "v1 (base, no adapter)" if adapter is None else str(adapter)
    print(f"assistant: {label}")
    tokenizer, model, device = load_assistant(args.base_model, adapter, args.device)
    print(f"device: {device} | caller: {args.caller_model}\n")

    calls = []
    for index, profile in enumerate(profiles, 1):
        print(f"[{index}/{len(profiles)}] {profile.name} — {profile.service} ({profile.persona})",
              flush=True)
        call = run_call(profile, tokenizer, model, device, args)
        call["audit"] = audit(call)
        call["assistant_model"] = label
        calls.append(call)
        flags = [k for k, v in call["audit"].items() if v is True and k != "read_back_phone"]
        print(f"      {call['turn_count']} turns"
              f" | phone read back correctly: {call['audit']['read_back_phone']}"
              f"{' | flags: ' + ', '.join(flags) if flags else ''}\n", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"assistant_model": label, "calls": calls}, indent=2))
    print(f"{len(calls)} calls written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
