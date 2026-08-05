#!/usr/bin/env python3
"""Build preference pairs that punish inventing caller details.

Each pair shares a conversation and differs only in the assistant's response: the
chosen response copies a detail the caller actually gave, the rejected one states
a plausible but wrong value. Training on these penalises the specific failure the
quality gate rejects at generation time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

PHONE = re.compile(r"\b(?:\d[\s\-().]*){7,}\d\b")
HOUR = re.compile(r"\b(\d{1,2})(:\d{2})?\s?(am|pm)\b", re.IGNORECASE)
WEEKDAY = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SUBSTITUTE_NAMES = [
    "Mr Bennett", "Mrs Clarke", "Ms Whitfield", "Mr Osei", "Dr Halloran",
    "Mr Nakamura", "Mrs Achebe", "Ms Lindqvist",
]


def corrupt_phone(text: str, rng: random.Random) -> str | None:
    """Swap the digits for different ones, keeping the original formatting."""
    match = PHONE.search(text)
    if not match:
        return None
    original = match.group()
    corrupted, changed = [], False
    for character in original:
        if character.isdigit():
            replacement = str(rng.randint(0, 9))
            if replacement != character:
                changed = True
            corrupted.append(replacement)
        else:
            corrupted.append(character)
    if not changed:
        return None
    return text[:match.start()] + "".join(corrupted) + text[match.end():]


def corrupt_time(text: str, rng: random.Random) -> str | None:
    match = HOUR.search(text)
    if not match:
        return None
    hour = int(match.group(1))
    shifted = hour + rng.choice([-2, -1, 1, 2])
    if not 1 <= shifted <= 12:
        shifted = 12 - (hour % 12) or 1
    if shifted == hour:
        return None
    replacement = f"{shifted}{match.group(2) or ''} {match.group(3)}"
    return text[:match.start()] + replacement + text[match.end():]


def corrupt_weekday(text: str, rng: random.Random) -> str | None:
    match = WEEKDAY.search(text)
    if not match:
        return None
    alternatives = [d for d in WEEKDAYS if d != match.group(1)]
    return text[:match.start()] + rng.choice(alternatives) + text[match.end():]


def inject_unearned_name(text: str, messages: list[dict], rng: random.Random) -> str | None:
    """Address the caller by a name they never gave."""
    if not text or text[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return None
    spoken = " ".join(m["content"] for m in messages if m["role"] == "user")
    candidates = [n for n in SUBSTITUTE_NAMES if n.split()[-1] not in spoken]
    if not candidates:
        return None
    name = rng.choice(candidates)
    first, rest = text.split(" ", 1) if " " in text else (text, "")
    return f"{first.rstrip(',')} {name}, {rest}" if rest else f"{first} {name}"


CORRUPTIONS = {
    "phone": lambda text, messages, rng: corrupt_phone(text, rng),
    "time": lambda text, messages, rng: corrupt_time(text, rng),
    "weekday": lambda text, messages, rng: corrupt_weekday(text, rng),
    "unearned_name": inject_unearned_name,
}


def caller_said(value: str, messages: list[dict]) -> bool:
    """Only corrupt details the caller actually supplied — those are the copies that matter."""
    digits = re.sub(r"\D", "", value)
    spoken = " ".join(m["content"] for m in messages if m["role"] == "user")
    return bool(digits) and digits in re.sub(r"\D", "", spoken)


def build_pairs(examples: list[dict], rng: random.Random, per_example: int) -> list[dict]:
    pairs = []
    for example in examples:
        response, messages = example["response"], example["messages"]
        phone_match = PHONE.search(response)
        applicable = []
        if phone_match and caller_said(phone_match.group(), messages):
            applicable.append("phone")
        if HOUR.search(response):
            applicable.append("time")
        if WEEKDAY.search(response):
            applicable.append("weekday")
        applicable.append("unearned_name")

        # Phone numbers are the costliest detail to get wrong and appear in the fewest
        # responses, so they take priority; unearned_name applies everywhere and is the
        # fallback that would otherwise swamp the mix.
        priority = {"phone": 0, "time": 1, "weekday": 2, "unearned_name": 3}
        applicable.sort(key=lambda kind: priority[kind])
        made = 0
        for kind in applicable:
            if made >= per_example:
                break
            rejected = CORRUPTIONS[kind](response, messages, rng)
            if not rejected or rejected == response:
                continue
            pairs.append({
                "messages": messages,
                "chosen": response,
                "rejected": rejected,
                "corruption": kind,
                "conversation_id": example.get("conversation_id"),
            })
            made += 1
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ethos_booking_v1.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/ethos_preferences_v1.jsonl"))
    parser.add_argument("--per-example", type=int, default=1,
                        help="maximum pairs generated from a single training example")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with args.data.open(encoding="utf-8") as handle:
        examples = [json.loads(line) for line in handle]

    pairs = build_pairs(examples, rng, args.per_example)
    with args.out.open("w", encoding="utf-8") as out_file:
        for pair in pairs:
            out_file.write(json.dumps(pair, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair["corruption"]] = counts.get(pair["corruption"], 0) + 1
    print(f"{len(examples)} examples -> {len(pairs)} preference pairs")
    for kind, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {kind:15s} {count}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
