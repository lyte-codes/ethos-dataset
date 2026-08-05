#!/usr/bin/env python3
"""Flag quality defects in a generated booking dataset."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

HONORIFIC = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Sir|Madam)\b\.?\s+([A-Z][a-z]+)?")
PHONE = re.compile(r"(?:\d[\s\-().]*){7,}")
CLOCK = re.compile(
    r"\b(?:\d{1,2}[:.]\d{2}|\d{1,2}\s?(?:am|pm|a\.m\.|p\.m\.)|\d{1,2}\s?o'?\s?clock"
    r"|(?:half|quarter)\s+(?:past|to)\s+\w+|noon|midday)",
    re.IGNORECASE,
)
DIGIT_RUN = re.compile(r"\d[\d\s\-().]{5,}\d")
PRICE = re.compile(r"[$£€]\s?\d+|\b\d+\s?(?:dollars|pounds|euros)\b", re.IGNORECASE)
HOUR = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.|o'?\s?clock)", re.IGNORECASE)
WEEKDAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)


def spoken_hours(text: str) -> set[str]:
    hours = set()
    for hour, minute, suffix in HOUR.findall(text):
        suffix = suffix.lower().replace(".", "").replace(" ", "")
        meridiem = "am" if suffix.startswith("am") else "pm" if suffix.startswith("pm") else "?"
        hours.add(f"{int(hour) % 12}:{minute or '00'}:{meridiem}")
    return hours


BARE_HOUR = re.compile(r"\b([1-9]|1[0-2])\b")


def hours_match(readback: str, earlier: str) -> set[str]:
    earlier_hours = spoken_hours(earlier)
    loose = {h.rsplit(":", 1)[0] for h in earlier_hours}
    loose |= {f"{int(n) % 12}:00" for n in BARE_HOUR.findall(earlier)}
    return {h for h in spoken_hours(readback)
            if h not in earlier_hours and h.rsplit(":", 1)[0] not in loose}
PLACEHOLDER = re.compile(r"\[[^\]]{2,30}\]|\bXXX+\b|\b(?:insert|placeholder)\s+\w+", re.IGNORECASE)
CAPITALIZED = re.compile(r"\b[A-Z][a-z]{2,}\b")
SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|[\"']\s*)([A-Z][a-z]{2,})")

COMMON_WORDS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Services", "Certainly",
    "Perfect", "Great", "Sure", "Thanks", "Thank", "Sorry", "Hello", "Okay",
    "Alright", "Understood", "Absolutely", "Apologies", "Could", "Would", "Just",
    "Excellent", "Wonderful", "Right", "Yes", "That", "This", "Your", "You",
    "What", "When", "Which", "Would", "Have", "Let", "One", "And", "For", "The",
}


def load_conversations(path: Path) -> dict[int, list[dict]]:
    conversations = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            conversations[row["conversation_id"]].append(row)
    return conversations


def rebuild(examples: list[dict]) -> tuple[list[tuple[str, str]], str, str]:
    last = examples[-1]
    turns = [(m["role"], m["content"]) for m in last["messages"][1:]]
    turns.append(("assistant", last["response"]))
    system = last["messages"][0]["content"]
    return turns, system, last["intent"]


def normalize(token: str) -> str:
    lowered = token.lower()
    return lowered[:-1] if lowered.endswith("s") and len(lowered) > 3 else lowered


def known_tokens(text: str) -> set[str]:
    return {normalize(token) for token in CAPITALIZED.findall(text)}


def check(examples: list[dict], business_words: set[str]) -> list[str]:
    turns, _, intent = rebuild(examples)
    return conversation_problems(turns, intent, business_words)


def conversation_problems(turns: list[tuple[str, str]], intent: str, business_words: set[str]) -> list[str]:
    problems = []
    seen = {normalize(word) for word in COMMON_WORDS | business_words}
    caller_text = " ".join(t for role, t in turns if role == "user")

    caller_so_far = ""
    for role, text in turns:
        if role == "user":
            seen |= known_tokens(text)
            caller_so_far += " " + text
            continue

        opener = {normalize(m) for m in SENTENCE_START.findall(text)}
        for token in known_tokens(text):
            if token not in seen and token not in opener:
                problems.append(f"assistant says {token!r} before the caller does")
        for match in HONORIFIC.finditer(text):
            surname = normalize(match.group(1)) if match.group(1) else None
            if surname and surname not in seen:
                problems.append(f"assistant uses honorific for unstated name {surname!r}")
            elif not surname:
                problems.append("assistant uses a bare honorific")
        for run in DIGIT_RUN.findall(text):
            digits = re.sub(r"\D", "", run)
            if digits not in re.sub(r"\D", "", caller_so_far):
                problems.append(f"assistant states digits {run.strip()!r} the caller never gave")
        if PLACEHOLDER.search(text):
            problems.append(f"placeholder text in assistant turn: {PLACEHOLDER.search(text).group()!r}")
        seen |= known_tokens(text)

    first_caller_turn = next((text for role, text in turns if role == "user"), "")
    if PHONE.search(first_caller_turn):
        problems.append("caller opens the call with a phone number")
    for role, text in turns:
        if role == "assistant" and PRICE.search(text):
            problems.append(f"assistant quotes a specific price ({PRICE.search(text).group().strip()!r})")
            break

    if intent == "new_booking" and not PHONE.search(caller_text):
        problems.append("no phone number given by the caller")
    if intent != "cancellation" and not CLOCK.search(" ".join(t for _, t in turns)):
        problems.append("no clock time anywhere in the call")

    closing = " ".join(text for role, text in turns if role == "assistant")[-400:]
    if intent != "cancellation" and not CLOCK.search(closing):
        problems.append("no time read back near the end of the call")

    readback = turns[-1][1]
    earlier = " ".join(text for _, text in turns[:-1])
    for invented in sorted(hours_match(readback, earlier)):
        problems.append(f"readback books a time never discussed ({invented.replace(':', ' ')})")
    for day in {d.lower() for d in WEEKDAY.findall(readback)} - {d.lower() for d in WEEKDAY.findall(earlier)}:
        problems.append(f"readback books {day.title()}, never discussed")

    return sorted(set(problems))


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "smoke_test.jsonl")
    business_words = set(sys.argv[2].split()) if len(sys.argv) > 2 else set()
    conversations = load_conversations(path)

    clean, defect_counts = 0, collections.Counter()
    for conversation_id in sorted(conversations):
        examples = conversations[conversation_id]
        system_words = known_tokens(examples[0]["messages"][0]["content"])
        problems = check(examples, business_words | system_words)
        if problems:
            print(f"FAIL conv {conversation_id} [{examples[0]['persona']}/{examples[0]['complication']}/{examples[0]['intent']}]")
            for problem in problems:
                print(f"     - {problem}")
                defect_counts[problem.split(":")[0].split("'")[0].strip()] += 1
        else:
            clean += 1
            print(f"PASS conv {conversation_id} [{examples[0]['persona']}/{examples[0]['complication']}/{examples[0]['intent']}]")

    total = len(conversations)
    print(f"\n{clean}/{total} clean ({clean / total * 100:.0f}%)" if total else "no conversations")
    for defect, count in defect_counts.most_common():
        print(f"  {count}x {defect}")
    return 0 if clean == total else 1


if __name__ == "__main__":
    sys.exit(main())
