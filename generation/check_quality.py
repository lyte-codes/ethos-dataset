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
NUMBER_WORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
)
# Spelled-out amounts count: an agent that says "about forty pounds" has committed its client
# to a price just as surely as one that says "£40".
PRICE = re.compile(
    rf"[$£€]\s?\d+|\b(?:\d+|{NUMBER_WORD}(?:[\s-]{NUMBER_WORD})*)\s?(?:dollars|pounds|euros|quid)\b",
    re.IGNORECASE,
)
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


BRIEF_PHONE = re.compile(r"Contact number:\s*([\d\s\-()+]{7,})")
BRIEF_NAME = re.compile(r"- Name:\s*(.+)")
BRIEF_AGENT = re.compile(r"You are ([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)?), a booking agent")


def check(examples: list[dict], business_words: set[str]) -> list[str]:
    turns, system, intent = rebuild(examples)
    # Standalone runs only have the file, so recover the brief from the system message the
    # generator baked into every example.
    match = BRIEF_PHONE.search(system)
    name = BRIEF_NAME.search(system)
    agent = BRIEF_AGENT.search(system)
    if agent:
        business_words = business_words | known_tokens(agent.group(1))
    return conversation_problems(turns, intent, business_words | known_tokens(system),
                                 brief_digits=match.group(1) if match else "",
                                 client_name=name.group(1).strip() if name else "")


def impersonation(text: str, client_name: str) -> str | None:
    """The agent calls *for* the client. Speaking as them is a role failure, not a phrasing one."""
    if not client_name:
        return None
    names = [client_name] + client_name.split()
    for name in names:
        pattern = re.compile(
            rf"\b(?:this is|my name is|i(?:'m| am))\s+(?:mr|mrs|ms|miss|dr)?\.?\s*{re.escape(name)}\b",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return match.group().strip()
    return None


def conversation_problems(turns: list[tuple[str, str]], intent: str, allowed_words: set[str],
                          brief_digits: str = "", client_name: str = "") -> list[str]:
    """Defects in an outbound call, where the assistant is the agent placing it.

    The assistant is the party that *supplies* details, so the rule it must not break is the
    mirror of the inbound one: it may say what the brief gave it and what the business has
    already said, and nothing else. `allowed_words` carries the brief's proper nouns and
    `brief_digits` the client's number.
    """
    problems = []
    seen = {normalize(word) for word in COMMON_WORDS | allowed_words}
    briefed_digits = re.sub(r"\D", "", brief_digits)

    business_so_far = ""
    for role, text in turns:
        if role == "user":
            seen |= known_tokens(text)
            business_so_far += " " + text
            continue

        opener = {normalize(m) for m in SENTENCE_START.findall(text)}
        for token in known_tokens(text):
            if token not in seen and token not in opener:
                problems.append(f"assistant says {token!r}, which is not in the brief")
        for match in HONORIFIC.finditer(text):
            surname = normalize(match.group(1)) if match.group(1) else None
            if surname and surname not in seen:
                problems.append(f"assistant uses honorific for a name not in the brief {surname!r}")
            elif not surname:
                problems.append("assistant uses a bare honorific")
        for run in DIGIT_RUN.findall(text):
            digits = re.sub(r"\D", "", run)
            known = re.sub(r"\D", "", business_so_far) + briefed_digits
            if digits not in known:
                problems.append(f"assistant states digits {run.strip()!r} that are not in the brief")
        if PLACEHOLDER.search(text):
            problems.append(f"placeholder text in assistant turn: {PLACEHOLDER.search(text).group()!r}")
        posing = impersonation(text, client_name)
        if posing:
            problems.append(f"assistant speaks as the client ({posing!r}) instead of on their behalf")
        seen |= known_tokens(text)

    assistant_text = " ".join(text for role, text in turns if role == "assistant")
    business_text = " ".join(text for role, text in turns if role == "user")

    # The agent commits its client to nothing it was not told to commit them to.
    if PRICE.search(assistant_text):
        problems.append(f"assistant commits to a price ({PRICE.search(assistant_text).group().strip()!r})")

    # An outbound call that never names its client is indistinguishable from the agent booking
    # for itself, which is the role confusion this dataset exists to train out.
    if client_name:
        opening = " ".join(text for role, text in turns if role == "assistant")[:400]
        if not any(part.lower() in opening.lower() for part in client_name.split()):
            problems.append("assistant never says who it is calling on behalf of")

    if briefed_digits and intent == "new_booking" and briefed_digits not in re.sub(r"\D", "", assistant_text):
        problems.append("assistant never gives the client's contact number")
    if intent != "cancellation" and not CLOCK.search(" ".join(t for _, t in turns)):
        problems.append("no clock time anywhere in the call")

    # The business is the one that reads the booking back, so the confirmation should appear
    # on its side of the call rather than the agent's.
    if intent != "cancellation" and not CLOCK.search(business_text[-600:]):
        problems.append("business never reads a time back near the end of the call")

    closing = " ".join(text for role, text in turns if role == "assistant")[-400:]
    if intent != "cancellation" and not CLOCK.search(closing):
        problems.append("assistant never confirms a time near the end of the call")

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
