#!/usr/bin/env python3
"""Replace real personal details with random, plausible stand-ins.

Used on the way into the correction store, so a real call's details never reach
training data. The rules that matter:

- **Plausible, not obviously fake.** A model trained on `[REDACTED]` or `XXX-XXXX`
  learns to say those. Stand-ins have to look exactly like the real thing so the
  model learns the shape of a phone number, not the shape of a redaction.
- **Consistent inside one call, unlinkable across calls.** Within a call the same
  real value always maps to the same stand-in, or the conversation stops making
  sense — the agent gives a number and the business reads a different one back.
  Across calls the mapping is re-randomised, so a stand-in cannot be used to tie
  two calls to the same person.
- **Near-misses stay near-misses.** If the business misread a number by one digit,
  the scrubbed version has to be wrong by one digit too. Mapping the two numbers
  independently would destroy the very relationship the correction is teaching.

Numbers come from 07700 900000–900999, which Ofcom reserves for drama, so no
generated number can ever reach a real person.
"""

from __future__ import annotations

import random
import re

PHONE = re.compile(r"(?:\d[\s\-().]*){6,}\d")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE)

FORENAMES = [
    "Marcus", "Priya", "Denise", "Alan", "Sophie", "Tomas", "Grace", "Hugh", "Nadia",
    "Ruth", "Callum", "Yusuf", "Eleanor", "Samir", "Bridget", "Imani", "Fergus", "Leila",
]
SURNAMES = [
    "Whitfield", "Raman", "Okoro", "Beattie", "Lindqvist", "Nowak", "Adeyemi", "Fairbairn",
    "Haddad", "Kelleher", "Doherty", "Demir", "Pryce", "Chaudhry", "Moloney", "Ferreira",
]


def digits_of(text: str) -> str:
    return re.sub(r"\D", "", text)


def near_miss(number: str, rng: random.Random) -> str:
    """Same number with one digit changed — how a misheard number actually differs."""
    positions = [i for i, character in enumerate(number) if character.isdigit()]
    # Leave the leading 077009 alone so the result stays inside the drama range.
    changeable = positions[-3:]
    index = rng.choice(changeable)
    replacement = rng.choice([d for d in "0123456789" if d != number[index]])
    return number[:index] + replacement + number[index + 1:]


class Scrubber:
    """One scrubber per call. Reuse it across that call's turns, never between calls."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.numbers: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.other: dict[str, str] = {}

    def _fresh_number(self) -> str:
        while True:
            candidate = f"07700 900{self.rng.randint(0, 999):03d}"
            if candidate not in self.numbers.values():
                return candidate

    def number_for(self, real: str) -> str:
        key = digits_of(real)
        if key in self.numbers:
            return self.numbers[key]

        # A number that differs from one already seen by a digit or two is almost
        # certainly a misreading of it, and the correction depends on that closeness.
        for known, replacement in self.numbers.items():
            if len(known) == len(key) and sum(a != b for a, b in zip(known, key)) <= 2:
                candidate = near_miss(replacement, self.rng)
                if candidate not in self.numbers.values():
                    self.numbers[key] = candidate
                    return candidate

        self.numbers[key] = self._fresh_number()
        return self.numbers[key]

    def name_for(self, real: str) -> str:
        key = real.strip().lower()
        if key not in self.names:
            while True:
                candidate = f"{self.rng.choice(FORENAMES)} {self.rng.choice(SURNAMES)}"
                if candidate not in self.names.values():
                    self.names[key] = candidate
                    break
        return self.names[key]

    def scrub(self, text: str, known_names: list[str] | None = None) -> str:
        if not text:
            return text

        # Names first: a surname is not machine-detectable in general, so the caller has
        # to say which names are real. In practice the brief already names the client.
        for real in sorted(known_names or [], key=len, reverse=True):
            if not real.strip():
                continue
            fake = self.name_for(real)
            text = re.sub(rf"\b{re.escape(real)}\b", fake, text, flags=re.IGNORECASE)
            # Parts too, since a call says "Marcus" as often as "Marcus Whitfield".
            for part, stand_in in zip(real.split(), fake.split()):
                if len(part) > 2:
                    text = re.sub(rf"\b{re.escape(part)}\b", stand_in, text, flags=re.IGNORECASE)

        text = PHONE.sub(lambda m: self.number_for(m.group()), text)
        text = EMAIL.sub(lambda m: self._placeholder(m.group(), "email"), text)
        text = POSTCODE.sub(lambda m: self._placeholder(m.group(), "postcode"), text)
        return text

    def _placeholder(self, real: str, kind: str) -> str:
        if real.lower() not in self.other:
            index = len(self.other)
            self.other[real.lower()] = (
                f"contact{index}@example.com" if kind == "email" else f"SW1A {index}AA"
            )
        return self.other[real.lower()]


def scrub_correction(entry: dict, known_names: list[str], seed: int | None = None) -> dict:
    """Scrub a whole correction with one shared mapping, so the call stays coherent."""
    scrubber = Scrubber(seed)
    scrubbed = dict(entry)
    scrubbed["messages"] = [
        {**message, "content": scrubber.scrub(message["content"], known_names)}
        for message in entry["messages"]
    ]
    scrubbed["chosen"] = scrubber.scrub(entry["chosen"], known_names)
    scrubbed["rejected"] = scrubber.scrub(entry["rejected"], known_names)
    scrubbed["scrubbed"] = True
    return scrubbed
