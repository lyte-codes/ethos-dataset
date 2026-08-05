#!/usr/bin/env python3
"""Record a mistake a deployed agent made, so the next model can be trained out of it.

Every correction is one preference pair waiting to happen: what the agent said is the
rejected response, what it should have said is the chosen one. That is a far better
training signal than the synthetic corruptions in build_preference_pairs.py, because it
is a mistake the model actually made rather than one invented for it.

Everything written here lands under private/, which is gitignored in full. Real calls
carry real names and real numbers, and a model trained on them can repeat them back —
so this data stays on the machine that recorded it and never reaches the public repo.

    python3 training/record_correction.py \\
        --call data/fake_calls.json --index 0 \\
        --said "That's 07700 900312." \\
        --should-have-said "That's 07700 900734." \\
        --reason "gave a number that was not in the brief"

Or, from the audit of a call that already flagged itself:

    python3 training/record_correction.py --from-audit data/fake_calls.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STORE = Path("private/corrections.jsonl")

# Scrubbing happens on the way in, not on the way out. A store holding real details is one
# export away from leaking them, and nothing downstream needs them — the model has to learn
# "say the number from the brief", not any particular number.
BRIEF_NAME = re.compile(r"- Name:\s*(.+)")


def record(store: Path, entry: dict, scrub_details: bool = True,
           known_names: list[str] | None = None) -> None:
    if scrub_details:
        from scrub import scrub_correction

        names = list(known_names or [])
        match = BRIEF_NAME.search(entry["messages"][0]["content"])
        if match:
            names.append(match.group(1).strip())
        entry = scrub_correction(entry, known_names=names)

    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def from_call(call: dict, index: int, agent_name: str) -> tuple[list[dict], str]:
    """Rebuild the conversation up to the agent's last turn, plus what it said there."""
    system = call.get("system") or ""
    if not system:
        brief = call.get("brief", {})
        system = (f"You are {agent_name}, a booking agent.\n"
                  f"- Name: {brief.get('client_name', '')}\n"
                  f"- Contact number: {brief.get('client_phone', '')}\n"
                  f"- Needs: {brief.get('service', '')}\n"
                  f"- Available: {brief.get('availability', '')}")

    turns = call["turns"]
    last_agent = max(i for i, turn in enumerate(turns) if turn["role"] == "assistant")
    messages = [{"role": "system", "content": system}]
    for turn in turns[:last_agent]:
        messages.append({
            "role": "user" if turn["role"] == "business" else "assistant",
            "content": turn["text"],
        })
    return messages, turns[last_agent]["text"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--call", type=Path, help="a fake_calls.json-shaped file")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--said", help="what the agent actually said (the rejected response)")
    parser.add_argument("--should-have-said", help="what it should have said (the chosen response)")
    parser.add_argument("--reason", default="", help="why the turn was wrong, in your words")
    parser.add_argument("--build", default="unknown", help="which build made the mistake")
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--agent-name", default="Ethos")
    parser.add_argument("--name", action="append", default=[],
                        help="a real name to scrub beyond the client's; repeatable")
    parser.add_argument("--keep-real-details", action="store_true",
                        help="skip scrubbing — only for data that is already synthetic")
    args = parser.parse_args()

    if not args.call:
        parser.error("--call is required")
    payload = json.loads(args.call.read_text())
    calls = payload["calls"] if isinstance(payload, dict) else payload
    call = calls[args.index]

    messages, actually_said = from_call(call, args.index, args.agent_name)
    rejected = args.said or actually_said
    if not args.should_have_said:
        parser.error("--should-have-said is required: a correction needs the right answer, "
                     "not just the wrong one")

    record(args.store, {
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build": args.build,
        "reason": args.reason,
        "source": "human",
        "messages": messages,
        "chosen": args.should_have_said,
        "rejected": rejected,
    }, scrub_details=not args.keep_real_details, known_names=args.name)

    total = sum(1 for _ in args.store.open(encoding="utf-8"))
    print(f"recorded to {args.store} ({total} correction{'s' if total != 1 else ''} held)")
    if args.keep_real_details:
        print("WARNING: stored unscrubbed — only do this for already-synthetic data")
    else:
        print("names, numbers, emails and postcodes replaced with random plausible stand-ins")
    print("this file is gitignored and stays on this machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
