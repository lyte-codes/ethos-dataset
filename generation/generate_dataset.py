#!/usr/bin/env python3
"""Generate synthetic phone-booking conversations with a local Ollama model."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from check_quality import conversation_problems, known_tokens

SERVICE_TYPES = [
    "consultation",
    "repair",
    "fitting",
    "inspection",
    "installation",
    "follow-up appointment",
    "assessment",
]

PERSONAS = {
    "terse": "Answers in as few words as possible. Often one or two words. Never volunteers information that was not asked for.",
    "rambling": "Talks around the point, drifts into unrelated detail about their week, and buries the actual answer inside a long sentence.",
    "polite": "Warm and cooperative, thanks the assistant often, apologises for taking up time, answers clearly.",
    "impatient": "In a hurry, wants this over with, pushes back on being asked for details, interrupts with 'can we just get it booked'.",
    "confused": "Unsure what they actually need, mixes up dates and days of the week, asks the assistant to repeat things.",
}

COMPLICATIONS = {
    "vague_time": "The caller first gives a vague time reference such as 'sometime next week' or 'end of the month' and the assistant has to narrow it down to a specific day and time.",
    "change_of_mind": "Partway through the call the caller changes their mind about the day, the time, or the service, and the assistant has to update what it already collected.",
    "asks_before_committing": "Before agreeing to anything the caller asks about availability, how long it takes, or roughly what it costs. The assistant answers plausibly but does not invent exact prices, and steers back to booking.",
    "corrects_info": "The caller gives a detail (a phone number, a name spelling, a date) and then corrects it a few turns later. The assistant has to acknowledge the correction and use the new value.",
    "multiple_bookings": "The caller books two separate things in one call, and the assistant keeps them distinct and confirms both at the end.",
    "mishearing": "There is background noise on the line. The assistant mishears or cannot catch a detail once, asks the caller to repeat or spell it, and then continues.",
}

CALL_INTENTS = {
    "new_booking": "The caller wants to book a new appointment.",
    "reschedule": "The caller already has an appointment and wants to move it to a different day or time. They give their name and roughly when the existing appointment is.",
    "cancellation": "The caller wants to cancel an existing appointment. The assistant confirms which appointment, cancels it, and offers to rebook.",
}

INTENT_WEIGHTS = {"new_booking": 0.65, "reschedule": 0.2, "cancellation": 0.15}

ASSISTANT_SYSTEM_PROMPT = (
    "You are a phone booking assistant for {business_name}. {business_context}\n"
    "Be concise, confirm details clearly, and ask one question at a time. "
    "Collect the service type, the date and time, the duration when it is relevant, "
    "and the caller's name and phone number. "
    "Read the booking back to the caller before confirming it."
)

GENERATOR_SYSTEM_PROMPT = """You write realistic transcripts of phone calls to a small business's booking line.

You will be given a scenario: a caller persona, a complication, a service, and what the caller is calling about. Write the whole call as it would actually sound on the phone.

Rules:
- The caller speaks first. The assistant speaks last, and the last assistant turn confirms the outcome of the call.
- Speakers strictly alternate: caller, assistant, caller, assistant, and so on.
- Between 8 and 20 turns in total.
- Spoken language only. No stage directions, no "[background noise]", no speaker labels inside the text, no markdown.
- The assistant is concise, asks one question at a time, and confirms details back to the caller.
- The assistant knows nothing about the caller at the start of the call. It must never use a detail the caller has not already said out loud. In particular it must not use the caller's name, or any honorific such as Mr or Ms, until the caller has given their name. Before that point the assistant addresses the caller with no name at all.
- The assistant must ask for a callback phone number, and the caller must speak that number out loud in a caller turn before the assistant ever repeats it back. The assistant must never state a phone number, a name, a date or a time that the caller has not already said. The assistant must also collect the caller's name, the service, and a specific day and clock time.
- The final assistant turn reads the whole booking back: service, weekday and date, clock time, caller name, phone number.
- Use real-sounding specifics. Never write a placeholder such as [name], [date] or XXX-XXXX.
- The persona shapes the caller's turns, not the assistant's. The assistant stays professional throughout.

Reply with JSON only, in exactly this shape:
{"turns": [{"speaker": "caller", "text": "..."}, {"speaker": "assistant", "text": "..."}]}"""

GENERATOR_USER_PROMPT = """Business: {business_name}
Context: {business_context}
Service: {service}
Call type: {intent_description}
Caller persona ({persona}): {persona_description}
Complication: {complication_description}

Write the full call as JSON."""

BAR_WIDTH = 24
MIN_TURNS = 8
MIN_TURNS_CANCELLATION = 4
MIN_TURNS_TERSE = 6
MAX_TURNS = 40
MAX_TURN_CHARS = 800

SPEAKER_ALIASES = {
    "caller": "caller",
    "customer": "caller",
    "user": "caller",
    "client": "caller",
    "assistant": "assistant",
    "agent": "assistant",
    "receptionist": "assistant",
    "booking assistant": "assistant",
}

LABEL_PREFIX = re.compile(r"^\s*(caller|customer|user|assistant|agent|receptionist)\s*:\s*", re.IGNORECASE)
SALVAGE_TURN = re.compile(
    r'"(?:speaker|role)"\s*:\s*"([^"]+)"\s*,\s*"(?:text|content|message)"\s*:\s*"((?:[^"\\]|\\.)*)"'
)


class GenerationError(Exception):
    """Raised when a generation cannot be turned into a usable conversation."""


@dataclass(frozen=True)
class Scenario:
    service: str
    persona: str
    complication: str
    intent: str


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


def sample_scenario(rng: random.Random) -> Scenario:
    intents = list(INTENT_WEIGHTS)
    return Scenario(
        service=rng.choice(SERVICE_TYPES),
        persona=rng.choice(list(PERSONAS)),
        complication=rng.choice(list(COMPLICATIONS)),
        intent=rng.choices(intents, weights=[INTENT_WEIGHTS[i] for i in intents])[0],
    )


def call_ollama(base_url: str, model: str, temperature: float, timeout: int, scenario: Scenario,
                business_name: str, business_context: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "top_p": 0.95, "num_predict": 2048},
        "messages": [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GENERATOR_USER_PROMPT.format(
                    business_name=business_name,
                    business_context=business_context,
                    service=scenario.service,
                    intent_description=CALL_INTENTS[scenario.intent],
                    persona=scenario.persona,
                    persona_description=PERSONAS[scenario.persona],
                    complication_description=COMPLICATIONS[scenario.complication],
                ),
            },
        ],
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
        raise GenerationError(f"ollama HTTP {exc.code}: {exc.read().decode()[:200]}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GenerationError(f"ollama request failed: {exc}") from exc
    return body.get("message", {}).get("content", "")


def extract_json_object(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise GenerationError("no JSON object in output")
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise GenerationError(f"unparseable JSON: {exc}") from exc


def extract_turn_dicts(raw: str) -> list[dict]:
    try:
        parsed = extract_json_object(raw)
    except GenerationError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("turns", "conversation", "messages"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
    salvaged = [
        {"speaker": speaker, "text": json.loads(f'"{text}"')}
        for speaker, text in SALVAGE_TURN.findall(raw)
    ]
    if salvaged:
        return salvaged
    raise GenerationError("no turn list in output")


def repair_turns(turns: list[Turn]) -> list[Turn]:
    merged: list[Turn] = []
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker:
            merged[-1] = Turn(turn.speaker, f"{merged[-1].text} {turn.text}")
        else:
            merged.append(turn)
    while merged and merged[0].speaker != "caller":
        merged.pop(0)
    while merged and merged[-1].speaker != "assistant":
        merged.pop()
    return merged


def minimum_turns(scenario: Scenario) -> int:
    if scenario.intent == "cancellation":
        return MIN_TURNS_CANCELLATION
    if scenario.persona == "terse":
        return MIN_TURNS_TERSE
    return MIN_TURNS


def parse_conversation(raw: str, min_turns: int = MIN_TURNS) -> list[Turn]:
    turns: list[Turn] = []
    for item in extract_turn_dicts(raw):
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker") or item.get("role") or "").strip().lower()
        text = item.get("text") or item.get("content") or item.get("message") or ""
        if speaker not in SPEAKER_ALIASES and len(item) == 1:
            only_key, only_value = next(iter(item.items()))
            if str(only_key).strip().lower() in SPEAKER_ALIASES and isinstance(only_value, str):
                speaker, text = str(only_key).strip().lower(), only_value
        if speaker not in SPEAKER_ALIASES or not isinstance(text, str):
            continue
        text = LABEL_PREFIX.sub("", text).strip()
        if not text:
            continue
        if len(text) > MAX_TURN_CHARS:
            raise GenerationError("turn text implausibly long")
        turns.append(Turn(SPEAKER_ALIASES[speaker], text))

    turns = repair_turns(turns)

    if not min_turns <= len(turns) <= MAX_TURNS:
        raise GenerationError(f"turn count out of range ({len(turns)})")
    if turns[0].speaker != "caller":
        raise GenerationError("conversation does not start with the caller")
    if turns[-1].speaker != "assistant":
        raise GenerationError("conversation does not end with the assistant")
    for earlier, later in zip(turns, turns[1:]):
        if earlier.speaker == later.speaker:
            raise GenerationError("speakers do not alternate")
    return turns


def build_examples(turns: list[Turn], system_prompt: str, scenario: Scenario, conversation_id: int) -> list[dict]:
    examples = []
    history: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in turns:
        if turn.speaker == "caller":
            history.append({"role": "user", "content": turn.text})
            continue
        examples.append({
            "messages": list(history),
            "response": turn.text,
            "conversation_id": conversation_id,
            "service": scenario.service,
            "persona": scenario.persona,
            "complication": scenario.complication,
            "intent": scenario.intent,
        })
        history.append({"role": "assistant", "content": turn.text})
    return examples


def generate_conversation(args: argparse.Namespace, scenario: Scenario,
                          allowed_words: set[str]) -> tuple[list[Turn], list[str]]:
    errors = []
    for attempt in range(1, args.max_retries + 1):
        try:
            raw = call_ollama(
                args.base_url, args.model, args.temperature, args.timeout,
                scenario, args.business_name, args.business_context,
            )
            turns = parse_conversation(raw, minimum_turns(scenario))
            if not args.no_quality_gate:
                problems = conversation_problems(
                    [("user" if t.speaker == "caller" else "assistant", t.text) for t in turns],
                    scenario.intent, allowed_words,
                )
                if problems:
                    raise GenerationError("; ".join(problems))
            return turns, errors
        except GenerationError as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < args.max_retries:
                time.sleep(1)
    raise GenerationError("; ".join(errors))


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def render_status(done: int, total: int, examples: int, failed: int, elapsed: float, interactive: bool,
                  done_this_run: int | None = None) -> None:
    percent = done / total if total else 1.0
    filled = round(percent * BAR_WIDTH)
    bar = "#" * filled + "." * (BAR_WIDTH - filled)
    measured = done if done_this_run is None else done_this_run
    rate = elapsed / measured if measured else 0.0
    eta = format_duration(rate * (total - done)) if measured else "--"
    line = (f"[{bar}] {percent * 100:5.1f}%  {done}/{total} convs  "
            f"{examples} examples  {failed} failed  {rate:.0f}s/conv  ETA {eta}")
    if interactive:
        print(f"\r{line}", end="", flush=True)
    else:
        print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--count", type=int, default=500, help="conversations to generate")
    parser.add_argument("-o", "--out", type=Path, default=Path("booking_dataset.jsonl"), help="output JSONL path")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--business-name", default="Northgate Services")
    parser.add_argument(
        "--business-context",
        default="A local services business that books appointments by phone, weekdays 8am to 6pm and Saturday mornings.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--no-quality-gate", action="store_true",
                        help="keep conversations that fail the check_quality rules")
    parser.add_argument("--timeout", type=int, default=180, help="per-request timeout in seconds")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="conversations generated concurrently")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing output file, skipping conversations already in it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    system_prompt = ASSISTANT_SYSTEM_PROMPT.format(
        business_name=args.business_name, business_context=args.business_context
    )

    allowed_words = known_tokens(system_prompt) | known_tokens(args.business_name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failure_path = args.out.with_suffix(args.out.suffix + ".failures.log")

    generated = 0
    example_count = 0
    failures: list[tuple[Scenario, str]] = []
    started = time.time()

    interactive = sys.stdout.isatty()
    scenarios = {index: sample_scenario(rng) for index in range(1, args.count + 1)}

    already_done, mode = set(), "w"
    if args.resume and args.out.exists():
        with args.out.open(encoding="utf-8") as existing:
            for line in existing:
                row = json.loads(line)
                already_done.add(row["conversation_id"])
                example_count += 1
        generated = len(already_done)
        mode = "a"
        print(f"resuming: {generated} conversations already in {args.out}")

    pending = [(index, scenario) for index, scenario in scenarios.items() if index not in already_done]
    completed = resumed_count = generated
    render_status(completed, args.count, example_count, 0, 0.0, interactive)

    with args.out.open(mode, encoding="utf-8") as out_file, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_conversation, args, scenario, allowed_words): (index, scenario)
            for index, scenario in pending
        }
        for future in as_completed(futures):
            index, scenario = futures[future]
            try:
                turns, _ = future.result()
            except GenerationError as exc:
                failures.append((scenario, str(exc)))
                if interactive:
                    print()
                print(f"  skipped {index}/{args.count} ({scenario.persona}/{scenario.complication}): {exc}",
                      file=sys.stderr, flush=True)
            else:
                for example in build_examples(turns, system_prompt, scenario, index):
                    out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                    example_count += 1
                out_file.flush()
                generated += 1

            completed += 1
            render_status(completed, args.count, example_count, len(failures),
                          time.time() - started, interactive, completed - resumed_count)

    if interactive:
        print()

    if failures:
        with failure_path.open("w", encoding="utf-8") as log:
            for scenario, message in failures:
                log.write(json.dumps({
                    "service": scenario.service,
                    "persona": scenario.persona,
                    "complication": scenario.complication,
                    "intent": scenario.intent,
                    "error": message,
                }) + "\n")

    elapsed = time.time() - started
    print("\n--- summary ---")
    print(f"conversations requested: {args.count}")
    print(f"conversations written:   {generated}")
    print(f"turn-level examples:     {example_count}")
    print(f"failed generations:      {len(failures)}" + (f" (see {failure_path})" if failures else ""))
    if generated:
        print(f"examples per conversation: {example_count / generated:.1f}")
    print(f"elapsed: {elapsed / 60:.1f} min")
    print(f"output: {args.out}")
    return 0 if generated else 1


if __name__ == "__main__":
    sys.exit(main())
