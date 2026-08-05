#!/usr/bin/env python3
"""Generate synthetic phone-booking conversations with a local Ollama model."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
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
    "rambling": "Talks around the point, drifts into unrelated detail about the shop's week, and buries the actual answer inside a long sentence.",
    "polite": "Warm and helpful, thanks the agent often, apologises for any delay, answers clearly.",
    "impatient": "Busy and short on time, pushes to get the call over with, interrupts with 'what day do you want'.",
    "confused": "New to the desk, unsure what the diary shows, mixes up days of the week, asks the agent to repeat things.",
}

COMPLICATIONS = {
    "no_availability": "The first day or time the agent asks for is not free. The business offers alternatives and the agent has to pick one that fits the client's stated availability, or say it will check back with the client.",
    "asks_beyond_brief": "The business asks for something the brief does not cover — an account number, an access code, whether there is parking. The agent must say it will check with the client rather than answer.",
    "wrong_readback": "When the business reads the booking back, it gets one detail wrong — a digit of the phone number, the weekday, or the time. The agent notices and corrects it before agreeing.",
    "transfers_the_call": "The first person cannot book and passes the agent to someone who can. The agent restates the request concisely to the second person without changing any detail.",
    "asks_about_cost": "The business mentions a call-out charge or asks whether the client has approved a cost. The agent does not commit to a price and says it will confirm with the client.",
    "mishearing": "There is background noise on the line. The business mishears a detail once and the agent repeats or spells it out, using only what the brief says.",
}

CALL_INTENTS = {
    "new_booking": "The agent is calling to book a new appointment for the client.",
    "reschedule": "The client already has an appointment and the agent is calling to move it to a different day or time.",
    "cancellation": "The agent is calling to cancel the client's existing appointment, and to rebook only if the brief says to.",
}

INTENT_WEIGHTS = {"new_booking": 0.65, "reschedule": 0.2, "cancellation": 0.15}

AGENT_SYSTEM_PROMPT = (
    "You are a booking agent. You are phoning {business_name} on behalf of your client to "
    "arrange an appointment. {business_context}\n"
    "\n"
    "Your client's brief — these are the only facts you have:\n"
    "- Name: {client_name}\n"
    "- Contact number: {client_phone}\n"
    "- Needs: {service}\n"
    "- Available: {availability}\n"
    "\n"
    "Say who you are and who you are calling for. Ask for what your client needs and agree a "
    "specific weekday and clock time that falls inside your client's availability. Give a "
    "detail from the brief only when the business asks for it, and never state a name, number, "
    "date or time that is not in the brief. If the business asks for anything the brief does "
    "not cover, say you will check with your client rather than guessing. When the business "
    "reads the booking back, check every detail against the brief and correct anything they "
    "have wrong before you agree to it."
)

GENERATOR_SYSTEM_PROMPT = """You write realistic transcripts of phone calls made BY a booking agent TO a small business.

The assistant in these transcripts is the agent placing the call on behalf of a client. It is not the business. The business is the other speaker, and it is the one that answers the phone, checks the diary and reads the booking back.

You will be given the agent's brief — the client's name, contact number, what they need and when they are free — plus a persona for whoever answers at the business, and a complication. Write the whole call as it would actually sound on the phone.

Rules:
- The business speaks first, and its first turn answers the phone the way a real business does — naming the business, e.g. "Good morning, Northgate Services, how can I help?". Never open with a bare day or time.
- Every business turn is a real spoken sentence. Even a terse or impatient person says "Monday's full, what about Wednesday?" rather than "Monday".
- The assistant speaks last, and the last assistant turn closes the call.
- Speakers strictly alternate: business, assistant, business, assistant, and so on.
- Between 8 and 20 turns in total.
- Spoken language only. No stage directions, no "[background noise]", no speaker labels inside the text, no markdown.
- The assistant is NOT the client. It is a third party calling for them. It introduces itself as calling "on behalf of" the named client and refers to that client in the third person throughout ("she is free Monday", "his number is ..."). It must never say "this is <client name>", "my name is <client name>", or otherwise speak as though it were the client.
- The assistant knows ONLY what the brief says. It must never state a name, phone number, date or time that is not in the brief. It gives each detail only when the business asks for it.
- If the business asks for anything outside the brief, the assistant says it will check with the client. It never invents an answer and never commits to a price.
- The business and the assistant settle on a specific weekday and clock time that falls inside the client's stated availability.
- The business reads the booking back near the end. The assistant checks it against the brief and explicitly corrects any detail the business got wrong.
- Use real-sounding specifics. Never write a placeholder such as [name], [date] or XXX-XXXX.
- The persona shapes the business's turns, not the assistant's. The assistant stays professional throughout.

Reply with JSON only, in exactly this shape:
{"turns": [{"speaker": "business", "text": "..."}, {"speaker": "assistant", "text": "..."}]}"""

GENERATOR_USER_PROMPT = """Business being called: {business_name}
Context: {business_context}

The agent's brief:
- Client name: {client_name}
- Client contact number: {client_phone}
- Needs: {service}
- Client is available: {availability}

Call type: {intent_description}
Persona of whoever answers at the business ({persona}): {persona_description}
Complication: {complication_description}

{checklist}
Write the full call as JSON."""

BOOKING_CHECKLIST = """Before you finish, check the call contains all of these:
1. the assistant says who it is calling on behalf of, using the client name from the brief
2. the assistant speaks the client's contact number out loud, as digits, only after the business asks for it
3. the business and the assistant settle on a specific weekday and a clock time inside the client's availability
4. the business reads the booking back, and the assistant confirms or corrects it against the brief
5. the assistant never states a name, number, day or time that is not in the brief
"""

CANCELLATION_CHECKLIST = """Before you finish, check the call contains all of these:
1. the assistant identifies the client by the name in the brief and gives enough detail to find the appointment
2. the business confirms the cancellation explicitly
3. the assistant never states a name, number, day or time that is not in the brief
"""

BAR_WIDTH = 24
MIN_TURNS = 8
MIN_TURNS_CANCELLATION = 4
MIN_TURNS_TERSE = 6
MAX_TURNS = 40
MAX_TURN_CHARS = 800

# The agent places the call, so it is the assistant; whoever picks up at the business is the
# other speaker. "caller" is deliberately mapped to the business — a generator that slips back
# into the old framing labels the phone-answering turn "caller", and that is still the business.
SPEAKER_ALIASES = {
    "business": "business",
    "receptionist": "business",
    "staff": "business",
    "shop": "business",
    "caller": "business",
    "user": "business",
    "assistant": "assistant",
    "agent": "assistant",
    "booking agent": "assistant",
}

LABEL_PREFIX = re.compile(
    r"^\s*(business|receptionist|staff|shop|caller|user|assistant|agent)\s*:\s*", re.IGNORECASE
)
SALVAGE_TURN = re.compile(
    r'"(?:speaker|role)"\s*:\s*"([^"]+)"\s*,\s*"(?:text|content|message)"\s*:\s*"((?:[^"\\]|\\.)*)"'
)


class GenerationError(Exception):
    """Raised when a generation cannot be turned into a usable conversation."""


CLIENT_NAMES = [
    "Marcus Whitfield", "Priya Raman", "Denise Okoro", "Alan Beattie", "Sophie Lindqvist",
    "Tomas Nowak", "Grace Adeyemi", "Hugh Fairbairn", "Nadia Haddad", "Ruth Kelleher",
    "Callum Doherty", "Yusuf Demir", "Eleanor Pryce", "Samir Chaudhry", "Bridget Moloney",
]

AVAILABILITY = [
    "Tuesday or Wednesday morning",
    "any weekday afternoon after two",
    "Thursday or Friday, before midday",
    "Saturday morning only",
    "Monday afternoon, or Wednesday any time",
    "weekday mornings, not Tuesday",
]


@dataclass(frozen=True)
class ClientBrief:
    """What the agent knows before it dials — and the only thing it may say out loud."""
    name: str
    phone: str
    availability: str


@dataclass(frozen=True)
class Scenario:
    service: str
    persona: str
    complication: str
    intent: str
    client: ClientBrief


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


def sample_phone(rng: random.Random) -> str:
    """Ofcom reserves 07700 900000-900999 for drama, so these can never be a real number."""
    return f"07700 900{rng.randint(0, 999):03d}"


def sample_scenario(rng: random.Random) -> Scenario:
    intents = list(INTENT_WEIGHTS)
    return Scenario(
        service=rng.choice(SERVICE_TYPES),
        persona=rng.choice(list(PERSONAS)),
        complication=rng.choice(list(COMPLICATIONS)),
        intent=rng.choices(intents, weights=[INTENT_WEIGHTS[i] for i in intents])[0],
        client=ClientBrief(
            name=rng.choice(CLIENT_NAMES),
            phone=sample_phone(rng),
            availability=rng.choice(AVAILABILITY),
        ),
    )


def agent_system_prompt(business_name: str, business_context: str, scenario: Scenario) -> str:
    return AGENT_SYSTEM_PROMPT.format(
        business_name=business_name,
        business_context=business_context,
        client_name=scenario.client.name,
        client_phone=scenario.client.phone,
        service=scenario.service,
        availability=scenario.client.availability,
    )


def conversation_schema(min_turns: int, max_turns: int) -> dict:
    """Constrain decoding to the turn structure, so shape errors cannot occur at all."""
    return {
        "type": "object",
        "properties": {
            "turns": {
                "type": "array",
                "minItems": min_turns,
                "maxItems": max_turns,
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": ["business", "assistant"]},
                        "text": {"type": "string"},
                    },
                    "required": ["speaker", "text"],
                },
            }
        },
        "required": ["turns"],
    }


def call_ollama(base_url: str, model: str, temperature: float, timeout: int, scenario: Scenario,
                business_name: str, business_context: str, schema: dict | None = None,
                num_predict: int = 2048, keep_alive: str = "30m") -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": schema if schema is not None else "json",
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "top_p": 0.95, "num_predict": num_predict},
        "messages": [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GENERATOR_USER_PROMPT.format(
                    business_name=business_name,
                    business_context=business_context,
                    client_name=scenario.client.name,
                    client_phone=scenario.client.phone,
                    availability=scenario.client.availability,
                    service=scenario.service,
                    intent_description=CALL_INTENTS[scenario.intent],
                    persona=scenario.persona,
                    persona_description=PERSONAS[scenario.persona],
                    complication_description=COMPLICATIONS[scenario.complication],
                    checklist=(CANCELLATION_CHECKLIST if scenario.intent == "cancellation"
                               else BOOKING_CHECKLIST),
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
    while merged and merged[0].speaker != "business":
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
    if turns[0].speaker != "business":
        raise GenerationError("conversation does not start with the business")
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
        if turn.speaker == "business":
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
                          business_words: set[str]) -> tuple[list[Turn], list[str]]:
    # The brief is per-conversation, so what the agent is allowed to say is too: the client's
    # own name is legitimate for it to speak, and any other name is an invention.
    brief = scenario.client
    allowed_words = business_words | known_tokens(brief.name)
    errors = []
    for attempt in range(1, args.max_retries + 1):
        try:
            min_turns = minimum_turns(scenario)
            raw = call_ollama(
                args.base_url, args.model, args.temperature, args.timeout,
                scenario, args.business_name, args.business_context,
                schema=None if args.no_schema else conversation_schema(min_turns, args.max_turns),
                num_predict=args.num_predict,
            )
            turns = parse_conversation(raw, min_turns)
            if not args.no_quality_gate:
                problems = conversation_problems(
                    [("user" if t.speaker == "business" else "assistant", t.text) for t in turns],
                    scenario.intent, allowed_words, brief_digits=brief.phone,
                    client_name=brief.name,
                )
                if problems:
                    raise GenerationError("; ".join(problems))
            return turns, errors
        except GenerationError as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < args.max_retries:
                time.sleep(1)
    raise GenerationError("; ".join(errors))


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_manifest(args: argparse.Namespace, generated: int, examples: int, failed: int,
                   scenarios: dict[int, Scenario]) -> Path:
    """Record exactly what produced this file, so a dataset version is reproducible."""
    here = Path(__file__).resolve().parent
    counts = {
        field: dict(collections.Counter(getattr(s, field) for s in scenarios.values()))
        for field in ("persona", "complication", "intent", "service")
    }
    manifest = {
        "dataset_version": args.dataset_version,
        "created": datetime.date.today().isoformat(),
        "conversations": generated,
        "turn_level_examples": examples,
        "failed_generations": failed,
        "generator": {
            "model": args.model,
            "temperature": args.temperature,
            "max_retries": args.max_retries,
            "quality_gate": not args.no_quality_gate,
            "seed": args.seed,
        },
        "code": {
            "generate_dataset.py": file_digest(here / "generate_dataset.py"),
            "check_quality.py": file_digest(here / "check_quality.py"),
        },
        "business": {"name": args.business_name, "context": args.business_context},
        "agent_role": "outbound — the assistant phones the business on behalf of a client",
        # The brief differs per conversation, so the template is what identifies the dataset;
        # the filled-in version lives in each example's system message.
        "agent_system_prompt_template": AGENT_SYSTEM_PROMPT,
        "scenario_counts": counts,
        "data_sha256_16": file_digest(args.out) if args.out.exists() else None,
    }
    path = args.out.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


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
    parser.add_argument("--no-schema", action="store_true",
                        help="use free-form JSON instead of constrained decoding")
    parser.add_argument("--max-turns", type=int, default=16,
                        help="upper bound enforced by the schema")
    parser.add_argument("--num-predict", type=int, default=1400,
                        help="token budget per conversation")
    parser.add_argument("--no-quality-gate", action="store_true",
                        help="keep conversations that fail the check_quality rules")
    parser.add_argument("--timeout", type=int, default=180, help="per-request timeout in seconds")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-version", default="dev",
                        help="version label recorded in the manifest, e.g. v1")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="conversations generated concurrently")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing output file, skipping conversations already in it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    # Every conversation gets its own system prompt because the brief differs per call.
    business_words = known_tokens(args.business_name) | known_tokens(args.business_context)

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
            pool.submit(generate_conversation, args, scenario, business_words): (index, scenario)
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
                system_prompt = agent_system_prompt(
                    args.business_name, args.business_context, scenario)
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

    manifest_path = write_manifest(args, generated, example_count, len(failures), scenarios)

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
    print(f"manifest: {manifest_path} (dataset_version={args.dataset_version})")
    return 0 if generated else 1


if __name__ == "__main__":
    sys.exit(main())
