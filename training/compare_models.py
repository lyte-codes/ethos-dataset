#!/usr/bin/env python3
"""Run the same calls through several models and score them side by side.

Every model gets the identical scenarios, in the identical order, decoded at
temperature 0. Without that the comparison measures sampling noise as readily as it
measures the models — and on a handful of calls the noise is easily the larger effect.

    python3 training/compare_models.py \\
        --model v4:checkpoints/ethos-v4 \\
        --model v5:checkpoints/ethos-v5 \\
        --model base:                       # empty path means the untuned base model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_call


def score(calls: list[dict]) -> dict:
    """Collapse a model's calls into the numbers that decide whether it is better."""
    total = len(calls)
    if not total:
        return {}

    audits = [call["audit"] for call in calls]
    misread = [a for a in audits if a.get("business_misread")]

    return {
        "calls": total,
        # The hard gate. One invented detail is a failure however good everything else is.
        "invented_details": sum(1 for a in audits if a["invented_digits"]),
        "gave_correct_number": sum(1 for a in audits if a["gave_correct_number"]),
        "identified_itself": sum(1 for a in audits if a["identified_itself"]),
        "named_the_client": sum(1 for a in audits if a["named_the_client"]),
        "impersonated_the_client": sum(1 for a in audits if a["impersonated_the_client"]),
        "committed_to_a_price": sum(1 for a in audits if a["committed_to_a_price"]),
        # Only meaningful where the business actually misread something, so it is reported
        # as a fraction of those calls rather than of all of them.
        "misread_opportunities": len(misread),
        "challenged_the_misread": sum(1 for a in misread if a.get("agent_challenged_the_misread")),
        "mean_turns": round(sum(c["turn_count"] for c in calls) / total, 1),
    }


def verdict(results: dict[str, dict]) -> list[str]:
    """State plainly which model wins and on what, rather than leaving a table to squint at."""
    lines = []
    clean = {name: s for name, s in results.items()
             if s.get("invented_details", 1) == 0 and s.get("impersonated_the_client", 1) == 0}
    if not clean:
        lines.append("No model is clean: every one either invented a detail or posed as its "
                     "client. None of these would pass the hard gate.")
    else:
        lines.append("Clean on the hard gates (no invented details, no impersonation): "
                     + ", ".join(sorted(clean)))

    def best(metric: str, of: dict) -> str | None:
        if not of:
            return None
        top = max(of.values(), key=lambda s: s.get(metric, 0)).get(metric, 0)
        winners = sorted(name for name, s in of.items() if s.get(metric, 0) == top)
        return f"{', '.join(winners)} ({top})"

    pool = clean or results
    for metric, label in [
        ("gave_correct_number", "gave the client's number correctly"),
        ("challenged_the_misread", "pushed back when the business misread a detail"),
        ("identified_itself", "said who it was"),
    ]:
        winner = best(metric, pool)
        if winner:
            lines.append(f"Best at {label}: {winner}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True,
                        help="name:path — repeat per model; an empty path is the base model")
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("data/model_comparison.json"))
    parser.add_argument("--base-model", default=fake_call.BASE_MODEL)
    parser.add_argument("--business-model", default="llama3.2:3b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    arguments = parser.parse_args()

    try:
        fake_call.ollama_chat([{"role": "user", "content": "ping"}],
                              arguments.business_model, arguments.base_url, 0.0, timeout=30)
    except fake_call.OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results, transcripts = {}, {}
    for spec in arguments.model:
        name, _, path = spec.partition(":")
        adapter = Path(path) if path else None
        if adapter is not None and not adapter.exists():
            print(f"skipping {name}: {adapter} not found", file=sys.stderr)
            continue

        print(f"\n=== {name} ({adapter or 'untuned base'}) ===", flush=True)
        call_arguments = argparse.Namespace(
            agent_name="Ethos",
            business_name="Northgate Services",
            business_context="A local services business that books appointments by phone, "
                             "weekdays 8am to 6pm and Saturday mornings.",
            max_turns=14, max_new_tokens=110,
            # Temperature 0 on both sides: the comparison has to be about the models.
            temperature=0.0, business_temperature=0.0,
            business_model=arguments.business_model, base_url=arguments.base_url,
        )
        tokenizer, model, device = fake_call.load_agent(arguments.base_model, adapter, "auto")

        calls = []
        for index, scenario in enumerate(fake_call.SCENARIOS[:arguments.calls], 1):
            call = fake_call.run_call(scenario, tokenizer, model, device, call_arguments)
            call["audit"] = fake_call.audit(call, "Ethos")
            calls.append(call)
            flags = [k for k, v in call["audit"].items() if v not in (None, False, [], 0)]
            print(f"  [{index}] {scenario['brief'].client_name}: {', '.join(flags)}", flush=True)

        results[name] = score(calls)
        transcripts[name] = calls
        del model, tokenizer

    print("\n" + "=" * 72)
    keys = ["calls", "invented_details", "impersonated_the_client", "gave_correct_number",
            "identified_itself", "challenged_the_misread", "misread_opportunities",
            "committed_to_a_price", "mean_turns"]
    header = f"{'metric':<26}" + "".join(f"{name:>10}" for name in results)
    print(header)
    print("-" * len(header))
    for key in keys:
        print(f"{key:<26}" + "".join(f"{results[name].get(key, '-'):>10}" for name in results))

    print()
    for line in verdict(results):
        print(line)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(
        {"scores": results, "transcripts": transcripts}, indent=2))
    print(f"\nwritten to {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
