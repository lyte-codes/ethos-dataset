#!/usr/bin/env python3
"""One score for both training phases, fair to what each one actually optimises.

Held-out loss and preference accuracy each judge one phase well and the other phase
badly:

- **Held-out loss** measures whether the model still does the job — coherent, on-brief
  turns on conversations it never trained on. It is the right instrument for supervised
  training. It is the *wrong* one for DPO: DPO optimises the margin between chosen and
  rejected, not the likelihood of chosen in isolation, so a preference run can widen
  that margin while making the chosen response less likely too. That reads as a worse
  held-out loss on a better model — we measured exactly this for v5 already.

- **Preference accuracy** measures whether the model, given a business's turn, would
  rather say the thing that matches the brief than the thing that doesn't. It is the
  right instrument for judging invented details, and it says nothing about whether the
  model can hold a coherent conversation at all — a model that always refuses would
  score adequately here while being useless.

Neither one alone is a fair judge of a pipeline with both phases in it. This combines
them:

    fluency       = held-out loss, scaled 0-100 relative to the untuned base model
    faithfulness  = preference accuracy, weighted by how costly each corruption is —
                    phone digits count for more than an unearned name, because a wrong
                    number reaches a real business and a wrong name does not
    composite     = 0.4 x fluency + 0.6 x faithfulness

Faithfulness outweighs fluency because that is the priority docs/RELEASES.md already
states: invented details are a hard gate at every promotion level, fluency is not. A
model that is pleasant to listen to and confidently wrong about a phone number is the
one failure mode this whole pipeline exists to prevent — the weighting says so in the
score, not just in prose.

**A phone-accuracy floor overrides the arithmetic.** Below the floor, the composite is
capped regardless of how well everything else scores — a weighted average lets a bad
phone number hide behind good numbers everywhere else, and the actual gate does not.

    python3 training/composite_score.py
    python3 training/composite_score.py --fluency-weight 0.5 --phone-floor 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Priority mirrors generation/build_preference_pairs.py: phone first because it is the
# costliest detail to get wrong and the one that appears in the fewest responses, then
# time and weekday as roughly equal secondary failures, then an unearned name last,
# since it applies everywhere and would otherwise swamp the average.
CORRUPTION_WEIGHT = {"phone": 0.40, "time": 0.25, "weekday": 0.20, "unearned_name": 0.15}

DEFAULT_PHONE_FLOOR = 0.60   # a hair above chance; below this, capped regardless
CAPPED_SCORE = 35.0          # where a phone-floor breach lands, out of 100


def fluency_scores(heldout: dict) -> dict[str, float]:
    """0-100, base model pinned at 0. Scaled to the best loss in the batch rather than
    an arbitrary constant, so the scale reflects what has actually been achieved."""
    if "base" not in heldout or len(heldout) < 2:
        return {}
    base_loss = heldout["base"]["heldout_loss"]
    best_loss = min(v["heldout_loss"] for v in heldout.values())
    span = base_loss - best_loss
    if span <= 0:
        return {name: 0.0 for name in heldout}
    return {
        name: round(max(0.0, min(100.0, 100 * (base_loss - v["heldout_loss"]) / span)), 1)
        for name, v in heldout.items()
    }


def faithfulness_score(entry: dict) -> tuple[float, dict[str, float]]:
    """Weighted accuracy across corruption kinds actually observed for this model.

    Falls back to overall accuracy for weight the observed kinds do not cover, so a
    model missing a rare corruption type in its sample is not silently under-weighted.
    """
    by_kind = entry.get("by_corruption") or {}
    overall = entry.get("accuracy")
    if overall is None:
        return float("nan"), {}

    weighted_sum, weight_used = 0.0, 0.0
    breakdown = {}
    for kind, weight in CORRUPTION_WEIGHT.items():
        rate = by_kind.get(kind, overall)  # unseen kind: assume it behaves like the average
        breakdown[kind] = rate
        weighted_sum += weight * rate
        weight_used += weight
    score = 100 * weighted_sum / weight_used if weight_used else 100 * overall
    return round(score, 1), breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heldout", type=Path, default=Path("data/heldout_eval.json"))
    parser.add_argument("--preference", type=Path, default=Path("data/preference_eval.json"))
    parser.add_argument("--fluency-weight", type=float, default=0.4)
    parser.add_argument("--phone-floor", type=float, default=DEFAULT_PHONE_FLOOR,
                        help="phone-corruption accuracy below this caps the composite, "
                             "matching the hard promotion gate rather than averaging past it")
    parser.add_argument("--out", type=Path, default=Path("data/composite_score.json"))
    args = parser.parse_args()

    heldout = json.loads(args.heldout.read_text()) if args.heldout.exists() else {}
    preference = json.loads(args.preference.read_text()) if args.preference.exists() else {}
    fluency = fluency_scores(heldout)
    faithfulness_weight = 1 - args.fluency_weight

    names = sorted(set(fluency) | set(preference))
    if not names:
        print("no scores available yet — run eval_heldout.py and eval_preference.py first")
        return 1

    results = {}
    print(f"{'model':<8} {'fluency':>9} {'faithful':>10} {'composite':>11}   phone / time / weekday / name")
    for name in names:
        f_score = fluency.get(name)
        pref_entry = preference.get(name)
        faith_score, breakdown = faithfulness_score(pref_entry) if pref_entry else (None, {})

        capped = faith_score is not None and breakdown.get("phone", 1.0) < args.phone_floor
        if f_score is None or faith_score is None:
            composite = None
        elif capped:
            composite = min(CAPPED_SCORE, args.fluency_weight * f_score + faithfulness_weight * faith_score)
        else:
            composite = round(args.fluency_weight * f_score + faithfulness_weight * faith_score, 1)

        results[name] = {
            "fluency": f_score, "faithfulness": faith_score, "composite": composite,
            "phone_floor_breached": capped, "corruption_breakdown": breakdown,
        }

        row = (f"{name:<8} {fmt(f_score):>9} {fmt(faith_score):>10} "
               f"{fmt(composite):>11}{'  CAPPED' if capped else ''}")
        if breakdown:
            row += ("   " + " / ".join(fmt_pct(breakdown.get(k)) for k in CORRUPTION_WEIGHT))
        print(row)

    print(f"\nweights: fluency {args.fluency_weight:.0%}, faithfulness {faithfulness_weight:.0%} "
          f"| phone floor {args.phone_floor:.0%}")

    complete = {n: r for n, r in results.items() if r["composite"] is not None}
    if complete:
        best = max(complete.items(), key=lambda item: item[1]["composite"])
        print(f"best: {best[0]} ({best[1]['composite']})")
    incomplete = sorted(set(names) - set(complete))
    if incomplete:
        print(f"incomplete (missing one eval): {', '.join(incomplete)}")

    args.out.write_text(json.dumps({
        "weights": {"fluency": args.fluency_weight, "faithfulness": faithfulness_weight,
                    "phone_floor": args.phone_floor},
        "scores": results,
    }, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


if __name__ == "__main__":
    raise SystemExit(main())
