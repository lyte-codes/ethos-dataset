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

Neither alone is a fair judge of a pipeline with both phases in it, so this combines
them as fluency and faithfulness. Every number in that combination is derived or
mathematically defined rather than picked by feel — see the docstring on each function
for where its default comes from and why:

- **Faithfulness is built from margin, not raw accuracy — and both are macro, not by
  frequency.** Binary accuracy on these pairs saturates near ceiling for every model
  measured, *including the untrained base model*: preferring a number already stated in
  the conversation over a different one is a general copying capability transformers get
  from pretraining, not something DPO specifically installs, so "did it pick the right
  one" stops discriminating almost immediately. The margin — how much more likely the
  right answer is than the wrong one — keeps moving where accuracy does not, and tracks
  training investment cleanly (base +1.30, v4 +1.45, v6 +1.54, v5 +2.01, measured). Both
  margin and accuracy are macro-averaged per corruption kind rather than pooled, for the
  same reason: `unearned_name` is 41% of the training pairs and `phone` is 12%, so a
  pooled score would grade mostly on names and barely count invented numbers.

- **The phone floor is chance, adjusted upward only if the untrained model already
  beats chance.** A two-way preference judgment has a mathematically defined floor —
  50%, a coin flip — with nothing arbitrary about it. If the base model's own accuracy
  on phone corruptions sits above 50%, the floor moves up to that, because "no better
  than the untrained model" is itself disqualifying and that number comes from a
  measurement in the same run rather than a guess.

- **A model under the floor is capped at the base model's own composite**, not at a
  fixed number. If it cannot beat the untrained model at the one failure that matters
  most, it does not get credit for beating the untrained model overall. Both sides of
  that comparison are computed, not chosen.

- **Fluency and faithfulness are weighted equally by default.** The temptation is to
  skew this toward faithfulness because docs/RELEASES.md treats invented details as the
  higher-priority failure — but that priority is already enforced structurally by the
  phone floor above. Skewing the blend *as well* double-counts the same preference in
  two places for no principled reason. Once the hard constraint is handled by the floor,
  nothing in the data says fluency and faithfulness should be weighted unevenly, so they
  are not. `--fluency-weight` remains available for anyone who wants to state an
  explicit, deliberate preference — the default just should not pretend to be one.

    python3 training/composite_score.py
    python3 training/composite_score.py --fluency-weight 0.65   # an explicit choice, not a default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHANCE = 0.5  # a two-outcome preference judgment; not a parameter, a fact about the task


def fluency_scores(heldout: dict) -> dict[str, float]:
    """0-100, derived entirely from measured losses: the base model anchors 0, the best
    loss actually observed in this batch anchors 100. No fixed scale — the range is
    whatever the batch achieved, so the numbers describe this run rather than an assumed
    target."""
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


def macro_margin(entry: dict) -> float | None:
    """Mean margin across corruption kinds, each kind weighted equally rather than by
    how often it appears in the training pairs — see the module docstring."""
    by_kind = entry.get("margin_by_corruption") or {}
    if not by_kind:
        return entry.get("mean_margin")  # older eval_preference.json without the breakdown
    return sum(by_kind.values()) / len(by_kind)


def faithfulness_scores(preference: dict) -> dict[str, float]:
    """0-100, base-relative — same derivation as fluency_scores, applied to macro-margin
    instead of loss. Accuracy alone cannot do this job: it is saturated at or near
    ceiling for every model on this pair set, including the untrained base, so a
    base-relative scale built from accuracy would collapse to near-zero span. Margin has
    not saturated, so it is what the scale is built from."""
    margins = {name: macro_margin(entry) for name, entry in preference.items()}
    margins = {name: value for name, value in margins.items() if value is not None}
    if "base" not in margins or len(margins) < 2:
        return {}
    base_margin = margins["base"]
    best_margin = max(margins.values())
    span = best_margin - base_margin
    if span <= 0:
        return {name: 0.0 for name in margins}
    return {
        name: round(max(0.0, min(100.0, 100 * (value - base_margin) / span)), 1)
        for name, value in margins.items()
    }


def phone_floor(heldout_names_with_pref: dict) -> tuple[float, str]:
    """Chance (50%), raised to the base model's own phone accuracy if that is higher.

    Returns the floor and a one-line explanation of where it came from, since "why is
    the floor 63%" should always be answerable from this run's own numbers."""
    base = heldout_names_with_pref.get("base")
    base_phone = (base or {}).get("by_corruption", {}).get("phone") if base else None
    if base_phone is not None and base_phone > CHANCE:
        return base_phone, f"base model's own phone accuracy ({base_phone:.0%}), above chance"
    return CHANCE, "chance (50%) — no base-model phone score available, or it did not beat chance"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heldout", type=Path, default=Path("data/heldout_eval.json"))
    parser.add_argument("--preference", type=Path, default=Path("data/preference_eval.json"))
    parser.add_argument("--fluency-weight", type=float, default=0.5,
                        help="deliberate override; the default is 0.5 because the hard "
                             "constraint on faithfulness is already enforced by the phone "
                             "floor, not by skewing this blend")
    parser.add_argument("--phone-floor", type=float, default=None,
                        help="override the derived floor (chance, or the base model's own "
                             "phone accuracy if higher) with an explicit value")
    parser.add_argument("--out", type=Path, default=Path("data/composite_score.json"))
    args = parser.parse_args()

    heldout = json.loads(args.heldout.read_text()) if args.heldout.exists() else {}
    preference = json.loads(args.preference.read_text()) if args.preference.exists() else {}
    fluency = fluency_scores(heldout)
    faithfulness = faithfulness_scores(preference)
    faithfulness_weight = 1 - args.fluency_weight

    floor, floor_reason = (args.phone_floor, "explicit --phone-floor override") \
        if args.phone_floor is not None else phone_floor(preference)
    print(f"phone floor: {floor:.0%} ({floor_reason})\n")

    names = sorted(set(fluency) | set(preference))
    if not names:
        print("no scores available yet — run eval_heldout.py and eval_preference.py first")
        return 1

    # The reference for capping — computed, not chosen. If base itself cannot be scored,
    # there is no non-arbitrary number to cap against, so a capped model is honestly
    # reported as unscored rather than pinned to a made-up constant.
    base_composite = None
    if "base" in fluency and "base" in faithfulness:
        base_composite = round(args.fluency_weight * fluency["base"]
                               + faithfulness_weight * faithfulness["base"], 1)

    results = {}
    print(f"{'model':<8} {'fluency':>9} {'faithful':>10} {'composite':>11}   accuracy by corruption kind")
    for name in names:
        f_score = fluency.get(name)
        pref_entry = preference.get(name) or {}
        faith_score = faithfulness.get(name)
        breakdown = pref_entry.get("by_corruption") or {}

        phone_rate = breakdown.get("phone")
        capped = phone_rate is not None and phone_rate < floor

        if f_score is None or faith_score is None:
            composite = None
        else:
            raw = round(args.fluency_weight * f_score + faithfulness_weight * faith_score, 1)
            if capped and base_composite is not None:
                composite = min(raw, base_composite)
            elif capped:
                composite = None  # no base reference to cap against — do not invent one
            else:
                composite = raw

        results[name] = {
            "fluency": f_score, "faithfulness": faith_score, "composite": composite,
            "phone_floor_breached": capped, "corruption_breakdown": breakdown,
        }

        row = f"{name:<8} {fmt(f_score):>9} {fmt(faith_score):>10} {fmt(composite):>11}"
        if capped:
            row += " CAPPED" if composite is not None else " (no base reference)"
        if breakdown:
            row += "   " + " / ".join(f"{k}={fmt_pct(v)}" for k, v in sorted(breakdown.items()))
        print(row)

    print(f"\nweights: fluency {args.fluency_weight:.0%}, faithfulness {faithfulness_weight:.0%} "
          f"(equal by default — see --help for why)")

    complete = {n: r for n, r in results.items() if r["composite"] is not None}
    if complete:
        best = max(complete.items(), key=lambda item: item[1]["composite"])
        print(f"best: {best[0]} ({best[1]['composite']})")
    incomplete = sorted(set(names) - set(complete))
    if incomplete:
        print(f"incomplete or capped without a base reference: {', '.join(incomplete)}")

    args.out.write_text(json.dumps({
        "weights": {"fluency": args.fluency_weight, "faithfulness": faithfulness_weight,
                    "phone_floor": floor, "phone_floor_reason": floor_reason,
                    "capped_reference": base_composite},
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
