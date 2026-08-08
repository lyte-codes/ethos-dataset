#!/usr/bin/env python3
"""Evolve the supervised hyperparameters, spending compute where it is earning something.

A full supervised run costs about two hours on this dataset and five and a half on the real
one. A genetic search needs tens of evaluations, so giving every candidate a full run would
take days. Instead each generation climbs a ladder: every candidate is trained on a small
number of conversations, only the best few are retrained on more, and only the best of those
sees the whole dataset. A bad configuration is eliminated after twelve minutes rather than
two hours, and the compute it would have wasted goes to a candidate that might win.

Scores from different rungs are never compared. A model trained on twenty conversations and
one trained on a hundred are not competing on equal terms, so ranking happens within a rung
and only the ordering is carried upward.

Every rung is a superset of the one below it — the same conversations plus more — so a
candidate that improves is improving on the data it already had, not being handed easier
data.

    python3 training/search_hyperparameters.py --population 6 --generations 3
    python3 training/search_hyperparameters.py --rungs 20:4,60:2,189:1   # convs:survivors

The proxy ranks, it does not predict. Confirm the winner with a full run before changing any
defaults.

**What a candidate is scored on, and why it changed.** This search originally ranked on
held-out loss alone, and that was a mistake with a name: v7. Held-out loss is only the
*fluency* half of what a build has to be good at, so searching on it found the most fluent
configuration we have ever trained (held-out 1.260 against v4's 1.342, and the smallest
memorisation gap of any build) which was simultaneously *worse than the untrained base
model* at preferring a correct phone number to a corrupted one — base 0.802, v7 0.745. On
its test call it confirmed a misread number back to the business as accurate. The search
had no faithfulness term, so it optimised hard for half the problem and was blind to the
half that actually matters for a booking agent.

Candidates are now scored on the same absolute composite that composite_score.py --absolute
reports, imported from there rather than reimplemented so the two can never drift:

    harmonic_mean( exp(-held_out_loss), sigmoid(macro_margin) )

Both terms are bounded in (0, 1) and mean something on their own, which matters here in a
way it does not on a leaderboard: candidates are measured at different times against no
common batch, so a batch-relative score would be meaningless. The harmonic mean is what
stops a repeat of v7 — it refuses to let a superb fluency number carry a faithlessness
number, because it collapses toward whichever term is weaker.

**Higher is better now.** Held-out loss was minimised; this is maximised. Scores in the
journal from before this change are held-out losses and are not comparable, so they are
tagged by objective and the old ones are ignored on resume rather than silently mixed in.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# The metric lives in composite_score.py and is imported, not copied — a search that
# optimises a subtly different number from the one the builds are judged on is exactly
# the failure this change exists to fix.
from composite_score import harmonic_mean, macro_margin

JOURNAL = Path("logs/hpsearch.jsonl")
WORKDIR = Path("checkpoints/hpsearch")

# Bumped whenever the meaning of "score" changes. Journal rows carry it, and a resume
# reuses only rows whose objective matches the one being run now.
OBJECTIVE = "composite_floor_v1"

# Where the base model's own measurements live, for the floor below.
BASELINE = Path("data/preference_eval.json")


def phone_floor() -> float | None:
    """The untrained base model's own phone margin, or None if it has not been measured.

    Blending alone does not prevent another v7, and the numbers say why: across the five
    builds measured, exp(-loss) spans a factor of 1.87 while sigmoid(margin) spans only
    1.14. The fluency term simply has more room to move, so it dominates any weighted or
    harmonic combination — under the composite alone, v7 still outranks every other build
    despite being worse than *no training at all* at preferring a correct phone number.

    So faithfulness is enforced as a constraint rather than traded off. A candidate that
    cannot beat the untrained model at the failure this agent exists to avoid is
    disqualified regardless of how fluent it is. The threshold is measured, not chosen:
    it is base's own score on the same pairs.
    """
    try:
        entry = json.loads(BASELINE.read_text()).get("base")
        return (entry or {}).get("margin_by_corruption", {}).get("phone")
    except (OSError, json.JSONDecodeError):
        return None

# (conversations, how many survive to the next rung). The last rung's survivor count is the
# number that breed the next generation.
DEFAULT_RUNGS = [(20, 3), (60, 2), (189, 1)]

# Ranges, not arbitrary points: the search interpolates within these, so the bounds are
# the actual prior about what is worth trying.
SPACE = {
    "lora_rank": [8, 16, 32, 64],
    "lora_alpha_ratio": [1.0, 2.0, 4.0],      # alpha = rank * ratio, the usual parameterisation
    "lora_dropout": [0.0, 0.05, 0.1],
    "learning_rate": [5e-5, 1e-4, 2e-4, 4e-4],
    "gradient_accumulation_steps": [2, 4, 8],
    "attention_only": [False, True],           # drop the MLP projections from the adapter
}


@dataclass(frozen=True)
class Candidate:
    lora_rank: int
    lora_alpha_ratio: float
    lora_dropout: float
    learning_rate: float
    gradient_accumulation_steps: int
    attention_only: bool

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def random_candidate(rng: random.Random) -> Candidate:
    return Candidate(**{field: rng.choice(values) for field, values in SPACE.items()})


def breed(a: Candidate, b: Candidate, rng: random.Random, mutation: float) -> Candidate:
    """Uniform crossover, then mutate each gene independently.

    Uniform rather than single-point: these genes have no meaningful order, so a cut
    point would impose a linkage between neighbours that does not exist.
    """
    genes = {}
    for field, values in SPACE.items():
        genes[field] = getattr(a if rng.random() < 0.5 else b, field)
        if rng.random() < mutation:
            genes[field] = rng.choice(values)
    return Candidate(**genes)


def load_journal(path: Path) -> dict[tuple[str, int], float]:
    """Scores are keyed by candidate *and* rung: the same configuration on more data is a
    different measurement, not a repeat of one already paid for.

    Rows written under a different objective are skipped. Held-out losses (~1.3, lower
    better) and composite scores (~0.4, higher better) occupy overlapping ranges while
    meaning opposite things, so mixing them would not raise an error — it would quietly
    rank the search backwards.
    """
    if not path.exists():
        return {}
    scored, stale = {}, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("score") is None:
            continue
        # Rows predating the objective field are held-out losses by definition.
        if entry.get("objective", "heldout_loss") != OBJECTIVE:
            stale += 1
            continue
        scored[(entry["key"], entry.get("conversations", 0))] = entry["score"]
    if stale:
        print(f"ignoring {stale} journal entr{'y' if stale == 1 else 'ies'} scored under a "
              f"previous objective", flush=True)
    return scored


def parse_rungs(text: str) -> list[tuple[int, int]]:
    rungs = []
    for part in text.split(","):
        conversations, _, survivors = part.partition(":")
        rungs.append((int(conversations), int(survivors or 1)))
    return rungs


def evaluate(candidate: Candidate, args, index: int, conversations: int) -> float | None:
    """Train the candidate and return its absolute composite — higher is better.

    Both halves are measured: held-out loss for whether it can still hold a conversation,
    preference margin for whether it would rather say what the brief contained than
    something it made up. See the module docstring for why one without the other is not
    enough.
    """
    output = WORKDIR / f"cand-{index:03d}"
    modules = ("q_proj,k_proj,v_proj,o_proj" if candidate.attention_only
               else "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    train = [
        sys.executable, "training/train_lora.py",
        "--data", str(args.data), "--output", str(output),
        "--batch-size", "1", "--epochs", str(args.proxy_epochs),
        "--learning-rate", str(candidate.learning_rate),
        "--lora-rank", str(candidate.lora_rank),
        "--lora-alpha", str(int(candidate.lora_rank * candidate.lora_alpha_ratio)),
        "--lora-dropout", str(candidate.lora_dropout),
        "--grad-accum", str(candidate.gradient_accumulation_steps),
        "--target-modules", modules,
        "--limit-conversations", str(conversations),
    ]
    log = Path("logs") / f"hpsearch-{index:03d}.log"
    with log.open("w") as handle:
        if subprocess.run(train, stdout=handle, stderr=subprocess.STDOUT).returncode != 0:
            return None

    score_path = WORKDIR / f"score-{index:03d}.json"
    scoring = [
        sys.executable, "training/eval_heldout.py",
        "--adapter", f"c{index}:{output}", "--data", str(args.data),
        "--train-sample", "20", "--dtype", "bfloat16", "--out", str(score_path),
    ]
    with log.open("a") as handle:
        if subprocess.run(scoring, stdout=handle, stderr=subprocess.STDOUT).returncode != 0:
            return None

    # The faithfulness half. Roughly six minutes against a candidate's twenty-plus, which
    # is a cheap price for not shipping another model that invents phone numbers.
    pref_path = WORKDIR / f"pref-{index:03d}.json"
    preference = [
        sys.executable, "training/eval_preference.py",
        "--adapter", f"c{index}:{output}", "--data", str(args.data),
        "--dtype", "bfloat16", "--out", str(pref_path),
    ]
    with log.open("a") as handle:
        if subprocess.run(preference, stdout=handle, stderr=subprocess.STDOUT).returncode != 0:
            return None

    try:
        loss = next(iter(json.loads(score_path.read_text()).values()))["heldout_loss"]
        pref_entry = next(iter(json.loads(pref_path.read_text()).values()))
        margin = macro_margin(pref_entry)
        if margin is None:
            return None
        # Identical to composite_score.py --absolute: undo the log on the loss to get the
        # model's own per-token probability, and put the margin through the same sigmoid
        # DPO's loss is built from. Neither needs another model to mean something.
        composite = round(harmonic_mean(math.exp(-loss), 1 / (1 + math.exp(-margin))), 4)

        floor = phone_floor()
        phone = pref_entry.get("margin_by_corruption", {}).get("phone")
        if floor is not None and phone is not None and phone < floor:
            print(f"     DISQUALIFIED  composite {composite:.4f} but phone margin "
                  f"{phone:.3f} < base {floor:.3f}", flush=True)
            # Recorded as measured rather than failed: it ran fine and the result is real,
            # it just is not allowed to win. Zero sorts it last without pretending the run
            # never happened, which a None would.
            return 0.0
        return composite
    except (OSError, json.JSONDecodeError, StopIteration, KeyError):
        return None
    finally:
        # A candidate's weights are worthless once scored, and forty of them is 9GB.
        subprocess.run(["rm", "-rf", str(output)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/ethos_booking_v2.jsonl"))
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--rungs", default=",".join(f"{c}:{s}" for c, s in DEFAULT_RUNGS),
                        help="conversations:survivors per rung, cheapest first")
    parser.add_argument("--mutation", type=float, default=0.25)
    parser.add_argument("--proxy-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    args = parser.parse_args()

    rungs = parse_rungs(args.rungs)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seen = load_journal(args.journal)
    if seen:
        print(f"resuming with {len(seen)} scores already measured\n", flush=True)

    counts = [args.population] + [survivors for _, survivors in rungs[:-1]]
    print("ladder per generation:")
    for (conversations, _), how_many in zip(rungs, counts):
        print(f"  {how_many} candidate(s) x {conversations} conversations")
    print(f"over {args.generations} generations\n", flush=True)

    population = [random_candidate(rng) for _ in range(args.population)]
    evaluated = 0
    champion: tuple[float, Candidate, int] | None = None

    for generation in range(1, args.generations + 1):
        print(f"=== generation {generation}/{args.generations} ===", flush=True)
        climbing = list(population)
        top_of_ladder: list[tuple[float, Candidate]] = []

        for rung, (conversations, survivors) in enumerate(rungs):
            print(f"  -- rung {rung}: {len(climbing)} candidate(s), "
                  f"{conversations} conversations --", flush=True)
            scored: list[tuple[float, Candidate]] = []

            for candidate in climbing:
                cache_key = (candidate.key(), conversations)
                if cache_key in seen:
                    score = seen[cache_key]
                    print(f"     cached  {score:.4f}", flush=True)
                else:
                    started = time.time()
                    score = evaluate(candidate, args, evaluated, conversations)
                    evaluated += 1
                    with args.journal.open("a") as handle:
                        handle.write(json.dumps({
                            "generation": generation, "rung": rung,
                            "conversations": conversations, "key": candidate.key(),
                            "candidate": asdict(candidate), "score": score,
                            "objective": OBJECTIVE,
                            "seconds": round(time.time() - started),
                        }) + "\n")
                    if score is None:
                        print(f"     FAILED        {asdict(candidate)}", flush=True)
                        continue
                    seen[cache_key] = score
                    print(f"     {score:.4f}  ({(time.time() - started) / 60:.0f}m)  "
                          f"{asdict(candidate)}", flush=True)
                scored.append((score, candidate))

            if not scored:
                print("  nothing survived this rung", file=sys.stderr)
                break

            # Ranking happens inside a rung and only the ordering moves up. Scores from
            # different amounts of data are not comparable and are never compared.
            # Descending: the composite is maximised, unlike the held-out loss it replaced.
            scored.sort(key=lambda item: item[0], reverse=True)
            top_of_ladder = scored
            climbing = [c for _, c in scored[:survivors]]
            print(f"     best {scored[0][0]:.4f}, promoting {len(climbing)}\n", flush=True)

        if top_of_ladder:
            best_score, best_candidate = top_of_ladder[0]
            reached = rungs[min(len(rungs), max(1, len(rungs))) - 1][0]
            if champion is None or best_score > champion[0]:
                champion = (best_score, best_candidate, reached)

        if generation < args.generations and top_of_ladder:
            parents = [c for _, c in top_of_ladder[:max(2, len(top_of_ladder))]]
            # The best survives untouched. Without elitism a good configuration can be lost
            # to one unlucky generation of children.
            population = [top_of_ladder[0][1]] + [
                breed(rng.choice(parents), rng.choice(parents), rng, args.mutation)
                for _ in range(args.population - 1)
            ]

    if champion is None:
        print("no candidate completed the ladder", file=sys.stderr)
        return 1

    print("=== best configuration ===")
    print(json.dumps(asdict(champion[1]), indent=2))
    print(f"composite {champion[0]:.4f} at {champion[2]} conversations (higher is better)")
    print("\nThat score comes from a ladder built for ranking, not for predicting. Confirm it "
          "with a full run before changing the defaults in train_lora.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
