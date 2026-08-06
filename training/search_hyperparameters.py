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
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

JOURNAL = Path("logs/hpsearch.jsonl")
WORKDIR = Path("checkpoints/hpsearch")

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
    different measurement, not a repeat of one already paid for."""
    if not path.exists():
        return {}
    scored = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("score") is not None:
            scored[(entry["key"], entry.get("conversations", 0))] = entry["score"]
    return scored


def parse_rungs(text: str) -> list[tuple[int, int]]:
    rungs = []
    for part in text.split(","):
        conversations, _, survivors = part.partition(":")
        rungs.append((int(conversations), int(survivors or 1)))
    return rungs


def evaluate(candidate: Candidate, args, index: int, conversations: int) -> float | None:
    """Train the candidate on this rung's conversations and return its held-out loss."""
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

    try:
        scores = json.loads(score_path.read_text())
        return next(iter(scores.values()))["heldout_loss"]
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
            scored.sort(key=lambda item: item[0])
            top_of_ladder = scored
            climbing = [c for _, c in scored[:survivors]]
            print(f"     best {scored[0][0]:.4f}, promoting {len(climbing)}\n", flush=True)

        if top_of_ladder:
            best_score, best_candidate = top_of_ladder[0]
            reached = rungs[min(len(rungs), max(1, len(rungs))) - 1][0]
            if champion is None or best_score < champion[0]:
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
    print(f"held-out loss {champion[0]:.4f} at {champion[2]} conversations")
    print("\nThat score comes from a ladder built for ranking, not for predicting. Confirm it "
          "with a full run before changing the defaults in train_lora.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
