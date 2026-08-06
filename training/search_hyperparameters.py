#!/usr/bin/env python3
"""Evolve the supervised hyperparameters against a cheap proxy of the real run.

A full supervised run costs about five and a half hours here. A genetic search needs
tens of evaluations, so run against the real thing it would take days — the wrong tool
for that cost structure. Each candidate is therefore trained on a subset for a fraction
of the steps, ranked on held-out loss, and only the survivors are worth a full run.

The proxy is a ranking, not a prediction. It answers "is this configuration better than
that one" well enough to breed from, and says nothing trustworthy about what the final
loss will be. Confirm the winner with a full run before believing it.

    python3 training/search_hyperparameters.py --population 6 --generations 3
    python3 training/search_hyperparameters.py --resume    # after an interruption

Every candidate ever evaluated is appended to the journal, so an interrupted search
resumes instead of re-paying for scores it already has.
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


def load_journal(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    scored = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("score") is not None:
            scored[entry["key"]] = entry["score"]
    return scored


def evaluate(candidate: Candidate, args, index: int) -> float | None:
    """Train the candidate on the proxy and return its held-out loss. None if it failed."""
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
        "--limit", str(args.proxy_examples),
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
    parser.add_argument("--survivors", type=int, default=2, help="how many breed the next generation")
    parser.add_argument("--mutation", type=float, default=0.25)
    parser.add_argument("--proxy-examples", type=int, default=400)
    parser.add_argument("--proxy-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    args = parser.parse_args()

    WORKDIR.mkdir(parents=True, exist_ok=True)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seen = load_journal(args.journal)
    if seen:
        print(f"resuming with {len(seen)} candidates already scored\n", flush=True)

    population = [random_candidate(rng) for _ in range(args.population)]
    evaluated = 0
    best_overall: tuple[float, Candidate] | None = None

    for generation in range(1, args.generations + 1):
        print(f"=== generation {generation}/{args.generations} ===", flush=True)
        scored: list[tuple[float, Candidate]] = []

        for candidate in population:
            if candidate.key() in seen:
                score = seen[candidate.key()]
                print(f"  cached  {score:.4f}  {asdict(candidate)}", flush=True)
            else:
                started = time.time()
                score = evaluate(candidate, args, evaluated)
                evaluated += 1
                with args.journal.open("a") as handle:
                    handle.write(json.dumps({
                        "generation": generation, "key": candidate.key(),
                        "candidate": asdict(candidate), "score": score,
                        "seconds": round(time.time() - started),
                    }) + "\n")
                if score is None:
                    print(f"  FAILED        {asdict(candidate)}", flush=True)
                    continue
                seen[candidate.key()] = score
                print(f"  {score:.4f}  ({time.time() - started:.0f}s)  {asdict(candidate)}", flush=True)
            scored.append((score, candidate))

        if not scored:
            print("no candidate in this generation trained successfully", file=sys.stderr)
            return 1

        scored.sort(key=lambda item: item[0])
        if best_overall is None or scored[0][0] < best_overall[0]:
            best_overall = scored[0]
        print(f"  best this generation: {scored[0][0]:.4f}\n", flush=True)

        if generation < args.generations:
            parents = [c for _, c in scored[:max(2, args.survivors)]]
            # The best survives unchanged. Without that, a good configuration can be lost
            # to an unlucky generation of children.
            population = [scored[0][1]] + [
                breed(rng.choice(parents), rng.choice(parents), rng, args.mutation)
                for _ in range(args.population - 1)
            ]

    print("=== best configuration ===")
    print(json.dumps(asdict(best_overall[1]), indent=2))
    print(f"proxy held-out loss {best_overall[0]:.4f}")
    print("\nThis is a ranking, not a prediction. Confirm it with a full run before "
          "changing the defaults in train_lora.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
