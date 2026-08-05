# Releases

[VERSIONING.md](VERSIONING.md) covers the internal counters — which dataset produced which
adapter. This document covers the layer above it: what gets served, how a daily training run
becomes a shipped product, and how to undo it when a release turns out to be wrong.

The two layers exist because they answer different questions. `model v7` answers *which
experiment was this*. `2.1` answers *what will happen to my integration if I upgrade*. A
research counter cannot answer the second question, because it goes up for a learning-rate
tweak and for a total change of what the agent does.

---

## The two layers

```
lineage        dataset ethos-booking-v2  ->  model v7            immutable, internal
                                              |
release        2.1-nightly.20260805.a3f9c21                    immutable, published
                                              |
channel        nightly -> beta -> stable                         mutable pointers
```

**Builds are immutable.** Once `2.1-nightly.20260805.a3f9c21` exists it is never rebuilt or
overwritten. If it is wrong, it is superseded, never edited.

**Channels are pointers.** `stable` is a name for whichever build is currently shipped, the way
a git branch names a commit. Rolling back is repointing `stable` at the previous build — it
takes effect immediately and needs no retraining.

---

## Release numbers

Serving versions are **MAJOR.MINOR** — two numbers, no patch. The number describes **the
behaviour contract**, not the size of the code change.

| Bump | When | Example |
|---|---|---|
| **MAJOR** | What the agent *is* changes. Anyone relying on the old behaviour breaks. | `1.4 -> 2.0` when the assistant stopped being the business's receptionist and became the agent that calls the business for you |
| **MINOR** | Anything else that ships. New capability, better accuracy, a fixed failure mode. | handles rescheduling; covers a new service category; stops mangling phone numbers |

The rule for telling them apart: if a user's existing expectations would now be wrong, it is
MAJOR. Everything else is MINOR. Improving how well the agent reads a phone number back is
MINOR. Changing *who reads it back* is MAJOR.

There is no patch number because there is nothing for it to mean here. A patch level earns its
keep when users need to distinguish "same behaviour, security fix" from "new behaviour" — but
every release here is a retrained model, and a retrained model always behaves at least
slightly differently. Calling one of those a patch would imply a stability it does not have.
Minor goes up; that is the whole scheme.

### Prerelease identifiers

Semver precedence orders these correctly, which is why the format is worth keeping to:

```
2.1-nightly.20260805.a3f9c21   <   2.1-rc.3   <   2.1
```

- **nightly** — `<next-version>-nightly.<YYYYMMDD>.<git-sha8>`. One per training run.
- **rc** — `<next-version>-rc.<n>`. A nightly that cleared the full eval suite and is soaking.
- **stable** — `<major>.<minor>`. Promoted from an rc. Never built directly.

A nightly carries the version it is *heading toward*. Most days that is the next minor:
running against a shipped 2.0 produces `2.1-nightly.…`. If the role changed, it is
`3.0-nightly.…` from the first run after the change.

---

## Channels

| Channel | Points at | Who consumes it | Retention |
|---|---|---|---|
| `nightly` | Latest passing nightly build | You, and the eval harness | Last 14 |
| `beta` | Current release candidate | Opt-in users, staging | All rcs of the current release |
| `stable` | Currently shipped build | Production, default for every client | Every stable, forever |

Production never resolves `nightly`. Clients either pin `stable` or pin an exact version;
pinning exact is the right default for anyone who cares about reproducibility, and `stable`
is right for anyone who wants fixes automatically.

---

## The daily run

A nightly is not just "train again" — it is the full chain, and any step failing means no
nightly is published that day. Publishing nothing is a normal outcome and better than
publishing something unevaluated.

1. **Generate** the day's delta data with the pinned generator, appending a new dataset
   version if the generator or quality rules changed.
2. **Train** from the current base, writing `hyperparameters.json` and the dataset hash into
   the checkpoint.
3. **Evaluate** against the regression set (below). Fail the run on any hard gate.
4. **Publish** as `…-nightly.<date>.<sha>` and repoint the `nightly` channel.
5. **Record** the build in the registry with everything needed to rebuild it.

Because nightlies are immutable and cheap to keep, a regression that appears on the 12th is
bisectable against the 11th, the 10th, and so on.

---

## Promotion gates

A build only moves up a channel by clearing the gate below it. These are the metrics that
matter for an agent that phones a business on your behalf — the costly failure is not an
awkward sentence, it is confidently stating a phone number nobody gave it.

| Gate | nightly | beta | stable |
|---|---|---|---|
| Invented caller detail (phone, name, date, time) on the regression set | 0 | 0 | 0 |
| Booking completed within the turn budget | ≥ 80% | ≥ 90% | ≥ 90% |
| Business's readback correctly verified or corrected | ≥ 70% | ≥ 85% | ≥ 85% |
| Never quotes a price it was not given | pass | pass | pass |
| Soak time on the channel below | — | — | ≥ 3 days |
| Human review of 10 sampled calls | — | — | required |

Invented details are a **hard zero at every level**, including nightly. That is the whole
reason the preference-training stage exists, and a build that regresses on it should never
reach a channel anyone can resolve.

The regression set is a frozen list of scripted calls with known-correct answers, kept
separate from the training data and never regenerated — if it drifts, the numbers stop being
comparable across releases, which defeats the point.

---

## The registry

One row per build, append-only, never edited:

```json
{
  "version": "2.1-nightly.20260805.a3f9c21",
  "channel": "nightly",
  "created": "2026-08-05T03:14:00Z",
  "lineage": { "model": "v7", "dataset": "ethos-booking-v2" },
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_sha256": "…",
  "dataset_sha256": "…",
  "code_sha": "a3f9c21",
  "training": { "stage": "dpo", "from": "2.0", "hyperparameters": "…" },
  "eval": { "invented_details": 0, "completion_rate": 0.94, "readback_verified": 0.88 },
  "supersedes": "2.1-nightly.20260804.7c1e0b4"
}
```

`adapter_sha256` and `dataset_sha256` are what make a claim checkable later. The generation
manifest already records the dataset hash and the hash of the code that produced it, so most
of this is derivable from what the pipeline writes today.

---

## Rolling back

```bash
# stable is a pointer; move it
ethos release promote 2.0 --channel stable --reason "2.1 regressed on readback"
```

Three things make this safe, and all three are properties of the scheme rather than the
tooling:

- The old build still exists byte-for-byte, because builds are immutable.
- Clients pinned to `2.0` were never affected in the first place.
- The bad build stays in the registry with its eval numbers attached, so the regression is
  documented rather than erased.

Never fix a release by retraining under the same number. `2.1` that behaves differently on
Tuesday than it did on Monday makes every bug report unfalsifiable.

---

## Hosting notes

- **Serve adapters, not merged models.** One base model in memory with LoRA adapters swapped
  per version makes running `stable` and `beta` side by side cheap, and makes rollback a
  pointer change rather than a redeploy.
- **Pin the base model exactly.** A base model that silently updates underneath a pinned
  adapter version breaks the reproducibility the whole scheme is built on.
- **Version the system prompt with the model.** The brief format the agent is trained against
  is part of its contract; shipping a new prompt against an old adapter is a behaviour change
  with no version number on it.
- **Log the resolved version on every call.** When someone reports a bad booking, the first
  question is which build took it, and that has to be answerable without guessing.
