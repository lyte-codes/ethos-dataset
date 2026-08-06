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
release        2.1-nightly.20260805b                           immutable, published
                                              |
channel        nightly -> beta -> stable                         mutable pointers
```

**Builds are immutable.** Once `2.1-nightly.20260805b` exists it is never rebuilt or
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
2.1-nightly.20260805a   <   2.1-nightly.20260805b   <   2.1-rc.3   <   2.1
```

- **nightly** — `<next-version>-nightly.<YYYYMMDD><letter>`, where the letter counts builds that night from `a`.
- **rc** — `<next-version>-rc.<n>`. A nightly that cleared the full eval suite and is soaking.
- **stable** — `<major>.<minor>`. Promoted from an rc. Never built directly.

### The date is the night, not the clock

A full run crosses midnight — the supervised stages finish in the evening and the preference
stages land in the small hours. The date is pinned when the run starts and every build from
that run carries it, so `20260805a` through `20260805d` stay one batch.

Letting the clock decide instead would split a single night's builds across two dates and
reset the letter to `a` partway through, so the third build of a run would read as the first
build of a new night. Ordering survives either way — `20260805b` sorts before `20260806a` —
but the grouping is the point: those four builds only mean anything compared against each
other, and their names should say so.

### Why a letter

The per-night letter is not decoration. More than one build a day is the normal case, not the
exception — a supervised run and the preference run stacked on top of it finish hours apart on
the same date, and neither needs a code change between them. Keying on the date alone, or on
the date plus the commit, would give both the same identifier and two different sets of weights
would answer to one name. Alphanumeric prerelease identifiers compare lexically, so `a` before
`b` before `c` orders the day's builds correctly, which a git sha never could. Past `z` it
continues `aa`, `ab`, the way spreadsheet columns do — though twenty-six builds in a day is far
beyond what one machine can train.

The commit and the training stage are recorded in the registry rather than the version string.
They are worth knowing, but they are not what makes a build unique, and a version number that
carries every useful fact stops being a name and becomes a description.

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

## Cadence

Build on **change**, with one scheduled full run overnight as the floor. Nothing is gained by
rebuilding on a clock when the inputs have not moved — twenty-four models a day off an
unchanged dataset differ only by sampling noise, and every one of them is another candidate to
bisect through when a regression appears.

More than one build a day is normal on an active day, but they should not all be full builds.
The stages cost very different amounts, so a build only reruns from the earliest stage the
change actually touched:

| Change | Cheapest stage that covers it | Cost |
|---|---|---|
| Generator prompt, quality rules, scenario taxonomy | regenerate the dataset, then everything after | ~6h |
| Dataset version, LoRA rank, learning rate, epochs | supervised training, then preference training | ~4h |
| Preference pairs, DPO beta | preference training on the existing adapter | ~1–2h |
| Eval set, a metric definition, a threshold | evaluate the existing build | minutes |

That is what makes several builds a day affordable: one full run overnight, and anything
triggered during the day scoped to the stage that changed. Re-running preference training on an
adapter that already exists publishes in an hour or two rather than six.

An eval-only run is cheap enough to do freely, and worth doing whenever a threshold moves — it
re-scores existing builds against the new bar without producing a new artifact at all.

## What a full run does

A full build is not just "train again" — it is the whole chain, and any step failing means no
build is published. Publishing nothing is a normal outcome and better than publishing something
unevaluated.

1. **Generate** the day's delta data with the pinned generator, appending a new dataset
   version if the generator or quality rules changed.
2. **Train** from the current base, writing `hyperparameters.json` and the dataset hash into
   the checkpoint.
3. **Evaluate** against the regression set (below). Fail the run on any hard gate.
4. **Publish** as `…-nightly.<date><letter>` and repoint the `nightly` channel.
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
| No regression against the current `stable` on any blocking metric | — | required | required |
| Booking completed within the turn budget | ≥ 80% | ≥ 90% | ≥ 90% |
| Business's readback correctly verified or corrected | ≥ 70% | ≥ 85% | ≥ 85% |
| Never quotes a price it was not given | pass | pass | pass |
| Soak time on the channel below | — | — | ≥ 3 days |
| Human review of 10 sampled calls | — | — | required |

Invented details are a **hard zero at every level**, including nightly. That is the whole
reason the preference-training stage exists, and a build that regresses on it should never
reach a channel anyone can resolve.

### Two ways to fail

The table above is all absolute floors, and floors alone are not enough. A candidate can clear
every one of them and still be worse than the build it would replace — shipping it is a
downgrade for everyone already on `stable`, whatever the numbers say in isolation.

So a build fails if **either** is true:

1. **It misses a floor.** Non-negotiable, and independent of history. Without floors a series of
   releases each "no worse than the last" can drift a long way down, one imperceptible step at a
   time, with nothing ever failing.
2. **It regresses against the current `stable`.** Measured on the same frozen call set, a
   candidate must not be worse on any blocking metric. Better on one axis does not buy a
   regression on another — a build that completes more bookings but invents more details is not
   a trade worth making, it is the failure mode wearing a disguise.

The relative check needs a baseline, so it does not apply to the first release of a line. `0.1`
has no predecessor and is judged on floors alone. The same is true immediately after a MAJOR
bump: the agent's behaviour deliberately changed, so comparing it to the previous major would
be measuring the intended change rather than a regression. Re-baseline at the new `x.0` and
compare within the line after that.

### Why the comparison has to be paired

Both checks are worthless if the numbers move on their own. The regression set is a frozen list
of scripted calls with known-correct answers, kept out of the training data and never
regenerated — if it drifts, results stop being comparable across releases and every
"regression" is arguable.

Evaluation runs at temperature 0 for the same reason. Sampling noise on a few dozen calls is
easily larger than the difference you are trying to detect, and a gate that fires on noise gets
ignored within a week. Same calls, same decoding, so a difference between two builds is the
builds differing.

---

## The registry

One row per build, append-only, never edited:

```json
{
  "version": "2.1-nightly.20260805b",
  "channel": "nightly",
  "created": "2026-08-05T03:14:00Z",
  "lineage": { "model": "v7", "dataset": "ethos-booking-v2" },
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_sha256": "…",
  "dataset_sha256": "…",
  "code_sha": "a3f9c21",
  "training": { "stage": "dpo", "from": "2.1-nightly.20260805a", "hyperparameters": "…" },
  "eval": { "invented_details": 0, "completion_rate": 0.94, "readback_verified": 0.88 },
  "supersedes": "2.1-nightly.20260805a"
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
- **Publish the adapter, not the checkpoint.** A training checkpoint is roughly 227MB, but
  two thirds of that is optimizer state that exists only so training can resume. The adapter
  itself is around 70MB, and it is the only part a served build needs. Keeping whole
  checkpoints makes retention look three times more expensive than it is.
- **Pin the base model exactly.** A base model that silently updates underneath a pinned
  adapter version breaks the reproducibility the whole scheme is built on.
- **Version the system prompt with the model.** The brief format the agent is trained against
  is part of its contract; shipping a new prompt against an old adapter is a behaviour change
  with no version number on it.
- **Log the resolved version on every call.** When someone reports a bad booking, the first
  question is which build took it, and that has to be answerable without guessing.
