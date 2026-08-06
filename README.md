# Ethos

A synthetic dataset of phone-call booking conversations, and the code to fine-tune a small
model on it.

**Ethos** is the dataset: multi-turn transcripts of an agent phoning a business, decomposed
into turn-level training examples. **The model** is a LoRA fine-tune of
`Qwen2.5-1.5B-Instruct` that learns to be that agent — it is briefed with a client's details,
calls the business, answers whatever the person who picks up asks using only the brief, agrees
a time inside the client's availability, and checks the booking the business reads back.

The agent **places** the call. It is not the business, and it is not the client: it is a third
party acting for the client, and it never speaks as them.

Everything here is reproducible from scratch: the generator, the quality gate that decides
what is allowed into the dataset, the training script, and the exact hyperparameters.

---

## What is in the dataset

Each line of the dataset is one training example — a single agent turn plus the conversation
that led to it. The `user` role is whoever answered the phone at the business; the `assistant`
role is the agent:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a booking agent... Your client's brief...\n- Name: Priya Raman\n- Contact number: 07700 900318\n- Needs: fitting\n- Available: Tuesday or Wednesday morning"},
    {"role": "user",      "content": "Good morning, Northgate Services, how can I help?"},
    {"role": "assistant", "content": "Morning. I'm calling on behalf of Priya Raman to book a fitting."},
    {"role": "user",      "content": "Right, what day were you after?"}
  ],
  "response": "She's free Tuesday or Wednesday morning — would either suit?",
  "conversation_id": 1,
  "service": "fitting",
  "persona": "impatient",
  "complication": "no_availability",
  "intent": "new_booking"
}
```

The brief lives in the system message and differs per conversation, which is what makes the
data checkable: every concrete detail the agent says can be compared against what it was
actually told.

One conversation yields one example per agent turn, so an N-turn call produces several rows
that share a `conversation_id`. The trailing metadata fields are not needed for training —
they exist so you can slice quality by scenario and hold out whole conversations rather than
individual turns.

See [docs/DATASET.md](docs/DATASET.md) for the scenario taxonomy and the quality rules.

## Generating it

Requires [Ollama](https://ollama.com) with a local instruct model:

```bash
ollama pull llama3.2:3b
./generation/smoke.sh 5                          # preflight + 5 conversations + quality report
python3 generation/generate_dataset.py \
    -n 200 -w 3 -o data/ethos_booking_v2.jsonl   # the full run, 3 in parallel
```

The generator samples a client brief, a persona for whoever answers at the business, a
complication and a service for each call, asks the local model for the whole conversation as
JSON, then splits it into turn-level examples. Anything that fails the quality gate is
regenerated rather than written.

Bigger generators produce better dialogue at a steep cost in wall-clock: `llama3.2:3b` runs at
roughly a minute per conversation, `qwen2.5:14b-instruct` at over four.

`--resume` appends to an existing file and skips conversations already in it, so an
interrupted run is recoverable.

## Training on it

```bash
pip install -r requirements.txt
python3 training/train_lora.py --data data/ethos_booking_v2.jsonl --batch-size 1
python3 training/infer.py                     # talk to the result
python3 training/infer.py --base-only         # compare against untuned Qwen
```

The loss is computed on agent turns only — the business's text and the system prompt are
masked out. On Apple silicon, keep the batch size at 1: the defaults will exhaust MPS memory
on a 16GB machine, and in-loop eval is skipped there because the full-vocabulary logits
forward is what tips it over. The eval split holds out whole conversations, because splitting mid-conversation would
leak the same call into both sides.

Hyperparameters live in the `Hyperparameters` dataclass at the top of
[`training/train_lora.py`](training/train_lora.py) and are written to
`hyperparameters.json` alongside every checkpoint. See
[docs/TRAINING.md](docs/TRAINING.md) for what each one does and what to change first.

## Versions

Model v1 is the stock `Qwen2.5-1.5B-Instruct` with no fine-tuning — the baseline. Model and
dataset versions are tracked separately and pinned to each other in [VERSIONS.md](VERSIONS.md);
the rules for bumping either are in [docs/VERSIONING.md](docs/VERSIONING.md).

Those are internal counters. What gets **served** is versioned separately, as MAJOR.MINOR over
nightly/beta/stable channels — see [docs/RELEASES.md](docs/RELEASES.md).

Dataset v1 and the model v2 checkpoints are from an earlier design in which the assistant was
the business's receptionist rather than the agent calling it. They are kept for reference and
are not a base for anything.

Every generation run writes a manifest recording the generator model, temperature, seed,
scenario distribution, and SHA-256 hashes of both the data and the code that produced it.

## Layout

```
data/                     the dataset
generation/
  generate_dataset.py     conversation generator + turn-level decomposition
  check_quality.py        quality rules, used as a gate and as a standalone linter
  smoke.sh                preflight checks + small run + coverage report
training/
  train_lora.py           LoRA fine-tune
  infer.py                interactive check of a trained adapter
  train_dpo.py            preference training against invented details
  fake_call.py            run a whole fake call against a model, end to end
  render_call_audio.py    render a call transcript to MP3
  progress_server.py      live dashboard for a running training job
docs/
  DATASET.md              schema, scenario taxonomy, quality rules
  TRAINING.md             hyperparameters, hardware notes, what to tune
  VERSIONING.md           internal dataset/model counters
  RELEASES.md             what gets shipped, and how it is promoted
  CALLS.md                what a real phone call needs beyond the model
```

## Why the quality gate exists

Small instruct models generate these calls with a specific failure mode: the agent says
things its brief never contained. It gives a phone number a digit off, invents an access code
when the business asks for one, commits the client to "about forty pounds", or answers the
question rather than saying it will check.

There is a second failure that is easy to miss because the transcript still reads well — the
agent introducing itself as "this is Priya Raman" instead of calling *on behalf of* her. That
collapses the three-party structure the whole dataset is built on, and it slipped past an
earlier version of the gate.

Train on either and the model learns to speak for a client it does not actually know, which is
the behaviour you least want in something that phones real businesses. `check_quality.py`
encodes these as rules and the generator enforces them before writing, so defects trigger a
regeneration instead of entering the dataset.
