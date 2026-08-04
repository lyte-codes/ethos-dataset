# Ethos

A synthetic dataset of phone-call booking conversations, and the code to fine-tune a small
model on it.

**Ethos** is the dataset: multi-turn transcripts between a caller and a booking assistant,
decomposed into turn-level training examples. **The model** is a LoRA fine-tune of
`Qwen2.5-1.5B-Instruct` that learns to be that assistant — collect the service, the date and
time, and the contact details, then read the booking back and confirm it.

Everything here is reproducible from scratch: the generator, the quality gate that decides
what is allowed into the dataset, the training script, and the exact hyperparameters.

---

## What is in the dataset

Each line of `data/ethos_booking.jsonl` is one training example — a single assistant turn
plus the conversation that led to it:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a phone booking assistant for ..."},
    {"role": "user",      "content": "Hi, I need to book a fitting."},
    {"role": "assistant", "content": "Happy to help. What day works for you?"},
    {"role": "user",      "content": "Sometime next week?"}
  ],
  "response": "Would Tuesday the 12th at 10am suit?",
  "conversation_id": 1,
  "service": "fitting",
  "persona": "rambling",
  "complication": "vague_time",
  "intent": "new_booking"
}
```

One conversation yields one example per assistant turn, so an N-turn call produces several
rows that share a `conversation_id`. The trailing metadata fields are not needed for
training — they exist so you can slice quality by scenario and hold out whole conversations
rather than individual turns.

See [docs/DATASET.md](docs/DATASET.md) for the scenario taxonomy and the quality rules.

## Generating it

Requires [Ollama](https://ollama.com) with a local instruct model:

```bash
ollama pull llama3.1:8b
./generation/smoke.sh 5                       # preflight + 5 conversations + quality report
python3 generation/generate_dataset.py \
    -n 500 -w 10 -o data/ethos_booking.jsonl  # the full run, 10 in parallel
```

The generator samples a caller persona, a complication and a service for each call, asks the
local model for the whole conversation as JSON, then splits it into turn-level examples.
Anything that fails the quality gate is regenerated rather than written.

`--resume` appends to an existing file and skips conversations already in it, so an
interrupted run is recoverable.

## Training on it

```bash
pip install -r requirements.txt
python3 training/train_lora.py --data data/ethos_booking.jsonl
python3 training/infer.py                     # talk to the result
python3 training/infer.py --base-only         # compare against untuned Qwen
```

The loss is computed on assistant turns only — caller text and the system prompt are masked
out. The eval split holds out whole conversations, because splitting mid-conversation would
leak the same call into both sides.

Hyperparameters live in the `Hyperparameters` dataclass at the top of
[`training/train_lora.py`](training/train_lora.py) and are written to
`hyperparameters.json` alongside every checkpoint. See
[docs/TRAINING.md](docs/TRAINING.md) for what each one does and what to change first.

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
docs/
  DATASET.md              schema, scenario taxonomy, quality rules
  TRAINING.md             hyperparameters, hardware notes, what to tune
```

## Why the quality gate exists

Small instruct models generate booking calls with a specific failure mode: the assistant
states details the caller never gave. It greets an unnamed caller as "Mr. Bennett", reads
back a phone number nobody spoke, or confirms "Wednesday the 15th at 12pm" when the caller
rejected every time offered and never named one.

Train on that and the model learns to invent caller identities and appointment times, which
is precisely the behaviour you least want in a booking assistant. `check_quality.py` encodes
these as rules and the generator enforces them before writing, so defects trigger a
regeneration instead of entering the dataset.
