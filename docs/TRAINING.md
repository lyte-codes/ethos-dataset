# Training the Ethos model

A LoRA adapter over `Qwen2.5-1.5B-Instruct`. The base model is already instruction-tuned and
already speaks in turns; the fine-tune teaches it one specific job — run a booking call to
completion without inventing anything the caller did not say.

> **Status:** the configuration below is the reference setup this repo ships with, and is
> what `train_lora.py` runs by default. Measured results — loss curves, eval numbers,
> before/after comparisons — are not filled in yet, because the first full training run has
> not been done. Record them in [Results](#results) once you have them rather than trusting
> the defaults to be optimal.

## Setup

```bash
pip install -r requirements.txt
python3 training/train_lora.py --data data/ethos_booking.jsonl
```

The adapter and a `hyperparameters.json` recording the exact settings land in
`checkpoints/ethos-qwen-lora/`.

## Hyperparameters

Defined in the `Hyperparameters` dataclass at the top of `training/train_lora.py`.

| Parameter | Default | Why |
|---|---|---|
| `lora_rank` | 16 | Enough capacity for a single task with a few thousand examples. Rank 8 is usually fine too; past 32 you are mostly adding parameters that overfit. |
| `lora_alpha` | 32 | Held at 2x rank, the usual convention — effective scale is `alpha / rank`. |
| `lora_dropout` | 0.05 | Light regularization. Raise toward 0.1 if eval loss climbs while train loss falls. |
| `target_modules` | attention + MLP | All seven projections. Attention-only (`q,k,v,o`) trains faster and fits smaller cards, at some cost in instruction fidelity. |
| `learning_rate` | 2e-4 | Standard for LoRA — roughly 10x what you would use for a full fine-tune, because only the adapter moves. |
| `lr_scheduler` | cosine | With 3% warmup, so the first steps do not wreck the adapter. |
| `epochs` | 3 | Enough to learn the format without memorizing. Watch eval loss; synthetic data is repetitive and overfits readily. |
| `per_device_batch_size` | 4 | Tuned for a 1024-token window on modest hardware. |
| `gradient_accumulation_steps` | 4 | Effective batch of 16. Keep the product constant if you change either. |
| `max_sequence_length` | 1024 | Comfortably fits the longest examples — a late assistant turn carries the whole call as history. |
| `weight_decay` | 0.01 | Mild. |

### Loss masking

Prompt tokens are set to `-100` so the loss only covers the assistant response. Without this
the model spends capacity learning to imitate callers, which is not the job. This is the
single most important detail in the training script.

### Splitting

`split_by_conversation` holds out whole conversations. Turn-level examples from one call share
almost all their context, so a naive random split puts near-identical rows on both sides and
produces an eval loss that flatters the model.

## Hardware

| Setup | Notes |
|---|---|
| CUDA, 16GB+ | Comfortable. `bf16=True` is enabled automatically. |
| Apple Silicon (MPS) | Works and is what this repo was developed on, but slow for training. bf16 is disabled on MPS; expect fp32 speeds and keep the batch size low. |
| CPU | Only for verifying the script runs. |
| Colab T4 | A practical free option. Add 4-bit quantization (`bitsandbytes`) if memory is tight. |

A 1.5B model with a rank-16 adapter trains only a few million parameters, so this is a small
job by fine-tuning standards — the dataset size, not the model, is the limiting factor.

## Evaluating

```bash
python3 training/infer.py                 # talk to the adapter
python3 training/infer.py --base-only     # same prompts against untuned Qwen
```

Run the same handful of calls through both. What the fine-tune should improve:

- asking one question at a time instead of demanding every detail at once
- reading the booking back before confirming
- staying on task when the caller rambles or changes their mind
- **not inventing names, numbers or appointment times** — the failure this dataset is built
  to suppress

Eval loss alone will not tell you whether the last point improved. Check it by hand, or run
`check_quality.py`'s rules over generated conversations.

## Penalising invented details (model v3)

Supervised training rewards the right answer but never marks a wrong one as wrong. If the
model states a phone number the caller never gave, SFT simply does not reinforce it — there
is no gradient pushing against it. Preference training supplies that gradient.

```bash
python3 generation/build_preference_pairs.py \
    --data data/ethos_booking_v1.jsonl --out data/ethos_preferences_v1.jsonl
python3 training/train_dpo.py \
    --data data/ethos_preferences_v1.jsonl --sft-adapter checkpoints/ethos-qwen-lora
```

Pairs are built by corrupting a clean response: same conversation, same wording, only the
detail changed. The model sees a phone number copied correctly and the same sentence with
different digits, and learns which one the context supports.

| Parameter | Default | Why |
|---|---|---|
| `beta` | 0.1 | How far the model may drift from the SFT model. Lower punishes harder but risks degrading fluency; raise toward 0.3 if responses get terse or strange. |
| `learning_rate` | 5e-6 | Roughly 40x lower than the SFT rate. DPO destabilises quickly at SFT learning rates. |
| `epochs` | 1 | Preference data overfits fast. Watch the reward margin rather than adding epochs. |

`ref_model=None` is deliberate: with a PEFT model, `DPOTrainer` uses the adapter-disabled
model as the reference, so a second copy is never loaded.

### Checking it worked

Reward accuracy above ~0.9 in the training logs means the model reliably prefers the correct
detail. That is necessary but not sufficient — confirm on real generations:

```bash
python3 training/infer.py --adapter checkpoints/ethos-qwen-dpo
```

Give a phone number mid-call and see whether the readback reproduces it exactly. Then run the
same call against v2 and v1. If v2 already copies numbers faithfully, v3 is not buying you
anything, and that is worth knowing before you keep it.

## Results

Not yet measured. Fill in after the first run:

| Run | Examples | Epochs | Final train loss | Final eval loss | Notes |
|---|---|---|---|---|---|
| | | | | | |

## What to change first

1. **More data** before more epochs. Synthetic booking calls are repetitive; 500 conversations
   is a starting point, not a target.
2. **Epochs down to 2** if eval loss turns up after epoch 2.
3. **Rank down to 8** if the model becomes rigid and parrots dataset phrasings verbatim.
4. **Attention-only target modules** if you are memory constrained.
