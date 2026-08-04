# The Ethos dataset

## Schema

One JSON object per line. `messages` and `response` are what training consumes; the rest is
metadata for slicing and splitting.

| Field | Type | Meaning |
|---|---|---|
| `messages` | list | System prompt, then alternating caller (`user`) and assistant turns, ending with the caller turn being responded to |
| `response` | string | The assistant turn to learn |
| `conversation_id` | int | Groups every example that came from the same call |
| `service` | string | The service being booked |
| `persona` | string | Caller style for the call |
| `complication` | string | What makes the call non-trivial |
| `intent` | string | `new_booking`, `reschedule` or `cancellation` |

The system prompt is identical across every example and is prepended at generation time:

> You are a phone booking assistant for {business}. {context}
> Be concise, confirm details clearly, and ask one question at a time. Collect the service
> type, the date and time, the duration when it is relevant, and the caller's name and phone
> number. Read the booking back to the caller before confirming it.

Change the business with `--business-name` and `--business-context`; the dataset is otherwise
domain-generic.

## Scenario taxonomy

Each conversation samples one value from each axis independently.

**Personas** — `terse`, `rambling`, `polite`, `impatient`, `confused`. These shape the caller
only; the assistant stays professional regardless.

**Complications** — the thing that makes the call worth training on:

| Complication | What the assistant has to handle |
|---|---|
| `vague_time` | "Sometime next week" narrowed to a specific day and time |
| `change_of_mind` | Caller changes the day, time or service mid-call |
| `asks_before_committing` | Questions about availability, duration or price before booking |
| `corrects_info` | A detail given, then corrected several turns later |
| `multiple_bookings` | Two separate bookings in one call, kept distinct |
| `mishearing` | Background noise; assistant asks for a repeat or a spelling |

**Intents** — sampled at roughly 65% `new_booking`, 20% `reschedule`, 15% `cancellation`.

**Services** — a configurable list at the top of `generate_dataset.py`
(consultation, repair, fitting, inspection, installation, follow-up appointment, assessment).

## Quality rules

`check_quality.py` runs both as a gate inside the generator and as a standalone linter over a
finished file:

```bash
python3 generation/check_quality.py data/ethos_booking.jsonl "Northgate Services"
```

The rules exist because a small generator model hallucinates caller-side detail. Each one was
added in response to an observed defect:

| Rule | Defect it prevents |
|---|---|
| No capitalized name or honorific before the caller says it | "Thank you, Mr. Bennett" to an unnamed caller |
| No digit run the caller never spoke | Reading back an invented phone number |
| Readback times and weekdays must have been discussed | Booking "Wednesday at 12pm" when neither was ever mentioned |
| Phone number collected on new bookings | Call confirmed with no contact details |
| A clock time appears, and near the end of the call | Call that never actually settles a time |
| No placeholders (`[name]`, `XXX`) | Template text leaking into training data |

Minimum turn counts are scenario-aware — a terse cancellation is legitimately four turns,
while a rambling new booking should not be.

Rules are deliberately conservative about what counts as a violation. A false positive costs
one regeneration; a false negative puts a hallucination into the training data.

## Known limitations

- **Single generator model.** Every conversation comes from one model, so its stylistic tics
  are baked in. Generating across several models and mixing would give more variety.
- **No real ASR noise.** The `mishearing` complication is written as if there were background
  noise, but the text is clean. Real speech-to-text errors look different.
- **Dates are internally consistent only.** "Tuesday the 12th" need not be a real Tuesday.
  If your assistant must reason about a calendar, generate against real dates instead.
- **The assistant is always competent.** There are no examples of the assistant failing,
  escalating, or refusing — so the model has nothing to learn from for those cases.
- **English, one business at a time.** Regenerate per business or per locale as needed.
