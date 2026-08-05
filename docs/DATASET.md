# The Ethos dataset

Transcripts of an **agent phoning a business on a client's behalf**, decomposed into
turn-level training examples. The assistant is the party placing the call. It is not the
business, and it is not the client — it is a third party acting for them.

## Schema

One JSON object per line. `messages` and `response` are what training consumes; the rest is
metadata for slicing and splitting.

| Field | Type | Meaning |
|---|---|---|
| `messages` | list | System prompt carrying the client brief, then alternating business (`user`) and agent turns, ending with the business turn being responded to |
| `response` | string | The agent turn to learn |
| `conversation_id` | int | Groups every example that came from the same call |
| `service` | string | What the client needs booked |
| `persona` | string | Style of whoever answered the phone at the business |
| `complication` | string | What makes the call non-trivial |
| `intent` | string | `new_booking`, `reschedule` or `cancellation` |

Note that `user` is the **business**, not the client. The client is never on the call.

## The brief

Unlike a receptionist dataset, the system prompt is **not** identical across examples. Each
conversation carries its own client brief:

> You are a booking agent. You are phoning {business} on behalf of your client to arrange an
> appointment. {context}
>
> Your client's brief — these are the only facts you have:
> - Name: Priya Raman
> - Contact number: 07700 900318
> - Needs: fitting
> - Available: Tuesday or Wednesday morning
>
> …never state a name, number, date or time that is not in the brief. If the business asks for
> anything the brief does not cover, say you will check with your client rather than guessing.
> When the business reads the booking back, check every detail against the brief and correct
> anything they have wrong.

This is the part that makes the data checkable. Every concrete detail the agent utters can be
compared against what it was actually told, so "did it invent this?" is a mechanical question
rather than a judgement call. The manifest records the template; the filled-in brief lives in
each example's system message.

Contact numbers are drawn from `07700 900000–900999`, the range Ofcom reserves for drama, so
no generated number can reach a real person.

Change the business with `--business-name` and `--business-context`; the dataset is otherwise
domain-generic.

## Scenario taxonomy

Each conversation samples one value from each axis independently.

**Personas** — `terse`, `rambling`, `polite`, `impatient`, `confused`. These shape whoever
picks up at the business; the agent stays professional regardless.

**Complications** — the thing that makes the call worth training on. All six are situations
that only exist when you are the one calling:

| Complication | What the agent has to handle |
|---|---|
| `no_availability` | First choice is not free; pick an alternative that still fits the client's availability |
| `asks_beyond_brief` | Business asks something the brief does not cover — say you will check, do not guess |
| `wrong_readback` | Business gets a detail wrong when confirming; notice and correct it |
| `transfers_the_call` | Passed to someone else; restate the request without drift |
| `asks_about_cost` | Cost raised; do not commit the client to a price |
| `mishearing` | Background noise; repeat or spell a detail using only the brief |

**Intents** — sampled at roughly 65% `new_booking`, 20% `reschedule`, 15% `cancellation`.

**Services** — a configurable list at the top of `generate_dataset.py`
(consultation, repair, fitting, inspection, installation, follow-up appointment, assessment).

## Quality rules

`check_quality.py` runs both as a gate inside the generator and as a standalone linter over a
finished file:

```bash
python3 generation/check_quality.py data/ethos_booking_v2.jsonl "Northgate Services"
```

The rules exist because a small generator model puts words in the agent's mouth that its
brief never contained. Each was added in response to an observed defect:

| Rule | Defect it prevents |
|---|---|
| No capitalized name outside the brief | Inventing a spouse, a colleague, a second contact |
| No digit run the brief does not contain | Giving the business a phone number a digit off |
| The agent never speaks as its client | "This is Priya Raman calling" instead of "on behalf of Priya Raman" |
| The call names the client early | A call indistinguishable from the agent booking for itself |
| Contact number actually given on new bookings | Booking made with no way to reach the client |
| No price commitment, digits or words | "About forty pounds" commits the client just as "£40" does |
| Business reads a time back, and the agent confirms one | Call that never actually settles a time |
| No placeholders (`[name]`, `XXX`) | Template text leaking into training data |

The impersonation rule is the one worth dwelling on. A transcript where the agent says "this
is Priya Raman" reads perfectly well — it is fluent, polite and on-topic — and an earlier
version of the gate passed it. But it collapses the three-party structure the dataset exists
to teach, and a model trained on it will confidently claim to *be* whoever it is calling for.

Minimum turn counts are scenario-aware — a terse cancellation is legitimately four turns,
while a rambling new booking should not be.

Rules are deliberately conservative about what counts as a violation. A false positive costs
one regeneration; a false negative puts a fabrication into the training data.

## Known limitations

- **Single generator model.** Every conversation comes from one model, so its stylistic tics
  are baked in. Generating across several models and mixing would give more variety.
- **The business always eventually cooperates.** No call ends with the business refusing,
  never answering, or putting the agent on hold indefinitely — so the model has no examples of
  giving up and reporting back to the client.
- **No real ASR noise.** The `mishearing` complication is written as if there were background
  noise, but the text is clean. Real speech-to-text errors look different.
- **Dates are internally consistent only.** "Tuesday the 12th" need not be a real Tuesday.
  If your agent must reason about a calendar, generate against real dates instead.
- **The agent is always competent.** There are no examples of it failing, escalating, or
  refusing, so the model has nothing to learn from for those cases.
- **The client is never reachable mid-call.** The brief is fixed for the whole conversation,
  so "I'll check with my client" is always a promise and never a resolution.
- **English, one business at a time.** Regenerate per business or per locale as needed.
