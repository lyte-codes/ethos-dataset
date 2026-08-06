# Running it on a real call

A spec for the thing that stands between a trained adapter and a business's phone ringing.
The model is one component of five, and it is not the one that will cause you the most
trouble.

---

## Shape

```
        brief ─────────────┐
                           ▼
  telephony ──► ASR ──► the agent ──► guard ──► TTS ──► telephony
   (audio in)  (text)   (next turn)  (check)  (audio)   (audio out)
                           │            │
                           └── transcript + audit ──► corrections store
```

One turn is: caller speech arrives, is transcribed, the model produces a reply, the guard
checks that reply against the brief, and only then is it spoken. The guard is the part that
does not exist in a normal voice-agent stack and is the reason this one can be trusted with
somebody's phone number.

---

## 1. Telephony

You need a provider that will place an outbound PSTN call and give you a bidirectional audio
stream — Twilio Media Streams, Telnyx, Vonage, or a SIP trunk with your own media server.

What matters when choosing:

| Requirement | Why |
|---|---|
| Bidirectional streaming, not record-then-respond | Turn-taking is impossible if you only get audio after the caller stops |
| Raw PCM or μ-law frames | Anything that hands you an encoded file per utterance has already cost you a second |
| DTMF events | Phone menus are the first thing a real business will answer with |
| Call recording, separately consented | You need the audio to improve, and you need permission to keep it |

Telephone audio is **8kHz μ-law**, not the 22kHz your TTS produces. Resampling in both
directions is a real step, and doing it badly is the most common cause of an agent that
sounds worse on a call than it did in testing.

## 2. Speech to text

Whisper (`small` or `medium`) locally, or a streaming API. The distinction that matters is
**streaming versus batch**: a batch model cannot tell you the caller has stopped talking, so
you end up guessing with a silence timer and either interrupting people or leaving dead air.

For this domain specifically, ASR errors concentrate exactly where the stakes are. Phone
numbers, postcodes and unusual surnames are where transcription fails, and they are the
details the whole system exists to get right. Two mitigations:

- **Bias the decoder** toward the brief's contents where the API allows it. You know the
  client's name and number before the call starts.
- **Never trust a transcribed number.** If the business reads a number back and ASR renders
  it differently from the brief, the disagreement is at least as likely to be ASR as the
  business. Ask them to repeat rather than correcting them.

## 3. The agent

The model, given exactly the message shape it was trained on:

```python
messages = [
    {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(
        agent_name="Ethos", business_name=..., business_context=...,
        client_name=brief.name, client_phone=brief.phone,
        service=brief.service, availability=brief.availability)},
    {"role": "user",      "content": "Good morning, Northgate Services."},
    {"role": "assistant", "content": "Morning. I'm Ethos, calling on behalf of..."},
    {"role": "user",      "content": <what the business just said>},
]
```

`user` is the business. `assistant` is the agent. The brief goes in the system message and
nowhere else. Get this wrong and the model is being asked to do a task it was never trained
on, and it will fail in ways that look like a bad model rather than a bad prompt.

Generation settings: `max_new_tokens=110` is enough — a spoken turn is short, and a model
allowed to ramble will. Temperature 0.3–0.7; lower is duller but likelier to stick to the
brief.

## 4. The guard

**This is the part worth building carefully.** Before anything is spoken, the candidate turn
is checked against the brief, reusing the rules that already gate the training data:

```python
from check_quality import conversation_problems

problems = conversation_problems(
    turns_so_far + [("assistant", candidate)],
    intent, allowed_words, brief_digits=brief.phone, client_name=brief.name)
```

If it flags a problem — a phone number that is not the client's, a name nobody gave, a price
commitment, the agent claiming to *be* the client — regenerate. Two or three attempts, then
fall back to a fixed safe line:

> "Sorry, could you give me a moment? I'll need to check that with my client and call you
> back."

Ending a call politely is always available and never wrong. Saying a stranger's phone number
to a business is not recoverable.

The same failure the preference training targets is caught here deterministically. Training
lowers the rate; the guard bounds the damage. **Do not rely on the model alone for this** —
a rate of one invented number in fifty calls is fine as a training metric and unacceptable as
a production behaviour.

## 5. Text to speech

Kokoro (`af_heart`) already sounds fine and runs locally. The thing to get right is not
voice quality but **when it starts speaking**. Begin synthesising as the first tokens arrive
rather than waiting for the full turn, or you add the model's entire generation time to the
silence before the agent replies.

Numbers need saying digit by digit — "oh seven seven double-oh, nine hundred, four one two",
not "seven trillion". Do it in the text before it reaches TTS.

---

## Latency

The budget that decides whether it feels like a conversation or a bad video call:

| Stage | Budget | Notes |
|---|---|---|
| Silence detection | 300–500ms | Below this you interrupt people |
| ASR final transcript | 100–300ms | Streaming, on top of the silence wait |
| Model | 400–900ms | ~30 tokens at 30–70 tok/s |
| Guard | <10ms | Regex over one turn |
| TTS first audio | 150–400ms | If streaming; a full turn is much worse |
| **Total to first sound** | **≈1.0–2.0s** | |

Two seconds of silence is a long time on a phone. Two levers, and the second is better than
the first:

- **Speed up the model.** Quantise it, batch nothing, keep the KV cache warm across turns.
- **Fill the gap honestly.** A short "let me see…" while generating is what a person does,
  and it buys the entire model budget. Do not overuse it — an agent that says "um" before
  every sentence is worse than one that pauses.

A regeneration after a guard rejection costs a full model pass. Budget for one.

---

## What has to be built that does not exist yet

Everything in `training/` is offline. The runtime is new code:

1. **Call session** — holds the brief, the message list, and the turn state for one call.
2. **Audio bridge** — telephony frames to ASR, TTS to telephony frames, with resampling.
3. **Turn controller** — decides when the business has finished speaking, handles barge-in,
   and handles the case where nobody says anything for ten seconds.
4. **Guard** — `check_quality` at inference time, plus the retry and the safe fallback.
5. **Outcome extractor** — at the end, what got booked. A structured result, not a
   transcript, since the point of the call is a booking.
6. **Recorder** — transcript, audit, and outcome to the corrections store, scrubbed on the
   way in.

`training/fake_call.py` is already a working version of 1, 3 and 5 with the audio replaced by
a text loop. Building the runtime around it, rather than starting fresh, keeps testing honest:
the same harness that measures the model offline becomes the thing you run.

---

## Failure modes to design for now

These are not edge cases. They are the first five real calls.

| What happens | What the agent must do |
|---|---|
| A phone menu answers | Press a key, or hang up and flag for a human. Never talk to a robot for four minutes |
| Nobody picks up | Hang up after a set number of rings, retry later, do not call back four times in an hour |
| "Can I take your number and call you back?" | Give the client's number, and record that a callback is expected |
| The business asks something the brief does not cover | Say so and offer to check. Never guess — this is the trained behaviour and the guard backs it up |
| ASR mangles the business's slot offer | Ask them to repeat. Confirming a time you misheard is worse than sounding slow |
| The business is confused about who is calling | Restate plainly: an agent, on behalf of a named client |
| It is going badly | End the call politely and hand back to a human. Have this path built before the first real call, not after |

---

## Before it dials a real business

- **Consent and recording law.** Whether you may record, and whether you must announce it,
  varies by jurisdiction and by which end of the call you are on. Settle this before the
  first call, not after the first complaint.
- **Disclosure.** An automated system calling a business should say so. Beyond the ethics,
  an agent that gets caught pretending to be a person has damaged the client it was calling
  for.
- **A hard stop.** One key, one endpoint, that stops all calls immediately.
- **Rate limits per business.** A bug that redials in a loop is not a bug to that business,
  it is harassment.
- **A real phone number that is yours.** Not a client's, so a callback reaches you.

---

## The smallest thing that proves it works

Do not build all six components first. Build this:

1. A hardcoded brief, a fake business played by a script, and text instead of audio — this
   already exists as `fake_call.py`.
2. Swap the scripted business for a real phone call to **your own phone**, with you playing
   the business. Now the audio path is real and the risk is zero.
3. Add the guard, and try to make it fail on purpose: read a wrong number back and check that
   it corrects you rather than agreeing.
4. Only then call a real business, once, with someone watching and a finger on the stop.

Step 2 is where the interesting problems are — turn-taking, resampling, latency — and none of
them need a stranger on the other end to find.
