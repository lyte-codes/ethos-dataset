#!/usr/bin/env python3
"""Speech to text — component 2 of the call runtime (docs/CALLS.md).

Local Whisper. Batch, not streaming — good enough to prove the pipeline end to end,
not good enough for real turn-taking. Swap for a streaming API before this touches a
real call; see the note in docs/CALLS.md on why batch ASR cannot tell you the caller
has stopped talking.

    python3 training/transcribe.py audio-samples/1.mp3
    python3 training/transcribe.py audio-samples/1.mp3 --model small.en
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import whisper


def load(model_name: str) -> tuple["whisper.Whisper", str]:
    # MPS half-precision through whisper's conv stack has a history of silently wrong
    # output on some torch versions, and the failure mode is a bad transcript rather
    # than an error — the dangerous kind. fp32 costs some speed and is correct.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    return model, device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="medium.en")
    args = parser.parse_args()

    audio_path = args.audio.expanduser()
    if not audio_path.exists():
        parser.error(f"{audio_path} not found")

    model, device = load(args.model)
    print(f"model: {args.model} | device: {device} | fp16: False")

    result = whisper.transcribe(model, str(audio_path), fp16=False)
    print()
    print(result["text"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
