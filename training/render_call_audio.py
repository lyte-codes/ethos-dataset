#!/usr/bin/env python3
"""Turn a fake-call transcript into an MP3 of the call.

Each turn is spoken by macOS `say` in its own voice, written to its own clip,
and the clips are concatenated with a short pause between them so the result
sounds like a call rather than a list of sentences.

    python3 training/render_call_audio.py --calls data/fake_calls.json --index 0
    python3 training/render_call_audio.py --calls data/fake_calls.json --all
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CALLER_VOICE = "Samantha"
ASSISTANT_VOICE = "Daniel"
GAP_SECONDS = 0.45


def require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise SystemExit(f"error: {binary} not found on PATH")
    return path


def speakable(text: str) -> str:
    """`say` reads punctuation runs and markup literally; strip what would be heard wrong."""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # A UK mobile reads better digit by digit than as two large numbers.
    text = re.sub(r"\b(\d{5})\s?(\d{6})\b", lambda m: " ".join(m.group(1) + m.group(2)), text)
    return text


def synthesise(text: str, voice: str, rate: int, destination: Path) -> None:
    # WAV/LEF32 rather than AIFF: `say` rejects a little-endian format in an AIFF
    # container, and matching the silence clips' format keeps concat from resampling.
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", str(destination),
         "--data-format=LEF32@22050", speakable(text)],
        check=True,
    )


def silence(seconds: float, destination: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
         "-t", str(seconds), "-c:a", "pcm_f32le", str(destination)],
        check=True,
    )


def render(call: dict, out_path: Path, caller_voice: str, assistant_voice: str,
           rate: int, gap: float) -> Path:
    require("say")
    require("ffmpeg")

    with tempfile.TemporaryDirectory() as workspace:
        work = Path(workspace)
        clips: list[Path] = []

        pause = work / "pause.wav"
        silence(gap, pause)

        for index, turn in enumerate(call["turns"]):
            if not turn["text"].strip():
                continue
            clip = work / f"{index:03d}.wav"
            voice = caller_voice if turn["role"] == "caller" else assistant_voice
            synthesise(turn["text"], voice, rate, clip)
            clips.append(clip)
            clips.append(pause)

        if not clips:
            raise SystemExit("error: transcript has no speakable turns")

        listing = work / "clips.txt"
        listing.write_text("".join(f"file '{clip.as_posix()}'\n" for clip in clips))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(listing),
             "-ar", "22050", "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "64k",
             str(out_path)],
            check=True,
        )
    return out_path


def transcript_text(call: dict) -> str:
    lines = [f"# Fake call — {call['caller']['name']} ({call['caller']['persona']})",
             f"# assistant: {call.get('assistant_model', 'unknown')}",
             f"# wants: {call['caller']['service']}, {call['caller']['when']}",
             f"# phone:  {call['caller']['phone']}", ""]
    for turn in call["turns"]:
        speaker = "CALLER   " if turn["role"] == "caller" else "ASSISTANT"
        lines.append(f"{speaker}  {turn['text']}")
    if "audit" in call:
        lines += ["", "# audit"]
        for key, value in call["audit"].items():
            lines.append(f"#   {key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", type=Path, default=Path("data/fake_calls.json"))
    parser.add_argument("--index", type=int, default=0, help="which call to render")
    parser.add_argument("--all", action="store_true", help="render every call in the file")
    parser.add_argument("--out-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--caller-voice", default=CALLER_VOICE)
    parser.add_argument("--assistant-voice", default=ASSISTANT_VOICE)
    parser.add_argument("--rate", type=int, default=180, help="words per minute")
    parser.add_argument("--gap", type=float, default=GAP_SECONDS)
    args = parser.parse_args()

    payload = json.loads(args.calls.read_text())
    calls = payload["calls"] if isinstance(payload, dict) else payload
    chosen = list(enumerate(calls)) if args.all else [(args.index, calls[args.index])]

    tag = re.sub(r"[^a-z0-9]+", "-", str(payload.get("assistant_model", "model")).lower()).strip("-")
    for index, call in chosen:
        stem = f"{tag}-call{index + 1}-{call['caller']['name'].split()[-1].lower()}"
        audio = render(call, args.out_dir / f"{stem}.mp3",
                       args.caller_voice, args.assistant_voice, args.rate, args.gap)
        (args.out_dir / f"{stem}.txt").write_text(transcript_text(call))
        size = audio.stat().st_size / 1024
        print(f"{audio}  ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
