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
import sys
import tempfile
from pathlib import Path

BUSINESS_VOICE = "Daniel"     # whoever picks up at the business — macOS `say`
GAP_SECONDS = 0.45

# The agent is the voice a real customer would hear, so it gets a neural model rather than
# a `say` voice. The business is only ever a test harness, so a compact voice is fine there
# and keeps the two sides easy to tell apart.
#
# Three tiers, tried in order: Kokoro, then Piper, then `say`. Rendering should degrade rather
# than fail on a machine that has not had the models downloaded.
KOKORO_DIR = Path.home() / ".local/share/kokoro"
KOKORO_VOICE = "af_heart"
PIPER_VOICE = Path.home() / ".local/share/piper-voices/en_GB-jenny_dioco-medium.onnx"
AGENT_FALLBACK_VOICE = "Samantha"


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


class KokoroVoice:
    """Lazily loaded so a render that never reaches an agent turn pays nothing for it."""

    def __init__(self, directory: Path, voice: str):
        self.directory, self.voice, self._engine = directory, voice, None

    def available(self) -> bool:
        model = self.directory / "kokoro-v1.0.onnx"
        voices = self.directory / "voices-v1.0.bin"
        if not (model.exists() and voices.exists()):
            return False
        try:
            import kokoro_onnx  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError:
            return False
        return True

    def engine(self):
        if self._engine is None:
            from kokoro_onnx import Kokoro

            self._engine = Kokoro(str(self.directory / "kokoro-v1.0.onnx"),
                                  str(self.directory / "voices-v1.0.bin"))
        return self._engine

    def synthesise(self, text: str, destination: Path) -> None:
        import soundfile

        # "af_" voices are American, "bf_"/"bm_" British; the wrong tag mangles the vowels.
        language = "en-gb" if self.voice.startswith(("bf_", "bm_")) else "en-us"
        samples, rate = self.engine().create(speakable(text), voice=self.voice,
                                             speed=1.0, lang=language)
        soundfile.write(str(destination), samples, rate)


def synthesise_piper(text: str, model: Path, destination: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", str(model), "-f", str(destination)],
        input=speakable(text).encode(), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def piper_available(model: Path) -> bool:
    if not model.exists():
        return False
    probe = subprocess.run([sys.executable, "-c", "import piper"], capture_output=True)
    return probe.returncode == 0


def silence(seconds: float, destination: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
         "-t", str(seconds), "-c:a", "pcm_f32le", str(destination)],
        check=True,
    )


def render(call: dict, out_path: Path, business_voice: str, agent_voice: str,
           rate: int, gap: float, piper_model: Path | None = None,
           kokoro: "KokoroVoice | None" = None) -> Path:
    require("say")
    require("ffmpeg")

    use_kokoro = kokoro is not None and kokoro.available()
    use_piper = not use_kokoro and piper_model is not None and piper_available(piper_model)
    if kokoro is not None and not use_kokoro:
        print(f"note: kokoro unavailable in {kokoro.directory}, falling back for the agent voice",
              file=sys.stderr)
    if not use_kokoro and not use_piper:
        print("note: no neural voice available, using `say` for the agent", file=sys.stderr)

    with tempfile.TemporaryDirectory() as workspace:
        work = Path(workspace)
        clips: list[Path] = []

        pause = work / "pause.wav"
        silence(gap, pause)

        for index, turn in enumerate(call["turns"]):
            if not turn["text"].strip():
                continue
            clip = work / f"{index:03d}.wav"
            if turn["role"] == "business":
                synthesise(turn["text"], business_voice, rate, clip)
            elif use_kokoro or use_piper:
                raw = work / f"{index:03d}-neural.wav"
                if use_kokoro:
                    kokoro.synthesise(turn["text"], raw)
                else:
                    synthesise_piper(turn["text"], piper_model, raw)
                # Neural output is 22.05k/24k mono int16 or float; the concat list has to be
                # uniform or ffmpeg silently takes the first stream's format for all of them.
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(raw),
                     "-ar", "22050", "-ac", "1", "-c:a", "pcm_f32le", str(clip)],
                    check=True,
                )
            else:
                synthesise(turn["text"], agent_voice, rate, clip)
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
    brief = call["brief"]
    lines = [f"# Fake outbound call — agent booking for {brief['client_name']}",
             f"# agent model: {call.get('agent_model', 'unknown')}",
             f"# brief: {brief['service']}, available {brief['availability']}",
             f"# client number: {brief['client_phone']}",
             f"# business persona: {call.get('persona', '?')}", ""]
    for turn in call["turns"]:
        speaker = "BUSINESS " if turn["role"] == "business" else "AGENT    "
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
    parser.add_argument("--business-voice", default=BUSINESS_VOICE)
    parser.add_argument("--agent-voice", default=AGENT_FALLBACK_VOICE,
                        help="`say` voice used for the agent only if piper is unavailable")
    parser.add_argument("--kokoro-dir", type=Path, default=KOKORO_DIR)
    parser.add_argument("--kokoro-voice", default=KOKORO_VOICE,
                        help="af_/am_ are American, bf_/bm_ British")
    parser.add_argument("--piper-model", type=Path, default=PIPER_VOICE,
                        help="used only if kokoro is unavailable")
    parser.add_argument("--no-neural", action="store_true",
                        help="force the `say` voice for the agent too")
    parser.add_argument("--rate", type=int, default=180, help="words per minute")
    parser.add_argument("--gap", type=float, default=GAP_SECONDS)
    args = parser.parse_args()

    payload = json.loads(args.calls.read_text())
    calls = payload["calls"] if isinstance(payload, dict) else payload
    chosen = list(enumerate(calls)) if args.all else [(args.index, calls[args.index])]

    tag = re.sub(r"[^a-z0-9]+", "-", str(payload.get("agent_model", "model")).lower()).strip("-")
    for index, call in chosen:
        stem = f"{tag}-call{index + 1}-{call['brief']['client_name'].split()[-1].lower()}"
        audio = render(call, args.out_dir / f"{stem}.mp3",
                       args.business_voice, args.agent_voice, args.rate, args.gap,
                       None if args.no_neural else args.piper_model,
                       None if args.no_neural else KokoroVoice(args.kokoro_dir, args.kokoro_voice))
        (args.out_dir / f"{stem}.txt").write_text(transcript_text(call))
        size = audio.stat().st_size / 1024
        print(f"{audio}  ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
