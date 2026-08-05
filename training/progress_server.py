#!/usr/bin/env python3
"""Serve a live view of a training run.

Reads whatever the run has already written — the newest checkpoint's
trainer_state.json for the metric history, and the tail of the log for the
step the run is on right now — and serves both as JSON alongside a dashboard
that polls it. Nothing here touches the training process itself.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# " 25%|██▍       | 206/828 [47:51<2:10:29, 12.59s/it]" — also matches it/s for fast runs.
PROGRESS = re.compile(r"(\d+)/(\d+) \[(\d+:\d+(?::\d+)?)<([^,]+),\s*([\d.]+)(s/it|it/s)\]")

# "[#####...] 19.5%  39/200 convs  267 examples  3 failed  42s/conv  ETA 1h 51m"
GENERATION = re.compile(
    r"(\d+)/(\d+) convs\s+(\d+) examples\s+(\d+) failed\s+([\d.]+)s/conv\s+ETA ([^\[]*)"
)

HERE = Path(__file__).resolve().parent


def latest_checkpoint(output: Path) -> Path | None:
    checkpoints = [
        d for d in output.glob("checkpoint-*")
        if d.is_dir() and (d / "trainer_state.json").exists()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda d: int(d.name.rsplit("-", 1)[1]))


def parse_log_tail(log: Path) -> dict:
    """The newest progress reading in the log — the run is further along than the checkpoint."""
    if not log.exists():
        return {}
    with log.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 8192))
        tail = handle.read().decode("utf-8", "replace").replace("\r", "\n")

    matches = PROGRESS.findall(tail)
    if not matches:
        return {}
    step, total, elapsed, remaining, rate, unit = matches[-1]
    return {
        "step": int(step),
        "max_steps": int(total),
        "elapsed": elapsed,
        "remaining": remaining,
        "sec_per_step": float(rate) if unit == "s/it" else (1.0 / float(rate) if float(rate) else 0.0),
    }


def is_running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def parse_generation(log: Path) -> dict:
    """Where the dataset run has got to, read from the same progress bar a human would."""
    if not log.exists():
        return {}
    with log.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 8192))
        tail = handle.read().decode("utf-8", "replace").replace("\r", "\n")

    matches = GENERATION.findall(tail)
    if not matches:
        return {}
    done, total, examples, failed, rate, eta = matches[-1]
    done, failed = int(done), int(failed)
    attempted = done + failed
    return {
        "conversations": done,
        "target": int(total),
        "examples": int(examples),
        "failed": failed,
        # What the quality gate let through, which is the number worth watching: a run that
        # slows down usually means the gate is rejecting more, not that the model got slower.
        "pass_rate": round(done / attempted, 3) if attempted else None,
        "sec_per_conversation": float(rate),
        "remaining": eta.strip(),
    }


def collect(output: Path, log: Path, pattern: str,
            generation_log: Path | None = None) -> dict:
    state: dict = {"running": is_running(pattern), "log": [], "max_grad_norm": None}

    generation = parse_generation(generation_log) if generation_log else {}
    generating = is_running("generate_dataset.py")
    if generation:
        generation["running"] = generating
        state["generation"] = generation
    # Training is the headline once it starts; until then the dataset run is the live stage.
    state["stage"] = ("training" if state["running"]
                      else "generation" if generating
                      else "training" if state["log"] else "idle")

    checkpoint = latest_checkpoint(output)
    if checkpoint:
        saved = json.loads((checkpoint / "trainer_state.json").read_text())
        state["checkpoint"] = checkpoint.name
        state["max_steps"] = saved.get("max_steps")
        state["epochs"] = saved.get("num_train_epochs")
        state["save_steps"] = saved.get("save_steps")
        state["checkpoint_step"] = saved.get("global_step")
        state["log"] = [
            {
                "step": entry["step"],
                "loss": entry.get("loss"),
                "lr": entry.get("learning_rate"),
                "gn": entry.get("grad_norm"),
            }
            for entry in saved.get("log_history", [])
            if "loss" in entry
        ]

    state.update({k: v for k, v in parse_log_tail(log).items() if v is not None})
    if state["stage"] == "idle" and state.get("log"):
        state["stage"] = "training"

    # The clip ceiling is what actually reaches the weights, so the dashboard needs it
    # to put a spike in context. Only recorded in the checkpoint's training_args.
    if checkpoint and (checkpoint / "training_args.bin").exists():
        try:
            import warnings

            import torch

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arguments = torch.load(checkpoint / "training_args.bin", weights_only=False)
            state["max_grad_norm"] = getattr(arguments, "max_grad_norm", None)
        except Exception:
            pass

    return state


def make_handler(output: Path, log: Path, pattern: str, page: Path, generation_log: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler's interface
            if self.path.startswith("/data.json"):
                body = json.dumps(collect(output, log, pattern, generation_log)).encode()
                content_type = "application/json"
            else:
                body = page.read_bytes()
                content_type = "text/html; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # the polling would drown out anything worth reading

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ethos-v2"))
    parser.add_argument("--log", type=Path, default=Path("v2-train.log"))
    parser.add_argument("--page", type=Path, default=HERE / "progress.html")
    parser.add_argument("--generation-log", type=Path, default=Path("v2-generate.log"),
                        help="dataset generation log, shown while no training is running")
    parser.add_argument("--pattern", default="train_lora.py",
                        help="pgrep pattern used to tell whether the run is still alive")
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port),
                        make_handler(args.output, args.log, args.pattern, args.page,
                                     args.generation_log))
    print(f"live progress on http://127.0.0.1:{args.port}  (watching {args.output})")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
