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


# "  v4  train 0.628  held-out 1.342  gap +0.714" and "    v4 train: 50/80"
EVAL_RESULT = re.compile(r"^\s*(\S+)\s+train ([\d.]+)\s+held-out ([\d.]+)\s+gap ([+-][\d.]+)", re.M)
EVAL_PROGRESS = re.compile(r"^\s+(\S+) (train|held-out): (\d+)/(\d+)", re.M)
EVAL_SCORING = re.compile(r"^\s+scoring (\S+)…", re.M)


def parse_eval(log: Path) -> dict:
    """Held-out scoring: results already in, and how far through the current model it is."""
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")

    results = [
        {"model": name, "train": float(train), "heldout": float(held), "gap": float(gap)}
        for name, train, held, gap in EVAL_RESULT.findall(text)
    ]
    scoring = EVAL_SCORING.findall(text)
    progress = EVAL_PROGRESS.findall(text)

    state = {"results": results, "running": is_running("eval_heldout.py")}
    done = {r["model"] for r in results}
    if scoring and scoring[-1] not in done:
        state["current"] = scoring[-1]
        if progress:
            model, split, at, total = progress[-1]
            state["current_split"] = split
            state["current_progress"] = f"{at}/{total}"
    return state


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


# One DPO optimizer step costs more than one supervised step: it runs the chosen and the
# rejected response through both the policy and the reference model, and accumulates over
# twice as many examples. This multiplier is a starting guess, replaced by a measurement as
# soon as a real DPO run reports its own rate.
DPO_STEP_MULTIPLIER = 3.0
DPO_STEPS = 178  # 1421 pairs at batch 1, accumulating 8, for one epoch

# Measured rates are written here as they are observed, so an estimate improves as the
# night goes on instead of repeating the same guess. The second preference run is timed
# from the first one's real rate; a later night starts from tonight's measurements.
CALIBRATION = Path("logs/timings.json")


def load_calibration() -> dict:
    try:
        return json.loads(CALIBRATION.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_calibration(data: dict) -> None:
    try:
        CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION.write_text(json.dumps(data, indent=2))
    except OSError:
        pass  # a dashboard that cannot cache a timing is still a working dashboard


def dpo_progress(name: str) -> dict:
    """Progress of a preference run from its own log, if one is going."""
    log = Path("logs") / f"{name}.log"
    if not log.exists():
        return {}
    return parse_log_tail(log)


def published_versions() -> dict[str, str]:
    """What each build is actually called, read from the registry rather than assumed."""
    registry = Path("releases/registry.jsonl")
    if not registry.exists():
        return {}
    versions = {}
    with registry.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            versions[entry.get("lineage", {}).get("model", "")] = entry.get("version", "")
    return versions


def next_version(target: str, published: dict[str, str]) -> str:
    """What an unpublished build would be called if it were published now."""
    numbers = [int(v.rsplit(".", 1)[1]) for v in published.values()
               if v.startswith(f"{target}-build.") and v.rsplit(".", 1)[1].isdigit()]
    return f"{target}-build.{max(numbers, default=0) + 1}"


def held_out_scores() -> dict[str, dict]:
    """Each build's measured result, so a finished stage can show what it scored."""
    path = Path("data/heldout_eval.json")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def pipeline_stages(sft: dict) -> list[dict]:
    """The four builds, what state each is in, and how long until it exists.

    Times for stages that have not started are estimates and say so. A dashboard that
    presents a guess in the same voice as a measurement teaches you to distrust both.
    """
    step = sft.get("step") or 0
    total = sft.get("max_steps") or 1014
    rate = sft.get("sec_per_step") or 0
    epoch_two_step = round(total / 3 * 2)

    def seconds_to(target_step: int) -> float | None:
        if not rate or step >= target_step:
            return None
        return (target_step - step) * rate

    published = published_versions()
    unpublished = next_version("0.1", published)
    scores = held_out_scores()
    sft_running = is_running("train_lora.py")
    v4_done = Path("checkpoints/ethos-v4-epoch2/adapter_model.safetensors").exists()
    v6_done = Path("checkpoints/ethos-v4/adapter_model.safetensors").exists()

    stages = []

    stages.append({
        "key": "v4", "label": "v4 — supervised, epoch 2",
        "release": published.get("v4") or unpublished,
        "published": "v4" in published, "score": scores.get("v4"),
        "state": "done" if v4_done else "running" if sft_running else "waiting",
        "eta": None if v4_done else seconds_to(epoch_two_step),
        "estimated": False, "confidence": "measured",
        "detail": f"snapshot at step {epoch_two_step}",
    })

    stages.append({
        "key": "v6", "label": "v6 — supervised, epoch 3",
        "release": published.get("v6") or unpublished,
        "published": "v6" in published, "score": scores.get("v6"),
        "state": "done" if v6_done else "running" if sft_running else "waiting",
        "eta": None if v6_done else seconds_to(total),
        "estimated": False, "confidence": "measured",
        "detail": f"same run, through step {total}",
    })

    # The preference runs queue behind the supervised one, so their clock starts when it ends.
    queue = seconds_to(total) or 0
    calibration = load_calibration()
    if rate:
        calibration["sft_sec_per_step"] = round(rate, 2)

    # Anything a preference run has already reported beats the guess, for itself and for
    # the one queued behind it.
    live_runs = {key: dpo_progress(f"{key}-dpo") for key in ("v5", "v7")}
    for live in live_runs.values():
        if live.get("sec_per_step"):
            calibration["dpo_sec_per_step"] = round(live["sec_per_step"], 2)
            calibration["dpo_total_steps"] = live.get("max_steps") or DPO_STEPS
            if rate:
                calibration["dpo_multiplier"] = round(live["sec_per_step"] / rate, 2)

    dpo_rate = calibration.get("dpo_sec_per_step")
    dpo_steps = calibration.get("dpo_total_steps", DPO_STEPS)
    multiplier = calibration.get("dpo_multiplier", DPO_STEP_MULTIPLIER)
    if dpo_rate:
        per_run, confidence = dpo_rate * dpo_steps, "calibrated"
    else:
        per_run, confidence = (rate or 0) * multiplier * dpo_steps, "guess"

    pending = 0.0
    for key, base in [("v5", "v4")]:
        done = Path(f"checkpoints/ethos-{key}/adapter_model.safetensors").exists()
        live = live_runs[key]
        if done:
            state, eta, level = "done", None, "measured"
        elif live.get("step"):
            state = "running"
            eta = (live["max_steps"] - live["step"]) * (live.get("sec_per_step") or 0)
            level = "measured"
        else:
            state = "waiting"
            pending += per_run
            eta = queue + pending
            level = confidence
        stages.append({
            "key": key, "label": f"{key} — preference training on {base}",
            "release": published.get(key) or unpublished,
            "published": key in published, "score": scores.get(key),
            "state": state, "eta": eta, "estimated": level != "measured", "confidence": level,
            "detail": f"DPO from {base}",
        })

    save_calibration(calibration)
    return stages


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

    # A finished adapter beats whatever the log tail says. The log's last line is wherever
    # the run happened to stop being watched — for a run that was killed and later resumed
    # under a different log, that is a stale mid-run reading with a rate to match, and it
    # would otherwise be shown as if it were live.
    if (output / "adapter_model.safetensors").exists():
        state["step"] = state.get("max_steps") or state.get("step")
        state["complete"] = True
        state["remaining"] = "finished"
    else:
        state["complete"] = False

    if state["stage"] == "idle" and state.get("log"):
        state["stage"] = "training"

    state["stages"] = pipeline_stages(state)

    evaluation = parse_eval(Path("logs/heldout_bf16.log"))
    if evaluation.get("results") or evaluation.get("current"):
        state["evaluation"] = evaluation
        if evaluation["running"]:
            state["stage"] = "evaluating"

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
