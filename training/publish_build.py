#!/usr/bin/env python3
"""Publish a trained adapter as a GitHub release, and record it in the registry.

Builds go out as releases rather than commits. An adapter is ~70MB, past the point
where GitHub starts warning, and git history is permanent — a bad build committed is a
bad build you carry forever. Release assets can be replaced or deleted, stay out of
the repo's history, and map one-to-one onto the build ids in docs/RELEASES.md.

    python3 training/publish_build.py --adapter checkpoints/ethos-v4 --lineage v4
    python3 training/publish_build.py --adapter checkpoints/ethos-v4 --lineage v4 --dry-run

The version is derived, not passed: <target>-nightly.<YYYYMMDD><letter>, where the letter
counts that night's builds — 20260805a, then b, then c. Two builds on one night would
otherwise collide, and a build that answers to another build's name breaks everything
downstream of it.

The date is the night the run started, not the clock time a build finished. An overnight
pipeline routinely crosses midnight, and letting the date roll over mid-run would split one
batch of builds across two dates and reset the letter to a partway through — which reads as
the first build of a new night when it is the third of the same one. Pass --date to pin it;
run_pipeline.sh does that at launch so every build from one invocation agrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REGISTRY = Path("releases/registry.jsonl")
ASSETS = ["adapter_model.safetensors", "adapter_config.json", "hyperparameters.json"]


def digest(path: Path) -> str:
    """Full SHA-256. Truncated digests are fine for spotting a change and useless for
    proving one did not happen — a published artifact deserves the whole thing."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def dirty_tree() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return bool(result.stdout.strip())


def suffix_for(index: int) -> str:
    """0 -> a, 25 -> z, 26 -> aa. Spreadsheet columns, for the same reason they use them."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return letters


def index_of(suffix: str) -> int:
    value = 0
    for character in suffix:
        value = value * 26 + (ord(character) - ord("a") + 1)
    return value - 1


def next_suffix(target: str, day: str, registry: Path) -> str:
    """Builds already published today decide this one's letter."""
    if not registry.exists():
        return suffix_for(0)
    pattern = re.compile(rf"^{re.escape(target)}-nightly\.{day}([a-z]+)$")
    highest = -1
    with registry.open(encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(json.loads(line).get("version", ""))
            if match:
                highest = max(highest, index_of(match.group(1)))
    return suffix_for(highest + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--lineage", required=True, help="internal model version, e.g. v4")
    parser.add_argument("--target", default="0.1", help="the release this is heading toward")
    parser.add_argument("--dataset", default="ethos-booking-v2")
    parser.add_argument("--stage", default="", help="sft or dpo; inferred from the path if absent")
    parser.add_argument("--notes", default="")
    parser.add_argument("--date", default="",
                        help="YYYYMMDD of the build night; defaults to today. Pin this for a "
                             "run that will cross midnight so all its builds share a date.")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.adapter.exists():
        parser.error(f"{args.adapter} not found")
    weights = args.adapter / "adapter_model.safetensors"
    if not weights.exists():
        parser.error(f"{weights} not found — is this a finished adapter?")

    day = args.date or date.today().strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", day):
        parser.error(f"--date must be YYYYMMDD, got {day!r}")
    version = f"{args.target}-nightly.{day}{next_suffix(args.target, day, args.registry)}"
    stage = args.stage or ("dpo" if "dpo" in args.adapter.name or args.lineage in {"v5", "v7"} else "sft")

    # Hash every asset, not just the weights: a build is identified by everything that
    # ships with it, and a config that drifted is as much a different build as new weights.
    digests = {name: digest(args.adapter / name)
               for name in ASSETS if (args.adapter / name).exists()}
    dataset_digest = ""
    manifest = Path(f"data/{args.dataset.replace('ethos-booking-', 'ethos_booking_')}.manifest.json")
    if manifest.exists():
        dataset_digest = json.loads(manifest.read_text()).get("data_sha256_16", "")

    entry = {
        "version": version,
        "channel": "nightly",
        "created": f"{date.today().isoformat()}",
        "lineage": {"model": args.lineage, "dataset": args.dataset},
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_sha256": digests.get("adapter_model.safetensors", ""),
        "asset_sha256": digests,
        "dataset_sha256_16": dataset_digest,
        "code_sha": git_sha(),
        "training": {"stage": stage},
        "notes": args.notes,
    }

    print(json.dumps(entry, indent=2))
    if dirty_tree():
        # The recorded sha is what someone would check out to rebuild this. If the tree has
        # changes that are not in it, that promise is already false.
        print("\nWARNING: working tree is dirty — code_sha does not describe what was run",
              file=sys.stderr)

    if args.dry_run:
        print("\ndry run: nothing published")
        return 0

    assets = [str(args.adapter / name) for name in ASSETS if (args.adapter / name).exists()]

    # Shipped alongside the assets in the usual format, so a download can be checked with
    # `shasum -a 256 -c SHA256SUMS` rather than by eye.
    checksums = args.adapter / "SHA256SUMS"
    checksums.write_text("".join(f"{value}  {name}\n" for name, value in sorted(digests.items())))
    assets.append(str(checksums))

    rows = "\n".join(f"| `{name}` | `{value}` |" for name, value in sorted(digests.items()))
    body = (f"Automated nightly build.\n\n"
            f"- lineage: model {args.lineage} on `{args.dataset}`\n"
            f"- stage: {stage}\n"
            f"- code: `{entry['code_sha']}`\n"
            f"- dataset: `{dataset_digest or 'unrecorded'}`\n\n"
            f"**SHA-256**\n\n"
            f"| asset | sha256 |\n|---|---|\n{rows}\n\n"
            f"Verify with `shasum -a 256 -c SHA256SUMS`.\n\n"
            f"{args.notes}\n\n"
            f"Not promoted. Nightlies have not cleared the eval gate in "
            f"[docs/RELEASES.md](docs/RELEASES.md) and should not be served.")

    command = ["gh", "release", "create", version,
               "--title", f"{version} ({args.lineage}, {stage})",
               "--notes", body, "--prerelease", *assets]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return 1
    print(result.stdout.strip())

    args.registry.parent.mkdir(parents=True, exist_ok=True)
    with args.registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"recorded in {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
