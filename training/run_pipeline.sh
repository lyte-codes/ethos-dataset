#!/usr/bin/env bash
# Preference-train both supervised models and evaluate all four, unattended.
#
# Runs strictly in sequence. Two trainings at once on one GPU is slower than either
# alone and risks taking both down on an out-of-memory, which is a bad trade when the
# whole point is to leave it running overnight.
#
#   ./training/run_pipeline.sh
#
# Safe to re-run: any stage whose output already exists is skipped, so an interrupted
# night picks up where it stopped rather than starting over.

set -uo pipefail
cd "$(dirname "$0")/.."

PREFS=data/ethos_preferences_v2.jsonl

stage() {
    local name="$1" output="$2"
    shift 2
    if [ -e "$output" ]; then
        echo "[$(date +%H:%M)] $name — already done, skipping"
        return 0
    fi
    echo "[$(date +%H:%M)] $name — starting"
    if "$@" > "logs/${name}.log" 2>&1; then
        echo "[$(date +%H:%M)] $name — done"
    else
        # Keep going: a failed DPO on one adapter should not cost the other one's run.
        echo "[$(date +%H:%M)] $name — FAILED (see logs/${name}.log)" >&2
        return 1
    fi
}

mkdir -p logs

if [ ! -d checkpoints/ethos-v4 ]; then
    echo "checkpoints/ethos-v4 missing — supervised training has not finished" >&2
    exit 1
fi

stage v5-dpo checkpoints/ethos-v5/adapter_model.safetensors \
    python3 training/train_dpo.py \
        --data "$PREFS" \
        --sft-adapter checkpoints/ethos-v4 \
        --output checkpoints/ethos-v5 \
        --batch-size 1

stage v7-dpo checkpoints/ethos-v7/adapter_model.safetensors \
    python3 training/train_dpo.py \
        --data "$PREFS" \
        --sft-adapter checkpoints/ethos-v6 \
        --output checkpoints/ethos-v7 \
        --batch-size 1

echo "[$(date +%H:%M)] evaluating all four"
python3 training/compare_models.py \
    --model v4:checkpoints/ethos-v4 \
    --model v5:checkpoints/ethos-v5 \
    --model v6:checkpoints/ethos-v6 \
    --model v7:checkpoints/ethos-v7 \
    --out data/model_comparison.json 2>&1 | tee logs/compare.log

echo "[$(date +%H:%M)] pipeline finished"
