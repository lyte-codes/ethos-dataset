#!/usr/bin/env bash
# Preference-train both supervised adapters, publish all four builds in order, and
# score them against each other. Designed to be left running overnight.
#
#   ./training/run_pipeline.sh              # train, publish, compare
#   ./training/run_pipeline.sh --no-publish # train and compare only
#
# Builds are published v4, v5, v6, v7, so the day's ordinals (.1 .2 .3 .4) match the
# lineage order. That is not the order they finish in: v4 and v6 are the same
# supervised run, snapshotted at epoch 2 and epoch 3, so v6 exists long before v5 is
# trained. It waits for its turn rather than taking .2 off v5.
#
# Strictly sequential. Two trainings on one GPU is slower than either alone and risks
# losing both to a single out-of-memory, which is a poor trade for an unattended run.
#
# Safe to re-run: a stage whose output already exists is skipped, so an interrupted
# night resumes instead of starting over.

set -uo pipefail
cd "$(dirname "$0")/.."

PREFS=data/ethos_preferences_v2.jsonl
PUBLISH=1
[ "${1:-}" = "--no-publish" ] && PUBLISH=0

mkdir -p logs

train() {
    local name="$1" output="$2"
    shift 2
    if [ -e "$output" ]; then
        echo "[$(date +%H:%M)] $name — already trained, skipping"
        return 0
    fi
    echo "[$(date +%H:%M)] $name — training"
    if "$@" > "logs/${name}.log" 2>&1; then
        echo "[$(date +%H:%M)] $name — done"
    else
        echo "[$(date +%H:%M)] $name — FAILED (see logs/${name}.log)" >&2
        return 1
    fi
}

publish() {
    local adapter="$1" lineage="$2" notes="$3"
    [ "$PUBLISH" -eq 1 ] || { echo "[$(date +%H:%M)] $lineage — publishing disabled"; return 0; }
    [ -f "$adapter/adapter_model.safetensors" ] || {
        echo "[$(date +%H:%M)] $lineage — no adapter to publish, skipping" >&2; return 1; }
    if grep -q "\"model\": \"$lineage\"" releases/registry.jsonl 2>/dev/null; then
        echo "[$(date +%H:%M)] $lineage — already published, skipping"
        return 0
    fi
    echo "[$(date +%H:%M)] $lineage — publishing"
    python3 training/publish_build.py --adapter "$adapter" --lineage "$lineage" \
        --notes "$notes" 2>&1 | tee -a "logs/publish-${lineage}.log"
}

# --- v4: supervised, epoch 2 ------------------------------------------------------
publish checkpoints/ethos-v4-epoch2 v4 \
    "Supervised only, cut at the end of epoch 2. No preference training, so nothing
    penalises accepting a detail the brief never contained."

# --- v5: preference training on v4 ------------------------------------------------
train v5-dpo checkpoints/ethos-v5/adapter_model.safetensors \
    python3 training/train_dpo.py --data "$PREFS" \
        --sft-adapter checkpoints/ethos-v4-epoch2 --output checkpoints/ethos-v5 --batch-size 1

publish checkpoints/ethos-v5 v5 \
    "Preference training on top of v4. Isolates the effect of the penalty: v4 and v5
    differ by nothing else."

# --- v6: supervised, epoch 3 ------------------------------------------------------
# Trained hours ago as the tail of the same run that produced v4, held back so the
# published order matches the lineage order.
publish checkpoints/ethos-v4 v6 \
    "Supervised only, full three epochs. Against v4 this isolates the third epoch."

# --- v7: preference training on v6 ------------------------------------------------
train v7-dpo checkpoints/ethos-v7/adapter_model.safetensors \
    python3 training/train_dpo.py --data "$PREFS" \
        --sft-adapter checkpoints/ethos-v4 --output checkpoints/ethos-v7 --batch-size 1

publish checkpoints/ethos-v7 v7 \
    "Preference training on top of v6. Against v6 this confirms whether the penalty
    still helps once the supervised model has had a third epoch."

# --- compare ----------------------------------------------------------------------
echo "[$(date +%H:%M)] scoring all four on identical calls"
python3 training/compare_models.py \
    --model v4:checkpoints/ethos-v4-epoch2 \
    --model v5:checkpoints/ethos-v5 \
    --model v6:checkpoints/ethos-v4 \
    --model v7:checkpoints/ethos-v7 \
    --out data/model_comparison.json 2>&1 | tee logs/compare.log

echo "[$(date +%H:%M)] pipeline finished"
