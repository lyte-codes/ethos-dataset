#!/usr/bin/env bash
# Preference-train both supervised adapters, publish all four builds, and score them
# against each other. Designed to be left running overnight.
#
#   ./training/run_pipeline.sh              # train, publish, compare
#   ./training/run_pipeline.sh --no-publish # train and compare only
#
# Builds are published in the order they were actually made — v4, v6, v5, v7 — so the
# night's letters (a, b, c, d) say when each one came into existence rather than telling
# a tidier story than what happened. Each release's notes explain where it sits relative
# to the others.
#
# Why that order and not alternating SFT, DPO, SFT, DPO: v4 and v6 are a single
# supervised run, snapshotted at the end of epoch 2 and again at the end of epoch 3.
# Alternating would mean halting that run after epoch 2, training a preference model,
# then resuming the supervised run — and train_lora.py cannot resume, so epoch 3 would
# have to start from scratch. Both supervised snapshots therefore come first, and the
# two preference runs follow.
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

# Pinned once, at launch. The preference runs finish after midnight, and without this the
# date would roll over mid-pipeline: one night's four builds would land under two dates with
# the letter resetting to a partway through.
BUILD_NIGHT="${BUILD_NIGHT:-$(date +%Y%m%d)}"
[ "${1:-}" = "--no-publish" ] && PUBLISH=0

# Repeated at the foot of every release, so a build found on its own still explains how
# it relates to the other three.
SHARED_NOTES="**How the four builds relate**

\`v4\` and \`v6\` are the same supervised run: v4 is its state at the end of epoch 2, v6 at
the end of epoch 3. \`v5\` is v4 with preference training applied, \`v7\` is v6 with the same
preference training applied. Nothing else differs within either pair.

That gives two independent comparisons. \`v4\` against \`v6\` isolates the third epoch, with
the preference stage held constant at none. \`v4\` against \`v5\` — and \`v6\` against \`v7\` —
isolates the preference training, with the supervised model held constant.

They were built supervised-then-supervised, preference-then-preference, rather than
alternating. Alternating would have meant stopping the supervised run after epoch 2 to
train a preference model and then resuming it, and the training script cannot resume, so
epoch 3 would have had to start over. Release ordinals follow the order the builds were
actually made."

mkdir -p logs
echo "[$(date +%H:%M)] build night $BUILD_NIGHT"

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
        --date "$BUILD_NIGHT" \
        --notes "$notes

$SHARED_NOTES" 2>&1 | tee -a "logs/publish-${lineage}.log"
}

# --- 1st: v4, supervised, end of epoch 2 ------------------------------------------
publish checkpoints/ethos-v4-epoch2 v4 \
    "Supervised only, cut at the end of epoch 2 of a three-epoch run. Preference
training has not been applied, so nothing in this build penalises repeating a detail
the brief never contained — expect it to fail that gate."

# --- 2nd: v6, supervised, end of epoch 3 ------------------------------------------
# The same run as v4, allowed to finish. Published second because it finished second.
publish checkpoints/ethos-v4 v6 \
    "Supervised only, all three epochs — the completion of the very run that produced
v4 at its epoch-2 mark. Same data, same hyperparameters, one more pass. Still no
preference training."

# --- 3rd: v5, preference training on v4 -------------------------------------------
train v5-dpo checkpoints/ethos-v5/adapter_model.safetensors \
    python3 training/train_dpo.py --data "$PREFS" \
        --sft-adapter checkpoints/ethos-v4-epoch2 --output checkpoints/ethos-v5 --batch-size 1

publish checkpoints/ethos-v5 v5 \
    "Preference training applied to v4. Built third rather than second because v4 and
v6 came out of one continuous supervised run that was left to finish before any
preference training began."

# --- 4th: v7, preference training on v6 -------------------------------------------
train v7-dpo checkpoints/ethos-v7/adapter_model.safetensors \
    python3 training/train_dpo.py --data "$PREFS" \
        --sft-adapter checkpoints/ethos-v4 --output checkpoints/ethos-v7 --batch-size 1

publish checkpoints/ethos-v7 v7 \
    "Preference training applied to v6 — the same preference data and settings used for
v5, over the three-epoch supervised model instead of the two-epoch one."

# --- measure generalisation -------------------------------------------------------
# Behaviour on five calls is coarse. This is the direct test of whether the extra epoch
# taught the task or the training set: held-out loss against loss on data it did see.
echo "[$(date +%H:%M)] held-out loss for all four"
python3 training/eval_heldout.py \
    --adapter base: \
    --adapter v4:checkpoints/ethos-v4-epoch2 \
    --adapter v5:checkpoints/ethos-v5 \
    --adapter v6:checkpoints/ethos-v4 \
    --adapter v7:checkpoints/ethos-v7 \
    --out data/heldout_eval.json 2>&1 | tee logs/heldout.log

# --- compare ----------------------------------------------------------------------
echo "[$(date +%H:%M)] scoring all four on identical calls"
python3 training/compare_models.py \
    --model v4:checkpoints/ethos-v4-epoch2 \
    --model v5:checkpoints/ethos-v5 \
    --model v6:checkpoints/ethos-v4 \
    --model v7:checkpoints/ethos-v7 \
    --out data/model_comparison.json 2>&1 | tee logs/compare.log

echo "[$(date +%H:%M)] pipeline finished"
