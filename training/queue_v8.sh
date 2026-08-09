#!/usr/bin/env bash
# v8 — preference training on v7, then scored, then the search restarted behind it.
#
# Why this build is worth the GPU time: v7 is the best supervised model we have on
# fluency (held-out 1.260, the smallest memorisation gap of any build) and the worst on
# faithfulness (phone margin 0.745, below the untrained base model's 0.802). DPO is
# precisely the stage that installs faithfulness — it took v4's macro-margin from 1.067 to
# v5's 1.498, the largest single improvement we have measured. Applying it to a stronger
# supervised base is the obvious experiment.
#
# It may not work. v7's deficit is larger than v4's was, and DPO widens a margin rather
# than repairing a base that is actively worse than no training at all. That is the
# question this run answers.
#
#   nohup ./training/queue_v8.sh > logs/queue_v8.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

PREFS=data/ethos_preferences_v2.jsonl
SFT=checkpoints/ethos-v7
OUTPUT=checkpoints/ethos-v8

say() { echo "[$(date +%H:%M)] $*"; }

# --- clear the GPU ------------------------------------------------------------------
# The search may have been restarted since this was queued. One MPS workload at a time.
if pgrep -f "search_hyperparameter[s].py" > /dev/null; then
    say "stopping the hyperparameter search to free the GPU"
    pkill -f "search_hyperparameter[s].py"
    sleep 5
    pkill -f "train_lor[a].py" 2>/dev/null
    pkill -f "eval_heldou[t].py" 2>/dev/null
    pkill -f "eval_preferenc[e].py" 2>/dev/null
    sleep 20
fi

for _ in $(seq 1 60); do
    pgrep -f "train_lor[a].py" > /dev/null || break
    sleep 5
done
say "GPU clear"

[ -f "$SFT/adapter_model.safetensors" ] || { say "v7 adapter missing — nothing to train on"; exit 1; }

# --- train ---------------------------------------------------------------------------
if [ -e "$OUTPUT/adapter_model.safetensors" ]; then
    say "v8 already exists — skipping training"
else
    say "training v8 — DPO on v7"
    python3 training/train_dpo.py --data "$PREFS" \
        --sft-adapter "$SFT" --output "$OUTPUT" --batch-size 1 \
        > logs/v8-dpo.log 2>&1

    if [ ! -e "$OUTPUT/adapter_model.safetensors" ]; then
        say "v8 training FAILED or was killed — see logs/v8-dpo.log"
        say "restarting the search so the machine is not left idle"
        nohup python3 training/search_hyperparameters.py --population 6 --generations 3 \
            >> logs/hpsearch_run.log 2>&1 &
        exit 1
    fi
    say "v8 trained"
fi

# --- test call -------------------------------------------------------------------------
# The business side is a live model, so ollama has to be up. Two calls, same as v7 got,
# so the transcripts are comparable.
if ! curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
    say "starting ollama"
    OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve > logs/ollama.log 2>&1 &
    sleep 8
fi

say "running test calls with v8"
python3 training/fake_call.py --adapter "$OUTPUT" -n 2 \
    --out data/calls-v8.json > logs/v8-call.log 2>&1 \
    && say "calls generated" || say "calls FAILED — see logs/v8-call.log"

python3 training/render_call_audio.py --calls data/calls-v8.json \
    --out-dir data/audio >> logs/v8-call.log 2>&1 \
    && say "audio rendered" || say "audio render failed (non-fatal)"

# --- score -----------------------------------------------------------------------------
say "scoring v8 against every other build"
python3 training/eval_heldout.py \
    --adapter base: \
    --adapter v4:checkpoints/ethos-v4 \
    --adapter v6:checkpoints/ethos-v6 \
    --adapter v5:checkpoints/ethos-v5 \
    --adapter v7:"$SFT" \
    --adapter v8:"$OUTPUT" \
    --dtype bfloat16 --out data/heldout_eval.json > logs/v8-heldout.log 2>&1

python3 training/eval_preference.py \
    --adapter base: \
    --adapter v4:checkpoints/ethos-v4 \
    --adapter v6:checkpoints/ethos-v6 \
    --adapter v5:checkpoints/ethos-v5 \
    --adapter v7:"$SFT" \
    --adapter v8:"$OUTPUT" \
    --out data/preference_eval.json > logs/v8-prefs.log 2>&1

python3 training/composite_score.py > logs/v8-composite.log 2>&1
python3 training/composite_score.py --absolute > logs/v8-composite-absolute.log 2>&1

say "----- composite (relative) -----"
cat logs/v8-composite.log
say "----- composite (absolute) -----"
cat logs/v8-composite-absolute.log

# The question this build exists to answer, printed plainly rather than left to be
# worked out from two tables.
say "----- did DPO repair v7's phone deficit? -----"
python3 - <<'PY'
import json
from pathlib import Path
try:
    pref = json.loads(Path("data/preference_eval.json").read_text())
    base = pref["base"]["margin_by_corruption"]["phone"]
    print(f"  floor (untrained base): {base:.3f}")
    for name in ("v4", "v5", "v7", "v8"):
        if name in pref:
            p = pref[name]["margin_by_corruption"]["phone"]
            print(f"  {name}: {p:.3f}  {'PASSES' if p >= base else 'BELOW THE UNTRAINED MODEL'}")
except Exception as error:
    print(f"  could not read the comparison: {error}")
PY

# --- put the search back -----------------------------------------------------------------
say "restarting the hyperparameter search (now on the fixed objective)"
nohup python3 training/search_hyperparameters.py --population 6 --generations 3 \
    >> logs/hpsearch_run.log 2>&1 &
disown
sleep 5
pgrep -f "search_hyperparameter[s].py" > /dev/null \
    && say "search running again" || say "search FAILED to restart"

say "done — v8 trained and scored, NOT published. Review first."
