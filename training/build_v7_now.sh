#!/usr/bin/env bash
# Cuts the hyperparameter search short, builds v7 from the best configuration found so
# far, proves it on a real generated call, scores it against every other build, and then
# puts the search back where it left off.
#
# The search is safe to interrupt: logs/hpsearch.jsonl caches every finished
# (candidate, rung) pair, so restarting re-uses all of them and only the one candidate
# in flight at the moment of the kill is repeated.
#
# Strictly sequential throughout. Two MPS workloads on one GPU is slower than either
# alone and risks losing both to a single out-of-memory.
#
#   nohup ./training/build_v7_now.sh > logs/build_v7_now.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

OUTPUT=checkpoints/ethos-v7
EPOCHS=2

say() { echo "[$(date +%H:%M)] $*"; }

# --- 1. stop everything holding the GPU -------------------------------------------
# The armed waiter goes first: it is watching for generation 3's banner and would fire a
# second v7 build later if left running.
say "disarming the queued v7 waiter"
pkill -f "queue_v[7].sh" 2>/dev/null && say "  waiter stopped" || say "  no waiter running"

say "stopping the hyperparameter search"
pkill -f "search_hyperparameter[s].py" 2>/dev/null
sleep 5
# The search spawns train_lora.py / eval_heldout.py as children; killing the parent does
# not take them with it, and either one still holds GPU memory.
pkill -f "train_lor[a].py" 2>/dev/null
pkill -f "eval_heldou[t].py" 2>/dev/null

say "waiting for the GPU to clear"
for _ in $(seq 1 60); do
    if ! pgrep -f "train_lor[a].py" > /dev/null && ! pgrep -f "eval_heldou[t].py" > /dev/null; then
        break
    fi
    sleep 5
done
sleep 20   # let MPS actually release the allocation, not just lose the process
say "GPU clear"

# --- 2. pick the configuration ------------------------------------------------------
# Read from the journal rather than hardcoding, and compare only within the top rung —
# a score on 20 conversations and one on 189 are different measurements, and ranking
# across them would prefer whichever rung happened to be easier.
BEST=$(python3 - <<'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("logs/hpsearch.jsonl").read_text().splitlines() if l.strip()]
scored = [r for r in rows if r.get("score") is not None]
top = max(r["conversations"] for r in scored)
best = min((r for r in scored if r["conversations"] == top), key=lambda r: r["score"])
c = best["candidate"]
print(f'{c["lora_rank"]} {int(c["lora_rank"] * c["lora_alpha_ratio"])} {c["lora_dropout"]} '
      f'{c["learning_rate"]} {c["gradient_accumulation_steps"]} '
      f'{"q_proj,k_proj,v_proj,o_proj" if c["attention_only"] else "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"} '
      f'{best["score"]} {top}')
PY
)
read -r RANK ALPHA DROPOUT LR ACCUM MODULES SCORE CONVS <<< "$BEST"
say "best measured: $SCORE on $CONVS conversations"
say "v7 = rank $RANK, alpha $ALPHA, dropout $DROPOUT, lr $LR, accum $ACCUM, $EPOCHS epochs"

# --- 3. train ------------------------------------------------------------------------
if [ -e "$OUTPUT/adapter_model.safetensors" ]; then
    say "v7 already exists — skipping training"
else
    say "training v7 (roughly 1350 steps — several hours)"
    python3 training/train_lora.py \
        --data data/ethos_booking_v2.jsonl \
        --output "$OUTPUT" \
        --batch-size 1 \
        --epochs "$EPOCHS" \
        --learning-rate "$LR" \
        --lora-rank "$RANK" \
        --lora-alpha "$ALPHA" \
        --lora-dropout "$DROPOUT" \
        --grad-accum "$ACCUM" \
        --target-modules "$MODULES" \
        > logs/v7-train.log 2>&1

    if [ ! -e "$OUTPUT/adapter_model.safetensors" ]; then
        say "v7 training FAILED or was killed — see logs/v7-train.log"
        say "restarting the search anyway so the machine is not left idle"
        nohup python3 training/search_hyperparameters.py --population 6 --generations 3 \
            >> logs/hpsearch_run.log 2>&1 &
        exit 1
    fi
    say "v7 trained"
fi

# --- 4. test call --------------------------------------------------------------------
# The business side of the call is a live model, so ollama has to be up.
if ! curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
    say "starting ollama"
    OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve > logs/ollama.log 2>&1 &
    sleep 8
fi

say "running a test call with v7"
python3 training/fake_call.py --adapter "$OUTPUT" -n 2 \
    --out data/calls-v7.json > logs/v7-call.log 2>&1 \
    && say "call generated" || say "call FAILED — see logs/v7-call.log"

say "rendering the call to audio"
python3 training/render_call_audio.py --calls data/calls-v7.json \
    --out-dir data/audio >> logs/v7-call.log 2>&1 \
    && say "audio rendered" || say "audio render failed (non-fatal)"

# --- 5. score ------------------------------------------------------------------------
say "scoring v7 against every other build"
python3 training/eval_heldout.py \
    --adapter base: \
    --adapter v4:checkpoints/ethos-v4 \
    --adapter v6:checkpoints/ethos-v6 \
    --adapter v5:checkpoints/ethos-v5 \
    --adapter v7:"$OUTPUT" \
    --dtype bfloat16 --out data/heldout_eval.json > logs/v7-heldout.log 2>&1

python3 training/eval_preference.py \
    --adapter base: \
    --adapter v4:checkpoints/ethos-v4 \
    --adapter v6:checkpoints/ethos-v6 \
    --adapter v5:checkpoints/ethos-v5 \
    --adapter v7:"$OUTPUT" \
    --out data/preference_eval.json > logs/v7-prefs.log 2>&1

python3 training/composite_score.py > logs/v7-composite.log 2>&1
python3 training/composite_score.py --absolute > logs/v7-composite-absolute.log 2>&1

say "----- composite (relative) -----"
cat logs/v7-composite.log
say "----- composite (absolute) -----"
cat logs/v7-composite-absolute.log

# --- 6. put the search back ----------------------------------------------------------
# Resumes from the journal: every finished (candidate, rung) is cached, so only the
# candidate that was in flight when this script killed it gets repeated.
say "restarting the hyperparameter search"
nohup python3 training/search_hyperparameters.py --population 6 --generations 3 \
    >> logs/hpsearch_run.log 2>&1 &
disown
sleep 5
pgrep -f "search_hyperparameter[s].py" > /dev/null \
    && say "search running again" || say "search FAILED to restart"

say "done — v7 is trained and scored but NOT published. Review, then:"
say "  python3 training/publish_build.py --adapter $OUTPUT --lineage v7 --notes '...'"
