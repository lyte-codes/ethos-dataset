#!/usr/bin/env bash
# Waits for the hyperparameter search to finish generation 2, stops it, and trains v7 on
# the configuration that won.
#
# Why stop after generation 2 rather than letting generation 3 run: generation 1 was
# effectively random search, and generation 2 is the first one that breeds from a winner,
# so it is where the genetic part earns its cost. A third generation is another ~13 hours
# of the only GPU for what is, on the evidence so far, a refinement — and a configuration
# that already beats the shipped model is worth more as a trained build than as a slightly
# better number in a search log.
#
# Why two epochs and not three: v4 and v6 are the same run at its epoch-2 and epoch-3
# marks, and the third epoch measured *worse* on held-out data (1.516 against 1.342)
# despite a lower training loss. It memorised. The winning search score was measured at a
# single epoch, so two is already an extrapolation; three would repeat a known mistake.
#
#   nohup ./training/queue_v7.sh > logs/queue_v7.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

RUNLOG=logs/hpsearch_run.log
OUTPUT=checkpoints/ethos-v7

# The winner of generation 1, measured at 1.2632 on the full 189 conversations against
# v4's 1.342 — better generalisation on half the training. alpha is rank x 4.
RANK=16
ALPHA=64
DROPOUT=0.1
LR=0.0004
ACCUM=2
MODULES=q_proj,k_proj,v_proj,o_proj   # attention_only won
EPOCHS=2

say() { echo "[$(date +%H:%M)] $*"; }

say "waiting for generation 2 to finish (checking every 5 min)"

# The search prints its generation banner before starting one, so the appearance of
# generation 3's banner is the signal that generation 2 is complete. Bracket in the pgrep
# pattern stops this script's own command line from matching itself.
while true; do
    if ! pgrep -f "search_hyperparameter[s].py" > /dev/null; then
        say "search exited on its own — proceeding"
        break
    fi
    if grep -q "=== generation 3/" "$RUNLOG" 2>/dev/null; then
        say "generation 2 done, generation 3 starting — stopping the search here"
        pkill -f "search_hyperparameter[s].py"
        # Give the child train/eval processes a moment to die with it, so v7 does not
        # start while another MPS workload is still holding memory. Two on one GPU is
        # slower than either alone and risks losing both to a single out-of-memory.
        sleep 30
        pkill -f "train_lor[a].py" 2>/dev/null
        pkill -f "eval_heldou[t].py" 2>/dev/null
        sleep 10
        break
    fi
    sleep 300
done

# Re-read the winner from the journal rather than trusting the constants above: if
# generation 2 bred something better, that is what v7 should be. Falls back to the
# hardcoded generation-1 winner if the journal cannot be read.
BEST=$(python3 - <<'PY'
import json
from pathlib import Path
try:
    rows = [json.loads(l) for l in Path("logs/hpsearch.jsonl").read_text().splitlines() if l.strip()]
    scored = [r for r in rows if r.get("score") is not None]
    # Only the top rung is comparable — a score on 20 conversations and a score on 189
    # are different measurements, and ranking across them would prefer the cheap rung.
    top = max(r["conversations"] for r in scored)
    best = min((r for r in scored if r["conversations"] == top), key=lambda r: r["score"])
    c = best["candidate"]
    print(f'{c["lora_rank"]} {int(c["lora_rank"] * c["lora_alpha_ratio"])} {c["lora_dropout"]} '
          f'{c["learning_rate"]} {c["gradient_accumulation_steps"]} '
          f'{"q_proj,k_proj,v_proj,o_proj" if c["attention_only"] else "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"} '
          f'{best["score"]} {top}')
except Exception:
    pass
PY
)

if [ -n "$BEST" ]; then
    read -r RANK ALPHA DROPOUT LR ACCUM MODULES SCORE CONVS <<< "$BEST"
    say "best from the journal: $SCORE on $CONVS conversations"
fi

say "training v7 — rank $RANK, alpha $ALPHA, dropout $DROPOUT, lr $LR, accum $ACCUM, $EPOCHS epochs"
say "modules: $MODULES"

if [ -e "$OUTPUT/adapter_model.safetensors" ]; then
    say "v7 already exists — nothing to do"
    exit 0
fi

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
    exit 1
fi
say "v7 trained"

# Score it against everything else on both metrics. Held-out loss is the one that judges a
# supervised build; the preference eval is included so the composite score has a complete
# row for v7 rather than a half-filled one.
say "scoring v7"
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

say "v7 scored — composite:"
cat logs/v7-composite.log
say "absolute:"
cat logs/v7-composite-absolute.log

say "done. v7 is NOT published — review the numbers first, then run:"
say "  python3 training/publish_build.py --adapter $OUTPUT --lineage v7 --notes '...'"
