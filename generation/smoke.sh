#!/usr/bin/env bash
# Preflight + small smoke run of generate_dataset.py, with live status.
set -uo pipefail

COUNT="${1:-5}"
MODEL="${MODEL:-llama3.1:8b}"
BASE_URL="${BASE_URL:-http://localhost:11434}"
OUT="${OUT:-smoke_test.jsonl}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEPS=4
step() { printf '\033[1m[%d/%d %3d%%]\033[0m %s\n' "$1" "$STEPS" "$(( $1 * 100 / STEPS ))" "$2"; }
ok()   { printf '        \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '        \033[33mwarn\033[0m %s\n' "$1"; }
die()  { printf '        \033[31mfail\033[0m %s\n' "$1"; exit 1; }

step 1 "ollama binary"
command -v ollama >/dev/null || die "ollama not installed (brew install ollama)"
ok "$(command -v ollama)"

step 2 "server at $BASE_URL"
if ! curl -sf --max-time 3 "$BASE_URL/api/tags" >/dev/null; then
  warn "not responding, starting 'ollama serve'"
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for _ in $(seq 30); do
    curl -sf --max-time 2 "$BASE_URL/api/tags" >/dev/null && break
    sleep 1
  done
fi
curl -sf --max-time 3 "$BASE_URL/api/tags" >/dev/null || die "server did not come up (see /tmp/ollama-serve.log)"
ok "responding"

step 3 "model $MODEL"
AVAILABLE=$(curl -sf --max-time 5 "$BASE_URL/api/tags" | python3 -c 'import json,sys; print("\n".join(m["name"] for m in json.load(sys.stdin)["models"]))')
if ! grep -qx "$MODEL" <<<"$AVAILABLE"; then
  warn "$MODEL not pulled. Available:"
  sed 's/^/          - /' <<<"$AVAILABLE"
  FALLBACK=$(head -1 <<<"$AVAILABLE")
  [ -n "$FALLBACK" ] || die "no models at all (ollama pull $MODEL)"
  warn "falling back to $FALLBACK"
  MODEL="$FALLBACK"
fi
ok "using $MODEL"

step 4 "generating $COUNT conversations"
python3 "$HERE/generate_dataset.py" -n "$COUNT" -o "$OUT" --model "$MODEL" --base-url "$BASE_URL" "${@:2}"
STATUS=$?

echo
[ -s "$OUT" ] || die "no output written"
python3 - "$OUT" "$COUNT" <<'PY'
import collections, json, sys

path, requested = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
convs = collections.defaultdict(list)
for row in rows:
    convs[row["conversation_id"]].append(row)

firsts = [group[0] for group in convs.values()]
turns = [len(group) for group in convs.values()]
print(f"\033[1mquality check\033[0m  {len(convs)}/{requested} conversations "
      f"({len(convs) / requested * 100:.0f}%)  {len(rows)} examples  "
      f"{min(turns)}-{max(turns)} assistant turns each")
for field in ("persona", "complication", "intent", "service"):
    seen = collections.Counter(row[field] for row in firsts)
    print(f"  {field:13s} " + ", ".join(f"{k} x{v}" for k, v in seen.most_common()))

last = convs[sorted(convs)[0]][-1]
print("\n\033[1msample conversation\033[0m")
for message in last["messages"][1:] + [{"role": "assistant", "content": last["response"]}]:
    speaker = "caller   " if message["role"] == "user" else "assistant"
    print(f"  {speaker}> {message['content']}")
PY
exit $STATUS
