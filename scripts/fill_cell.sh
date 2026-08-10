#!/usr/bin/env bash
# Generate conversations until a cell reaches its target, random personas, parallel.
#   ./scripts/fill_cell.sh data/scenario/banking/impossible 275 6
set -uo pipefail

DIR="${1:?usage: fill_cell.sh <scenario_dir> <needed> [parallel]}"
NEEDED="${2:-275}"
PAR="${3:-6}"
DOMAIN="$(basename "$(dirname "$DIR")")"
PTYPE="$(basename "$DIR")"
TAG="${DOMAIN}_${PTYPE}"

mkdir -p logs
LOG="logs/${TAG}.log"
JOBS="logs/${TAG}_jobs.txt"

# Refresh the sfproxy PAT once up front and export it so the xargs children
# inherit it. Tokens are a snapshot: a run longer than the token lifetime will
# start logging 401s, so re-run this script (or `eval "$(sfproxy env)"`) if that
# shows up in the log.
if command -v sfproxy >/dev/null 2>&1; then
  eval "$(sfproxy env 2>/dev/null)" || true
fi
export SNOWFLAKE_PAT ANTHROPIC_CUSTOM_HEADERS
if [ -z "${SNOWFLAKE_PAT:-}" ]; then
  echo 'error: SNOWFLAKE_PAT not set and sfproxy env failed. Run: eval "$(sfproxy env)"' >&2
  exit 1
fi

count_valid() {
  uv run src/distribution.py --report 2>/dev/null \
    | grep -E "^${DOMAIN}[[:space:]]+${PTYPE}[[:space:]]" \
    | awk '{print $3}'
}

BATCH=$(( NEEDED * 3 ))   # overshoot ~50% to absorb eval failures

# random (scenario, persona) pairs
python - "$DIR" "$BATCH" > "$JOBS" <<'PY'
import json, random, sys
from pathlib import Path
d, n = Path(sys.argv[1]), int(sys.argv[2])
personas = [p["id"] for p in json.load(open("data/personas/catalog.json"))["personas"]]
scen = sorted(str(f) for f in d.glob("*.json") if f.name != "tools.json")
rng = random.Random()
pairs = set()
while len(pairs) < min(n, len(scen) * len(personas)):
    pairs.add((rng.choice(scen), rng.choice(personas)))
for s, p in pairs:
    print(s, p)
PY

TOTAL=$(wc -l < "$JOBS")
echo "cell=$TAG  need=$NEEDED  queued=$TOTAL  parallel=$PAR  start=$(date)"
echo "valid before: $(count_valid)"

xargs -P "$PAR" -n 2 sh -c \
  'uv run src/simulate.py "$1" --persona-id "$2" --run-eval \
     --model gpt-5.1 --max-turns 20 >> '"$LOG"' 2>&1' _ < "$JOBS"

echo "valid after: $(count_valid)   done=$(date)"