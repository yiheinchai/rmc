#!/usr/bin/env bash
# Run HotPotQA upstream as two parallel Codex shards (cases 0-49 and 50-99).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/hotpot-parallel-${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

MAIN="papers/rse/results/competitive-latest.json"
WORK="papers/rse/results/hotpot-workspace"
SHARD2="papers/rse/results/hotpot-shard2"
SPLIT=50
STEP2_PATTERN='run_competitive_evals.py --agent codex --samples 1 --skip-upstream'
LOCK="/tmp/hotpot-parallel.lock"

echo "=== HotPotQA parallel shards (split at case $SPLIT) ==="
echo "log: $LOG"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another HotPotQA parallel runner is active — exiting"
  exit 0
fi

echo "waiting for competitive suite step 2 process to finish..."
while pgrep -f "$STEP2_PATTERN" >/dev/null 2>&1; do
  echo "$(date -u +%H:%M:%S) step 2 still running..."
  sleep 120
done

echo "waiting for step-2 markers in $MAIN..."
while ! python3 -c "
import json, sys
d = json.load(open('$MAIN'))
ok = (
    d.get('agent') == 'codex'
    and d.get('rmc_bench')
    and d.get('wikiskill_probe')
    and d.get('memgpt_nested_kv')
    and d.get('session_study')
)
sys.exit(0 if ok else 1)
" 2>/dev/null; do
  echo "$(date -u +%H:%M:%S) waiting for competitive-latest.json step-2 sections..."
  sleep 30
done

if ! python3 scripts/merge_competitive_upstream.py --check --competitive "$MAIN" | grep -qx hotpotqa-dev; then
  echo "HotPotQA already complete in $MAIN"
  exit 0
fi

mkdir -p "$WORK" "$SHARD2"

# Stop single-threaded HotPot driver if still running (replaced by this orchestrator).
if pgrep -f "run_competitive_evals.py.*hotpotqa-dev" >/dev/null 2>&1; then
  echo "stopping existing HotPotQA driver..."
  pkill -f "run_competitive_evals.py.*hotpotqa-dev" || true
  sleep 3
fi

SHARD2_OFFSET="$SPLIT"
if [[ -f "$WORK/competitive-latest.json" ]]; then
  SHARD2_OFFSET=$(python3 -c "
import json, re
d = json.load(open('$WORK/competitive-latest.json'))
cases = (d.get('upstream') or {}).get('hotpotqa-dev') or {}
ids = [c.get('case_id','') for c in cases.get('cases') or [] if c.get('case_id')]
nums = [int(re.search(r'(\d+)$', i).group(1)) for i in ids if re.search(r'(\d+)$', i)]
start = ($SPLIT if not nums else max(max(nums) + 1, $SPLIT))
print(min(start, 99))
" 2>/dev/null || echo "$SPLIT")
fi
echo "shard2 offset from checkpoint: $SHARD2_OFFSET"

echo "$(date -u +%H:%M:%S) shard1: cases 0-$((SPLIT - 1)) (limit=$SPLIT, resume)"
SHARD1_PID=""
if python3 -c "
import json, re, sys
p = '$WORK/competitive-latest.json'
if not __import__('pathlib').Path(p).exists():
    sys.exit(0)
d = json.load(open(p))
cases = (d.get('upstream') or {}).get('hotpotqa-dev') or {}
ids = [c.get('case_id','') for c in cases.get('cases') or [] if c.get('case_id')]
nums = [int(re.search(r'(\d+)$', i).group(1)) for i in ids if re.search(r'(\d+)$', i)]
sys.exit(0 if (nums and min(nums) == 0 and max(nums) >= $SPLIT - 1) else 1)
" 2>/dev/null; then
  echo "shard1 already complete through case $((SPLIT - 1)) — skipping"
else
  python3 scripts/run_competitive_evals.py \
    --agent codex \
    --samples 1 \
    --skip-bench \
    --skip-probe \
    --skip-memgpt \
    --skip-session \
    --merge "$MAIN" \
    --out "$WORK" \
    --resume \
    --limit "$SPLIT" \
    --upstream hotpotqa-dev &
  SHARD1_PID=$!
fi

echo "$(date -u +%H:%M:%S) shard2: cases $SHARD2_OFFSET-99 (offset=$SHARD2_OFFSET)"
python3 scripts/run_competitive_evals.py \
  --agent codex \
  --samples 1 \
  --skip-bench \
  --skip-probe \
  --skip-memgpt \
  --skip-session \
  --merge "$MAIN" \
  --out "$SHARD2" \
  --offset "$SHARD2_OFFSET" \
  --upstream hotpotqa-dev &
PID2=$!

if [[ -n "$SHARD1_PID" ]]; then
  wait "$SHARD1_PID" "$PID2"
else
  wait "$PID2"
fi

echo "$(date -u +%H:%M:%S) merging HotPot shards into $WORK/competitive-latest.json"
python3 scripts/merge_competitive_upstream.py \
  --competitive "$WORK/competitive-latest.json" \
  --merge-shards "$WORK/competitive-latest.json" "$SHARD2/competitive-latest.json" \
  --stem hotpotqa-dev

python3 scripts/merge_competitive_upstream.py \
  --competitive "$MAIN" \
  --from-competitive "$WORK/competitive-latest.json" \
  --stem hotpotqa-dev

echo "=== HotPotQA parallel done ==="
