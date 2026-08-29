#!/usr/bin/env bash
# Run SealQA upstream as two parallel Codex shards (cases 0-55 and 56-110).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

BENCH="evals/upstream/sealqa-test.jsonl"
MAIN_OUT="papers/rse/results"
SHARD2_OUT="papers/rse/results/sealqa-shard2"
SPLIT=56
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/sealqa-parallel-${STAMP}.log"
LOCK="/tmp/sealqa-parallel.lock"

exec > >(tee -a "$LOG") 2>&1
echo "=== SealQA parallel shards (split at case $SPLIT) ==="
echo "log: $LOG"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another SealQA parallel runner is active — exiting"
  exit 0
fi

mkdir -p "$SHARD2_OUT"

# Stop single-threaded SealQA driver if still running (replaced by this orchestrator).
if pgrep -f "run_wikiskill_evals.py --agent codex --bench $BENCH" >/dev/null 2>&1; then
  echo "stopping existing SealQA wikiskill driver..."
  pkill -f "run_wikiskill_evals.py --agent codex --bench $BENCH" || true
  sleep 3
fi

echo "$(date -u +%H:%M:%S) shard1: cases 0-$((SPLIT - 1)) (limit=$SPLIT, resume)"
python3 scripts/run_wikiskill_evals.py \
  --agent codex \
  --bench "$BENCH" \
  --samples 1 \
  --checkpoint \
  --resume \
  --limit "$SPLIT" &
PID1=$!

echo "$(date -u +%H:%M:%S) shard2: cases $SPLIT-110 (offset=$SPLIT)"
python3 scripts/run_wikiskill_evals.py \
  --agent codex \
  --bench "$BENCH" \
  --samples 1 \
  --checkpoint \
  --offset "$SPLIT" \
  --out "$SHARD2_OUT" &
PID2=$!

wait "$PID1" "$PID2"

echo "$(date -u +%H:%M:%S) merging shards into $MAIN_OUT/wikiskill-latest.json"
python3 scripts/merge_wikiskill_shards.py \
  --out "$MAIN_OUT/wikiskill-latest.json" \
  "$MAIN_OUT/wikiskill-latest.json" \
  "$SHARD2_OUT/wikiskill-latest.json"

python3 scripts/merge_competitive_upstream.py || true
python3 scripts/inject_paper_results.py || true

echo "=== SealQA parallel done ==="
