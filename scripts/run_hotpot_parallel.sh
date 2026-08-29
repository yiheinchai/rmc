#!/usr/bin/env bash
# Run HotPotQA upstream in parallel with SealQA (separate workspace, merge at end).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/hotpot-parallel-${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

MAIN="papers/rse/results/competitive-latest.json"
WORK="papers/rse/results/hotpot-workspace"
STEP2_PATTERN='run_competitive_evals.py --agent codex --samples 1 --skip-upstream'

echo "=== HotPotQA parallel runner ==="
echo "log: $LOG"

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
echo "$(date -u +%H:%M:%S) step 2 complete — injecting Codex results into manuscript"
python3 scripts/inject_paper_results.py || true
echo "$(date -u +%H:%M:%S) starting HotPotQA"

if ! python3 scripts/merge_competitive_upstream.py --check --competitive "$MAIN" | grep -qx hotpotqa-dev; then
  echo "HotPotQA already complete in $MAIN"
  exit 0
fi

mkdir -p "$WORK"
python3 scripts/run_competitive_evals.py \
  --agent codex \
  --samples 1 \
  --skip-bench \
  --skip-probe \
  --skip-memgpt \
  --skip-session \
  --merge "$MAIN" \
  --out "$WORK" \
  --upstream hotpotqa-dev

python3 scripts/merge_competitive_upstream.py \
  --competitive "$MAIN" \
  --from-competitive "$WORK/competitive-latest.json" \
  --stem hotpotqa-dev

echo "=== HotPotQA parallel done ==="
