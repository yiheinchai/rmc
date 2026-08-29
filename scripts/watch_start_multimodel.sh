#!/usr/bin/env bash
# Start multi-model eval once HotPotQA upstream hits 100 tasks (before SealQA finishes).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

OUT="papers/rse/results/multimodel-latest.json"
HOTPOT_WS="papers/rse/results/hotpot-workspace/competitive-latest.json"
INTERVAL="${1:-60}"

echo "=== Multimodel early-start watcher (poll every ${INTERVAL}s) ==="

ready() {
  [[ -f "$OUT" ]] && python3 -c "
import json, sys
d = json.load(open('$OUT'))
sys.exit(0 if len(d.get('models') or {}) >= 3 else 1)
" 2>/dev/null
}

hotpot_done() {
  [[ -f "$HOTPOT_WS" ]] && python3 -c "
import json, sys
d = json.load(open('$HOTPOT_WS'))
u = (d.get('upstream') or {}).get('hotpotqa-dev') or {}
t = ((u.get('arms') or {}).get('full-inject') or {}).get('total', 0)
sys.exit(0 if t >= 100 else 1)
" 2>/dev/null
}

while true; do
  if ready; then
    echo "$(date -u +%H:%M:%S) multimodel already complete (>=3 models)"
    exit 0
  fi
  if hotpot_done; then
    echo "$(date -u +%H:%M:%S) HotPotQA 100/100 — launching multimodel parallel"
    bash ./scripts/run_multimodel_parallel.sh
    exit $?
  fi
  if ! pgrep -f 'run_hotpot_parallel.sh' >/dev/null 2>&1 && \
     ! pgrep -f 'hotpotqa-dev' >/dev/null 2>&1; then
    hp_total=0
    if [[ -f "$HOTPOT_WS" ]]; then
      hp_total=$(python3 -c "
import json
d=json.load(open('$HOTPOT_WS'))
print(((d.get('upstream') or {}).get('hotpotqa-dev') or {}).get('arms',{}).get('full-inject',{}).get('total',0))
" 2>/dev/null || echo 0)
    fi
    if [[ "$hp_total" -lt 100 ]]; then
      echo "$(date -u +%H:%M:%S) HotPotQA runner exited early ($hp_total/100) — deferring to sequential step 4"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
