#!/usr/bin/env bash
# Wait for an in-flight Codex pipeline, then run steps the old script may have skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

LOG="/tmp/pipeline-followup-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Pipeline follow-up watcher ==="
echo "log: $LOG"

wait_for() {
  local pattern="$1"
  local label="$2"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    echo "$(date -u +%H:%M:%S) waiting on $label..."
    sleep 120
  done
  echo "$(date -u +%H:%M:%S) $label finished"
}

# Wait for the sequential Codex pipeline, then merge/finalize.
wait_for "run_sequential_codex_evals.sh" "sequential pipeline"

echo "=== Merge full SealQA from wikiskill-latest into competitive ==="
python3 scripts/merge_competitive_upstream.py || true

echo "=== Running upstream follow-up (HotPotQA + incomplete splits) ==="
bash ./scripts/run_upstream_followup.sh || true

echo "=== Final regeneration ==="
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rse && make

echo "=== Follow-up watcher done ==="
