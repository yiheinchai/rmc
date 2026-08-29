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

# Current long-running competitive suite from the 05:01 sequential run.
wait_for "run_competitive_evals.py --agent codex" "competitive suite"
wait_for "run_sequential_codex_evals.sh" "sequential pipeline"

echo "=== Running upstream follow-up (HotPotQA) ==="
bash ./scripts/run_upstream_followup.sh || true

echo "=== Final regeneration ==="
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rse && make

echo "=== Follow-up watcher done ==="
