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

# Wait for the sequential Codex pipeline, then any parallel upstream evals.
wait_for "run_sequential_codex_evals.sh" "sequential pipeline"
wait_for "run_sealqa_parallel.sh" "SealQA parallel orchestrator"
wait_for "run_wikiskill_evals.py --agent codex --bench evals/upstream/sealqa-test.jsonl" "SealQA wikiskill"
wait_for "run_competitive_evals.py.*hotpotqa-dev" "HotPotQA upstream"
wait_for "run_multimodel_evals.py" "multimodel eval"

echo "=== Merge SealQA shards (if parallel) + competitive upstream ==="
SHARD2="papers/rse/results/sealqa-shard2/wikiskill-latest.json"
WIKI="papers/rse/results/wikiskill-latest.json"
if [[ -f "$SHARD2" ]]; then
  python3 scripts/merge_wikiskill_shards.py --out "$WIKI" "$WIKI" "$SHARD2" || true
fi
python3 scripts/merge_competitive_upstream.py || true

echo "=== Running upstream follow-up (HotPotQA + incomplete splits) ==="
bash ./scripts/run_upstream_followup.sh || true

echo "=== Final regeneration ==="
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rse && make

echo "=== Competitive bar audit ==="
python3 scripts/audit_competitive_bar.py || true

echo "=== Follow-up watcher done ==="
