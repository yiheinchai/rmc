#!/usr/bin/env bash
# Sequential Codex eval pipeline — avoids parallel API contention.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/codex-sequential-${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Log: $LOG ==="

echo "=== [1/6] Cross-model transfer (Codex) ==="
if [[ -f papers/rse/results/cross-transfer-latest.json ]] && \
   python3 -c "import json; d=json.load(open('papers/rse/results/cross-transfer-latest.json')); exit(0 if 'codex' in d.get('table',{}) else 1)"; then
  echo "skip: cross-transfer-latest.json already has Codex results"
else
  python3 scripts/run_cross_transfer.py --graders codex --samples 3
fi

echo "=== [2/6] Competitive suite (RMC-Bench + probe + MemGPT + session; upstream in steps 3/6) ==="
python3 scripts/run_competitive_evals.py \
  --agent codex \
  --samples 1 \
  --skip-upstream
cp papers/rse/results/competitive-latest.json "papers/rse/results/competitive-codex-${STAMP}.json" 2>/dev/null || true
python3 scripts/inject_paper_results.py || true

python3 scripts/inject_paper_results.py || true

echo "=== [3/6] Full upstream SealQA (111 tasks, checkpoint; exclusive Codex) ==="
python3 scripts/run_wikiskill_evals.py \
  --agent codex \
  --bench evals/upstream/sealqa-test.jsonl \
  --samples 1 \
  --checkpoint \
  --resume

echo "=== Merge SealQA into competitive + inject manuscript ==="
python3 scripts/merge_competitive_upstream.py || true
python3 scripts/inject_paper_results.py || true

echo "=== [4/6] HotPotQA + multi-model (parallel after SealQA) ==="
HOTPOT_LOG="/tmp/hotpot-parallel-${STAMP}.log"
bash ./scripts/run_hotpot_parallel.sh > >(tee -a "$HOTPOT_LOG") 2>&1 &
HOTPOT_PID=$!
echo "HotPotQA background pid=$HOTPOT_PID log=$HOTPOT_LOG"

MULTI_LOG="/tmp/multimodel-parallel-${STAMP}.log"
bash ./scripts/run_multimodel_parallel.sh > >(tee -a "$MULTI_LOG") 2>&1 &
MULTI_PID=$!
echo "Multi-model background pid=$MULTI_PID log=$MULTI_LOG"

echo "=== waiting for HotPotQA (pid=$HOTPOT_PID) and multi-model (pid=$MULTI_PID) ==="
wait "$HOTPOT_PID" || true
if kill -0 "$MULTI_PID" 2>/dev/null; then
  wait "$MULTI_PID" || true
elif [[ ! -f papers/rse/results/multimodel-latest.json ]] || \
     ! python3 -c "import json; d=json.load(open('papers/rse/results/multimodel-latest.json')); exit(0 if len(d.get('models') or {})>=3 else 1)"; then
  bash ./scripts/run_multimodel_parallel.sh
fi

echo "=== [5/6] Regenerate reports and figures ==="
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rse && make

echo "=== [6/6] Upstream follow-up (HotPotQA etc.) ==="
bash ./scripts/run_upstream_followup.sh

echo "=== Competitive bar audit ==="
python3 scripts/audit_competitive_bar.py || true

echo "=== Done ==="
