#!/usr/bin/env bash
# Sequential Codex eval pipeline — avoids parallel API contention.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/codex-sequential-${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Log: $LOG ==="

echo "=== [1/5] Cross-model transfer (Codex) ==="
if [[ -f papers/rse/results/cross-transfer-latest.json ]] && \
   python3 -c "import json; d=json.load(open('papers/rse/results/cross-transfer-latest.json')); exit(0 if 'codex' in d.get('table',{}) else 1)"; then
  echo "skip: cross-transfer-latest.json already has Codex results"
else
  python3 scripts/run_cross_transfer.py --graders codex --samples 3
fi

#!/usr/bin/env bash
# Sequential Codex eval pipeline — avoids parallel API contention.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/codex-sequential-${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Log: $LOG ==="

echo "=== [1/5] Cross-model transfer (Codex) ==="
if [[ -f papers/rse/results/cross-transfer-latest.json ]] && \
   python3 -c "import json; d=json.load(open('papers/rse/results/cross-transfer-latest.json')); exit(0 if 'codex' in d.get('table',{}) else 1)"; then
  echo "skip: cross-transfer-latest.json already has Codex results"
else
  python3 scripts/run_cross_transfer.py --graders codex --samples 3
fi

echo "=== [2/5] Competitive suite (50-task upstream + expanded RMC-Bench) ==="
python3 scripts/run_competitive_evals.py \
  --agent codex \
  --samples 1 \
  --limit 50
# Write incremental snapshot after competitive suite
cp papers/rse/results/competitive-latest.json "papers/rse/results/competitive-codex-${STAMP}.json" 2>/dev/null || true

echo "=== [3/5] Full upstream SealQA (111 tasks, checkpoint) ==="
python3 scripts/run_wikiskill_evals.py \
  --agent codex \
  --bench evals/upstream/sealqa-test.jsonl \
  --samples 1 \
  --checkpoint

echo "=== [4/5] Multi-model WikiSkill probe (Codex) ==="
python3 scripts/run_multimodel_evals.py --agents codex --samples 3

echo "=== [5/5] Regenerate reports and figures ==="
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rse && make

echo "=== Done ==="

echo "=== [3/5] Full upstream SealQA (111 tasks, checkpoint) ==="
python3 scripts/run_wikiskill_evals.py \
  --agent codex \
  --bench evals/upstream/sealqa-test.jsonl \
  --samples 1 \
  --checkpoint

echo "=== [4/5] Multi-model WikiSkill probe (Codex) ==="
python3 scripts/run_multimodel_evals.py --agents codex --samples 3

echo "=== [5/5] Regenerate reports and figures ==="
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rse && make

echo "=== Done ==="
