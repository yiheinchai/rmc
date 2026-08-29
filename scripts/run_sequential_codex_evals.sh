#!/usr/bin/env bash
# Sequential Codex eval pipeline — avoids parallel API contention.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "=== [1/4] Cross-model transfer (Codex) ==="
python3 scripts/run_cross_transfer.py --graders codex --samples 3

echo "=== [2/4] SealQA upstream (50 tasks, all arms, samples=1) ==="
python3 scripts/run_wikiskill_evals.py \
  --agent codex \
  --bench evals/upstream/sealqa-test.jsonl \
  --limit 50 \
  --samples 1

echo "=== [3/4] Competitive suite refresh ==="
python3 scripts/run_competitive_evals.py \
  --agent codex \
  --samples 1 \
  --limit 50 \
  --skip-memgpt

echo "=== [4/4] Regenerate reports and figures ==="
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py

echo "=== Done ==="
