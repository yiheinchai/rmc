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

echo "=== [3/6] Full upstream SealQA (111 tasks, checkpoint) ==="
python3 scripts/run_wikiskill_evals.py \
  --agent codex \
  --bench evals/upstream/sealqa-test.jsonl \
  --samples 1 \
  --checkpoint \
  --resume

echo "=== [4/6] Multi-model WikiSkill probe ==="
if python3 -c "from scripts.generate_submission_report import _claude_authenticated; import sys; sys.exit(0 if _claude_authenticated() else 1)"; then
  python3 scripts/run_multimodel_evals.py --agents claude codex --samples 3
else
  mapfile -t _CODEX_MODELS < <(python3 -c "from rmc.grader_specs import extra_codex_models; print('\n'.join(extra_codex_models()))")
  echo "note: claude not authenticated; Codex multi-model with: codex ${_CODEX_MODELS[*]}"
  python3 scripts/run_multimodel_evals.py --agents codex --codex-models "${_CODEX_MODELS[@]}" --samples 3
fi

echo "=== [5/6] Regenerate reports and figures ==="
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rse && make

echo "=== [6/6] Upstream follow-up (HotPotQA etc.) ==="
bash ./scripts/run_upstream_followup.sh

echo "=== Done ==="
