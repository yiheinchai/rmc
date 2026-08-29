#!/usr/bin/env bash
# Run multi-model WikiSkill probe while the main sequential pipeline is busy.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/multimodel-parallel-${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

OUT="papers/rose/results/multimodel-latest.json"
LOCK="/tmp/multimodel-parallel.lock"

echo "=== Multi-model parallel runner ==="
echo "log: $LOG"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another multi-model parallel runner is active — exiting"
  exit 0
fi

if [[ -f "$OUT" ]]; then
  n=$(python3 -c "import json; d=json.load(open('$OUT')); print(len(d.get('models') or {}))")
  if [[ "$n" -ge 3 ]]; then
    echo "skip: $OUT already has $n models"
    exit 0
  fi
fi

if python3 -c "from scripts.generate_submission_report import _claude_authenticated; import sys; sys.exit(0 if _claude_authenticated() else 1)"; then
  python3 scripts/run_multimodel_evals.py --agents claude codex --samples 3
else
  mapfile -t _CODEX_MODELS < <(python3 -c "from rose.grader_specs import extra_codex_models; print('\n'.join(extra_codex_models()))")
  echo "note: claude not authenticated; Codex multi-model with: codex ${_CODEX_MODELS[*]}"
  python3 scripts/run_multimodel_evals.py --agents codex --codex-models "${_CODEX_MODELS[@]}" --samples 3 --resume
fi

echo "=== Multi-model parallel done ==="
