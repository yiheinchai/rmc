#!/usr/bin/env bash
# Run upstream benchmark evals missing from competitive-latest.json (post main pipeline).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

COMPETITIVE="papers/rse/results/competitive-latest.json"
MISSING=()

for stem in hotpotqa-dev; do
  if ! python3 -c "
import json, sys
from pathlib import Path
p = Path('$COMPETITIVE')
if not p.exists():
    sys.exit(1)
d = json.loads(p.read_text())
arms = (d.get('upstream') or {}).get('$stem', {}).get('arms') or {}
fi = arms.get('full-inject') or {}
sys.exit(0 if fi.get('total', 0) else 1)
"; then
    MISSING+=("$stem")
  else
    echo "skip $stem: already in competitive-latest.json"
  fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo "no upstream follow-up needed"
  exit 0
fi

echo "=== Upstream follow-up: ${MISSING[*]} ==="
python3 scripts/run_competitive_evals.py \
  --agent codex \
  --samples 1 \
  --skip-bench \
  --skip-probe \
  --skip-memgpt \
  --skip-session \
  --merge "$COMPETITIVE" \
  --upstream "${MISSING[@]}"

python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rse && make

echo "=== Upstream follow-up done ==="
