#!/usr/bin/env bash
# Run upstream benchmark evals missing from competitive-latest.json (post main pipeline).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

COMPETITIVE="papers/rose/results/competitive-latest.json"
WIKISKILL="papers/rose/results/wikiskill-latest.json"

echo "=== Merge full SealQA from wikiskill-latest (if step 3 completed) ==="
python3 scripts/merge_competitive_upstream.py \
  --competitive "$COMPETITIVE" \
  --wikiskill "$WIKISKILL" || true

HOTPOT_WS="papers/rose/results/hotpot-workspace/competitive-latest.json"
HOTPOT2="papers/rose/results/hotpot-shard2/competitive-latest.json"
if [[ -f "$HOTPOT_WS" && -f "$HOTPOT2" ]]; then
  python3 scripts/merge_competitive_upstream.py \
    --competitive "$HOTPOT_WS" \
    --merge-shards "$HOTPOT_WS" "$HOTPOT2" \
    --stem hotpotqa-dev || true
  python3 scripts/merge_competitive_upstream.py \
    --competitive "$COMPETITIVE" \
    --from-competitive "$HOTPOT_WS" \
    --stem hotpotqa-dev || true
fi

mapfile -t MISSING < <(python3 scripts/merge_competitive_upstream.py --check --competitive "$COMPETITIVE")

if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo "no upstream follow-up needed (all full splits present)"
  exit 0
fi

echo "=== Upstream follow-up needed: ${MISSING[*]} ==="
python3 scripts/run_competitive_evals.py \
  --agent codex \
  --samples 1 \
  --skip-bench \
  --skip-probe \
  --skip-memgpt \
  --skip-session \
  --merge "$COMPETITIVE" \
  --resume \
  --upstream "${MISSING[@]}"

python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
python3 scripts/inject_paper_results.py || true
cd papers/rose && make

echo "=== Upstream follow-up done ==="
