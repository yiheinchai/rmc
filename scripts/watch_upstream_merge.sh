#!/usr/bin/env bash
# Periodically merge partial upstream checkpoints into competitive-latest.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

INTERVAL="${1:-120}"
MAIN="papers/rse/results/competitive-latest.json"
HOTPOT="papers/rse/results/hotpot-workspace/competitive-latest.json"
WIKI="papers/rse/results/wikiskill-latest.json"

echo "=== Upstream merge watcher (every ${INTERVAL}s) ==="

while true; do
  if [[ -f "$WIKI" ]]; then
    python3 scripts/merge_competitive_upstream.py \
      --competitive "$MAIN" \
      --wikiskill "$WIKI" 2>/dev/null || true
  fi
  if [[ -f "$HOTPOT" ]]; then
    python3 scripts/merge_competitive_upstream.py \
      --competitive "$MAIN" \
      --from-competitive "$HOTPOT" \
      --stem hotpotqa-dev 2>/dev/null || true
  fi
  sleep "$INTERVAL"
done
