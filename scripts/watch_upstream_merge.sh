#!/usr/bin/env bash
# Periodically merge partial upstream checkpoints into competitive-latest.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

INTERVAL="${1:-60}"
MAIN="papers/rse/results/competitive-latest.json"
HOTPOT="papers/rse/results/hotpot-workspace/competitive-latest.json"
WIKI="papers/rse/results/wikiskill-latest.json"
SHARD2="papers/rse/results/sealqa-shard2/wikiskill-latest.json"

echo "=== Upstream merge watcher (every ${INTERVAL}s) ==="

while true; do
  ts=$(date -u +%H:%M:%S)
  merged=false
  if [[ -f "$SHARD2" ]]; then
    if python3 scripts/merge_wikiskill_shards.py --out "$WIKI" "$WIKI" "$SHARD2" 2>&1; then
      merged=true
    fi
  fi
  if [[ -f "$WIKI" ]]; then
    if python3 scripts/merge_competitive_upstream.py \
      --competitive "$MAIN" \
      --wikiskill "$WIKI" 2>&1; then
      merged=true
    fi
  fi
  if [[ -f "$HOTPOT" ]]; then
    if python3 scripts/merge_competitive_upstream.py \
      --competitive "$MAIN" \
      --from-competitive "$HOTPOT" \
      --stem hotpotqa-dev 2>&1; then
      merged=true
    fi
  fi
  if [[ "$merged" == false ]]; then
    echo "$ts merge tick (no change)"
  fi
  sleep "$INTERVAL"
done
