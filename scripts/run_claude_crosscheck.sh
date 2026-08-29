#!/usr/bin/env bash
# Run Claude-graded RMC-Bench cross-check (paper §4.7).
# Requires: `claude` on PATH and authenticated (`claude` then /login).

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v claude >/dev/null 2>&1; then
  echo "claude not on PATH. Install: npm install @anthropic-ai/claude-code --prefix ~/.local" >&2
  echo "Then: ln -sf ~/.local/node_modules/@anthropic-ai/claude-code-linux-x64/claude ~/.local/bin/claude" >&2
  exit 1
fi

if claude -p "ping" --no-session-persistence 2>&1 | grep -qi "not logged in"; then
  echo "claude is not authenticated. Run: claude  then  /login" >&2
  exit 1
fi

echo "Running harness preflight..."
python3 scripts/validate_agent_harness.py --agent claude

echo "Running RMC-Bench (claude, 3 samples)..."
python3 -m rmc.cli bench --agent claude --samples 3

echo "Updating submission report..."
python3 scripts/generate_submission_report.py

echo "Done. Compare papers/rse/results/rmc-bench-latest.json (codex) vs new claude output."
