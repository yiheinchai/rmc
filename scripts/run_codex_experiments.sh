#!/usr/bin/env bash
# Authenticate Codex CLI and run the full ROSE experiment suite.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v codex >/dev/null 2>&1; then
  echo "codex not found; installing to ~/.local ..."
  npm install -g @openai/codex --prefix "${HOME}/.local"
fi

echo "Codex: $(codex --version)"

# Auth: prefer ChatGPT session token, then API key.
if [[ -n "${CODEX_ACCESS_TOKEN:-}" ]]; then
  echo "Logging in with CODEX_ACCESS_TOKEN ..."
  printf '%s' "${CODEX_ACCESS_TOKEN}" | codex login --with-access-token
elif [[ -n "${CODEX_API_KEY:-}" ]]; then
  echo "Logging in with CODEX_API_KEY ..."
  printf '%s' "${CODEX_API_KEY}" | codex login --with-api-key
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "Logging in with OPENAI_API_KEY ..."
  printf '%s' "${OPENAI_API_KEY}" | codex login --with-api-key
else
  echo "No Codex credentials in environment."
  echo "Set one of: CODEX_ACCESS_TOKEN, CODEX_API_KEY, OPENAI_API_KEY"
  codex login status || true
  exit 1
fi

codex login status

echo "Harness validation (cheap; ~4 Codex calls) ..."
cd "${ROOT}"
python3 scripts/validate_agent_harness.py --agent codex

SAMPLES="${1:-3}"
echo "Running full experiment suite (samples=${SAMPLES}) ..."
python3 scripts/run_all_experiments.py --agent codex --samples "${SAMPLES}"
