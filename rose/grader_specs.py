"""Parse grader specs like ``codex:gpt-5.6-sol`` for multi-model evals."""

from __future__ import annotations

import os
import shutil
import subprocess


def parse_grader_spec(spec: str) -> tuple[str, str | None, str]:
    """Return (backend, model, label) for a grader CLI token."""
    text = (spec or "").strip()
    if ":" in text:
        backend, model = text.split(":", 1)
        backend = backend.strip().lower()
        model = model.strip() or None
        return backend, model, text
    return text.lower(), None, text


def _claude_authenticated() -> bool:
    if not shutil.which("claude"):
        return False
    try:
        proc = subprocess.run(
            ["claude", "-p", "ping"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        combined = (proc.stdout + proc.stderr).lower()
        return "not logged in" not in combined
    except (OSError, subprocess.TimeoutExpired):
        return False


def default_multimodel_specs(*, min_models: int = 3) -> list[str]:
    """Grader specs for WikiSkill Table 1/2 multi-model bar (≥3 when possible)."""
    specs: list[str] = []
    if _claude_authenticated():
        specs.append("claude")
    specs.append("codex")
    extra = [m.strip() for m in os.environ.get("CODEX_MULTI_MODELS", "gpt-5.6-sol").split() if m.strip()]
    for model in extra:
        token = f"codex:{model}"
        if token not in specs:
            specs.append(token)
    if len(specs) < min_models:
        fallback = [m.strip() for m in os.environ.get("CODEX_MULTI_MODELS_FALLBACK", "o4-mini").split() if m.strip()]
        for model in fallback:
            token = f"codex:{model}"
            if token not in specs:
                specs.append(token)
            if len(specs) >= min_models:
                break
    return specs


def extra_codex_models() -> list[str]:
    """Codex -m overrides from default_multimodel_specs (excludes bare ``codex``)."""
    out: list[str] = []
    for spec in default_multimodel_specs():
        backend, model, _ = parse_grader_spec(spec)
        if backend == "codex" and model and model not in out:
            out.append(model)
    return out
