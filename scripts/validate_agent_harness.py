#!/usr/bin/env python3
"""Cheap preflight for real agent backends before a full experiment run.

Exits 0 only when probe, blind-grade, and one bench case can complete without
harness errors (e.g. invalid JSON schemas, broken subprocess plumbing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rose.adapters import get_adapter
from rose.bench import bench_adapter, load_bench, score_transfer
from rose.evaluate import _grade
from rose.judge import RELEVANCE_SCHEMA
from rose.prompts import BLIND_JUDGE, JUDGE_SCHEMA


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _ok(msg: str) -> None:
    print(f"ok  {msg}")


def validate(agent: str) -> int:
    raw = get_adapter(agent)
    if not raw.available():
        return _fail(f"{agent} backend not available on PATH")

    adapter = bench_adapter(raw) if agent == "mock" else raw
    _ok(f"{agent} adapter available")

    ping = adapter.run("Reply with exactly: PING", timeout=120)
    if not ping.ok or "PING" not in (ping.text or "").upper():
        return _fail(f"basic exec failed: {(ping.error or ping.text or 'no output')[:200]}")
    _ok("basic exec")

    verdict = adapter.run(
        BLIND_JUDGE.format(task="say hello", expected="greets the user", candidate="Hello!"),
        schema=JUDGE_SCHEMA,
        timeout=120,
    )
    if not verdict.ok or not verdict.data:
        err = (verdict.error or verdict.text or "no structured output")[:300]
        if "invalid_json_schema" in err or "grader unavailable" in err.lower():
            return _fail(f"blind judge schema broken: {err}")
        return _fail(f"blind judge failed: {err}")
    if "pass" not in verdict.data:
        return _fail(f"blind judge missing 'pass' key: {verdict.data}")
    _ok("blind judge schema")

    rel = adapter.run(
        "Question: deploy the app\n\nCandidates:\n- n_1: use port 8080\n\n"
        "Which lessons are relevant? Return JSON picks.",
        schema=RELEVANCE_SCHEMA,
        timeout=120,
    )
    if not rel.ok or rel.data is None:
        err = (rel.error or rel.text or "no structured output")[:300]
        if "invalid_json_schema" in err:
            return _fail(f"relevance schema broken: {err}")
        return _fail(f"relevance judge failed: {err}")
    _ok("relevance schema")

    cases, _ = load_bench()
    case = next(c for c in cases if c.kind == "trap")
    row = score_transfer(adapter, case, arm="L0", pack=case.lesson or "", samples=1, timeout=120)
    if "grader unavailable" in (row.reason or "").lower():
        return _fail(f"bench transfer grading broken: {row.reason[:200]}")
    _ok(f"bench case {case.id} (passed={row.passed})")

    print(f"\nHarness ready for full {agent} run.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate agent harness before expensive evals")
    parser.add_argument("--agent", choices=["mock", "claude", "codex"], default="codex")
    args = parser.parse_args()
    return validate(args.agent)


if __name__ == "__main__":
    raise SystemExit(main())
