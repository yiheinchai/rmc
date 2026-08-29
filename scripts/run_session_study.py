#!/usr/bin/env python3
"""Run session-length paired study (EXPERIMENTS §7 proxy)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rose.adapters import get_adapter
from rose.bench import bench_adapter
from rose.session_study import run, to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Session-length paired study")
    parser.add_argument("--agent", choices=["mock", "claude", "codex"], default="mock")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rose" / "results")
    args = parser.parse_args()

    raw = get_adapter(args.agent)
    if not raw.available():
        print(f"agent {args.agent} not available; using mock", file=sys.stderr)
        raw = get_adapter("mock")
    adapter = bench_adapter(raw) if args.agent == "mock" else raw

    print(f"Running session paired study (agent={adapter.name}, samples={args.samples})...")
    report = run(adapter, samples=args.samples)
    text = report.render()
    print(text)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = to_dict(report)
    payload["generated_at"] = stamp
    payload["samples"] = args.samples

    latest = out / "session-study-latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
