#!/usr/bin/env python3
"""Run WikiSkill-comparable benchmark evals with RSE recall arms."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rmc.adapters import get_adapter
from rmc.bench import bench_adapter
from rmc.wikiskill import run, to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="WikiSkill-comparable RSE benchmark")
    parser.add_argument("--agent", choices=["mock", "claude", "codex"], default="mock")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rse" / "results")
    args = parser.parse_args()

    raw = get_adapter(args.agent)
    if not raw.available():
        print(f"agent {args.agent} not available; using mock", file=sys.stderr)
        raw = get_adapter("mock")
    adapter = bench_adapter(raw) if args.agent == "mock" else raw

    print(f"Running WikiSkill-comparable bench (agent={adapter.name}, samples={args.samples})...")
    report = run(adapter, samples=args.samples)
    text = report.render()
    print(text)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = to_dict(report)
    payload["generated_at"] = stamp
    payload["samples"] = args.samples

    latest = out / "wikiskill-latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / f"wikiskill-{stamp}.json").write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "wikiskill-latest.txt").write_text(text, encoding="utf-8")
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
