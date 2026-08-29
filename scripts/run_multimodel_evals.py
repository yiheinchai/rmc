#!/usr/bin/env python3
"""Run WikiSkill probe across all available agent backends (WikiSkill Table 1 style)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rmc.adapters import available_backends, get_adapter
from rmc.bench import bench_adapter
from rmc.wikiskill import CORE_ARMS, run, to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-model WikiSkill probe")
    parser.add_argument("--agents", nargs="*", default=None, help="default: all available")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rse" / "results")
    args = parser.parse_args()

    agents = args.agents or [a for a in available_backends() if a != "mock"]
    if not agents:
        agents = ["mock"]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: dict[str, dict] = {"generated_at": stamp, "samples": args.samples, "models": {}}

    for name in agents:
        raw = get_adapter(name)
        if not raw.available():
            print(f"skip {name}: not available", file=sys.stderr)
            continue
        adapter = bench_adapter(raw) if name == "mock" else raw
        print(f"\n=== {name} ===")
        report = run(adapter, samples=args.samples, arms=CORE_ARMS)
        print(report.render())
        results["models"][name] = to_dict(report)

    latest = out / "multimodel-latest.json"
    latest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
