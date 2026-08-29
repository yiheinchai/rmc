#!/usr/bin/env python3
"""Run the ROSE publication evaluation suite and write results to papers/rose/results/.

Prefer scripts/run_all_experiments.py for the complete suite (recall ablations,
compaction, walkthrough, retention curve). This script runs bench + scaling only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rose.adapters import get_adapter
from rose.bench import run as run_bench, to_dict as bench_to_dict
from rose.scaling import render_table, run_scaling


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ROSE paper evaluation suite")
    parser.add_argument("--full", action="store_true", help="run complete suite via run_all_experiments")
    parser.add_argument("--agent", choices=["mock", "claude", "codex"], default="mock")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rose" / "results")
    args = parser.parse_args()

    if args.full:
        from scripts.run_all_experiments import main as run_all_main
        sys.argv = ["run_all_experiments", "--agent", args.agent, "--samples", str(args.samples)]
        return run_all_main()

    adapter = get_adapter(args.agent)
    if not adapter.available():
        print(f"agent {args.agent!r} is not available; falling back to mock")
        adapter = get_adapter("mock")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    bench_report = run_bench(adapter, samples=args.samples)
    bench_path = out / f"rose-bench-{args.agent}-{stamp}.json"
    bench_path.write_text(json.dumps(bench_to_dict(bench_report), indent=2), encoding="utf-8")
    (out / "rose-bench-latest.json").write_text(bench_path.read_text(encoding="utf-8"), encoding="utf-8")
    bench_txt = out / f"rose-bench-{args.agent}-{stamp}.txt"
    bench_txt.write_text(bench_report.render(), encoding="utf-8")

    scaling_rows = run_scaling([25, 100, 500])
    scaling_data = {
        "agent": args.agent,
        "rows": [r.to_dict() for r in scaling_rows],
        "table": render_table(scaling_rows),
    }
    scaling_path = out / f"scaling-{args.agent}-{stamp}.json"
    scaling_path.write_text(json.dumps(scaling_data, indent=2), encoding="utf-8")
    (out / "scaling-latest.json").write_text(scaling_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "generated_at": stamp,
        "agent": args.agent,
        "bench": bench_to_dict(bench_report),
        "scaling": scaling_data,
        "artifacts": {
            "bench_json": str(bench_path.relative_to(ROOT)),
            "bench_txt": str(bench_txt.relative_to(ROOT)),
            "scaling_json": str(scaling_path.relative_to(ROOT)),
        },
    }
    summary_path = out / "summary-latest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(bench_report.render())
    print()
    print("Scaling")
    print(scaling_data["table"])
    print()
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
