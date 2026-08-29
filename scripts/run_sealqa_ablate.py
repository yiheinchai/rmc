#!/usr/bin/env python3
"""Run SealQA ROSE feature ablations (probe replay, compaction presets)."""

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
from rose.sealqa_ablate import PRESETS, run


def main() -> int:
    parser = argparse.ArgumentParser(description="SealQA ROSE ablation harness")
    parser.add_argument("--agent", choices=["mock", "claude", "codex"], default="mock")
    parser.add_argument(
        "--bench",
        type=Path,
        default=ROOT / "evals" / "sealqa-ablation" / "probe-dev.yaml",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help=f"subset of presets (default: all). Choices: {', '.join(PRESETS)}",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rose" / "results")
    parser.add_argument("--no-task-score", action="store_true")
    args = parser.parse_args()

    raw = get_adapter(args.agent)
    if not raw.available():
        print(f"agent {args.agent} not available; using mock", file=sys.stderr)
        raw = get_adapter("mock")
    adapter = bench_adapter(raw) if args.agent == "mock" else raw

    presets = tuple(args.presets) if args.presets else None
    report = run(
        adapter,
        path=args.bench,
        presets=presets,
        score_tasks=not args.no_task_score,
    )
    text = report.render()
    print(text)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = report.to_dict()
    payload["generated_at"] = stamp
    payload["samples"] = 1

    latest = args.out / "sealqa-ablation-latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / f"sealqa-ablation-{stamp}.json").write_text(latest.read_text(encoding="utf-8"))
    (args.out / "sealqa-ablation-latest.txt").write_text(text, encoding="utf-8")
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
