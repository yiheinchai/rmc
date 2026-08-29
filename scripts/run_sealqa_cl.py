#!/usr/bin/env python3
"""Run SealQA continual-learning eval (train → compact → test)."""

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
from rose.sealqa_cl import DEFAULT_BENCH, run


def main() -> int:
    parser = argparse.ArgumentParser(description="SealQA continual-learning harness")
    parser.add_argument("--agent", choices=["mock", "claude", "codex"], default="mock")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rose" / "results")
    args = parser.parse_args()

    raw = get_adapter(args.agent)
    if not raw.available():
        print(f"agent {args.agent} not available; using mock", file=sys.stderr)
        raw = get_adapter("mock")
    adapter = bench_adapter(raw) if args.agent == "mock" else raw

    print(
        f"Running SealQA continual learning (agent={adapter.name}, bench={args.bench.name})...",
        flush=True,
    )
    report = run(
        adapter,
        path=args.bench,
        timeout=args.timeout,
        compact=not args.no_compact,
    )
    text = report.render()
    print(text)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = report.to_dict()
    payload["generated_at"] = stamp

    latest = args.out / "sealqa-cl-latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / f"sealqa-cl-{stamp}.json").write_text(latest.read_text(encoding="utf-8"))
    txt = args.out / "sealqa-cl-latest.txt"
    if not txt.exists() or txt.parent.name:
        try:
            txt.write_text(text, encoding="utf-8")
        except OSError:
            pass
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
