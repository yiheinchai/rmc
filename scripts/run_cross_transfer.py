#!/usr/bin/env python3
"""Cross-model skill transfer eval (WikiSkill Table 2)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rmc.adapters import available_backends
from rmc.cross_transfer import run_cross_transfer, to_dict as cross_transfer_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-model skill transfer")
    parser.add_argument("--graders", nargs="*", default=None)
    parser.add_argument("--bench", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rse" / "results")
    args = parser.parse_args()

    graders = args.graders or [g for g in available_backends() if g != "mock"]
    if not graders:
        graders = ["mock"]

    report = run_cross_transfer(graders, bench_path=args.bench, samples=args.samples, limit=args.limit)
    text = report.render()
    print(text)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = cross_transfer_to_dict(report)
    payload["generated_at"] = stamp
    payload["graders"] = graders
    payload["samples"] = args.samples

    latest = out / "cross-transfer-latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / f"cross-transfer-{stamp}.json").write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
