#!/usr/bin/env python3
"""Merge parallel WikiSkill shard checkpoints into one wikiskill-latest.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rmc.wikiskill import _bench_paths_match, from_checkpoint_dict, merge_reports, to_dict


def merge_shard_paths(paths: list[Path]) -> dict:
    reports = []
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("cases"):
            continue
        reports.append(from_checkpoint_dict(data))
    if not reports:
        raise ValueError("no shard data to merge")
    bench = reports[0].bench_path
    reports = [r for r in reports if _bench_paths_match(r.bench_path, bench)]
    merged = merge_reports(*reports)
    payload = to_dict(merged)
    payload["generated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if any(
        json.loads(p.read_text(encoding="utf-8")).get("checkpoint")
        for p in paths
        if p.exists()
    ):
        payload["checkpoint"] = True
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge WikiSkill parallel shard JSON files")
    parser.add_argument("shards", nargs="+", type=Path, help="wikiskill-latest.json shard paths")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "papers" / "rse" / "results" / "wikiskill-latest.json",
    )
    args = parser.parse_args()

    payload = merge_shard_paths(args.shards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    n_cases = len({c["case_id"] for c in payload.get("cases") or []})
    print(f"merged {len(args.shards)} shards → {args.out} ({n_cases} cases, {len(payload.get('cases') or [])} scores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
