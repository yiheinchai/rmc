#!/usr/bin/env python3
"""Merge full upstream wikiskill evals into competitive-latest.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "papers" / "rse" / "results"

# Full split sizes (not curated subsets).
EXPECTED_TASKS = {
    "sealqa-test": 111,
    "hotpotqa-dev": 100,
}

STEM_FOR_BENCH = {
    "sealqa": "sealqa-test",
    "hotpotqa": "hotpotqa-dev",
}


def _stem_from_bench_path(path: str) -> str | None:
    text = path.lower()
    for key, stem in STEM_FOR_BENCH.items():
        if key in text:
            return stem
    return None


def _upstream_blob(data: dict) -> dict | None:
    """Normalize a wikiskill result dict into competitive upstream entry."""
    arms = data.get("arms") or {}
    fi = arms.get("full-inject") or {}
    if not fi.get("total"):
        return None
    keep = {
        "agent",
        "bench_path",
        "arms",
        "cases",
        "comparisons",
        "significance_vs_full_inject",
        "render",
        "samples",
        "checkpoint",
    }
    return {k: data[k] for k in keep if k in data}


def merge_wikiskill_into_competitive(
    competitive: Path,
    wikiskill: Path,
    *,
    stem: str | None = None,
) -> bool:
    if not competitive.exists() or not wikiskill.exists():
        return False
    comp = json.loads(competitive.read_text(encoding="utf-8"))
    ws = json.loads(wikiskill.read_text(encoding="utf-8"))
    if comp.get("agent") != "codex" and ws.get("agent") == "codex":
        comp["agent"] = "codex"
    resolved = stem or _stem_from_bench_path(str(ws.get("bench_path", "")))
    if not resolved:
        return False
    blob = _upstream_blob(ws)
    if not blob:
        return False
    comp.setdefault("upstream", {})
    existing = (comp["upstream"].get(resolved) or {}).get("arms", {}).get("full-inject", {})
    new_total = (blob.get("arms") or {}).get("full-inject", {}).get("total", 0)
    old_total = existing.get("total", 0)
    if new_total <= old_total:
        return False
    comp["upstream"][resolved] = blob
    comp["generated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    competitive.write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print(f"merged {resolved}: {old_total} → {new_total} tasks into {competitive}")
    return True


def merge_upstream_from_competitive(
    competitive: Path,
    other: Path,
    *,
    stems: tuple[str, ...] | None = None,
) -> bool:
    """Merge upstream.* sections from another competitive JSON payload."""
    if not competitive.exists() or not other.exists():
        return False
    comp = json.loads(competitive.read_text(encoding="utf-8"))
    src = json.loads(other.read_text(encoding="utf-8"))
    comp.setdefault("upstream", {})
    changed = False
    for stem, blob in (src.get("upstream") or {}).items():
        if stems and stem not in stems:
            continue
        fi = (blob.get("arms") or {}).get("full-inject") or {}
        if not fi.get("total"):
            continue
        existing = (comp["upstream"].get(stem) or {}).get("arms", {}).get("full-inject", {})
        new_total = fi.get("total", 0)
        old_total = existing.get("total", 0)
        if new_total <= old_total:
            continue
        comp["upstream"][stem] = blob
        changed = True
        print(f"merged {stem}: {old_total} → {new_total} tasks into {competitive}")
    if not changed:
        return False
    if comp.get("agent") != "codex" and src.get("agent") == "codex":
        comp["agent"] = "codex"
    comp["generated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    competitive.write_text(json.dumps(comp, indent=2), encoding="utf-8")
    return True


def upstream_needs_run(competitive: Path, stem: str) -> bool:
    expected = EXPECTED_TASKS.get(stem)
    if not expected:
        return False
    if not competitive.exists():
        return True
    comp = json.loads(competitive.read_text(encoding="utf-8"))
    arms = (comp.get("upstream") or {}).get(stem, {}).get("arms") or {}
    fi = arms.get("full-inject") or {}
    return fi.get("total", 0) < expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge upstream evals into competitive-latest.json")
    parser.add_argument("--competitive", type=Path, default=RESULTS / "competitive-latest.json")
    parser.add_argument("--wikiskill", type=Path, default=RESULTS / "wikiskill-latest.json")
    parser.add_argument("--stem", default=None)
    parser.add_argument(
        "--from-competitive",
        type=Path,
        default=None,
        help="merge upstream.* from another competitive-latest.json",
    )
    parser.add_argument("--check", action="store_true", help="print stems needing full upstream eval")
    args = parser.parse_args()

    if args.check:
        for stem in EXPECTED_TASKS:
            if upstream_needs_run(args.competitive, stem):
                print(stem)
        return 0

    if args.from_competitive:
        stems = (args.stem,) if args.stem else None
        if merge_upstream_from_competitive(args.competitive, args.from_competitive, stems=stems):
            return 0
        print("no upstream merge from competitive performed", file=sys.stderr)
        return 1

    if merge_wikiskill_into_competitive(args.competitive, args.wikiskill, stem=args.stem):
        return 0
    print("no merge performed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
