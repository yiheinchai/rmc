#!/usr/bin/env python3
"""Build full SealQA continual-learning bench from HF upstream (111 tasks)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rose import yamlish
from rose.sealqa_evidence import default_lesson, hf_row_to_case

DEFAULT_OUT = ROOT / "evals" / "sealqa-ablation" / "upstream-cl.yaml"


def _assign_splits(cases: list[dict]) -> list[dict]:
    by_axis: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_axis[str(case.get("axis") or "answer-format")].append(case)
    out: list[dict] = []
    for axis, group in sorted(by_axis.items()):
        group.sort(key=lambda c: c["id"])
        for i, case in enumerate(group):
            # Alternate train/test within axis; odd last train-only when count is odd.
            if len(group) == 1:
                case["split"] = "train"
            elif i % 2 == 0:
                case["split"] = "train"
            else:
                case["split"] = "test"
            out.append(case)
    out.sort(key=lambda c: c["id"])
    return out


def build(*, fetch_wikipedia: bool = True) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("pip install datasets") from exc

    ds = load_dataset("vtllms/sealqa", "seal_0", split="test")
    cases = [
        hf_row_to_case(row, idx=i + 1, fetch_wikipedia=fetch_wikipedia)
        for i, row in enumerate(ds)
    ]
    cases = _assign_splits(cases)
    train = sum(1 for c in cases if c.get("split") == "train")
    test = sum(1 for c in cases if c.get("split") == "test")
    qualities: dict[str, int] = defaultdict(int)
    for c in cases:
        qualities[str(c.get("evidence_quality") or "unknown")] += 1
    return {
        "version": 1,
        "benchmark": "SealQA",
        "source": "vtllms/sealqa seal_0 test (111 tasks)",
        "lesson": default_lesson(),
        "stats": {
            "total": len(cases),
            "train": train,
            "test": test,
            "evidence_quality": dict(qualities),
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SealQA upstream CL bench")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-fetch", action="store_true", help="skip Wikipedia enrichment")
    args = parser.parse_args()

    payload = build(fetch_wikipedia=not args.no_fetch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yamlish.dump(payload), encoding="utf-8")
    stats = payload["stats"]
    print(f"Wrote {args.out}")
    print(f"  total={stats['total']} train={stats['train']} test={stats['test']}")
    print(f"  evidence_quality={stats['evidence_quality']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
