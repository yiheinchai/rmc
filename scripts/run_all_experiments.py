#!/usr/bin/env python3
"""Run the complete ROSE experiment suite and write all results."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rose.experiments import render_summary, run_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all ROSE publication experiments")
    parser.add_argument("--agent", choices=["mock", "claude", "codex"], default="mock")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rose" / "results")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"Running full experiment suite (agent={args.agent}, samples={args.samples})...")
    suite = run_all(agent=args.agent, samples=args.samples)
    text = render_summary(suite)
    print(text)

    payload = suite.to_dict()
    payload["generated_at"] = stamp

    full_path = out / "experiments-full-latest.json"
    full_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / f"experiments-full-{stamp}.json").write_text(
        full_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    txt_path = out / "experiments-full-latest.txt"
    txt_path.write_text(text, encoding="utf-8")

    # Keep summary-latest in sync for paper tooling
    summary = {
        "generated_at": stamp,
        "agent": suite.agent,
        "samples": suite.samples,
        "bench": suite.bench,
        "scaling": suite.scaling,
        "recall": suite.recall,
        "compaction": suite.compaction,
        "walkthrough": suite.walkthrough,
        "retention_curve": suite.retention_curve,
        "wikiskill": suite.wikiskill,
        "session_study": suite.session_study,
        "notes": suite.notes,
    }
    (out / "summary-latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "rose-bench-latest.json").write_text(json.dumps(suite.bench, indent=2), encoding="utf-8")
    (out / "scaling-latest.json").write_text(json.dumps(suite.scaling, indent=2), encoding="utf-8")
    (out / "recall-ablations-latest.json").write_text(
        json.dumps(suite.recall, indent=2), encoding="utf-8"
    )
    (out / "compaction-ablation-latest.json").write_text(
        json.dumps(suite.compaction, indent=2), encoding="utf-8"
    )
    if suite.wikiskill:
        (out / "wikiskill-latest.json").write_text(json.dumps(suite.wikiskill, indent=2), encoding="utf-8")
    if suite.session_study:
        (out / "session-study-latest.json").write_text(
            json.dumps(suite.session_study, indent=2), encoding="utf-8"
        )

    print(f"\nWrote {full_path}")
    print(f"Wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
