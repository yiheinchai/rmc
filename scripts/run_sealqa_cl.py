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
from rose.sealqa_cl import (
    DEFAULT_BENCH,
    UPSTREAM_BENCH,
    from_checkpoint_dict,
    run,
    scored_test_keys,
)

DEFAULT_OUT = ROOT / "papers" / "rose" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="SealQA continual-learning harness")
    parser.add_argument("--agent", choices=["mock", "claude", "codex"], default="mock")
    parser.add_argument(
        "--bench",
        type=Path,
        default=UPSTREAM_BENCH,
        help=f"YAML bench (default: upstream 111-task {UPSTREAM_BENCH.name})",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="cap train cases (debug)")
    parser.add_argument("--checkpoint", action="store_true", help="write partial JSON after each step")
    parser.add_argument("--resume", action="store_true", help="resume from out/sealqa-cl-latest.json")
    args = parser.parse_args()

    raw = get_adapter(args.agent)
    if not raw.available():
        print(f"agent {args.agent} not available; using mock", file=sys.stderr)
        raw = get_adapter("mock")
    adapter = bench_adapter(raw) if args.agent == "mock" else raw

    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "sealqa-cl-latest.json"
    store_base = args.out / "sealqa-cl-store"
    existing = None
    if args.resume and latest.exists():
        existing = from_checkpoint_dict(json.loads(latest.read_text(encoding="utf-8")))
        n_train = len(existing.train_steps)
        n_test = len(scored_test_keys(existing))
        print(f"Resuming: {n_train} train steps, {n_test} test scores", flush=True)

    def _write_checkpoint(report) -> None:
        payload = report.to_dict()
        payload["generated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload["checkpoint"] = True
        latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n_train = len(report.train_steps)
        n_test = len({(s.case_id, s.arm) for s in report.test_scores})
        print(
            f"  progress: train={n_train} test_scores={n_test}",
            flush=True,
        )

    on_progress = _write_checkpoint if args.checkpoint else None

    print(
        f"Running SealQA continual learning (agent={adapter.name}, bench={args.bench.name})...",
        flush=True,
    )
    report = run(
        adapter,
        path=args.bench,
        timeout=args.timeout,
        compact=not args.no_compact,
        store_base=store_base,
        existing=existing,
        train_limit=args.limit,
        on_progress=on_progress,
    )
    text = report.render()
    print(text)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = report.to_dict()
    payload["generated_at"] = stamp
    payload.pop("checkpoint", None)

    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / f"sealqa-cl-{stamp}.json").write_text(latest.read_text(encoding="utf-8"))
    try:
        (args.out / "sealqa-cl-latest.txt").write_text(text, encoding="utf-8")
    except OSError:
        pass
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
