#!/usr/bin/env python3
"""Run competitive-bar eval suite (WikiSkill / Reflexion / MemGPT aligned)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rmc.adapters import available_backends, get_adapter
from rmc.bench import bench_adapter, run as run_bench
from rmc.bench import to_dict as bench_to_dict
from rmc.memgpt_bench import run as run_memgpt
from rmc.memgpt_bench import to_dict as memgpt_to_dict
from rmc.session_study import run as run_session
from rmc.session_study import to_dict as session_to_dict
from rmc.wikiskill import CORE_ARMS, run as run_wikiskill
from rmc.wikiskill import to_dict as wikiskill_to_dict

UPSTREAM_DIR = ROOT / "evals" / "upstream"


def main() -> int:
    parser = argparse.ArgumentParser(description="Competitive-bar evaluation suite")
    parser.add_argument("--agent", default="mock")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="cap upstream tasks per split")
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rse" / "results")
    parser.add_argument("--skip-upstream", action="store_true")
    parser.add_argument("--skip-memgpt", action="store_true")
    args = parser.parse_args()

    raw = get_adapter(args.agent)
    if not raw.available():
        print(f"agent {args.agent} unavailable; using mock", file=sys.stderr)
        raw = get_adapter("mock")
    adapter = bench_adapter(raw) if raw.name == "mock" else raw

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest = out / "competitive-latest.json"

    def _flush() -> None:
        latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  (checkpoint → {latest})", flush=True)

    payload: dict = {
        "generated_at": stamp,
        "agent": adapter.name,
        "samples": args.samples,
        "available_backends": available_backends(),
    }

    print("=== RMC-Bench ===", flush=True)
    bench_report = run_bench(adapter, samples=args.samples)
    payload["rmc_bench"] = bench_to_dict(bench_report)
    print(bench_report.render(), flush=True)
    _flush()

    print("\n=== WikiSkill probe (core arms) ===", flush=True)

    def _probe_checkpoint(report) -> None:
        payload["wikiskill_probe"] = wikiskill_to_dict(report)
        _flush()

    ws_probe = run_wikiskill(
        adapter,
        samples=args.samples,
        arms=CORE_ARMS,
        on_progress=_probe_checkpoint,
    )
    payload["wikiskill_probe"] = wikiskill_to_dict(ws_probe)
    print(ws_probe.render(), flush=True)
    _flush()

    if not args.skip_upstream:
        upstream_files = sorted(UPSTREAM_DIR.glob("*.jsonl"))
        payload["upstream"] = {}
        for path in upstream_files:
            print(f"\n=== Upstream: {path.name} ===", flush=True)

            def _upstream_checkpoint(report, stem=path.stem) -> None:
                payload["upstream"][stem] = wikiskill_to_dict(report)
                _flush()

            report = run_wikiskill(
                adapter,
                path=path,
                samples=args.samples,
                limit=args.limit,
                arms=CORE_ARMS + ("trace2skill", "evoskill", "skillopt", "keyword-rag", "oracle-skill"),
                on_progress=_upstream_checkpoint,
            )
            payload["upstream"][path.stem] = wikiskill_to_dict(report)
            print(report.render(), flush=True)
            _flush()

    if not args.skip_memgpt:
        print("\n=== MemGPT nested KV proxy ===", flush=True)
        memgpt_report = run_memgpt(adapter, samples=args.samples)
        payload["memgpt_nested_kv"] = memgpt_to_dict(memgpt_report)
        print(memgpt_report.render(), flush=True)
        _flush()

    print("\n=== Session paired study (Reflexion-style continuity) ===", flush=True)
    session_report = run_session(adapter, samples=args.samples)
    payload["session_study"] = session_to_dict(session_report)
    print(session_report.render(), flush=True)
    _flush()

    latest = out / "competitive-latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / f"competitive-{stamp}.json").write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
