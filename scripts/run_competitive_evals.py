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
UPSTREAM_ARMS = CORE_ARMS + ("trace2skill", "evoskill", "skillopt", "keyword-rag", "oracle-skill")


def _load_payload(merge: Path | None, *, stamp: str, adapter, samples: int) -> dict:
    if merge and merge.exists():
        payload = json.loads(merge.read_text(encoding="utf-8"))
        payload["generated_at"] = stamp
        payload["agent"] = adapter.name
        payload["samples"] = samples
        payload["available_backends"] = available_backends()
        return payload
    return {
        "generated_at": stamp,
        "agent": adapter.name,
        "samples": samples,
        "available_backends": available_backends(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Competitive-bar evaluation suite")
    parser.add_argument("--agent", default="mock")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="cap upstream tasks per split")
    parser.add_argument("--out", type=Path, default=ROOT / "papers" / "rse" / "results")
    parser.add_argument("--skip-upstream", action="store_true")
    parser.add_argument("--skip-memgpt", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-session", action="store_true")
    parser.add_argument(
        "--upstream",
        nargs="*",
        default=None,
        help="only run these upstream JSONL stems (e.g. hotpotqa-dev)",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        help="merge into an existing competitive-latest.json (keeps skipped sections)",
    )
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
    merge_path = args.merge

    payload = _load_payload(merge_path, stamp=stamp, adapter=adapter, samples=args.samples)

    def _preserve_upstream_from_disk() -> None:
        """Keep upstream splits merged by parallel jobs when this run re-flushes."""
        if not latest.exists():
            return
        try:
            on_disk = json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        disk_up = on_disk.get("upstream") or {}
        if not disk_up:
            return
        payload.setdefault("upstream", {})
        for stem, blob in disk_up.items():
            disk_total = (
                (blob.get("arms") or {}).get("full-inject") or {}
            ).get("total", 0)
            pay_total = (
                (payload["upstream"].get(stem) or {}).get("arms", {}).get("full-inject") or {}
            ).get("total", 0)
            if disk_total > pay_total:
                payload["upstream"][stem] = blob

    def _flush() -> None:
        _preserve_upstream_from_disk()
        latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  (checkpoint → {latest})", flush=True)

    if not args.skip_bench:
        print("=== RMC-Bench ===", flush=True)
        bench_report = run_bench(adapter, samples=args.samples)
        payload["rmc_bench"] = bench_to_dict(bench_report)
        print(bench_report.render(), flush=True)
        _flush()
    elif payload.get("rmc_bench"):
        print("=== RMC-Bench === (skipped, kept from merge)", flush=True)

    if not args.skip_probe:
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
    elif payload.get("wikiskill_probe"):
        print("\n=== WikiSkill probe === (skipped, kept from merge)", flush=True)

    if not args.skip_upstream:
        upstream_files = sorted(UPSTREAM_DIR.glob("*.jsonl"))
        if args.upstream:
            allowed = set(args.upstream)
            upstream_files = [p for p in upstream_files if p.stem in allowed]
        payload.setdefault("upstream", {})
        for path in upstream_files:
            print(f"\n=== Upstream: {path.name} ===", flush=True)

            def _upstream_checkpoint(report, stem=path.stem) -> None:
                payload["upstream"][stem] = wikiskill_to_dict(report)
                n_cases = len({c.case_id for c in report.cases})
                print(
                    f"  upstream progress: {len(report.cases)} scores ({n_cases} cases)",
                    flush=True,
                )
                _flush()

            report = run_wikiskill(
                adapter,
                path=path,
                samples=args.samples,
                limit=args.limit,
                arms=UPSTREAM_ARMS,
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
    elif payload.get("memgpt_nested_kv"):
        print("\n=== MemGPT nested KV proxy === (skipped, kept from merge)", flush=True)

    if not args.skip_session:
        print("\n=== Session paired study (Reflexion-style continuity) ===", flush=True)
        session_report = run_session(adapter, samples=args.samples)
        payload["session_study"] = session_to_dict(session_report)
        print(session_report.render(), flush=True)
        _flush()
    elif payload.get("session_study"):
        print("\n=== Session paired study === (skipped, kept from merge)", flush=True)

    latest = out / "competitive-latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / f"competitive-{stamp}.json").write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
