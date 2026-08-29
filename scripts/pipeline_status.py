#!/usr/bin/env python3
"""Print one-screen status for competitive-bar Codex pipelines."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "papers" / "rse" / "results"

EXPECTED = {"sealqa-test": 111, "hotpotqa-dev": 100}


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _running(pattern: str) -> bool:
    """Match python driver processes only (ignore bash wait-loop wrappers)."""
    try:
        proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if proc.returncode != 0:
            return False
        for pid in proc.stdout.strip().split():
            if not pid:
                continue
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode("latin-1").replace("\0", " ")
            except OSError:
                continue
            if "python3" in cmd and pattern in cmd:
                return True
        return False
    except OSError:
        return False


def _running_script(script: str) -> bool:
    """True if a bash driver for ``scripts/<script>`` is active."""
    needle = f"scripts/{script}"
    try:
        proc = subprocess.run(["pgrep", "-f", needle], capture_output=True, text=True)
        if proc.returncode != 0:
            return False
        for pid in proc.stdout.strip().split():
            if not pid:
                continue
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode("latin-1").replace("\0", " ")
            except OSError:
                continue
            if needle in cmd and "python3" not in cmd:
                return True
        return False
    except OSError:
        return False


def _mtime(path: Path) -> str:
    if not path.exists():
        return "—"
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def main() -> int:
    comp = _load("competitive-latest.json")
    ws = _load("wikiskill-latest.json")
    mm = _load("multimodel-latest.json")
    ct = _load("cross-transfer-latest.json")

    print("=== Competitive bar pipeline status ===")
    print(f"competitive-latest.json  agent={comp.get('agent', '—')}  updated={_mtime(RESULTS / 'competitive-latest.json')}")
    if comp.get("rmc_bench"):
        lift = comp["rmc_bench"].get("lift")
        print(f"  rmc_bench lift={lift}")
    for stem, expected in EXPECTED.items():
        blob = (comp.get("upstream") or {}).get(stem) or {}
        total = ((blob.get("arms") or {}).get("full-inject") or {}).get("total", 0)
        print(f"  upstream {stem}: {total}/{expected}")

    ws_total = ((ws.get("arms") or {}).get("full-inject") or {}).get("total", 0)
    ws_cases = len({c.get("case_id") for c in ws.get("cases", []) if c.get("case_id")})
    ws_scores = len(ws.get("cases") or [])
    ws_path = RESULTS / "wikiskill-latest.json"
    print(
        f"wikiskill-latest.json    agent={ws.get('agent', '—')}  "
        f"cases={ws_cases}/111  scores={ws_scores}  ckpt={ws.get('checkpoint', False)}  "
        f"updated={_mtime(ws_path)}"
    )
    print(f"multimodel-latest.json   models={len(mm.get('models') or {})}  updated={_mtime(RESULTS / 'multimodel-latest.json')}")
    print(f"cross-transfer-latest    models={list((ct.get('table') or {}).keys())}")

    jobs = [
        ("sequential pipeline", "run_sequential_codex_evals.sh", "script"),
        ("competitive step 2", "scripts/run_competitive_evals.py --agent codex --samples 1 --skip-upstream", "py"),
        ("SealQA upstream", "scripts/run_wikiskill_evals.py --agent codex", "py"),
        ("multimodel", "scripts/run_multimodel_evals.py", "py"),
        ("HotPotQA parallel", "run_hotpot_parallel.sh", "script"),
    ]
    print("\nRunning:")
    any_running = False
    for label, pattern, kind in jobs:
        active = _running_script(pattern.replace("scripts/", "")) if kind == "script" else _running(pattern)
        if active:
            print(f"  • {label}")
            any_running = True
    if not any_running:
        print("  (none)")

    logs = sorted(Path("/tmp").glob("codex-sequential-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        text = logs[0].read_text(encoding="utf-8")
        transfers = re.findall(r"rmc-bench transfer (\d+)/(\d+)", text)
        upstream_scores = re.findall(r"upstream progress: (\d+) scores", text)
        if "=== [3/6]" in text or _running("scripts/run_wikiskill_evals.py --agent codex"):
            n_cases = ws_cases if ws_cases else ws_total
            n_scores = ws_scores if ws_scores else (int(upstream_scores[-1]) if upstream_scores else 0)
            pct = 100 * n_cases / EXPECTED["sealqa-test"]
            print(
                f"\nStep 3 SealQA upstream: {n_cases}/111 cases "
                f"({pct:.0f}%, {n_scores} arm-scores, target 999)"
            )
        elif "=== WikiSkill probe" in text:
            print("\nStep 2 phase: WikiSkill probe (10-task)")
        elif "=== MemGPT" in text or "memgpt" in text.lower() and "=== [2/6]" not in text:
            print("\nStep 2 phase: MemGPT nested-KV")
        elif "=== Session" in text or "session_study" in text:
            print("\nStep 2 phase: session paired study")
        elif transfers:
            cur, total = transfers[-1]
            pct = 100 * int(cur) / int(total)
            remaining = int(total) - int(cur)
            if int(cur) >= int(total):
                print("\nStep 2 phase: RMC-Bench retrieval/retention (post-transfer)")
            else:
                print(
                    f"\nStep 2 phase: RMC-Bench transfer {cur}/{total} "
                    f"({pct:.0f}%, {remaining} cases left)"
                )
        tail = text.strip().splitlines()[-1]
        print(f"Latest sequential log ({logs[0].name}):")
        print(f"  {tail}")
        log_ts = datetime.fromtimestamp(logs[0].stat().st_mtime, tz=timezone.utc)
        print(f"  (log updated {log_ts.strftime('%H:%M:%S UTC')})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
