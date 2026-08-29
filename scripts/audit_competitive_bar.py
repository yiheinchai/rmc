#!/usr/bin/env python3
"""Verify competitive submission bar artifacts (WikiSkill / Reflexion / MemGPT)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "papers" / "rse" / "results"
FIGURES = ROOT / "papers" / "rse" / "figures"
PAPER = ROOT / "papers" / "rse" / "paper.tex"

EXPECTED_UPSTREAM = {
    "sealqa-test": 111,
    "hotpotqa-dev": 100,
}

BASELINE_ARMS = (
    "no-skill",
    "keyword-rag",
    "trace2skill",
    "evoskill",
    "skillopt",
    "full-inject",
    "recall-judge",
    "recall-agentic",
    "oracle-skill",
)

REQUIRED_FIGURES = (
    "fig_architecture.pdf",
    "fig_case_study.pdf",
    "fig_competitive_baselines.pdf",
    "fig_cross_transfer.pdf",
    "fig_multimodel.pdf",
)


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _upstream_total(comp: dict, stem: str) -> int:
    blob = (comp.get("upstream") or {}).get(stem) or {}
    return ((blob.get("arms") or {}).get("full-inject") or {}).get("total", 0)


def _arms_with_bootstrap(blob: dict) -> list[str]:
    arms = blob.get("arms") or {}
    out = []
    for name, row in arms.items():
        if row.get("bootstrap_ci") and row.get("total", 0) > 0:
            out.append(name)
    return out


def _table_populated(paper_text: str, begin: str, end: str) -> bool:
    start = paper_text.find(begin)
    stop = paper_text.find(end)
    if start < 0 or stop < 0 or stop <= start:
        return False
    block = paper_text[start:stop]
    return "\\begin{table}" in block


def audit() -> list[str]:
    failures: list[str] = []
    comp = _load("competitive-latest.json")
    mm = _load("multimodel-latest.json")
    ct = _load("cross-transfer-latest.json")

    if comp.get("agent") != "codex":
        failures.append("competitive-latest.json: agent must be codex")

    rb = comp.get("rmc_bench") or {}
    transfer = (rb.get("transfer") or {}).get("total", 0)
    if transfer < 20:
        failures.append(f"rmc_bench: expanded transfer cases expected (got {transfer})")

    for stem, expected in EXPECTED_UPSTREAM.items():
        total = _upstream_total(comp, stem)
        if total < expected:
            failures.append(f"upstream {stem}: {total}/{expected} tasks")
        else:
            blob = (comp.get("upstream") or {}).get(stem) or {}
            arms = blob.get("arms") or {}
            missing = [a for a in BASELINE_ARMS if not (arms.get(a) or {}).get("total")]
            if missing:
                failures.append(f"upstream {stem}: missing arms {missing}")
            if not _arms_with_bootstrap(blob):
                failures.append(f"upstream {stem}: no bootstrap_ci on any arm")

    models = mm.get("models") or {}
    if len(models) < 3:
        failures.append(f"multimodel: need >=3 models (got {len(models)})")

    if "codex" not in (ct.get("table") or {}):
        failures.append("cross-transfer: missing codex table")

    for fig in REQUIRED_FIGURES:
        if not (FIGURES / fig).exists():
            failures.append(f"missing figure {fig}")

    if PAPER.exists():
        text = PAPER.read_text(encoding="utf-8")
        for begin, end, label in [
            ("% AUTO:SEALQA_TABLE_BEGIN", "% AUTO:SEALQA_TABLE_END", "SealQA"),
            ("% AUTO:HOTPOT_TABLE_BEGIN", "% AUTO:HOTPOT_TABLE_END", "HotPotQA"),
            ("% AUTO:MULTIMODEL_TABLE_BEGIN", "% AUTO:MULTIMODEL_TABLE_END", "multimodel"),
        ]:
            if not _table_populated(text, begin, end):
                failures.append(f"paper.tex: {label} table not injected")
    else:
        failures.append("paper.tex missing")

    return failures


def main() -> int:
    failures = audit()
    if failures:
        print("Competitive bar: INCOMPLETE", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("Competitive bar: COMPLETE")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
