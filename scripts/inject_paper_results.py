#!/usr/bin/env python3
"""Inject upstream SealQA result numbers into paper.tex from wikiskill-latest.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "papers" / "rse" / "results"
PAPER = ROOT / "papers" / "rse" / "paper.tex"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}\\%"


def _ci(ci: dict | None) -> str:
    if not ci:
        return ""
    return f" (95\\% CI: {_pct(ci['low'])}--{_pct(ci['high'])})"


def build_sealqa_table(data: dict) -> str:
    arms = data.get("arms") or {}
    order = [
        "no-skill",
        "keyword-rag",
        "trace2skill",
        "evoskill",
        "skillopt",
        "full-inject",
        "recall-judge",
        "recall-agentic",
        "oracle-skill",
    ]
    n = (arms.get("full-inject") or {}).get("total", 0)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{Upstream SealQA ({n} tasks, Codex-graded, 1 sample).}}",
        "  \\label{tab:sealqa_upstream}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        "    Arm & Accuracy & Mean tokens \\\\",
        "    \\midrule",
    ]
    for arm in order:
        row = arms.get(arm)
        if not row or not row.get("total"):
            continue
        acc = row.get("accuracy", 0)
        ci = row.get("bootstrap_ci")
        tok = row.get("mean_tokens", 0)
        label = arm.replace("-", "\\mbox{-}")
        lines.append(
            f"    {label} & {_pct(acc)}{_ci(ci)} & {tok} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def inject(paper_path: Path, table: str) -> bool:
    text = paper_path.read_text(encoding="utf-8")
    marker_start = "% AUTO:SEALQA_TABLE_BEGIN"
    marker_end = "% AUTO:SEALQA_TABLE_END"
    block = f"{marker_start}\n{table}\n{marker_end}"
    if marker_start in text:
        text = re.sub(
            rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}",
            block,
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        anchor = "\\label{fig:competitive}"
        if anchor not in text:
            print("anchor not found in paper.tex", file=sys.stderr)
            return False
        text = text.replace(anchor, f"{anchor}\n\n{block}", 1)
    paper_path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    path = RESULTS / "wikiskill-latest.json"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    if "sealqa" not in str(data.get("bench_path", "")):
        print("wikiskill-latest.json is not upstream SealQA", file=sys.stderr)
        return 1
    arms = data.get("arms") or {}
    if not arms.get("full-inject", {}).get("total", 0):
        print("no upstream SealQA results yet", file=sys.stderr)
        return 1
    table = build_sealqa_table(data)
    if inject(PAPER, table):
        print(f"Updated {PAPER}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
