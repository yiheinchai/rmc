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


def build_rmc_bench_rows(rb: dict) -> dict[str, str]:
    transfer = rb.get("transfer") or {}
    retrieval = rb.get("retrieval") or {}
    cases = rb.get("cases") or []
    l0_toks = [c["tokens"] for c in cases if c.get("arm") == "L0" and c.get("tokens")]
    mean_tok = sum(l0_toks) // len(l0_toks) if l0_toks else 0

    by_kind: dict[str, tuple[int, int]] = {}
    for kind in ("detail", "trap", "multi", "principle"):
        rows = [c for c in cases if c.get("kind") == kind and c.get("arm") == "L0"]
        if rows:
            by_kind[kind] = (sum(1 for c in rows if c.get("passed")), len(rows))

    def _fmt(kind: str) -> str:
        p, t = by_kind.get(kind, (0, 0))
        return f"{p}/{t} ({p * 100 // t if t else 0}\\%)" if t else "—"

    lift = rb.get("lift")
    lift_s = f"\\textbf{{{lift:+.0%}}}" if lift is not None else "—"
    tr_p, tr_t = transfer.get("passed", 0), transfer.get("total", 0)
    tr_s = f"\\textbf{{{tr_p}/{tr_t} ({tr_p * 100 // tr_t if tr_t else 0}\\%)}}"
    ret_p, ret_t = retrieval.get("passed", 0), retrieval.get("total", 0)
    ret_s = f"\\textbf{{{ret_p}/{ret_t} ({ret_p * 100 // ret_t if ret_t else 0}\\%)}}"

    return {
        "lift": lift_s,
        "transfer": tr_s,
        "detail": _fmt("detail"),
        "trap": _fmt("trap"),
        "multi": _fmt("multi"),
        "principle": _fmt("principle"),
        "mean_tokens": f"\\textbf{{{mean_tok}}}",
        "retrieval": ret_s,
    }


def inject_rmc_bench(paper_path: Path, rb: dict) -> bool:
    rows = build_rmc_bench_rows(rb)
    block = f"""% AUTO:RMC_BENCH_TABLE_BEGIN
  \\begin{{tabular}}{{lc}}
    \\toprule
    Metric & Result \\\\
    \\midrule
    Lift (L0 $-$ control, core kinds) & {rows['lift']} \\\\
    Transfer @ L0 & {rows['transfer']} \\\\
    Detail / trap / multi transfer & {rows['detail']}, {rows['trap']}, {rows['multi']} \\\\
    Principle transfer & {rows['principle']} \\\\
    Mean L0 tokens & {rows['mean_tokens']} \\\\
    Bench retrieval axis & {rows['retrieval']} \\\\
    \\bottomrule
  \\end{{tabular}}
% AUTO:RMC_BENCH_TABLE_END"""
    text = paper_path.read_text(encoding="utf-8")
    marker_start = "% AUTO:RMC_BENCH_TABLE_BEGIN"
    marker_end = "% AUTO:RMC_BENCH_TABLE_END"
    if marker_start in text:
        text = re.sub(
            rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}",
            block,
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        return False
    paper_path.write_text(text, encoding="utf-8")
    return True


def inject_sealqa(paper_path: Path, table: str) -> bool:
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
            return False
        text = text.replace(anchor, f"{anchor}\n\n{block}", 1)
    paper_path.write_text(text, encoding="utf-8")
    return True


def _sealqa_upstream_payload(data: dict) -> dict | None:
    """Extract SealQA upstream arms from competitive or wikiskill result blobs."""
    upstream = (data.get("upstream") or {}).get("sealqa-test")
    if upstream and (upstream.get("arms") or {}).get("full-inject", {}).get("total", 0):
        return upstream
    if "sealqa" in str(data.get("bench_path", "")):
        arms = data.get("arms") or {}
        if arms.get("full-inject", {}).get("total", 0):
            return data
    return None


def main() -> int:
    updated = False
    comp = RESULTS / "competitive-latest.json"
    if comp.exists():
        data = json.loads(comp.read_text(encoding="utf-8"))
        if data.get("agent") == "codex" and data.get("rmc_bench"):
            if inject_rmc_bench(PAPER, data["rmc_bench"]):
                print(f"Updated RMC-Bench table in {PAPER}")
                updated = True
        sealqa = _sealqa_upstream_payload(data)
        if sealqa:
            table = build_sealqa_table(sealqa)
            if inject_sealqa(PAPER, table):
                print(f"Updated SealQA table from competitive-latest in {PAPER}")
                updated = True

    path = RESULTS / "wikiskill-latest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        sealqa = _sealqa_upstream_payload(data)
        if sealqa and not (
            comp.exists()
            and _sealqa_upstream_payload(json.loads(comp.read_text(encoding="utf-8")))
        ):
            table = build_sealqa_table(sealqa)
            if inject_sealqa(PAPER, table):
                print(f"Updated SealQA table from wikiskill-latest in {PAPER}")
                updated = True

    if not updated:
        print("no Codex results to inject yet", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
