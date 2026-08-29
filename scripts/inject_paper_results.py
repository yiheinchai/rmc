#!/usr/bin/env python3
"""Inject upstream SealQA result numbers into paper.tex from wikiskill-latest.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "papers" / "rose" / "results"
PAPER = ROOT / "papers" / "rose" / "paper.tex"

MIN_UPSTREAM_TASKS = {
    "sealqa-test": 111,
    "hotpotqa-dev": 100,
}


def _pct(x: float) -> str:
    return f"{x * 100:.0f}\\%"


def _ci(ci: dict | None) -> str:
    if not ci:
        return ""
    return f" (95\\% CI: {_pct(ci['low'])}--{_pct(ci['high'])})"


def build_upstream_baseline_table(data: dict, *, benchmark: str, label: str) -> str:
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
        f"  \\caption{{Upstream {benchmark} ({n} tasks, Codex-graded, 1 sample).}}",
        f"  \\label{{tab:{label}}}",
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
        arm_label = arm.replace("-", "\\mbox{-}")
        lines.append(
            f"    {arm_label} & {_pct(acc)}{_ci(ci)} & {tok} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def build_multimodel_table(data: dict, *, min_models: int = 3) -> str | None:
    models = data.get("models") or {}
    if len(models) < min_models:
        return None
    samples = data.get("samples", 3)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{Multi-model WikiSkill probe (recall-agentic vs. full-inject, {samples} samples).}}",
        "  \\label{tab:multimodel}",
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    Model & full-inject & recall-agentic & Mean tokens \\\\",
        "    \\midrule",
    ]
    for label in sorted(models):
        arms = models[label].get("arms") or {}
        fi = arms.get("full-inject") or {}
        ra = arms.get("recall-agentic") or {}
        if not ra.get("total"):
            continue
        escaped_label = label.replace("_", r"\_")
        lines.append(
            f"    {escaped_label} & {_pct(fi.get('accuracy', 0))} & "
            f"{_pct(ra.get('accuracy', 0))}{_ci(ra.get('bootstrap_ci'))} & {ra.get('mean_tokens', 0)} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def build_sealqa_table(data: dict) -> str:
    return build_upstream_baseline_table(data, benchmark="SealQA", label="sealqa_upstream")


def build_hotpot_table(data: dict) -> str:
    return build_upstream_baseline_table(data, benchmark="HotPotQA", label="hotpot_upstream")


def build_cross_transfer_table(data: dict, *, model: str = "codex") -> str:
    bench_table = (data.get("table") or {}).get(model) or {}
    model_blob = (data.get("models") or {}).get(model) or {}
    arms = model_blob.get("arms") or {}
    fi = arms.get("full-inject") or {}
    ra = arms.get("recall-agentic") or {}
    samples = data.get("samples", 3)

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{Cross-model transfer on WikiSkill probe ({model}, {samples} samples).}}",
        "  \\label{tab:cross_transfer}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        "    Benchmark & full-inject & recall-agentic \\\\",
        "    \\midrule",
    ]
    for bench in sorted(bench_table):
        row = bench_table[bench]
        fi_acc = row.get("full-inject", 0)
        ra_acc = row.get("recall-agentic", 0)
        ra_cell = f"\\textbf{{{_pct(ra_acc)}}}" if ra_acc >= fi_acc else _pct(ra_acc)
        lines.append(f"    {bench} & {_pct(fi_acc)} & {ra_cell} \\\\")
    ra_ci = ra.get("bootstrap_ci")
    ra_overall = f"\\textbf{{{_pct(ra.get('accuracy', 0))}}}{_ci(ra_ci)}"
    lines += [
        "    \\midrule",
        f"    Overall & {_pct(fi.get('accuracy', 0))} & {ra_overall} \\\\",
        f"    Mean tokens & {fi.get('mean_tokens', 0)} & \\textbf{{{ra.get('mean_tokens', 0)}}} \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def build_rose_bench_rows(rb: dict) -> dict[str, str]:
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
    if lift is not None:
        lift_pct = f"{lift:+.0%}".replace("%", "\\%")
        lift_s = f"\\textbf{{{lift_pct}}}"
    else:
        lift_s = "—"
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


def inject_rose_bench(paper_path: Path, rb: dict) -> bool:
    rows = build_rose_bench_rows(rb)
    block = f"""% AUTO:ROSE_BENCH_TABLE_BEGIN
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
% AUTO:ROSE_BENCH_TABLE_END"""
    text = paper_path.read_text(encoding="utf-8")
    marker_start = "% AUTO:ROSE_BENCH_TABLE_BEGIN"
    marker_end = "% AUTO:ROSE_BENCH_TABLE_END"
    if marker_start in text:
        pattern = rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}"

        def _repl(_match: re.Match[str]) -> str:
            return block

        text = re.sub(pattern, _repl, text, count=1, flags=re.DOTALL)
    else:
        return False
    paper_path.write_text(text, encoding="utf-8")
    return True


def inject_marked_table(paper_path: Path, table: str, marker_start: str, marker_end: str, *, anchor: str = "") -> bool:
    text = paper_path.read_text(encoding="utf-8")
    block = f"{marker_start}\n{table}\n{marker_end}"
    if marker_start in text:
        pattern = rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}"

        def _repl(_match: re.Match[str]) -> str:
            return block

        text = re.sub(pattern, _repl, text, count=1, flags=re.DOTALL)
    elif anchor and anchor in text:
        text = text.replace(anchor, f"{anchor}\n\n{block}", 1)
    else:
        return False
    paper_path.write_text(text, encoding="utf-8")
    return True


def inject_sealqa(paper_path: Path, table: str) -> bool:
    return inject_marked_table(
        paper_path,
        table,
        "% AUTO:SEALQA_TABLE_BEGIN",
        "% AUTO:SEALQA_TABLE_END",
        anchor="\\label{fig:competitive}",
    )


def inject_hotpot(paper_path: Path, table: str) -> bool:
    return inject_marked_table(
        paper_path,
        table,
        "% AUTO:HOTPOT_TABLE_BEGIN",
        "% AUTO:HOTPOT_TABLE_END",
        anchor="\\label{sec:hotpot}",
    )


def inject_cross_transfer(paper_path: Path, table: str) -> bool:
    return inject_marked_table(
        paper_path,
        table,
        "% AUTO:CROSS_TRANSFER_TABLE_BEGIN",
        "% AUTO:CROSS_TRANSFER_TABLE_END",
        anchor="\\label{sec:cross_transfer}",
    )


def inject_multimodel(paper_path: Path, table: str) -> bool:
    return inject_marked_table(
        paper_path,
        table,
        "% AUTO:MULTIMODEL_TABLE_BEGIN",
        "% AUTO:MULTIMODEL_TABLE_END",
        anchor="\\label{tab:wikiskill}",
    )


def _upstream_payload(data: dict, stem: str) -> dict | None:
    upstream = (data.get("upstream") or {}).get(stem)
    if upstream and _upstream_ready(upstream, stem):
        return upstream
    if stem.replace("-", "") in str(data.get("bench_path", "")).replace("-", "").lower():
        arms = data.get("arms") or {}
        if data.get("agent") == "codex" and _upstream_ready(data, stem):
            return data
    return None


def _upstream_ready(blob: dict, stem: str) -> bool:
    arms = blob.get("arms") or {}
    fi = arms.get("full-inject") or {}
    total = fi.get("total", 0)
    minimum = MIN_UPSTREAM_TASKS.get(stem, 1)
    return total >= minimum


def _sealqa_upstream_payload(data: dict) -> dict | None:
    return _upstream_payload(data, "sealqa-test")


def _hotpot_upstream_payload(data: dict) -> dict | None:
    return _upstream_payload(data, "hotpotqa-dev")


def main() -> int:
    updated = False
    comp = RESULTS / "competitive-latest.json"
    if comp.exists():
        data = json.loads(comp.read_text(encoding="utf-8"))
        if data.get("agent") == "codex" and data.get("rose_bench"):
            if inject_rose_bench(PAPER, data["rose_bench"]):
                print(f"Updated ROSE-Bench table in {PAPER}")
                updated = True
        sealqa = _sealqa_upstream_payload(data)
        if sealqa:
            table = build_sealqa_table(sealqa)
            if inject_sealqa(PAPER, table):
                print(f"Updated SealQA table from competitive-latest in {PAPER}")
                updated = True
        hotpot = _hotpot_upstream_payload(data)
        if hotpot:
            table = build_hotpot_table(hotpot)
            if inject_hotpot(PAPER, table):
                print(f"Updated HotPotQA table from competitive-latest in {PAPER}")
                updated = True

    ct_path = RESULTS / "cross-transfer-latest.json"
    if ct_path.exists():
        ct = json.loads(ct_path.read_text(encoding="utf-8"))
        for model in sorted((ct.get("table") or {}).keys()):
            blob = (ct.get("models") or {}).get(model) or {}
            if blob.get("agent") == "codex" or model == "codex":
                table = build_cross_transfer_table(ct, model=model)
                if inject_cross_transfer(PAPER, table):
                    print(f"Updated cross-transfer table ({model}) in {PAPER}")
                    updated = True
                break

    mm_path = RESULTS / "multimodel-latest.json"
    if mm_path.exists():
        mm = json.loads(mm_path.read_text(encoding="utf-8"))
        table = build_multimodel_table(mm)
        if table and inject_multimodel(PAPER, table):
            print(f"Updated multi-model table in {PAPER}")
            updated = True

    path = RESULTS / "wikiskill-latest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        sealqa = _sealqa_upstream_payload(data)
        if sealqa and data.get("agent") == "codex" and not (
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
