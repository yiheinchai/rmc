"""WikiSkill-comparable benchmark runner for RSE.

Loads YAML or JSONL benchmark files and scores multiple inference arms:

* **no-skill** — bare task (WikiSkill "No skill" baseline)
* **full-inject** — all store lessons concatenated (WikiSkill test-time injection)
* **trace2skill** — single best keyword-matched skill (Trace2Skill proxy)
* **evoskill** — top-2 keyword-matched skills (EvoSkill proxy)
* **skillopt** — optimized single-skill header inject (SkillOpt proxy)
* **keyword-rag** — MemGPT/RAG-style top-k lexical retrieval
* **recall-judge** — RSE recall with judge-walk selector
* **recall-agentic** — RSE recall with agentic selector
* **oracle-skill** — ground-truth task skill (upper bound)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import index as index_mod
from . import yamlish
from .adapters import Adapter
from .adapters.mock import MockAdapter, _candidates, _section
from .bench import bench_adapter, mock_grade, score_transfer
from .evaluate import CONTROL
from .node import Node
from .prompts import JUDGE_SCHEMA
from .recall import recall_pack
from .skill_baselines import (
    evoskill_pack,
    keyword_rag_pack,
    oracle_skill_pack,
    skillopt_pack,
    trace2skill_pack,
)
from .stats import bootstrap_ci, paired_bootstrap_test
from .store import Store
from .util import count_tokens, utcnow

DEFAULT_BENCH = Path(__file__).resolve().parents[1] / "evals" / "wikiskill-bench.yaml"

CORE_ARMS = ("no-skill", "full-inject", "recall-judge", "recall-agentic")
BASELINE_ARMS = ("trace2skill", "evoskill", "skillopt", "keyword-rag", "oracle-skill")
ARMS = CORE_ARMS + BASELINE_ARMS


@dataclass
class WikiSkillCase:
    id: str
    benchmark: str
    task: str
    expected: str
    skill: str
    family: str = "default"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WikiSkillCase":
        return cls(
            id=str(raw.get("id") or ""),
            benchmark=str(raw.get("benchmark") or ""),
            task=str(raw.get("task") or "").strip(),
            expected=str(raw.get("expected") or "").strip(),
            skill=str(raw.get("skill") or "").strip(),
            family=str(raw.get("family") or raw.get("benchmark") or "default").lower(),
        )


@dataclass
class ArmScore:
    case_id: str
    benchmark: str
    arm: str
    passed: bool
    pass_rate: float
    tokens: int
    reason: str = ""
    samples_passed: int = 0
    samples_total: int = 0


@dataclass
class WikiSkillReport:
    cases: list[ArmScore] = field(default_factory=list)
    agent: str = "mock"
    bench_path: str = ""

    def by_arm(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for arm in ARMS:
            rows = [c for c in self.cases if c.arm == arm]
            passed = sum(1 for c in rows if c.passed)
            out[arm] = (passed, len(rows))
        return out

    def by_benchmark(self, arm: str) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        rows = [c for c in self.cases if c.arm == arm]
        for bench in sorted({c.benchmark for c in rows}):
            sub = [c for c in rows if c.benchmark == bench]
            passed = sum(1 for c in sub if c.passed)
            out[bench] = (passed, len(sub))
        return out

    def accuracy(self, arm: str) -> float:
        passed, total = self.by_arm().get(arm, (0, 0))
        return passed / total if total else 0.0

    def mean_tokens(self, arm: str) -> int:
        rows = [c for c in self.cases if c.arm == arm]
        return sum(c.tokens for c in rows) // len(rows) if rows else 0

    def render(self) -> str:
        lines = [
            f"WikiSkill-comparable bench — {len(self.cases)} arm scores, agent={self.agent}",
            f"bench: {self.bench_path or DEFAULT_BENCH}",
            "",
            "Accuracy by arm (WikiSkill Table 1 style):",
        ]
        for arm in ARMS:
            passed, total = self.by_arm().get(arm, (0, 0))
            rate = f"{passed / total:.0%}" if total else "—"
            tok = self.mean_tokens(arm)
            lines.append(f"  {arm:<16} {passed}/{total}  ({rate})  mean_tokens={tok}")
        lines += ["", "Per-benchmark (full-inject vs recall-judge):"]
        full = self.by_benchmark("full-inject")
        recall = self.by_benchmark("recall-judge")
        for bench in sorted(set(full) | set(recall)):
            fp, ft = full.get(bench, (0, 0))
            rp, rt = recall.get(bench, (0, 0))
            fr = f"{fp / ft:.0%}" if ft else "—"
            rr = f"{rp / rt:.0%}" if rt else "—"
            lines.append(f"  {bench:<14} full={fp}/{ft} ({fr})  recall={rp}/{rt} ({rr})")
        return "\n".join(lines)


def load_bench(path: Path | None = None) -> tuple[list[WikiSkillCase], list[str]]:
    bench_path = path or DEFAULT_BENCH
    if bench_path.suffix == ".jsonl":
        cases = []
        for line in bench_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cases.append(WikiSkillCase.from_dict(json.loads(line)))
        benchmarks = sorted({c.benchmark for c in cases})
        return cases, benchmarks
    raw = yamlish.load(bench_path.read_text(encoding="utf-8"))
    cases = [WikiSkillCase.from_dict(c) for c in (raw.get("cases") or []) if c.get("id")]
    benchmarks = [str(b) for b in (raw.get("benchmarks") or [])]
    return cases, benchmarks


def build_store(cases: list[WikiSkillCase], base: Path, *, dedupe_families: bool = False) -> Store:
    store = Store.init(base)
    seen_families: set[str] = set()
    for case in cases:
        if not case.skill:
            continue
        if dedupe_families:
            key = case.family or case.id
            if key in seen_families:
                continue
            seen_families.add(key)
            node_id = key
        else:
            node_id = case.id
        store.save_node(
            Node(
                id=node_id,
                family=case.family,
                level=0,
                title=node_id.replace("-", " "),
                gist=f"{case.benchmark}: {node_id}",
                body=case.skill,
                created=utcnow(),
                updated=utcnow(),
            )
        )
    index_mod.rebuild(store)
    return store


def full_inject_pack(store: Store) -> str:
    """WikiSkill test-time injection: all active skills in the prompt."""
    parts = []
    for node in sorted(store.nodes(), key=lambda n: n.id):
        if node.status == "archived":
            continue
        parts.append(node.body.strip())
    return "\n\n---\n\n".join(parts)


def _pack_for_arm(
    store: Store,
    adapter: Adapter,
    case: WikiSkillCase,
    arm: str,
    *,
    full_pack: str,
) -> str:
    if arm == "no-skill":
        return ""
    if arm == "full-inject":
        return full_pack
    if arm == "oracle-skill":
        return oracle_skill_pack(case)
    if arm == "trace2skill":
        return trace2skill_pack(store, case.task)
    if arm == "evoskill":
        return evoskill_pack(store, case.task)
    if arm == "skillopt":
        return skillopt_pack(store, case.task)
    if arm == "keyword-rag":
        return keyword_rag_pack(store, case.task)
    selector = "judge" if arm == "recall-judge" else "agentic"
    prev = store.config.get("recall.selector", "agentic")
    store.config.set("recall.selector", selector)
    try:
        pack = recall_pack(store, case.task, adapter)
        return pack.text
    finally:
        store.config.set("recall.selector", prev)


def wikiskill_adapter(base: Adapter | None, cases: list[WikiSkillCase]) -> Adapter:
    """Mock routing that applies lessons to produce expected-shaped answers."""

    by_task = {c.task.strip(): c for c in cases}

    def router(prompt: str, schema: dict | None) -> Any:
        head = (prompt or "")[:4000].lower()
        if schema is JUDGE_SCHEMA or (schema and "pass" in (schema.get("properties") or {})):
            task = _section(prompt, "TASK")
            expected = _section(prompt, "KNOWN-GOOD") or _section(prompt, "EXPECTED")
            candidate = _section(prompt, "CANDIDATE")
            ok, reason = mock_grade(expected, candidate, kind="trap")
            return {"pass": ok, "reason": reason}
        if "describe the approach" in head and "<<<lesson" in head:
            pack = _section(prompt, "LESSON")
            task = _section(prompt, "TASK").strip()
            case = by_task.get(task)
            if case and pack and "(no lesson available)" not in pack:
                return case.expected
            return "Use generic defaults without domain-specific procedure."
        if schema and "picks" in (schema.get("properties") or {}):
            question = _section(prompt, "WORK") or _section(prompt, "QUESTION")
            picks = []
            q = question.lower()
            for ident, text in _candidates(prompt):
                relevant = any(
                    word in text.lower()
                    for word in q.split()
                    if len(word) > 5
                ) or any(c.id == ident and c.task.strip()[:40] in question for c in cases)
                picks.append(
                    {
                        "id": ident,
                        "verdict": "relevant" if relevant else "unrelated",
                        "descend": False,
                        "why": "task overlap" if relevant else "no overlap",
                    }
                )
            return {"picks": picks}
        if base is not None and getattr(base, "router", None):
            return base.router(prompt, schema)
        return MockAdapter()._builtin(prompt, schema)

    if base is not None and getattr(base, "name", "") != "mock":
        return base
    return MockAdapter(router=router)


def from_checkpoint_dict(data: dict[str, Any]) -> WikiSkillReport:
    """Restore a partial WikiSkillReport from a checkpoint JSON payload."""
    report = WikiSkillReport(
        agent=str(data.get("agent", "?")),
        bench_path=str(data.get("bench_path", "")),
    )
    for raw in data.get("cases") or []:
        report.cases.append(
            ArmScore(
                case_id=str(raw.get("case_id", "")),
                benchmark=str(raw.get("benchmark", "")),
                arm=str(raw.get("arm", "")),
                passed=bool(raw.get("passed")),
                pass_rate=float(raw.get("pass_rate", 0)),
                tokens=int(raw.get("tokens", 0)),
                reason=str(raw.get("reason", "")),
                samples_passed=int(raw.get("samples_passed", 0)),
                samples_total=int(raw.get("samples_total", 0)),
            )
        )
    return report


def scored_keys(report: WikiSkillReport) -> set[tuple[str, str]]:
    return {(c.case_id, c.arm) for c in report.cases}


def _bench_paths_match(left: str | Path | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return False
    return Path(left).resolve() == Path(right).resolve()


def merge_reports(*reports: WikiSkillReport) -> WikiSkillReport:
    """Combine shard reports, deduplicating by (case_id, arm)."""
    if not reports:
        return WikiSkillReport()
    merged = WikiSkillReport(
        agent=reports[0].agent,
        bench_path=reports[0].bench_path,
    )
    seen: set[tuple[str, str]] = set()
    for report in reports:
        if report.bench_path and not merged.bench_path:
            merged.bench_path = report.bench_path
        if report.agent and merged.agent == "?":
            merged.agent = report.agent
        for row in report.cases:
            key = (row.case_id, row.arm)
            if key in seen:
                continue
            seen.add(key)
            merged.cases.append(row)
    return merged


def run(
    adapter: Adapter,
    *,
    path: Path | None = None,
    samples: int = 1,
    timeout: int = 180,
    store: Store | None = None,
    tmp_base: Path | None = None,
    arms: tuple[str, ...] | None = None,
    offset: int = 0,
    limit: int | None = None,
    on_progress: Callable[[WikiSkillReport], None] | None = None,
    existing: WikiSkillReport | None = None,
) -> WikiSkillReport:
    cases, _ = load_bench(path)
    if offset:
        cases = cases[offset:]
    if limit is not None:
        cases = cases[:limit]
    bench_path = str(path or DEFAULT_BENCH)
    use_arms = arms or ARMS
    wrapped = wikiskill_adapter(adapter if getattr(adapter, "name", "") == "mock" else adapter, cases)
    if getattr(adapter, "name", "") != "mock":
        wrapped = adapter
    if existing is not None and _bench_paths_match(existing.bench_path, bench_path):
        report = existing
        report.agent = getattr(wrapped, "name", report.agent)
    else:
        report = WikiSkillReport(agent=getattr(wrapped, "name", "?"), bench_path=bench_path)

    if store is None:
        import tempfile

        dedupe = path is not None and Path(path).suffix == ".jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            store = build_store(cases, Path(tmp) / "repo", dedupe_families=dedupe)
            _score_cases(
                report, wrapped, store, cases, samples=samples, timeout=timeout, arms=use_arms,
                on_progress=on_progress,
            )
    else:
        _score_cases(
            report, wrapped, store, cases, samples=samples, timeout=timeout, arms=use_arms,
            on_progress=on_progress,
        )

    return report


def _score_cases(
    report: WikiSkillReport,
    adapter: Adapter,
    store: Store,
    cases: list[WikiSkillCase],
    *,
    samples: int,
    timeout: int,
    arms: tuple[str, ...],
    on_progress: Callable[[WikiSkillReport], None] | None = None,
) -> None:
    from .bench import _grade, _probe

    full_pack = full_inject_pack(store)
    skip = scored_keys(report)
    for case in cases:
        for arm in arms:
            if (case.id, arm) in skip:
                continue
            pack = _pack_for_arm(store, adapter, case, arm, full_pack=full_pack)
            passed_runs = 0
            reasons: list[str] = []
            for _ in range(samples):
                probe_arm = CONTROL if arm == "no-skill" else "L0"
                answer = _probe(adapter, case.task, pack if probe_arm != CONTROL else "", timeout)
                ok, why = _grade(adapter, case.task, case.expected, answer, timeout)
                if getattr(adapter, "name", "") == "mock" or not ok:
                    ok2, why2 = mock_grade(case.expected, answer, kind="trap")
                    if getattr(adapter, "name", "") == "mock":
                        ok, why = ok2, why2
                passed_runs += 1 if ok else 0
                reasons.append(why)
            rate = passed_runs / samples if samples else 0.0
            report.cases.append(
                ArmScore(
                    case_id=case.id,
                    benchmark=case.benchmark,
                    arm=arm,
                    passed=rate >= 0.5,
                    pass_rate=rate,
                    tokens=count_tokens(pack) if pack else 0,
                    reason=reasons[0] if reasons else "",
                    samples_passed=passed_runs,
                    samples_total=samples,
                )
            )
        if on_progress is not None:
            on_progress(report)


def _as_bench_case(case: WikiSkillCase):
    """Adapt WikiSkillCase for bench.score_transfer (expects BenchCase fields)."""
    from .bench import BenchCase

    return BenchCase(
        id=case.id,
        kind="trap",
        task=case.task,
        expected=case.expected,
        trap="",
        family=case.family,
        lesson=case.skill,
    )


def _case_pass_rates(report: WikiSkillReport, arm: str) -> list[float]:
    seen: dict[str, float] = {}
    for row in report.cases:
        if row.arm == arm:
            seen[row.case_id] = row.pass_rate
    return list(seen.values())


def significance_vs_full_inject(report: WikiSkillReport) -> dict[str, Any]:
    full = _case_pass_rates(report, "full-inject")
    out: dict[str, Any] = {}
    for arm in ARMS:
        if arm in ("no-skill", "full-inject"):
            continue
        rates = _case_pass_rates(report, arm)
        if len(rates) == len(full) and rates:
            out[arm] = paired_bootstrap_test(rates, full, iterations=1000, seed=7).as_dict()
    return out


def to_dict(report: WikiSkillReport) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARMS:
        passed, total = report.by_arm().get(arm, (0, 0))
        rates = _case_pass_rates(report, arm)
        ci = bootstrap_ci(rates, iterations=1000, seed=3) if rates else None
        arms[arm] = {
            "passed": passed,
            "total": total,
            "accuracy": report.accuracy(arm),
            "mean_tokens": report.mean_tokens(arm),
            "bootstrap_ci": ci.as_dict() if ci else None,
            "by_benchmark": {
                bench: {"passed": p, "total": t, "accuracy": p / t if t else 0.0}
                for bench, (p, t) in report.by_benchmark(arm).items()
            },
        }
    return {
        "agent": report.agent,
        "bench_path": report.bench_path,
        "arms": arms,
        "cases": [
            {
                "case_id": c.case_id,
                "benchmark": c.benchmark,
                "arm": c.arm,
                "passed": c.passed,
                "pass_rate": c.pass_rate,
                "samples_passed": c.samples_passed,
                "samples_total": c.samples_total,
                "tokens": c.tokens,
                "reason": c.reason,
            }
            for c in report.cases
        ],
        "render": report.render(),
        "significance_vs_full_inject": significance_vs_full_inject(report),
        "comparisons": {
            "full_inject_vs_no_skill": {
                "delta_accuracy": report.accuracy("full-inject") - report.accuracy("no-skill"),
            },
            "recall_judge_vs_full_inject": {
                "delta_accuracy": report.accuracy("recall-judge") - report.accuracy("full-inject"),
                "token_savings": report.mean_tokens("full-inject") - report.mean_tokens("recall-judge"),
            },
            "recall_agentic_vs_full_inject": {
                "delta_accuracy": report.accuracy("recall-agentic") - report.accuracy("full-inject"),
                "token_savings": report.mean_tokens("full-inject") - report.mean_tokens("recall-agentic"),
            },
            "recall_agentic_vs_recall_judge": {
                "delta_accuracy": report.accuracy("recall-agentic") - report.accuracy("recall-judge"),
            },
        },
    }
