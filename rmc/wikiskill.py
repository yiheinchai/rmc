"""WikiSkill-comparable benchmark runner for RSE.

Loads ``evals/wikiskill-bench.yaml`` and scores four arms per task:

* **no-skill** — bare task (WikiSkill "No skill" baseline)
* **full-inject** — all store lessons concatenated (WikiSkill test-time injection)
* **recall-judge** — RSE recall with judge-walk selector
* **recall-agentic** — RSE recall with agentic selector (cold search when no session)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import index as index_mod
from . import yamlish
from .adapters import Adapter
from .adapters.mock import MockAdapter, _candidates, _section
from .bench import bench_adapter, mock_grade, score_transfer
from .evaluate import CONTROL
from .node import Node
from .prompts import JUDGE_SCHEMA
from .recall import recall_pack
from .store import Store
from .util import count_tokens, utcnow

DEFAULT_BENCH = Path(__file__).resolve().parents[1] / "evals" / "wikiskill-bench.yaml"

ARMS = ("no-skill", "full-inject", "recall-judge", "recall-agentic")


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
    raw = yamlish.load(bench_path.read_text(encoding="utf-8"))
    cases = [WikiSkillCase.from_dict(c) for c in (raw.get("cases") or []) if c.get("id")]
    benchmarks = [str(b) for b in (raw.get("benchmarks") or [])]
    return cases, benchmarks


def build_store(cases: list[WikiSkillCase], base: Path) -> Store:
    store = Store.init(base)
    for case in cases:
        if not case.skill:
            continue
        store.save_node(
            Node(
                id=case.id,
                family=case.family,
                level=0,
                title=case.id.replace("-", " "),
                gist=f"{case.benchmark}: {case.id}",
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


def run(
    adapter: Adapter,
    *,
    path: Path | None = None,
    samples: int = 1,
    timeout: int = 180,
    store: Store | None = None,
    tmp_base: Path | None = None,
) -> WikiSkillReport:
    cases, _ = load_bench(path)
    bench_path = str(path or DEFAULT_BENCH)
    wrapped = wikiskill_adapter(adapter if getattr(adapter, "name", "") == "mock" else adapter, cases)
    if getattr(adapter, "name", "") != "mock":
        wrapped = adapter
    report = WikiSkillReport(agent=getattr(wrapped, "name", "?"), bench_path=bench_path)

    if store is None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = build_store(cases, Path(tmp) / "repo")
            _score_cases(report, wrapped, store, cases, samples=samples, timeout=timeout)
    else:
        _score_cases(report, wrapped, store, cases, samples=samples, timeout=timeout)

    return report


def _score_cases(
    report: WikiSkillReport,
    adapter: Adapter,
    store: Store,
    cases: list[WikiSkillCase],
    *,
    samples: int,
    timeout: int,
) -> None:
    full_pack = full_inject_pack(store)
    for case in cases:
        for arm in ARMS:
            pack = _pack_for_arm(store, adapter, case, arm, full_pack=full_pack)
            probe_arm = CONTROL if arm == "no-skill" else "L0"
            score = score_transfer(
                adapter,
                _as_bench_case(case),
                arm=probe_arm,
                pack=pack,
                samples=samples,
                timeout=timeout,
            )
            passed_runs = int(score.passed) * samples  # score_transfer uses majority
            report.cases.append(
                ArmScore(
                    case_id=case.id,
                    benchmark=case.benchmark,
                    arm=arm,
                    passed=score.passed,
                    pass_rate=1.0 if score.passed else 0.0,
                    tokens=count_tokens(pack) if pack else 0,
                    reason=score.reason,
                )
            )


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


def to_dict(report: WikiSkillReport) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARMS:
        passed, total = report.by_arm().get(arm, (0, 0))
        arms[arm] = {
            "passed": passed,
            "total": total,
            "accuracy": report.accuracy(arm),
            "mean_tokens": report.mean_tokens(arm),
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
                "tokens": c.tokens,
                "reason": c.reason,
            }
            for c in report.cases
        ],
        "render": report.render(),
        "comparisons": {
            "full_inject_vs_no_skill": {
                "delta_accuracy": report.accuracy("full-inject") - report.accuracy("no-skill"),
            },
            "recall_judge_vs_full_inject": {
                "delta_accuracy": report.accuracy("recall-judge") - report.accuracy("full-inject"),
                "token_savings": report.mean_tokens("full-inject") - report.mean_tokens("recall-judge"),
            },
            "recall_agentic_vs_recall_judge": {
                "delta_accuracy": report.accuracy("recall-agentic") - report.accuracy("recall-judge"),
            },
        },
    }
