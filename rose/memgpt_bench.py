"""MemGPT-style nested key-value retrieval benchmark (multi-hop proxy).

Mirrors MemGPT nested KV retrieval: values may be keys requiring chained lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import yamlish
from .adapters import Adapter
from .bench import mock_grade, score_transfer
from .evaluate import CONTROL
from .recall import recall_pack
from .store import Store

DEFAULT_BENCH = Path(__file__).resolve().parents[1] / "evals" / "memgpt-nested-kv.yaml"

ARMS = ("fixed-context", "recall-agentic")


@dataclass
class KvCase:
    id: str
    task: str
    expected: str
    catalog: str
    hops: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KvCase":
        return cls(
            id=str(raw.get("id") or ""),
            task=str(raw.get("task") or "").strip(),
            expected=str(raw.get("expected") or "").strip(),
            catalog=str(raw.get("catalog") or "").strip(),
            hops=int(raw.get("hops") or 1),
        )


@dataclass
class KvScore:
    case_id: str
    arm: str
    passed: bool
    tokens: int
    hops: int


@dataclass
class MemgptReport:
    cases: list[KvScore] = field(default_factory=list)
    agent: str = "mock"
    bench_path: str = ""

    def accuracy(self, arm: str) -> float:
        rows = [c for c in self.cases if c.arm == arm]
        return sum(1 for c in rows if c.passed) / len(rows) if rows else 0.0

    def render(self) -> str:
        lines = [f"MemGPT nested KV — agent={self.agent}", ""]
        for arm in ARMS:
            rows = [c for c in self.cases if c.arm == arm]
            passed = sum(1 for c in rows if c.passed)
            lines.append(f"  {arm:<16} {passed}/{len(rows)} ({self.accuracy(arm):.0%})")
        return "\n".join(lines)


def load_bench(path: Path | None = None) -> list[KvCase]:
    bench_path = path or DEFAULT_BENCH
    raw = yamlish.load(bench_path.read_text(encoding="utf-8"))
    return [KvCase.from_dict(c) for c in (raw.get("cases") or []) if c.get("id")]


def run(adapter: Adapter, *, path: Path | None = None, samples: int = 3, timeout: int = 120) -> MemgptReport:
    from .bench import BenchCase
    from .wikiskill import WikiSkillCase, build_store

    cases = load_bench(path)
    report = MemgptReport(agent=getattr(adapter, "name", "?"), bench_path=str(path or DEFAULT_BENCH))

    ws_cases = [
        WikiSkillCase(
            id=c.id,
            benchmark="NestedKV",
            task=c.catalog,
            expected=c.catalog,
            skill=c.catalog,
            family="nested-kv",
        )
        for c in cases
    ]

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = build_store(ws_cases, Path(tmp) / "repo")
        for case in cases:
            fixed_pack = case.catalog[:2000]
            for arm, pack in (
                ("fixed-context", fixed_pack),
                ("recall-agentic", recall_pack(store, case.task, adapter).text),
            ):
                bench_case = BenchCase(
                    id=case.id,
                    kind="trap",
                    task=case.task,
                    expected=case.expected,
                    trap="",
                    family="nested-kv",
                    lesson=case.catalog,
                )
                score = score_transfer(
                    adapter, bench_case, arm=CONTROL if not pack else "L0", pack=pack, samples=samples, timeout=timeout
                )
                report.cases.append(
                    KvScore(case_id=case.id, arm=arm, passed=score.passed, tokens=score.tokens, hops=case.hops)
                )
    return report


def to_dict(report: MemgptReport) -> dict[str, Any]:
    return {
        "agent": report.agent,
        "bench_path": report.bench_path,
        "arms": {
            arm: {
                "passed": sum(1 for c in report.cases if c.arm == arm and c.passed),
                "total": sum(1 for c in report.cases if c.arm == arm),
                "accuracy": report.accuracy(arm),
            }
            for arm in ARMS
        },
        "render": report.render(),
    }
