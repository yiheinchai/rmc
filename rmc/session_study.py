"""Paired session-length study: does recall on a follow-up task beat memory-off?

EXPERIMENTS §7 asks whether a recalled lesson shortens the next session. This
harness approximates that with scripted session pairs: a warmup narrative
(session 1) and a related follow-up task (session 2). The store holds lessons
captured in session 1; memory-on runs ``recall_pack``, memory-off does not.

This is not a live multi-turn agent session — it is a controlled probe of whether
RSE recall on turn 2 improves outcomes vs a fresh context with only the warmup
narrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import yamlish
from .adapters import Adapter
from .bench import build_bench_store, load_bench, score_transfer
from .evaluate import CONTROL
from .node import Node
from .recall import recall_pack
from .store import Store
from .util import count_tokens, utcnow

DEFAULT_PAIRS = Path(__file__).resolve().parents[1] / "evals" / "session-pairs.yaml"


@dataclass
class SessionPair:
    id: str
    warmup: str
    task: str
    expected: str
    lesson_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SessionPair":
        return cls(
            id=str(raw.get("id") or ""),
            warmup=str(raw.get("warmup") or "").strip(),
            task=str(raw.get("task") or "").strip(),
            expected=str(raw.get("expected") or "").strip(),
            lesson_ids=[str(x) for x in (raw.get("lesson_ids") or [])],
        )


@dataclass
class PairScore:
    pair_id: str
    arm: str
    passed: bool
    tokens: int
    reason: str = ""


@dataclass
class SessionStudyReport:
    pairs: list[PairScore] = field(default_factory=list)
    agent: str = "mock"

    def by_arm(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for arm in ("memory-off", "memory-on"):
            rows = [p for p in self.pairs if p.arm == arm]
            passed = sum(1 for p in rows if p.passed)
            out[arm] = (passed, len(rows))
        return out

    def accuracy(self, arm: str) -> float:
        passed, total = self.by_arm().get(arm, (0, 0))
        return passed / total if total else 0.0

    def mean_tokens(self, arm: str) -> int:
        rows = [p for p in self.pairs if p.arm == arm]
        return sum(p.tokens for p in rows) // len(rows) if rows else 0

    def render(self) -> str:
        off = self.by_arm().get("memory-off", (0, 0))
        on = self.by_arm().get("memory-on", (0, 0))
        lines = [
            f"Session-length paired study — agent={self.agent}",
            "",
            f"  memory-off  {off[0]}/{off[1]} ({self.accuracy('memory-off'):.0%})  "
            f"mean_tokens={self.mean_tokens('memory-off')}",
            f"  memory-on   {on[0]}/{on[1]} ({self.accuracy('memory-on'):.0%})  "
            f"mean_tokens={self.mean_tokens('memory-on')}",
            "",
            f"  lift (memory-on − off): {self.accuracy('memory-on') - self.accuracy('memory-off'):+.0%}",
        ]
        return "\n".join(lines)


def load_pairs(path: Path | None = None) -> list[SessionPair]:
    pairs_path = path or DEFAULT_PAIRS
    raw = yamlish.load(pairs_path.read_text(encoding="utf-8"))
    return [SessionPair.from_dict(p) for p in (raw.get("pairs") or []) if p.get("id")]


def build_pair_store(pair: SessionPair, base: Path) -> Store:
    cases, _ = load_bench()
    by_id = {c.id: c for c in cases}
    store = Store.init(base)
    for lid in pair.lesson_ids:
        case = by_id.get(lid)
        if not case or not case.lesson:
            continue
        store.save_node(
            Node(
                id=case.id,
                family=case.family,
                level=0,
                title=case.id.replace("-", " "),
                gist=case.id.replace("-", " "),
                body=case.lesson,
                created=utcnow(),
                updated=utcnow(),
            )
        )
    from . import index as index_mod

    index_mod.rebuild(store)
    return store


def _bench_case(pair: SessionPair):
    from .bench import BenchCase

    return BenchCase(
        id=pair.id,
        kind="trap",
        task=pair.task,
        expected=pair.expected,
        trap="",
        family="session",
    )


def run(
    adapter: Adapter,
    *,
    path: Path | None = None,
    samples: int = 1,
    timeout: int = 180,
) -> SessionStudyReport:
    import tempfile

    pairs = load_pairs(path)
    report = SessionStudyReport(agent=getattr(adapter, "name", "?"))

    with tempfile.TemporaryDirectory() as tmp:
        for pair in pairs:
            store = build_pair_store(pair, Path(tmp) / pair.id)
            prompt = f"{pair.warmup}\n\n{pair.task}"
            narrative_task = prompt

            # memory-off: prior session narrative only, no structured recall
            off_case = _bench_case(pair)
            off_case.task = narrative_task
            off_score = score_transfer(
                adapter,
                off_case,
                arm=CONTROL,
                pack="",
                samples=samples,
                timeout=timeout,
            )
            report.pairs.append(
                PairScore(
                    pair_id=pair.id,
                    arm="memory-off",
                    passed=off_score.passed,
                    tokens=0,
                    reason=off_score.reason,
                )
            )

            pack = recall_pack(store, prompt, adapter).text
            on_score = score_transfer(
                adapter,
                _bench_case(pair),
                arm="L0",
                pack=pack,
                samples=samples,
                timeout=timeout,
            )
            report.pairs.append(
                PairScore(
                    pair_id=pair.id,
                    arm="memory-on",
                    passed=on_score.passed,
                    tokens=count_tokens(pack),
                    reason=on_score.reason,
                )
            )

    return report


def to_dict(report: SessionStudyReport) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ("memory-off", "memory-on"):
        passed, total = report.by_arm().get(arm, (0, 0))
        arms[arm] = {
            "passed": passed,
            "total": total,
            "accuracy": report.accuracy(arm),
            "mean_tokens": report.mean_tokens(arm),
        }
    return {
        "agent": report.agent,
        "arms": arms,
        "lift": report.accuracy("memory-on") - report.accuracy("memory-off"),
        "pairs": [
            {
                "pair_id": p.pair_id,
                "arm": p.arm,
                "passed": p.passed,
                "tokens": p.tokens,
                "reason": p.reason[:200],
            }
            for p in report.pairs
        ],
        "render": report.render(),
    }
