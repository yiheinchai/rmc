"""SealQA continual-learning harness.

Unlike upstream ``sealqa-test.jsonl`` (broken snippets, one static skill, no
learning loop), this module runs a proper train → compact → test protocol on
curated evidence-handling cases:

* **train** — one variant per axis; attempt with recall, ingest lesson + probe
* **compact** — replay-gated compression against seeded probes
* **test** — held-out variant per axis; score no-memory vs static vs continual

The bench defaults to ``evals/sealqa-ablation/probe-dev.yaml`` with split
``*-1`` → train and ``*-2`` → test.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import index as index_mod
from . import probes as probes_mod
from . import yamlish
from .adapters import Adapter
from .bench import _grade, _probe, mock_grade
from .compact import validate
from .node import Node
from .recall import recall_pack
from .skill_baselines import oracle_skill_pack
from .store import Store
from .util import count_tokens, utcnow
from .wikiskill import WikiSkillCase, compact_probe_store, wikiskill_adapter

DEFAULT_BENCH = (
    Path(__file__).resolve().parents[1] / "evals" / "sealqa-ablation" / "probe-dev.yaml"
)

TEST_ARMS = ("no-memory", "static-composite", "continual-recall", "oracle")


@dataclass
class CLCase(WikiSkillCase):
    axis: str = ""
    split: str = "train"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_split: str = "train") -> "CLCase":
        case_id = str(raw.get("id") or "")
        axis = str(raw.get("axis") or case_id)
        split = str(raw.get("split") or default_split)
        if not raw.get("split"):
            if case_id.endswith("-1"):
                split = "train"
            elif case_id.endswith("-2"):
                split = "test"
        return cls(
            id=case_id,
            benchmark=str(raw.get("benchmark") or "SealQA"),
            task=str(raw.get("task") or "").strip(),
            expected=str(raw.get("expected") or "").strip(),
            skill=str(raw.get("skill") or "").strip(),
            family=str(raw.get("family") or "sealqa"),
            axis=axis,
            split=split,
        )


@dataclass
class CLBench:
    lesson: str
    train: list[CLCase]
    test: list[CLCase]
    axes: dict[str, str]

    @classmethod
    def load(cls, path: Path | None = None) -> "CLBench":
        bench_path = path or DEFAULT_BENCH
        raw = yamlish.load(bench_path.read_text(encoding="utf-8"))
        lesson = str(raw.get("lesson") or "").strip()
        train: list[CLCase] = []
        test: list[CLCase] = []
        axes: dict[str, str] = {}
        for item in raw.get("cases") or []:
            if not item.get("id"):
                continue
            case = CLCase.from_dict(item)
            axes[case.id] = case.axis
            if case.split == "test":
                test.append(case)
            else:
                train.append(case)
        return cls(lesson=lesson, train=train, test=test, axes=axes)


@dataclass
class AttemptScore:
    case_id: str
    axis: str
    arm: str
    passed: bool
    tokens: int
    reason: str = ""
    answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "axis": self.axis,
            "arm": self.arm,
            "passed": self.passed,
            "tokens": self.tokens,
            "reason": self.reason[:240],
            "answer": self.answer[:200],
        }


@dataclass
class TrainStep:
    case_id: str
    axis: str
    passed_before_learn: bool
    recall_tokens: int
    probe_count: int
    compacted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "axis": self.axis,
            "passed_before_learn": self.passed_before_learn,
            "recall_tokens": self.recall_tokens,
            "probe_count": self.probe_count,
            "compacted": self.compacted,
        }


@dataclass
class CLReport:
    bench_path: str
    agent: str
    train_steps: list[TrainStep] = field(default_factory=list)
    test_scores: list[AttemptScore] = field(default_factory=list)
    probe_pass_rate: float = 0.0
    probe_passed: int = 0
    probe_total: int = 0
    compaction_accepted: bool = False
    compaction_reason: str = ""
    tokens_before_compact: int = 0
    tokens_after_compact: int = 0

    def accuracy(self, arm: str) -> tuple[int, int, float]:
        rows = [s for s in self.test_scores if s.arm == arm]
        passed = sum(1 for s in rows if s.passed)
        total = len(rows)
        return passed, total, passed / total if total else 0.0

    def by_axis(self, arm: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in self.test_scores:
            if row.arm != arm:
                continue
            bucket = out.setdefault(row.axis, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += 1 if row.passed else 0
        for bucket in out.values():
            bucket["pass_rate"] = bucket["passed"] / bucket["total"] if bucket["total"] else 0.0
        return out

    def render(self) -> str:
        lines = [
            f"SealQA continual learning — {self.bench_path}",
            f"agent={self.agent}  train={len(self.train_steps)}  test={len({s.case_id for s in self.test_scores})}",
            "",
            "Train stream (recall before ingest):",
        ]
        for step in self.train_steps:
            mark = "ok" if step.passed_before_learn else "miss"
            lines.append(
                f"  {step.case_id:<28} {step.axis:<16} before={mark}  probes={step.probe_count}"
            )
        lines += [
            "",
            f"Compaction: accepted={self.compaction_accepted}  "
            f"probes={self.probe_passed}/{self.probe_total} ({self.probe_pass_rate:.0%})  "
            f"tokens {self.tokens_before_compact}→{self.tokens_after_compact}",
            f"  reason: {self.compaction_reason or '(none)'}",
            "",
            f"{'arm':<20} {'accuracy':>10} {'mean_tok':>9}",
        ]
        for arm in TEST_ARMS:
            passed, total, acc = self.accuracy(arm)
            rows = [s for s in self.test_scores if s.arm == arm]
            mean_tok = sum(s.tokens for s in rows) // len(rows) if rows else 0
            lines.append(f"{arm:<20} {passed}/{total} ({acc:.0%}) {mean_tok:>9}")
        cont = self.by_axis("continual-recall")
        nomem = self.by_axis("no-memory")
        axes = sorted(set(cont) | set(nomem))
        if axes:
            lines.append("")
            lines.append("Per-axis test (continual-recall vs no-memory):")
            for axis in axes:
                c = cont.get(axis, {})
                n = nomem.get(axis, {})
                lines.append(
                    f"  {axis:<18} continual={c.get('passed', 0)}/{c.get('total', 0)}"
                    f"  no-memory={n.get('passed', 0)}/{n.get('total', 0)}"
                )
        lift_p, lift_t, lift_acc = self.accuracy("continual-recall")
        base_p, base_t, base_acc = self.accuracy("no-memory")
        lines.append("")
        lines.append(f"Lift (continual − no-memory): {lift_acc - base_acc:+.0%}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bench_path": self.bench_path,
            "agent": self.agent,
            "train_steps": [s.to_dict() for s in self.train_steps],
            "test_scores": [s.to_dict() for s in self.test_scores],
            "probe_pass_rate": self.probe_pass_rate,
            "probe_passed": self.probe_passed,
            "probe_total": self.probe_total,
            "compaction_accepted": self.compaction_accepted,
            "compaction_reason": self.compaction_reason,
            "tokens_before_compact": self.tokens_before_compact,
            "tokens_after_compact": self.tokens_after_compact,
            "arms": {
                arm: {
                    "passed": self.accuracy(arm)[0],
                    "total": self.accuracy(arm)[1],
                    "accuracy": self.accuracy(arm)[2],
                }
                for arm in TEST_ARMS
            },
            "lift_vs_no_memory": self.accuracy("continual-recall")[2] - self.accuracy("no-memory")[2],
            "render": self.render(),
        }


def _empty_store(base: Path) -> Store:
    return Store.init(base)


def _static_store(bench: CLBench, base: Path) -> Store:
    store = Store.init(base)
    node = Node(
        id="sealqa-composite",
        family="sealqa",
        title="SealQA evidence handling",
        gist="Composite procedural memory for SealQA",
        body=bench.lesson,
        level=0,
        created=utcnow(),
        updated=utcnow(),
    )
    store.save_node(node)
    index_mod.rebuild(store)
    return store


def _ingest_lesson(store: Store, case: CLCase) -> int:
    node_id = f"sealqa-{case.axis}"
    body = case.skill.strip()
    node = store.get(node_id)
    if node is None:
        node = Node(
            id=node_id,
            family="sealqa",
            title=case.axis.replace("-", " "),
            gist=case.axis,
            body=body,
            level=0,
            created=utcnow(),
            updated=utcnow(),
        )
    elif body and body not in node.body:
        node.body = f"{node.body.rstrip()}\n{body}"
    node.stats.successes = max(node.stats.successes, 1)
    store.save_node(node)
    probe = probes_mod.seed_from_case(
        store,
        case_id=case.id,
        node_id=node.id,
        task=case.task,
        expected=case.expected,
        axis=case.axis,
        enforce_cap=False,
    )
    probes_mod.prune_to_cap(store, node.id)
    index_mod.rebuild(store)
    store.invalidate()
    return len(probes_mod.load_for_node(store, node.id))


def _attempt(
    adapter: Adapter,
    case: CLCase,
    pack: str,
    *,
    timeout: int,
) -> AttemptScore:
    answer = _probe(adapter, case.task, pack, timeout=timeout)
    ok, why = _grade(adapter, case.task, case.expected, answer, timeout=timeout)
    if getattr(adapter, "name", "") == "mock":
        ok, why = mock_grade(case.expected, answer, kind="trap")
    return AttemptScore(
        case_id=case.id,
        axis=case.axis,
        arm="",
        passed=ok,
        tokens=count_tokens(pack) if pack else 0,
        reason=why,
        answer=answer,
    )


def _probe_replay_rate(store: Store, adapter: Adapter) -> tuple[float, int, int]:
    probes: list[Any] = []
    bodies: dict[str, str] = {}
    for node in store.nodes():
        if not node.is_apex:
            continue
        node_probes = store.regression_set(node)
        if not node_probes:
            continue
        probes.extend(node_probes)
        bodies[node.id] = node.body
    if not probes:
        return 0.0, 0, 0
    passed = 0
    for node in store.nodes():
        if not node.is_apex:
            continue
        node_probes = store.regression_set(node)
        if not node_probes:
            continue
        outcomes = validate(store, adapter, node.body, node_probes)
        passed += sum(1 for o in outcomes if o.ok)
    total = sum(len(store.regression_set(n)) for n in store.nodes() if n.is_apex and store.regression_set(n))
    return passed / total if total else 0.0, passed, total


def train_stream(
    store: Store,
    adapter: Adapter,
    cases: list[CLCase],
    *,
    timeout: int = 180,
) -> list[TrainStep]:
    steps: list[TrainStep] = []
    for case in cases:
        pack = recall_pack(store, case.task, adapter)
        attempt = _attempt(adapter, case, pack.text, timeout=timeout)
        probe_count = _ingest_lesson(store, case)
        steps.append(
            TrainStep(
                case_id=case.id,
                axis=case.axis,
                passed_before_learn=attempt.passed,
                recall_tokens=pack.tokens,
                probe_count=probe_count,
            )
        )
    return steps


def score_test_arm(
    store: Store | None,
    adapter: Adapter,
    bench: CLBench,
    arm: str,
    *,
    timeout: int = 180,
) -> list[AttemptScore]:
    scores: list[AttemptScore] = []
    static_pack = bench.lesson if arm == "static-composite" else ""
    for case in bench.test:
        if arm == "no-memory":
            pack = ""
        elif arm == "static-composite":
            pack = static_pack
        elif arm == "oracle":
            pack = oracle_skill_pack(case)
        elif arm == "continual-recall":
            if store is None:
                raise ValueError("continual-recall requires a trained store")
            pack = recall_pack(store, case.task, adapter).text
        else:
            raise ValueError(f"unknown arm {arm!r}")
        attempt = _attempt(adapter, case, pack, timeout=timeout)
        attempt.arm = arm
        scores.append(attempt)
    return scores


def _wrap_adapter(adapter: Adapter, bench: CLBench) -> Adapter:
    if getattr(adapter, "name", "") == "mock":
        return wikiskill_adapter(adapter, [*bench.train, *bench.test])
    return adapter


def run(
    adapter: Adapter,
    *,
    path: Path | None = None,
    timeout: int = 180,
    compact: bool = True,
) -> CLReport:
    bench = CLBench.load(path)
    adapter = _wrap_adapter(adapter, bench)
    report = CLReport(
        bench_path=str(path or DEFAULT_BENCH),
        agent=getattr(adapter, "name", "?"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        store = _empty_store(Path(tmp) / "repo")
        report.train_steps = train_stream(store, adapter, bench.train, timeout=timeout)

        report.tokens_before_compact = sum(n.tokens for n in store.nodes() if n.is_apex)
        if compact and any(probes_mod.collect(store, n) for n in store.nodes() if n.is_apex):
            before = {n.id: n.tokens for n in store.nodes()}
            compressed = compact_probe_store(store, adapter)
            after = {nid: store.get(nid).tokens if store.get(nid) else 0 for nid in before}
            report.tokens_after_compact = sum(after.values())
            report.compaction_accepted = bool(compressed)
            report.compaction_reason = "compacted " + ", ".join(compressed) if compressed else "no compaction accepted"
        else:
            report.tokens_after_compact = report.tokens_before_compact
            report.compaction_reason = "compaction skipped"

        rate, passed, total = _probe_replay_rate(store, adapter)
        report.probe_pass_rate = rate
        report.probe_passed = passed
        report.probe_total = total

        for arm in TEST_ARMS:
            if arm == "continual-recall":
                scores = score_test_arm(store, adapter, bench, arm, timeout=timeout)
            elif arm == "static-composite":
                scores = score_test_arm(None, adapter, bench, arm, timeout=timeout)
            else:
                scores = score_test_arm(None, adapter, bench, arm, timeout=timeout)
            report.test_scores.extend(scores)

    return report
