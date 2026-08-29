"""ROSE-Bench: procedural memory under compression and retrieval.

Loads ``evals/rose-bench.yaml`` and scores four axes from the eval README:

* **Transfer** — does the lesson change behaviour (control vs treatment)?
* **Retention** — does transfer survive mock compression (L0 vs L1)?
* **Retrieval** — does the selector serve the right lessons (distractor/null/multi)?
* **Cost** — tokens paid per prompt.

The mock backend uses term overlap against ``expected`` for blind grading so the
full pipeline runs without API keys. Pass ``--agent claude`` for model grading.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import index as index_mod
from . import yamlish
from .adapters import Adapter
from .adapters.mock import MockAdapter, _candidates, _section
from .compact import compress_node
from .evaluate import CONTROL, _grade, _probe
from .judge import Judge
from .node import Node
from .prompts import JUDGE_SCHEMA
from .store import Episode, Store
from .util import count_tokens, new_id, utcnow

DEFAULT_BENCH = Path(__file__).resolve().parents[1] / "evals" / "rose-bench.yaml"

_TERM_RE = re.compile(
    r"`([^`]+)`|"
    r"\b([A-Z][A-Z0-9_]{2,})\b|"
    r"\b(\d+(?:\.\d+)?(?:ms|s)?)\b|"
    r"(?:±\s*\d+%)|"
    r"\?purge=true|"
    r"Idempotency-Key|"
    r"forward-only|"
    r"detached"
)

_STOP = frozenset(
    "that this with from should write what when which about into your their "
    "have will would could make need want using used also only just".split()
)


@dataclass
class BenchCase:
    id: str
    kind: str
    task: str
    expected: str
    trap: str
    family: str = "default"
    lesson: str = ""
    requires: list[str] = field(default_factory=list)
    distractor_for: list[str] = field(default_factory=list)
    conflicting: list[str] = field(default_factory=list)
    extra_lesson: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchCase":
        return cls(
            id=str(raw.get("id") or ""),
            kind=str(raw.get("kind") or "trap"),
            task=str(raw.get("task") or "").strip(),
            expected=str(raw.get("expected") or "").strip(),
            trap=str(raw.get("trap") or "").strip(),
            family=str(raw.get("family") or "default"),
            lesson=str(raw.get("lesson") or "").strip(),
            requires=[str(x) for x in (raw.get("requires") or [])],
            distractor_for=[str(x) for x in (raw.get("distractor_for") or [])],
            conflicting=[str(x) for x in (raw.get("conflicting") or [])],
            extra_lesson=str(raw.get("extra_lesson") or "").strip(),
        )


@dataclass
class CaseScore:
    case_id: str
    kind: str
    arm: str
    passed: bool
    tokens: int
    reason: str = ""
    retrieval_ok: bool | None = None


@dataclass
class AxisScore:
    axis: str
    passed: int
    total: int
    kinds: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class BenchReport:
    cases: list[CaseScore] = field(default_factory=list)
    retention: dict[str, tuple[int, int]] = field(default_factory=dict)
    agent: str = "mock"
    bench_path: str = ""

    def lift(self) -> float:
        """L0 transfer minus control on core kinds."""
        core = {"trap", "detail", "principle", "multi"}
        ctrl = [c for c in self.cases if c.arm == CONTROL and c.kind in core]
        treat = [c for c in self.cases if c.arm == "L0" and c.kind in core]
        if not ctrl or not treat:
            return 0.0
        return (sum(1 for c in treat if c.passed) / len(treat)) - (
            sum(1 for c in ctrl if c.passed) / len(ctrl)
        )

    def axis(self, name: str, *, kinds: set[str] | None = None) -> AxisScore:
        rows = list(self.cases)
        if kinds is not None:
            rows = [c for c in rows if c.kind in kinds]
        if name == "transfer":
            rows = [c for c in rows if c.arm == "L0"]
        elif name == "retrieval":
            rows = [c for c in rows if c.retrieval_ok is not None]
        passed = sum(1 for c in rows if (c.retrieval_ok if name == "retrieval" else c.passed))
        kinds_out: dict[str, tuple[int, int]] = {}
        for kind in sorted({c.kind for c in rows}):
            sub = [c for c in rows if c.kind == kind]
            p = sum(1 for c in sub if (c.retrieval_ok if name == "retrieval" else c.passed))
            kinds_out[kind] = (p, len(sub))
        return AxisScore(axis=name, passed=passed, total=len(rows), kinds=kinds_out)

    def render(self) -> str:
        transfer = self.axis("transfer", kinds={"trap", "detail", "principle", "multi"})
        retrieval = self.axis("retrieval")
        lines = [
            f"ROSE-Bench — {len(self.cases)} scored arm(s), agent={self.agent}",
            f"bench: {self.bench_path or DEFAULT_BENCH}",
            "",
            f"Lift (L0 − control, core kinds): {self.lift():+.0%}",
            "",
            "Transfer @ L0 (trap|detail|principle|multi)",
            f"  {transfer.passed}/{transfer.total}  ({transfer.rate:.0%})",
        ]
        for kind, (p, n) in transfer.kinds.items():
            rate = f"{p / n:.0%}" if n else "—"
            lines.append(f"    {kind:<12} {p}/{n}  ({rate})")
        if self.retention:
            lines += ["", "Retention (detail cases, L0 vs L1)"]
            for level in sorted(self.retention):
                p, n = self.retention[level]
                rate = f"{p / n:.0%}" if n else "—"
                lines.append(f"  {level:<4} {p}/{n}  ({rate})")
        if retrieval.total:
            lines += [
                "",
                "Retrieval (distractor|null|conflict|multi)",
                f"  {retrieval.passed}/{retrieval.total}  ({retrieval.rate:.0%})",
            ]
            for kind, (p, n) in retrieval.kinds.items():
                rate = f"{p / n:.0%}" if n else "—"
                lines.append(f"    {kind:<12} {p}/{n}  ({rate})")
        cost = sum(c.tokens for c in self.cases if c.arm == "L0")
        n_l0 = sum(1 for c in self.cases if c.arm == "L0")
        if n_l0:
            lines += ["", f"Cost (mean L0 tokens): {cost // n_l0}"]
        return "\n".join(lines)


def load_bench(path: Path | None = None) -> tuple[list[BenchCase], dict[str, BenchCase]]:
    bench_path = path or DEFAULT_BENCH
    raw = yamlish.load(bench_path.read_text(encoding="utf-8"))
    cases = [BenchCase.from_dict(c) for c in (raw.get("cases") or [])]
    by_id = {c.id: c for c in cases if c.id}
    return cases, by_id


def key_terms(text: str) -> list[str]:
    found: list[str] = []
    for match in _TERM_RE.finditer(text or ""):
        term = next((g for g in match.groups() if g), match.group(0))
        term = term.strip()
        if len(term) >= 2:
            found.append(term)
    for phrase in re.findall(r"`[^`]+`|\?purge=true|forward-only|Idempotency-Key", text or ""):
        found.append(phrase.strip("`"))
    out: list[str] = []
    seen: set[str] = set()
    for term in found:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
    if out:
        return out[:12]
    return [text.split(".")[0][:40]] if text else []


def mock_grade(expected: str, candidate: str, *, kind: str) -> tuple[bool, str]:
    terms = key_terms(expected)
    if not candidate.strip():
        return False, "empty answer"
    lower = candidate.lower()
    hits = sum(1 for t in terms if t.lower() in lower)
    ratio = hits / len(terms) if terms else 0.0
    if kind in ("distractor", "null"):
        content_hits = sum(1 for t in terms if t.lower() in lower)
        ok = content_hits >= max(1, len(terms) * 0.35)
        return ok, f"content terms {content_hits}/{len(terms)}"
    if kind == "conflict":
        conflict_words = ("contradict", "conflict", "incompatible", "5433", "5434", "which", "verify")
        has_conflict = any(w in lower for w in conflict_words)
        return has_conflict, f"conflict={has_conflict}, terms {hits}/{len(terms)}"
    threshold = 0.5 if kind in ("principle", "multi") else 0.4
    ok = ratio >= threshold
    return ok, f"matched {hits}/{len(terms)} key terms ({ratio:.0%})"


def _overlap(task: str, body: str) -> float:
    task_words = {
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", task)
        if w.lower() not in _STOP
    }
    body_words = {
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", body)
        if w.lower() not in _STOP
    }
    if not task_words:
        return 0.0
    return len(task_words & body_words) / len(task_words)


def bench_adapter(base: Adapter | None = None) -> Adapter:
    """Adapter with bench-aware probe, grade, and relevance behaviour."""

    def router(prompt: str, schema: dict | None) -> Any:
        head = (prompt or "")[:4000].lower()
        if schema is JUDGE_SCHEMA or (schema and "pass" in (schema.get("properties") or {})):
            task = _section(prompt, "TASK")
            expected = _section(prompt, "KNOWN-GOOD") or _section(prompt, "EXPECTED")
            candidate = _section(prompt, "CANDIDATE")
            kind = "trap"
            if "regex" in task.lower():
                kind = "null"
            elif "5433" in task or "5434" in task:
                kind = "conflict"
            elif "ci pipeline" in task.lower() or "marketing site" in task.lower():
                kind = "distractor"
            ok, reason = mock_grade(expected, candidate, kind=kind)
            return {"pass": ok, "reason": reason}
        if "describe the approach" in head and "<<<lesson" in head:
            lesson = _section(prompt, "LESSON")
            if not lesson or lesson.strip() == "(no lesson available)":
                return "Use defaults: standard retry, plain DELETE, up/down migrations."
            return lesson[:1200]
        if schema and "picks" in (schema.get("properties") or {}):
            question = _section(prompt, "WORK") or _section(prompt, "QUESTION")
            picks = []
            qfacts = set(MockAdapter.facts(question))
            for ident, text in _candidates(prompt):
                shared = MockAdapter.facts(text) & qfacts
                # @recall-* tags are the primary signal; fall back to word overlap.
                score = _overlap(question, text)
                verdict = "relevant" if shared or score >= 0.22 else "unrelated"
                picks.append(
                    {
                        "id": ident,
                        "verdict": verdict,
                        "descend": False,
                        "why": f"shared={sorted(shared)}" if shared else f"overlap {score:.0%}",
                    }
                )
            return {"picks": picks}
        if base is not None and getattr(base, "router", None):
            return base.router(prompt, schema)
        return MockAdapter()._builtin(prompt, schema)

    if base is not None and getattr(base, "name", "") != "mock":
        return base
    return MockAdapter(router=router)


def pack_for_case(case: BenchCase, by_id: dict[str, BenchCase]) -> str:
    if case.kind in ("null",):
        return ""
    if case.kind == "multi":
        parts = [by_id[i].lesson for i in case.requires if i in by_id and by_id[i].lesson]
        return "\n\n---\n\n".join(parts)
    if case.kind == "conflict":
        parts = []
        for cid in case.conflicting:
            if cid in by_id and by_id[cid].lesson:
                parts.append(by_id[cid].lesson)
        if case.extra_lesson:
            parts.append(case.extra_lesson)
        return "\n\n---\n\n".join(parts)
    if case.kind == "distractor":
        wrong = [by_id[i].lesson for i in case.distractor_for if i in by_id and by_id[i].lesson]
        return "\n\n---\n\n".join(wrong[:1])
    return case.lesson


def ideal_pack(case: BenchCase) -> set[str]:
    if case.kind in ("null", "distractor"):
        return set()
    if case.kind == "multi":
        return set(case.requires)
    if case.kind == "conflict":
        ids = set(case.conflicting)
        if case.extra_lesson and "integration-pg-port-moved" in case.conflicting:
            ids.add("integration-pg-port-moved")
        return ids
    if case.lesson:
        return {case.id}
    return set()


def score_transfer(
    adapter: Adapter,
    case: BenchCase,
    *,
    arm: str,
    pack: str,
    samples: int,
    timeout: int,
) -> CaseScore:
    passed_runs = 0
    reasons: list[str] = []
    for _ in range(samples):
        answer = _probe(adapter, case.task, pack if arm != CONTROL else "", timeout)
        ok, why = _grade(adapter, case.task, case.expected, answer, timeout)
        if getattr(adapter, "name", "") == "mock" or not ok:
            ok2, why2 = mock_grade(case.expected, answer, kind=case.kind)
            if getattr(adapter, "name", "") == "mock":
                ok, why = ok2, why2
        passed_runs += 1 if ok else 0
        reasons.append(why)
    tokens = count_tokens(pack) if pack else 0
    rate = passed_runs / samples if samples else 0.0
    return CaseScore(
        case_id=case.id,
        kind=case.kind,
        arm=arm,
        passed=rate >= 0.5,
        tokens=tokens,
        reason=reasons[0] if reasons else "",
    )


def score_retrieval(
    store: Store,
    adapter: Adapter,
    case: BenchCase,
) -> CaseScore | None:
    if case.kind not in ("distractor", "null", "conflict", "multi"):
        return None
    ideal = ideal_pack(case)
    served = store.nodes()
    picks = Judge(store, adapter).relevance(case.task, served)
    kept = {p.id for p in picks if p.verdict != "unrelated"}
    if case.kind in ("null", "distractor"):
        ok = len(kept) == 0
    elif case.kind == "multi":
        ok = ideal <= kept
    else:
        ok = ideal <= kept
    return CaseScore(
        case_id=case.id,
        kind=case.kind,
        arm="retrieval",
        passed=ok,
        tokens=sum(n.tokens for n in served if n.id in kept),
        retrieval_ok=ok,
        reason=f"kept {sorted(kept)} want {sorted(ideal)}",
    )


def _extra_node_id(case: BenchCase) -> str | None:
    if not case.extra_lesson:
        return None
    match = re.search(r"^id:\s*(\S+)", case.extra_lesson, re.MULTILINE)
    return match.group(1) if match else f"{case.id}-alt"


def build_bench_store(cases: list[BenchCase], base: Path) -> Store:
    store = Store.init(base)
    written: set[str] = set()
    for case in cases:
        if case.lesson and case.id not in written:
            node = Node(
                id=case.id,
                family=case.family,
                level=0,
                title=case.id.replace("-", " "),
                gist=case.id.replace("-", " "),
                body=case.lesson,
                created=utcnow(),
                updated=utcnow(),
            )
            store.save_node(node)
            written.add(case.id)
        if case.extra_lesson:
            extra_id = _extra_node_id(case)
            if extra_id and extra_id not in written:
                body = re.sub(r"^id:\s*\S+\s*\n", "", case.extra_lesson, count=1, flags=re.MULTILINE)
                extra = Node(
                    id=extra_id,
                    family=case.family,
                    level=0,
                    title=extra_id.replace("-", " "),
                    gist=extra_id.replace("-", " "),
                    body=body.strip(),
                    created=utcnow(),
                    updated=utcnow(),
                )
                store.save_node(extra)
                written.add(extra_id)
    index_mod.rebuild(store)
    return store


def run_retention(
    store: Store,
    adapter: Adapter,
    case: BenchCase,
    *,
    samples: int,
    timeout: int,
) -> dict[str, tuple[bool, int]]:
    node = store.get(case.id)
    if node is None or not case.lesson:
        return {}
    episode = Episode(
        id=new_id("e"),
        family=case.family,
        prompt=case.task,
        outcome="success",
        used=[node.id],
        accepted_summary=case.expected,
    )
    store.save_episode(episode)
    node.stats.successes = 3
    store.save_node(node)
    store.invalidate()
    result = compress_node(
        store,
        adapter,
        node,
        skip_replay=getattr(adapter, "name", "") == "mock",
    )
    out: dict[str, tuple[bool, int]] = {}
    levels: list[tuple[str, str]] = [("L0", node.body)]
    if result.new_node:
        levels.append(("L1", result.new_node.body))
    for label, body in levels:
        passed = 0
        for _ in range(samples):
            answer = _probe(adapter, case.task, body, timeout)
            ok, _ = mock_grade(case.expected, answer, kind=case.kind)
            passed += 1 if ok else 0
        out[label] = (passed >= (samples + 1) // 2, count_tokens(body))
    return out


def run(
    adapter: Adapter,
    *,
    path: Path | None = None,
    samples: int = 1,
    retention: bool = True,
    retrieval: bool = True,
    kinds: set[str] | None = None,
    timeout: int = 180,
    on_progress: Callable[[BenchReport], None] | None = None,
) -> BenchReport:
    cases, by_id = load_bench(path)
    if kinds:
        cases = [c for c in cases if c.kind in kinds]
    report = BenchReport(agent=getattr(adapter, "name", "?"), bench_path=str(path or DEFAULT_BENCH))
    wrapped = bench_adapter(adapter)

    transfer_kinds = {"trap", "detail", "principle", "multi", "distractor", "null", "conflict"}
    transfer_cases = [c for c in cases if c.kind in transfer_kinds]
    for i, case in enumerate(transfer_cases, 1):
        pack = pack_for_case(case, by_id)
        for arm in (CONTROL, "L0"):
            score = score_transfer(
                wrapped,
                case,
                arm=arm,
                pack="" if arm == CONTROL else pack,
                samples=samples,
                timeout=timeout,
            )
            report.cases.append(score)
        print(f"  rose-bench transfer {i}/{len(transfer_cases)}: {case.id}", flush=True)
        if on_progress is not None:
            on_progress(report)

    if retrieval:
        with tempfile.TemporaryDirectory() as tmp:
            store = build_bench_store(cases, Path(tmp) / "repo")
            for case in cases:
                if case.kind in ("distractor", "null", "conflict", "multi"):
                    row = score_retrieval(store, wrapped, case)
                    if row:
                        report.cases.append(row)

    if retention:
        detail_cases = [c for c in cases if c.kind == "detail" and c.lesson][:3]
        retention_totals: dict[str, list[bool]] = {"L0": [], "L1": []}
        with tempfile.TemporaryDirectory() as tmp:
            for case in detail_cases:
                store = build_bench_store([case], Path(tmp) / case.id)
                levels = run_retention(store, wrapped, case, samples=samples, timeout=timeout)
                for level, (ok, _) in levels.items():
                    retention_totals.setdefault(level, []).append(ok)
        for level, vals in retention_totals.items():
            report.retention[level] = (sum(1 for v in vals if v), len(vals))

    return report


def to_dict(report: BenchReport) -> dict[str, Any]:
    transfer = report.axis("transfer", kinds={"trap", "detail", "principle", "multi"})
    retrieval = report.axis("retrieval")
    return {
        "agent": report.agent,
        "bench_path": report.bench_path,
        "lift": report.lift(),
        "transfer": {"passed": transfer.passed, "total": transfer.total, "rate": transfer.rate},
        "retrieval": {"passed": retrieval.passed, "total": retrieval.total, "rate": retrieval.rate},
        "retention": {k: {"passed": v[0], "total": v[1]} for k, v in report.retention.items()},
        "cases": [
            {
                "case_id": c.case_id,
                "kind": c.kind,
                "arm": c.arm,
                "passed": c.passed,
                "tokens": c.tokens,
                "retrieval_ok": c.retrieval_ok,
                "reason": c.reason,
            }
            for c in report.cases
        ],
    }
