"""SealQA ablation harness — component tests for ROSE compaction + probes.

Unlike WikiSkill upstream scoring (end-to-end QA accuracy on broken evidence),
this module holds task prompts fixed and varies ROSE configuration presets to
measure probe replay, compaction acceptance, and optional task accuracy.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import index as index_mod
from . import probes as probes_mod
from . import yamlish
from .adapters import Adapter
from .adapters.mock import MockAdapter
from .bench import _grade, _probe, mock_grade
from .compact import compress_node, validate
from .node import Node
from .store import Episode, Store
from .wikiskill import WikiSkillCase, wikiskill_adapter

DEFAULT_PROBE_DEV = (
    Path(__file__).resolve().parents[1] / "evals" / "sealqa-ablation" / "probe-dev.yaml"
)

PRESETS = (
    "baseline",
    "compact-no-replay",
    "compact-probe-replay",
    "compact-episode-replay",
    "probes-off",
)


@dataclass
class ProbeDevBench:
    lesson: str
    cases: list[WikiSkillCase]
    axes: dict[str, str]  # case_id -> axis

    @classmethod
    def load(cls, path: Path | None = None) -> "ProbeDevBench":
        bench_path = path or DEFAULT_PROBE_DEV
        raw = yamlish.load(bench_path.read_text(encoding="utf-8"))
        lesson = str(raw.get("lesson") or "").strip()
        cases: list[WikiSkillCase] = []
        axes: dict[str, str] = {}
        for c in raw.get("cases") or []:
            if not c.get("id"):
                continue
            case_id = str(c["id"])
            axes[case_id] = str(c.get("axis") or case_id)
            cases.append(
                WikiSkillCase.from_dict(
                    {**c, "skill": c.get("skill") or lesson, "benchmark": "SealQA"}
                )
            )
        return cls(lesson=lesson, cases=cases, axes=axes)


@dataclass
class PresetResult:
    preset: str
    probe_pass_rate: float = 0.0
    probe_passed: int = 0
    probe_total: int = 0
    compaction_accepted: bool = False
    compaction_reason: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    probe_count: int = 0
    episode_count: int = 0
    by_axis: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_accuracy: float | None = None
    task_passed: int | None = None
    task_total: int | None = None
    replay_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "probe_pass_rate": self.probe_pass_rate,
            "probe_passed": self.probe_passed,
            "probe_total": self.probe_total,
            "compaction_accepted": self.compaction_accepted,
            "compaction_reason": self.compaction_reason,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "probe_count": self.probe_count,
            "episode_count": self.episode_count,
            "by_axis": self.by_axis,
            "task_accuracy": self.task_accuracy,
            "task_passed": self.task_passed,
            "task_total": self.task_total,
            "replay_details": self.replay_details,
        }


def _build_store(bench: ProbeDevBench, base: Path) -> tuple[Store, Node]:
    store = Store.init(base)
    node = Node(
        id="sealqa-master",
        family="sealqa",
        title="SealQA evidence handling",
        gist="SealQA: answer from snippets with correct format",
        body=bench.lesson,
        level=0,
    )
    store.save_node(node)
    index_mod.rebuild(store)
    return store, node


def _seed_probes_with_axes(store: Store, node: Node, bench: ProbeDevBench) -> int:
    count = 0
    for case in bench.cases:
        probe = probes_mod.seed_from_case(
            store,
            case_id=case.id,
            node_id=node.id,
            task=case.task,
            expected=case.expected,
            axis=bench.axes.get(case.id, case.id),
            enforce_cap=False,
        )
        if probe is not None:
            count += 1
    probes_mod.prune_to_cap(store, node.id)
    node.stats.successes = max(node.stats.successes, 3)
    store.save_node(node)
    store.invalidate()
    return count


def _seed_episodes(store: Store, node: Node, cases: list[WikiSkillCase]) -> int:
    count = 0
    for case in cases:
        ep = Episode(
            id=f"e_{case.id}",
            family="sealqa",
            prompt=probes_mod.minimal_task(case.task),
            outcome="success",
            served=[node.id],
            used=[node.id],
            accepted_summary=case.expected,
        )
        store.save_episode(ep)
        if ep.id not in node.covers_tasks:
            node.covers_tasks = sorted({*node.covers_tasks, ep.id})
        count += 1
    store.save_node(node)
    store.invalidate()
    return count


def _replay_adapter_for_cases(cases: list[WikiSkillCase]) -> MockAdapter:
    by_task = {probes_mod.minimal_task(c.task): c for c in cases}
    by_id = {f"p_{c.id}": c for c in cases}

    def router(prompt: str, schema: dict | None) -> Any:
        head = (prompt or "")[:4000].lower()
        if schema and "pass" in (schema.get("properties") or {}):
            from .adapters.mock import _section

            expected = _section(prompt, "KNOWN-GOOD") or _section(prompt, "EXPECTED")
            candidate = _section(prompt, "CANDIDATE")
            ok, reason = mock_grade(expected, candidate, kind="trap")
            return {"pass": ok, "reason": reason}
        if "describe the approach" in head:
            from .adapters.mock import _section

            task = _section(prompt, "TASK").strip()
            case = by_task.get(task)
            pack = _section(prompt, "LESSON")
            if case and pack and "(no lesson available)" not in pack:
                return case.expected
            return "Answer: unknown"
        return {"body": "Answer from snippets only. @header-metadata @explicit-count", "dropped": [], "lossless": True}

    return MockAdapter(router=router)


def _compress_router_aggressive() -> Callable:
    """Simulator: compression that drops a norm to test replay gate."""

    def router(prompt: str, schema: dict | None) -> Any:
        if "ROSE:compress" in (prompt or ""):
            return {
                "body": (
                    "Answer from snippets. Format: Answer: <fact>. "
                    "Prefer header metadata when present."
                ),
                "dropped": [
                    {"claim": "never count et al.", "kind": "edge-case"},
                    {"claim": "reject false premises", "kind": "precondition"},
                ],
                "lossless": False,
            }
        if "ROSE:worth" in (prompt or ""):
            return {"keep": True, "generality": "more", "why": "shorter and broader"}
        return {"pass": True, "reason": "ok", "missing": []}

    return router


def _probe_replay_pass_rate(
    store: Store,
    adapter: Adapter,
    node: Node,
    bench: ProbeDevBench,
) -> tuple[float, list[dict[str, Any]]]:
    probe_list = store.regression_set(node)
    if not probe_list:
        return 0.0, []
    outcomes = validate(store, adapter, node.body, probe_list)
    details = []
    passed = 0
    axis_by_id = {f"p_{cid}": axis for cid, axis in bench.axes.items()}
    for outcome, episode in zip(outcomes, probe_list):
        ok = outcome.ok
        passed += 1 if ok else 0
        details.append(
            {
                "probe_id": episode.id,
                "axis": axis_by_id.get(episode.id, "unknown"),
                "ok": ok,
                "reason": outcome.reason[:200],
            }
        )
    rate = passed / len(outcomes) if outcomes else 0.0
    return rate, details


def _aggregate_by_axis(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in details:
        axis = row.get("axis") or "unknown"
        bucket = out.setdefault(axis, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += 1 if row.get("ok") else 0
    for axis, bucket in out.items():
        bucket["pass_rate"] = bucket["passed"] / bucket["total"] if bucket["total"] else 0.0
    return out


def _task_accuracy(
    store: Store,
    adapter: Adapter,
    node: Node,
    cases: list[WikiSkillCase],
) -> tuple[float, int, int]:
    wrapped = wikiskill_adapter(adapter, cases)
    pack = node.body
    passed = 0
    for case in cases:
        answer = _probe(wrapped, case.task, pack, timeout=120)
        ok, _ = _grade(wrapped, case.task, case.expected, answer, timeout=120)
        if getattr(wrapped, "name", "") == "mock":
            ok, _ = mock_grade(case.expected, answer, kind="trap")
        passed += 1 if ok else 0
    return passed / len(cases) if cases else 0.0, passed, len(cases)


def run_preset(
    bench: ProbeDevBench,
    preset: str,
    adapter: Adapter,
    *,
    score_tasks: bool = True,
) -> PresetResult:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; want one of {PRESETS}")

    replay_adapter = _replay_adapter_for_cases(bench.cases)
    compress_adapter = MockAdapter(router=_compress_router_aggressive())

    with tempfile.TemporaryDirectory() as tmp:
        store, node = _build_store(bench, Path(tmp) / "repo")
        result = PresetResult(preset=preset, tokens_before=node.tokens)

        if preset != "probes-off":
            result.probe_count = _seed_probes_with_axes(store, node, bench)
        if preset == "compact-episode-replay":
            result.episode_count = _seed_episodes(store, node, bench.cases)
            # Episodes only: remove probes so regression_set falls back to episodes
            for probe in probes_mod.load_for_node(store, node.id):
                probes_mod.delete(store, probe)
            node.probes = []
            store.save_node(node)
            store.invalidate()

        node = store.get(node.id)
        lesson_body = node.body

        if preset in ("compact-no-replay", "compact-probe-replay", "compact-episode-replay"):
            compact = compress_node(
                store,
                compress_adapter,
                node,
                skip_replay=(preset == "compact-no-replay"),
            )
            result.compaction_accepted = compact.accepted
            result.compaction_reason = compact.reason
            result.tokens_after = compact.after_tokens or node.tokens
            if compact.accepted and compact.new_node is not None:
                node = compact.new_node
                lesson_body = node.body
            elif compact.after_tokens:
                lesson_body = compact.new_node.body if compact.new_node else node.body

        rate, details = _probe_replay_pass_rate(
            store, replay_adapter, store.get(node.id) or node, bench
        )
        result.probe_pass_rate = rate
        result.probe_passed = sum(1 for d in details if d.get("ok"))
        result.probe_total = len(details)
        result.replay_details = details
        result.by_axis = _aggregate_by_axis(details)

        if score_tasks:
            acc, tp, tt = _task_accuracy(store, adapter, store.get(node.id) or node, bench.cases)
            result.task_accuracy = acc
            result.task_passed = tp
            result.task_total = tt

        if preset in ("baseline", "probes-off"):
            result.compaction_reason = "no compaction"
            result.tokens_after = result.tokens_before

        return result


@dataclass
class AblationReport:
    bench_path: str
    agent: str
    presets: list[PresetResult] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"SealQA ablation — {self.bench_path}",
            f"agent={self.agent}  cases={self.presets[0].probe_total if self.presets else 0}",
            "",
            f"{'preset':<26} {'probe_pass':>10} {'compact':>8} {'tok_before':>10} {'tok_after':>9} {'task_acc':>8}",
        ]
        for row in self.presets:
            acc = f"{row.task_accuracy:.0%}" if row.task_accuracy is not None else "—"
            lines.append(
                f"{row.preset:<26} {row.probe_passed}/{row.probe_total} ({row.probe_pass_rate:.0%})"
                f"  {str(row.compaction_accepted):>8} {row.tokens_before:>10} {row.tokens_after:>9} {acc:>8}"
            )
        lines.append("")
        lines.append("Per-axis probe pass (compact-probe-replay vs compact-no-replay):")
        replay = next((p for p in self.presets if p.preset == "compact-probe-replay"), None)
        noreplay = next((p for p in self.presets if p.preset == "compact-no-replay"), None)
        if replay and noreplay:
            axes = sorted(set(replay.by_axis) | set(noreplay.by_axis))
            for axis in axes:
                r = replay.by_axis.get(axis, {})
                n = noreplay.by_axis.get(axis, {})
                lines.append(
                    f"  {axis:<20} replay={r.get('passed', 0)}/{r.get('total', 0)}"
                    f"  no-replay={n.get('passed', 0)}/{n.get('total', 0)}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bench_path": self.bench_path,
            "agent": self.agent,
            "presets": [p.to_dict() for p in self.presets],
            "render": self.render(),
        }


def run(
    adapter: Adapter,
    *,
    path: Path | None = None,
    presets: tuple[str, ...] | None = None,
    score_tasks: bool = True,
) -> AblationReport:
    bench = ProbeDevBench.load(path)
    use = presets or PRESETS
    report = AblationReport(
        bench_path=str(path or DEFAULT_PROBE_DEV),
        agent=getattr(adapter, "name", "?"),
    )
    for preset in use:
        report.presets.append(
            run_preset(bench, preset, adapter, score_tasks=score_tasks)
        )
    return report
