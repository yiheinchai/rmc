"""Publication experiment suite — runs every quantitative eval ROSE needs.

All experiments are runnable without API keys (mock backend). When Claude/Codex
is available, pass ``agent='claude'`` for model-graded results.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import index as index_mod
from .adapters import get_adapter
from .adapters.mock import MockAdapter, MockWorld
from .bench import (
    DEFAULT_BENCH,
    bench_adapter,
    build_bench_store,
    load_bench,
    mock_grade,
    run as run_bench,
    to_dict as bench_to_dict,
)
from .compact import compress_node
from .evaluate import CONTROL, _probe
from .eval_recall import compare, run as run_recall
from .node import Node
from .recall import recall_pack, solve_with_descent
from .scaling import render_table, run_scaling
from .store import Episode, Store
from .util import count_tokens, new_id, utcnow
from .session_study import run as run_session_study
from .session_study import to_dict as session_study_to_dict
from .wikiskill import run as run_wikiskill
from .wikiskill import to_dict as wikiskill_to_dict

# Walkthrough lesson — multi-block so mock compression works.
WALKTHROUGH_LESSON = """When calling flaky remote services in this codebase, follow these rules.

- Retry only idempotent operations. @idempotent

- Use jittered exponential backoff. @backoff

- S3 can return HTTP 200 with an error body. Parse the body. @s3-body
"""

WALKTHROUGH_EPISODES = [
    ("e_http", "add retry to the http client", {"idempotent"}, "Retried idempotent ops only."),
    ("e_db", "make the db writer retry safely", {"idempotent", "backoff"}, "Used backoff."),
    ("e_queue", "queue consumer needs backoff", {"backoff"}, "Used jittered backoff."),
]


@dataclass
class ExperimentSuite:
    agent: str
    samples: int
    bench: dict[str, Any] = field(default_factory=dict)
    scaling: dict[str, Any] = field(default_factory=dict)
    recall: dict[str, Any] = field(default_factory=dict)
    compaction: dict[str, Any] = field(default_factory=dict)
    walkthrough: dict[str, Any] = field(default_factory=dict)
    retention_curve: dict[str, Any] = field(default_factory=dict)
    wikiskill: dict[str, Any] = field(default_factory=dict)
    session_study: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "samples": self.samples,
            "bench": self.bench,
            "scaling": self.scaling,
            "recall": self.recall,
            "compaction": self.compaction,
            "walkthrough": self.walkthrough,
            "retention_curve": self.retention_curve,
            "wikiskill": self.wikiskill,
            "session_study": self.session_study,
            "notes": self.notes,
        }


def _adapter(name: str):
    raw = get_adapter(name)
    if not raw.available():
        return bench_adapter(MockAdapter()), "mock"
    return bench_adapter(raw) if name == "mock" else raw, name


def _recall_report_dict(report) -> dict[str, Any]:
    return {
        "arm": report.arm,
        "precision": report.precision,
        "recall": report.recall_rate,
        "hits": report.hits,
        "kept": report.kept,
        "used": report.used,
        "misses": report.misses,
        "noise_tokens": report.noise_tokens,
        "useful_tokens": report.useful_tokens,
        "baseline_tokens": report.baseline_tokens,
        "episodes": len(report.scores),
        "searches": report.searches,
    }


def build_recall_fixture(base: Path) -> Store:
    """Bench lessons + episodes where served sets include noise lessons."""
    cases, _by_id = load_bench()
    store = build_bench_store(cases, base)
    lesson_ids = [c.id for c in cases if c.lesson]
    for case in cases:
        if not case.lesson and case.kind not in ("multi", "conflict"):
            continue
        tag = f"@recall-{case.id}"
        node = store.get(case.id)
        if node:
            if tag not in node.body:
                node.body = f"{node.body.rstrip()}\n\n{tag}"
            node.gist = f"{node.gist} {tag}"
            store.save_node(node)
        prompt = f"{case.task[:500]}\n\n{tag}"
        if case.kind == "multi":
            served = list(case.requires)
            used = list(case.requires)
            prompt = case.task[:500]
            for rid in case.requires:
                prompt += f" @recall-{rid}"
        elif case.kind == "conflict":
            served = list(case.conflicting)
            used = list(case.conflicting)
            prompt = case.task[:500]
        elif case.kind in ("distractor", "null"):
            served = case.distractor_for[:1] if case.distractor_for else []
            used = []
            prompt = case.task[:500]
        else:
            served = [case.id]
            noise = [i for i in lesson_ids if i != case.id][:2]
            served = served + noise
            used = [case.id]
        if not served:
            continue
        store.save_episode(
            Episode(
                id=new_id("e"),
                family=case.family,
                prompt=prompt,
                outcome="success",
                served=served,
                used=used,
                accepted_summary=case.expected[:400],
            )
        )
    index_mod.rebuild(store)
    return store


def run_recall_ablations(adapter, store: Store) -> dict[str, Any]:
    reports = {}
    for arm in ("serve-all", "judge", "agentic"):
        reports[arm] = run_recall(store, adapter, arm=arm)
    arms = {name: _recall_report_dict(r) for name, r in reports.items()}
    comparisons = {}
    if "judge" in reports and "serve-all" in reports:
        comparisons["judge_vs_serve_all"] = compare(reports["serve-all"], reports["judge"])
    return {"arms": arms, "comparisons": comparisons}


def run_compaction_ablation(adapter) -> dict[str, Any]:
    """Meta-tested vs skip-replay on a compressible walkthrough lesson."""
    world = MockWorld(
        {eid: facts for eid, _, facts, _ in WALKTHROUGH_EPISODES}
        | {"t_s3": {"idempotent", "s3-body"}}
    )
    replay_adapter = MockAdapter(world=world)

    def _seed_store(root: Path) -> tuple[Store, Node]:
        store = Store.init(root)
        node = Node(
            id="n_L0",
            family="retry",
            title="Retrying flaky services",
            body=WALKTHROUGH_LESSON,
            level=0,
            gist="retry flaky services",
        )
        store.save_node(node)
        for eid, prompt, _facts, summary in WALKTHROUGH_EPISODES:
            store.save_episode(
                Episode(
                    id=eid,
                    family="retry",
                    prompt=prompt,
                    outcome="success",
                    served=[node.id],
                    used=[node.id],
                    accepted_summary=summary,
                )
            )
        node = store.get(node.id)
        node.stats.successes = len(WALKTHROUGH_EPISODES)
        store.save_node(node)
        store.invalidate()
        return store, node

    with tempfile.TemporaryDirectory() as tmp:
        store_a, node_a = _seed_store(Path(tmp) / "a")
        with_meta = compress_node(store_a, replay_adapter, node_a, skip_replay=False)

        store_b, node_b = _seed_store(Path(tmp) / "b")
        without_meta = compress_node(store_b, replay_adapter, node_b, skip_replay=True)

        return {
            "with_meta_testing": {
                "accepted": with_meta.accepted,
                "reason": with_meta.reason,
                "before_tokens": with_meta.before_tokens,
                "after_tokens": with_meta.after_tokens,
                "pass_rate": with_meta.pass_rate,
                "replays": len(with_meta.replays),
            },
            "without_meta_testing": {
                "accepted": without_meta.accepted,
                "reason": without_meta.reason,
                "before_tokens": without_meta.before_tokens,
                "after_tokens": without_meta.after_tokens,
            },
        }


def run_walkthrough_cycle(adapter) -> dict[str, Any]:
    world = MockWorld(
        {eid: facts for eid, _, facts, _ in WALKTHROUGH_EPISODES}
        | {"t_s3": {"idempotent", "s3-body"}}
    )
    mock = MockAdapter(world=world) if adapter.name == "mock" else adapter
    with tempfile.TemporaryDirectory() as tmp:
        store = Store.init(Path(tmp) / "repo")
        base = Node(
            id="n_L0",
            family="retry",
            title="Retrying flaky services",
            body=WALKTHROUGH_LESSON,
            level=0,
            gist="retry flaky services",
        )
        store.save_node(base)
        for eid, prompt, _facts, summary in WALKTHROUGH_EPISODES:
            store.save_episode(
                Episode(
                    id=eid,
                    family="retry",
                    prompt=prompt,
                    outcome="success",
                    served=[base.id],
                    accepted_summary=summary,
                )
            )
        base = store.get(base.id)
        base.stats.successes = len(WALKTHROUGH_EPISODES)
        store.save_node(base)
        store.invalidate()

        result = compress_node(store, mock, store.get(base.id))
        apex = result.new_node
        pack_before = recall_pack(store, "the http client needs retry logic", mock)

        def verify(run, pack_text):
            ok, missing = world.solves("t_s3", pack_text)
            return ok, "missing: " + " ".join(f"@{m}" for m in sorted(missing))

        descent = solve_with_descent(
            store,
            adapter=MockAdapter(world=world),
            task_id="t_s3",
            task="handle the s3 upload response correctly",
            family="retry",
            verify=verify,
        )

        return {
            "compression_accepted": result.accepted,
            "compression_ratio": result.ratio,
            "replay_pass_rate": result.pass_rate,
            "apex_level": apex.level if apex else None,
            "apex_tokens": apex.tokens if apex else None,
            "dropped_claims": len(apex.dropped) if apex else 0,
            "recall_tokens_before": pack_before.tokens,
            "descent_attempts": len(descent.attempts),
            "descent_rescued": descent.rescued_by.label if descent.rescued_by else None,
            "descent_escalated": descent.escalated,
        }


def run_retention_curve(adapter, samples: int) -> dict[str, Any]:
    """Transfer@level on walkthrough lesson after real compression."""
    world = MockWorld(
        {eid: facts for eid, _, facts, _ in WALKTHROUGH_EPISODES}
        | {"t_s3": {"idempotent", "s3-body"}}
    )
    mock = MockAdapter(world=world)
    with tempfile.TemporaryDirectory() as tmp:
        store = Store.init(Path(tmp) / "repo")
        base = Node(
            id="n_L0",
            family="retry",
            title="Retrying flaky services",
            body=WALKTHROUGH_LESSON,
            level=0,
            gist="retry flaky services",
        )
        store.save_node(base)
        for eid, prompt, facts, summary in WALKTHROUGH_EPISODES:
            store.save_episode(
                Episode(
                    id=eid,
                    family="retry",
                    prompt=prompt,
                    outcome="success",
                    served=[base.id],
                    used=[base.id],
                    accepted_summary=summary,
                )
            )
        base = store.get(base.id)
        base.stats.successes = len(WALKTHROUGH_EPISODES)
        store.save_node(base)
        store.invalidate()

        result = compress_node(store, mock, store.get(base.id))
        levels: dict[str, dict[str, Any]] = {}
        task_id = "t_s3"
        task = "handle the s3 upload response correctly"

        def score_body(body: str) -> float:
            passes = 0
            for _ in range(samples):
                ans = _probe(adapter, task, body, 60)
                ok, _ = world.solves(task_id, ans if body else "")
                passes += int(ok)
            return passes / samples

        levels[CONTROL] = {"pass_rate": score_body(""), "tokens": 0}
        levels["L0"] = {"pass_rate": score_body(base.body), "tokens": base.tokens}

        if result.accepted and result.new_node:
            levels[f"L{result.new_node.level}"] = {
                "pass_rate": score_body(result.new_node.body),
                "tokens": result.new_node.tokens,
                "compression_accepted": True,
                "replay_pass_rate": result.pass_rate,
            }
        return {"levels": levels, "compression": {"accepted": result.accepted, "ratio": result.ratio}}


def run_all(*, agent: str = "mock", samples: int = 3) -> ExperimentSuite:
    adapter, effective = _adapter(agent)
    suite = ExperimentSuite(agent=effective, samples=samples)
    if effective != agent:
        suite.notes.append(f"requested agent {agent!r} unavailable; used mock")

    bench_report = run_bench(adapter, samples=samples)
    suite.bench = bench_to_dict(bench_report)
    suite.bench["render"] = bench_report.render()

    scaling_rows = run_scaling([25, 100, 500, 1000])
    suite.scaling = {
        "rows": [r.to_dict() for r in scaling_rows],
        "table": render_table(scaling_rows),
    }

    with tempfile.TemporaryDirectory() as tmp:
        store = build_recall_fixture(Path(tmp) / "repo")
        suite.recall = run_recall_ablations(adapter, store)
        if effective == "mock":
            suite.notes.append(
                "agentic recall arm requires claude/codex subprocess; skipped on mock"
            )
            suite.recall["arms"].pop("agentic", None)
        else:
            try:
                agentic = run_recall(store, adapter, arm="agentic")
                suite.recall["arms"]["agentic"] = _recall_report_dict(agentic)
            except Exception as exc:
                suite.notes.append(f"agentic recall failed: {exc}")

    suite.compaction = run_compaction_ablation(adapter)
    suite.walkthrough = run_walkthrough_cycle(adapter)
    suite.retention_curve = run_retention_curve(adapter, samples)
    ws_report = run_wikiskill(adapter, samples=samples)
    suite.wikiskill = wikiskill_to_dict(ws_report)
    ss_report = run_session_study(adapter, samples=samples)
    suite.session_study = session_study_to_dict(ss_report)

    return suite


def render_summary(suite: ExperimentSuite) -> str:
    lines = [
        f"ROSE Experiment Suite — agent={suite.agent}, samples={suite.samples}",
        "",
        "=== ROSE-Bench ===",
        suite.bench.get("render", ""),
        "",
        "=== Scaling ===",
        suite.scaling.get("table", ""),
        "",
        "=== Recall ablations ===",
    ]
    for arm, data in suite.recall.get("arms", {}).items():
        lines.append(
            f"  {arm:<12} prec={data['precision']:.0%}  rec={data['recall']:.0%}  "
            f"noise={data['noise_tokens']} tok"
        )
    lines += ["", "=== Compaction ablation ==="]
    for key in ("with_meta_testing", "without_meta_testing"):
        row = suite.compaction.get(key, {})
        lines.append(f"  {key}: accepted={row.get('accepted')} ratio={row.get('after_tokens', 0)}/{row.get('before_tokens', 0)}")
    lines += ["", "=== Walkthrough cycle ==="]
    for k, v in suite.walkthrough.items():
        lines.append(f"  {k}: {v}")
    lines += ["", "=== Retention curve (S3 task) ==="]
    for level, data in suite.retention_curve.get("levels", {}).items():
        lines.append(f"  {level:<8} transfer={data.get('pass_rate', 0):.0%}  tokens={data.get('tokens', 0)}")
    if suite.wikiskill.get("render"):
        lines += ["", "=== WikiSkill-comparable ===", suite.wikiskill["render"]]
    if suite.session_study.get("render"):
        lines += ["", "=== Session paired study ===", suite.session_study["render"]]
    if suite.notes:
        lines += ["", "Notes:", *[f"  - {n}" for n in suite.notes]]
    return "\n".join(lines)
