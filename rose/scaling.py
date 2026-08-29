"""Synthetic scaling study for ROSE retrieval cost.

Generates stores of varying size and measures index footprint, apex count,
and mock retrieval behaviour. Used for the paper's scaling table.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import index as index_mod
from .adapters.mock import MockAdapter
from .bench import bench_adapter
from .eval_recall import run as run_recall
from .node import Node
from .store import Episode, Store
from .util import count_tokens, new_id, utcnow


@dataclass
class ScalingRow:
    lessons: int
    apexes: int
    index_tokens: int
    index_lines: int
    routing_tokens_per_apex: int = 55
    mean_served: int = 0
    judge_precision: float = 0.0
    judge_recall: float = 0.0

    @property
    def routing_tokens(self) -> int:
        return self.apexes * self.routing_tokens_per_apex

    def to_dict(self) -> dict[str, Any]:
        return {
            "lessons": self.lessons,
            "apexes": self.apexes,
            "index_tokens": self.index_tokens,
            "index_lines": self.index_lines,
            "routing_tokens_per_prompt": self.routing_tokens,
            "mean_served": self.mean_served,
            "judge_precision": self.judge_precision,
            "judge_recall": self.judge_recall,
        }


def _synthetic_nodes(n: int) -> list[Node]:
    nodes: list[Node] = []
    for i in range(n):
        family = f"family-{i % max(1, n // 10)}"
        body = (
            f"Lesson {i}: when working on {family}, prefer approach pattern-{i % 7}. "
            f"Trap to avoid: default-{i % 5}. @fact-{i % 13}"
        )
        nodes.append(
            Node(
                id=new_id("n"),
                family=family,
                level=0,
                title=f"lesson {i}",
                gist=f"pattern for {family}",
                body=body,
                created=utcnow(),
                updated=utcnow(),
            )
        )
    return nodes


def build_synthetic_store(n: int, base: Path) -> Store:
    store = Store.init(base)
    for node in _synthetic_nodes(n):
        store.save_node(node)
    index_mod.rebuild(store)
    return store


def _seed_episodes(store: Store, n_episodes: int = 12) -> None:
    nodes = store.nodes()
    if not nodes:
        return
    for i in range(n_episodes):
        served = [nodes[i % len(nodes)].id]
        used = served if i % 3 else []
        store.save_episode(
            Episode(
                id=new_id("e"),
                family=nodes[i % len(nodes)].family,
                prompt=f"task about {nodes[i % len(nodes)].family} pattern-{i % 7}",
                outcome="success",
                served=served,
                used=used,
                accepted_summary=f"used pattern-{i % 7}",
            )
        )


def measure(store: Store, adapter: MockAdapter | None = None) -> ScalingRow:
    nodes = store.nodes()
    apexes = [n for n in nodes if n.is_apex]
    index_path = store.root / "index.md"
    index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    lines = [ln for ln in index_text.splitlines() if ln.startswith("|") and "n_" in ln]
    row = ScalingRow(
        lessons=len(nodes),
        apexes=len(apexes),
        index_tokens=count_tokens(index_text),
        index_lines=len(lines),
    )
    episodes = store.episodes()
    if episodes:
        row.mean_served = round(sum(len(e.served) for e in episodes) / len(episodes))
    if adapter is not None and episodes:
        report = run_recall(store, adapter, limit=len(episodes), arm="judge")
        row.judge_precision = report.precision
        row.judge_recall = report.recall_rate
    return row


def run_scaling(
    sizes: list[int] | None = None,
    *,
    episodes: int = 12,
) -> list[ScalingRow]:
    sizes = sizes or [25, 100, 500]
    adapter = bench_adapter(MockAdapter())
    rows: list[ScalingRow] = []
    for n in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            store = build_synthetic_store(n, Path(tmp) / "repo")
            _seed_episodes(store, episodes)
            rows.append(measure(store, adapter))
    return rows


def render_table(rows: list[ScalingRow]) -> str:
    lines = [
        "| lessons | apexes | index tok | routing tok/prompt | judge prec | judge rec |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.lessons} | {row.apexes} | {row.index_tokens} | "
            f"{row.routing_tokens} | {row.judge_precision:.0%} | {row.judge_recall:.0%} |"
        )
    return "\n".join(lines)
