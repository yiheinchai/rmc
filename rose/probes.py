"""Minimal regression probes: distilled tasks that drive compaction.

Each attributed use of a lesson can spawn a probe — a stripped-down task that
captures the *crux* of what the lesson enabled, not the session it came from.
Probes accumulate across contexts so compression must generalize to pass them all,
not concatenate scenarios.

Probes live outside ``nodes/`` (in ``probes/``) so they stay replay artefacts,
not retrievable knowledge. A node keeps probe ids in frontmatter for bookkeeping;
the regression set walks the node's subtree and selects up to ``max_probes``
orthogonal cases for replay.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .prompts import DISTILL_PROBE, DISTILL_PROBE_SCHEMA
from .util import new_id, truncate, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .adapters import Adapter
    from .node import Node
    from .store import Store

PROBES_DIRNAME = "probes"


@dataclass
class Probe:
    """One minimal task that a compressed lesson must still transfer."""

    id: str
    node_id: str
    task: str
    outcome: str
    axis: str = "general"
    source_episode: str = ""
    created: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "task": self.task,
            "outcome": self.outcome,
            "axis": self.axis,
            "source_episode": self.source_episode,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Probe":
        return cls(
            id=str(raw.get("id") or ""),
            node_id=str(raw.get("node_id") or ""),
            task=str(raw.get("task") or ""),
            outcome=str(raw.get("outcome") or ""),
            axis=str(raw.get("axis") or "general"),
            source_episode=str(raw.get("source_episode") or ""),
            created=str(raw.get("created") or utcnow()),
        )


def probes_dir(store: "Store") -> Path:
    return store.root / PROBES_DIRNAME


def _path(store: "Store", probe_id: str) -> Path:
    return probes_dir(store) / f"{probe_id}.json"


def save(store: "Store", probe: Probe) -> Path:
    directory = probes_dir(store)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path(store, probe.id)
    path.write_text(json.dumps(probe.to_dict(), indent=2), encoding="utf-8")
    return path


def delete(store: "Store", probe: Probe) -> None:
    path = _path(store, probe.id)
    if path.exists():
        path.unlink()


def get(store: "Store", probe_id: str) -> Probe | None:
    path = _path(store, probe_id)
    if not path.exists():
        return None
    try:
        return Probe.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def load_for_node(store: "Store", node_id: str) -> list[Probe]:
    directory = probes_dir(store)
    if not directory.is_dir():
        return []
    out: list[Probe] = []
    for path in sorted(directory.glob("*.json")):
        try:
            probe = Probe.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if probe.node_id == node_id:
            out.append(probe)
    return sorted(out, key=lambda p: p.created)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def _norm_axis(axis: str) -> str:
    return _norm(axis) or "general"


def _overlap(a: str, b: str) -> float:
    wa, wb = set(_norm(a).split()), set(_norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _duplicate(existing: Probe, candidate: Probe) -> bool:
    if _norm_axis(existing.axis) == _norm_axis(candidate.axis):
        return True
    if _overlap(existing.task, candidate.task) >= 0.72:
        return True
    return False


def _subtree_ids(store: "Store", node: "Node") -> set[str]:
    ids = {node.id}
    for desc in store.descendants(node):
        ids.add(desc.id)
    return ids


def collect(store: "Store", node: "Node") -> list[Probe]:
    """All probes on this node and its descendants."""
    ids = _subtree_ids(store, node)
    directory = probes_dir(store)
    if not directory.is_dir():
        return []
    out: list[Probe] = []
    for path in sorted(directory.glob("*.json")):
        try:
            probe = Probe.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if probe.node_id in ids:
            out.append(probe)
    return sorted(out, key=lambda p: p.created)


def select_for_replay(store: "Store", node: "Node", *, limit: int | None = None) -> list[Probe]:
    """Pick orthogonal probes for compaction replay — one per axis first."""
    probes = collect(store, node)
    if not probes:
        return []
    limit = limit if limit is not None else len(probes)

    selected: list[Probe] = []
    seen_axes: set[str] = set()
    for probe in sorted(probes, key=lambda p: p.created, reverse=True):
        axis = _norm_axis(probe.axis)
        if axis in seen_axes:
            continue
        selected.append(probe)
        seen_axes.add(axis)
        if len(selected) >= limit:
            return selected

    for probe in sorted(probes, key=lambda p: p.created, reverse=True):
        if probe in selected:
            continue
        selected.append(probe)
        if len(selected) >= limit:
            break
    return selected[:limit]


def render_for_compress(probes: list[Probe]) -> str:
    if not probes:
        return "(none)"
    lines = []
    for probe in probes:
        lines.append(
            f"- [{probe.id}] axis={probe.axis}: {probe.task.strip()} "
            f"→ expect: {probe.outcome.strip()}"
        )
    return "\n".join(lines)


def _attach_to_node(store: "Store", node_id: str, probe_id: str) -> None:
    node = store.get(node_id)
    if node is None:
        return
    if probe_id not in node.probes:
        node.probes = sorted({*node.probes, probe_id})
        store.save_node(node)


def _detach_from_node(store: "Store", node_id: str, probe_id: str) -> None:
    node = store.get(node_id)
    if node is None or probe_id not in node.probes:
        return
    node.probes = [p for p in node.probes if p != probe_id]
    store.save_node(node)


def add(store: "Store", probe: Probe, *, enforce_cap: bool = True) -> Probe | None:
    """Store a probe if it adds a new axis; optionally maintain the cap."""
    if not probe.task.strip() or not probe.outcome.strip():
        return None

    max_probes = int(store.config.get("compaction.max_probes", 10))
    existing = load_for_node(store, probe.node_id)
    if any(_duplicate(p, probe) for p in existing):
        return None

    if enforce_cap and len(existing) >= max_probes:
        existing.sort(key=lambda p: p.created)
        drop = existing[0]
        delete(store, drop)
        _detach_from_node(store, probe.node_id, drop.id)

    save(store, probe)
    _attach_to_node(store, probe.node_id, probe.id)
    store.log(
        "probe",
        probe=probe.id,
        node=probe.node_id,
        axis=probe.axis,
        source=probe.source_episode or None,
    )
    return probe


def distill(
    store: "Store",
    adapter: "Adapter",
    *,
    node_id: str,
    lesson_body: str,
    task: str,
    outcome: str,
    context: str = "",
    source_episode: str = "",
) -> Probe | None:
    """Distil a real use into a minimal standalone probe."""
    if not store.config.get("compaction.probe_distill", True):
        return None
    task, outcome = task.strip(), outcome.strip()
    if not task or not outcome:
        return None

    run = adapter.run(
        DISTILL_PROBE.format(
            lesson=truncate(lesson_body, 4000),
            task=truncate(task, 2000),
            outcome=truncate(outcome, 1500),
            context=truncate(context, 3000) or "(no extra context)",
        ),
        schema=DISTILL_PROBE_SCHEMA,
        timeout=int(store.config.get("limits.agent_timeout_s", 180)),
    )
    if not run.ok or not run.data:
        return None

    minimal_task = str(run.data.get("task") or "").strip()
    minimal_outcome = str(run.data.get("outcome") or "").strip()
    axis = str(run.data.get("axis") or "general").strip()
    if not minimal_task or not minimal_outcome:
        return None

    probe = Probe(
        id=new_id("p"),
        node_id=node_id,
        task=minimal_task,
        outcome=minimal_outcome,
        axis=axis or "general",
        source_episode=source_episode,
    )
    return add(store, probe)


def minimal_task(text: str) -> str:
    """Strip evidence blocks; keep the standalone question."""
    lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(
            ("evidence snippets", "reference context", "reference urls", "answer with")
        ):
            break
        lines.append(stripped)
    out = " ".join(lines).strip()
    return out[:600] if out else (text or "").strip()[:600]


def prune_to_cap(store: "Store", node_id: str) -> list[str]:
    """Keep the most orthogonal probes up to ``compaction.max_probes``."""
    node = store.get(node_id)
    if node is None:
        return []
    max_probes = int(store.config.get("compaction.max_probes", 10))
    kept = select_for_replay(store, node, limit=max_probes)
    keep_ids = {p.id for p in kept}
    dropped: list[str] = []
    for probe in load_for_node(store, node_id):
        if probe.id not in keep_ids:
            delete(store, probe)
            _detach_from_node(store, node_id, probe.id)
            dropped.append(probe.id)
    return dropped


def seed_from_case(
    store: "Store",
    *,
    case_id: str,
    node_id: str,
    task: str,
    expected: str,
    axis: str = "",
    enforce_cap: bool = False,
) -> Probe | None:
    """Heuristic probe from a benchmark case (no model call)."""
    probe = Probe(
        id=f"p_{case_id}",
        node_id=node_id,
        task=minimal_task(task),
        outcome=expected.strip(),
        axis=axis or case_id,
    )
    return add(store, probe, enforce_cap=enforce_cap)


def distill_and_add(
    store: "Store",
    adapter: "Adapter | None",
    *,
    node_id: str,
    lesson_body: str,
    task: str,
    outcome: str,
    context: str = "",
    source_episode: str = "",
) -> Probe | None:
    if adapter is None:
        return None
    return distill(
        store,
        adapter,
        node_id=node_id,
        lesson_body=lesson_body,
        task=task,
        outcome=outcome,
        context=context,
        source_episode=source_episode,
    )
