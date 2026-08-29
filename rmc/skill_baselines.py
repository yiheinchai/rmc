"""Inference-time skill baselines mirroring WikiSkill comparison methods.

These are *inference arms* for evaluation — not full offline evolution loops.
They approximate how Trace2Skill, EvoSkill, and SkillOpt behave at test time
(full or partial skill injection) so RSE recall can be compared on equal footing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import Store
    from .wikiskill import WikiSkillCase

_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")


def _task_tokens(task: str) -> set[str]:
    return set(_TOKEN_RE.findall((task or "").lower()))


def _score_node(task: str, body: str) -> int:
    task_tok = _task_tokens(task)
    body_tok = _task_tokens(body)
    return len(task_tok & body_tok)


def rank_skills(store: Store, task: str) -> list[tuple[str, str, int]]:
    """Return [(node_id, body, score), ...] descending by keyword overlap."""
    ranked: list[tuple[str, str, int]] = []
    for node in store.nodes():
        if node.status == "archived":
            continue
        score = _score_node(task, node.body)
        ranked.append((node.id, node.body.strip(), score))
    ranked.sort(key=lambda row: (-row[2], row[0]))
    return ranked


def oracle_skill_pack(case: WikiSkillCase) -> str:
    """Upper bound: inject only the task's ground-truth skill."""
    return case.skill.strip()


def trace2skill_pack(store: Store, task: str) -> str:
    """Proxy: single best-matching evolved skill (Trace2Skill-style one skill)."""
    ranked = rank_skills(store, task)
    if not ranked or ranked[0][2] == 0:
        return ""
    return ranked[0][1]


def evoskill_pack(store: Store, task: str, *, k: int = 2) -> str:
    """Proxy: top-k skills by overlap (EvoSkill-style multi-skill inject)."""
    ranked = [body for _, body, score in rank_skills(store, task) if score > 0][:k]
    return "\n\n---\n\n".join(ranked)


def skillopt_pack(store: Store, task: str) -> str:
    """Proxy: best skill plus index hints (SkillOpt-style structured inject)."""
    ranked = rank_skills(store, task)
    if not ranked or ranked[0][2] == 0:
        return ""
    node_id, body, _ = ranked[0]
    header = f"# Selected skill: {node_id}\n# Optimized for task keyword overlap\n\n"
    return header + body


def keyword_rag_pack(store: Store, task: str, *, k: int = 2) -> str:
    """MemGPT/RAG-style: retrieve top-k skill snippets by lexical overlap."""
    return evoskill_pack(store, task, k=k)
