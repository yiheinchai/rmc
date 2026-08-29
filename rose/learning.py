"""Lesson-authoring guidance: ROSE applied to its own minting.

A learning lesson is what reflection discovers about *how to write* a good
lesson — not knowledge about the work itself. "When the human corrected output
format, capture the required shape, not just the fact." "When a trap came from
trial and error, write the blind spot the detour exposed, not only the command
that worked."

Like selection lessons in ``routing/``, these live outside ``nodes/`` so recall
never treats authoring norms as domain knowledge. They are injected into
``REFLECT`` (and similar meta-calls) under their own token cap, ranked by
``helped`` / ``wasted`` when the cap binds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import yamlish
from .util import count_tokens, new_id, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import Store

LEARNING_DIRNAME = "learning"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass
class Rule:
    """One authoring lesson: when this kind of session appears, mint like this."""

    id: str
    when: str
    then: str = ""
    created: str = field(default_factory=utcnow)
    updated: str = field(default_factory=utcnow)
    shown: int = 0
    helped: int = 0
    wasted: int = 0
    origin: str = "reflection"  # reflection | manual
    status: str = "active"
    path: Path | None = None

    @property
    def tokens(self) -> int:
        return count_tokens(self.render())

    @property
    def posterior(self) -> float:
        return (self.helped + 1) / (self.helped + self.wasted + 2)

    def render(self) -> str:
        return f"- When {self.when.strip().rstrip('.')}: {self.then.strip()}"

    def to_frontmatter(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "when": self.when,
            "created": self.created,
            "updated": self.updated,
            "origin": self.origin,
            "status": self.status,
            "tokens": self.tokens,
            "stats": {"shown": self.shown, "helped": self.helped, "wasted": self.wasted},
        }

    def to_markdown(self) -> str:
        fm = yamlish.dump(self.to_frontmatter()).rstrip("\n")
        return f"---\n{fm}\n---\n\n{self.then.strip()}\n"

    @classmethod
    def from_markdown(cls, text: str, path: Path | None = None) -> "Rule":
        match = _FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError(f"learning rule has no frontmatter: {path}")
        meta = yamlish.load(match.group(1)) or {}
        if not isinstance(meta, dict):
            raise ValueError(f"learning frontmatter is not a mapping: {path}")
        stats = meta.get("stats") or {}
        return cls(
            id=str(meta.get("id") or ""),
            when=str(meta.get("when") or ""),
            then=match.group(2).strip(),
            created=meta.get("created") or utcnow(),
            updated=meta.get("updated") or utcnow(),
            shown=int(stats.get("shown") or 0),
            helped=int(stats.get("helped") or 0),
            wasted=int(stats.get("wasted") or 0),
            origin=str(meta.get("origin") or "reflection"),
            status=str(meta.get("status") or "active"),
            path=path,
        )

    def touch(self) -> None:
        self.updated = utcnow()


def learning_dir(store: "Store") -> Path:
    return store.root / LEARNING_DIRNAME


def load(store: "Store") -> list[Rule]:
    rules: dict[str, Rule] = {}
    if getattr(store, "parent", None) is not None:
        rules = {r.id: r for r in load(store.parent)}
    directory = learning_dir(store)
    if directory.is_dir():
        for path in sorted(directory.glob("*.md")):
            try:
                rule = Rule.from_markdown(path.read_text(encoding="utf-8"), path)
            except Exception:
                continue
            if rule.id and rule.status == "active":
                rules[rule.id] = rule
    return sorted(rules.values(), key=lambda r: (-r.posterior, r.id))


def save(store: "Store", rule: Rule) -> Path:
    directory = learning_dir(store)
    directory.mkdir(parents=True, exist_ok=True)
    rule.touch()
    path = rule.path or (directory / f"{rule.id}.md")
    path.write_text(rule.to_markdown(), encoding="utf-8")
    rule.path = path
    return path


def delete(rule: Rule) -> None:
    if rule.path and rule.path.exists():
        rule.path.unlink()


def get(store: "Store", rule_id: str) -> Rule | None:
    return next((r for r in load(store) if r.id == rule_id), None)


def mint(store: "Store", *, when: str, then: str, origin: str = "reflection") -> Rule | None:
    when, then = (when or "").strip(), (then or "").strip()
    if not when or not then:
        return None
    rule = Rule(id=new_id("l"), when=when, then=then, origin=origin)
    if any(_same(rule, other) for other in load(store)):
        return None
    save(store, rule)
    return rule


def _same(a: Rule, b: Rule) -> bool:
    return _norm(a.when) == _norm(b.when)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def fit(rules: list[Rule], budget: int) -> list[Rule]:
    out: list[Rule] = []
    used = 0
    for rule in sorted(rules, key=lambda r: (-r.posterior, -r.helped, r.id)):
        cost = rule.tokens
        if used + cost > budget:
            continue
        out.append(rule)
        used += cost
    return out


def render(rules: list[Rule]) -> str:
    if not rules:
        return ""
    return "\n".join(r.render() for r in rules)


def pack_for_reflect(store: "Store") -> tuple[str, list[Rule]]:
    """Guidance text and rules injected for one reflect/mint call."""
    if not store.config.get("learning_rules.enabled", True):
        return "", []
    budget = int(store.config.get("learning_rules.max_tokens", 600))
    rules = fit(load(store), budget)
    text = render(rules)
    return text, rules


def credit(store: "Store", *, helped: list[str], wasted: list[str], shown: list[str]) -> None:
    by_id = {r.id: r for r in load(store)}
    touched: set[str] = set()
    for rule_id in shown:
        rule = by_id.get(rule_id)
        if rule is not None:
            rule.shown += 1
            touched.add(rule_id)
    for rule_id in helped:
        rule = by_id.get(rule_id)
        if rule is not None:
            rule.helped += 1
            touched.add(rule_id)
    for rule_id in wasted:
        rule = by_id.get(rule_id)
        if rule is not None:
            rule.wasted += 1
            touched.add(rule_id)
    for rule_id in touched:
        save(store, by_id[rule_id])


def growth(store: "Store") -> dict[str, Any]:
    rules = load(store)
    nodes = [n for n in store.nodes() if n.status != "archived"]
    return {
        "rules": len(rules),
        "nodes": len(nodes),
        "ratio": (len(rules) / len(nodes)) if nodes else 0.0,
        "tokens": sum(r.tokens for r in rules),
    }
