"""Evidence enrichment for SealQA continual-learning benches."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import yamlish

_GARBLED_RE = re.compile(
    r",-,|\[12\]|^the world\.|^\(b,American|Baby Shark\[,|Found \d+ airline members",
    re.IGNORECASE,
)
_HOW_MANY_RE = re.compile(r"\bhow many\b", re.IGNORECASE)

AXIS_PRIORITY: tuple[tuple[str, str], ...] = (
    ("false-premise", "no-guess"),
    ("temporal tracking", "header-metadata"),
    ("cross-lingual reasoning", "cross-lingual"),
    ("entity/event disambiguation", "entity-match"),
    ("advanced reasoning", "advanced-reasoning"),
)

AXIS_SKILLS: dict[str, str] = {
    "answer-format": 'Format every answer as: Answer: <short fact>.',
    "explicit-count": 'Count only explicitly named entries; never count "et al." as extra names.',
    "entity-match": "Prefer snippets whose title or entity matches the question subject.",
    "no-guess": "When the premise is wrong or evidence is missing, say so — do not guess.",
    "table-row": "When a table row is labeled Total X, use that row.",
    "stated-count": 'For "how many" questions, use a number only if the snippet states it.',
    "header-metadata": "Prefer header/metadata year, date, or count over surrounding prose.",
    "advanced-reasoning": "Answer only from provided evidence; combine snippets carefully.",
    "cross-lingual": "Use the evidence language and entities as given in snippets.",
}

DEFAULT_LESSON = ""  # loaded lazily via default_lesson()


def default_lesson() -> str:
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "evals" / "sealqa-ablation" / "probe-dev.yaml"
    raw = yamlish.load(path.read_text(encoding="utf-8"))
    return str(raw.get("lesson") or "").strip()


def _normalize_urls(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(u).strip() for u in raw if str(u).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(u).strip() for u in parsed if str(u).strip()]
        except json.JSONDecodeError:
            pass
    return [text]


def _url_text_snippets(urls: list[str]) -> list[str]:
    snippets: list[str] = []
    for url in urls:
        if ":~:text=" not in url:
            continue
        tail = url.split(":~:text=", 1)[1]
        for frag in re.split(r",-,", tail):
            text = urllib.parse.unquote(frag.replace("+", " ")).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) >= 8:
                snippets.append(text)
    return snippets


def _url_page_hints(urls: list[str]) -> list[str]:
    hints: list[str] = []
    for url in urls:
        if "wikipedia.org/wiki/" not in url:
            continue
        path = url.split("/wiki/", 1)[1]
        page, _, frag = path.partition("#")
        title = urllib.parse.unquote(page).replace("_", " ")
        if title:
            hints.append(f"Wikipedia article: {title}")
        if frag and not frag.startswith("~:text"):
            anchor = frag.split(":~:", 1)[0]
            hints.append(f"Section: {urllib.parse.unquote(anchor).replace('_', ' ')}")
    return hints


def _fetch_wikipedia_extract(title: str, *, max_chars: int = 900) -> str:
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "rose-sealqa-cl/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return ""
    extract = str(data.get("extract") or "").strip()
    extract = re.sub(r"\s+", " ", extract)
    return extract[:max_chars]


def _wikipedia_titles(urls: list[str]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if "wikipedia.org/wiki/" not in url:
            continue
        page = url.split("/wiki/", 1)[1].split("#")[0]
        title = urllib.parse.unquote(page).replace("_", " ")
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def classify_snippets(snippets: list[str]) -> str:
    if not snippets:
        return "missing"
    joined = " ".join(snippets)
    if _GARBLED_RE.search(joined):
        return "garbled"
    if len(joined) < 24:
        return "garbled"
    return "usable"


def infer_axis(question: str, question_types: list[str] | None) -> str:
    if _HOW_MANY_RE.search(question):
        return "stated-count"
    types = {str(t).strip().lower() for t in (question_types or [])}
    for needle, axis in AXIS_PRIORITY:
        if needle in types:
            return axis
    return "answer-format"


def skill_for_axis(axis: str) -> str:
    return AXIS_SKILLS.get(axis, AXIS_SKILLS["answer-format"])


def enrich_evidence(
    *,
    question: str,
    urls: list[str],
    fetch_wikipedia: bool = True,
    sleep_s: float = 0.05,
) -> tuple[list[str], str]:
    """Return (snippet lines, evidence_quality tag)."""
    snippets = _url_text_snippets(urls)
    quality = classify_snippets(snippets)

    if fetch_wikipedia and (quality in ("missing", "garbled") or not snippets):
        for title in _wikipedia_titles(urls)[:2]:
            extract = _fetch_wikipedia_extract(title)
            if extract:
                snippets.append(extract)
                time.sleep(sleep_s)
        if snippets and quality == "missing":
            quality = "wikipedia"
        elif snippets and quality == "garbled":
            quality = "wikipedia_supplement"

    if not snippets:
        hints = _url_page_hints(urls)
        if hints:
            quality = "page_hint"
            return hints, quality
        if urls:
            return [u for u in urls[:3]], "bare_url"
        return [], "missing"

    return snippets[:6], quality


def build_task(question: str, snippets: list[str], *, label: str = "Evidence snippets") -> str:
    block = "\n".join(f"- {s}" for s in snippets)
    return (
        f"{question.strip()}\n\n{label}:\n{block}\n\n"
        "Answer with a short factual response starting with 'Answer:'."
    )


def hf_row_to_case(
    row: dict[str, Any],
    *,
    idx: int,
    fetch_wikipedia: bool = True,
) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    urls = _normalize_urls(row.get("urls"))
    snippets, quality = enrich_evidence(
        question=question,
        urls=urls,
        fetch_wikipedia=fetch_wikipedia,
    )
    axis = infer_axis(question, row.get("question_types"))
    label = "Evidence snippets (enriched)" if quality.startswith("wikipedia") else "Evidence snippets"
    if quality == "page_hint":
        label = "Reference context"
    elif quality == "bare_url":
        label = "Reference URLs"
    task = build_task(question, snippets, label=label)
    expected = answer if answer.lower().startswith("answer") else f"Answer: {answer}"
    return {
        "id": f"sealqa-cl-{idx:04d}",
        "benchmark": "SealQA",
        "family": "sealqa",
        "axis": axis,
        "task": task,
        "expected": expected,
        "skill": skill_for_axis(axis),
        "evidence_quality": quality,
        "topic": str(row.get("topic") or ""),
    }
