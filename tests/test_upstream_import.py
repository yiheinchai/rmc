"""Tests for upstream benchmark import."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "import_upstream_bench.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("import_upstream_bench", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["import_upstream_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def imp():
    return _load_module()


def test_normalize_urls_list(imp) -> None:
    assert imp._normalize_urls(["http://a", "http://b"]) == ["http://a", "http://b"]


def test_normalize_urls_string_list(imp) -> None:
    raw = "['http://example.com']"
    assert imp._normalize_urls(raw) == ["http://example.com"]


def test_url_text_snippets(imp) -> None:
    url = (
        "https://en.wikipedia.org/wiki/Test#:~:text=Hello%20World"
        ",-,Second%20snippet"
    )
    snippets = imp._url_text_snippets([url])
    assert len(snippets) >= 1
    assert any("Hello" in s or "Second" in s for s in snippets)


def test_row_to_case_includes_evidence(imp) -> None:
    spec = {
        "id": "test",
        "benchmark": "SealQA",
        "family": "sealqa",
        "task_field": "question",
        "expected_field": "answer",
        "skill": "test skill",
    }
    row = {
        "question": "Who won?",
        "answer": "Alice",
        "urls": ["https://x#:~:text=Alice%20won%20the%20prize"],
    }
    case = imp._row_to_case(row, spec, 1)
    assert "Evidence snippets" in case["task"]
    assert case["expected"] == "Answer: Alice"


def test_row_to_case_hotpot(imp) -> None:
    spec = {
        "id": "hotpotqa-dev",
        "benchmark": "HotPotQA",
        "family": "hotpotqa",
        "task_field": "question",
        "expected_field": "answer",
        "evidence_mode": "hotpot",
        "skill": "hotpot skill",
    }
    row = {
        "question": "Same nationality?",
        "answer": "yes",
        "context": {
            "title": ["Alice", "Bob"],
            "sentences": [["Alice is American."], ["Bob is American."]],
        },
        "supporting_facts": {"title": ["Alice", "Bob"], "sent_id": [0, 0]},
    }
    case = imp._row_to_case(row, spec, 1)
    assert "Evidence snippets" in case["task"]
    assert "American" in case["task"]
    assert case["expected"] == "Answer: yes"


def test_row_to_case_page_hints(imp) -> None:
    spec = {
        "id": "test",
        "benchmark": "SealQA",
        "family": "sealqa",
        "task_field": "question",
        "expected_field": "answer",
        "skill": "test skill",
    }
    row = {
        "question": "How many?",
        "answer": "12",
        "urls": ["https://en.wikipedia.org/wiki/List_of_NBA_single-game_scoring_leaders#Single-game_leaders"],
    }
    case = imp._row_to_case(row, spec, 1)
    assert "Reference context" in case["task"]
    assert "NBA" in case["task"]
