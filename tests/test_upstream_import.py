"""Tests for upstream benchmark import."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "import_upstream_bench.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("import_upstream_bench", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["import_upstream_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestUpstreamImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.imp = _load_module()

    def test_normalize_urls_list(self) -> None:
        self.assertEqual(
            self.imp._normalize_urls(["http://a", "http://b"]),
            ["http://a", "http://b"],
        )

    def test_normalize_urls_string_list(self) -> None:
        raw = "['http://example.com']"
        self.assertEqual(self.imp._normalize_urls(raw), ["http://example.com"])

    def test_url_text_snippets(self) -> None:
        url = (
            "https://en.wikipedia.org/wiki/Test#:~:text=Hello%20World"
            ",-,Second%20snippet"
        )
        snippets = self.imp._url_text_snippets([url])
        self.assertGreaterEqual(len(snippets), 1)
        self.assertTrue(any("Hello" in s or "Second" in s for s in snippets))

    def test_row_to_case_includes_evidence(self) -> None:
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
        case = self.imp._row_to_case(row, spec, 1)
        self.assertIn("Evidence snippets", case["task"])
        self.assertEqual(case["expected"], "Answer: Alice")

    def test_row_to_case_hotpot(self) -> None:
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
        case = self.imp._row_to_case(row, spec, 1)
        self.assertIn("Evidence snippets", case["task"])
        self.assertIn("American", case["task"])
        self.assertEqual(case["expected"], "Answer: yes")

    def test_row_to_case_page_hints(self) -> None:
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
            "urls": [
                "https://en.wikipedia.org/wiki/List_of_NBA_single-game_scoring_leaders#Single-game_leaders"
            ],
        }
        case = self.imp._row_to_case(row, spec, 1)
        self.assertIn("Reference context", case["task"])
        self.assertIn("NBA", case["task"])
