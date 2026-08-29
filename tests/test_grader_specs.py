"""Tests for grader_specs and HotPotQA import."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_import():
    spec = importlib.util.spec_from_file_location(
        "import_upstream_bench", ROOT / "scripts" / "import_upstream_bench.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["import_upstream_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestGraderSpecs(unittest.TestCase):
    def test_parse_grader_spec(self) -> None:
        from rmc.grader_specs import parse_grader_spec

        self.assertEqual(parse_grader_spec("codex"), ("codex", None, "codex"))
        self.assertEqual(
            parse_grader_spec("codex:gpt-5.6-sol"),
            ("codex", "gpt-5.6-sol", "codex:gpt-5.6-sol"),
        )

    def test_hotpot_evidence(self) -> None:
        imp = _load_import()
        row = {
            "context": {
                "title": ["Alice", "Bob"],
                "sentences": [["Alice is a chef."], ["Bob is a pilot."]],
            },
            "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
        }
        snippets = imp._hotpot_evidence(row)
        self.assertEqual(len(snippets), 1)
        self.assertIn("chef", snippets[0])

    def test_default_multimodel_specs_count(self) -> None:
        from rmc.grader_specs import default_multimodel_specs

        specs = default_multimodel_specs()
        self.assertGreaterEqual(len(specs), 3)
        self.assertIn("codex", specs)
