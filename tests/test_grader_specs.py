"""Tests for grader_specs and HotPotQA import."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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


def test_parse_grader_spec() -> None:
    from rmc.grader_specs import parse_grader_spec

    assert parse_grader_spec("codex") == ("codex", None, "codex")
    assert parse_grader_spec("codex:gpt-5.6-sol") == ("codex", "gpt-5.6-sol", "codex:gpt-5.6-sol")


def test_hotpot_evidence() -> None:
    imp = _load_import()
    row = {
        "context": {
            "title": ["Alice", "Bob"],
            "sentences": [["Alice is a chef."], ["Bob is a pilot."]],
        },
        "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
    }
    snippets = imp._hotpot_evidence(row)
    assert len(snippets) == 1
    assert "chef" in snippets[0]


def test_default_multimodel_specs_count() -> None:
    from rmc.grader_specs import default_multimodel_specs

    specs = default_multimodel_specs()
    assert len(specs) >= 3
    assert "codex" in specs
