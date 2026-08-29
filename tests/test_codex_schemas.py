"""Offline checks that schemas satisfy Codex strict output-schema rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rose.judge import RELEVANCE_SCHEMA
from rose.prompts import JUDGE_SCHEMA, SELECT_SCHEMA


def _all_props_required(schema: dict) -> list[str]:
    """Return property keys not listed in required (recursive for object items)."""
    missing: list[str] = []
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        for key in props:
            if key not in required:
                missing.append(key)
            if isinstance(props[key], dict):
                missing.extend(_all_props_required(props[key]))
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        missing.extend(_all_props_required(schema["items"]))
    return missing


class TestCodexSchemas(unittest.TestCase):
    def test_judge_schema_strict(self) -> None:
        self.assertEqual(_all_props_required(JUDGE_SCHEMA), [])

    def test_relevance_pick_schema_strict(self) -> None:
        pick = RELEVANCE_SCHEMA["properties"]["picks"]["items"]
        self.assertEqual(_all_props_required(pick), [])

    def test_select_schema_strict(self) -> None:
        self.assertEqual(_all_props_required(SELECT_SCHEMA), [])


if __name__ == "__main__":
    unittest.main()
