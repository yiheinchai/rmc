"""Tests for merge_competitive_upstream.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "merge_competitive_upstream",
        ROOT / "scripts" / "merge_competitive_upstream.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["merge_competitive_upstream"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestMergeCompetitiveUpstream(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_merge_upgrades_sealqa_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            comp = Path(tmp) / "competitive.json"
            ws = Path(tmp) / "wikiskill.json"
            comp.write_text(
                json.dumps(
                    {
                        "agent": "codex",
                        "upstream": {
                            "sealqa-test": {
                                "arms": {"full-inject": {"total": 50, "passed": 30}},
                            }
                        },
                    }
                )
            )
            ws.write_text(
                json.dumps(
                    {
                        "agent": "codex",
                        "bench_path": "/workspace/evals/upstream/sealqa-test.jsonl",
                        "arms": {
                            "full-inject": {"total": 111, "passed": 80, "accuracy": 0.72}
                        },
                        "cases": [],
                    }
                )
            )
            self.assertTrue(self.mod.merge_wikiskill_into_competitive(comp, ws))
            data = json.loads(comp.read_text())
            self.assertEqual(
                data["upstream"]["sealqa-test"]["arms"]["full-inject"]["total"],
                111,
            )

    def test_upstream_needs_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            comp = Path(tmp) / "competitive.json"
            comp.write_text(
                json.dumps(
                    {
                        "upstream": {
                            "sealqa-test": {"arms": {"full-inject": {"total": 50}}},
                        }
                    }
                )
            )
            self.assertTrue(self.mod.upstream_needs_run(comp, "sealqa-test"))
            self.assertTrue(self.mod.upstream_needs_run(comp, "hotpotqa-dev"))

    def test_merge_upstream_from_competitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            comp = Path(tmp) / "competitive.json"
            other = Path(tmp) / "other.json"
            comp.write_text(
                json.dumps(
                    {
                        "agent": "codex",
                        "rose_bench": {"lift": 0.25},
                        "upstream": {},
                    }
                )
            )
            other.write_text(
                json.dumps(
                    {
                        "agent": "codex",
                        "upstream": {
                            "hotpotqa-dev": {
                                "arms": {
                                    "full-inject": {"total": 100, "passed": 55, "accuracy": 0.55}
                                }
                            }
                        },
                    }
                )
            )
            self.assertTrue(
                self.mod.merge_upstream_from_competitive(comp, other, stems=("hotpotqa-dev",))
            )
            data = json.loads(comp.read_text())
            self.assertEqual(
                data["upstream"]["hotpotqa-dev"]["arms"]["full-inject"]["total"],
                100,
            )
