"""Tests for audit_competitive_bar.py."""

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
        "audit_competitive_bar",
        ROOT / "scripts" / "audit_competitive_bar.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_competitive_bar"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestAuditCompetitiveBar(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_incomplete_on_mock_competitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            self.mod.RESULTS = results
            self.mod.PAPER = results / "paper.tex"
            self.mod.FIGURES = results / "figures"
            self.mod.FIGURES.mkdir()
            for fig in self.mod.REQUIRED_FIGURES:
                (self.mod.FIGURES / fig).write_text("%PDF")
            self.mod.PAPER.write_text(
                "% AUTO:SEALQA_TABLE_BEGIN\n% AUTO:SEALQA_TABLE_END\n"
            )
            (results / "competitive-latest.json").write_text(
                json.dumps({"agent": "mock", "rmc_bench": {"transfer": {"total": 10}}})
            )
            failures = self.mod.audit()
            self.assertTrue(any("agent must be codex" in f for f in failures))
            self.assertTrue(any("sealqa-test" in f for f in failures))

    def test_complete_minimal_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            self.mod.RESULTS = results
            self.mod.PAPER = results / "paper.tex"
            self.mod.FIGURES = results / "figures"
            self.mod.FIGURES.mkdir()
            for fig in self.mod.REQUIRED_FIGURES:
                (self.mod.FIGURES / fig).write_text("%PDF")

            def _upstream(stem: str, n: int) -> dict:
                arms = {
                    a: {"total": n, "accuracy": 0.5, "bootstrap_ci": {"low": 0.4, "high": 0.6}}
                    for a in self.mod.BASELINE_ARMS
                }
                return {"arms": arms}

            comp = {
                "agent": "codex",
                "rmc_bench": {"transfer": {"total": 25}},
                "upstream": {
                    "sealqa-test": _upstream("sealqa-test", 111),
                    "hotpotqa-dev": _upstream("hotpotqa-dev", 100),
                },
            }
            (results / "competitive-latest.json").write_text(json.dumps(comp))
            (results / "multimodel-latest.json").write_text(
                json.dumps({"models": {"a": {}, "b": {}, "c": {}}})
            )
            (results / "cross-transfer-latest.json").write_text(
                json.dumps({"table": {"codex": {}}})
            )
            self.mod.PAPER.write_text(
                "\n".join(
                    [
                        "% AUTO:SEALQA_TABLE_BEGIN",
                        "\\begin{table}",
                        "% AUTO:SEALQA_TABLE_END",
                        "% AUTO:HOTPOT_TABLE_BEGIN",
                        "\\begin{table}",
                        "% AUTO:HOTPOT_TABLE_END",
                        "% AUTO:MULTIMODEL_TABLE_BEGIN",
                        "\\begin{table}",
                        "% AUTO:MULTIMODEL_TABLE_END",
                    ]
                )
            )
            failures = self.mod.audit()
            self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
