"""Tests for inject_paper_results.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inject_paper_results.py"


def _load():
    spec = importlib.util.spec_from_file_location("inject_paper_results", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inject_paper_results"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestInjectPaperResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inj = _load()

    def test_build_rmc_bench_rows(self) -> None:
        rb = {
            "lift": 0.25,
            "transfer": {"passed": 19, "total": 20},
            "retrieval": {"passed": 5, "total": 8},
            "cases": [
                {"kind": "trap", "arm": "L0", "passed": True, "tokens": 80},
                {"kind": "detail", "arm": "L0", "passed": True, "tokens": 82},
            ],
        }
        rows = self.inj.build_rmc_bench_rows(rb)
        self.assertTrue("+25" in rows["lift"] or "25" in rows["lift"])
        self.assertIn("19/20", rows["transfer"])

    def test_sealqa_upstream_payload_competitive(self) -> None:
        data = {
            "agent": "codex",
            "upstream": {
                "sealqa-test": {
                    "arms": {
                        "full-inject": {"accuracy": 0.6, "total": 111, "mean_tokens": 90},
                    }
                }
            },
        }
        payload = self.inj._sealqa_upstream_payload(data)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["arms"]["full-inject"]["total"], 111)

    def test_sealqa_upstream_rejects_small_mock_subset(self) -> None:
        data = {
            "upstream": {
                "sealqa-test": {
                    "arms": {
                        "full-inject": {"accuracy": 1.0, "total": 15, "mean_tokens": 90},
                    }
                }
            },
        }
        self.assertIsNone(self.inj._sealqa_upstream_payload(data))

    def test_build_hotpot_table(self) -> None:
        data = {
            "arms": {
                "full-inject": {
                    "accuracy": 0.5,
                    "total": 100,
                    "mean_tokens": 120,
                    "bootstrap_ci": {"low": 0.4, "high": 0.6},
                },
            }
        }
        table = self.inj.build_hotpot_table(data)
        self.assertIn("100 tasks", table)
        self.assertIn("HotPotQA", table)

    def test_build_sealqa_table(self) -> None:
        data = {
            "arms": {
                "full-inject": {
                    "accuracy": 0.7,
                    "total": 111,
                    "mean_tokens": 100,
                    "bootstrap_ci": {"low": 0.6, "high": 0.8},
                },
                "recall-agentic": {"accuracy": 0.8, "total": 111, "mean_tokens": 40},
            }
        }
        table = self.inj.build_sealqa_table(data)
        self.assertIn("111 tasks", table)
        self.assertIn("70\\%", table)
        self.assertIn("80\\%", table)

    def test_build_cross_transfer_table(self) -> None:
        data = {
            "samples": 3,
            "table": {
                "codex": {
                    "SealQA": {"full-inject": 1.0, "recall-agentic": 1.0},
                    "ALFWorld": {"full-inject": 0.5, "recall-agentic": 1.0},
                }
            },
            "models": {
                "codex": {
                    "agent": "codex",
                    "arms": {
                        "full-inject": {"accuracy": 0.7, "mean_tokens": 534},
                        "recall-agentic": {
                            "accuracy": 0.8,
                            "mean_tokens": 59,
                            "bootstrap_ci": {"low": 0.47, "high": 0.93},
                        },
                    },
                }
            },
        }
        table = self.inj.build_cross_transfer_table(data)
        self.assertIn("SealQA", table)
        self.assertIn("80\\%", table)
        self.assertIn("47", table)
