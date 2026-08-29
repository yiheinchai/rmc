"""Tests for the publication experiment suite."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rose.experiments import run_all


class TestExperiments(unittest.TestCase):
    def test_run_all_completes(self) -> None:
        suite = run_all(agent="mock", samples=1)
        self.assertIn("transfer", suite.bench)
        self.assertGreaterEqual(len(suite.scaling.get("rows", [])), 3)
        self.assertIn("serve-all", suite.recall.get("arms", {}))
        self.assertTrue(suite.walkthrough.get("compression_accepted"))
        levels = suite.retention_curve.get("levels", {})
        self.assertEqual(levels.get("L0", {}).get("pass_rate"), 1.0)
        self.assertEqual(levels.get("L1", {}).get("pass_rate"), 0.0)
        self.assertIn("no-skill", suite.wikiskill.get("arms", {}))


if __name__ == "__main__":
    unittest.main()
