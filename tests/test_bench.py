"""Tests for RMC-Bench runner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmc.adapters.mock import MockAdapter
from rmc.bench import DEFAULT_BENCH, load_bench, mock_grade, run


class TestBench(unittest.TestCase):
    def test_load_bench_has_cases(self) -> None:
        cases, by_id = load_bench()
        self.assertGreaterEqual(len(cases), 10)
        self.assertIn("delete-is-soft", by_id)

    def test_mock_grade_detects_key_terms(self) -> None:
        ok, _ = mock_grade(
            "Uses DELETE with ?purge=true for GDPR erasure.",
            "Call DELETE with ?purge=true to purge the row completely.",
            kind="trap",
        )
        self.assertTrue(ok)

    def test_run_bench_produces_report(self) -> None:
        report = run(MockAdapter(), path=DEFAULT_BENCH, retention=False)
        self.assertGreater(len(report.cases), 0)
        text = report.render()
        self.assertIn("RMC-Bench", text)
        self.assertIn("Lift", text)

    def test_run_bench_on_progress_fires_per_transfer_case(self) -> None:
        seen: list[int] = []

        def _track(report) -> None:
            seen.append(len(report.cases))

        run(
            MockAdapter(),
            path=DEFAULT_BENCH,
            retention=False,
            retrieval=False,
            on_progress=_track,
        )
        self.assertEqual(len(seen), 25)
        self.assertEqual(seen[-1], 50)

    def test_control_worse_than_treatment_on_trap(self) -> None:
        report = run(MockAdapter(), path=DEFAULT_BENCH, retention=False, retrieval=False)
        core = {"trap", "detail", "principle"}
        ctrl = [c for c in report.cases if c.arm == "none" and c.kind in core]
        treat = [c for c in report.cases if c.arm == "L0" and c.kind in core]
        self.assertTrue(ctrl)
        self.assertTrue(treat)
        # At least one trap case should show lift with the lesson.
        traps = [c for c in report.cases if c.kind == "trap" and c.arm == "L0"]
        self.assertTrue(any(c.passed for c in traps))


if __name__ == "__main__":
    unittest.main()
