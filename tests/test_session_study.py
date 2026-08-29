"""Session paired study tests."""

from __future__ import annotations

import unittest

from rmc.adapters.mock import MockAdapter
from rmc.bench import bench_adapter
from rmc.session_study import load_pairs, run, to_dict


class TestSessionStudy(unittest.TestCase):
    def test_loads_pairs(self) -> None:
        pairs = load_pairs()
        self.assertGreaterEqual(len(pairs), 5)
        self.assertTrue(all(p.lesson_ids for p in pairs))

    def test_mock_run_completes(self) -> None:
        report = run(bench_adapter(MockAdapter()), samples=1)
        self.assertEqual(len(report.pairs), 10)  # 5 pairs × 2 arms
        self.assertIn("memory-off", report.by_arm())
        self.assertIn("memory-on", report.by_arm())

    def test_to_dict_has_lift(self) -> None:
        report = run(bench_adapter(MockAdapter()), samples=1)
        data = to_dict(report)
        self.assertIn("lift", data)
        self.assertIn("arms", data)


if __name__ == "__main__":
    unittest.main()
