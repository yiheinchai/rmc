"""WikiSkill-comparable benchmark tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from rmc.adapters.mock import MockAdapter
from rmc.bench import bench_adapter
from rmc.wikiskill import (
    ARMS,
    build_store,
    full_inject_pack,
    load_bench,
    run,
    to_dict,
)


class TestWikiSkillBench(unittest.TestCase):
    def test_loads_all_five_benchmarks(self) -> None:
        cases, benchmarks = load_bench()
        self.assertEqual(len(benchmarks), 5)
        self.assertEqual(len(cases), 10)
        seen = {c.benchmark for c in cases}
        self.assertEqual(seen, set(benchmarks))

    def test_full_inject_includes_every_skill(self) -> None:
        import tempfile

        cases, _ = load_bench()
        with tempfile.TemporaryDirectory() as tmp:
            store = build_store(cases, Path(tmp))
            pack = full_inject_pack(store)
            for case in cases:
                self.assertIn(case.skill.split()[0], pack)

    def test_mock_run_produces_all_arms(self) -> None:
        report = run(bench_adapter(MockAdapter()), samples=1)
        self.assertEqual(len(report.cases), 10 * len(ARMS))
        for arm in ARMS:
            self.assertGreater(report.by_arm()[arm][1], 0)

    def test_to_dict_has_comparisons(self) -> None:
        report = run(bench_adapter(MockAdapter()), samples=1)
        data = to_dict(report)
        self.assertIn("full_inject_vs_no_skill", data["comparisons"])
        self.assertIn("arms", data)
        self.assertIn("no-skill", data["arms"])


if __name__ == "__main__":
    unittest.main()
