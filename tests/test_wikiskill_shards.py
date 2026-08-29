"""Tests for WikiSkill shard merge and offset."""

from __future__ import annotations

import unittest

from rose.adapters.mock import MockAdapter
from rose.bench import bench_adapter
from rose.wikiskill import merge_reports, run, to_dict


class TestWikiSkillShards(unittest.TestCase):
    def test_offset_limits_case_window(self) -> None:
        first = run(bench_adapter(MockAdapter()), samples=1, limit=3, arms=("no-skill",))
        second = run(
            bench_adapter(MockAdapter()),
            samples=1,
            offset=3,
            limit=2,
            arms=("no-skill",),
        )
        self.assertEqual(len(first.cases), 3)
        self.assertEqual(len(second.cases), 2)
        ids_first = {c.case_id for c in first.cases}
        ids_second = {c.case_id for c in second.cases}
        self.assertFalse(ids_first & ids_second)

    def test_merge_reports_dedupes(self) -> None:
        a = run(bench_adapter(MockAdapter()), samples=1, limit=2, arms=("no-skill",))
        b = run(
            bench_adapter(MockAdapter()),
            samples=1,
            offset=2,
            limit=2,
            arms=("no-skill",),
        )
        merged = merge_reports(a, b)
        self.assertEqual(len(merged.cases), 4)
        payload = to_dict(merged)
        self.assertEqual(payload["arms"]["no-skill"]["total"], 4)


if __name__ == "__main__":
    unittest.main()
