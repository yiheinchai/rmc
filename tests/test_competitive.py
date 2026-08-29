"""Tests for skill baseline packs and MemGPT bench."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rmc.adapters.mock import MockAdapter
from rmc.bench import bench_adapter
from rmc.memgpt_bench import load_bench as load_memgpt_bench, run as run_memgpt
from rmc.skill_baselines import evoskill_pack, oracle_skill_pack, trace2skill_pack
from rmc.wikiskill import WikiSkillCase, build_store, load_bench as load_wikiskill_bench


class TestSkillBaselines(unittest.TestCase):
    def test_oracle_returns_case_skill(self) -> None:
        cases, _ = load_wikiskill_bench()
        self.assertTrue(oracle_skill_pack(cases[0]))

    def test_trace2skill_picks_overlap(self) -> None:
        cases, _ = load_wikiskill_bench()
        with tempfile.TemporaryDirectory() as tmp:
            store = build_store(cases, Path(tmp))
            pack = trace2skill_pack(store, cases[0].task)
            self.assertIn("extremal", pack.lower())


class TestMemgptBench(unittest.TestCase):
    def test_loads_cases(self) -> None:
        self.assertGreaterEqual(len(load_memgpt_bench()), 8)

    def test_mock_run(self) -> None:
        report = run_memgpt(bench_adapter(MockAdapter()), samples=1)
        self.assertGreater(len(report.cases), 0)
