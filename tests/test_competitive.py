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


class TestCrossTransfer(unittest.TestCase):
    def test_mock_cross_transfer(self) -> None:
        from rmc.cross_transfer import run_cross_transfer, to_dict

        report = run_cross_transfer(["mock"], samples=1, limit=3)
        payload = to_dict(report)
        self.assertIn("table", payload)
        self.assertTrue(report.cells)


class TestCompetitiveEvalHelpers(unittest.TestCase):
    def test_load_payload_merge(self) -> None:
        import importlib.util
        import json
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "run_competitive_evals",
            Path(__file__).resolve().parents[1] / "scripts" / "run_competitive_evals.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            merge = Path(tmp) / "comp.json"
            merge.write_text(json.dumps({"rmc_bench": {"lift": 0.25}, "upstream": {}}))
            adapter = bench_adapter(MockAdapter())
            payload = mod._load_payload(merge, stamp="t", adapter=adapter, samples=1)
            self.assertEqual(payload["rmc_bench"]["lift"], 0.25)
            self.assertEqual(payload["agent"], "mock")
