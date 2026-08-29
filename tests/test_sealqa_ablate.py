"""SealQA ablation harness tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from rose.adapters.mock import MockAdapter
from rose.bench import bench_adapter
from rose.sealqa_ablate import PRESETS, ProbeDevBench, run, run_preset


class TestSealQAAblation(unittest.TestCase):
    def test_probe_dev_bench_loads(self) -> None:
        bench = ProbeDevBench.load()
        self.assertGreater(len(bench.cases), 0)
        self.assertTrue(bench.lesson.strip())
        self.assertEqual(len(bench.axes), len(bench.cases))

    def test_all_presets_run_with_mock(self) -> None:
        bench = ProbeDevBench.load()
        adapter = bench_adapter(MockAdapter())
        for preset in PRESETS:
            result = run_preset(bench, preset, adapter)
            self.assertEqual(result.preset, preset)
            if preset == "probes-off":
                self.assertEqual(result.probe_total, 0)
                self.assertEqual(result.tokens_after, result.tokens_before)
            elif preset == "baseline":
                self.assertFalse(result.compaction_accepted)
                self.assertEqual(result.tokens_after, result.tokens_before)
            else:
                self.assertTrue(result.compaction_accepted)

    def test_report_renders_and_serializes(self) -> None:
        report = run(bench_adapter(MockAdapter()), score_tasks=True)
        text = report.render()
        self.assertIn("compact-probe-replay", text)
        data = report.to_dict()
        self.assertEqual(len(data["presets"]), len(PRESETS))
        self.assertIn("render", data)

    def test_custom_bench_path(self) -> None:
        path = Path(__file__).resolve().parents[1] / "evals" / "sealqa-ablation" / "probe-dev.yaml"
        report = run(bench_adapter(MockAdapter()), path=path, presets=("baseline",))
        self.assertEqual(len(report.presets), 1)
        self.assertEqual(report.presets[0].preset, "baseline")


if __name__ == "__main__":
    unittest.main()
