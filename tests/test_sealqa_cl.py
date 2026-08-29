"""SealQA continual-learning harness tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from rose.adapters.mock import MockAdapter
from rose.bench import bench_adapter
from rose.sealqa_cl import CLBench, TEST_ARMS, run


class TestSealQAContinualLearning(unittest.TestCase):
    def test_bench_splits_train_and_test(self) -> None:
        bench = CLBench.load()
        self.assertEqual(len(bench.train), 6)
        self.assertEqual(len(bench.test), 6)
        train_axes = {c.axis for c in bench.train}
        test_axes = {c.axis for c in bench.test}
        self.assertEqual(train_axes, test_axes)

    def test_mock_run_produces_all_arms(self) -> None:
        report = run(bench_adapter(MockAdapter()), compact=True)
        self.assertEqual(len(report.train_steps), 6)
        for arm in TEST_ARMS:
            passed, total, _ = report.accuracy(arm)
            self.assertEqual(total, 6, msg=arm)
            self.assertGreaterEqual(passed, 0, msg=arm)
        self.assertTrue(report.render())

    def test_to_dict_has_lift(self) -> None:
        report = run(bench_adapter(MockAdapter()))
        data = report.to_dict()
        self.assertIn("lift_vs_no_memory", data)
        self.assertIn("arms", data)
        self.assertEqual(set(data["arms"]), set(TEST_ARMS))

    def test_custom_bench_path(self) -> None:
        path = Path(__file__).resolve().parents[1] / "evals" / "sealqa-ablation" / "probe-dev.yaml"
        report = run(bench_adapter(MockAdapter()), path=path)
        self.assertIn("probe-dev.yaml", report.bench_path)


if __name__ == "__main__":
    unittest.main()
