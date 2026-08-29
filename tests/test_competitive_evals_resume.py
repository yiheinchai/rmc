"""Tests for upstream resume in run_competitive_evals.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_competitive_evals",
        ROOT / "scripts" / "run_competitive_evals.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_competitive_evals"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCompetitiveEvalsResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_load_upstream_existing_from_wikiskill_checkpoint(self) -> None:
        from rmc.adapters.mock import MockAdapter
        from rmc.bench import bench_adapter
        from rmc.wikiskill import run, to_dict

        bench = ROOT / "evals" / "upstream" / "sealqa-test.jsonl"
        report = run(
            bench_adapter(MockAdapter()),
            path=bench,
            samples=1,
            limit=1,
            arms=("no-skill", "full-inject"),
        )
        ckpt = to_dict(report)
        ckpt["checkpoint"] = True

        with tempfile.TemporaryDirectory() as tmp:
            wiki = Path(tmp) / "wikiskill-latest.json"
            wiki.write_text(json.dumps(ckpt), encoding="utf-8")
            orig = ROOT / "papers" / "rse" / "results" / "wikiskill-latest.json"
            backup = None
            if orig.exists():
                backup = orig.read_text(encoding="utf-8")
            orig.parent.mkdir(parents=True, exist_ok=True)
            orig.write_text(wiki.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                existing = self.mod._load_upstream_existing(
                    "sealqa-test",
                    bench,
                    out_dir=Path(tmp),
                    payload={},
                    resume=True,
                )
                self.assertIsNotNone(existing)
                self.assertEqual(len(existing.cases), 2)
            finally:
                if backup is not None:
                    orig.write_text(backup, encoding="utf-8")
                elif orig.exists():
                    orig.unlink()

    def test_resume_disabled_returns_none(self) -> None:
        bench = ROOT / "evals" / "upstream" / "sealqa-test.jsonl"
        existing = self.mod._load_upstream_existing(
            "sealqa-test",
            bench,
            out_dir=Path("/tmp"),
            payload={},
            resume=False,
        )
        self.assertIsNone(existing)


if __name__ == "__main__":
    unittest.main()
