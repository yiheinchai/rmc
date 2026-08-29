"""Tests for paper figure generation."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "papers" / "rse" / "figures"
RESULTS = ROOT / "papers" / "rse" / "results"


@unittest.skipUnless(
    (RESULTS / "submission-latest.json").exists(),
    "submission-latest.json missing",
)
class TestGeneratePaperFigures(unittest.TestCase):
    def test_generate_paper_figures_creates_pdfs(self) -> None:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_paper_figures.py")],
            check=True,
            cwd=ROOT,
        )
        for name in (
            "fig_architecture",
            "fig_case_study",
            "fig_wikiskill",
            "fig_recall_ablations",
            "fig_scaling",
            "fig_transfer_tokens",
            "fig_session_study",
        ):
            self.assertTrue((FIGURES / f"{name}.pdf").exists())


@unittest.skipUnless(
    (RESULTS / "submission-latest.json").exists(),
    "submission-latest.json missing",
)
class TestSubmissionReport(unittest.TestCase):
    def test_submission_report_has_manuscript_fields(self) -> None:
        path = RESULTS / "submission-latest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("headline_findings", data)
        self.assertTrue(data.get("wikiskill"))
