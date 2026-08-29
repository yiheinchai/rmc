"""Tests for paper figure generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "papers" / "rse" / "figures"
RESULTS = ROOT / "papers" / "rse" / "results"


def test_generate_paper_figures_creates_pdfs() -> None:
    if not (RESULTS / "submission-latest.json").exists():
        pytest.skip("submission-latest.json missing")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_paper_figures.py")],
        check=True,
        cwd=ROOT,
    )
    for name in (
        "fig_wikiskill",
        "fig_recall_ablations",
        "fig_scaling",
        "fig_transfer_tokens",
        "fig_session_study",
    ):
        assert (FIGURES / f"{name}.pdf").exists()


def test_submission_report_has_manuscript_fields() -> None:
    path = RESULTS / "submission-latest.json"
    if not path.exists():
        pytest.skip("submission-latest.json missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "headline_findings" in data
    assert data.get("wikiskill")
