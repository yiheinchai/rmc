"""Tests for inject_paper_results.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inject_paper_results.py"


def _load():
    spec = importlib.util.spec_from_file_location("inject_paper_results", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inject_paper_results"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def inj():
    return _load()


def test_build_sealqa_table(inj) -> None:
  data = {
      "arms": {
          "full-inject": {"accuracy": 0.7, "total": 50, "mean_tokens": 100, "bootstrap_ci": {"low": 0.6, "high": 0.8}},
          "recall-agentic": {"accuracy": 0.8, "total": 50, "mean_tokens": 40},
      }
  }
  table = inj.build_sealqa_table(data)
  assert "50 tasks" in table
  assert "70\\%" in table
  assert "80\\%" in table
