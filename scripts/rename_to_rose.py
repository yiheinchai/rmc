#!/usr/bin/env python3
"""One-shot RMC/RSE → ROSE rename across the repository."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Order matters: longer / more specific patterns first.
REPLACEMENTS: list[tuple[str, str]] = [
    ("Recursive Memory Compaction", "Recursive Online Skill Evolution"),
    ("Recursive Skill Evolution", "Recursive Online Skill Evolution"),
    ("RMC-Bench", "ROSE-Bench"),
    ("rmc-bench", "rose-bench"),
    ("rmc_bench", "rose_bench"),
    ("RMC_HOME", "ROSE_HOME"),
    ("RMC_CHILD", "ROSE_CHILD"),
    ("RMC_DISABLE", "ROSE_DISABLE"),
    ("RMC_BACKGROUND", "ROSE_BACKGROUND"),
    ("RMC_TOKENIZER", "ROSE_TOKENIZER"),
    ("RMC_PYTHON", "ROSE_PYTHON"),
    ("rmc-memory", "rose-memory"),
    ("rmc-codex", "rose-codex"),
    ("papers/rse", "papers/rose"),
    ("yiheinchai.com/rmc", "yiheinchai.com/rose"),
    ("github.com/yiheinchai/rmc", "github.com/yiheinchai/rose"),
    ("yiheinchai/rmc", "yiheinchai/rose"),
    ("/bin/rmc", "/bin/rose"),
    ("bin/rmc", "bin/rose"),
    ("commands/rmc.md", "commands/rose.md"),
    ("evals/rmc-bench.yaml", "evals/rose-bench.yaml"),
    ("from rmc.", "from rose."),
    ("from rmc import", "from rose import"),
    ("import rmc", "import rose"),
    ("-m rmc", "-m rose"),
    ('"rmc.cli:main"', '"rose.cli:main"'),
    ('include = ["rmc*"]', 'include = ["rose*"]'),
    ('"rmc"', '"rose"'),
    ("_rmc", "_rose"),
    ('id="rmc-search"', 'id="rose-search"'),
    ("RSE Submission", "ROSE Submission"),
    ("# RSE ", "# ROSE "),
    ("Publishing RSE", "Publishing ROSE"),
    ("AUTO:RMC_BENCH", "AUTO:ROSE_BENCH"),
    ("RMC_BENCH_TABLE", "ROSE_BENCH_TABLE"),
    ("Updated RMC-Bench", "Updated ROSE-Bench"),
    ("=== RMC-Bench", "=== ROSE-Bench"),
    (".rmc/", ".rose/"),
    (".rmc'", ".rose'"),
    ('".rmc"', '".rose"'),
    ("`.rmc`", "`.rose`"),
    ("~/.rmc", "~/.rose"),
    ("STORE_DIRNAME = \".rmc\"", 'STORE_DIRNAME = ".rose"'),
    ("RMC_", "ROSE_"),  # env config prefix after specific keys above
    ("\\bRSE\\b", "ROSE"),
    ("\\bRMC\\b", "ROSE"),
    ("\\brmc\\b", "rose"),
]

SKIP_SUFFIXES = {
    ".pyc",
    ".png",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".woff",
    ".woff2",
    ".ico",
    ".zip",
    ".gz",
    ".whl",
}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "node_modules", ".venv", "venv"}


def git_mv(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=True)


def rename_paths() -> None:
    moves = [
        (ROOT / "rmc", ROOT / "rose"),
        (ROOT / "papers" / "rose", ROOT / "papers" / "rose"),
        (ROOT / "evals" / "rmc-bench.yaml", ROOT / "evals" / "rose-bench.yaml"),
        (ROOT / "bin" / "rmc", ROOT / "bin" / "rose"),
        (ROOT / "commands" / "rmc.md", ROOT / "commands" / "rose.md"),
        (ROOT / "tests" / "test_rmc.py", ROOT / "tests" / "test_rose.py"),
        (ROOT / "skills" / "recursive-memory", ROOT / "skills" / "rose"),
    ]
    for src, dst in moves:
        git_mv(src, dst)
    bench_latest = ROOT / "papers" / "rose" / "results" / "rmc-bench-latest.json"
    if bench_latest.exists():
        git_mv(bench_latest, bench_latest.with_name("rose-bench-latest.json"))


def should_process(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.suffix in SKIP_SUFFIXES:
        return False
    if path.name == "rename_to_rose.py":
        return False
    return path.is_file()


def replace_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        if old.startswith("\\b"):
            text = re.sub(old, new, text)
        else:
            text = text.replace(old, new)
    return text


def process_files() -> int:
    changed = 0
    for path in sorted(ROOT.rglob("*")):
        if not should_process(path):
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        updated = replace_text(raw)
        if updated != raw:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    rename_paths()
    n = process_files()
    print(f"Updated {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
