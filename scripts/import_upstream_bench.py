#!/usr/bin/env python3
"""Import upstream benchmark splits into evals/upstream/*.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re
from urllib.parse import unquote

from rmc import yamlish

MANIFEST = ROOT / "evals" / "upstream" / "manifest.yaml"
OUT_DIR = ROOT / "evals" / "upstream"


def _normalize_urls(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(u).strip() for u in raw if str(u).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(u).strip() for u in parsed if str(u).strip()]
        except json.JSONDecodeError:
            pass
    return [text]


def _url_text_snippets(urls: list[str]) -> list[str]:
    """Extract Wikipedia #:~:text= fragments as synthetic evidence snippets."""
    snippets: list[str] = []
    for url in urls:
        if ":~:text=" not in url:
            continue
        tail = url.split(":~:text=", 1)[1]
        for frag in re.split(r",-,", tail):
            text = unquote(frag.replace("+", " ")).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) >= 8:
                snippets.append(text)
    return snippets


def _row_to_case(row: dict, spec: dict, idx: int) -> dict:
    task_field = spec["task_field"]
    expected_field = spec["expected_field"]
    question = str(row.get(task_field) or "").strip()
    expected = str(row.get(expected_field) or "").strip()
    urls = _normalize_urls(row.get("urls"))
    snippets = _url_text_snippets(urls)
    evidence = ""
    if snippets:
        evidence = "\n\nEvidence snippets (from reference URLs):\n" + "\n".join(
            f"- {s}" for s in snippets[:4]
        )
    elif urls:
        evidence = "\n\nReference URLs:\n" + "\n".join(f"- {u}" for u in urls[:3])
    task = f"{question}{evidence}\n\nAnswer with a short factual response starting with 'Answer:'."
    case_id = f"{spec['id']}-{idx:04d}"
    return {
        "id": case_id,
        "benchmark": spec["benchmark"],
        "family": spec.get("family") or spec["benchmark"].lower(),
        "task": task,
        "expected": f"Answer: {expected}" if not expected.lower().startswith("answer") else expected,
        "skill": str(spec.get("skill") or "").strip(),
    }


def import_source(spec: dict) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("pip install datasets") from exc

    ds = load_dataset(
        spec["hf_dataset"],
        name=spec.get("hf_config"),
        split=spec["hf_split"],
    )
    limit = spec.get("limit")
    if limit:
        ds = ds.select(range(min(int(limit), len(ds))))

    cases = [_row_to_case(row, spec, i + 1) for i, row in enumerate(ds)]
    out = OUT_DIR / f"{spec['id']}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} cases to {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Import upstream benchmark JSONL files")
    parser.add_argument("--source", action="append", help="manifest source id (repeatable)")
    parser.add_argument("--all", action="store_true", help="import every source in manifest")
    args = parser.parse_args()

    raw = yamlish.load(MANIFEST.read_text(encoding="utf-8"))
    sources = {s["id"]: s for s in (raw.get("sources") or [])}
    ids = list(sources) if args.all else (args.source or [])
    if not ids:
        print("Specify --all or --source <id>", file=sys.stderr)
        return 1

    for sid in ids:
        if sid not in sources:
            print(f"unknown source: {sid}", file=sys.stderr)
            return 1
        try:
            import_source(sources[sid])
        except Exception as exc:
            print(f"WARN: failed to import {sid}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
