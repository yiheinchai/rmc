"""Cross-model skill transfer evaluation (WikiSkill Table 2 style).

Skills are stored in an RMC store (evolved offline). Each *inference model*
grades the same tasks with the same skill set injected — measuring whether
procedural knowledge transfers across model families when retrieval is held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import Adapter, get_adapter
from .bench import bench_adapter
from .grader_specs import parse_grader_spec
from .wikiskill import CORE_ARMS, WikiSkillCase, build_store, load_bench, run, to_dict as wikiskill_to_dict

DEFAULT_BENCH = Path(__file__).resolve().parents[1] / "evals" / "wikiskill-bench.yaml"


@dataclass
class TransferCell:
    inference_model: str
    skill_source: str
    benchmark: str
    arm: str
    accuracy: float
    passed: int
    total: int


@dataclass
class CrossTransferReport:
    cells: list[TransferCell] = field(default_factory=list)
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    bench_path: str = ""

    def render(self) -> str:
        lines = ["Cross-model transfer (WikiSkill Table 2 style)", ""]
        models = sorted({c.inference_model for c in self.cells})
        for model in models:
            rows = [c for c in self.cells if c.inference_model == model and c.arm == "recall-agentic"]
            if not rows:
                continue
            avg = sum(c.accuracy for c in rows) / len(rows)
            lines.append(f"  {model:<12} recall-agentic avg={avg:.0%}  ({len(rows)} benchmarks)")
        return "\n".join(lines)


def run_cross_transfer(
    graders: list[str],
    *,
    bench_path: Path | None = None,
    skill_source: str = "wikiskill-probe",
    samples: int = 3,
    limit: int | None = None,
    arms: tuple[str, ...] = ("full-inject", "recall-agentic"),
) -> CrossTransferReport:
    """Evaluate the same skill store under multiple inference/grading backends."""
    path = bench_path or DEFAULT_BENCH
    cases, _ = load_bench(path)
    if limit:
        cases = cases[:limit]

    report = CrossTransferReport(bench_path=str(path))

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = build_store(
            cases,
            Path(tmp) / "repo",
            dedupe_families=path.suffix == ".jsonl",
        )
        for grader_spec in graders:
            backend, model, label = parse_grader_spec(grader_spec)
            raw = get_adapter(backend, model=model)
            if not raw.available():
                continue
            adapter = bench_adapter(raw) if backend == "mock" else raw
            ws_report = run(
                adapter,
                path=path,
                samples=samples,
                store=store,
                arms=arms,
                limit=limit,
            )
            report.by_model[label] = {
                "skill_source": skill_source,
                "agent": backend,
                "model": model,
                **wikiskill_to_dict(ws_report),
            }
            for arm in arms:
                for bench, stats in ws_report.by_benchmark(arm).items():
                    p, t = stats
                    report.cells.append(
                        TransferCell(
                            inference_model=label,
                            skill_source=skill_source,
                            benchmark=bench,
                            arm=arm,
                            accuracy=p / t if t else 0.0,
                            passed=p,
                            total=t,
                        )
                    )
    return report


def to_dict(report: CrossTransferReport) -> dict[str, Any]:
    table: dict[str, dict[str, dict[str, float]]] = {}
    for cell in report.cells:
        table.setdefault(cell.inference_model, {}).setdefault(cell.benchmark, {})[cell.arm] = cell.accuracy

    return {
        "bench_path": report.bench_path,
        "table": table,
        "models": report.by_model,
        "render": report.render(),
    }
