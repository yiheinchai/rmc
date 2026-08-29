#!/usr/bin/env python3
"""Generate publication figures from papers/rse/results/*.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "papers" / "rse" / "results"
FIGURES = ROOT / "papers" / "rse" / "figures"


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def fig_wikiskill(report: dict) -> Path:
    arms = report.get("wikiskill", {})
    labels = ["no-skill", "full-inject", "recall-judge", "recall-agentic"]
    acc = [arms.get(a, {}).get("accuracy", 0) * 100 for a in labels]
    tok = [arms.get(a, {}).get("mean_tokens", 0) for a in labels]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, acc, w, label="Accuracy (%)", color="#4C72B0")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace("-", "\n") for l in labels], fontsize=8)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w / 2, tok, w, label="Mean tokens", color="#DD8452")
    ax2.set_ylabel("Mean injected tokens")
    ax2.set_ylim(0, max(tok) * 1.2 if tok else 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)
    ax1.set_title("WikiSkill-comparable subset (10 tasks, Codex-graded)")
    fig.tight_layout()
    out = FIGURES / "fig_wikiskill.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_recall_ablations(report: dict) -> Path:
    arms = report.get("recall_ablations", {})
    labels = ["serve-all", "judge", "agentic"]
    prec = [arms.get(a, {}).get("precision", 0) * 100 for a in labels]
    rec = [arms.get(a, {}).get("recall", 0) * 100 for a in labels]
    noise = [arms.get(a, {}).get("noise_tokens", 0) for a in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w / 2, prec, w, label="Precision", color="#55A868")
    ax1.bar(x + w / 2, rec, w, label="Recall", color="#4C72B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("%")
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.set_title("Retrieval quality")

    ax2.bar(labels, noise, color="#C44E52")
    ax2.set_ylabel("Noise tokens")
    ax2.set_title("Injection noise")
    ax2.set_yscale("log")
    fig.suptitle("Recall ablations (fixture store)", y=1.02)
    fig.tight_layout()
    out = FIGURES / "fig_recall_ablations.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_scaling(report: dict) -> Path:
    rows = report.get("scaling", [])
    if not rows:
        return FIGURES / "fig_scaling.pdf"
    lessons = [r["lessons"] for r in rows]
    routing = [r["routing_tokens_per_prompt"] for r in rows]
    index_tok = [r["index_tokens"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(lessons, routing, "o-", label="Routing tokens / prompt", color="#4C72B0")
    ax.set_xlabel("Lessons in store")
    ax.set_ylabel("Routing tokens per prompt")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(lessons, index_tok, "s--", label="Index size (not injected)", color="#DD8452", alpha=0.8)
    ax2.set_ylabel("Index tokens (catalog)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.set_title("Scaling: routing cost grows; index stays out of context")
    fig.tight_layout()
    out = FIGURES / "fig_scaling.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_transfer_tokens(report: dict, bench: dict) -> Path:
    """Transfer rate vs mean tokens across bench arms and retention levels."""
    points: list[tuple[str, float, float]] = []

    transfer = bench.get("transfer") or {}
    if transfer.get("total"):
        cases = bench.get("cases") or []
        none_cases = [c for c in cases if c.get("arm") == "none" and c.get("kind") in ("trap", "detail", "principle", "multi")]
        l0_cases = [c for c in cases if c.get("arm") == "L0" and c.get("kind") in ("trap", "detail", "principle", "multi")]
        if none_cases:
            rate = sum(1 for c in none_cases if c.get("passed")) / len(none_cases)
            points.append(("control", rate * 100, 0))
        if l0_cases:
            rate = sum(1 for c in l0_cases if c.get("passed")) / len(l0_cases)
            mean_tok = sum(c.get("tokens", 0) for c in l0_cases) / len(l0_cases)
            points.append(("L0", rate * 100, mean_tok))

    retention = report.get("retention_curve", {}).get("levels", {})
    for level in ("L0", "L1"):
        row = retention.get(level, {})
        if row:
            points.append((f"held-out {level}", row.get("pass_rate", 0) * 100, row.get("tokens", 0)))

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for label, rate, tok in points:
        ax.scatter(tok, rate, s=80, zorder=3)
        ax.annotate(label, (tok, rate), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean injected tokens")
    ax.set_ylabel("Transfer / pass rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Transfer vs injection cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "fig_transfer_tokens.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_session_study(report: dict) -> Path:
    ss = report.get("session_study", {})
    if not ss:
        return FIGURES / "fig_session_study.pdf"
    labels = ["memory-off", "memory-on"]
    acc = [ss.get(a, {}).get("accuracy", 0) * 100 for a in labels]
    tok = [ss.get(a, {}).get("mean_tokens", 0) for a in labels]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = np.arange(len(labels))
    ax.bar(x, acc, color=["#AAAAAA", "#4C72B0"])
    for i, (a, t) in enumerate(zip(acc, tok)):
        ax.text(i, a + 2, f"{a:.0f}%\n({t} tok)", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Narrative only", "RSE recall"])
    ax.set_ylabel("Follow-up accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Session paired study (5 pairs)")
    fig.tight_layout()
    out = FIGURES / "fig_session_study.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _style()
    report = _load("submission-latest.json")
    bench = _load("rmc-bench-latest.json") or _load("experiments-full-latest.json").get("bench", {})

    paths = [
        fig_wikiskill(report),
        fig_recall_ablations(report),
        fig_scaling(report),
        fig_transfer_tokens(report, bench),
        fig_session_study(report),
    ]
    for p in paths:
        if p.exists():
            print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
