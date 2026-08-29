# Publishing ROSE — manuscript and reproduction guide

This directory contains the **Recursive Online Skill Evolution (ROSE)** submission package:
LaTeX manuscript, figures, evaluation results, and reproducibility appendix.

## Manuscript

| File | Purpose |
|---|---|
| `paper.tex` | Main LaTeX source (9 pages + appendix) |
| `appendix.tex` | Reproducibility checklist + supplementary tables |
| `references.bib` | Bibliography (15 citations) |
| `paper.pdf` | Built PDF (`make`) |
| `figures/` | Publication figures (PDF + PNG) |
| `SUBMISSION.md` | Pre-submission checklist |
| `paper.md` | Markdown draft (kept in sync) |

```bash
cd papers/rose
make          # build paper.pdf
make figures  # regenerate figures from results JSON
```

## Quick start (evaluations)

```bash
pip install -e ".[dev,paper]"

# Run ALL experiments (bench, scaling, recall, compaction, retention, wikiskill, session)
python3 scripts/run_all_experiments.py --agent codex --samples 3

# Individual suites
python3 scripts/run_wikiskill_evals.py --agent codex --samples 3
python3 scripts/run_session_study.py --agent codex --samples 3
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py
```

Results land in `papers/rose/results/`:

| File | Contents |
|---|---|
| `submission-latest.json` | Unified report: all evals + comparative baselines |
| `experiments-full-latest.json` | Complete suite |
| `wikiskill-latest.json` | WikiSkill-comparable four-arm comparison |
| `session-study-latest.json` | Session paired study |
| `rose-bench-latest.json` | Per-case transfer/retrieval scores |
| `recall-ablations-latest.json` | serve-all vs judge vs agentic |
| `scaling-latest.json` | Synthetic scaling table |

## What is measured

### ROSE-Bench (`evals/rose-bench.yaml`)

Four axes: transfer, retention, retrieval, cost.

### WikiSkill-comparable (`evals/wikiskill-bench.yaml`)

Five domains, four arms: no-skill, full-inject, recall-judge, recall-agentic.

### Session paired study (`evals/session-pairs.yaml`)

Memory-off vs memory-on follow-up tasks.

### Scaling study (`rose/scaling.py`)

Synthetic stores at 25/100/500/1000 lessons.

## Submission status

See `SUBMISSION.md` for the full checklist.

**Ready now:** LaTeX manuscript, figures, reproducibility appendix, Codex-graded results.

**Author action:** swap in official venue LaTeX template, run Claude cross-check (`./scripts/run_claude_crosscheck.sh` after `claude` login), anonymize for double-blind if required.
