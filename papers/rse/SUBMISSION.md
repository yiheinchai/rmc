# Manuscript submission checklist

This document tracks readiness for conference submission (NeurIPS / ICML / ICLR).

## Artifacts

| Artifact | Path | Status |
|---|---|---|
| LaTeX source | `paper.tex` | Ready |
| Appendix + reproducibility | `appendix.tex` | Ready |
| Bibliography | `references.bib` | Ready (15 citations) |
| Figures (PDF + PNG) | `figures/` | Generated from eval JSON |
| PDF build | `paper.pdf` | `make` in this directory |
| Markdown draft (synced) | `paper.md` | Kept in sync with LaTeX |
| Eval results | `results/submission-latest.json` | Complete (Codex) |
| Figure generator | `../../scripts/generate_paper_figures.py` | Ready |

## Pre-submission steps

### Done in-repo

- [x] LaTeX manuscript with abstract, method, experiments, limitations, related work
- [x] Figures: WikiSkill, recall ablations, scaling, transfer–token, session study
- [x] Reproducibility appendix (NeurIPS-style checklist)
- [x] Consistent numbers (mean L0 tokens = 139 throughout)
- [x] Removed internal "remaining for submission" draft section
- [x] Expanded related work and bibliography
- [x] Unified eval report (`scripts/generate_submission_report.py`)

### Author action required

- [ ] **Swap LaTeX style** — replace `article` + `geometry` with official venue template (`neurips_2024.sty`, `iclr2025_conference.sty`, or ICML style) before upload
- [ ] **Claude cross-check** — run `claude` → `/login`, then `./scripts/run_claude_crosscheck.sh`
- [ ] **Anonymize** for double-blind venues (remove author block, de-identify repo URLs if required)
- [ ] **Page limit check** — current draft is ~8 pages + appendix; verify against venue limits
- [ ] **Broader eval scale** (recommended for main track, not blocking arXiv):
  - Scale WikiSkill to upstream task files (124–280 tasks/benchmark)
  - Expand RMC-Bench beyond 10 hand-written cases
  - Add bootstrap CIs with `--samples 5` or higher
  - Run external baselines (SkillRouter, MemGPT, vanilla RAG)

## Build commands

```bash
# Regenerate figures from latest eval results
python3 scripts/generate_submission_report.py
python3 scripts/generate_paper_figures.py

# Build PDF
cd papers/rse && make

# Full eval refresh (requires Codex auth)
python3 scripts/validate_agent_harness.py --agent codex
python3 scripts/run_all_experiments.py --agent codex --samples 3
```

## Venue-specific notes

| Venue | Template | Page limit | Notes |
|---|---|---|---|
| NeurIPS | `neurips_2024.sty` | 9 main + unlimited appendix | Reproducibility checklist required |
| ICML | `icml2025.sty` | 8 main + appendix | Broader baselines expected |
| ICLR | `iclr2025_conference.sty` | 9 main + appendix | OpenReview upload |
| arXiv | `article` (current) | No limit | Ready now with current template |

## Honest scope statement

The manuscript is **submission-formatted** with complete Codex-graded results and reproducible harnesses.
For **competitive main-track** acceptance, reviewers will likely request: full WikiSkill splits, cross-model grading, statistical significance, and stronger compaction results on held-out tasks.
For **arXiv or workshop** submission, the current package is ready after style swap and optional Claude cross-check.
