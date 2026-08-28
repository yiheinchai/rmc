# Publishing RSE — reproduction guide

This directory contains the **Recursive Skill Evolution (RSE)** paper draft and
machine-readable evaluation results.

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run ALL experiments (bench, scaling, recall ablations, compaction, retention curve)
python3 scripts/run_all_experiments.py --samples 3

# Bench + scaling only
python3 scripts/run_paper_evals.py

# With a real agent when available
python3 scripts/run_all_experiments.py --agent claude --samples 3
```

Results land in `papers/rse/results/`:

| File | Contents |
|---|---|
| `experiments-full-latest.json` | Complete suite: bench + scaling + recall + compaction + retention |
| `recall-ablations-latest.json` | serve-all vs judge precision/recall |
| `compaction-ablation-latest.json` | meta-testing on/off |
| `rmc-bench-latest.json` | Per-case transfer/retrieval scores |
| `scaling-latest.json` | Synthetic store scaling table |

## What is measured

### RMC-Bench (`evals/rmc-bench.yaml`)

Four axes from `evals/README.md`:

1. **Transfer** — control vs L0 on trap/detail/principle/multi cases
2. **Retention** — L0 vs L1 after meta-tested compression (detail cases)
3. **Retrieval** — distractor/null/conflict/multi selection quality
4. **Cost** — mean tokens injected per prompt

### Scaling study (`rmc/scaling.py`)

Synthetic stores at 25/100/500 lessons. Reports index size, apex count, routing
token estimate, and mock judge precision/recall on seeded episodes.

### Dogfood numbers (`EXPERIMENTS.md`)

Real-store measurements from one month of RMC usage — retrieval filtering,
tuning, negative results. Cite alongside benchmark numbers.

## Paper status

| Section | Status |
|---|---|
| Abstract + contributions | Draft in `paper.md` |
| Method (§3) | Mapped to `DESIGN.md` / codebase |
| RMC-Bench + full suite | **Run** — `experiments-full-latest.json` |
| Recall ablations | **Run** — judge 100% prec vs serve-all 47% |
| Retention curve | **Run** — L0 100% → L1 0% on held-out S3 |
| WikiSkill-comparable benchmarks | Not yet integrated |

## Next steps for submission-quality numbers

1. `python3 scripts/run_all_experiments.py --agent claude --samples 3`
2. Wire WikiSkill benchmark harness with RSE recall
3. End-to-end session-length paired study (EXPERIMENTS.md §7)
