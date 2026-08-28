# Publishing RSE — reproduction guide

This directory contains the **Recursive Skill Evolution (RSE)** paper draft and
machine-readable evaluation results.

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run the full publication suite (mock backend — no API keys)
python3 scripts/run_paper_evals.py

# Run with a real agent when available
python3 scripts/run_paper_evals.py --agent claude --samples 3

# Run RMC-Bench only
python3 -m rmc.cli bench --agent mock --json

# Recall ablations (requires a populated .rmc store with episodes)
rmc eval-recall --arm judge --save judge
rmc eval-recall --arm serve-all --against judge
rmc eval-recall --arm agentic --against judge
```

Results land in `papers/rse/results/`:

| File | Contents |
|---|---|
| `summary-latest.json` | Combined bench + scaling snapshot |
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
| RMC-Bench results | Mock numbers in `results/`; **re-run with Claude for submission** |
| WikiSkill-comparable benchmarks | Not yet integrated — see `paper.md` §Future work |
| Ablations (no meta-test, no retrieval tune) | CLI flags exist (`--skip-replay`, recall arms) |

## Next steps for submission-quality numbers

1. `python3 scripts/run_paper_evals.py --agent claude --samples 3`
2. Wire WikiSkill benchmark harness (SpreadSheet, ALFWorld, etc.) with RSE recall
3. Run paired on/off session-length study (EXPERIMENTS.md §7 — product claim)
4. Fill ablation table: meta-testing off, serve-all, tune off
