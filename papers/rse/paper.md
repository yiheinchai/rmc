# Recursive Skill Evolution: Co-Evolving Procedural Knowledge and Retrieval with Meta-Tested Compaction

**Working title.** RSE is the research framing for RMC (Recursive Memory Compaction).

---

## Abstract

Agent skills package procedural knowledge into reusable resources, but most skill-evolution methods treat retrieval as a solved problem or ignore it entirely. We introduce **Recursive Skill Evolution (RSE)**, a framework that co-evolves procedural lessons with the retrieval system that finds them. RSE separates raw execution traces, a compressible lesson DAG, and a dynamic context-discovery layer that searches a greppable index rather than injecting the whole store. Lessons compress recursively only when **meta-testing** — fresh-process episode replay — confirms behavioral equivalence; when abstraction fails, a delta manifest enables descent to recover dropped detail. Retrieval co-evolves through selection rules and gated self-tuning that keeps changes only when precision and recall both improve. On **RMC-Bench**, a benchmark for procedural memory under compression, RSE achieves **+70% lift** over control on core transfer cases and **90%** L0 transfer at **~111 tokens** per prompt (mock grading). A scaling study shows index cost stays at **0 injected tokens** while routing cost grows with apex count — motivating dynamic context discovery over full injection. We report negative results from production dogfooding — cheap-model routers hurt, unaudited metrics misled optimization — and outline the path to WikiSkill-comparable end-to-end benchmarks.

---

## 1. Introduction

General-purpose agents need domain-specific procedure. Skills (filesystem modules with `SKILL.md`) make that knowledge auditable and reusable. Recent work evolves skills from execution traces (EvoSkill, Trace2Skill, SkillOpt, WikiSkill), but three gaps remain:

1. **Retrieval is ignored.** WikiSkill full-injects skills at test time to isolate skill *quality*. Production stores outgrow context; selection is the problem.
2. **Compaction is unvalidated.** Skill edits shrink text without a regression gate; lost detail fails silently.
3. **Knowledge and retrieval are not co-evolved.** Selection rules and prompts do not improve from measured outcomes.

RSE addresses all three. It is an ambient loop — recall before work, reflect after, compact when earned, tune retrieval when measured — not a batch training pipeline.

**Contributions:**

1. **RSE framework** — three layers (raw traces, lesson DAG, dynamic retrieval) with a continual co-evolution loop.
2. **Meta-tested recursive compaction** — compression accepted only when episode replay passes at 100%; delta manifests enable descent on failure.
3. **Dynamic context discovery** — agentic index search: 0 tokens for the catalog, bounded injection for selected bodies.
4. **Retrieval co-evolution** — `rmc tune` proposes prompt/config changes; keeps only if precision **and** recall improve.
5. **RMC-Bench + scaling study** — four-axis procedural memory benchmark with reproducible harness; synthetic scaling table.

---

## 2. Problem Setup

Let \(\mathcal{D}\) be tasks an agent performs over time. Unlike WikiSkill's fixed train/val/test split, RSE operates **online**: each session produces traces \(\tau_i\), optional lessons \(L\), and attribution of which lessons bore on success.

State at step \(k\): \((\mathcal{N}_k, \mathcal{R}_k, \mathcal{I}_k)\)

- \(\mathcal{N}_k\) — lesson DAG (nodes with levels L0…Ln, `dropped[]` manifests)
- \(\mathcal{R}_k\) — retrieval policy (selection rules, selector prompts, config)
- \(\mathcal{I}_k\) — greppable index (searched, never injected wholesale)

**Metrics:**

| Symbol | Meaning |
|---|---|
| \(\mathcal{R}_{task}\) | Task success / blind-graded transfer |
| \(\text{Prec}, \text{Rec}\) | Retrieval precision/recall vs `episode.used` |
| \(C_{inj}\) | Injection tokens per prompt |
| \(C_{route}\) | Routing/selection tokens per prompt |

Thesis: transfer stays flat while \(C_{inj}\) falls under compression, and \(\text{Prec}\) stays high as \(|\mathcal{N}|\) grows.

---

## 3. Method

### 3.1 Three-layer architecture

```
Raw layer       episodes/, sessions/, events.jsonl
Knowledge layer nodes/<family>/*.md  (DAG, delta manifests)
Retrieval layer index.md + routing/*.md + selector agent
```

**Principle:** harness owns structure; model owns meaning. All semantic judgements flow through a single judge interface (relevance, compress, replay, assess).

### 3.2 Loop

1. **Recall** — fork live session, grep index + nodes, return lesson ids (`select_agent.py`)
2. **Execute** — agent works with context pack
3. **Reflect** — mint L0 lessons from session digest (`reflect.py`)
4. **Attribute** — record `episode.used` for lessons that bore on work
5. **Compact** — when due: compress → meta-test replay → promote or reject (`compact.py`)
6. **Tune** — measure recall, propose one change, keep iff prec **and** rec improve (`tune.py`)

### 3.3 Meta-tested compaction

Accept candidate \(L_{k+1}\) only if:

\[
\text{pass\_rate}(L_{k+1}, \mathcal{E}_{reg}) = 1.0 \quad \land \quad |L_{k+1}| \leq \rho \cdot |L_k|
\]

Replay runs in a **fresh process** with only the compressed body — no context leak. Failures record `preserve[]` hints; repeated rescues trigger **repair** (fold detail back).

**Ablation:** `--skip-replay` disables meta-testing for controlled degradation studies.

### 3.4 Dynamic context discovery

The index holds one line per lesson (~0 tokens injected). The selector agent searches it and `nodes/` on demand. Selection rules (`routing/*.md`) capture meta-knowledge: "when task X, look in family Y."

Warm-prefix caching (`--resume --fork-session`) amortizes routing cost across prompts.

### 3.5 Retrieval co-evolution

`rmc eval-recall` scores served lessons against `episode.used`. Arms:

| Arm | What it tests |
|---|---|
| `serve-all` | No filtering (baseline noise) |
| `judge` | Apex-walk relevance filter |
| `agentic` | Cold search over whole store |

`rmc tune` closes the loop on \(\mathcal{R}_k\).

---

## 4. Experiments

### 4.1 RMC-Bench (procedural memory)

Hand-written cases (`evals/rmc-bench.yaml`) covering trap, detail, principle, multi, distractor, conflict, null kinds. Scored on four axes per `evals/README.md`.

**Mock results** (reproducible, no API keys) — `papers/rse/results/summary-latest.json`:

| Metric | Result |
|---|---|
| Lift (L0 − control, core kinds) | **+70%** |
| Transfer @ L0 | **9/10 (90%)** |
| Detail transfer | 3/3 (100%) |
| Trap transfer | 3/3 (100%) |
| Mean L0 tokens | **111** |
| Retrieval (judge heuristic) | 4/7 (57%) |

Run: `python3 scripts/run_paper_evals.py` or `rmc bench --agent mock`

**Submission:** re-run with `python3 scripts/run_paper_evals.py --agent claude --samples 3`.

### 4.2 Scaling study

Synthetic stores (25 / 100 / 500 lessons):

| lessons | apexes | index tok | routing tok/prompt | judge prec | judge rec |
|---:|---:|---:|---:|---:|---:|
| 25 | 25 | 740 | 1,375 | 67% | 100% |
| 100 | 100 | 2,503 | 5,500 | 67% | 100% |
| 500 | 500 | 12,303 | 27,500 | 67% | 100% |

Index is searched, not injected — the scaling argument is that **full injection fails at 500 lessons (~12k tokens catalog alone)** while search stays bounded by what the selector opens.

### 4.3 Dogfood retrieval (real store, one user, ~29 nodes)

From `EXPERIMENTS.md` (Aug 2026):

| Configuration | Precision | Recall | Noise tokens |
|---|---|---|---|
| Serve everything | 28% | 100% | 15,917 |
| Judge filter | **48%** | **100%** | **7,146** |
| Haiku router | 35% | 75% | 8,818 |
| After tune (1 round) | **51%** | **88%** | — |

**Negative result:** cheap-model routing is worse than no filter — include in paper.

### 4.4 Ablations (planned)

| Config | Expected effect |
|---|---|
| RSE full | baseline |
| − meta-testing (`--skip-replay`) | retention drops at L2+ |
| − compaction (L0 only) | tokens rise, transfer flat |
| − judge filter (`serve-all`) | precision ↓, noise ↑ |
| − tune | recall stagnates |

### 4.5 Comparison to WikiSkill

WikiSkill co-evolves skills + wiki offline on fixed benchmarks with full skill injection. RSE targets the **production setting** WikiSkill defers: growing stores, retrieval under budget, validated compaction. Complementary, not competing — future work integrates WikiSkill benchmarks with RSE recall.

---

## 5. Analysis

### 5.1 Transfer–token curve (headline figure)

Plot transfer@level vs mean tokens across compression levels. Thesis holds if curve is flat while tokens fall. Mock bench establishes pipeline; Claude run fills the curve.

### 5.2 Case study

`examples/walkthrough.py` demonstrates: L0 lesson → episodes → compress → failure on dropped `@s3-body` → descent via delta manifest → rescue.

### 5.3 Honest limitations

- RMC-Bench is hand-written; cases from real captures would be more representative.
- Mock grading approximates blind judge; submission needs Claude/Gemini runs.
- Dogfood store is N=29; steady-state apex reduction unproven at scale.
- End-to-end "does recall shorten the next session?" not yet measured (EXPERIMENTS §7).
- Blocking recall latency ~34s with accurate judge vs ~5s CLI floor.

---

## 6. Related Work

- **Skill evolution:** EvoSkill, Trace2Skill, SkillOpt, WikiSkill — batch loops, full injection at test.
- **Skill retrieval:** SkillRouter, SkillRet — retrieval only, no compaction co-evolution.
- **Agent memory:** MemGPT, various RAG — typically no validated recursive compression.

RSE = WikiSkill's persistent knowledge insight + explicit retrieval co-evolution + meta-tested compaction.

---

## 7. Conclusion

RSE compiles agent experience into recursively compressible lessons while co-evolving the retrieval that finds them. Meta-testing prevents silent behavioral drift; dynamic context discovery keeps catalog cost at zero injected tokens. RMC-Bench and scaling studies provide reproducible quantitative hooks; dogfood measurements confirm retrieval filtering and tuning matter in practice. Next: WikiSkill-comparable benchmark runs with recall enabled, completing the ablation table for submission.

---

## References

- Tang et al. WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution. arXiv:2608.27454, 2026.
- RMC design and experiment log: `DESIGN.md`, `EXPERIMENTS.md` in this repository.

---

*Generated evaluation artifacts: `papers/rse/results/`. Reproduce: `python3 scripts/run_paper_evals.py`.*
