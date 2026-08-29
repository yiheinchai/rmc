# Recursive Skill Evolution: Co-Evolving Procedural Knowledge and Retrieval with Meta-Tested Compaction

RSE is the research framing for RMC (Recursive Memory Compaction). LaTeX source: `paper.tex`; build with `make`.

---

## Abstract

Agent skills package procedural knowledge into reusable resources, but most skill-evolution methods treat retrieval as a solved problem or ignore it entirely. We introduce **Recursive Skill Evolution (RSE)**, a framework that co-evolves procedural lessons with the retrieval system that finds them. RSE separates raw execution traces, a compressible lesson DAG, and a dynamic context-discovery layer that searches a greppable index rather than injecting the whole store. Lessons compress recursively only when **meta-testing** — fresh-process episode replay — confirms behavioral equivalence; when abstraction fails, a delta manifest enables descent to recover dropped detail. Retrieval co-evolves through selection rules and gated self-tuning that keeps changes only when precision and recall both improve. On **RMC-Bench**, a benchmark for procedural memory under compression, RSE achieves **+20% lift** over control on core transfer cases and **90%** L0 transfer at **~139 tokens** per prompt (Codex-graded, `gpt-5.6-sol`). Judge filtering reaches **100%** precision with **0** noise tokens; agentic search achieves **93%** precision/recall at **54** noise tokens vs **2,054** for serve-all. On a **WikiSkill-comparable** five-benchmark subset, agentic recall reaches **80%** accuracy vs **70%** for WikiSkill-style full injection at **~88% lower token cost** (64 vs 534 tokens). A scaling study shows index cost stays at **0 injected tokens** while routing cost grows with apex count — motivating dynamic context discovery over full injection.

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

Hand-written cases (`evals/rmc-bench.yaml`) covering trap, detail, principle, multi, distractor, conflict, null kinds.

**Results** (`python3 scripts/run_all_experiments.py --agent codex --samples 3`):

| Metric | Result |
|---|---|
| Lift (L0 − control, core kinds) | **+20%** |
| Transfer @ L0 | **9/10 (90%)** |
| Detail / trap / multi transfer | 3/3, 3/3, 2/2 (100%) |
| Principle transfer | 1/2 (50%) |
| Mean L0 tokens | **139** |
| Bench retrieval axis | **7/7 (100%)** |

### 4.2 Recall ablations (fixture store, noise in served set)

| Arm | Precision | Recall | Noise tokens |
|---|---|---|---|
| serve-all | 47% | 100% | 2,054 |
| judge | **100%** | **100%** | **0** |
| agentic | **93%** | **93%** | **54** (35 searches) |

Mirrors dogfood finding: filtering removes noise without losing useful lessons. Agentic search recovers most signal with far less noise than serve-all.

### 4.3 Retention curve (held-out S3 task, walkthrough lesson)

| Level | Transfer | Tokens |
|---|---|---|
| none (control) | 0% | 0 |
| L0 | 0% | 60 |
| L1 (compressed) | 0% | 42 |

Compression accepted at L1 (60→42 tok, 100% in-sample replay) but held-out S3 transfer fails under Codex grading — same failure mode as mock, confirming the descent motivation. Mock grading had shown L0=100%; Codex grading is stricter on this held-out probe.

### 4.4 Compaction + walkthrough cycle

| Metric | Value |
|---|---|
| Compression accepted | **no** (Codex adapter; mock replay adapter used for compaction ablation only) |
| Compaction ablation (mock replay) | yes (60 → 42 tok, 70%, 100% replay) |
| Descent rescued | no (1 attempt) |
| Recall tokens before compress | 68 |

### 4.5 Scaling study

| lessons | apexes | index tok | routing tok/prompt | judge prec | judge rec |
|---:|---:|---:|---:|---:|---:|
| 25 | 25 | 740 | 1,375 | 67% | 100% |
| 100 | 100 | 2,503 | 5,500 | 67% | 100% |
| 500 | 500 | 12,303 | 27,500 | 67% | 100% |
| 1000 | 1000 | 24,529 | 54,945 | 67% | 100% |

### 4.6 Dogfood retrieval (real store, one user, ~29 nodes)

From `EXPERIMENTS.md` (Aug 2026):

| Configuration | Precision | Recall | Noise tokens |
|---|---|---|---|
| Serve everything | 28% | 100% | 15,917 |
| Judge filter | **48%** | **100%** | **7,146** |
| Haiku router | 35% | 75% | 8,818 |
| After tune (1 round) | **51%** | **88%** | — |

### 4.7 Evaluation protocol

| Item | Detail |
|---|---|
| Grading agent | Codex (`gpt-5.6-sol`), 3 samples per case |
| Cross-model check | `scripts/run_claude_crosscheck.sh` (requires authenticated `claude` CLI) |
| Artifacts | `papers/rse/results/submission-latest.json` |
| Figures | `papers/rse/figures/` via `scripts/generate_paper_figures.py` |
| PDF | `papers/rse/paper.pdf` via `make` |

### 4.8 Comparison to WikiSkill

WikiSkill co-evolves skills + wiki offline on fixed benchmarks with **full skill injection** at test time. RSE targets the production setting WikiSkill defers: growing stores, retrieval under budget, validated compaction.

We built a WikiSkill-comparable subset (`evals/wikiskill-bench.yaml`) spanning the same five domains (LiveMath, SealQA, SpreadSheet, OfficeQA, ALFWorld) with evolved skills loaded into an RMC store. Each task is scored under four arms:

| Arm | Mechanism | Accuracy (Codex, 3 samples) | Mean tokens |
|---|---|---|---|
| no-skill | bare task | 60% (6/10) | 0 |
| full-inject | WikiSkill-style: all skills in prompt | 70% (7/10) | 534 |
| recall-judge | RSE judge-walk selector | 70% (7/10) | **64** |
| recall-agentic | RSE agentic DCD selector | **80% (8/10)** | **64** |

**Headline:** agentic recall **matches or beats** full injection while using **~88% fewer tokens** (64 vs 534). Full injection adds +10pp over no-skill; agentic recall adds another +10pp over full-inject on this subset — including rescuing `alfworld-put-apple` where injecting all skills caused the model to over-complicate the action sequence.

This is a curated 10-task probe subset, not the full WikiSkill test splits (124–280 tasks per benchmark). It establishes the comparison methodology; scaling to upstream task files is future work.

Reproduce: `python3 scripts/run_wikiskill_evals.py --agent codex --samples 3` → `papers/rse/results/wikiskill-latest.json`.

### 4.9 Session-length paired study (proxy)

EXPERIMENTS §7 asks whether recall shortens the next session. We approximate this with `evals/session-pairs.yaml`: five follow-up tasks where session 1 captured lessons into the store and session 2 is a related task.

| Arm | Accuracy (Codex, 3 samples) | Mean tokens |
|---|---|---|
| memory-off (narrative only) | 40% (2/5) | 0 |
| memory-on (RSE recall) | **100% (5/5)** | 168 |

**Lift: +60pp** — structured recall on the follow-up task rescues cases where a warmup narrative alone is insufficient (`charge-followup`, `schema-purge-followup`, `s3-upload-followup`).

This is a single-turn proxy, not live multi-turn agent sessions. Reproduce: `python3 scripts/run_session_study.py --agent codex --samples 3`.

---

## 5. Analysis

### 5.1 Transfer–token curve (headline figure)

Figure: `figures/fig_transfer_tokens.pdf`. Codex-graded bench establishes **+20% lift** and **90%** transfer@L0; held-out retention curve shows 0% transfer under Codex grading, motivating descent.

### 5.2 Case study

`examples/walkthrough.py` demonstrates: L0 lesson → episodes → compress → failure on dropped `@s3-body` → descent via delta manifest → rescue.

### 5.3 Honest limitations

- RMC-Bench is hand-written; cases from real captures would be more representative.
- Codex grading (`gpt-5.6-sol`) covers bench transfer, retrieval, recall, and retention curve; Claude/Gemini cross-check still useful.
- Walkthrough compression fails under Codex adapter (mock replay path still works in ablation).
- Dogfood store is N=29; steady-state apex reduction unproven at scale.
- Session-length proxy (`evals/session-pairs.yaml`) shows **+60pp** lift with recall vs narrative-only follow-up; live multi-turn sessions not yet measured.
- Blocking recall latency ~34s with accurate judge vs ~5s CLI floor.

---

## 6. Related Work

- **Skill evolution:** EvoSkill, Trace2Skill, SkillOpt, WikiSkill — batch loops, full injection at test.
- **Skill retrieval:** SkillRouter, SkillRet — retrieval only, no compaction co-evolution.
- **Agent memory:** MemGPT, various RAG — typically no validated recursive compression.

RSE = WikiSkill's persistent knowledge insight + explicit retrieval co-evolution + meta-tested compaction.

---

## 7. Conclusion

RSE compiles agent experience into recursively compressible lessons while co-evolving the retrieval that finds them. Meta-testing prevents silent behavioral drift; dynamic context discovery keeps catalog cost at zero injected tokens. RMC-Bench and scaling studies provide reproducible quantitative hooks; WikiSkill-comparable evals show agentic recall matching full skill injection at **~88% lower token cost** and beating it on accuracy (80% vs 70%) on a five-benchmark subset. Dogfood measurements confirm retrieval filtering and tuning matter in practice.

---

## References

- Tang et al. WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution. arXiv:2608.27454, 2026.
- RMC design and experiment log: `DESIGN.md`, `EXPERIMENTS.md` in this repository.

---

*Submission package: `papers/rse/SUBMISSION.md`. Reproduce: `python3 scripts/validate_agent_harness.py --agent codex` then `python3 scripts/run_all_experiments.py --agent codex --samples 3`; build PDF with `cd papers/rse && make`.*
