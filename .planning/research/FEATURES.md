# Feature Landscape

**Domain:** LongMemEval benchmark harness for a Rust memory engine
**Researched:** 2026-03-15

## Table Stakes

Features users (readers, reviewers, competitors) expect. Missing = results are not credible.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| All 500 questions scored | LongMemEval is defined as 500 questions. Partial runs must be disclosed. Every published system reports on all 500 | Low | Category counts: single-session-user (70), single-session-assistant (56), single-session-preference (30), multi-session (133), knowledge-update (78), temporal-reasoning (133). The 30 abstention questions are drawn from the other types and tagged separately |
| Per-category accuracy breakdown | The paper reports results per question type. All published evaluators (Backboard, Hindsight, EverMemOS, Supermemory, Mastra) report per-category tables | Low | 7 categories: the 6 base types plus abstention as a cross-cutting 7th. The paper groups these into 5 "core memory abilities": Information Extraction (user+assistant), Multi-Session Reasoning, Knowledge Updates, Temporal Reasoning, Abstention |
| GPT-4o judge (or explicit alternative) | The paper uses GPT-4o-2024-08-06 as the canonical judge with >97% agreement with human experts. Deviating without disclosure invalidates comparability | Low | Backboard used GPT-4o-mini and got 93.4%; Hindsight used the paper's GPT-4o setup. Elephant should default to GPT-4o for comparability, with configurable override (matching LoCoMo pattern) |
| Per-instance bank isolation | Each of the 500 questions has its own conversation history (~40-50 sessions in S, ~500 in M). Banks cannot be shared across questions | High | This is the fundamental architectural difference from LoCoMo (10 shared conversations). 500 separate banks is the main engineering challenge |
| LongMemEval-S support | The S setting (~115k tokens, ~40-50 sessions per question) is the standard evaluation setting. Every published result uses at least S | Med | ~57.5M total tokens across 500 instances. M (~500 sessions, ~1.5M tokens per question) is aspirational |
| `question_date` passthrough to reflect | Questions have a `question_date` field (tq) representing when the user asks the question. The paper specifies tq > tN (after all history). Temporal reasoning depends on this | Low | Must be passed to the reflect agent so it knows "today's date" for temporal questions like "how long ago did X happen" |
| Overall accuracy metric | Single headline number for comparability. Correct / total across all 500 questions | Low | This is the number that goes in comparison tables |
| Dataset fingerprinting | Reproducibility requires proving you ran against the exact same dataset. LoCoMo already does this | Low | Hash of `longmemeval_s_cleaned.json` or `longmemeval_m_cleaned.json` |
| Full manifest/provenance | Model versions, prompt hashes, runtime config, git commit, dataset fingerprint, consolidation mode. LoCoMo already has this pattern | Low | Direct carry-over from LoCoMo infrastructure. Same reproducibility contract |
| Three-artifact output | Summary JSON + question JSONL sidecar + debug JSONL sidecar. Established by LoCoMo, expected for serious benchmarks | Med | Artifact schema needs LongMemEval-specific fields (question_type, per-instance bank_id) but same structural pattern |
| Abstention scoring | The 30 false-premise questions must be scored. The judge evaluates whether the system correctly refuses to answer | Low | The paper's GPT-4o judge handles this with its standard prompt. Abstention questions are skipped for retrieval metrics only, not for QA accuracy |
| run/ingest/qa subcommands | Separating bank construction from QA evaluation enables iteration. LoCoMo already has this pattern; LongMemEval's per-instance model makes it even more important (500 banks = expensive to rebuild) | Med | `ingest` builds 500 banks. `qa` scores against existing banks. `run` does both. Critical for iteration speed |

## Differentiators

Features that set Elephant's evaluation apart. Not expected, but valued for publication credibility.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Stage-level cost metrics (retain, consolidate, reflect, judge) | Only EverMemOS publishes stage-level token breakdowns. Elephant already does this for LoCoMo. Showing where tokens go (extraction vs reflection vs judging) is a strong credibility signal | Low | Direct carry-over from LoCoMo. The per-instance model means per-instance stage metrics are also natural |
| Retrieval provenance (retrieved facts, reflect traces) | The paper defines Recall@k and NDCG@k as optional retrieval metrics when systems expose retrieval results. Elephant's debug sidecar already captures reflect traces and retrieved context | Med | LongMemEval provides `has_answer: true` turn annotations and `answer_session_ids` for ground truth retrieval location. Mapping Elephant's retrieved facts back to source sessions enables retrieval metrics. Most competitors do not report these |
| Evidence recall against `answer_session_ids` | The dataset tags which sessions contain the answer. Computing whether Elephant's retrieval surfaced those sessions is a novel analysis most competitors skip (the paper notes retrieval metrics are optional) | Med | Analogous to LoCoMo's evidence recall but using session-level rather than turn-level ground truth |
| Multi-judge variance analysis | Backboard validated with 4 judge models (GPT-4, GPT-4o-mini, Gemini 3 Pro, GPT-5.2) showing 91.2-93.4% range. Reporting judge variance is a strong signal of result stability | Med | LoCoMo protocol already calls for "at least one rerun or judge-variance check." Extending to multi-judge is straightforward with configurable judge |
| LongMemEval-M support | Only the paper itself evaluates on M (~500 sessions, ~1.5M tokens per question). Running M proves the system handles production-scale history. No published competitor has run M | Very High | ~750M total tokens across 500 instances. Extremely expensive ($$$) and slow. Aspirational for full publication; S is sufficient for competitive claims |
| merge subcommand | Assembling full 500-question runs from batch slices. LoCoMo already has this with strict compatibility checks | Med | With 500 per-instance banks, batch execution is likely necessary. Merge must validate disjoint question_ids and matching protocol |
| Consolidation mode ablation | Testing `end` vs `per-session` vs `off` consolidation. Shows how Elephant's consolidation affects different LongMemEval categories | Med | Knowledge-update and temporal-reasoning categories are most sensitive to consolidation quality. This is a natural ablation for the paper |
| Bank construction stats per instance | Counting facts, entities, observations per bank shows how the memory system processes each question's unique history | Low | LoCoMo has per-conversation bank stats. LongMemEval needs per-instance stats. With 500 instances, aggregate statistics (mean, p50, p95) are more useful than individual listings |
| Smoke profile for fast iteration | A profile that runs a small subset (e.g., 20 questions, 2-3 per category) for development and CI | Low | LoCoMo has `smoke`. Essential for development velocity given 500-question scale |
| Cross-benchmark comparison view | Comparing Elephant's LoCoMo and LongMemEval results side by side. Shows whether the system is generally strong or has benchmark-specific weaknesses | Med | Separate view tool as specified in PROJECT.md. Could share display logic but maintains independent artifact schemas |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Auto-download from HuggingFace | Adds HF SDK dependency for a one-time download. Manual download is simpler and the dataset rarely changes | Provide `wget`/`curl` instructions in README. Check for file presence at startup with clear error message |
| Custom history compilation | LongMemEval's paper describes an attribute-controlled pipeline for generating variable-length histories. The cleaned S and M datasets already have fixed histories | Use `longmemeval_s_cleaned.json` and `longmemeval_m_cleaned.json` directly. These are the standard evaluation datasets |
| MemoryChat optimizations | The paper proposes session decomposition, fact-augmented key expansion, and time-aware query expansion as their own improvements. Implementing these would benchmark MemoryChat, not Elephant | Benchmark Elephant's existing retain/recall/reflect pipeline. If Elephant needs temporal query expansion, that is a separate engine improvement, not a harness feature |
| Retrieval-only mode | The paper reports Recall@k and NDCG@k separately, but these require exposing raw retrieval rankings before the reader/reflect stage processes them. Building a separate retrieval-only evaluation path adds complexity for a secondary metric | Report retrieval provenance in the debug sidecar (which Elephant already does via reflect traces). Compute evidence session overlap as a derived metric from existing data, not a separate evaluation mode |
| Extending the LoCoMo view tool | Coupling the viewers creates maintenance burden and risks breaking LoCoMo's stable tooling | Build `longmemeval-view` as a separate binary. Share rendering utilities (table formatting, color output) via shared library code if appropriate |
| Modifying reflect for abstention | If Elephant's reflect agent can't handle false-premise questions naturally, fixing that is an engine change, not a harness feature. The harness should measure current behavior faithfully | Score all 500 questions including the 30 abstention questions. If abstention accuracy is low, that is a finding, not a bug in the harness |
| Per-question concurrency across instances | Running multiple question instances in parallel where each instance has its own bank. This creates contention on the database and embedding service without clear benefit | Use `--instance-jobs` for parallel bank construction during ingest (which is I/O bound), but run QA sequentially per instance. The parallelism that matters is within-instance session ingestion |
| Shared bank across questions | Tempting to group questions with overlapping histories into shared banks. This violates the benchmark protocol where each question has independent evaluation context | One bank per question. No exceptions. The dataset is designed this way intentionally |

## Feature Dependencies

```
Dataset loading → Per-instance bank creation → Ingestion (session-level)
Per-instance bank creation → Consolidation → QA evaluation
QA evaluation → Judge scoring → Per-category aggregation → Overall accuracy
QA evaluation → Debug sidecar (reflect traces, retrieved context)

run subcommand = ingest + QA in sequence
qa subcommand requires existing ingest artifact (bank_ids)
merge subcommand requires multiple compatible subset artifacts

Profile system → Controls ingest mode, consolidation mode, concurrency, question subset
Manifest/provenance → Requires profile, dataset fingerprint, prompt hashes, git commit

question_date passthrough → Required before temporal-reasoning questions make sense
Abstention scoring → Requires judge prompt that handles "correctly refused to answer"

Evidence session tracking → Requires answer_session_ids from dataset + source tracking in retrieval
Stage metrics → Already exists in LoCoMo infrastructure, wire through to LongMemEval
Bank stats → Already exists in LoCoMo infrastructure, aggregate across 500 instances
```

## MVP Recommendation

Prioritize these features for a first credible LongMemEval result:

1. **Dataset loading and per-instance bank isolation** (table stakes) -- without this nothing works
2. **run/ingest/qa subcommands** (table stakes) -- iteration speed is critical with 500 banks
3. **All 500 questions scored with GPT-4o judge** (table stakes) -- the headline number
4. **Per-category accuracy breakdown** (table stakes) -- the standard results table
5. **question_date passthrough** (table stakes) -- temporal reasoning depends on this
6. **Full manifest/provenance and three-artifact output** (table stakes) -- reproducibility contract
7. **Smoke profile** (differentiator) -- essential for development iteration
8. **Stage-level cost metrics** (differentiator) -- low-cost competitive advantage, already built for LoCoMo

Defer:
- **LongMemEval-M**: Defer until S results are competitive. M is extremely expensive and no competitor has published M results
- **Evidence session tracking / retrieval metrics**: Defer to second iteration. The reflect trace in the debug sidecar captures raw retrieval data; computing Recall@k against `answer_session_ids` can be added later without rerunning
- **Multi-judge variance analysis**: Defer until a baseline result exists. Run GPT-4o judge first, then rerun with alternatives
- **merge subcommand**: Defer if full runs can complete in one batch. Build when batch execution becomes necessary
- **Cross-benchmark comparison view**: Defer until both LoCoMo and LongMemEval have canonical results

## Competitive Landscape (Published LongMemEval-S Scores)

For context on what "competitive" means:

| System | Overall | Judge | Year | Source |
|--------|---------|-------|------|--------|
| Mastra Observational Memory (GPT-5-mini) | 94.9% | GPT-4o | 2026 | [Mastra Research](https://mastra.ai/research/observational-memory) |
| Backboard | 93.4% | GPT-4o-mini | 2026 | [GitHub](https://github.com/Backboard-io/Backboard-longmemEval-results) |
| Hindsight (Gemini 3 Pro) | 91.4% | GPT-4o | 2025 | [Vectorize PR](https://www.prnewswire.com/news-releases/vectorize-breaks-90-on-longmemeval-with-open-source-ai-agent-memory-system-302643146.html) |
| Hindsight (OSS-120B) | 89.0% | GPT-4o | 2025 | [hindsight-benchmarks](https://github.com/vectorize-io/hindsight-benchmarks) |
| Emergence (RAG) | 86.0% | GPT-4o | 2025 | [Emergence blog](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) |
| EverMemOS | 83.0% | GPT-4o | 2026 | [arXiv:2601.02163](https://arxiv.org/pdf/2601.02163.pdf) |
| Supermemory (GPT-4o) | 81.6% | GPT-4o | 2026 | [Supermemory Research](https://supermemory.ai/research) |
| TiMem (GPT-4o-mini) | 76.9% | GPT-4o | 2026 | LongMemEval benchmark topic |
| Zep/Graphiti (GPT-4o) | 71.2% | GPT-4o | 2025 | [Zep blog](https://blog.getzep.com/state-of-the-art-agent-memory/) |
| GPT-4o full-context baseline | 60.6% | GPT-4o | 2024 | [LongMemEval paper](https://arxiv.org/abs/2410.10813) |
| ChatGPT (interactive) | 57.7% | GPT-4o | 2024 | [LongMemEval paper](https://arxiv.org/abs/2410.10813) |

The competitive target is 85%+ to be credible, 90%+ to be noteworthy. Elephant's LoCoMo temporal-consolidation result (94.2% on conv-26) suggests the engine has the capability, but LongMemEval's per-instance isolation and harder question distribution (especially multi-session and temporal) will likely produce lower scores.

## Key Differences from LoCoMo

| Dimension | LoCoMo | LongMemEval |
|-----------|--------|-------------|
| Conversations | 10 shared | 500 per-instance |
| Questions | 1,540 (Cat 1-4) | 500 (7 types) |
| History per unit | 1 conversation, ~35 sessions | ~40-50 sessions (S), ~500 sessions (M) |
| Adversarial/Abstention | Cat 5 excluded | 30 false-premise questions scored |
| Evidence ground truth | Per-turn `evidence` refs | `has_answer` turn field + `answer_session_ids` |
| Retrieval metrics | Evidence recall (turn-level) | Recall@k, NDCG@k (session-level, optional) |
| Judge | Configurable (default Sonnet) | GPT-4o canonical, configurable for comparability |
| Bank reuse | QA reuses same bank across questions | Each question gets its own bank |
| Ingestion cost | 10 banks, amortized | 500 banks, dominant cost |
| Image handling | BLIP-2 captions inline | No images in dataset |

## Sources

- [LongMemEval paper (ICLR 2025)](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
- [LongMemEval project page](https://xiaowu0162.github.io/long-mem-eval/)
- [Backboard LongMemEval results](https://github.com/Backboard-io/Backboard-longmemEval-results)
- [Hindsight benchmarks](https://github.com/vectorize-io/hindsight-benchmarks)
- [Vectorize PR on 91.4%](https://www.prnewswire.com/news-releases/vectorize-breaks-90-on-longmemeval-with-open-source-ai-agent-memory-system-302643146.html)
- [EverMemOS paper](https://arxiv.org/pdf/2601.02163.pdf)
- [Mastra Observational Memory research](https://mastra.ai/research/observational-memory)
- [Supermemory research](https://supermemory.ai/research)
- [Emergence blog on LongMemEval](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag)
- [Zep blog on agent memory](https://blog.getzep.com/state-of-the-art-agent-memory/)

