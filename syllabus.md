# Retrieval-Augmented Generation — A Senior Engineer's Course

> **Learner profile.** A working software/ML engineer (3+ years shipping systems) targeting
> **Senior AI Engineer** and **Senior RAG Engineer** roles. Assumed: Python fluency, comfort with
> HTTP APIs and SQL, basic linear algebra (dot products, cosine similarity), and no prior
> retrieval or IR background.
> **Depth target.** As broad and deep as possible. Every module ends in a decision a senior
> engineer actually signs off on — an ADR, a schema, a budget, a threshold.
> **Frozen at.** Authoring baseline, 2025-10. Version-sensitive claims carry their version.

---

## 1. Scope knobs (frozen)

| Knob | Value | Notes |
|---|---|---|
| **Reference stack** | **LlamaIndex (Python framework)** | Single stack. Every API/feature claim is grounded against `developers.llamaindex.ai/python/framework/...`. No stack claim rests on memory. |
| **Backends** | **LiteLLM-routed providers** — OpenAI, Azure OpenAI, Anthropic, Gemini, Cohere, and self-hosted (Ollama/vLLM) | Labs target a LiteLLM proxy so model swap is a config change, not a code change. |
| **Hands-on labs** | ✅ ON | One runnable lab per module in `labs/NN-slug/`. |
| **Capstone** | ✅ ON | `capstone/` — one system carried from M1's brief to M22's production readiness review. |
| **Assessments** | ✅ ON | Per-module self-check + mid-term (after M11) + final (after M22) in `assessments/`. |
| **Tooling map** | ✅ ON | `maps/vector-db-map.md` — the vector-DB and tooling landscape, with sources and licenses. |
| **Production ops track** | ✅ ON | Part VII (M18–M22). Cost, latency, freshness, observability, security, scaling. |
| **Deferred (explicit)** | Multi-modal retrieval (image/audio embeddings) and agentic orchestration beyond M22 | Named here so they are *absent by decision*, not by omission. M22 and the map file flag them as extension points. |

### Version pin

Labs are written against **`llama-index-core` ≥ 0.12** on **Python ≥ 3.11**. Where an API is
young or in flux, the module says so in place rather than asserting a stable surface.

---

## 2. The running domain spine: **LogiTrack**

Every module's worked example is drawn from one invented company, so concepts compound instead
of restarting. LogiTrack is a mid-size freight-forwarding and customs-brokerage operator.

**The corpus** — 3,000 documents (~13.5M words, ~18.1M tokens) at pilot scale in M1.
`labs/generate_corpus.py` produces it deterministically; every figure derived from it in any
module is reproducible from that script:

| Source | Format | Why it is in the corpus |
|---|---|---|
| Carrier service agreements | PDF, ~40–120 pages, dense legal prose | Long, hierarchical, clause-numbered — forces structure-aware chunking (M3) |
| Customs procedure manuals | PDF + DOCX, per-jurisdiction | Jurisdiction/year metadata — forces filtered retrieval (M6, M9) |
| Tariff and rate tables | XLSX, HTML tables | Numeric, tabular — the classic chunking failure (M3) |
| Dangerous-goods (DG) rules | PDF with cross-references | Cross-document dependency — forces graph/hierarchical retrieval (M8, M10) |
| Support tickets + ops wiki | Markdown, short | Short, high-signal — the contrast case for chunk size (M3) |
| Service bulletins | Email/HTML, dated | Freshness-sensitive — forces incremental ingest (M14, M20) |

**The hard limit numbers** (invented, so no citation burden — they exist to force decisions):
P95 end-to-end latency ≤ **2.5 s**; monthly spend ≤ **$18,000**; known-item recall@10 ≥ **0.85**;
index freshness ≤ **15 min**; 8,500 queries/day by M18; 40 jurisdictions.

**The evaluation set** — invented alongside the corpus: 60 labelled queries frozen in M1,
grown to 400 by M13, and to 1,200 by M18. The count is a schedule, not a fixed size; every module
names the size it is working at. It is a *component of the spine*: M1 creates it, M13 makes it
the regression gate, M18 attaches it to CI, M22 reports on it in the readiness review.

> **Citation exemption (course-wide).** LogiTrack, its corpus, its numbers, its queries, and every
> worked example are **fiction** and carry no citation burden. Any *factual* claim embedded in
> them — an embedding dimension, a benchmark figure, a complexity class, a spec limit — is **not**
> exempt and must carry a source.

---

## 3. Reference stack grounding

LlamaIndex framework documentation is the sole authority for stack claims:

- Root: [LlamaIndex Framework](https://developers.llamaindex.ai/python/framework/) — `[LI]`
- RAG overview: [Introduction to RAG](https://developers.llamaindex.ai/python/framework/understanding/rag/) — `[LI]`
- Ingestion: [Loading](https://developers.llamaindex.ai/python/framework/understanding/rag/loading/), [Indexing](https://developers.llamaindex.ai/python/framework/understanding/rag/indexing/), [Storing](https://developers.llamaindex.ai/python/framework/understanding/rag/storing/), [Querying](https://developers.llamaindex.ai/python/framework/understanding/rag/querying/) — `[LI]`
- Component guides: [Ingestion Pipeline](https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/), [Node Parsers](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/), [Retriever](https://developers.llamaindex.ai/python/framework/module_guides/querying/retriever/), [Node Postprocessor](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/), [Response Synthesizer](https://developers.llamaindex.ai/python/framework/module_guides/querying/response_synthesizers/), [Evaluating](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/), [Routers](https://developers.llamaindex.ai/python/framework/module_guides/querying/router/), [Storing](https://developers.llamaindex.ai/python/framework/module_guides/storing/), [Observability](https://developers.llamaindex.ai/python/framework/module_guides/observability/) — `[LI]`

**Survey-by-weight rule.** Every module's payload prose carries a citation — a `[LI]` stack-doc
link for API/feature claims, or a `[key]` paper for conceptual/scientific claims. Only novel
sentences are exempt. Nothing else is.

**Explicitly out of scope for this course:** Java/TypeScript framework bindings, LLM
pre-training and fine-tuning (except where embedding-model fine-tuning is the retrieval lever,
M6/M15), and building a vector database engine from scratch (M5 analyzes the algorithms but does
not implement one beyond a teaching toy).

**Thinly documented, flagged in place:** LlamaIndex's per-parameter chunking defaults are
documented unevenly across parser classes; M3 records which claims are doc-backed and which are
observed behaviour with no doc page.

---

## 4. Module list

Three-state vocabulary — used in the table below and in every module's Verification note:

- `⬜ planned` — sketched in this syllabus; prose not written.
- `🟡 written` — README complete; **not** verified.
- `✅ verified` — **both** gates (Step 7 memory-less logic review, Step 8 literature survey) have
  run, with rounds and residual findings recorded in the module's Verification note.

**Written is not verified.**

| # | Module | Status | Verification gate state |
|---|---|---|---|
| M1 | Why RAG Exists and How to Think About It | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M2 | The RAG Pipeline End to End | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M3 | Corpus Acquisition, Parsing, and the Chunking Decision | 🟡 written | Step 7: **3 rounds run**, substantive findings fixed each time. Step 8: not run. |
| M4 | Embeddings and Vector Space | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M5 | Vector Databases and ANN Indexes | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M6 | Metadata, Filters, and Multi-Tenancy | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M7 | Sparse, Dense, and Hybrid Retrieval | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M8 | Index Structures Beyond Flat Vectors | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M9 | Ranking, Fusion, and Reranking | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M10 | Query Understanding and Query Transformation | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M11 | Context Assembly, Prompting, and the Temperature Dial | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M12 | Grounding, Citations, and Attribution | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M13 | Evaluation of Retrieval and Generation | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M14 | Ingestion at Scale, Incremental Sync, and Deletion | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M15 | Fine-Tuning Embeddings and Rerankers for a Domain | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M16 | Advanced Architectures: GraphRAG, Self-RAG, Agentic Retrieval | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M17 | Multi-Tenant, Permissioned, and Regulated RAG | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M18 | Cost Engineering and Latency Budgets | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M19 | Observability, Tracing, and Online Evaluation | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M20 | Failure Modes, Drift, and Continuous Improvement | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M21 | Security, Prompt Injection, and Adversarial Retrieval | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |
| M22 | Architecture Review and the Senior Engineer's Playbook | 🟡 written | Authoring checks only (anatomy, links, citation graph). Neither gate run. |

Cross-module progress marker read at the end of each README:
`M1 ✅ · M2 ✅ · … · M22 ⬜`

---

## 5. Module sketches

Each sketch is the lean brief the module is written from: **core question · content · domain beat · exercise**.

### Part I — Foundations

**M1 · Why RAG Exists and How to Think About It**
*Question:* Which problems is retrieval actually the right answer to?
*Content:* parametric vs non-parametric knowledge; the hallucination/attribution/freshness triad;
context-window scaling and why "just use a bigger window" does not dissolve retrieval; the
cost/latency/quality triangle; RAG vs fine-tuning vs long-context — the decision procedure.
*Domain:* LogiTrack's brief — 3,000 documents, 8,500 daily questions, $18k/month, 2.5 s P95. The
first 60 labelled queries are written here.
*Lab:* Build the smallest honest baseline — `VectorStoreIndex` over 20 documents, no tuning —
and record how it fails. The failures become the course's problem list.
*Exercise:* Write the ADR: RAG vs fine-tune vs long-context for LogiTrack, with the rejected
alternatives and the metric that would reverse the decision.

**M2 · The RAG Pipeline End to End**
*Question:* What are the stages, and what does each one cost when it is wrong?
*Content:* load → parse → chunk → embed → index → retrieve → rerank → assemble → generate →
attribute; the LlamaIndex object graph (`Document` → `Node` → `Index` → `Retriever` →
`NodePostprocessor` → `ResponseSynthesizer`) as the concrete skeleton of that pipeline; the
**8-failure-mode taxonomy** introduced and mapped onto stages; where evaluation attaches.
*Domain:* Walk one LogiTrack question ("What is the DG surcharge for lithium batteries on the
Rotterdam–Singapore lane?") through all ten stages, naming the failure each stage can inject.
*Lab:* Instrument the M1 baseline stage by stage; print the node set at each hop.
*Exercise:* Draw the pipeline for LogiTrack as a schema, annotate each stage with its failure
mode and its owner, and mark the three stages you would instrument first.

### Part II — Ingestion

**M3 · Corpus Acquisition, Parsing, and the Chunking Decision** — *gold standard module*
*Question:* How do you cut a document into retrievable units without destroying meaning?
*Content:* the chunking decision as a budget allocation; fixed-size token windows with overlap;
`SentenceSplitter`, `TokenTextSplitter`, `SentenceWindowNodeParser`, `HierarchicalNodeParser`,
`SemanticSplitterNodeParser`, `MarkdownNodeParser`, `CodeSplitter` — when each applies; chunk
size vs retrieval precision/recall vs answer completeness; overlap's cost; metadata that must
travel with each chunk; table and layout handling.
*Domain:* the carrier-agreement PDF and the rate-table XLSX — the same corpus, two chunkers, two
failure profiles measured against the M1 query set.
*Lab:* Chunk LogiTrack's carrier agreements five ways; measure recall@10 and answer completeness
for each.

**M4 · Embeddings and Vector Space**
*Question:* What is an embedding model actually optimizing, and how do you choose one?
*Content:* the bi-encoder architecture and contrastive training objective; cosine vs dot vs
Euclidean — and why normalizing makes them agree; the anisotropy / cone-effect problem;
dimensionality, Matryoshka truncation, and storage math; symmetric vs asymmetric retrieval and
instruction-prefixed models; MTEB and BEIR as selection instruments and their limits.
*Domain:* pick LogiTrack's embedding model against a $0.02/1M-token budget and 13.5M words.
*Lab:* Embed the same 200 LogiTrack chunks with three models; plot the similarity distribution
and find the near-duplicate threshold empirically.

**M5 · Vector Databases and ANN Indexes**
*Question:* How does approximate nearest-neighbour search actually work, and which engine earns its operational cost?
*Content:* exact search and its cost; HNSW (graph, `M`/`efConstruction`/`efSearch`); IVF and
IVF-PQ (clustering, product quantization, recall–memory tradeoff); DiskANN/Vamana; quantization
(scalar, binary, product); the recall/latency/memory triangle; filtered ANN and pre- vs
post-filtering; the engine landscape — Pinecone, Weaviate, Qdrant, Milvus, pgvector,
Elasticsearch/OpenSearch, Chroma, LanceDB, Vespa, Turbopuffer — and the operational axes that
actually decide it (managed vs self-hosted, license, hybrid support, filter expressiveness,
tenant isolation, cost model).
*Domain:* select LogiTrack's store under the $18k ceiling — and write down what would change the
answer.
*Lab:* Same 200k vectors into two engines; measure recall@10 vs p99 latency vs memory.
*Map:* `maps/vector-db-map.md` is owned here.

**M6 · Metadata, Filters, and Multi-Tenancy**
*Question:* How do you make retrieval respect scope — jurisdiction, date, client, clearance?
*Content:* metadata schema design; filter push-down vs post-filter and the cardinality trap;
pre-filtering with ANN (why it can destroy recall); hybrid metadata + semantic routing;
per-tenant isolation strategies (shared index + filter, namespace, collection, DB-per-tenant)
with their blast-radius and cost profiles.
*Domain:* 40 jurisdictions × 12 clients × document date — filter the DG corpus without leaking
across clients.
*Lab:* Add filters to the M5 store; measure the recall cost of a selective filter.
*Exercise:* Design the metadata schema and isolation model; state the leak scenario you are
defending against.

### Part III — Indexing and Retrieval

**M7 · Sparse, Dense, and Hybrid Retrieval**
*Question:* When does lexical search beat embeddings, and how do you combine them?
*Content:* BM25 (term frequency saturation, IDF, length normalization) and why exact tokens
(clause IDs, part numbers) defeat dense retrieval; SPLADE and learned sparse; fusion — RRF vs
weighted score fusion vs convex combination; the score-incomparability problem; tuning α.
*Domain:* "clause 14.2(b)" and "UN3480" queries that dense retrieval misses entirely.
*Lab:* BM25 vs dense vs hybrid on the M1 query set, split into lexical-only and semantic-only
subsets.

**M8 · Index Structures Beyond Flat Vectors**
*Question:* Which non-vector index earns its complexity for which question shape?
*Content:* `SummaryIndex`, `DocumentSummaryIndex`, `KeywordTableIndex`, `TreeIndex`,
`KnowledgeGraphIndex`, `PropertyGraphIndex`; metadata-replacement vs metadata-filter; recursive
retrieval and auto-merging as index-time structures; when a graph beats a flat index (multi-hop,
cross-document, aggregate queries) and its extraction cost; the `IndexStore` and persisted
`DocStore`/`IndexStore`/`VectorStore` split.
*Domain:* the DG rules' cross-references ("see §4.1 of the carrier agreement") — a multi-hop
question no flat index answers.
*Lab:* Build a `DocumentSummaryIndex` and a `PropertyGraphIndex` over the same corpus; compare
on the multi-hop subset.

**M9 · Ranking, Fusion, and Reranking**
*Question:* How do you reorder a candidate set so the top-k is actually the best-k?
*Content:* the retriever/reranker split (recall@k then precision@k); cross-encoders vs
late-interaction (ColBERT) vs LLM rerankers — quality/cost/latency; `SentenceTransformerRerank`,
`LLMRerank`, `CohereRerank`, `LongContextReorder`; the lost-in-the-middle position effect; MMR
for diversity; the candidate-set size decision.
*Domain:* 40 candidates → 6 for a 4,000-token answer budget.
*Lab:* Measure nDCG@10 with and without a reranker, and the latency you paid for it.

**M10 · Query Understanding and Query Transformation**
*Question:* How do you fix retrieval when the user's words are not the corpus's words?
*Content:* query rewriting, decomposition, and expansion; HyDE; multi-query and RAG-Fusion;
step-back prompting; routing (`RouterRetriever`, `LLMSingleSelector`, `PydanticSingleSelector`)
across indexes; intent/ambiguity detection and the clarify-vs-guess decision.
*Domain:* "can I ship that battery thing to Singapore?" — vague, colloquial, needs decomposition
and jurisdiction resolution.
*Lab:* Transform 50 LogiTrack queries by four techniques; measure retrieval delta and added cost.
*Exercise:* Choose which transformations ship; price each in tokens and latency against the 2.5 s
budget.

### Part IV — Generation

**M11 · Context Assembly, Prompting, and the Temperature Dial**
*Question:* Given the right chunks, how do you get the right answer — and what does temperature actually change?
*Content:* context ordering and the lost-in-the-middle effect applied at assembly time; packing
vs truncation; `ResponseSynthesizer` modes (`refine`, `compact`, `tree_summarize`,
`simple_summarize`, `accumulate`, `compact_accumulate`) and their N-call cost; system-prompt
design and instruction-following under retrieved evidence; **temperature**: what the sampling
distribution change does, why low temperature is not determinism, why some providers expose
top-p/seed and some do not, and the temperature × task matrix (factual extraction ~0, synthesis
~0.2–0.4, query rewriting ~0, diversification ~0.7+) with the evaluation protocol to re-derive
those numbers for your own system.
*Domain:* the DG surcharge answer — a numeric extraction that must be verbatim-correct, versus a
"summarize our exposure across 12 carriers" synthesis.
*Lab:* Sweep temperature {0, 0.2, 0.5, 0.7, 1.0} × 5 runs on the LogiTrack set; measure answer
variance, faithfulness, and cost. Re-derive the matrix rather than trusting it.

**M12 · Grounding, Citations, and Attribution**
*Question:* How does the system prove its answer came from the corpus?
*Content:* span-level vs document-level citation; faithfulness vs answer-relevance vs
context-relevance as separate metrics; the "answered from parametric memory" failure and how to
detect it; abstention and refusal as first-class outcomes; citation verifiability (can a human
check the claim against the cited span?) and citation precision/recall.
*Domain:* a customs ruling where a wrong citation is a compliance incident, not a bad UX.
*Lab:* Add citations with source-node provenance; measure citation precision by hand on 40 answers.

### Part V — Measurement

**M13 · Evaluation of Retrieval and Generation**
*Question:* How do you know it got better, and how do you keep it from getting worse?
*Content:* retrieval metrics (recall@k, precision@k, MRR, nDCG, MAP) and what each ignores;
generation metrics (faithfulness, answer relevance, context precision/recall); the
`FaithfulnessEvaluator` / `RelevancyEvaluator` / `RetrieverEvaluator` and `DatasetGenerator`;
LLM-as-judge and its biases (position, verbosity, self-preference) plus mitigation; synthetic
query generation and its trap (generating queries from the chunks you will retrieve); the golden
set's construction, stratification, and maintenance; regression gates.
*Domain:* grow the M1 60-query set into 400 stratified queries; make it the merge gate.
*Lab:* Build the eval harness; compute the M3–M12 systems' scores on one table.

### Part VI — Advanced and Scale

**M14 · Ingestion at Scale, Incremental Sync, and Deletion**
*Question:* How do you keep a 480k-document index fresh without rebuilding it?
*Content:* the `IngestionPipeline` and its cache (`IngestionCache`), docstore upsert semantics,
`ref_doc_id` and node relationships; dedup vs update vs insert; hard deletes and the right-to-be-
forgotten problem; backfills and re-embedding on model change (the full-corpus re-embed as a
budgeted project); parallel ingest, batching, and rate limits.
*Domain:* 3,000 → 480,000 documents over 12 months; a carrier agreement revision that must
supersede its predecessor within 15 minutes.
*Lab:* Incremental run vs full rebuild: measure wall-clock, token spend, and staleness window.

**M15 · Fine-Tuning Embeddings and Rerankers for a Domain**
*Question:* When is fine-tuning the retrieval model cheaper than better prompting or a better index?
*Content:* the domain-gap diagnosis; contrastive fine-tuning of a bi-encoder from (query, positive,
hard-negative) triples; hard-negative mining and its failure (false negatives); synthetic
query-document pair generation; reranker fine-tuning; catastrophic forgetting and the held-out
check; cost/benefit against re-chunking, hybrid search, and reranking — in that order.
*Domain:* LogiTrack's jargon-heavy DG vocabulary against a general-purpose embedding model.
*Lab:* Fine-tune a small bi-encoder on 2,000 mined pairs; measure recall@10 delta on a held-out set.

**M16 · Advanced Architectures: GraphRAG, Self-RAG, Agentic Retrieval**
*Question:* When does the advanced architecture pay for its complexity, and when is it a tax?
*Content:* iterative and self-correcting retrieval (Self-RAG, CRAG); GraphRAG and community
summarization for global/aggregate queries; adaptive and route-based retrieval; multi-hop and
decomposition pipelines; agentic retrieval loops (`FunctionAgent`/`AgentWorkflow`) with cost
ceilings; modular RAG as the organizing taxonomy; the honest comparison — every one of these
costs N× tokens and adds a failure surface.
*Domain:* "Which of our 12 carriers has the worst DG exposure in the EU?" — a global question no
top-k retrieval answers.
*Lab:* Build one advanced pipeline; measure quality gain per added dollar against the M13 harness.

### Part VII — Production

**M17 · Multi-Tenant, Permissioned, and Regulated RAG**
*Question:* How do you ship RAG where a permission bug is a breach and a wrong answer is a liability?
*Content:* ACL propagation into the index and onto every node; the post-retrieval authorization
filter as the non-negotiable backstop; document-level vs span-level permissions; audit logging
and answer provenance retention; data residency and PII redaction before embedding; GDPR/HIPAA-
flavoured deletion obligations; model-provider data-handling terms and zero-retention options.
*Domain:* 12 clients, 40 jurisdictions, and one shared index — with a red-team test for leakage.
*Lab:* Implement the authorisation backstop; write the test that proves cross-client leakage fails.

**M18 · Cost Engineering and Latency Budgets**
*Question:* Where do the dollars and the milliseconds actually go, and what do you cut first?
*Content:* a full unit-economic model (embedding at ingest, storage, query embedding, ANN search,
rerank, generation in/out tokens, eval); the latency budget as a waterfall with a per-stage
allocation summing to 2.5 s; caching layers (embedding cache, semantic response cache, prefix
cache on the generation prompt); batching and concurrency; model right-sizing per stage and
cascade routing (cheap model first, escalate on low confidence); the cost/quality frontier and
where you choose to sit on it.
*Domain:* $18,000/month against 8,500 queries/day — price every architecture in M7–M16 and cut
to fit.
*Lab:* Compute measured p50/p95 latency per stage; find the bottleneck; halve cost without losing
recall.
*Exercise:* The budget ADR — what you cut, what you refused to cut, and the SLO you renegotiated.

**M19 · Observability, Tracing, and Online Evaluation**
*Question:* What do you instrument so that a quality drop is diagnosable before a user reports it?
*Content:* spans for every pipeline stage; logging the retrieved node set with scores and lineage;
online signals (thumbs, regeneration rate, citation-click, abandonment, escalation-to-human) as
proxy labels; drift detection on query distribution and embedding distribution; A/B and
interleaving for retrieval changes; alert design on quality metrics rather than only on uptime;
the feedback loop back into the golden set.
*Domain:* a Tuesday-afternoon quality drop traced to a carrier-agreement revision ingested with
a broken parser.
*Lab:* Instrument the pipeline end to end; build the dashboard that would have caught it.

**M20 · Failure Modes, Drift, and Continuous Improvement**
*Question:* How do you run a RAG system as a system with a known failure budget?
*Content:* the 8 failure modes from M2 revisited with M3–M19's mitigations mapped onto each;
drift taxonomy (corpus drift, query drift, model drift beneath a pinned API); the failure budget
and incident classes; triage decision tree from symptom to stage; the improvement loop
(hypothesis → change → eval gate → canary → monitor); what to do when the vendor deprecates the
model under you.
*Domain:* a full incident post-mortem — the M19 drop, root-caused to parse → chunk → stale nodes.
*Lab:* Inject four planned failures; confirm your instrumentation localizes each to its stage.

**M21 · Security, Prompt Injection, and Adversarial Retrieval**
*Question:* How does an attacker use your own corpus against you, and what holds?
*Content:* indirect prompt injection via retrieved documents; the instruction/data boundary and
why it is porous; poisoned corpus and content-sanitization on ingest; exfiltration through
citations and markdown image rendering; the confused-deputy problem in agentic retrieval; tool
and query injection; defences graded by strength (output filtering, provenance constraints,
isolation, dual-LLM patterns) and why prompt-level defences alone fail; red-team methodology.
*Domain:* an attacker plants a bulletin in LogiTrack's corpus; a client tries to read another
client's contracts.
*Lab:* Plant three injections; measure the baseline attack success rate, then the defended rate.

**M22 · Architecture Review and the Senior Engineer's Playbook**
*Question:* What does a senior RAG engineer actually decide, and how do they defend it?
*Content:* synthesis of the tradeoff ledger across all 22 modules; the reference architecture
assembled from the course's decisions; build vs buy boundaries; the team and capability model;
the interview-shaped artifacts a senior RAG engineer is expected to produce (design doc,
eval plan, incident review, capacity model, cost model) and how to argue them under challenge;
the honest "what I would do differently" section; extension points deliberately left out
(multi-modal, fine-tuned generators, online learning).
*Domain:* the LogiTrack production-readiness review — the capstone oral.
*Exercise:* Defend the capstone architecture against three adversarial challenges from the
instructor's question bank.

---

## 6. Cross-cutting threads

### Thread A — Failure modes (the taxonomy)

Introduced in **M2**, mapped onto stages, and referenced by name in every module thereafter.
Every module's opening scene ends on the failure *it* prevents, taken from this set:

| ID | Failure mode | Owned by the stage | Module that mitigates it |
|---|---|---|---|
| **F1** | **Retrieval miss** — the answer existed and was never retrieved | retrieve | M3, M7, M8, M10 |
| **F2** | **Retrieval noise** — the right chunk was buried under plausible wrong ones | rank | M9, M10 |
| **F3** | **Chunk-boundary loss** — the answer was split across two chunks | chunk | M3, M8 |
| **F4** | **Representation mismatch** — user vocabulary and corpus vocabulary never met | embed | M4, M7, M15 |
| **F5** | **Context overflow** — evidence retrieved, then truncated or lost mid-prompt | assemble | M11, M18 |
| **F6** | **Unfaithful synthesis** — fluent answer not supported by the retrieved evidence | generate | M11, M12, M16 |
| **F7** | **Staleness** — index and corpus diverged | ingest | M14, M20 |
| **F8** | **Budget blowout** — cost or latency exceeded | every | M18, M19 |

M21 adds a ninth, adversarial class (**F9 · Poisoned context**) and M17 adds **F10 · Scope leak**
— both flagged as extensions of the taxonomy rather than retrofits.

### Thread B — Tradeoff ledger

Every module contributes entries in a fixed shape:

> **Entry M#-n.** *Decision.* Costs X. Accepted because Y. Reverses if Z.

The ledger is the capstone's decision record; M22 reads it back as the architecture.

### Thread C — LlamaIndex at a glance

One verified primitive per module, in a fixed block, each traceable to a doc page:

| Module | Primitive |
|---|---|
| M1 | `VectorStoreIndex`, `SimpleDirectoryReader` |
| M2 | `Document` → `Node` → `Index` → `Retriever` → `NodePostprocessor` → `ResponseSynthesizer` |
| M3 | `IngestionPipeline`, `SentenceSplitter`, `SentenceWindowNodeParser`, `HierarchicalNodeParser` |
| M4 | `BaseEmbedding`, `HuggingFaceEmbedding`, `OpenAIEmbedding` |
| M5 | `BasePydanticVectorStore`, `VectorStoreIndex.from_vector_store` |
| M6 | `MetadataFilters`, `MetadataFilter`, `VectorStoreIndex` filters |
| M7 | `BM25Retriever`, `QueryFusionRetriever` |
| M8 | `DocumentSummaryIndex`, `KnowledgeGraphIndex`, `PropertyGraphIndex`, `RecursiveRetriever`, `AutoMergingRetriever` |
| M9 | `BaseNodePostprocessor`, `SentenceTransformerRerank`, `LLMRerank`, `LongContextReorder`; MMR as a **vector-store query mode** (`vector_store_query_mode="mmr"`, `vector_store_kwargs={"mmr_threshold": …}`) — **not** an `MMRPostprocessor`, which is not a documented class |
| M10 | `HyDEQueryTransform`, `MultiStepQueryEngine`, `RouterRetriever`, `SubQuestionQueryEngine` |
| M11 | `get_response_synthesizer` (`refine`/`compact`/`tree_summarize`), `Settings.llm`, `Settings.context_window` |
| M12 | `Response.source_nodes`, `NodeWithScore`, `CitationQueryEngine` |
| M13 | `RetrieverEvaluator`, `FaithfulnessEvaluator`, `RelevancyEvaluator`, `DatasetGenerator` |
| M14 | `IngestionCache`, `docstore`, `ref_doc_id`, `upsert` |
| M15 | `SentenceSplitter` + fine-tuned `BaseEmbedding`; `SentenceTransformerRerank` fine-tune |
| M16 | `SubQuestionQueryEngine`, `PropertyGraphIndex`, `FunctionAgent`, `AgentWorkflow` |
| M17 | `MetadataFilters` as the authorisation backstop; `IngestionPipeline` transformations |
| M18 | `Settings.callback_manager`, token counting, `IngestionPipeline` cache |
| M19 | `CallbackManager`, `LlamaDebugHandler`, instrumentation/span handlers |
| M20 | `IngestionPipeline` docstore lineage, `RefDocInfo` |
| M21 | `IngestionPipeline` sanitization node, output filters |
| M22 | Assembled reference architecture; ledger read-back |

### Reserved keys — and a caution about them

**Two further key corrections**, caught by the M12–M15 author and verified:

- `[TREC23]` was assigned to "the TREC RAG track." **There is no TREC 2023 RAG track** — the
  track's first edition was TREC 2024, published as *Ragnarök* (Pradeep, Thakur,
  Sharifymoghaddam, Zhang, Nguyen, Campos, Craswell & Lin,
  [arXiv:2406.16828](https://arxiv.org/abs/2406.16828)), with a support-assessment study at
  [arXiv:2504.15205](https://arxiv.org/abs/2504.15205). The comprehensive RAG benchmark is a
  *different* artifact, CRAG ([arXiv:2406.04744](https://arxiv.org/abs/2406.04744), NeurIPS 2024
  Datasets & Benchmarks). M13 therefore leaves `[TREC23]` unassigned, cites the real artifacts as
  `[RAGNAROK24]` and `[TRECSUP25]`, and records the mis-assignment in its unsupported note.
- `[SELFJUDGE23]` was assigned a 2023 year. The self-preference-bias paper is Wataoka, Takahashi &
  Ri, [arXiv:2410.21819](https://arxiv.org/abs/2410.21819) — **2024**, not 2023. M13 attaches the
  key to the verified paper and flags the year mismatch rather than restating the wrong year.

The ownership map fixes which module *surveys* a claim, not which arXiv ID owns it. **Two IDs
assigned to reserved keys during blueprint authoring turned out to belong to entirely different
papers** when fetched:

- `[LONGFORM23]` was planned as arXiv:2308.00479 for "long-form RAG response synthesis." That ID
  is actually *"Retrieval Augmented Generation and Representative Vector Summarization for large
  unstructured textual data in Medical Education"* (Manathunga & Illangasekara, 2023), which owns
  a narrower claim. The real owners of M11's `refine` and tree-summarization behaviour are
  [Shao et al., arXiv:2305.15294](https://arxiv.org/abs/2305.15294) (iterative
  retrieval–generation synergy) and [Wu et al., arXiv:2109.10862](https://arxiv.org/abs/2109.10862)
  (recursive summarization).
- `[MULTIHOP20]` was planned as arXiv:2011.01060 for "multi-hop decomposition query engines." That
  ID is actually the *2WikiMultiHopQA dataset* paper (Ho, Nguyen, Sugawara & Aizawa, COLING 2020).
  The real owner of multi-hop *dense retrieval* is
  [Xiong et al., arXiv:2009.12756](https://arxiv.org/abs/2009.12756).

Neither key appears in a written module — M8/M9 caught both during authoring and cited the true
owners instead, flagging the mismatch in place. The lesson is the course's own:
**a plausible-looking identifier is not a verified one.** No module may cite an arXiv ID that the
module's author has not fetched and read the title, author list, and year off. This is the
failure mode `lit-survey` calls the near-miss identifier, and it is the reason Step 8 exists.

Prevents the same paper being surveyed five times. Each recurring claim is owned and surveyed
**once**, in its owning module; later modules cite it by the same key and do not re-survey it.

| Recurring claim | Owning module | Later modules that cite the key |
|---|---|---|
| Original RAG formulation; retriever-generator split | **M1** | M2, M11, M16, M22 |
| Dense retrieval / DPR; bi-encoder training | **M4** | M7, M15 |
| BEIR zero-shot IR benchmark | **M4** | M13 |
| ANN algorithms (HNSW) | **M5** | M9, M18 |
| Product quantization / IVF | **M5** | M18 |
| BM25; sparse retrieval and learned sparse (SPLADE) | **M7** | M9, M18 |
| Reciprocal rank fusion | **M7** | M9 |
| Cross-encoder reranking; late interaction (ColBERT) | **M9** | M13, M18 |
| Iterative retrieval–generation synergy (`refine`) | **M11** | M16 |
| Recursive/tree summarization | **M11** | M13 |
| Multi-hop dense retrieval; decomposition over evidence paths | **M8** | M10, M16 |
| Lost in the middle (position effect) | **M9** | M11, F5 |
| MMR: query mode vs postprocessor, and its score rescaling | **M9** | M11, M13 |
| Lost in the middle — assembly-side application | **M11** | (cites M9; does not re-survey) |
| Query rewriting/decomposition, HyDE | **M10** | M16 |
| RAG evaluation frameworks; LLM-as-judge; RAGAS | **M13** | M16, M19, M20 |
| Self-RAG, CRAG, Self-correcting retrieval | **M16** | M20 |
| GraphRAG / community summarization | **M16** | M8 (cross-cite), M22 |
| Modular RAG taxonomy | **M16** | M22 |
| Long-context vs RAG | **M1** | M18, M22 |
| Prompt injection; indirect injection | **M21** | M17, M22 |
| Fine-tuning retrievers; hard negatives | **M15** | M4, M9 |
| MRL / Matryoshka representation truncation | **M4** | M18 |
| MTEB leaderboard as a selection instrument | **M4** | M13 |
| BRIGHT: reasoning-intensive retrieval; leaderboard-vs-task gap | **M4** | M7, M16 |
| Dimension-bounded single-vector expressivity | **M4** | M7, M16 |
| Ingestion-time text-to-vector inversion risk | **M4** | M17 |
| Filtered-ANN algorithms (ACORN, window-filtered search) | **M6** | M5, M17 |
| Cardinality trap in filtered retrieval | **M6** | M17 |
| Multi-tenant isolation models and cost multipliers | **M6** | M5, M17 |
| Per-engine licences, tenancy and quantization capabilities | **M5** | M6, M17 |
| Learned sparse expansion (SPLADE family) | **M7** | M9 |
| Score incomparability across retrievers | **M7** | M9 |
| BM25Retriever internals; the lexical path's filter contract | **M7** | M6 |

---

## 7. Repo structure

```
RAG/
├── syllabus.md                 ← this file; the blueprint and the status table
├── README.md                   ← course index and how to work through it
├── modules/
│   └── NN-slug/
│       ├── README.md           ← the module (fixed anatomy)
│       ├── lab/                ← runnable lab for this module
│       └── …
├── labs/                       ← shared lab harness (corpus generator, eval runner)
├── workshops/                  ← optional, short focused build sessions
├── assessments/
│   ├── midterm.md              ← after M11
│   └── final.md                ← after M22
├── discussions/                ← durable Q&A and worked examples raised during authoring
├── maps/
│   └── vector-db-map.md        ← vector DB + tooling landscape (M5 owns)
├── capstone/                   ← the end-to-end system spec, built across the course
└── tmp/                        ← Step 7 verifier scratch dumps (deleted after each pass)
```

---

## 8. Module anatomy (fixed, repeated in every module)

Every `modules/NN-slug/README.md` is exactly this skeleton, same blocks, same order:

```markdown
# M# · <Topic>

> **Module question:** …
> **Cross-cutting threads:** Failure modes · Tradeoff ledger · LlamaIndex at a glance
> **Domain spine:** LogiTrack — <the beat>
> **Prerequisites:** <module links>

---

## Opening scene — <narrative ending on the failure this module prevents>
> **Failure mode (the module in one line):** …

## <Payload sections — one concept per section, prose + table + mermaid>
> **LlamaIndex at a glance:** …
> **Tradeoff (the ledger entry):** …

## Worked example: <LogiTrack beat>

## Design exercise
## Where the literature disagrees
## Sources
**Verification note:** <status line + Step 7 rounds + residual findings + Step 8 status>
**Next module:** [M## — <title>](../NN-slug/README.md) — <one line>
```

The Verification note is the one block that must **not** be a single summary line. Step 7 requires
the round count and the residual findings actually accepted, and Step 8 requires the bibliography
status — so the note is a short block carrying all three. A module whose note merely says
"verified" cannot be marked `✅`, because the note is the evidence that the gates ran.

Gold standard = **M3** (first claim-dense module). It fixes the key format, entry format, and
section order; every later module mirrors it.

---

## 9. Verification protocol

Per module, in order, before it may be marked `✅ verified`:

1. **Step 7 — memory-less logic review.** Dump the claim-bearing README to `tmp/<slug>.md`,
   dispatch a fresh subagent with the fixed Verifier prompt (it sees only that file), fix the
   artifact, re-dispatch the **byte-identical** prompt. Expect decay: substantive first, cosmetic
   later. Stop when a round is cosmetic-only, or at the cap (**3** rounds prose-only, **5–8**
   for modules carrying derivations, budgets, or thresholds). Record rounds + residuals.
2. **Step 8 — literature survey.** Run the `lit-survey` skill on the module. Zero-step
   derivations are exempt: opening scenes, LogiTrack fiction, and design exercises carry no
   citation burden. Respect the ownership map in §6 — cite, do not re-survey.
3. **Step 9 — fold refinements.** Any edit touching verified content re-opens 1 and 2 for that
   module (or returns it to `🟡 written`).

## 10. Completion criteria

The course is complete when the status table in §4 is **true** — every `✅` backed by a recorded
two-gate Verification note and a cosmetic-only (or capped) final round; every module carries the
full anatomy and ends in a decision-forcing exercise plus a next-module link; the citation graph
in every module is one-to-one; every stack claim traces to a LlamaIndex doc page; every
cross-reference resolves; and no file is orphaned.
