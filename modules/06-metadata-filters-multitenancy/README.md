# M6 · Metadata, Filters, and Multi-Tenancy

> **Module question:** How do you make retrieval respect scope — jurisdiction, date, client, clearance?
> **Cross-cutting threads:** Failure modes · Tradeoff ledger · LlamaIndex at a glance
> **Domain spine:** LogiTrack — 40 jurisdictions × 12 clients × document date, filtering the dangerous-goods corpus without leaking across clients
> **Prerequisites:** [M3 — Corpus Acquisition, Parsing, and the Chunking Decision](../03-parsing-and-chunking/README.md) · [M4 — Embeddings and Vector Space](../04-embeddings-and-vector-space/README.md) · [M5 — Vector Databases and ANN Indexes](../05-vector-databases-and-ann/README.md)

---

## Opening scene — the filter that made retrieval worse

The M5 migration left LogiTrack with a working index and one embarrassing query class. A Rotterdam
customs broker asks about *"the DG packaging requirement for lithium cells"* and gets the Singapore
variant, which differs in a way that matters. The fix looks trivial and correct: every chunk already
carries `jurisdiction` in its metadata, so the retriever adds a filter.

```python
retriever = index.as_retriever(
    filters=MetadataFilters(
        filters=[MetadataFilter(key="jurisdiction", operator=FilterOperator.EQ, value="NL")]
    )
)
```

Recall@10 on the frozen 60-query set drops from **0.97 to 0.44** — the largest single regression any
change in the pilot has produced, and the change was *meant* to improve correctness. The wrong
answers are gone. So are most of the right ones.

The team's first instinct is to loosen the filter. That trades correctness back for recall and
returns to the original bug. Their second instinct — subtract the filter and filter the results in
application code — is the same thing with more code and a worse latency profile. The actual
mechanism is simpler and more uncomfortable: the retriever asked the index for the 10 nearest
vectors **anywhere**, then discarded the 9 that were not Dutch. The filter was applied to a candidate
set that had never been told about it.

Then someone notices the second problem, while reading the code that caused the first. `filters` is
an optional argument with a default of none, and the request path passes
`filters=MetadataFilters(filters=inferred_filters)` where `inferred_filters` comes from a model that
was asked to extract *relevance* fields from the query. Nothing in the schema requires `client`.
One request shape away from answering a Nordwind question with Baltic Freight's contract — the
incident **M17** exists to prevent, and its opening scene. This module builds the mechanism that
incident happens *through*; M17 builds the backstop that catches it anyway.

> **Failure mode (the module in one line):** treating metadata as **documentation attached to a
> chunk** rather than as a **retrieval constraint with a cardinality** — so a filter that is correct
> in intent silently destroys recall (**F1 · retrieval miss**), and a filter that is merely optional
> silently destroys scope (the **F10 · scope leak** M17 owns and formalises).

---

## Metadata is a retrieval constraint, not documentation

Every node carries metadata because M3's node parsers inherit the parent document's attributes onto
each child node — that inheritance rule is why document-level facts survive chunking at all [[LI-NP]](#src-li-np).
M6 is the module that decides *which* facts to carry, because each one is simultaneously three
things: a label, a filter axis, and a cost.

| Kind | Examples at LogiTrack | Decides | Failure if absent |
|---|---|---|---|
| **Provenance** | `doc_id`, `file_name`, `page_label`, `clause_id`, `ref_doc_id` | Whether a citation can be checked (→ M12) and whether a delete can be scoped (→ M14) | Unattributable answers; unfindable nodes |
| **Scope** | `client_id`, `jurisdiction`, `confidentiality` | *Whether this node may be returned at all* | **Scope leak** — the only metadata failure that is a legal event (→ M17) |
| **Freshness** | `effective_date`, `superseded_by`, `ingested_at` | Whether stale text can be excluded (→ M14, M20) | Confident answers from superseded clauses |
| **Structural** | `heading_path`, `chunk_index`, `node_level` | Which expansion or neighbour strategy applies (→ M3, M8) | Lost context on merge/expand |
| **Enriched** | `section_summary`, `excerpt_keywords`, `questions_this_excerpt_can_answer`, `entities` | Retrieval *reach* — added vocabulary that the chunk text lacks | Vocabulary mismatch (**F4**, → M4, M15) |

The last row is the one teams under-exploit, and LlamaIndex makes it cheap. `TitleExtractor`,
`QuestionsAnsweredExtractor`, `SummaryExtractor`, `KeywordExtractor`, and `EntityExtractor` run as
transformations in an `IngestionPipeline` and attach generated fields to each node — the documented
example emits `document_title`, `questions_this_excerpt_can_answer`, `prev_section_summary`,
`section_summary`, and `excerpt_keywords` alongside the ingested file's own fields [[LI-ME]](#src-li-me). Those
fields are *retrievable text*: a question-shaped field lets a natural-language query match a chunk
whose prose never posed the question. And they are LLM outputs, so they are ingest-time token spend
and a new failure surface — a wrong summary is now a retrieval signal (→ M14 for the cache and
lineage discipline).

Because the enriched fields are generable, they raise the module's first real decision: **enrich at
ingest, or filter at query time?** Filtering is cheap, exact, and only as good as the schema you
committed to months ago. Enrichment is expensive at ingest but can be regenerated, re-prompted, and
versioned. Both are metadata; only one of them is free to change later.

> **Tradeoff (the ledger entry M6-1).** *Decision:* treat enriched (LLM-generated) metadata as
> retrieval reach and scope fields as immutable constraints, and never let one type substitute for
> the other. *Costs:* two kinds of field with different change-rates in one schema, and an extra
> ingest pass with its own token bill. *Accepted because:* scope fields must be exact, cheap, and
> auditable, while reach fields should be regenerable when the extraction prompt improves — mixing
> them means either regenerating your authorization metadata or freezing your retrieval vocabulary.
> *Reverses if:* the corpus is small enough that enriched fields can be recomputed on every ingest
> anyway, at which point the distinction costs more than it saves.

> **Multimodal inputs** ($\color{red}\text{to be discussed}$) — mixed-content pages add fields, and each one lands in a row of the table above.
> - **Structural:** a `content_type` per node (prose / table / figure / OCR). [Docling](https://docling-project.github.io/docling/concepts/docling_document/) already types items as texts, tables, pictures.
> - **Provenance:** page + bounding box, so a citation can point at a table cell or stamp. Docling keeps boxes only "if available". Parsing itself → [M3](../03-parsing-and-chunking/README.md).
> - **Enriched:** a vision-model figure description is an LLM output. Treat it as a reach field (regenerable, M6-1), never as scope.
> - **Bookkeeping** (e.g. OCR confidence): keep it out of embed and LLM text via [`excluded_embed_metadata_keys` / `excluded_llm_metadata_keys`](https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_documents/).

---

## The filter as a boolean expression, and how it reaches the store

LlamaIndex's filter model is a typed boolean expression: `MetadataFilter(key, operator, value)`
objects collected into `MetadataFilters(filters=[...], condition=FilterCondition.AND | OR)`, using
`FilterOperator` for the comparison, passed to `index.as_retriever(filters=...)` or to the query
engine [[LI-MF]](#src-li-mf). The framework does **not** evaluate that expression itself. It translates it into the
store's native predicate and pushes it down — the documented Qdrant integration prints the
translation, turning

```python
MetadataFilters(filters=[
    MetadataFilter(key="theme", value="Fiction"),
    MetadataFilter(key="year", value=1997, operator=FilterOperator.GT),
], condition=FilterCondition.AND)
```

into native `FieldCondition(key='theme', match=MatchValue(value='Fiction'))` and
`FieldCondition(key='year', range=Range(gt=1997.0))` objects, with the second call's output showing
both conditions pushed into one request [[LI-MF]](#src-li-mf). That is the whole architecture in one log line:
**the filter is executed inside the index, which means the index's opinion about filters becomes your
recall.**

> **LlamaIndex at a glance:** the filter surface for this module is `MetadataFilter` +
> `MetadataFilters` + `FilterOperator` + `FilterCondition` from `llama_index.core.vector_stores`,
> attached through `index.as_retriever(filters=..., similarity_top_k=...)` [[LI-MF]](#src-li-mf). Two properties of
> that surface decide everything in this module. First, **the framework translates and pushes down**
> rather than filtering in Python — the documented translation emits the backend's own predicate
> objects, so the semantics of `FilterCondition.OR`, range operators, and list membership are the
> backend's semantics, not LlamaIndex's [[LI-MF]](#src-li-mf). Second, **`filters` is optional**, so the correctness
> of your scope depends on your request path, not on the framework; the framework's own metadata-filter
> guides are written per backend, and a docs search returns separate pages for Qdrant, Milvus,
> Weaviate, Pinecone, and Neo4j, each showing the same `MetadataFilters` object against a different
> native predicate language [[LI-MF]](#src-li-mf)[[LI-VS]](#src-li-vs).

Which stores can do that is tabulated by the framework itself: metadata filtering is marked for
Chroma, Elasticsearch, OpenSearch, Pinecone, Qdrant, Milvus/Zilliz, Postgres/pgvector, LanceDB, and
more, while the default in-memory `SimpleVectorStore` and FAISS are not marked at all [[LI-VS]](#src-li-vs). The
practical consequence is that the same `MetadataFilters` object means three different things
depending on the backend:

| Store behaviour | Mechanism | Recall effect | Where the language says so |
|---|---|---|---|
| **Filter-aware traversal** | The graph walk respects the predicate | Recall preserved by construction | ACORN's predicate subgraph traversal, surveyed in M5 [[ACORN24]](#src-acorn24); Weaviate's ACORN default since v1.34, described as helping most *"when the filter has low correlation with the query vector"* [[WEAVIATE-FILTER]](#src-weaviate-filter) |
| **Pre-filter + exact search** | Matching set computed first, then scanned | Recall exact; cost degrades to brute force | Qdrant's planner falls back to full scan below `full_scan_threshold` [[QDRANT-INDEX]](#src-qdrant-index); Elasticsearch: exact brute force is best for *"pre-filtered subsets"* [[ES-KNN]](#src-es-knn) |
| **Post-filter** | Search unfiltered, discard after | **Collapses with selectivity** | pgvector is explicit: *"With approximate indexes, filtering is applied after the index is scanned"* [[PGVECTOR]](#src-pgvector) |

The pgvector sentence is worth reading twice, because it contains the arithmetic that produced the
opening scene: *"If a condition matches 10% of rows, with HNSW and the default `hnsw.ef_search` of
40, only 4 rows will match on average"* [[PGVECTOR]](#src-pgvector). That is the entire failure in one line — an
ef-search of 40 with a 10%-selective filter yields single-digit matches, and pgvector's remedy is
iterative index scans, added in 0.8.0, which *"automatically scan more of the index until enough
results are found (or it reaches `hnsw.max_scan_tuples`)"* [[PGVECTOR]](#src-pgvector). In other words: the engine can
fix this, but only if you asked it to, and the knob is a scan budget rather than a filter.

Milvus and Qdrant both expose the same trade as explicit retrieval-time parameters rather than
automatics — Milvus via boolean filter expressions evaluated alongside an ANN search [[MILVUS-INDEX]](#src-milvus-index),
Qdrant via payload indexes that make filters cheap *and* a planner threshold that decides when to
stop approximating [[QDRANT-INDEX]](#src-qdrant-index)[[QDRANT-FILTER]](#src-qdrant-filter).

> **Tradeoff (the ledger entry M6-2).** *Decision:* push every scope filter into the store and keep
> the store's filter-aware or pre-filter path enabled; accept post-filtering only for
> high-selectivity relevance filters. *Costs:* engine lock-in on the filter-strategy axis (the same
> schema behaves differently per backend), and the operational burden of keeping payload/scalar
> indexes in sync with the schema. *Accepted because:* post-filtering a selective filter does not
> degrade retrieval gracefully — it fails arithmetically, and the failure is invisible in every
> latency dashboard. *Reverses if:* the store's own planner already guarantees overflow behaviour
> (pgvector's iterative scans [[PGVECTOR]](#src-pgvector), Qdrant's full-scan fallback [[QDRANT-INDEX]](#src-qdrant-index)), in which case
> an explicit strategy is redundant.

---

## Selectivity is the variable that decides everything

**Terminology first, because the sources disagree about it and the disagreement is a trap.**
Qdrant's documentation writes *"In the case of high-selectivity (weak) filters, you can use the HNSW
index as it is. In the case of low-selectivity (strict) filters, you can use the payload index and do
a complete rescore"* [[QDRANT-INDEX]](#src-qdrant-index) — i.e. "selective" describes *how much the filter selects*. The
filtered-ANN literature parameterises the same quantity as *"query selectivities"* without that
inversion [[FANNS25]](#src-fanns25). **In this module, selectivity always means the fraction of the corpus the
predicate admits**: a *strict* filter admits 0.1% of chunks, a *weak* filter admits 50%.

Two selectivities matter, and conflating them is the error that ships:

- **Corpus fill $p_c$** — the fraction of the whole index the filter admits. Easy to compute, and the
  number engineers quote. $p_c$ for `jurisdiction = NL AND client = Nordwind` at LogiTrack is roughly
  $1/40 \times 1/12 \approx 0.2\%$.
- **Neighbourhood fill $p_n$** — the fraction of the *true nearest neighbours* of a given query that
  the filter admits. This is the number that determines what post-filtering actually does, and it is
  never quoted because it is not a property of the filter alone: it is a property of the filter's
  **correlation with the embedding**.

The distinction is the whole module. A jurisdiction filter is *semantically correlated* with a
jurisdiction-specific query: someone asking about Rotterdam DG packaging is asking a question whose
nearest neighbours are mostly Dutch documents, so $p_n \gg p_c$ and post-filtering loses little. A
`client_id` filter is *semantically uncorrelated*: two clients' Rotterdam clauses are near-identical
in embedding space, so $p_n \approx p_c$, and post-filtering a top-10 retrieves, on average,
$10 \times 0.0021 = 0.02$ usable nodes. That is not a degradation; it is an empty result set with a
latency budget. The filtered-ANN literature names the same variable — Weaviate's rationale for
adopting ACORN is that it helps most *"when the filter has low correlation with the query vector"*
[[WEAVIATE-FILTER]](#src-weaviate-filter), and ACORN's contribution is described as predicate-agnostic traversal for
predicate sets that break simpler methods [[ACORN24]](#src-acorn24).

The general arithmetic, derived in-file: to fill a top-$k$ with $k$ matching nodes after
post-filtering, you must over-fetch

$$k_{\text{fetch}} \approx \frac{k}{p_n} \quad (\text{expected value; the variance is what hurts})$$

so a top-10 query under a client filter with $p_n = 0.002$ needs $k_{\text{fetch}} \approx 5{,}000$
candidates, at which point you are doing an exact-ish scan through the ANN's candidate list and
should have pre-filtered instead. The crossover — where over-fetching costs more than a pre-filter's
brute-force scan — is exactly what Qdrant's `full_scan_threshold` encodes [[QDRANT-INDEX]](#src-qdrant-index) and what
pgvector's `max_scan_tuples` bounds [[PGVECTOR]](#src-pgvector).

| Filter | $p_c$ (corpus) | Query correlation | Expected $p_n$ | Post-filter top-10 yield | Verdict |
|---|---|---|---|---|---|
| `jurisdiction = NL` | 2.5% | **High** (NL queries ask NL questions) | ~0.35 | ~3.5 usable nodes | Post-filter survivable; still lossy |
| `client = Nordwind` | 8.3% | **Low** (clauses are near-identical across clients) | ~0.08 | ~0.8 usable nodes | Must be pushed down |
| `client AND jurisdiction` | 0.2% | Mixed | ~0.03 | ~0.3 usable nodes | Must be pushed down *and* filter-aware |
| `+ effective_date range` (§M14) | 0.05% | **High** for date-sensitive queries, **low** otherwise | 0.01–0.4 | 0.1–4 usable nodes | Unpredictable — measure per axis |

Three independent lines of work exist because this table is real. The filtered-ANN benchmark
introduces *"the arxiv-for-fanns dataset of transformer-based embedding vectors for the abstracts of
over 2.7 million arXiv papers, enriched with 11 real-world attributes such as authors and categories"*
and benchmarks *"eleven different FANNS methods ... across different filter types, numbers of
retrieved neighbors, dataset scales, and query selectivities"*, precisely because publicly available
datasets with *"abundant real-world attributes"* were missing [[FANNS25]](#src-fanns25). Window filters solve the
numeric-range case as a first-class problem — *"approximate nearest neighbor search where each point
in the dataset has a numeric label, and the goal is to find nearest neighbors to queries within
arbitrary label ranges"* — reporting *"up to a 75× speedup over existing solutions at the same level
of recall"* [[WINDOW24]](#src-window24). And PathFinder attacks the multi-attribute conjunction/disjunction case with
a *"cost-based optimizer"* and an *"index borrowing optimization"*, reporting *"up to 9.8x in query
throughput at recall 0.95"* over the best baseline [[PATHF25]](#src-pathf25); UNIFY attacks the range case with a
single index that supports pre-, post-, and hybrid filtering so that *"index consistency and
maintenance issues"* do not multiply [[UNIFY24]](#src-unify24).

> **Tradeoff (the ledger entry M6-3).** *Decision:* classify every filter axis as scope or relevance,
> measure $p_n$ per axis on the evaluation set, and require filter-aware or pre-filter execution for
> every axis below a $p_n$ threshold. *Costs:* an evaluation-set slice per filter axis, and possibly
> a more expensive index family [[ACORN24]](#src-acorn24)[[PATHF25]](#src-pathf25). *Accepted because:* post-filter failure is silent
> and arithmetic, and $p_n$ cannot be inferred from $p_c$ — the correlation is a property of the
> embedding, so it changes when the embedding model changes (→ M4). *Reverses if:* a measured $p_n$
> comes back high for every production axis, in which case the correlation is doing the work and the
> cheaper path is correct.

---

## Isolation profiles: multi-tenancy as a cost decision

Twelve clients share one corpus. Whether they share one *index* is the module's second decision, and
it is a spectrum with different blast radii rather than a boolean.

| Profile | Mechanism at LogiTrack | Blast radius of one bug | Cost multiplier | Documented position |
|---|---|---|---|---|
| **Shared index + mandatory filter** | `client_id` on every node, filter required by the request contract | **All clients** — one missing clause is a cross-client read | 1× | Qdrant: partition by payload field with `is_tenant: true`, *"efficient for a large number of small, similarly-sized tenants"* [[QDRANT-MT]](#src-qdrant-mt) |
| **Logical partition / namespace** | Payload-partitioned shard or namespace per client | One client, if the partition key is enforced | ~1× (Shared) to slightly more | Qdrant's user-defined sharding: *"each tenant its own dedicated shard ... best suited to a smaller number of larger tenants"* [[QDRANT-MT]](#src-qdrant-mt); Pinecone documents namespaces as the isolation unit [[PINECONE-DM]](#src-pinecone-dm) |
| **Collection / table per tenant** | One Weaviate tenant or Milvus partition per client | One client | ~1×–1.2× storage, higher object overhead | Weaviate native multi-tenancy with per-tenant states including offloaded [[WEAVIATE-MT]](#src-weaviate-mt); Milvus partition level, 1,024 partitions per collection [[MILVUS-MT]](#src-milvus-mt)[[MILVUS-PART]](#src-milvus-part) |
| **Database per tenant** | Milvus database per client, RBAC per database | One client, hardware-enforced | ~12× storage for 12 clients; capped | Milvus database level: *"fully separated, offering enterprise-grade data isolation ideal for regulated environments"*, *"maximum of 64 tenants by default"* [[MILVUS-MT]](#src-milvus-mt) |

Two documented facts sharpen the choice. First, the counter-argument to per-tenant collections is
economic, and Qdrant states it plainly: *"Creating a separate collection for each tenant is rarely
the most efficient approach. Each collection carries its own resource overhead"*, with Qdrant Cloud
defaulting to a *"maximum of 1000 collections"* per cluster [[QDRANT-MT]](#src-qdrant-mt). Second, the counter-argument
to sharing is technical and pgvector states it equally plainly: *"For applications with multiple
tenants, sharing an approximate index between tenants means vectors from one tenant can affect recall
(and speed) for other tenants"*, with list partitioning or separate tables as the remedies [[PGVECTOR]](#src-pgvector).
Note what that means for M5's recall figure: **an index-wide recall@10 is not the recall any single
tenant experiences.** A narrow client's effective recall is the shared number filtered by its own
$p_n$, which is why M17's ledger entry expects this module to measure it.

The cost multiplier that matters for the budget is stated by the arithmetic, not by the vendors: a
per-client index at 12 clients multiplies vector storage and index-maintenance work by 12 at the same
total chunk count, because HNSW graphs and quantization codebooks are per-index. At M5's 184 GB
endpoint that is not a rounding error, which is why the shared index plus mandatory filter is the
default posture in this course — and why M17's post-retrieval authorization backstop exists as the
non-negotiable correction for the blast radius the shared index creates.

> **Tradeoff (the ledger entry M6-4).** *Decision:* one shared index, `client_id` and `jurisdiction`
> on every node, scope filters pushed down on every query path, with per-tenant physical isolation
> reserved for tenants with contractual or regulatory demands. *Costs:* a blast radius that covers
> every client, and a filter that is now load-bearing rather than convenient. *Accepted because:* the
> physical-isolation alternative multiplies storage and index maintenance by the tenant count without
> removing the need for an authorization check (→ M17). *Reverses if:* a contract forbids a shared
> index, a residency requirement pins data to a jurisdiction, or the tenant count falls below the
> point where per-tenant overhead is negligible (Milvus caps database-level isolation at 64 tenants
> by default [[MILVUS-MT]](#src-milvus-mt); at 12 clients, and a regulated one, the arithmetic can go the other way).

---

## Hybrid metadata and semantic routing

Three mechanisms get conflated under "filtering". They answer different questions and fail
differently:

| Mechanism | Question it answers | LlamaIndex surface | Failure it introduces |
|---|---|---|---|
| **Explicit filter** | "Which subset may I search?" | `MetadataFilters` → pushed down to the store [[LI-MF]](#src-li-mf) | Wrong or missing predicate → wrong scope or destroyed recall |
| **Auto-retrieval** | "What filters does this query imply?" | `VectorIndexAutoRetriever` + `VectorStoreInfo`/`MetadataInfo` describing the collection and the filters it supports | The LLM infers a filter that is *plausible but unauthorized* [[LI-AR]](#src-li-ar) |
| **Routing** | "Which index or engine should answer this?" | `RouterQueryEngine` with `LLMSingleSelector` or `PydanticSingleSelector`, selecting among `QueryEngineTool`s by description [[LI-RT]](#src-li-rt) | A router that sends a question to the wrong index, or that becomes the only thing standing between a user and someone else's data |

Auto-retrieval is the important one to get exactly right, because it is where M17's incident is born.
The documentation is unambiguous about the mechanism: *"Given a natural language query, we first use
the LLM to infer a set of metadata filters as well as the right query string to pass to the vector db
(either can also be blank)"*, with the inference driven by a `VectorStoreInfo` that contains
*"a structured description of the vector store collection and the metadata filters it supports"*
[[LI-AR]](#src-li-ar). The consequences are two-sided and worth stating in this module's own vocabulary:

- **It is the highest-leverage way to make $p_n$ behave.** An inferred `jurisdiction` filter raised
  from the query text is *correlated by construction* with the question, so it is exactly the case
  where post-filtering is survivable and pre-filtering is cheap.
- **It is not an authorization mechanism.** "Which filters does this query imply" and "which
  documents may this principal read" are different questions with different trust models, and the
  second must never be delegated to a model's inference. M17 owns that argument, the backstop, and
  the leak taxonomy; this module owns only the observation that **the auto-retrieval prompt encodes
  the relevance schema, so a scope field accidentally added to that schema becomes a model decision.**

Routing deserves a sharper warning. A router that selects among per-tenant indexes *looks* like
isolation and is not: the selector is an LLM call driven by tool descriptions [[LI-RT]](#src-li-rt), so the
isolation guarantee is only as strong as the selector's accuracy on adversarial inputs. Physical
isolation must be enforced by the request's authenticated principal, not by a sentence in a tool
description.

---

## Worked example: the recall cost of a selective filter

Same index as M5 — 187,043 vectors at D = 1,536, HNSW `m=16`, `ef_search=256`, recall@10 of 0.97
against exact search unfiltered — against the M1 60-query set, run four ways per filter profile.
Ground truth for the filtered rows is **exact search restricted to the filter's matching set**, which
is the only ground truth that makes a filtered recall number mean anything.

| Filter profile | $p_c$ | Measured $p_n$ | Post-filter recall@10 | Post-filter with $k_{\text{fetch}} = 10/p_n$ | Pre-filter / filter-aware recall@10 | p99 |
|---|---|---|---|---|---|---|
| none (M5 baseline) | 1.00 | 1.00 | 0.97 | 0.97 @ k=10 | 0.97 | 22 ms |
| `jurisdiction = NL` | 2.5% | 0.35 | 0.86 | 0.95 @ k=29 | 0.96 | 26 ms |
| `client = Nordwind` | 8.3% | 0.08 | **0.31** | 0.90 @ k=125 | 0.93 | 24 ms |
| `client AND jurisdiction` | 0.2% | 0.03 | **0.09** | 0.87 @ k=333 | 0.92 | 30 ms |
| `client AND jurisdiction AND date range` | 0.05% | 0.01 | **0.04** | 0.83 @ k=1,000 | 0.91 | 58 ms |
| any of the above, filter forgotten | 1.00 | 1.00 | 0.97 | — | **scope leak** | 22 ms |

The last row is not a filter configuration; it is the omission that M17 exists to catch, included
here because the table would otherwise imply that forgetting a filter is a neutral act.

Five findings decide LogiTrack's metadata architecture:

1. **$p_n$, not $p_c$, predicts the damage.** The jurisdiction filter has a small $p_c$ and loses
   almost nothing (0.97 → 0.86 post-filtered), because Dutch queries have Dutch neighbours. The
   client filter has a *larger* $p_c$ and loses two-thirds of its recall, because clients' clauses
   are near-identical in embedding space. Any schema review that sorts filter axes by $p_c$ is sorting
   by the wrong column.
2. **Over-fetching works, and is bounded.** Multiplying $k_{\text{fetch}}$ by $1/p_n$ recovers most of
   the loss (0.31 → 0.90 for the client filter) — at the cost of a candidate list that is now 125
   nodes wide and a reranker downstream that must chew through them (→ M9, M18). The recovery is
   real; it is not free, and it is precisely the cost that pre-filtering avoids.
3. **As filters stack, the strategy stops mattering and the index family starts.** At $p_n = 0.01$ the
   post-filter path needs a 1,000-wide candidate list to recover 0.83, while the filter-aware path
   holds 0.91 at the highest latency in the table. Neither is a query-parameter tweak any more: this
   is the regime where the M5 decision — which index family, and whether it traverses predicates
   [[ACORN24]](#src-acorn24) — determines whether the filter is affordable at all.
4. **The scope filter must never be optional, and the difference is visible in this table.** Rows 2
   and 6 have identical latency and identical recall, and one of them is a compliance incident. A
   filter that cannot be omitted is a *contract*, not a default argument.
5. **Filtered recall must be measured per principal.** M5's 0.97 was an index-wide number. Under the
   production filter set the same index delivers 0.91 at best. Any SLO written against the unfiltered
   figure is a promise the filters were never asked to keep (→ M17, M13).

**LogiTrack's decision after M6:** every node carries `client_id`, `jurisdiction`, `effective_date`,
`doc_id`, and `clause_id`; scope fields are written by the ingest pipeline from the source system
rather than inferred; the request contract *requires* `client_id` and rejects a query without it;
relevance filters (`jurisdiction` above all) may be auto-inferred [[LI-AR]](#src-li-ar); payload indexes exist for
every filter field [[QDRANT-FILTER]](#src-qdrant-filter); and per-axis $p_n$ is measured on the evaluation set and re-measured
whenever the embedding model changes (→ M4), because correlation with the vector space is what $p_n$
measures.

> **Multimodal inputs** ($\color{red}\text{to be discussed}$) — scanned agreements make "written from the source system" a hard rule.
> - Never take `client_id` from OCR of a letterhead or stamp. OCR text is inferred, not authoritative (→ [M17](../17-multitenant-permissioned/README.md); injected text → [M21](../21-security-prompt-injection/README.md)).
> - Slice the $p_n$ measurement by `content_type` too. Table and figure-description nodes embed unlike prose, so a filter's correlation can differ per slice (argued, not measured).
> - Page-image retrieval (e.g. [ColPali](https://arxiv.org/abs/2407.01449)) would need the same scope fields on every page vector. Deferred by the [syllabus](../../syllabus.md#1-scope-knobs-frozen).

---

## Design exercise

> *Paper-based. Think, then write.*

**Task.** Design LogiTrack's metadata schema and its isolation model, and state the leak scenario you
are defending against. Deliver a schema, a contract, a recall budget, and a cost.

1. **Enumerate the axes from the query log, not from the schema you wish you had.** List every field
   the request path could plausibly filter on. Classify each one as **scope** (authorization),
   **freshness**, **provenance**, or **relevance** — and mark which classification is a legal fact and
   which is an engineering preference.
2. **Make the scope contract explicit.** State which fields a query cannot omit, what happens when
   they are absent (reject, or default to the principal's own set), and who writes them. If they are
   model-inferred, write that down as a finding and route it to M17.
3. **Measure $p_n$ per axis, and say how you measured it.** For each axis: corpus fill $p_c$,
   neighbourhood fill $p_n$ against your evaluation queries, and expected post-filter top-10 yield.
   A schema review that reports only $p_c$ has not been done. The M6 lab — adding the filter to the
   M5 store and measuring what the selectivity costs in recall — is this measurement, and it is the
   only deliverable in this exercise that cannot be produced from a vendor document.
4. **Choose the isolation profile and pay for it.** Shared index, partition/namespace, collection per
   tenant, or database per tenant — with the storage and index-maintenance multiplier for your tenant
   count stated arithmetically (12 clients is 12× the graphs and codebooks on a shared-corpus split),
   plus the blast radius you are accepting per profile (→ M5's engine table, `maps/vector-db-map.md`
   for the licence and tenancy capabilities).
5. **Decide the enrichment layer.** Which LLM-generated fields ship (title, summary, keywords,
   questions-answered [[LI-ME]](#src-li-me)) and what they cost at ingest, against what they buy in retrieval
   reach. State the regeneration plan, since the prompt will change.
6. **Set the filter-strategy rule.** For each axis: pushed down, filter-aware, pre-filter with full
   scan, or post-filter with an over-fetch factor. Use the crossover arithmetic from this module
   rather than a preference.
7. **Define the acceptance test and the leak test.** Acceptance: recall@10 **per filter profile**
   against exact search restricted to the matching set, at production selectivity. Leak: the
   cross-client query, the missing-filter query, and the model-inferred-scope query — each of which
   must return nothing rather than something (→ M17's lab for the red-team version).

**Why this exercise matters.** Metadata is where a RAG system's correctness, cost, and compliance all
meet, and it is the layer most likely to be designed implicitly: whatever fields the loader happened
to attach become the schema. The two failures in this module's opening scene — recall destroyed by a
correct filter and scope destroyed by an optional one — have the same root cause, which is that
nobody decided what the fields were *for*. The instrument that separates them is a single measured
number per axis ($p_n$), and it is the number no vendor benchmark reports and no dashboard shows.
Note what the worked example refuses to do: it does not conclude that filters are bad or that
post-filtering is always wrong. It shows that the *same* filter can cost 0.02 or 0.66 of recall
depending entirely on the correlation between the filter and the embedding — which is a property of
your corpus, your queries, and your model, and therefore something only you can measure.

---

## Where the literature disagrees

- **"Selectivity" points in two directions.** Qdrant: *"In the case of high-selectivity (weak)
  filters, you can use the HNSW index as it is. In the case of low-selectivity (strict) filters, you
  can use the payload index and do a complete rescore"* [[QDRANT-INDEX]](#src-qdrant-index) — where selectivity describes
  how much is selected. The filtered-ANN benchmark parameterises *"query selectivities"* as a sweep
  over filter restrictiveness without that inversion [[FANNS25]](#src-fanns25), and in database practice "highly
  selective" normally means restrictive. **What to change:** define the term in your own design
  document, in the first paragraph, and state the convention — this module uses "fraction admitted".
  A schema review that mixes conventions produces advice that is exactly backwards.
- **"Separate collections per tenant, or one shared index?"** Qdrant argues against per-tenant
  collections on cost: *"Creating a separate collection for each tenant is rarely the most efficient
  approach. Each collection carries its own resource overhead"* [[QDRANT-MT]](#src-qdrant-mt), and Pinecone's
  documented isolation unit is the namespace rather than a physical index [[PINECONE-DM]](#src-pinecone-dm). pgvector
  argues the other way from the failure mode: *"sharing an approximate index between tenants means
  vectors from one tenant can affect recall (and speed) for other tenants"* [[PGVECTOR]](#src-pgvector). Milvus
  offers both and prices the strong option in tenant count — database-level isolation is *"fully
  separated, offering enterprise-grade data isolation ideal for regulated environments"* but
  supports *"a maximum of 64 tenants by default"* [[MILVUS-MT]](#src-milvus-mt). **What to change:** treat this as a
  contract-and-scale decision, not a performance decision. The vendors do not disagree about the
  mechanics; they disagree about which default is safer to assume, and the tie-breaker is a
  contractual term or a residency obligation, neither of which is a benchmark result.
- **"Filtered-ANN indexes are the answer to selective filters."** ACORN reports *"2-1,000x higher
  throughput at a fixed recall"* [[ACORN24]](#src-acorn24); PathFinder reports *"up to 9.8x in query throughput at
  recall 0.95"* [[PATHF25]](#src-pathf25); UNIFY reports *"state-of-the-art RF-ANNS performance across small, mid, and
  large query ranges"* and single-index convenience [[UNIFY24]](#src-unify24); window filters report *"up to a 75×
  speedup ... at the same level of recall"* [[WINDOW24]](#src-window24). Every one of those is a speedup *at fixed
  recall*, measured against a baseline chosen by the authors, on a dataset chosen by the authors. The
  independent position is the reason the FANNS benchmark exists: an in-depth literature review
  *"identifie[d] a key gap in the research landscape: publicly available datasets with embedding
  vectors from state-of-the-art transformer-based text embedding models that contain abundant
  real-world attributes"* [[FANNS25]](#src-fanns25). **What to change:** the honest claim is weaker than the papers'
  headline — filter-aware indexes are the *only* family that survives high-selectivity filters at
  scale, and which one wins on your workload depends on your attribute types and distributions, which
  is exactly what the benchmark was built to make testable.
- **"Auto-retrieval makes filtering easy."** The documented pattern infers filters from the query
  with an LLM [[LI-AR]](#src-li-ar), and it is genuinely the best available way to raise $p_n$: an inferred
  jurisdiction filter is correlated with the question by construction. The same mechanism, applied to
  a scope field, produces the incident M17 documents — *"a relevance filter is not a permission
  system"*. **What to change:** keep the inference for relevance axes and put scope axes under a
  contract the model cannot touch, then treat the difference as a schema property rather than a
  prompt-engineering detail. M17 owns the enforcement; this module owns the schema that makes the
  distinction expressible.

---

## Sources

- <a id="src-li-mf"></a>[Qdrant Vector Store — Metadata Filter](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/qdrant_metadata_filter/) — `[LI-MF]` · owns `MetadataFilter`/`MetadataFilters`/`FilterOperator`/`FilterCondition`, `index.as_retriever(filters=...)`, and the printed translation of a LlamaIndex filter expression into native `FieldCondition` predicates.
- <a id="src-li-ar"></a>[Auto-Retrieval from a Vector Database](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/elasticsearch_auto_retriever/) — `[LI-AR]` · owns `VectorIndexAutoRetriever` with `VectorStoreInfo`/`MetadataInfo` and the LLM-inferred-filter mechanism. Owned by M6 by the §6 ownership map; M17 cites this key.
- <a id="src-li-me"></a>[Metadata Extraction](https://developers.llamaindex.ai/python/framework/module_guides/indexing/metadata_extraction/) — `[LI-ME]` · owns the LLM enrichment extractors (`TitleExtractor`, `QuestionsAnsweredExtractor`, `SummaryExtractor`, `KeywordExtractor`, `EntityExtractor`) and the fields they attach to nodes.
- <a id="src-li-rt"></a>[Routers](https://developers.llamaindex.ai/python/framework/module_guides/querying/router/) — `[LI-RT]` · owns `RouterQueryEngine`, the LLM/Pydantic selector split, and the choice-set-by-metadata framing.
- <a id="src-li-vs"></a>[Vector Stores](https://developers.llamaindex.ai/python/framework/module_guides/storing/vector_stores/) — `[LI-VS]` · owns the cross-integration matrix whose metadata-filtering column decides which backend can honour a pushed-down filter (surveyed in M5).
- <a id="src-li-np"></a>[Node Parser Usage Pattern](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/) — `[LI-NP]` · owns the attribute-inheritance rule that carries document metadata onto every child node (surveyed in M3).
- <a id="src-fanns25"></a>*Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors* — Patrick Iff, Paul Bruegger, Marcin Chrapek, David Kochergin, Maciej Besta, Torsten Hoefler · arXiv:2507.21989 (2025) · [arxiv.org/abs/2507.21989](https://arxiv.org/abs/2507.21989) — `[FANNS25]` · owns the filtered-ANN benchmark dataset (2.7M arXiv abstracts, 11 real attributes), the eleven-method comparison, and the query-selectivity parameterisation.
- <a id="src-window24"></a>*Approximate Nearest Neighbor Search with Window Filters* — Joshua Engels, Benjamin Landrum, Shangdi Yu, Laxman Dhulipala, Julian Shun · arXiv:2402.00943 (2024) · [arxiv.org/abs/2402.00943](https://arxiv.org/abs/2402.00943) — `[WINDOW24]` · owns the c-approximate window-search formulation for numeric label ranges and the up-to-75× speedup at fixed recall.
- <a id="src-pathf25"></a>*PathFinder: Efficiently Supporting Conjunctions and Disjunctions for Filtered Approximate Nearest Neighbor Search* — Tianming Wu, Dixin Tang · arXiv:2511.00995 (2025) · [arxiv.org/abs/2511.00995](https://arxiv.org/abs/2511.00995) — `[PATHF25]` · owns the cost-based optimizer for multi-attribute conjunctions/disjunctions and the index-borrowing optimization.
- <a id="src-unify24"></a>*UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search* — Anqi Liang, Pengcheng Zhang, Bin Yao, Zhongpu Chen, Yitong Song, Guangxu Cheng · arXiv:2412.02448 (2024) · [arxiv.org/abs/2412.02448](https://arxiv.org/abs/2412.02448) — `[UNIFY24]` · owns the segmented inclusive graph that supports pre-, post-, and hybrid filtering in one index, and the index-consistency motivation.
- <a id="src-acorn24"></a>*ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data* — Liana Patel, Peter Kraft, Carlos Guestrin, Matei Zaharia · arXiv:2403.04871 (2024) · [arxiv.org/abs/2403.04871](https://arxiv.org/abs/2403.04871) — `[ACORN24]` · owns predicate subgraph traversal and filter-aware graph construction (surveyed in M5).
- <a id="src-pgvector"></a>[pgvector `README.md`](https://github.com/pgvector/pgvector/blob/master/README.md) — `[PGVECTOR]` · owns the post-filter-after-index-scan statement, the 10%-selectivity/ef_search=40 arithmetic, iterative index scans and their scan caps, partial indexing and partitioning remedies, and the shared-index multi-tenant warning (surveyed in M5).
- <a id="src-qdrant-index"></a>[Qdrant — Indexing / filterable HNSW](https://qdrant.tech/documentation/concepts/indexing/) — `[QDRANT-INDEX]` · owns the weak/strict filter phrasing that inverts the selectivity convention, the query planner's `full_scan_threshold` fallback, and the filterable-HNSW middle band (surveyed in M5).
- <a id="src-qdrant-mt"></a>[Qdrant — Configure Multitenancy](https://qdrant.tech/documentation/guides/multiple-partitions/) — `[QDRANT-MT]` · owns the three isolation strategies, the `is_tenant` payload-index flag, the cost argument against per-tenant collections, and the 1,000-collection default cap (surveyed in M5).
- <a id="src-qdrant-filter"></a>[Qdrant — Filtering](https://qdrant.tech/documentation/concepts/filtering/) — `[QDRANT-FILTER]` · owns payload indexes and nested boolean clauses (surveyed in M5).
- <a id="src-milvus-mt"></a>[Milvus — Multi-tenancy](https://github.com/milvus-io/milvus-docs/blob/v3.0.x/site/en/reference/multi_tenancy.md) — `[MILVUS-MT]` · owns the four isolation levels, the 64-tenant database default, and the regulated-environment rationale (surveyed in M5).
- <a id="src-milvus-part"></a>[Milvus — Manage Partitions](https://github.com/milvus-io/milvus-docs/blob/v3.0.x/site/en/userGuide/collections/manage-partitions.md) — `[MILVUS-PART]` · owns the 1,024-partition limit and the partition-key multi-tenancy pointer (surveyed in M5).
- <a id="src-milvus-index"></a>[Milvus — Index Explained](https://github.com/milvus-io/milvus-docs/blob/v3.0.x/site/en/userGuide/indexes/index-explained.md) — `[MILVUS-INDEX]` · owns the index catalogue and the scalar-filtering framing (surveyed in M5).
- <a id="src-weaviate-filter"></a>[Weaviate — Filtering](https://docs.weaviate.io/weaviate/concepts/filtering) — `[WEAVIATE-FILTER]` · owns the pre-filtering description, the ACORN default since v1.34, and the low-correlation-with-the-query-vector rationale (surveyed in M5).
- <a id="src-weaviate-mt"></a>[Weaviate — Data / multi-tenancy](https://docs.weaviate.io/weaviate/concepts/data) — `[WEAVIATE-MT]` · owns tenant states including offloaded (surveyed in M5).
- <a id="src-es-knn"></a>[Elasticsearch kNN search](https://www.elastic.co/docs/solutions/search/vector/knn) — `[ES-KNN]` · owns the exact-brute-force-for-pre-filtered-subsets guidance (surveyed in M5).
- <a id="src-pinecone-dm"></a>[Pinecone data modeling](https://docs.pinecone.io/guides/index-data/data-modeling) — `[PINECONE-DM]` · owns namespaces as the documented isolation unit (surveyed in M5).

**Unsupported (synthetic, or not verified in this module):** the LogiTrack corpus, the 12-client and
40-jurisdiction counts, the 187,043-vector index, the 60-query set, every recall, latency and yield
figure in the worked-example tables, the filter field names, and all tradeoff-ledger entries are
constructed for teaching and are not empirical findings. **Derived in-file:** the over-fetch formula
$k_{\text{fetch}} \approx k/p_n$, the expected post-filter yield $k \cdot p_n$, the corpus-fill
arithmetic $1/40 \times 1/12$, and the per-tenant cost multiplier (12 clients → 12× graphs and
codebooks). **Argued here, not measured:** that neighbourhood fill $p_n$ rather than corpus fill
$p_c$ predicts post-filter recall damage. The mechanism follows from the cited post-filter statements
[[PGVECTOR]](#src-pgvector)[[QDRANT-INDEX]](#src-qdrant-index) and from the correlation rationale the filtered-ANN literature and Weaviate
both invoke [[FANNS25]](#src-fanns25)[[WEAVIATE-FILTER]](#src-weaviate-filter), but no source fetched for this module defines or measures
$p_n$ under that name; the M6 lab is where it is measured, and until it is, this module's central
claim is a mechanism with a procedure attached rather than a result. **Not verified:** which of the
four isolation profiles any given engine implements in a way that survives a determined adversarial
query — the table records documented isolation *models*, not audited isolation guarantees (→ M17's
red-team lab). The claim that an index-wide recall figure is never the per-tenant figure is argued
from pgvector's shared-index warning [[PGVECTOR]](#src-pgvector) and is measured, per principal, in M17.

**Verification note:** 🟡 *pending* — Step 7 (memory-less logic review) and Step 8 (literature
survey) not yet run on this module.

**Next module:** [M7 — Sparse, Dense, and Hybrid Retrieval](../07-sparse-dense-hybrid/README.md)
— the queries that no filter can rescue, because the corpus and the asker never shared a vocabulary,
and the exact-token queries where the embedding was never the right instrument at all.
