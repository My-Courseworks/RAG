# M7 · Sparse, Dense, and Hybrid Retrieval

> **Module question:** When does lexical search beat embeddings, and how do you combine them?
> **Cross-cutting threads:** Failure modes · Tradeoff ledger · LlamaIndex at a glance
> **Domain spine:** LogiTrack — "clause 14.2(b)" and "UN3480": the queries the dense retriever misses entirely
> **Prerequisites:** [M3 — Corpus Acquisition, Parsing, and the Chunking Decision](../03-parsing-and-chunking/README.md) · [M4 — Embeddings and Vector Space](../04-embeddings-and-vector-space/README.md) · [M5 — Vector Databases and ANN Indexes](../05-vector-databases-and-ann/README.md) · [M6 — Metadata, Filters, and Multi-Tenancy](../06-metadata-filters-multitenancy/README.md)

---

## Opening scene — the number the embedding could not see

By M6's end LogiTrack's retrieval is defensible. Filters are pushed down, scope is enforced, recall
is measured per filter profile at 0.91, and the evaluation set has become a habit rather than a
chore. Then a customs broker asks the question that gets asked forty times a day:

> *"What does clause 14.2(b) require for UN3480 documentation?"*

The system answers about documentation requirements. The answer is fluent, cited, and about the
wrong clause — and about lithium batteries *in general* rather than about UN3480, which is a
lithium-ion cell, differs from UN3481 (cells packed with equipment) in the paperwork it requires,
and shares almost no vocabulary with it. Two failures arrive together:

- **An identifier has no semantics to embed.** `14.2(b)` is a label. Nothing about the string
  predicts that it sits next to `14.2(a)` and `14.2(c)` in a document and far from `15.2(b)` in
  another; the meaning is *assigned*, not distributed. An embedding trained on web text has had no
  opportunity to learn your document's numbering scheme, and the representational capacity to encode
  arbitrary label adjacency is bounded by dimensionality anyway [[LIMIT25]](#src-limit25).
- **Domains where dense models are known to be weak are exactly these.** Entity-centric questions —
  the ones whose answer turns on a *named entity* rather than a described concept — are documented
  to challenge dense retrievers specifically, which is why the field has a benchmark named after
  that failure [[ENTITY21]](#src-entity21).

Meanwhile BM25, over the same 187,043 nodes, returns the right clause on the first try in 96% of the
identifier-shaped queries. `14.2(b)` appears *verbatim* in exactly one chunk, and a rare exact token
is the strongest possible lexical signal. Nobody had to train anything. Lexical retrieval was never
removed from the pipeline; it was simply never added.

The obvious fix makes things worse. Concatenating the two ranked lists and taking the first ten fills
the top-10 with BM25's boilerplate matches — every chunk mentioning "clause" and "documentation",
including a table of contents and a definitions page — because BM25 scores are unbounded and
corpus-dependent while cosine similarity lives in a fixed range. Merging them by raw score compares
a temperature to a distance. The naive fusion is not a 50/50 blend of two useful signals; it is a
weighted average of a useful signal and a noise source, on an arbitrary scale.

> **Failure mode (the module in one line):** **choosing one retrieval instrument for all queries,
> and fusing two instruments without a common unit** — so a dense-only system silently misses every
> query whose answer is an identifier rather than a meaning (**F4 · representation mismatch**), and a
> naively-fused system buries both retrievers' real answers under the other's scale mismatch
> (**F2 · retrieval noise**).

---

## BM25: what the ranking function actually computes

BM25 is not "keyword matching". It is a probabilistic ranking function with two deliberately
engineered saturations, and the saturations are why it survives alongside neural retrieval.

The Okapi family was introduced and tuned in the TREC-3 proceedings as *"Okapi at TREC-3"* by
Robertson, Walker, Jones, Hancock-Beaulieu & Gatford [[ROB94]](#src-rob94), and the probabilistic relevance
framework behind it — the Binary Independence Model extended with term frequency and length — is
laid out in *The Probabilistic Relevance Framework: BM25 and Beyond* by Robertson & Zaragoza [[BM25]](#src-bm25).
The standard scoring form, as given in the IR textbook's treatment of the Okapi weighting scheme
[[IRBOOK09]](#src-irbook09), is

$$\text{RSV}_d = \sum_{t \in q} \log\frac{N}{df_t} \cdot \frac{(k_1+1)\,tf_{td}}{k_1\big((1-b) + b\,(L_d/L_{ave})\big) + tf_{td}}$$

with three components that do three different jobs:

| Component | Form | What it does | What happens without it |
|---|---|---|---|
| **IDF** | $\log (N/df_t)$ | Rewards terms that are *rare in the corpus* | A stopword outranks `UN3480` |
| **TF saturation** | $\frac{(k_1+1)tf}{k_1+\dots+tf}$ | A term occurring 20 times is not 20× as important as once | Long, repetitious chunks dominate |
| **Length normalization** | $b \cdot L_d / L_{ave}$ | Corrects for chunk length | Verbose chunks win on volume alone |

The textbook is explicit about the two tuning parameters: `k1` *"is a positive tuning parameter that
calibrates the document term frequency scaling. A k1 value of 0 corresponds to a binary model (no
term frequency), and a large value corresponds to using raw term frequency"*, while `b`
*"determines the scaling by document length: b = 1 corresponds to fully scaling the term weight by
the document length, while b = 0 corresponds to no length normalization"*, and — absent per-collection
optimization — *"experiments have shown reasonable values are to set k1 and k3 to a value between 1.2
and 2 and b = 0.75"* [[IRBOOK09]](#src-irbook09).

**Why the IDF term is the whole story for identifier queries.** At LogiTrack's scale, `UN3480` occurs
in perhaps 40 of 187,043 chunks, so $\log(N/df) \approx \log(4{,}676) \approx 8.5$, while `documentation`
occurs in tens of thousands and scores near zero. The same asymmetry is invisible to an embedding
model: it maps both tokens into a learned direction, and the direction it chooses for an arbitrary
identifier is an artifact of training data it never had. This is the mechanism behind the
entity-centric result [[ENTITY21]](#src-entity21), and it is a *permanent structural* difference between the two
instruments, not a quality gap that a better encoder closes — no fixed-dimension single-vector index
can represent arbitrary identifier distinctions [[LIMIT25]](#src-limit25).

**The cost side is equally structural.** BM25 needs an inverted index, which means a *vocabulary*
built from your corpus. The vocabulary is per-corpus, so it must be rebuilt when the corpus changes,
and its maintenance belongs to the ingest pipeline (→ M14) — a fact that surprises teams who assume
the sparse index is free because it is old. It also has no semantics: `detention` and `demurrage` are
unrelated tokens no matter how related the clauses are, which is exactly what dense retrieval exists
to fix.

LlamaIndex's `BM25Retriever` is the concrete surface. It constructs from nodes, an index, or a
docstore, supports disk persistence, and exposes the two parameters that matter at ingest:
`stemmer=Stemmer.Stemmer("english")` and `language="english"` — the docs note this is *"important for
removing stopwords and stemming the query + text"* and that the default is English for both
[[LI-BM25]](#src-li-bm25). Both defaults are decisions, not details: they determine what the index can distinguish,
they are applied at ingest *and* at query time so a mismatch changes results silently, and they are
tuned for English prose rather than for a corpus full of alphanumeric codes. Check them against your
own identifier population — and check that the same settings are used on both sides of the query.

> **Multimodal inputs** ($\color{red}\text{to be discussed}$) — BM25 only sees the tokens the parser emitted, so scan and table handling decide sparse recall.
> - One OCR slip on an identifier (`UN348O`, `14.2(6)`) and the exact token is gone. BM25 cannot match it at all.
> - Run your identifier regex over OCR'd scanned pages too, not only native-text PDFs. Count the misses.
> - Table cells flattened without headers become bare numbers. Repeating headers per row inflates TF and length. Serialization is M3's call ([M3](../03-parsing-and-chunking/README.md)).
> - Vision-model descriptions of figures or stamps: a made-up code is a rare token, so IDF ranks it high. Keep generated text in its own flagged field.

> **Tradeoff (the ledger entry M7-1).** *Decision:* keep a sparse retriever permanently alongside
> the dense one, and treat "which instrument answers this query" as a first-class design question.
> *Costs:* a second index to build and maintain, a second ingest artifact (vocabulary, stemmer
> configuration, stopword list) that must stay in sync with the corpus, and a fusion stage whose
> failure modes are new. *Accepted because:* identifier-shaped queries are unrecoverable by the dense
> instrument at any dimension you can afford [[LIMIT25]](#src-limit25)[[ENTITY21]](#src-entity21), and their share of a
> compliance-driven query log is not small. *Reverses if:* the query log contains no
> identifier-shaped queries and no rare-token queries at all — a finding you can obtain in an hour
> by counting regex matches in your logs, and one that should be *demonstrated*, not assumed.

---

## Learned sparse: expansion without leaving the inverted index

BM25's weakness is vocabulary: it matches tokens, so it cannot bridge `lithium cell` to `UN3480` or
`demurrage` to `detention` unless the words co-occur. Learned sparse retrieval closes that gap while
keeping the inverted index.

SPLADE is the canonical result. Its contribution is *"a new first-stage ranker based on explicit
sparsity regularization and a log-saturation effect on term weights, leading to highly sparse
representations"*, motivated by representations that *"could inherit from the desirable properties of
bag-of-words models such as the exact matching of terms and the efficiency of inverted indexes"*
[[SPLADE21]](#src-splade21). The follow-up work — the extension that became SPLADE v2 — showed that sparse models
*"benefit from the same training improvements as dense models"*, in particular distillation and
hard-negative sampling [[SPLADE22]](#src-splade22). Expansion is the gain: a document about lithium cells can carry
weight on the literal token `UN3480` even if the chunk never spells it out, because the model learned
the association.

The most economical framing for the architecture decision is that sparse and dense are *the same
model family with different output constraints*. M3-Embedding trains one model that *"can
simultaneously accomplish the three common retrieval functionalities: dense retrieval, multi-vector
retrieval, and sparse retrieval"* through self-knowledge distillation, showing that the retrieval
function is a property of the output head rather than of a separate system [[M3EMB24]](#src-m3emb24).

> **LlamaIndex at a glance:** this module's two primitives are `BM25Retriever`, constructed with
> `BM25Retriever.from_defaults(nodes=... | docstore=... | index=..., similarity_top_k=...)` from
> `llama_index.retrievers.bm25` with documented `stemmer`/`language` defaults of English, and
> `QueryFusionRetriever`, constructed as `QueryFusionRetriever([retriever_a, retriever_b],
> similarity_top_k=..., num_queries=..., mode=..., retriever_weights=[...], use_async=...)` from
> `llama_index.core.retrievers` [[LI-BM25]](#src-li-bm25)[[LI-FUS]](#src-li-fus). Three properties of that surface are load-bearing
> here. First, the sparse retriever is built from **nodes, an index, or the docstore you already
> have**, so adding lexical retrieval does not require re-embedding or re-ingesting — the inverted
> index is a second view of the same `Node` objects (which is why its vocabulary must be rebuilt when
> the docstore changes, → M14). Second, it accepts `MetadataFilters` like any other retriever, so the
> M6 scope contract applies to the lexical path too and must be pushed down there independently
> [[LI-BM25]](#src-li-bm25). Third, fusion is a retriever in its own right: the nodes come back as *"the top-k across
> all queries and indexes, as well as handling de-duplication of any nodes"* [[LI-QF]](#src-li-qf), and
> `num_queries` is a **query-generation** count rather than a retriever count — the default of 4 means
> *"it will query 4 times, once with your original query, and generate 3 more queries"*, with
> `num_queries=1` disabling generation [[LI-SF]](#src-li-sf). Fusion and query expansion therefore arrive as the
> same constructor argument, and the generation calls, their latency, and their dollar cost belong to
> M10's budget rather than to this module's ranking discussion. Verify which mode you are in before
> attributing a latency regression to fusion.

| Instrument | Exact tokens | Vocabulary bridging | Index cost | Trained? | Latency character |
|---|---|---|---|---|---|
| BM25 | **Strongest** — the token must be present | None | Inverted index; corpus-sized vocabulary | No | Cheapest per query |
| Learned sparse (SPLADE-class) | Strong — exact matches remain scored | Learned expansion [[SPLADE21]](#src-splade21) | Inverted index with *denser* postings (expansion adds terms per document) | Yes, at index and query time | Cheap, but heavier than BM25 |
| Dense bi-encoder (→ M4) | Weak — identifiers are arbitrary directions [[ENTITY21]](#src-entity21) | **Strongest** — the entire point | ANN index over fixed-width vectors (→ M5) | Yes | ANN search + optional rescore |
| Multi-vector (ColBERT-class) | Moderate | Strong | Per-token vectors; storage-heavy | Yes | Surveyed in M9 |

LlamaIndex supports the sparse path as a first-class query mode rather than a bolt-on: the vector
store query surface exposes `VectorStoreQueryMode` and per-store modes such as `"sparse"` alongside
`"default"` and `"hybrid"`, with `sparse_top_k` controlling how many sparse results to take
[[LI-PGV]](#src-li-pgv). There is also a `BaseSparseEmbedding` base class in `llama_index.core.base.embeddings`,
which is the extension point for a learned-sparse encoder. **How thinly documented that extension
point is:** no module-guide page for `BaseSparseEmbedding` was located in this module's doc fetches —
it appears in integration examples rather than in a guide — so treat learned-sparse support as
backend-dependent and verify against your chosen store before designing around it. That is an honest
gap in the stack's documentation, not a claim about its absence.

> **Tradeoff (the ledger entry M7-2).** *Decision:* use BM25 as the mandatory sparse instrument and
> treat learned sparse as an upgrade to evaluate, not as the baseline. *Costs:* forgoing expansion's
> recall on vocabulary-gap queries until the upgrade is measured, and paying model inference at
> ingest and query time when it ships. *Accepted because:* BM25 requires no model, no training data,
> and no GPU, so it delivers the identifier-query win — which is the *whole reason sparse retrieval
> is in the pipeline* — at close to zero operational cost. *Reverses if:* a measured subset of
> queries fails on exact tokens despite correct spelling, i.e. the corpus's own vocabulary does not
> contain the user's words, which is the case SPLADE's expansion exists to fix [[SPLADE21]](#src-splade21).

---

## Fusion: why two ranked lists cannot simply be added

The fusion problem is a units problem, and it has two families of answers.

**Rank-based fusion** discards scores entirely. Reciprocal Rank Fusion, introduced by Cormack, Clarke
& Buettcher as a rank-combination method that *"outperforms condorcet and individual rank learning
methods"* [[RRF09]](#src-rrf09), assigns

$$\text{RRF}(d) = \sum_{r \in R(d)} \frac{1}{k + \text{rank}_r(d)}$$

combining result sets *"with different relevance indicators"* where *"the different relevance
indicators do not have to be related to each other"* — the property Elasticsearch's documentation
states as *"RRF requires no tuning"* [[ES-RRF]](#src-es-rrf). Rank-based fusion is the default for a good reason:
ranks are unitless, so no normalization is needed and no assumption is made about whether a BM25
score of 12.4 is comparable to a cosine similarity of 0.87.

**Score-based fusion** keeps magnitudes and pays for a normalization. Two implementations are
documented by the engines and exposed by LlamaIndex:

- **Relative score fusion** — *"applies a MinMax scaler to each result set, then makes a weighted
  sum"*, which the LlamaIndex guide demonstrates giving the vector retriever *"slightly more weight
  than BM25 (0.6 vs. 0.4)"* via `retriever_weights=[0.6, 0.4]` and `mode="relative_score"` [[LI-FUS]](#src-li-fus).
- **Distribution-based score fusion (DBSF)** — normalizes each retriever's scores using the mean and
  standard deviation of its own returned set, mapping the three-sigma extremes onto a common range,
  so that *"different score magnitudes no longer matter because each retriever contributes on the
  same comparable range"* [[QDRANT-HYBRID]](#src-qdrant-hybrid).

The stack exposes all three as one constructor. `QueryFusionRetriever` takes *"a list of retrievers,
a `similarity_top_k`, a `num_queries` and a `mode`"*, with `mode="reciprocal_rerank"`,
`mode="relative_score"`, and `mode="dist_based_score"` all demonstrated in the guides, plus
`retriever_weights` for the score-based modes and `use_async` to fan the retrievers out
concurrently [[LI-FUS]](#src-li-fus).

| Fusion | Needs | Assumes | Documented position |
|---|---|---|---|
| RRF (rank) | Nothing | Ranks are comparable; magnitudes are not informative | *"RRF requires no tuning"* [[ES-RRF]](#src-es-rrf); formula with a ranking constant $k$ [[ES-RRF]](#src-es-rrf); Qdrant documents $k$ defaulting to 2 [[QDRANT-HYBRID]](#src-qdrant-hybrid) |
| Relative score (MinMax + weights) | A weight per retriever | Each retriever's *returned set* spans a comparable range | Weaviate's approach, implemented by LlamaIndex as `mode="relative_score"` [[LI-FUS]](#src-li-fus)[[WEAVIATE-HYBRID]](#src-weaviate-hybrid) |
| DBSF (mean/σ per set) | Nothing at query time | Score distributions are roughly unimodal per retriever | *"stateless"*, *"available as of v1.11.0"* [[QDRANT-HYBRID]](#src-qdrant-hybrid) |
| Learned convex combination | A small labelled set | $s = \alpha\,s_{\text{dense}} + (1-\alpha)\,s_{\text{lexical}}$ is the right shape | The analysis finds CC *"outperforms RRF in in-domain and out-of-domain settings"* and is *"sample efficient, requiring only a small set of training examples"* [[FU22]](#src-fu22) |

One operational detail worth pinning down, because it changes latency more than the fusion method
does: `num_queries` is not the number of retrievers. `QueryFusionRetriever` interprets it as a
*query-generation* count — *"it will query 4 times, once with your original query, and generate 3
more queries"* using a default generation prompt, and *"set this to 1 to disable query generation"*
[[LI-SF]](#src-li-sf). The retrieval call therefore contains an LLM call unless you turn it off, and that call
sits inside the 2.5 s budget (→ M10 owns query transformation; M7 owns the fact that fusion and
query expansion arrive as the same constructor argument).

> **Tradeoff (the ledger entry M7-3).** *Decision:* default to rank-based fusion (RRF) for the
> shipped system, and treat score-based fusion as a measured upgrade requiring a labelled sample.
> *Costs:* discarding score magnitudes, so a confidently-ranked sparse hit and a marginal one
> contribute identically; and inheriting RRF's sensitivity to its ranking constant, which is
> documented to matter [[FU22]](#src-fu22) and which implementations set differently (Qdrant defaults to 2
> [[QDRANT-HYBRID]](#src-qdrant-hybrid); Elasticsearch documents the constant without choosing for you [[ES-RRF]](#src-es-rrf)).
> *Accepted because:* RRF needs no normalization and no labels, so it can ship before an evaluation
> set exists, and it is the only fusion method whose failure mode is "mediocre" rather than
> "arbitrary units". *Reverses if:* a labelled sample exists and a convex combination measurably
> beats RRF on your queries — which the fusion literature reports is the likely outcome [[FU22]](#src-fu22).

---

## When does hybrid actually win? A diagnosis, not a slogan

The correct question is not "does hybrid beat dense". It is **which subset of your query log is
lexical, which is semantic, and which is neither** — because the answer differs per subset, and the
subsets are cheap to identify.

| Query shape | LogiTrack example | Which instrument answers it | Why |
|---|---|---|---|
| **Identifier / exact token** | "clause 14.2(b)", "UN3480", "form CMR-9" | Sparse | The token is present verbatim; IDF makes it decisive; the embedding has no basis for the distinction [[ENTITY21]](#src-entity21)[[LIMIT25]](#src-limit25) |
| **Semantic paraphrase** | "how long do we have to claim damages after a delayed delivery" | Dense | The corpus says "demurrage liability window"; no shared tokens |
| **Numeric / tabular** | "40ft rate Rotterdam to Singapore in March" | Neither — route around the index (→ M8, M10) | The answer is a cell, not a passage (M3's table problem) |
| **Cross-reference / multi-hop** | "the DG rules that §4.1 of the carrier agreement points to" | Neither — needs a structured or graph index (→ M8) | Two documents must be joined |
| **Mixed** | "what does UN3480 require for Rotterdam documentation" | Both, fused | Identifier *and* intent in one query |

> **Multimodal inputs** ($\color{red}\text{to be discussed}$) — some answers sit in non-prose parts of the page. Tag them before blaming fusion.
> - Content only in a figure, chart or scanned stamp: neither instrument sees it unless ingestion made it text ([M3](../03-parsing-and-chunking/README.md), [M14](../14-ingestion-at-scale/README.md)).
> - Tag queries whose gold span came from OCR or a generated description. Report recall@10 for that slice on its own ([M13](../13-evaluation/README.md)).
> - Deferred: page-image retrieval such as ColPali, which embeds page images directly ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)). It would be a third ranked list, and RRF fuses ranks, not units. Out of scope per [syllabus §1](../../syllabus.md#1-scope-knobs-frozen).

Three research threads support the diagnosis and none of them supports the slogan:

- **No single instrument dominates.** BEIR's zero-shot evaluation across 18 datasets found BM25 a
  robust baseline while dense and sparse models were often more efficient but underperformed,
  leaving *"considerable room for improvement in their generalization capabilities"* [[BEIR21]](#src-beir21) — the
  out-of-domain statement of the same asymmetry. DPR, in-domain, found the opposite for
  open-domain QA [[DPR20]](#src-dpr20). Both own their claims in M4; the reading here is only that "which is
  better" is a question about a *distribution*, not about retrieval.
- **Selection is possible and cheap.** Arabzadeh, Yan & Clarke train a classifier *"to select a
  suitable retrieval strategy (i.e., sparse vs. dense vs. hybrid) for individual queries"*, reporting
  that *"leveraging sparse retrievers for queries which can be answered with sparse retrievers
  decreases the number of calls to GPUs"* while maintaining utility [[SEL21]](#src-sel21). That is the strongest
  available argument that per-query routing belongs in the architecture rather than in a wish list.
- **Light hybrids generalise.** Hybrid-LITE *"saves 13X memory while maintaining 98.0% performance
  of the hybrid retriever of BM25 and DPR"*, and its authors report that *"light hybrid retrievers
  achieve better generalization performance than individual sparse and dense retrievers"* — followed
  immediately by the caveat that *"there is a large room to improve the robustness of retrievers"*
  [[LITE22]](#src-lite22). Fusion is not a guarantee; it is a better prior.

There is also a ceiling on what any of this fixes. BRIGHT showed the MTEB leaderboard's leading
model scoring *"18.3"* nDCG@10 against its MTEB headline of *"59.0"* on reasoning-intensive queries
[[BRIGHT24]](#src-bright24), and the dimension-bounded expressivity result sets a hard limit on what a single-vector
retriever can return at all [[LIMIT25]](#src-limit25) — both owned by M4. Hybrid retrieval helps with the *lexical*
half of that gap. It does not create the ability to reason (→ M10, M16).

> **Tradeoff (the ledger entry M7-4).** *Decision:* run both retrievers on every query rather than
> routing to one, and keep the fusion candidate set wide enough for the reranker to fix.
> *Costs:* the sparse index's build and maintenance, one extra retrieval per query, a wider
> candidate set passed downstream (→ M9, M18), and the tempting `num_queries > 1` default that
> silently inserts a generation call [[LI-SF]](#src-li-sf). *Accepted because:* the query's instrument is not
> determinable before the query runs, and mis-routing is a silent recall loss of exactly the kind
> this course keeps trying to price. *Reverses if:* the log shows identifier-shaped queries are
> deterministically detectable (a regex over clause IDs and UN numbers is a two-line feature) — at
> which point the measured routing gain in [[SEL21]](#src-sel21) becomes available and the always-run cost becomes
> waste.

---

## Worked example: BM25 vs dense vs hybrid on the 60-query set

Same corpus, same 187,043 nodes, same HNSW configuration as M5, filters from M6 applied. The
60-query set is partitioned into the five shapes above; dense is M4's chosen model, sparse is
`BM25Retriever` with the documented English stemmer and stopword defaults [[LI-BM25]](#src-li-bm25), fusion is
`QueryFusionRetriever` in `reciprocal_rerank` and `relative_score` modes [[LI-FUS]](#src-li-fus). Ground truth is
span-level, as in M3.

| Query subset | n | Dense recall@10 | BM25 recall@10 | RRF recall@10 | Relative score (α=0.6) | Best single |
|---|---|---|---|---|---|---|
| Semantic paraphrase | 20 | **0.94** | 0.58 | 0.92 | 0.93 | Dense |
| Identifier / exact token | 12 | 0.38 | **0.96** | 0.94 | 0.91 | **BM25** |
| Numeric / tabular | 10 | 0.31 | 0.44 | 0.41 | 0.40 | Neither (route) |
| Cross-reference / multi-hop | 8 | 0.52 | 0.35 | **0.58** | 0.55 | Fusion (weakly) |
| Mixed | 10 | 0.71 | 0.68 | **0.83** | 0.81 | Fusion |
| **Overall (60)** | 60 | 0.63 | 0.62 | **0.78** | 0.77 | Fusion |
| p99 latency (one retriever + fuse) | — | 24 ms | 8 ms | 31 ms | 34 ms | — |
| p99 with `num_queries=4` | — | — | — | **640 ms** | 660 ms | — |

Fusion-method sweep on the same 60 queries:

| Method | Constant / weight | Overall recall@10 |
|---|---|---|
| RRF | $k = 2$ [[QDRANT-HYBRID]](#src-qdrant-hybrid) | 0.78 |
| RRF | $k = 60$ | 0.75 |
| Relative score | α = 0.5 | 0.74 |
| Relative score | α = 0.6 | 0.77 |
| Relative score | α = 0.7 | 0.76 |
| Relative score | α = 0.3 | 0.70 |

Five findings decide LogiTrack's retrieval architecture:

1. **The two instruments are complementary, and their overall averages hide it.** Dense 0.63 and
   BM25 0.62 overall — nearly identical — while differing by 0.58 on the identifier subset and 0.36
   on paraphrase. A single aggregate number would have declared them interchangeable. **Report
   retrieval metrics per query shape, or you cannot see the decision.**
2. **Fusion beats both overall (0.78 vs 0.63/0.62) and still loses to BM25 alone on identifiers
   (0.94 vs 0.96).** This is the most important row in the table: hybrid is not uniformly better than
   its parts. A fused top-10 spends slots on dense's identifier misses that BM25 would have filled,
   which is the concrete form of the "hybrid does not always help" claim.
3. **The fusion constant and the weight are both real dials.** RRF moves 0.78 → 0.75 across $k = 2$
   → 60, and relative-score moves 0.70 → 0.77 across α = 0.3 → 0.6. The parameters are not noise, and
   the literature's claim that RRF is parameter-sensitive [[FU22]](#src-fu22) shows up as a 0.03 swing on 60
   queries — small, but the same size as several decisions this course has made deliberately.
4. **Query generation is a budget decision, not a retrieval decision.** `num_queries=4` takes p99
   from 31 ms to 640 ms — 25% of the 2.5 s end-to-end budget, spent before reranking or generation
   begins [[LI-SF]](#src-li-sf). It must be argued for on the M13 harness, not enabled by default (→ M10).
5. **Two subsets still need a different index, not a different fusion.** Identifier and numeric
   queries cap out below 0.6 for *every* fusion configuration in the table, which is the M3 finding
   arriving with better numbers: no combination of retrieval modes repairs a chunking or routing
   problem. The routing plan is M8's and M10's, and this table is its justification.

**LogiTrack's decision after M7:** sparse + dense with `QueryFusionRetriever` in
`reciprocal_rerank` mode, `num_queries=1` (query generation off until M10 argues otherwise), a
per-subset regression gate rather than a single overall recall number, and an explicit plan to route
identifier-shaped queries to a sparse-weighted path once a deterministic identifier detector exists.
The α/weight sweep is re-run whenever the embedding model changes (→ M4) or the fusion constant is
changed, because both move the answer.

---

## Design exercise

> *Paper-based. Think, then write.*

**Task.** Write the retrieval-mode ADR for a corpus you own — or use LogiTrack's. It must name, per
query shape, which instrument answers, what the fusion is, and what the routing rule is.

1. **Partition your query log by instrument, with counts.** Take 100 real queries and label each as
   identifier-shaped (exact tokens that must be matched), paraphrase (meaning without vocabulary
   overlap), numeric/tabular, multi-hop, or mixed. Report the proportions. If your log is 95%
   paraphrase, sparse retrieval is still worth having for the 5% — but you now know what you are
   paying for, and this count is the single most decision-relevant number in the module.
2. **Choose the sparse instrument and justify it against its cost.** BM25 (no model, no GPU, a
   corpus-specific vocabulary that must be rebuilt on change [[LI-BM25]](#src-li-bm25)) versus learned sparse
   (expansion, at index- and query-time inference cost, backend-dependent support) [[SPLADE21]](#src-splade21).
   State which identifier classes you expect each to recover.
3. **Decide the fusion method and the constant.** Rank-based or score-based, with the constant or
   weight *stated as a number* and a labelled sample if you claim a tuned value. If you choose RRF,
   state which $k$ your implementation defaults to, because implementations differ [[QDRANT-HYBRID]](#src-qdrant-hybrid)[[ES-RRF]](#src-es-rrf).
4. **Price the retrieval stage in latency.** Per-query: each retriever, the fusion, and — explicitly
   — whether query generation is on. `num_queries > 1` adds an LLM call inside retrieval [[LI-SF]](#src-li-sf);
   say what fraction of your end-to-end budget that consumes.
5. **Write the routing rule, or write down why you have none.** If identifier detection is
   deterministic on your queries (a regex over clause numbers, standard codes, part numbers), a
   route beats a blend on that subset [[SEL21]](#src-sel21). If you cannot detect them, say so — that is a finding
   about your query log worth recording.
6. **Set the acceptance test per subset, not in aggregate.** Recall@10 (and nDCG@10 once M9 exists)
   for each instrument band, plus the delta that would make you revert fusion. An overall average is
   the metric that hides this module's entire finding.

**Why this exercise matters.** Lexical-versus-dense is the oldest argument in retrieval and the one
most often settled by fashion: teams either rip out BM25 when they adopt embeddings, or bolt it back
on with an arbitrary weight after an incident. The worked example shows why neither is a decision:
the two instruments are near-identical in aggregate and complementary by subset, so the architecture
is a *routing and budgeting* question — which subsets, which fusion, which latency — and every one of
those sub-questions has a number attached. Note what the table refuses to conclude: it does not
report that hybrid wins, because on the identifier subset hybrid loses to BM25 alone. The senior
version of this decision is the one that can be read back a year later with the subset counts and
the constants still in it.

---

## Where the literature disagrees

- **"Hybrid reliably beats dense-only."** The case for hybrid is strong and comes from several
  directions: light hybrids *"achieve better generalization performance than individual sparse and
  dense retrievers"* and cut indexing memory 13× at 98% of a BM25+DPR hybrid's performance [[LITE22]](#src-lite22);
  BM25's robustness out of domain is a headline BEIR result [[BEIR21]](#src-beir21); and per-query strategy
  selection beats committing to one instrument [[SEL21]](#src-sel21). Against that, the worked example above has
  the fused system losing to BM25 alone on the identifier subset, and the entity-centric finding says
  why the dense half of any fusion contributes noise there [[ENTITY21]](#src-entity21). **What to change:** measure
  hybrid *per query band*, and accept that the honest claim is "hybrid is a better prior across a
  mixed query distribution", not "hybrid dominates dense". A team that adopts fusion by slogan will
  not notice the subset where it costs recall, because the aggregate will have improved.
- **"RRF requires no tuning."** Elasticsearch's documentation states it plainly: *"RRF requires no
  tuning, and the different relevance indicators do not have to be related to each other to achieve
  high-quality results"* [[ES-RRF]](#src-es-rrf). Bruch, Gai & Ingber report the opposite: *"Contrary to existing
  studies, we find RRF to be sensitive to its parameters; that the learning of a CC fusion is
  generally agnostic to the choice of score normalization; that CC outperforms RRF in in-domain and
  out-of-domain settings; and finally, that CC is sample efficient, requiring only a small set of
  training examples to tune its only parameter to a target domain"* [[FU22]](#src-fu22). The two claims are
  directly contradictory, and the second is a *contrary to existing studies* claim, i.e. the authors
  are explicitly disagreeing with the position the first states. The implementation landscape adds
  evidence for the disagreement: Qdrant documents $k = 2$ as its default [[QDRANT-HYBRID]](#src-qdrant-hybrid) while
  Elasticsearch presents the constant without choosing [[ES-RRF]](#src-es-rrf), so two systems calling themselves
  RRF are not necessarily computing the same thing. **What to change:** if you ship RRF, record the
  constant and the implementation, and if you have a labelled sample, spend it on a convex
  combination rather than trusting *no tuning* [[FU22]](#src-fu22).
- **"Score normalization is the fix for incomparable scores."** DBSF exists precisely because raw
  scores from different retrievers are not comparable, normalizing each retriever's returned set by
  its own mean and standard deviation so that *"different score magnitudes no longer matter"*
  [[QDRANT-HYBRID]](#src-qdrant-hybrid); relative score fusion makes the same move with a MinMax scaler [[LI-FUS]](#src-li-fus). Bruch et
  al. report that *"the learning of a CC fusion is generally agnostic to the choice of score
  normalization"* [[FU22]](#src-fu22) — which cuts the other way: if the learned combination does not care how you
  normalize, then normalization is not where the gain is, and the weight is. **What to change:** treat
  normalization as hygiene and the weight as the decision; spend measurement effort on α rather than
  on choosing between MinMax and z-score.
- **"Dense retrieval subsumes lexical retrieval."** DPR's in-domain result — a dense retriever
  beating a strong Lucene-BM25 baseline by 9–19% absolute in top-20 passage accuracy [[DPR20]](#src-dpr20) — was
  the origin of the subsumption claim, and it is a real result on its benchmarks. It coexists with
  BEIR's out-of-domain finding that dense models often underperform a robust BM25 [[BEIR21]](#src-beir21), with the
  entity-centric benchmark that documents dense failure on named-entity questions [[ENTITY21]](#src-entity21), and
  with the dimensional limit showing that some query→document-set mappings cannot be represented by
  a single vector at any affordable dimension [[LIMIT25]](#src-limit25). **What to change:** scope every "dense
  beats lexical" claim to the distribution it was measured on. All four claims are true; they are
  statements about different corpora, and your corpus is a fifth.

---

## Sources

- <a id="src-li-qf"></a>[Query Fusion (reciprocal rerank fusion)](https://developers.llamaindex.ai/python/examples/retrievers/reciprocal_rerank_fusion/) — `[LI-QF]` · owns the `QueryFusionRetriever` construction, the multi-query fusion behaviour and its de-duplication guarantee.
- <a id="src-li-sf"></a>[Simple Fusion Retriever](https://developers.llamaindex.ai/python/examples/retrievers/simple_fusion/) — `[LI-SF]` · owns the `num_queries` semantics, the documented equivalence of `num_queries=1` to disabling query generation, and the default query-generation prompt.
- <a id="src-li-bm25"></a>[BM25 Retriever](https://developers.llamaindex.ai/python/framework/integrations/retrievers/bm25_retriever/) — `[LI-BM25]` · owns `BM25Retriever.from_defaults` (nodes/docstore/index), disk persistence, and the `stemmer`/`language` defaults and their stopword-and-stemming role.
- <a id="src-li-fus"></a>[Relative Score and Distribution-Based Score Fusion](https://developers.llamaindex.ai/python/framework/integrations/retrievers/relative_score_dist_fusion/) — `[LI-FUS]` · owns `QueryFusionRetriever`'s `mode` values (`reciprocal_rerank`, `relative_score`, `dist_based_score`), `retriever_weights`, and `use_async`.
- <a id="src-li-pgv"></a>[Postgres Vector Store](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/postgres/) — `[LI-PGV]` · owns `hybrid_search=True`, `vector_store_query_mode` (`"default"`/`"sparse"`/`"hybrid"`), `VectorStoreQueryMode`, and `sparse_top_k` defaulting to `similarity_top_k`.
- <a id="src-rob94"></a>*Okapi at TREC-3* — S. E. Robertson, S. Walker, S. Jones, M. M. Hancock-Beaulieu, M. Gatford · NIST TREC-3 proceedings, p. 109 (1994) · [trec.nist.gov/pubs/trec3/t3_proceedings.html](https://trec.nist.gov/pubs/trec3/t3_proceedings.html) — `[ROB94]` · owns the original Okapi/BM25 system description and its TREC-3 evaluation. The paper itself is distributed as PostScript ([city.ps.gz](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz)); the venue and author list were verified from the official proceedings index, not from the PostScript body.
- <a id="src-bm25"></a>*The Probabilistic Relevance Framework: BM25 and Beyond* — Stephen Robertson, Hugo Zaragoza · Foundations and Trends in Information Retrieval 4(1–2):1–174 (2009) · DOI [10.1561/1500000019](https://doi.org/10.1561/1500000019) — `[BM25]` · owns the probabilistic relevance framework that BM25 formalises. **Not on arXiv**; verified from the publisher's Crossref record.
- <a id="src-irbook09"></a>*Introduction to Information Retrieval*, Chapter 11 (Probabilistic information retrieval) — Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze · Cambridge University Press (2008; online edition 2009) · [nlp.stanford.edu/IR-book/pdf/11prob.pdf](https://nlp.stanford.edu/IR-book/pdf/11prob.pdf) — `[IRBOOK09]` · owns the Okapi BM25 scoring form (eq. 11.32) quoted in this module, the roles of `k1` and `b`, and the statement that reasonable values are *"k1 and k3 to a value between 1.2 and 2 and b = 0.75"*.
- <a id="src-splade21"></a>*SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking* — Thibault Formal, Benjamin Piwowarski, Stéphane Clinchant · arXiv:2107.05720 (SIGIR 2021) · [arxiv.org/abs/2107.05720](https://arxiv.org/abs/2107.05720) — `[SPLADE21]` · owns learned sparse retrieval with explicit sparsity regularization, log saturation, and inverted-index compatibility.
- <a id="src-splade22"></a>*From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective* — Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant · arXiv:2205.04733 (SIGIR 2022) · [arxiv.org/abs/2205.04733](https://arxiv.org/abs/2205.04733) — `[SPLADE22]` · owns the finding that sparse models benefit from the same training improvements (distillation, hard negatives) as dense models.
- <a id="src-m3emb24"></a>*M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* — Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu · arXiv:2402.03216 (2024) · [arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216) — `[M3EMB24]` · owns the single model that produces dense, multi-vector, and sparse retrieval representations.
- <a id="src-entity21"></a>*Simple Entity-Centric Questions Challenge Dense Retrievers* — Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, Danqi Chen · arXiv:2109.08535 (EMNLP 2021) · [arxiv.org/abs/2109.08535](https://arxiv.org/abs/2109.08535) — `[ENTITY21]` · owns the entity-centric failure of dense retrievers, which is the mechanism behind this module's identifier-query subset.
- <a id="src-lite22"></a>*A Study on the Efficiency and Generalization of Light Hybrid Retrievers* — Man Luo, Shashank Jain, Anchit Gupta, Arash Einolghozati, Barlas Oguz, Debojeet Chatterjee, Xilun Chen, Chitta Baral, Peyman Heidari · arXiv:2210.01371 (ACL 2023) · [arxiv.org/abs/2210.01371](https://arxiv.org/abs/2210.01371) — `[LITE22]` · owns the light-hybrid memory/performance result and the generalization comparison against individual retrievers.
- <a id="src-sel21"></a>*Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection* — Negar Arabzadeh, Xinyi Yan, Charles L. A. Clarke · arXiv:2109.10739 (2021) · [arxiv.org/abs/2109.10739](https://arxiv.org/abs/2109.10739) — `[SEL21]` · owns per-query retrieval-strategy selection and its cost/utility result.
- <a id="src-rrf09"></a>*Reciprocal rank fusion outperforms condorcet and individual rank learning methods* — Gordon V. Cormack, Charles L A Clarke, Stefan Buettcher · Proceedings of the 32nd international ACM SIGIR conference, pp. 758–759 (2009) · DOI [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114) — `[RRF09]` · owns reciprocal rank fusion as a rank-combination method.
- <a id="src-fu22"></a>*An Analysis of Fusion Functions for Hybrid Retrieval* — Sebastian Bruch, Siyu Gai, Amir Ingber · arXiv:2210.11934 (2022) · [arxiv.org/abs/2210.11934](https://arxiv.org/abs/2210.11934) — `[FU22]` · owns the convex-combination-versus-RRF analysis, RRF's parameter sensitivity, the normalization-agnostic finding, and the sample-efficiency result.
- <a id="src-es-rrf"></a>[Elasticsearch reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) — `[ES-RRF]` · owns the RRF formula with its ranking constant and the *"RRF requires no tuning"* statement (surveyed in M5).
- <a id="src-qdrant-hybrid"></a>[Qdrant — Hybrid Queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) — `[QDRANT-HYBRID]` · owns the RRF formula with $k$ defaulting to 2, and DBSF's three-sigma normalization and its v1.11.0 introduction (surveyed in M5).
- <a id="src-weaviate-hybrid"></a>[Weaviate — Hybrid search](https://docs.weaviate.io/weaviate/search/hybrid) — `[WEAVIATE-HYBRID]` · owns the `alpha` fusion weight and score explanation (surveyed in M5).
- <a id="src-dpr20"></a>*Dense Passage Retrieval for Open-Domain Question Answering* — Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih · arXiv:2004.04906 (EMNLP 2020) · [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) — `[DPR20]` · owns dense retrieval's in-domain advantage over tuned BM25 (owned and surveyed by M4; cited here, not re-surveyed). · mirrored for local self-containment; **M4** owns the survey.
- <a id="src-beir21"></a>*BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models* — Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych · arXiv:2104.08663 (NeurIPS 2021) · [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) — `[BEIR21]` · owns the out-of-domain robustness of BM25 and the generalization gap of dense retrievers (owned and surveyed by M4; cited here). · mirrored for local self-containment; **M4** owns the survey.
- <a id="src-limit25"></a>*On the Theoretical Limitations of Embedding-Based Retrieval* — Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee · arXiv:2508.21038 (ICLR 2026) · [arxiv.org/abs/2508.21038](https://arxiv.org/abs/2508.21038) — `[LIMIT25]` · owns the dimension-bounded expressivity limit of single-vector retrieval (owned and surveyed by M4; cited here as the ceiling on the dense instrument).
- <a id="src-bright24"></a>*BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval* — Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan Shi, Zachary S. Siegel, Michael Tang, Ruoxi Sun, Jinsung Yoon, Sercan O. Arik, Danqi Chen, Tao Yu · arXiv:2407.12883 (2024) · [arxiv.org/abs/2407.12883](https://arxiv.org/abs/2407.12883) — `[BRIGHT24]` · owns the leaderboard-leader collapse on reasoning-intensive retrieval (owned and surveyed by M4; cited here).

**Unsupported (synthetic, or not verified in this module):** the LogiTrack corpus, its query log and
its five query-shape proportions, the 60-query evaluation set, every recall, latency and sweep figure
in the worked-example tables, the identifier examples (`14.2(b)`, `UN3480`, `CMR-9` and their
shipping-rule differences), and all tradeoff-ledger entries are constructed for teaching and are not
empirical findings. The IDF arithmetic for `UN3480` and the query-count weighting behind the overall
rows are derived in-file from the stated corpus size. **Not verified:** the internal parameters of
`BM25Retriever` beyond the documented `stemmer`/`language` defaults — the module does not assert
which `k1`/`b` values the implementation uses, and states only that the textbook's reasonable range
is 1.2–2 and 0.75 [[IRBOOK09]](#src-irbook09); if you need the implementation's values, read them off your build. The
claim that stemming and stopword removal can damage identifier recall is an argued mechanism, not a
measured result. **Thinly documented, flagged in place:** `BaseSparseEmbedding` was observed in
integration examples but no module-guide page for it was located in this module's doc fetches, so
learned-sparse support is treated as backend-dependent rather than as a stable framework surface.
**Ownership note:** dense-versus-sparse effectiveness claims are owned by M4 [[DPR20]](#src-dpr20)[[BEIR21]](#src-beir21), the
dimensional limit by M4 [[LIMIT25]](#src-limit25), and reranking of the fused candidate set by M9; this module cites
those keys and does not re-survey them.

**Verification note:** 🟡 *pending* — Step 7 (memory-less logic review) and Step 8 (literature
survey) not yet run on this module.

**Next module:** [M8 — Index Structures Beyond Flat Vectors](../08-index-structures-beyond-flat/README.md)
— the two subsets this module could not rescue (numeric and cross-reference), and the index shapes
that exist because a top-*k* over one flat vector index is not the right data structure for them.
