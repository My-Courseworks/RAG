# Vector Database and Tooling Map

> **Owner:** [M5 — Vector Databases and ANN Indexes](../modules/05-vector-databases-and-ann/README.md)
> **Purpose:** the landscape behind M5's selection decision. Every capability and license claim here
> was verified against the project's own repository or documentation during authoring; the source
> link is on the entry. Where a claim could not be verified, it says so instead of guessing.
> **Verified:** authoring baseline (2025-10). Licenses and capabilities move — re-verify before
> citing in a design review.

---

## How to read this map

The question "which vector database?" is usually asked as a feature comparison and answered as a
popularity contest. M5 reframes it: a vector store is **operational infrastructure**, and the axes
that actually decide the choice are the ones that cost you money or money-shaped things —
engineering time, blast radius, and lock-in. Popularity is not one of those axes.

Four axes do most of the deciding:

1. **Managed vs self-hosted** — who carries the pager at 03:00.
2. **License** — not a legal formality; it decides whether the *specific feature you need* is
   available to you without a commercial agreement.
3. **Filter expressiveness and pre-filtering** — whether metadata filtering preserves recall, and
   how expressive the filter language is (→ M6).
4. **Hybrid support and the cost model** — whether you can do sparse+dense in one system (→ M7),
   and whether you pay for provisioned capacity or consumed work.

---

## The landscape

| Engine | Delivery | License (verified) | Hybrid / lexical | Notable for |
|---|---|---|---|---|
| **Pinecone** | Managed only (SaaS; BYOC option) | Proprietary (closed source; client SDKs Apache-2.0) | Yes — sparse vectors + full-text search | Zero-ops; serverless and pod-based index modes |
| **Weaviate** | Self-hosted + managed cloud | **Split:** BSD-3-Clause outside `wl/`, enterprise license inside `wl/` | Yes — BM25 + vector, native hybrid | Hybrid search as a first-class query mode |
| **Qdrant** | Self-hosted + managed cloud | Apache-2.0 | Yes — sparse vectors, plus full-text filtering | Rust implementation; rich filter payloads |
| **Milvus** | Self-hosted + managed (Zilliz) | Apache-2.0 | Yes — sparse vectors | Scale-out architecture; multiple index types |
| **pgvector** | PostgreSQL extension | PostgreSQL License (permissive, BSD-like) | Via Postgres FTS (`tsvector`) alongside | No new infrastructure if you already run Postgres |
| **OpenSearch** | Self-hosted + managed | Apache-2.0 | Yes — native, it *is* a search engine first | Lexical search is the core competency, vectors added |
| **Elasticsearch** | Self-hosted + managed | **Triple:** AGPL-3.0-only, SSPL-1.0, or Elastic License 2.0 | Yes — native BM25 + vectors | Same: search-engine-first |
| **Chroma** | Embedded + server | Apache-2.0 | Limited (no native BM25 to the same depth) | Fastest path from zero to a working index |
| **LanceDB** | Embedded (library) + cloud | Apache-2.0 | Yes — full-text search alongside | Embedded, columnar (Lance) format; no server to run |
| **Vespa** | Self-hosted + managed cloud | Apache-2.0 | Yes — strong, it is a serving engine first | Very large-scale ranking and serving features |
| **Turbopuffer** | Managed only (SaaS) | Proprietary (closed source) | Yes — BM25 full-text + vector + filters, combinable | Object-storage-native; cost model for large cold corpora |

### License verification notes

These are the details that a feature matrix usually gets wrong:

- **Weaviate is not simply "BSD-3-Clause."** The repository states that source is "variously
  licensed under the BSD-3-Clause License, or the Weaviate License," with the deciding factor
  being location: outside the `wl/` directory it is BSD-3-Clause, and within `wl/` and its
  subdirectories it "is only available under a separate enterprise license," with the BSD license
  explicitly not granting "the right to use these features or circumvent any license key"
  ([weaviate/weaviate LICENSE](https://github.com/weaviate/weaviate/blob/main/LICENSE)). The
  operational consequence: **check whether the feature you need lives in `wl/` before you
  architect around it.**
- **Elasticsearch is not simply "open source."** The repository license file states that source is
  covered by "(i) a triple license under the 'GNU Affero General Public License v3.0 only', 'the
  Server Side Public License, v 1', and the 'Elastic License 2.0', or (ii) an 'Apache License 2.0'
  compatible" license depending on the file
  ([elastic/elasticsearch LICENSE.txt](https://github.com/elastic/elasticsearch/blob/main/LICENSE.txt)).
  AGPL and SSPL both carry obligations that many enterprises will not accept for a modified
  deployment, and the Elastic License 2.0 restricts offering the software as a managed service.
- **OpenSearch is the clean-room fork** and is Apache-2.0
  ([opensearch-project/OpenSearch LICENSE.txt](https://github.com/opensearch-project/OpenSearch/blob/main/LICENSE.txt)),
  which is why it is frequently the licence-driven choice where Elasticsearch's tri-license is
  unacceptable.
- **Apache-2.0 engines:** [Qdrant](https://github.com/qdrant/qdrant/blob/master/LICENSE),
  [Milvus](https://github.com/milvus-io/milvus/blob/master/LICENSE),
  [Chroma](https://github.com/chroma-core/chroma/blob/main/LICENSE),
  [LanceDB](https://github.com/lancedb/lancedb/blob/main/LICENSE),
  [Vespa](https://github.com/vespa-engine/vespa/blob/master/LICENSE).
- **pgvector** is under the PostgreSQL License
  ([pgvector LICENSE](https://github.com/pgvector/pgvector/blob/master/LICENSE)), the same
  permissive BSD-style licence as PostgreSQL itself — which is what makes it viable in
  organisations that have a policy against adding a new datastore.
- **Pinecone and Turbopuffer are managed, closed-source services.** Their *client SDKs* are
  open source (Pinecone's Python client is Apache-2.0:
  [pinecone-python-client LICENSE](https://github.com/pinecone-io/pinecone-python-client/blob/main/LICENSE)),
  but the engines are not. With these, your exit cost is a data-export and re-index project, not
  a `docker pull` — which belongs in the ledger entry, not in the footnotes.

---

## The cost-model axis, which the feature matrices omit

Two structurally different cost models are in play, and they dominate at the extremes of corpus
size and query volume:

| Model | How you are billed | Favours | Risk |
|---|---|---|---|
| **Provisioned capacity** | You pay for RAM/CPU/replicas reserved, whether or not queries arrive | Steady, high query volume; predictable load | Idle capacity is pure waste; scaling down has lead time |
| **Consumption / serverless** | You pay for stored vectors and for queries/reads performed | Spiky or low traffic; large cold corpora | Cost is a function of your own traffic growth — budget surprises scale with success |
| **Object-storage-native** | Storage is cheap (object storage); compute is cache | Very large corpora with long-tail access | Cold reads are slow; you must engineer around the cache |

**Turbopuffer's own documentation is unusually candid about this tradeoff**, and it is worth
quoting because it is exactly the kind of statement most vendors omit: it describes its tradeoffs
for "excellent economics and scalability" as "higher write latency (p90=248ms for 512KB upserts)
from writing directly to object storage, and occasional cold queries for uncached or unpinned
data (p90=1214ms on 1M documents)"
([turbopuffer docs — Introduction](https://turbopuffer.com/docs)). It also documents >90% recall
for vector search combineable with filters, and dense+sparse support in the same query.

Read that against LogiTrack's spine: a **P95 end-to-end latency budget of 2.5 s** and a
**15-minute freshness SLA**. A p90 cold query of 1.21 s consumes half the entire budget in one
hop, and a p90 upsert of 248 ms is fine for freshness but shapes how you batch. This is why the
cost model belongs in M5's decision table rather than in a footnote: the storage economics that
make object-storage-native attractive are purchased with a latency distribution that may not fit
your SLO.

> **Tradeoff (the ledger entry M5-MAP-1).** *Decision:* choose the store by its cost model and
> its license-fit before its feature checklist. *Costs:* you may accept a system with fewer
> features, or pay more per query, to avoid a provisioning or licence trap. *Accepted because:*
> features are portable across engines with a rewrite; a data-residency or licence violation is
> not, and a cost model that inverts under growth is a re-platforming project. *Reverses if:*
> a required capability (e.g. native hybrid search) exists in exactly one engine that passes your
> other two filters — then the feature does the deciding.

---

## Which tool for which job

A map is only useful if it recommends. These are decision-shaped, not ranking-shaped:

| If your binding constraint is… | Reach for … | Because |
|---|---|---|
| "We already run Postgres and cannot add a datastore" | **pgvector** | Same backup, auth, and ops story; permissive licence; a real FTS path exists in Postgres for hybrid (→ M7) |
| "Zero ops, and we will pay for it" | **Pinecone** | Fully managed, with a documented BYOC path if data residency becomes binding (→ M17) |
| "Hybrid search is the core requirement" | **Weaviate**, **Elasticsearch/OpenSearch**, or **Vespa** | Lexical+dense fusion is native rather than bolted on — but check the Weaviate `wl/` boundary and the Elastic tri-license first |
| "Self-hosted, permissive licence, strong payload filtering" | **Qdrant** | Apache-2.0 with rich filter payloads (→ M6) |
| "Embedded, no server process at all" | **LanceDB**, **Chroma** | Library-first; the right call for tests, local dev, and small corpora — and a liability pattern if it quietly becomes production |
| "Billion-scale, self-hosted, multiple index types" | **Milvus**, **Vespa** | Built for scale-out; you are now running a distributed system |
| "Very large corpus, long-tail access, cost-dominated" | **Turbopuffer** | Object-storage-native economics — priced in cold-read latency, so verify against your SLO |
| "Search relevance is the product" | **OpenSearch**, **Elasticsearch**, **Vespa** | You want a search engine that added vectors, not a vector store that added search |

**The two-question test.** Before any of this: *(1) Is our corpus large enough that ANN is
required at all?* Below roughly a few hundred thousand vectors, exact search on a well-provisioned
instance is often fast enough, and the entire ANN machinery — recall tuning, quantization,
parameter sweeps — is complexity you can defer (→ M5, M18). *(2) Is our binding constraint
capability or operations?* If capability, compare features. If operations, compare licences and
cost models first, because those are the ones that fail late.

---

## Deliberately out of scope

- **Graph databases and graph-RAG stacks** (Neo4j and similar) — the index-side decision is M8's;
  the graph-vs-flat tradeoff is M16's.
- **Document parsing and OCR platforms** (including LlamaParse, LlamaIndex's own parsing
  platform, and open alternatives) — owned by **M3**, which is where the parse-then-chunk
  ordering is decided.
- **Embedding model vendors** — the model-selection decision is **M4**'s.
- **Reranking providers** — **M9**'s.
- **Managed RAG platforms that bundle ingest+index+serve** — M22's build-vs-buy discussion, since
  choosing one is an architecture decision rather than a component choice.

---

## Sources

- [Qdrant LICENSE](https://github.com/qdrant/qdrant/blob/master/LICENSE) — Apache-2.0, verified.
- [Weaviate LICENSE](https://github.com/weaviate/weaviate/blob/main/LICENSE) — split BSD-3-Clause / enterprise (`wl/`), verified.
- [Milvus LICENSE](https://github.com/milvus-io/milvus/blob/master/LICENSE) — Apache-2.0, verified.
- [Chroma LICENSE](https://github.com/chroma-core/chroma/blob/main/LICENSE) — Apache-2.0, verified.
- [LanceDB LICENSE](https://github.com/lancedb/lancedb/blob/main/LICENSE) — Apache-2.0, verified.
- [Vespa LICENSE](https://github.com/vespa-engine/vespa/blob/master/LICENSE) — Apache-2.0, verified.
- [pgvector LICENSE](https://github.com/pgvector/pgvector/blob/master/LICENSE) — PostgreSQL License, verified.
- [OpenSearch LICENSE.txt](https://github.com/opensearch-project/OpenSearch/blob/main/LICENSE.txt) — Apache-2.0, verified.
- [Elasticsearch LICENSE.txt](https://github.com/elastic/elasticsearch/blob/main/LICENSE.txt) — triple AGPL-3.0 / SSPL-1.0 / Elastic License 2.0, verified.
- [Pinecone Python client LICENSE](https://github.com/pinecone-io/pinecone-python-client/blob/main/LICENSE) — Apache-2.0 (client only; the service is proprietary), verified.
- [Pinecone documentation](https://docs.pinecone.io/guides/get-started/overview) — capability and deployment-mode reference.
- [turbopuffer documentation](https://turbopuffer.com/docs) — architecture, latency tradeoffs, and recall claims, quoted above.
- [LlamaIndex vector store integrations](https://developers.llamaindex.ai/python/framework/module_guides/storing/index.md) — the framework side of connecting any of these (→ M5's `VectorStoreIndex.from_vector_store`).

**Unsupported / unverified:** license *versions* and pricing figures are deliberately absent —
both change, and a stale number in a design document is worse than no number. The LogiTrack
latency and freshness figures referenced above are course fiction (syllabus §2). The "two-question
test" thresholds are stated as practical guidance, not measured results; no source establishes a
universal corpus size below which ANN is unnecessary, because it depends on hardware, dimension,
and query concurrency.
