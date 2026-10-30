# M14 · Ingestion at Scale, Incremental Sync, and Deletion

> **Module question:** How do you keep a 480k-document index fresh without rebuilding it?
> **Cross-cutting threads:** Failure modes · Tradeoff ledger · LlamaIndex at a glance
> **Domain spine:** LogiTrack — 3,000 → 480,000 documents over twelve months, and a carrier agreement revision that must supersede its predecessor within 15 minutes
> **Prerequisites:** [M13 — Evaluation of Retrieval and Generation](../13-evaluation/README.md), [M3 — Corpus Acquisition, Parsing, and the Chunking Decision](../03-parsing-and-chunking/README.md)

---

## Opening scene — two versions of clause 4.2, both live

Maersk files a revised service agreement at 14:02 on a Tuesday; clause 4.2 changes the DG
documentation surcharge from 3.5% to 4.1% effective 1 October. LogiTrack's ingest runs on a
nightly cron, so the revision is picked up at 02:00 — twelve hours later, the staleness window
the team knows about and has priced.

What nobody priced is what the revision *does*. A new PDF with a new filename enters as a new
document with a new `doc_id`; its 41 pages parse into 312 chunks that are near-duplicates of
the previous agreement's for every clause except 4.2 and two others, because a revised contract
is 97% the same text. A broker asks about the DG surcharge at 09:40 and gets **3.5%**, cited to
the *superseded* agreement, which the retriever prefers because its embedding is marginally
closer to the query wording. The answer is faithful (→ M12), the citation is real, the
retrieval is correct by every metric the harness computes (→ M13), and the number stopped being
true twelve hours ago.

Root cause is not the cron. The pipeline has exactly one incremental operation — *insert* — and
the corpus needs four: insert, update, dedup, and **supersede**. Without supersession every
revision permanently doubles the corpus for the clauses it touches, and the evaluation set
silently measures a corpus with two conflicting truths in it. The M13 ground-truth mappings for
a dozen queries now point at the retired document, and the recall@10 number they contribute
measures nothing at all.

> **Failure mode (the module in one line):** treating ingestion as a one-way flow from corpus to
> index rather than as **a sync problem with four operations and a staleness budget** — so
> revisions accumulate as duplicates, deletions never propagate, and every downstream metric is
> computed against an index that no longer matches the corpus it claims to represent.

---

## Staleness is a latency, not a boolean

**Staleness window** $= t_{\text{served}} - t_{\text{corpus changed}}$, where
$t_{\text{served}}$ is the first moment the index answers a query using the new content.
LogiTrack's budget is P95 ≤ **15 minutes** [syllabus]. Every component of the loop contributes,
and the contributions are additive — which is why a nightly cron with a 12-hour window is not
"slow", it is *100× over budget*, and why shaving the parser is irrelevant until the schedule
changes.

| Contributor to the window | Typical lever | Cost of shrinking it |
|---|---|---|
| Detection lag (how you learn the document changed) | Poll interval → webhook/CDC | Webhook infrastructure per source system; poll granularity bounded by the rate limit of the *source*, not the pipeline |
| Queue lag (how long it waits for a batch) | Batch size → per-document processing | Per-document processing multiplies per-request overhead; batching trades freshness for throughput |
| Transform lag (parse, chunk, extract metadata, embed) | `num_workers`, concurrent embedding calls | Provider rate limits (→ M18); CPU for local extractors |
| Index-write lag (embedding → queryable) | Vector-store upsert visibility; some stores index asynchronously | Engine-specific; the M5 selection already gated on this |
| **Supersession lag** (old content stops being served) | Explicit delete or filter on effective date | A delete touches four stores plus backups; a filter is cheap but must be applied to *every* query path (→ M6) |

The last row is the one teams forget, and it is structurally different: the first four rows
make new content *arrive*, the fifth makes old content *stop*. A pipeline can be perfectly
fresh and still wrong — freshness is a property of the pair (new present, old absent).

---

## The pipeline and its cache

LlamaIndex's `IngestionPipeline` is the abstraction for the transform half of the loop. It
"uses a concept of `Transformations` that are applied to input data", and "the resulting nodes
are either returned or inserted into a vector database (if given)"; the canonical construction
chains a node parser, extractors, and an embedding model [LI-IP].

Three documented properties decide the design:

1. **The cache is keyed on content plus transformation.** "Each node + transformation pair is
   cached, so that subsequent runs (if the cache is persisted) with the same
   node+transformation combination can use the cached result and save you time" [LI-IP]. It can
   be persisted and reloaded — `pipeline.persist("./pipeline_storage")` and
   `new_pipeline.load(...)`, after which the second run "will run instantly due to the cache"
   [LI-IP] — or kept remote, with `RedisCache`, `MongoDBCache`, and `FirestoreCache` documented
   as supported backends and `IngestionCache(cache=RedisCache.from_host_and_port(...),
   collection=...)` wiring one in [LI-IP]. The remote path benefits because "no persist step is
   needed, since everything is cached as you go" [LI-IP].
2. **Embeddings must be a pipeline stage if you write to a vector store.** "If you are
   connecting your pipeline to a vector store, embeddings must be a stage of your pipeline or
   your later instantiation of the index will fail. You can omit embeddings from your pipeline
   if you are not connecting to a vector store" [LI-IP].
3. **Parallelism is process-based.** "The `run` method of `IngestionPipeline` can be executed
   with parallel processes. It does so by making use of `multiprocessing.Pool` distributing
   batches of nodes to across processors", enabled with `pipeline.run(documents=[...],
   num_workers=4)`; and "the `IngestionPipeline` also has support for async operation" via
   `await pipeline.arun(documents=documents)` [LI-IP].

> **LlamaIndex at a glance:** `IngestionCache` is the module's primitive, and its correctness
> property is the one to internalise: **the cache key is content-derived, not
> transformation-version-derived.** The docs say the key is the node + transformation combination
> [LI-IP]. A chunk-*parser* code change, an extractor prompt change, or a re-tuned embedding model
> therefore does not change the input node's content hash, so the cache can serve the previous run's
> output — and the docs do not specify an invalidation policy for either case. Treat cache
> invalidation as a first-class release step: bump the cache collection name on any transformation
> change, so a stale-cache class of bug cannot reach production at all.

---

## Docstore upsert semantics: what "we already have it" means

The docstore is what makes incremental ingestion safe, and its semantics are documented
precisely enough to design against. "Attaching a `docstore` to the ingestion pipeline will
enable document management. Using the `document.doc_id` or `node.ref_doc_id` as a grounding
point, the ingestion pipeline will actively look for duplicate documents", by "storing a map of
`doc_id` -> `document_hash`" [LI-IP]. The decision table is then exactly:

| State | With a vector store attached | Without a vector store |
|---|---|---|
| Duplicate `doc_id`, hash **changed** | "the document will be re-processed and upserted" [LI-IP] | Not detectable — see the caveat below |
| Duplicate `doc_id`, hash **unchanged** | "the node is skipped" [LI-IP] | Not detectable |
| New `doc_id` | Processed and inserted | Processed |
| `doc_id` absent from the input set | **Nothing happens** — no delete is inferred | Nothing happens |
| Duplicate input within one run | Deduped | Deduped: "Checks all existing hashes for each node. If a duplicate is found, the node is skipped. Otherwise, the node is processed" [LI-IP] |

The caveat is the single most consequential line on this page: "**NOTE:** If we do not attach a
vector store, we can only check for and remove duplicate inputs" [LI-IP]. Without a vector
store, the docstore's hash map can suppress a duplicate *input* and do nothing else — no
upsert, no supersession, no removal of previously ingested content. Such a pipeline silently
accumulates every version of every document it has ever seen, and the only symptom is a
retrieval corpus 1.9× its stated size.

Two further documented facts:

- **Document identity is not node identity.** `ref_doc_id` links a node back to its source
  document, and `index.ref_doc_info` returns a map from each ingested `doc_id` to a
  `RefDocInfo` carrying "their associated `node_ids` of the nodes they were split into" plus
  the original `metadata` [LI-DM]. That mapping is the deletion path — you cannot delete a
  document whose node ids you do not know.
- **Node ids are random by default**, as the retrieval-evaluation notebook states in a code
  comment: "by default, the node ids are set to random uuids. To ensure same id's per run, we
  manually set them" [LI-EVR2]. Re-ingesting produces *different* node ids: you cannot dedup
  nodes across runs by id, every node-id ground-truth mapping in M13's evaluation set is
  invalidated, and a delete targeting node ids from last week's log hits nodes that no longer
  exist. Set ids deterministically at load time — `filename_as_id=True` on
  `SimpleDirectoryReader` sets the document `doc_id` "to be the full path to each document"
  [LI-DOC] — and derive node ids from content plus a stable identifier, not from the clock.

> **Tradeoff (the ledger entry M14-1).** *Decision:* attach a docstore **and** a vector store to
> every ingestion pipeline, and set deterministic document and node identifiers at load time.
> *Costs:* a docstore to operate and keep in sync (an extra store in the backup and restore path),
> plus the discipline of an id scheme that must be stable across parser and model changes.
> *Accepted because:* the documented alternative — no vector store attached — cannot upsert or
> delete, only suppress duplicate inputs [LI-IP], which makes supersession and the right to be
> forgotten both unimplementable. *Reverses if:* the corpus is genuinely append-only and immutable
> (a frozen archive), where the docstore's hash map has nothing to do and its operational cost is
> pure overhead.

---

## The incremental-sync state machine

Incremental ingestion is a state machine over documents, not a function over files. The states
are observable, the transitions are the four operations the opening scene was missing, and the
failure modes attach to specific transitions.

```mermaid
stateDiagram-v2 [*] --> Discovered Discovered --> Hashed: fetch + parse + hash Hashed -->
Absent: doc_id not in docstore Hashed --> Unchanged: hash matches Hashed --> Changed: hash
differs Hashed --> Superseding: revision of a known doc Absent --> Embedding: insert path
Changed --> Embedding: upsert path Superseding --> Retiring: delete predecessor nodes Retiring
--> Embedding: insert successor nodes Unchanged --> [*]: skip (no embed cost) Embedding -->
Indexed: vector store upsert Indexed --> Verified: query the doc_id, assert content Verified
--> [*] Verified --> Quarantined: assertion fails Quarantined --> [*]: alert; predecessor stays
live Deleted --> Retiring: tombstone received note right of Superseding Requires an explicit
revision link. Filename or metadata version is the only signal the corpus gives you. end note
note right of Verified No transition may reach Verified by assumption; every path re-queries.
end note
```

Two transitions carry almost all of the risk.

**`Superseding → Retiring`.** A revision arrives as a new document. Nothing in an
`IngestionPipeline` knows that `Maersk-CSA-2024.pdf` and `Maersk-CSA-2025.pdf` are the same
contract, because the docstore's key is `doc_id` and the ids differ [LI-IP]. Supersession must
be *supplied*: load revisions under a stable `doc_id` (losing the old version) or carry an
explicit `supersedes: <doc_id>` chain and delete the predecessor. The choice is a ledger entry,
not an implementation detail:

| Design | Mechanism | Buys | Costs |
|---|---|---|---|
| **Stable `doc_id` per logical document** | Load every revision with the same `doc_id`; the hash change drives the upsert [LI-IP] | Nothing to reconcile; the docstore does the work | The superseded text is gone — no historical answer, no audit of what was said in September |
| **New `doc_id` + explicit supersession link** | `metadata["supersedes"] = predecessor_id`; a job deletes the predecessor | History retained; `effective_from`/`effective_to` become filterable (→ M6) | A reconciliation job that must itself be correct, idempotent, and monitored; two documents are briefly live simultaneously |
| **New `doc_id` + effective-date filter** | Both versions stay in the index; queries filter `effective_date <= today` (→ M6) | Cheapest write path; point-in-time queries for free | Every query path must apply the filter — a single unfiltered path serves stale content, and M6 established that filters are easy to forget at exactly one call site |

LogiTrack's answer is the middle row for contracts and the third for bulletins, because the two
sources have different temporal semantics: a contract revision *replaces* an obligation
(retrospective queries are rare and legally fraught), whereas a service bulletin *adds* to a
timeline (queries about last month's bulletin are routine).

**`Indexed → Verified`.** The step that makes the whole machine trustworthy, and the one teams
skip. "Indexed" is what the pipeline *believes*; "verified" is what a query *returns*. The
check is cheap — re-query by `doc_id`, assert the canonical chunk is present and its text hash
matches — and it is the only defence against the failure class where the pipeline reports
success while the vector store silently rejected the upsert (schema mismatch, dimension
mismatch, metadata type change, quota). The diagram encodes a design choice: `Quarantined` is
*terminal*, and the `note` says the predecessor stays live. A failed supersession must fail
*closed* — serving the old number one more cycle is a freshness incident; serving an unverified
new number is the compliance incident.

---

## Deletion, and the right to be forgotten

Deletion is documented, supported, and — read carefully — narrower than it looks.

```
index.delete_ref_doc("doc_id_0", delete_from_docstore=True)
```

"`delete_from_docstore` will default to `False` in case you are sharing nodes between indexes
using the same docstore. However, these nodes will not be used when querying when this is set
to `False` as they will be deleted from the `index_struct` of the index, which keeps track of
which nodes can be used for querying" [LI-DM]. Three consequences follow:

1. **The default leaves the text in the docstore.** `delete_from_docstore=False` removes the
   nodes from the index's queryable set, but the document text survives in the docstore. For a
   right-to-be-forgotten request that is the difference between compliance and non-compliance,
   and the default is the non-compliant option. You must pass `True` — deliberately, per call
   site.
2. **Deletion support is not uniform.** The same page states "**NOTE**: the tree index
   currently does not support deletion" [LI-DM]. Any architecture relying on a tree structure
   (→ M8, and RAPTOR-style hierarchies in M16) inherits a deletion gap closed only by rebuild,
   by a tombstone filter, or by accepting that the structure cannot be incrementally
   maintained.
3. **Deleting from the index is not deleting the document.** The framework's storage layer is
   explicitly decomposed — document stores, index stores, vector stores, property-graph stores, and
   chat stores, bound together by a `StorageContext` [LI-DS] — and the docstore and index store each
   have persistent backends that write as nodes are ingested [LI-DOCS][LI-IS]. That is why the
   inventory below has more rows than the deletion API has parameters, and why a deletion that misses
   one store did not happen. The inventory:

| Where the text lives | Reached by `delete_ref_doc`? | What you must do |
|---|---|---|
| Vector store nodes | Yes, with `delete_from_docstore=True` and a vector store attached [LI-IP][LI-DM] | Pass `True`; verify the count actually dropped |
| Docstore | Only with `delete_from_docstore=True` [LI-DM] | Same call, same flag |
| Index store (`index_struct`) | Removed from the queryable set [LI-DM] | Nothing further, but do not assume the bytes are gone |
| `IngestionCache` | **No** | Cached node+transformation outputs persist; a re-ingest of the deleted document can be served entirely from cache. The delivery mechanism is unaddressed by the docs; the mitigation available is `cache.clear()` [LI-IP] or a cache-scoped delete |
| Derived structures (summaries, graphs, trees) | Partly — trees do not support deletion [LI-DM] | Rebuild the derivation, or tombstone |
| Raw document store / S3 origin | No | Delete at the source of truth first; the index is a replica |
| Backups and snapshots | No | Retention policy is the only answer; this is a contracts problem, not a code problem |
| Logs, traces, and eval dumps | No | PII must not be logged, or log retention must be short enough to satisfy the obligation (→ M17, M19) |

The research answer is aligned with this reading. RAG-based unlearning proposes using the
external knowledge base *as* the forgetting mechanism — "a lightweight behavioral unlearning
framework based on Retrieval-Augmented Generation (RAG) technology. By modifying the external
knowledge base of RAG, we simulate the effects of forgetting without directly interacting with
the unlearned LLM" — favoured over weight-level methods that "often suffer from high
computational demands, limited applicability, or the risk of catastrophic forgetting"
[RAGUNLEARN24]. That supports deletion as an index operation, and says nothing about caches,
replicas, backups, or logs, where the inventory above actually lives. The paper's five criteria
are "effectiveness, universality, harmlessness, simplicity, and robustness" [RAGUNLEARN24];
none is *completeness across stores*, the criterion an auditor applies.

> **Tradeoff (the ledger entry M14-2).** *Decision:* implement deletion as an explicit,
> idempotent, verified job over an enumerated set of stores — vector store, docstore, index store,
> cache, derived structures — and keep the store inventory under version control as an artifact.
> *Costs:* the job must be maintained as the architecture grows (every new derived structure, cache,
> or store replica adds a row to the inventory), and verification costs a read per deletion.
> *Accepted because:* a partially complete deletion is indistinguishable from no deletion to an
> auditor and strictly worse than an acknowledged gap, because it produces a false assurance.
> *Reverses if:* the corpus contains no personal or client-confidential data, where the obligation
> reduces to hygiene and a count check on the primary store suffices.

---

## Backfills: the re-embedding project

Changing the embedding model — for a vendor upgrade, a domain fine-tune (→ M15), or a dimension
reduction (→ M4) — invalidates every vector in the index. There is no partial version of this
operation: a query embedded with model $B$ cannot be compared against vectors built with model
$A$, so the backfill is all-or-nothing and it is a *project*, with a budget, a schedule, and a
rollback plan.

Size it before you argue about it. LogiTrack's M1 corpus of 3,000 documents is ~13.5M words, so
the twelve-month target of 480,000 documents scales by $480{,}000/3{,}000 = 160\times$:

$$14\text{M words} \times 160 = 2.24\text{B words} \approx \frac{2.24\text{B}}{0.75} \approx 2.99\text{B tokens}$$

and with M3's overlap decision at $O/C = 0.20$, the stored/embedded volume is multiplied by
$\frac{1}{1-O/C} = 1.25$:

$$2.99\text{B} \times 1.25 \approx 3.74\text{B tokens per full re-embed}$$

The dollar figure is a single multiplication whose *answer depends entirely on a rate this
module does not source*, so state it parametrically:

| Illustrative rate $R$ (per 1M tokens) | Full re-embed at 3.74B tokens |
|---|---|
| \$0.01 | \$37,400 |
| \$0.02 | \$74,800 |
| \$0.13 | \$486,200 |

The rates above are **illustrative placeholders, not sourced prices** (see Unsupported). The
decision-relevant observation survives any rate: **a full re-embed is a
five-figure-to-six-figure project that lands against a monthly budget of \$18k, so it is always
a budgeted initiative and never an incident response.** It is also why M3's overlap entry is a
capacity decision: at $O/C = 0.20$ the backfill costs 25% more than it needs to, forever, on
every model change.

The operational pattern that makes a backfill survivable is **dual-write with an alias swap**:
build the new index alongside the old, keep ingesting into both, verify the new one against the
M13 harness, then flip the query path atomically and delete the old index after a retention
window. The properties that matter:

| Requirement | Why |
|---|---|
| Both indexes fed during the build | Otherwise the new index is stale on arrival and the swap creates a freshness incident |
| The M13 gate run against the *new* index before the swap | The backfill is the largest single change the system will ever receive; it must clear the harness that gates everything else |
| Ground-truth node ids re-derived | Node ids are random by default [LI-EVR2], so a re-embed invalidates every node-id mapping in the evaluation set (→ M13) |
| Cache namespaced per model | The `IngestionCache` key is content + transformation [LI-IP]; a model change is a transformation change, so the old cache must not be reachable |
| Atomic cutover, one-way rollback | A swap that can half-happen produces an index with two embedding spaces in it, which is the worst of all states |

---

## Parallel ingest, batching, and rate limits

The levers are few and they interact badly.

| Lever | Documented mechanism | Bound by | Failure mode when pushed |
|---|---|---|---|
| `num_workers` on `run` | `multiprocessing.Pool` "distributing batches of nodes to across processors" [LI-IP] | Local CPU; per-process memory; provider concurrency | Process-per-worker memory multiplies; local embedding models saturate the machine before the provider does |
| `arun` | Async pipeline run [LI-IP] | Python event loop; provider rate limits | Unbounded concurrency produces 429s and retry storms — the M13 notebook shows the framework's own retry path firing repeatedly under batch load [LI-EVR2] |
| Batch size | Fewer requests per token | Provider request-size caps | Requests too large are rejected wholesale; a batch is atomic, so one bad document can fail 999 good ones |
| Scheduling | Cron → continuous worker | Detection lag budget | A continuous worker with no idempotency re-ingests on every restart |
| Cache | Skip unchanged node+transformation pairs [LI-IP] | Cache correctness (above) | A stale cache silently serves old outputs at zero cost, which looks like a win |

Two invariants make the parallelism safe rather than merely fast:

1. **Every unit of work must be idempotent and keyed.** Because node ids are random by default
   [LI-EVR2], a retried batch produces different node ids and duplicates rather than
   deduplicating. Deterministic ids are a *precondition* for parallel ingest, not an
   optimisation of it.
2. **The batch boundary must be a transaction boundary you can observe.** With `num_workers=8`,
   the pipeline reports success for the run, not for a document. Without the `Indexed →
   Verified` transition, a partially failed run leaves the index in a state no one can
   describe, and the earliest symptom is a retrieval metric drifting down over three weeks (→
   M20).

> **Tradeoff (the ledger entry M14-3).** *Decision:* run continuous incremental sync with
> deterministic ids, idempotent per-document work, and a per-document verification read — accepting
> a lower ingest throughput than a parallel batch can achieve. *Costs:* roughly a doubling of ingest
> wall-clock and index read load, plus the code to make each document's processing retry-safe.
> *Accepted because:* at 480k documents the steady-state change rate is small (a few hundred
> documents a day), so throughput is not the binding constraint — freshness and correctness are,
> and both are damaged by batched nondeterminism. *Reverses if:* the corpus turns over fast enough
> that the daily change volume approaches the pipeline's per-document capacity, at which point
> batching with deterministic ids becomes mandatory and the verification read moves to a nightly
> sampled audit.

---

## Worked example: full rebuild vs four incremental designs

The lab is exactly this comparison: run both paths against a corpus with a known change set and
measure wall-clock, token spend, and the staleness window. LogiTrack's twelve-month target:
480,000 documents, ~2.24B words, with a daily change set of ~600 documents (carrier revisions,
bulletins, rate-table refreshes) plus ~4,000 new ops-wiki pages.

| Design | Wall-clock for daily delta | Tokens/day (delta) | Staleness window P95 | Supersession | Re-embed cost | Notes |
|---|---|---|---|---|---|---|
| Nightly full rebuild | 9 h 20 m | 3.74B (full corpus) | 12 h | Free — the old index is discarded | Full, every night | Cannot meet 15 min by construction; also re-embeds 480k docs to absorb 600 changed ones |
| Nightly rebuild + daily delta insert | 9 h 20 m + 11 m | 3.74B + 2.1M | 12 h (new content 11 m; *supersession* still 24 h) | None | Full, every night | The opening scene's design: new content lands quickly, old content never retires |
| Hourly incremental, no vector store attached | 6 m | 2.1M | 60 m | None | Incremental | Freshness improves, supersession does not — the docs are explicit that without a vector store you "can only check for and remove duplicate inputs" [LI-IP] |
| Continuous incremental, docstore + vector store, stable `doc_id` per contract | 40 s P95 | 2.1M | **8 m** | Implicit — supersession *is* the update path | Incremental | Meets 15 min with headroom. Loses superseded contract text |
| Continuous incremental + supersession links + verified writes | 1 m 10 s P95 | 2.1M | **8 m** (new content) / **11 m** (retired content) | Explicit delete job | Incremental | LogiTrack's choice for contracts; +0.5M tokens/day for the verification reads |

Four findings decide the design:

1. **Rebuilding nightly costs 1,780× the tokens of the incremental path to publish 600 changed
   documents** — 3.74B tokens versus 2.1M. Even at the cheapest rate above that is a
   five-figure monthly line item, and M18 prices it against a \$18k ceiling. The rebuild is not
   "safe"; it is *expensive enough to distort every other decision in the system*.
2. **Freshness and supersession are independent, and the naive incremental design fixes only
   one.** Compare rows 2 and 3: hourly incremental ingest cuts detection lag from 12 hours to
   60 minutes and leaves the opening scene's *actual* failure intact, because the failure was
   never detection lag. This is the module's central claim in one table row.
3. **The winning design is not the fastest, it is the one with a verification step.** Row 5 is
   slower than row 4 by 30 seconds per cycle and buys the only transition that distinguishes
   "the pipeline thinks it upserted" from "a query returns the new clause". Retired content (11
   m) is slightly *worse* than new content (8 m) because deletion is a second job — and that
   asymmetry is the honest number to report, because the compliance risk lives on the
   retirement side.
4. **Detecting the change is now the bottleneck.** With ingest at 8 minutes P95 against a
   15-minute budget, the remaining 7 minutes are the source system's notification path.
   Optimising the parser further buys nothing; this is where M19's instrumentation should
   point.

**LogiTrack's decision after M14:** continuous incremental sync with a docstore and vector
store attached, deterministic document and node ids set at load, supersession links for
contracts and effective-date metadata for bulletins, a per-document verification read after
every upsert, a store inventory under version control for the deletion job, and full re-embeds
run as dual-write projects gated by the M13 harness.

---

## Design exercise

> *Paper-based. Think, then write. Deliver a decision, not a survey.*

**Task.** Write the ingestion and sync ADR for a corpus you own, or for LogiTrack's
twelve-month target: a state machine plus the four operations it must implement, with a
staleness budget on each transition.

1. **Set the staleness budget in two numbers, not one.** State the maximum age of *new* content
   and the maximum time *retired* content may remain served — different machinery, different
   values. A single number hides the retirement side, where the compliance risk sits.
2. **Enumerate the four operations for your corpus.** For each of insert, update, dedup, and
   supersede: what signal identifies it, what function implements it, and what happens on the
   day the signal is absent. "Absent" is the common case — a revised contract with a new
   filename carries no supersession signal — and if you cannot name the signal you do not have
   the operation.
3. **Choose the identity scheme and justify it against node-id randomness.** State the document
   `doc_id` scheme and the node id scheme explicitly, then what breaks when you re-ingest:
   which ids change, which stored mappings (evaluation ground truth, debug logs, deletion jobs)
   become invalid, and how they are re-derived. If your answer is "the ids are stable", say
   why, because the framework's default is random UUIDs [LI-EVR2].
4. **Draw the state machine and mark the fail-closed transitions.** Include discovery, hashing,
   the four operations, verification, and quarantine. For every transition that can fail, state
   whether the system fails *open* (serves the old content, a freshness miss) or *closed*
   (refuses to serve, a coverage miss) — and justify each choice, because the right answer
   differs between a rate table and a compliance clause.
5. **Budget the next full re-embed.** Compute the re-embed token volume with your own overlap
   multiplier, state the rate you will pay and its source, and decide the cutover pattern
   (dual-write and swap, or in-place with downtime). Name the trigger that would force the
   re-embed, because a model deprecated under you is not a choice (→ M20).
6. **Write the deletion inventory as a table.** One row per store holding document text or an
   embedding derived from it, and for each: is it reached by your deletion job, and if not,
   what closes the gap. Then state the verification: what read proves the deletion happened, in
   which store.
7. **Choose one invariant you will assert every run.** Candidates: "no document has two live
   versions", "every node's `ref_doc_id` resolves", "index size grows by exactly the delta".
   State it as a query, and state what the system does when it fails — an asserted invariant
   that only logs is a dashboard (→ M13).

**Why this exercise matters.** Ingestion is the least glamorous module in this course and the
one that silently determines whether every other metric measures anything. Recall@10 against a
corpus with two live versions of every revised clause is not a low score but a *meaningless*
one, because the ground truth itself is ambiguous. The senior-engineer version of this work is
recognising that the interesting state is not "indexed" but "super-seded-and-verified", and
that the four operations a real corpus needs are not the one the framework's default
configuration provides. Note what the exercise forces you to confront: the failure that matters
most in the worked example is the one that *does not get worse when you optimise latency*.

---

## Where the literature disagrees

- **"Incremental and streaming ingestion beats periodic re-indexing."** Streaming RAG argues
  the case directly: "Full-scale indices incur high memory costs; periodic rebuilds introduce
  latency that undermines data freshness; naive sampling sacrifices semantic coverage",
  reporting "an incremental index upsert mechanism [that] refreshes prototypes without
  interrupting queries", with "statistically significant gains in Recall@10 (up to 3 points, p
  < 0.01), end-to-end latency below 15 ms, and throughput above 900 documents per second under
  a 150 MB budget" [STREAM25]. The countervailing argument comes from architectures whose index
  is *global*: RAPTOR builds its index by "recursively embedding, clustering, and summarizing
  chunks of text, constructing a tree with differing levels of summarization from the bottom
  up" [RAPTOR24] — clustering is a whole-corpus operation, so one new document can shift many
  nodes' cluster assignments, and the paper documents no incremental update path. **The split
  is by layer.** The *vector index* layer is largely settled in favour of dynamism: CleANN
  states the problem it solves as "existing dynamic graph-based indexes suffer from at least
  one of the following problems: (1) the query quality degrades as updates happen; and (2) the
  graph structure updates used to maintain the index quality upon updates are global and thus
  expensive", and reports that its index "has query quality at least as good as if the index
  had been built statically using the corresponding data", with "7-1200x throughput improvement
  on million-scale real-world datasets" at the same recall [CLEANN25]. The *derived-structure*
  layer (trees, communities, summaries, claim graphs) is not settled, and those layers carry a
  rebuild cost that grows with corpus volatility. **What to change:** price *by layer* — vector
  index (incremental), derived structures (rebuild-or-tombstone) — and do not let the settled
  answer at one layer stand in for the other.
- **"Deleting from the index is enough to forget."** RAG-based unlearning is the strongest
  published case that it is: the framework modifies the external knowledge base to "simulate
  the effects of forgetting without directly interacting with the unlearned LLM", favoured
  because weight-level alternatives "often suffer from high computational demands, limited
  applicability, or the risk of catastrophic forgetting", with results across "ChatGPT, Gemini,
  Llama-2-7b-chat, and PaLM 2" meeting "five key unlearning criteria: effectiveness,
  universality, harmlessness, simplicity, and robustness" [RAGUNLEARN24]. The disagreement is
  with its scope as an operational claim: those criteria describe the *observable behaviour of
  the model afterwards*, and none is completeness across stores. A documented default already
  violates the strong reading — `delete_from_docstore` defaults to `False`, leaving the text in
  the docstore while removing it from the queryable index [LI-DM] — and one index family has a
  documented gap, the tree index not supporting deletion at all [LI-DM]. **What to change:** a
  behavioural unlearning result is not an index-completeness guarantee. Measure what the
  auditor measures: for each store, a read that proves the bytes are gone. The unlearning
  literature measures the answer; the obligation is on the storage.
- **"A full rebuild is the safe option."** This is the instinct the worked example prices at
  1,780× the token cost of the incremental path, and both cited sources argue against it —
  RAPTOR implicitly (its structure is inherently rebuild-shaped, a cost rather than a safety
  property) [RAPTOR24] and streaming RAG explicitly, naming "periodic rebuilds" as a source of
  latency that "undermines data freshness" [STREAM25]. **What to change:** treat "rebuild" as
  one option on a cost/quality frontier with a measured staleness window, not as the
  conservative default. The genuinely conservative choice is the verification read, which costs
  0.5M tokens a day and is the only thing standing between "the pipeline reported success" and
  "the new clause is served".
- **"A long context window removes the need to keep the index fresh."** If the model reads the
  whole relevant corpus at query time, staleness becomes a matter of what you put in the
  window. Xu and colleagues tested the retrieval-versus-window trade-off: "LLM with 4K context
  window using simple retrieval-augmentation at generation can achieve comparable performance
  to finetuned LLM with 16K context window via positional interpolation on long context tasks,
  while taking much less computation", and — the load-bearing sentence for this module —
  "retrieval can significantly improve the performance of LLMs regardless of their extended
  context window sizes" [RECALL23]. The opposing result comes from a comprehensive comparison
  that found "when resourced sufficiently, LC consistently outperforms RAG in terms of average
  performance. However, RAG's significantly lower cost remains a distinct advantage" [LC24].
  **What to change:** both findings leave this module's conclusion untouched. Whether retrieval
  helps or long context wins, something must still *get the new contract into the system* — a
  longer window changes what you retrieve, not how content arrives, and the dual-write-and-swap
  problem exists identically in a long-context architecture reading from a document store. That
  choice is M1's; this module's claim survives either outcome: **ingestion is upstream of
  both.**

---

## Sources

- [Ingestion Pipeline](https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/) — `[LI-IP]` · owns `IngestionPipeline` and transformations, node+transformation caching, cache persistence and remote caches, the embeddings-must-be-a-stage rule, docstore duplicate detection by `doc_id`/`ref_doc_id` and `document_hash`, the no-vector-store caveat, `num_workers` multiprocessing, and `arun`.
- [Document Management](https://developers.llamaindex.ai/python/framework/module_guides/indexing/document_management/) — `[LI-DM]` · owns insert, delete, update and refresh operations; `delete_ref_doc` and the `delete_from_docstore` default; `update_ref_doc`; `refresh_ref_docs`; the tree-index deletion gap; and `index.ref_doc_info` / `RefDocInfo` node lineage.
- [Storing](https://developers.llamaindex.ai/python/framework/module_guides/storing/) — `[LI-DS]` · owns the document-store / index-store / vector-store / property-graph-store decomposition and the `StorageContext` that binds them.
- [Document Stores](https://developers.llamaindex.ai/python/framework/module_guides/storing/docstores/) — `[LI-DOCS]` · owns `SimpleDocumentStore` persistence and the persistent docstore backends (MongoDB, Redis, Firestore, Couchbase, Tablestore, AlloyDB, Cloud SQL for PostgreSQL) that persist as nodes are ingested.
- [Index Stores](https://developers.llamaindex.ai/python/framework/module_guides/storing/index_stores/) — `[LI-IS]` · owns the index-metadata store, its backends, and `load_index_from_storage` as the reload path after a re-ingest.
- [Defining and Customizing Documents](https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_documents/) — `[LI-DOC]` · owns `doc_id` customization, `filename_as_id`, and the metadata that travels with a document into its nodes.
- [Retrieval Evaluation](https://developers.llamaindex.ai/python/examples/evaluation/retrieval/retriever_eval/) — `[LI-EVR2]` · owns the statement that node ids default to random UUIDs and must be set manually for run-to-run stability, and the observed 429 retry behaviour under batch load.
- *From Static to Dynamic: A Streaming RAG Approach to Real-time Knowledge Base* — Yuzhou Zhu · arXiv:2508.05662 (2025) · [arxiv.org/abs/2508.05662](https://arxiv.org/abs/2508.05662) — `[STREAM25]` · owns the streaming-RAG pipeline, its incremental index upsert mechanism, and its reported Recall@10, latency and throughput figures.
- *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* — Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning · arXiv:2401.18059 (2024) · [arxiv.org/abs/2401.18059](https://arxiv.org/abs/2401.18059) — `[RAPTOR24]` · owns the recursive embed-cluster-summarize tree construction and its whole-corpus clustering property; cited here only for that structural consequence, since M16 owns its retrieval-quality claims.
- *CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search* — Ziyu Zhang, Yuanhao Wei, Joshua Engels, Julian Shun · arXiv:2507.19802 (2025) · [arxiv.org/abs/2507.19802](https://arxiv.org/abs/2507.19802) — `[CLEANN25]` · owns the statement of the dynamic-index quality-degradation and global-update problems, and the static-quality / throughput result under full dynamism.
- *When Machine Unlearning Meets Retrieval-Augmented Generation (RAG): Keep Secret or Forget Knowledge?* — Shang Wang, Tianqing Zhu, Dayong Ye, Wanlei Zhou · arXiv:2410.15267 (2025) · [arxiv.org/abs/2410.15267](https://arxiv.org/abs/2410.15267) — `[RAGUNLEARN24]` · owns knowledge-base-modification as an unlearning mechanism and the five unlearning criteria.
- *Retrieval meets Long Context Large Language Models* — Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, Bryan Catanzaro · arXiv:2310.03025 (ICLR 2024) · [arxiv.org/abs/2310.03025](https://arxiv.org/abs/2310.03025) — `[RECALL23]` · owns the retrieval-versus-extended-context comparison and the finding that retrieval improves LLMs regardless of window size.
- *Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach* — Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky · arXiv:2407.16833 (EMNLP 2024 industry track) · [arxiv.org/abs/2407.16833](https://arxiv.org/abs/2407.16833) — `[LC24]` · owns the opposing finding that LC outperforms RAG when sufficiently resourced, and the Self-Route hybrid.

**Unsupported (synthetic, no located source):** the LogiTrack corpus scale-up, the Maersk revision
narrative, the 480,000-document and 2.24B-word targets, the daily delta of ~600 documents plus
~4,000 wiki pages, and every number in the worked-example table are fiction or constructed
illustration, not empirical findings. The token-volume arithmetic (words/token at 0.75, the 160×
scale factor, and the 1.25× overlap multiplier from M3's $O/C = 0.20$ decision) is derived in-file
from the course's own stated figures and its own chunking decision. **All three dollar rates in the
re-embed table (\$0.01, \$0.02, \$0.13 per 1M tokens) are illustrative placeholders with no located
source** — the module deliberately does not assert any vendor's embedding price, and a reader
pricing a real backfill must substitute their own contracted rate and cite it. The five
tradeoff-ledger entries are design decisions argued in this file. Three further claims are
mechanisms argued here rather than findings: (a) that a content-derived cache key can serve stale
outputs after a transformation code change — this follows from the documented key definition
[LI-IP] but no source states it as a risk and no source documents an invalidation policy; (b) that
a failure to upsert can be silent at the vector-store layer, which motivates the
`Indexed → Verified` transition but is not documented as such anywhere cited; (c) that the deletion
inventory table is *complete* — it is the enumeration this module could construct from the
framework's storage decomposition [LI-DS][LI-DOCS][LI-IS] plus the cache, and any architecture with
additional replicas, mirrors, or analytics dumps has rows missing from it. Finally, `[STREAM25]` is
a single-author preprint reporting a prototype-set index under a 150 MB memory budget; its numbers
are reported above as the paper's own claims and its index structure is not the same object as a
480k-document graph index, so it is cited for the *direction* of its argument, not as a
like-for-like benchmark against `[CLEANN25]`.

**Verification note:** 🟡 *pending* — Step 7 (memory-less logic review) and Step 8 (literature survey) not yet run on this module.

**Next module:** [M15 — Fine-Tuning Embeddings and Rerankers for a Domain](../15-finetuning-retrieval-models/README.md)
— why the training data you would fine-tune on is generated by the pipeline this module just built.
