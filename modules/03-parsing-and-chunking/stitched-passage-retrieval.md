# Retrieval with stitched passages

> $\color{red}\text{to be discussed}$ — the query-time pipeline that turns overlapping chunk hits
> into stitched passages, reranks them, and fills the LLM's evidence budget. It expands M3's
> "Mitigation: measure, stitch, then cut overlap" collapsible ([README](README.md#overlap-what-it-buys-and-what-it-costs)).

## Remember these six things

1. **Overlapping chunks from one document get merged, not dropped.**
   - Dropping a duplicate also throws away the part of it that *doesn't* overlap.
2. **Stitching stops at only two things:** the reranker's input limit, and a structural boundary.
   - Every hit in the top N earned its place, so stitching never drops one.
3. **Fetch many, rerank fewer, give the LLM a token budget.**
   - Fetch N ≈ 100 → rerank the top C ≈ 30–50 → fill B tokens.
4. **Embeddings do two jobs:** find the N hits, and choose which C passages the reranker sees.
   - After reranking, only the reranker's score counts.
5. **Variety comes from the question, not from a cap.**
   - A question needing several sources is split into one sub-question per source, and each one runs
     the whole pipeline.
6. **Never pad.** If the budget isn't full, flag the question as low-evidence.

## Terms

| Term | Meaning |
|---|---|
| **Chunk** (node) | What the parser made at ingest. It has a vector, a `doc_id`, a `version`, character offsets and structure metadata |
| **Hit** | A chunk returned by the vector index for this query, with its cosine score |
| **Stitched passage** | One or more hits from the same document merged into one contiguous span. It exists only at query time |
| **Sub-question** | One part of a question that needs several sources, made by the router; it runs the pipeline on its own |
| **N** | The number of hits fetched from the vector index |
| **C** | The cap: the maximum number of stitched passages sent to the reranker |
| **B** | The evidence budget: the tokens of passages the LLM may receive |
| **L** | The reranker's input limit, in the **reranker's** tokens |

## What this pipeline fixes, and what it doesn't

| Problem | What happens | Fixed by |
|---|---|---|
| **Overlap duplicates** | The same text in one document takes two slots, so fewer distinct facts reach the LLM | **Stitching** (step 4) |
| **One document crowding out a source the question needs** | A comparison question fills B from one carrier's document, and the other carrier's is missing | **Decomposing the question** (step 1): each source is asked for explicitly |
| **Near-identical text in different documents** (a clause copied between contracts) | Offsets can't see it, because each document has its own offsets | **Not this pipeline.** Use deduplication at ingest (MinHash or a text hash, M14) or MMR at query time (M9) |

- **Many passages from one document is often correct.** When the answer lives in one document (rules
  from one tariff), the reranker's order should fill B from it. That's why there's **no per-document
  cap.**

## The pipeline

### 0 · What ingestion must already have stored
- **Docstore:** the full text of every document, keyed by `doc_id` and `version`.
- **Every chunk carries:**
  - `doc_id` and `version`;
  - `start_char_idx` / `end_char_idx`, its offsets into the document's text;
  - **structure metadata:** its heading path, and its region kind (prose, table);
  - its embedding, in the vector index (HNSW).
- **Chunk size ≤ L.** Otherwise the reranker cuts off even a single hit.
- **N, C, B and the score floor** have been tuned on the evaluation set (see
  [the settings](#the-settings-and-where-each-comes-from)).

### 1 · Route the question, and split it if it needs several sources
- The router (M10) labels the question:
  - **single-source:** a rate, a clause, one policy;
  - **several sources:** comparing carriers, a policy plus an exception in another document.
- **Several sources → decompose** it into **s** sub-questions, one per source.
  - Each sub-question runs **steps 2–7 on its own**, with its own share of the budget: **B ÷ s**.
- **Single-source →** one run of steps 2–7 with the whole of B.
- **The router's decisions are measurable:** score its labels and its decompositions on the labelled
  set (M13).

### 2 · Embed the query
- Use the **same embedding model** the index was built with.

### 3 · Retrieve N hits
- Ask the vector index for the **N ≈ 100** nearest chunks.
- Set `ef_search` ≥ N (`hnsw_ef` in Qdrant). It's how many candidates HNSW keeps in play while
  searching ([M5 · HNSW](../05-vector-databases-and-ann/README.md#hnsw-three-dials-only-one-of-them-free)).
- Apply metadata filters in the same call: tenant, current version (M6).
- **Why so many:** fetching is cheap (a few milliseconds more for 100 than for 20). Reranking is
  the expensive step.
- **Output:** N hits, each with its cosine score.

### 4 · Stitch the hits into passages
1. **Group** the hits by (`doc_id`, `version`).
   - Never merge across versions: offsets from two versions of a document don't line up (M14).
2. **Sort** each group by `start_char_idx`.
3. **Merge** the next hit into the current passage when it **overlaps or touches** it (a gap ≤ g
   characters).
4. **Stop extending** the passage, and start a new one, at the **first** of:
   - **the reranker's input limit:** the passage would exceed L tokens, counted with the
     **reranker's** tokenizer;
   - **a structural boundary:** a different heading path, or a change of region kind (prose to table).
     Text on either side belongs to different units, so a passage across it mixes two topics.
5. **Drop nothing.** A group too long for one passage becomes **several consecutive passages,** split
   at hit boundaries. Every hit lands in exactly one passage.
6. **Cut each passage's text from the docstore:** `doc.text[start : end]`.
   - Gluing chunks together would repeat the overlapping text.
   - If only nodes were stored, rebuild the text by trimming each node's overlap using its offsets.
7. **Score each passage** with the **highest cosine** among its hits.
8. **A hit with no offsets** (LlamaIndex leaves them `None` when it can't find the node's text in the
   document): fall back to its `PREVIOUS`/`NEXT` relationships or its text hash, and log it.
- **Output:** stitched passages, each with a cosine score.

### 5 · Cap what goes to the reranker
- Sort the stitched passages by cosine score and keep the **top C**.
- **This is the last use of embeddings.** A good passage ranked below C never reaches the reranker,
  so embedding quality caps the whole pipeline.

### 6 · Rerank
- The cross-encoder scores each (query, stitched-passage text) pair, run in batches.
- Re-order by the **reranker's score.** The cosine scores are discarded.

### 7 · Fill the budget
The budget is **B** for a single-source question, or **B ÷ s** for each sub-question. Walk the
reranked list from the top:
- **Below the score floor** (on the reranker's scale)? Stop; everything after it scores lower.
- **Doesn't fit in the tokens left** (counted with the **generator's** tokenizer)? Skip it; a
  shorter passage further down may still fit.
- Otherwise, **add it.**

Stop when the budget is full or the list runs out.

- **Sub-questions with tokens left over** pass them to the others.
  - Hand them out **round-robin by rank** (each other sub-question's next passage in turn), not by
    score. Reranker scores for different sub-questions answer different queries, so they aren't
    comparable.

### 8 · If B isn't full
- **Don't pad** with weaker passages: they add noise (F2).
- **Flag the question as low-evidence,** so generation can say it can't fully answer (M12), and log
  it.
  - For a decomposed question, flag **which sub-question** came up short.
- **No refill loop.** One large fetch in step 3 is cheaper and more predictable.

### 9 · Generate
- The LLM gets the question plus the chosen passages, **grouped by sub-question** when it was
  decomposed.
- Each passage is cited by `doc_id` + offsets, which map back to a page (M12).

### 10 · Log for monitoring (M19)
- The merge ratio (hits per stitched passage), at p50 and p95.
- How often stitching stopped at L, and how often at a structural boundary.
- How full B got, and the share of questions (and sub-questions) that leave it under-filled.
- The share of questions decomposed, and the number of sub-questions.
- How many passages the score floor cut.
- Latency per stage: route, retrieve, stitch, rerank, fill.
- **Alert when under-filled questions rise.** It usually means the index drifted (re-ingestion, a
  parser change) or a new kind of question arrived.

## The settings and where each comes from

| Setting | What it controls | How it's set |
|---|---|---|
| **N** | Hits fetched | Large enough that B is rarely under-filled, given the measured merge ratio. Cheap to raise |
| **`ef_search`** | Search effort in HNSW | ≥ N; raise it until recall stops improving (M5) |
| **g** | Whether two hits are close enough to merge | Small, around one sentence |
| **L** | Where stitching stops on length | **Not tuned:** the reranker model's input limit, in its own tokens |
| **C** | Stitched passages sent to the reranker | See [How C is chosen](#how-c-is-chosen) |
| **Score floor** | The minimum reranker score allowed in | Calibrated on the reranker's scale on the labelled set (M9 and M13). No sensible default exists |
| **s** | Sub-questions for a question needing several sources | **Not tuned:** the router's decomposition decides it, and it's scored on the labelled set |
| **B** | Evidence tokens given to the LLM | The context budget (M11) |

- Tune N, C and the score floor **per tenant** if corpora differ a lot. A 100-page tariff merges very
  differently from a set of short memos.

### How C is chosen
A cross-encoder runs one model pass per (query, passage) pair, so its cost grows with the number of
passages and their length. C must satisfy two limits:

1. **The latency ceiling:**
   - C_max = (the rerank share of the p95 latency target) ÷ (time per pair at your passage length).
   - Measure the time per pair on real hardware, with batching on, at real stitched-passage lengths.
   - For a hosted reranker, the price per call (and any limit on documents per call) is a second
     ceiling.
   - **A decomposed question reranks s times,** so either the latency target allows for it, or each
     sub-question gets a C that's s times smaller.
2. **The recall floor:**
   - **recall@C:** how often the passage holding the answer is in the top C *before* reranking.
   - Take the **smallest C at which recall@C stops rising.**

- **C = that smallest value, as long as it's ≤ C_max.**
- **If recall@C is still rising at C_max,** latency is binding. Choose one or more of:
  - a faster reranker;
  - a reranker with a smaller L, so passages are shorter;
  - a better first-stage retriever (embeddings or hybrid search, M7), so good passages rank higher.
- Re-measure both limits whenever the reranker, the hardware, the passage lengths or the corpus
  change.

## Building it

### In LlamaIndex
- **Routing and decomposition:** a router query engine or a sub-question query engine (M10). Each
  sub-question gets its own retriever run.
- **Stitching:** a custom `BaseNodePostprocessor`. LlamaIndex core has none that merges by offsets.
  `PrevNextNodePostprocessor` adds neighbours, which is a different job.
- **Order of postprocessors:** stitching → the cap → the reranker postprocessor → a final step that
  fills the budget.

### Decisions still open before coding

| Decision | What it depends on |
|---|---|
| **B** | The generator and its context budget (M11) |
| **The reranker** | Its model, which also fixes L |
| **The score floor** | A labelled question set scored by the chosen reranker |
| **The router** | How it labels questions and decomposes them, scored on the labelled set (M10 and M13) |
| **Structure metadata on chunks** | Ingestion must write the heading path and region kind |
| **A version id on chunks** | Ingestion must write it (M14) |

### Testing
- **Real questions over the real index, with no mocks.** Use M1's 60-question set, including
  questions that need several sources.
- **Measure before and after stitching:**
  - the merge ratio;
  - how full B gets;
  - recall and precision of distinct evidence in the final context.
- **For decomposed questions,** check that every source the question needs appears in the final
  context.
