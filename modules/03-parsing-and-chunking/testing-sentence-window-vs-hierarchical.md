# Testing `SentenceWindowNodeParser` vs `HierarchicalNodeParser`

> $\color{red}\text{to be discussed}$ — a test plan, not yet run. Back to [M3 parsers](README.md#the-parsers-and-when-each-one-earns-its-place) · auto-merging internals in [M8](../08-index-structures-beyond-flat/README.md#structure-at-index-time-recursive-retrieval-and-auto-merging).
>
> Both techniques also apply beyond prose (markdown, tables, HTML, JSON, code): [expansion-beyond-prose.md](expansion-beyond-prose.md).

This is a test set built to make each one win where it should and fail where it should, with everything else held constant.

## The corpus

- **One corpus for both:** LogiTrack carrier agreements as **flat text** (naive extraction). Neither gets structure for free.
- **Everything else fixed:** same embedder, same LLM, same query set. Only the parser and retriever change.
- **A baseline:** `SentenceSplitter(512, 50)`, so every win and loss is measured against something.

## Queries with planted, controlled answer spans

Use character offsets so each case is constructed deliberately, not found by luck:

| Case | Answer span | Placement | Built to show |
|---|---|---|---|
| **A. Opening scene** | 2 sentences (~60 tokens) | Straddles a 128-token leaf boundary | Window **wins**; hierarchical **fails** (1 of 5, no merge) |
| **B. Single fact** | 1 sentence | Inside one leaf | Both win; window more precisely |
| **C. Long clause** | 6-8 sentences (~300-400 tokens) | Within one 512-token parent | Hierarchical **wins** (3+ of 5 merge); window **fails** if the hit is at the clause's edge (only N + 1 sentences covered) |
| **D. Section question** | "What are the obligations under clause 14?" | The whole section | Hierarchical wins; window returns fragments |
| **E. Parent-boundary straddle** | ~100 tokens | Across two 512-token parents | Hierarchical **fails** (different parents); window wins |
| **F. Section-edge sentence** | The last sentence of 14.2 | Next to "15. Fuel Surcharge" | Window **fails on noise**: it pulls in unrelated clause 15 |
| **G. Anaphoric sentence** | "Such period shall not apply to reefers." | The meaning sits in the previous sentence | Window **fails on retrieval**: the sentence alone embeds poorly, and an unretrieved hit can't be expanded |
| **H. Rate-table lookup** | One row | A linearised table | **Both fail**, a parsing problem in both cases |

## Ablations: the misconfigurations people actually make

- **Sentence window without `MetadataReplacementPostProcessor`:** completeness should collapse to single sentences.
- **Hierarchical with every level embedded:** duplicate parent and child hits in top-k, 1.3× the vectors, and 2.7× the embedded tokens.
- **Settings to sweep:**
  - `window_size`: 1 / 3 / 5;
  - `simple_ratio_thresh`: 0.3 / 0.5 / 0.7;
  - `similarity_top_k`: 6 / 12 / 24. A merge needs 3 of 5 siblings, so a small top-k never merges.

## Metrics per case

- **recall@k:** is the answer span in the retrieved set?
- **Completeness:** is the *whole* span in the text the LLM actually receives? This is the one that separates the two.
- **Noise ratio:** the share of context tokens outside the answer span. It shows window's neighbour pull-in and merge's unrelated siblings.
- **Answer correctness:** an LLM judge against the ground truth.
- **Costs:** nodes, vectors, memory, and query latency (docstore reads for merges and windows).

## Expected pattern (the hypothesis the test should confirm or break)

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| Sentence window | ✅ | ✅ | ⚠️ | ❌ | ✅ | ⚠️ noise | ❌ | ❌ |
| Hierarchical + auto-merge | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ❌ |
| Baseline 512 | ⚠️ | ✅ | ⚠️ | ❌ | ⚠️ | ✅ | ✅ | ❌ |

## What a good outcome teaches

- **Neither one dominates.** They fail on *opposite* span sizes: window fails on large answers, hierarchical on small straddling ones.
- **Both depend on structure they can't see.** Cases E, F and H are only fixed by structure-aware boundaries, which is M3's point that structure recovery comes first.
- **Misconfiguration costs more than parser choice.** The ablations should show bigger drops than the parser comparison does.
