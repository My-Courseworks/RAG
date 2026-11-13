# Measuring what one node really weighs — test plan

> $\color{red}\text{to be discussed}$ — a plan, not yet run. It replaces the **estimates** in M3's
> "What one node weighs" table ([README](README.md#overlap-what-it-buys-and-what-it-costs)) with
> **measurements** taken on a production-shaped stack.

## Ground rules

- **No mocks, no fakes, no stand-ins.** Not `MockEmbedding`, not in-memory vector stores, and not a
  hand-built JSON file pretending to be a docstore.
- **Real everything:**
  - a real embedding API, called over the network under real rate limits;
  - a real vector database server with persistent storage;
  - a real persistent docstore;
  - real documents.
- **The whole corpus whenever it's affordable.** At these volumes the full pilot costs under a
  dollar to embed (see the budget below), so there's no excuse to sample.
- **Every number is reproducible.** Pin every version, record the hardware, keep the raw logs, and
  store every vector so a re-run never has to pay twice.
- **A prediction comes before every measurement.** The README's estimate is the prediction; the
  write-up explains every miss.

## What gets tested

| # | Claim in the README | Predicted | How it's decided |
|---|---|---|---|
| H1 | Vector bytes per node | 6,144 B (1,536 × float32) | On-disk vector storage ÷ points |
| H2 | Window node's embedded text | ~116 B (29 tokens × 4 B) | Measured text bytes and tokens per node |
| H3 | `window` metadata | ~818 B | Measured bytes of the `window` field |
| H4 | Ids and relationships | ~500 B | Measured bytes of the id and relationship fields |
| H5 | HNSW graph at `m = 16` | ~140 B per node | On-disk HNSW index ÷ points |
| H6 | The vector store copies the whole node into its payload | +~1.2 KB per node | Read a stored point back; payload storage ÷ points |
| H7 | Per node, all-in | ≈ 8.2 KB (window), ≈ 9.2 KB (512-token chunk) | Sum of H1–H6 against total disk ÷ points |
| H8 | Window node count on the pilot corpus | 615,445 | Count the parser's output |
| H9 | Index size | ≈ 5.0 GB (window), ≈ 0.37 GB (512-token chunk) | Total storage on disk and in RAM |

**Expected to fail: H8.** The course generator's `_para()` builds "sentences" of 280–680 words
(noted in [practising-scale-under-constraints.md](../../practising-scale-under-constraints.md)), so
the window parser may produce far fewer nodes than 615,445. That is exactly why a real corpus (C2)
runs alongside.

## Corpora

| ID | Corpus | Why it's here |
|---|---|---|
| **C1** | The course corpus: `labs/generate_corpus.py --seed 1337 --docs 3000` | Comparable with every number in the README |
| **C2** | **Real commercial contracts:** CUAD v1, 510 contracts (⚠️ confirm the licence, believed CC BY 4.0, before use) | Real legal prose with real sentence lengths — the closest public match to carrier agreements |
| **C3** | **Real carrier documents:** published demurrage and detention terms and tariffs from carriers' websites (⚠️ check each site's terms of use) | The actual LogiTrack document type, including real PDFs, tables and layout |

- C2 and C3 are PDFs, so they go through a **real layout-aware reader**. Record which one and its
  version. The reader is part of what's being measured, since it decides the text and the metadata.
- Store every source document and a checksum so the run can be repeated exactly.

## The stack

Pin the exact version of every item at run time and record it in the run manifest.

| Layer | Choice | Why |
|---|---|---|
| Framework | `llama-index-core` 0.14.25 | The version every M3 finding was observed on |
| Embedding | OpenAI `text-embedding-3-small`, 1,536 dimensions, **real API**, your actual account tier | Matches the table's D = 1,536; its price and limits are in M3's `[PRICES]` source |
| Tokenizer | `tiktoken` `cl100k_base` | The tokenizer `text-embedding-3-small` bills in |
| Vector store | **Qdrant server** in Docker, persistent volume, through `llama-index-vector-stores-qdrant` | M5's engine; a real server, not a local file |
| HNSW | `m = 16`, `ef_construct = 200` | M5's decision |
| Docstore | **PostgreSQL** through LlamaIndex's Postgres docstore integration (⚠️ confirm the package name) | A real persistent docstore; the parents and windows live here in production |
| Hardware | Record CPU, RAM, disk type and OS | Disk and RAM figures depend on it |

## Configurations

| ID | Parser | Notes |
|---|---|---|
| P1 | `SentenceWindowNodeParser.from_defaults(window_size=3)` | The table's window column |
| P2 | `SentenceSplitter(chunk_size=512, chunk_overlap=50)` | The table's 512-token column |
| P3 | `HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128], chunk_overlap=20)`, leaves in Qdrant, all nodes in Postgres | The design LogiTrack chose |
| P4 | Same parser, every level in Qdrant | The multi-granularity design, for the size difference |

Every configuration runs on C1 and C2. C3 runs P1 and P2 at least.

## Procedure

### 0 · Environment
- Start Qdrant and Postgres from a pinned compose file with named volumes. Record image digests.
- Write a run manifest: package versions, image digests, hardware, account tier, date.

### 1 · Parse and count
- Run each parser over each corpus. Record node count, tokens per node (min, median, p95, max), and
  bytes of text per node.
- **H8 is decided here.**

### 2 · Break each node down by field
For every node, record the UTF-8 byte size of each part of its serialized JSON:

| Part | What it covers |
|---|---|
| `text` | What gets embedded |
| `metadata.window`, `metadata.original_text` | Window parser only |
| Other metadata | Document fields inherited by the node |
| Relationships | PARENT, PREVIOUS, NEXT, SOURCE, CHILD |
| Ids and hashes | `id_`, `hash`, the ids inside relationships |
| Framework fields | Excluded-key lists, templates, separators |

- Report the median and p95 per configuration. **H2, H3 and H4 are decided here.**

### 3 · Embed, for real
- Call the real API through LlamaIndex's embedding class. Use your real batch size, and log it.
- Log for every request: tokens, inputs, latency, HTTP status, retries.
- Handle 429s with the client's real backoff, and count them; don't hide them.
- Write every vector to disk keyed by node id and text hash, so storage steps can be repeated without
  paying again.
- Record total tokens, dollars at the account's real price, wall-clock time, and the limit that bound
  (TPM or RPM). This checks the throughput table in M3 as a side result.

### 4 · Load the vector store
- Create one Qdrant collection per configuration, with `m = 16` and `ef_construct = 200`.
- Insert through `QdrantVectorStore`, exactly as production would.
- **Wait until the collection is fully optimized** (status green, indexed vector count = point count)
  before measuring anything.
- **Read a few points back** and inspect the payload: is the whole node JSON there? Is the `window`
  stored twice? **H6 is decided here.**

### 5 · Measure storage
- **Disk:** the size of each collection's storage directory, split by segment component (vectors,
  payload, HNSW index, payload index, WAL). ⚠️ Confirm the directory layout on the pinned Qdrant
  version first. Report the WAL separately; it isn't steady-state.
- **RAM:** the container's resident memory after a restart and a warm-up query load, plus Qdrant's
  own telemetry.
- **Docstore:** Postgres table sizes including indexes and TOAST, divided by node count.
- **H1, H5, H7 and H9 are decided here.**

### 6 · Separate fixed costs from per-node costs
- Repeat steps 4–5 at roughly 10k, 100k and full size for P1 and P2, reusing the stored vectors.
- Fit bytes = fixed + per-node × N. The per-node slope is the number that belongs in the README.

### 7 · Variants that production would actually consider
| Variant | Question it answers |
|---|---|
| Payload on disk vs in RAM | How much RAM does the payload really take? |
| `window` kept only in Postgres, not in the payload | What does the "pointers, not text" option save? (M3 estimates ≈ 0.57 GB) |
| int8 scalar quantization with rescoring | Does the M5-2 memory saving appear on this data? |
| 768-dimension vectors (`dimensions=768`, re-embedded for real) | Does truncation save the ≈ 1.9 GB M3 predicts? |

## Validity checks

- **Three repeats** of steps 4–5 per configuration; report the spread.
- **One process at a time** on the machine during measurement; record anything else running.
- **Count check:** points in Qdrant = nodes parsed = rows in Postgres. Any mismatch voids the run.
- **Spot check:** re-embed 100 random nodes and confirm cosine ≈ 1.0 with the stored vectors.
- **Don't trust a single tool for sizes:** compare the filesystem's numbers with Qdrant's and
  Postgres's own reports.

## Pass and fail

- An estimate **passes** if the measurement is within ±20%.
- A miss means: replace the number in the README, and write down which assumption was wrong.
- H8 failing on C1 means the README's window row needs C2's numbers, or a corrected generator.

## Budget

| Item | Size | Cost |
|---|---|---|
| C1, P1 + P2 embeddings | ~18M + ~20M tokens | ≈ $0.76 at $0.02/1M |
| C1, P3 + P4 embeddings | ~22M + ~59M tokens | ≈ $1.62 |
| C2 and C3 | Measured in step 1 | Expected well under $1 |
| 768-dimension variant | One configuration again | Under $0.50 |
| Wall-clock | Bound by the account's TPM | ≈ 20 min per 20M tokens at 1M TPM |

Under **$5** in API costs for the whole plan. The cost is time, not money.

## Deliverables

- A measured version of the "What one node weighs" table, with C1 and C2 side by side.
- Raw CSVs for every step, the run manifest, and the compose file.
- One paragraph per hypothesis: predicted, measured, and the reason for any miss.

## Risks

| Risk | What it would break | Mitigation |
|---|---|---|
| The generator's long "sentences" | H8, and the whole window row | C2 and C3 run alongside |
| Rate limits | Wall-clock, not results | Real backoff, logged; the stored vectors make re-runs free |
| Integration payload format differs by version | H6 | Pin the integration version; inspect points directly |
| Licence or terms of use for C2 and C3 | Use of the corpus | Confirm before downloading; keep only what's allowed |
| Qdrant storage layout differs by version | The split of disk sizes | Confirm the layout on the pinned version first |
