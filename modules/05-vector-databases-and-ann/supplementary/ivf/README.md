# IVF across nine implementations

Companion to the module's [IVF section](../../README.md#ivf-partitions-the-second-family). One file per
provider; this page is the side-by-side. Every provider was given the same brief: read the source,
document the algorithm and every dial, run the module's 8-value toy at `nlist = 3`, and run one scale
test on one shared generator so the numbers are comparable.

**Shared scale test.** N = 187,043 (the pilot count) × 256-d float32, a 50-topic Gaussian mixture
(`rng(0)`, topic spread 0.35), 1,000 queries from the same mixture, recall@10 against brute-force L2.
Generator and ground truth are identical in every `lab/ivf_<provider>_scale.py`. Synthetic data: the
absolute recall numbers are not LogiTrack's; the *shape* (what moves, what does not) is the lesson.

## What ran here, and what did not

| Provider | File | Real library ran on this Mac? | Flavour |
|---|---|---|---|
| FAISS 1.16 | [faiss.md](faiss.md) | Yes | IVF-Flat, IVF-PQ, +refine; the reference |
| pgvector 0.6.2 | [pgvector.md](pgvector.md) | Yes (PostgreSQL 16 via `pgserver`) | IVF-Flat only |
| Knowhere (Milvus Lite 3.2) | [knowhere.md](knowhere.md) | Yes | IVF_FLAT (patched FAISS); SQ/PQ/RaBitQ documented |
| LanceDB 0.41 | [lancedb.md](lancedb.md) | Yes | IVF_FLAT, IVF_PQ (default), adaptive probing |
| Elasticsearch 9.2 [DiskBBQ](elasticsearch.md#3-the-algorithm-es-920) | [elasticsearch.md](elasticsearch.md) | Yes (Docker); toy simulated | Hierarchical k-means + SOAR + 1-bit codes + rescore |
| Annoy 1.17 | [annoy.md](annoy.md) | Yes, after rebuilding without `-ffast-math` | Random-projection tree forest (partitioning, not k-means) |
| ScaNN 1.4 | [scann.md](scann.md) | No (Linux wheels only); simulated | k-means tree + anisotropic quantization + reorder |
| SPANN / SPTAG | [spann.md](spann.md) | No (C++ build); simulated | Heads in RAM, capped postings on SSD, RNG replication |
| cuVS 26.12 | [cuvs.md](cuvs.md) | No (needs NVIDIA GPU); simulated | Balanced hierarchical k-means, IVF-Flat / IVF-PQ |

"Simulated" means a Python re-implementation of the cited source lines, labelled as such in the file;
no scale numbers were produced for those three.

## The frontier: smallest `nprobe` reaching recall@10 ≥ 0.95

| Provider | `nlist` ≈ 4096 | `nlist` ≈ 1024 | Notes |
|---|---|---|---|
| FAISS `IVF,Flat` | **64** | **32** (16 gives 0.947) | 1.97 % of corpus scanned at the knee |
| pgvector `ivfflat` | **64** | **16** | also `lists=187` (its own rule): 4; ≈1 ms at the knee in all three |
| Knowhere `IVF_FLAT` | **64** | **16** | Lite built two segments, 4096 lists each |
| LanceDB `IVF_FLAT` | **256** (0.92–0.97 across 3 builds) | **32** (adaptive default: 1.000) | hierarchical k-means above 256 partitions; 186 empty lists |
| Elasticsearch `bbq_disk` | **≈64** (2.19 % visited, `cluster_size=64` → 2,923 lists) | **≈16** (1.57 %, `cluster_size=183`) | needs rescoring: 1-bit ceiling without it is 0.38 |
| Annoy | `n_trees=10, search_k=10,000` (≈5 % of N as candidates) | | forest size barely moves recall at fixed `search_k` |
| ScaNN (simulated) | 128 | 32 | half the leaves re-seeded every iteration at forced 4096; see file |
| FAISS `IVF4096,PQ32x8` | never (plateau 0.235) | | `+RFlat, k_factor=64` → 0.968 |
| LanceDB `IVF_PQ m=32` | never (0.236; `refine_factor=10` → 0.694) | | silently built 730 partitions, not 4096 |

Three independent IVF-Flat implementations land on the same number (64 at 4096, 16–32 at 1024) because
the number is a property of the data and `nlist`, not of the library. The two that deviate do so for
reasons visible in their source (LanceDB's hierarchical training; Elasticsearch's 1-bit codes), which
is the module's "compare implementations, not algorithms" point with numbers attached.

## The same dial under nine names

| Concept | FAISS | pgvector | Knowhere | LanceDB | Elasticsearch | Annoy | ScaNN | SPANN | cuVS |
|---|---|---|---|---|---|---|---|---|---|
| number of lists | `nlist` | `lists` | `nlist` | `num_partitions` | `cluster_size` (= N/nlist, inverted) | `_K` leaf size × `n_trees` (implicit) | `num_leaves` | `Ratio` (heads ≈ 12–20 % of N) | `n_lists` |
| lists opened per query | `nprobe` | `ivfflat.probes` | `nprobe` | `nprobes` (or adaptive) | `visit_percentage` (fraction of vectors) | `search_k` (candidate count) | `num_leaves_to_search` | `SearchInternalResultNum` | `n_probes` |
| default for that | 1 | 1 | 8 | adaptive (6–24) | dynamic (≈1.5 %) | `n · n_trees` | — | 64 | 20 |
| rescoring with full vectors | `IndexRefine` / `k_factor` | none | `refine`, `refine_k` | `refine_factor` | `rescore_vector.oversample` (default 3) | always (exact rerank) | `reorder` | always (full vectors on SSD) | `refine()` call |
| multiple assignment | no | no | no | no | SOAR (λ = 1), ~every vector twice | forest of trees | SOAR (dot product only) | RNG rule, `ReplicaCount` 8 | no |
| filter handling | per code inside probed lists | post-filter on index TIDs; `iterative_scan` 0.8+ | bitset inside probed lists | prefilter / postfilter | filter-aware list extension | none | none (OSS) | none (static) | bitset inside probed lists |
| insert after build | nearest existing centroid | nearest existing centroid | per-segment index | `optimize()` → existing centroids | new segment, re-clustered on merge | rebuild only | nearest centroid; `incremental_threshold` splits | SPFresh split/merge | `extend`, `adaptive_centers` |
| seed exposed | `cp.seed` (1234) | no | FAISS default | no (OS entropy) | fixed `Random(42)` | `set_seed` | `clustering_seed` | — | fixed (`i mod k`) |

## Findings that cut across providers

- **`nprobe` is read off a curve, never set.** Every real run needed a sweep to find the knee, and the
  knee moved with `nlist`, with the training procedure, and (in the module's own earlier test) 20× with
  the data distribution. Three libraries expose the sweep as a feature: FAISS `ParameterSpace`,
  LanceDB adaptive probing, Elasticsearch's dynamic `visit_percentage`.
- **`nlist` is a per-library, per-N choice.** FAISS wants ≥39 points per centroid; pgvector's rule is
  rows/1000; Lance caps quantized indexes at N/`sample_rate`; Elasticsearch parameterises the inverse
  (`cluster_size`, floor 64); ScaNN's defaults break at a forced 4096 on 187k points. The course's
  "4096" was honoured by four of them and silently or loudly changed by three.
- **PQ hits a resolution wall that `nprobe` cannot fix.** FAISS and LanceDB both plateau near 0.235
  at `m = 32` on this generator regardless of lists opened; only rescoring with full vectors recovers
  it (FAISS to 0.968 at `k_factor = 64`; Lance's `refine_factor = 10` reaches 0.694). Elasticsearch's
  1-bit codes show the same shape: 0.38 ceiling, 0.99 with `oversample = 3` and a wide candidate pool.
  This is tradeoff M5-3 measured: compression costs recall that rescoring buys back, provided the
  originals are kept.
- **Filters starve post-hoc.** pgvector returned 0 rows for an out-of-topic filter at every setting up
  to the knee; FAISS, Knowhere and cuVS apply the filter inside probed lists, which keeps results
  correct but still leaves them short; Elasticsearch extends the list budget when a filter is present;
  SPANN and OSS ScaNN do not filter at all.
- **Engines add behaviour the library does not have.** Milvus Lite trained one IVF per sealed segment
  (two indexes, not one) and its client ignored `nlist` on one API path; Lance switched training
  algorithm above 256 partitions; Elasticsearch re-clusters on every segment merge. None of this is
  visible from a FAISS notebook, and all of it changes the recall you measure.
- **Multiple assignment is the index-time answer to boundary misses.** SPANN's RNG replication and
  Elasticsearch's SOAR both store boundary vectors in a second list; the SPANN toy shows a query that
  plain IVF misses at `nprobe = 1` and replicated IVF hits. Cost: posting storage roughly doubles
  (Elasticsearch's `.clivf` holds ~2× codes).
- **The toy is a k-means lottery.** On 8 one-dimensional values, three different partitions appear
  across rebuilds in pgvector, LanceDB and Knowhere; whether query 9.5 or 5.5 misses depends on which
  one you drew. Libraries with a fixed seed (FAISS, Elasticsearch, cuVS) are reproducible; the
  engines that hide the seed are not.

## Reproduce

Each file ends with its own venv, install line and commands. The scripts are in [`../../lab/`](../../lab/)
as `ivf_<provider>_toy.py` and `ivf_<provider>_scale.py`; run them with `python -I`. The three
simulated providers' scale scripts print a skip message and exit 0 on hardware that cannot run them.
