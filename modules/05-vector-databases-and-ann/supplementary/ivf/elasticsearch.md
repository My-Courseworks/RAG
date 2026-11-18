# Elasticsearch DiskBBQ (`bbq_disk`): IVF with 1-bit codes, hierarchical k-means and SOAR

- **Flavour**: IVF whose postings hold **1-bit "BBQ" (optimized-scalar-quantized) codes relative to the list centroid** plus 14 bytes of corrections per vector; centroids are stored int7-quantized; **SOAR** writes most vectors into a *second* list; search re-scores with the raw floats by default. No PQ, no residual PQ, no graph over centroids (flat scan, or a two-level parent/child scan when there are >256 centroids).
- **Where it runs**: the format is Elasticsearch's own (`org.elasticsearch.index.codec.vectors.diskbbq`, GA in ES 9.2, Oct 2025); it is *not* in Apache Lucene `main` (Lucene only carries a sandbox `KMeans.java`). Needs a JVM server; here it ran as the official Docker image.
- **Ran here**: **yes, the real thing** for the scale test (ES 9.2.0 single node, Docker, 4 GB). The toy is a **SIMULATION** of the 9.2.0 clustering code because ES refuses `cluster_size < 64` and folds ≤64 vectors into one centroid; a real 8-vector/1-centroid run is also shown.

## 2. Source read

Checked out tag **v9.2.0** of `github.com/elastic/elasticsearch`, commit `25d88452371273dd27356c98598287b669a03eae` (2025-10-21) — the exact build that ran. Paths below are relative to `server/src/main/java/org/elasticsearch/` unless stated. Line numbers are from that tag.

| File | What it does |
|---|---|
| `index/codec/vectors/diskbbq/ES920DiskBBQVectorsFormat.java` | Constants: `DYNAMIC_VISIT_RATIO=0` (L70), `DEFAULT_VECTORS_PER_CLUSTER=384`, `MIN=64`, `MAX=65536` (L71-73), `DEFAULT_CENTROIDS_PER_PARENT_CLUSTER=16` (L74); three files per segment `.cenivf` / `.clivf` / `.mivf` (L54-58). |
| `index/codec/vectors/cluster/HierarchicalKMeans.java` | **Train**: `cluster()` L56-88 (≤targetSize → single mean centroid L63-77; recursive partition then one neighbour-aware refinement + SOAR L80-85); `clusterAndSplit()` L90-165 (k formula L95, sample size L96, split-if-134 % rule L135, empty-cluster removal L141-160). Constants L23-27: `MAXK=128`, `MAX_ITERATIONS_DEFAULT=6`, `SAMPLES_PER_CLUSTER_DEFAULT=64`, `DEFAULT_SOAR_LAMBDA=1.0`. |
| `index/codec/vectors/cluster/KMeansLocal.java` | `pickInitialCentroids` L54-70 (reservoir, `new Random(42L)`); `stepLloyd` L72-134; `getBestCentroidFromNeighbours` L136-187; **SOAR** `assignSpilled` L214-300; sampled Lloyd + final full pass `cluster` L354-386. |
| `index/codec/vectors/cluster/NeighborHood.java` | Centroid neighbourhoods for the refinement pass: brute force below 10 k centroids, else an HNSW graph *over centroids* (L32-40). Training-time only. |
| `index/codec/vectors/cluster/SampleReader.java` | Reservoir sampling of training vectors, seed 42 (L84-104). |
| `libs/simdvec/src/main/java/org/elasticsearch/simdvec/internal/vectorization/DefaultESVectorUtilSupport.java` | `soarDistance` L154-164: `‖x−c‖² + λ·((x−c)·(x−c₁))² / ‖x−c₁‖²`. |
| `index/codec/vectors/diskbbq/ES920DiskBBQVectorsWriter.java` | **Assign + store**: `calculateCentroids` L505-538; `buildAndWritePostingsLists` L67-159 (SOAR copies counted and appended L77-104; raw centroid + its dot product written at the head of each list L126-130; doc ids delta-coded in blocks of 16 L139-150); `writeCentroids` L355-370 → parents when `numCentroids > 16²` L365, `buildCentroidGroups` L451-492 (k-means *over centroids*, targetSize 16, SOAR off); per-vector layout `writeQuantizedValue` L540-548. |
| `index/codec/vectors/diskbbq/IVFVectorsWriter.java` | `flush` L172-215: every segment flush clusters its own vectors; `mergeOneFieldIVF` L272-420: a merge **re-clusters all merged vectors from scratch** (L341). |
| `index/codec/vectors/diskbbq/IVFVectorsReader.java` | **Search** L258-332: dynamic visit ratio L278-287, `maxVectorVisited = 2·visitRatio·N` L289, list loop L307-318, filter-aware extension L319-331. |
| `index/codec/vectors/diskbbq/ES920DiskBBQVectorsReader.java` | Query quantized int7 against the global centroid L95-105; flat centroid scan L139-176; parent/child scan L178-275 with `centroidOversampling = numCentroids/(2·numParents)` L113; `MemorySegmentPostingsVisitor` L362-588: query re-quantized to **4 bits per list** against that list's centroid L583, bulk-16 scoring L526-535, block skip when all 16 docs are filtered L516-519. |
| `search/vectors/AbstractIVFKnnVectorQuery.java` | Collector size `2·k` (SOAR duplicates) L120; dynamic budget `log10(N)²·min(10000, max(num_candidates, 5k))` L134-141; per-segment merge to k L152. |
| `search/vectors/KnnSearchBuilder.java` | Top-level `knn` becomes a query with **k = num_candidates per shard** (L501). |
| `index/mapper/vectors/DenseVectorFieldMapper.java` | `bbq_disk` options L1506-1544 (`cluster_size`, `default_visit_percentage`, default `rescore_vector.oversample=3`); `adjustedK = ceil(k·oversample)` L2567-2582; `visitRatio = visit_percentage/100` L2594-2606. `supportsDimension` returns `true` for `bbq_disk` (the 64-dim floor applies to `bbq_hnsw`/`bbq_flat` only) — confirmed: a `dims: 1` `bbq_disk` mapping is accepted. |
| `docs/reference/elasticsearch/mapping-reference/dense-vector.md` L356-371; `docs/reference/query-languages/query-dsl/query-dsl-knn-query.md` L93-94 | Official option docs (`cluster_size` 64..65536 default 384; `visit_percentage` 0..100, "~1 % per shard per 1 M vectors", `num_candidates` ignored when set). |

Also looked at: `main` (commit `4eba9b52f5a2…`, 2026-11-15). The 9.2 code has since grown into `ES940`/`ES950`/`ESNext` formats with **balanced** k-means (`BalancedOTKMeansLocal`, Sinkhorn), auto-calibration of the visit budget (`IvfAutoCalibration`), bfloat16 raw vectors and an "ASH" quantizer. None of that is in 9.2.0; this note documents 9.2.0. Apache Lucene `main` (`8d85d89405fb…`) has no IVF format — only `lucene/sandbox/.../quantization/KMeans.java`.

Blog: "DiskBBQ: … beyond HNSW" (Elastic Search Labs, 2025-10-23, https://www.elastic.co/search-labs/blog/diskbbq-elasticsearch-introduction).

## 3. The algorithm (ES 9.2.0)

1. **Train — hierarchical k-means, per Lucene segment.** `targetSize = cluster_size`. If `N ≤ targetSize` → one centroid = mean of all vectors, no k-means (`HierarchicalKMeans.java:63-77`). Otherwise `clusterAndSplit`:
   - `k = clamp(int((N + targetSize/2) / targetSize), 2, 128)` (L95) — at most 128 centroids per level.
   - Seeds: first `k` vectors, then reservoir replacement with `Random(42)` (`KMeansLocal.java:54-70`).
   - Lloyd's on a reservoir **sample of `k·64` vectors** (L96, seed 42), **≤6 iterations**, stop early when no assignment changes; if sampled, one final full pass over all vectors (`KMeansLocal.java:375-385`). Only centroids whose membership changed are recomputed.
   - Any cluster with `count > 1.34·targetSize` is **recursively re-clustered** with the same rule (L135-140); empty clusters are dropped (L141-160). Final `nlist ≈ N / cluster_size` emerges, it is not set.
   - One more Lloyd pass over *all* centroids (sample `numCentroids·32`), **neighbour-aware**: each vector only compares against the ≤128 nearest centroids of its current one, with an early-out when it is already closer than that neighbourhood's max intra-distance (`KMeansLocal.java:136-187`; neighbourhoods from `NeighborHood.java`).
2. **Assign + SOAR.** Primary = nearest centroid. Second ("spilled") assignment = the centroid among the ≤128 neighbours minimising `soar(x,c) = ‖x−c‖² + λ·((x−c)·(x−c₁))²/‖x−c₁‖²`, λ = 1 (`KMeansLocal.java:214-300`, `DefaultESVectorUtilSupport.java:154-164`). Penalises second centroids whose residual is *parallel* to the first residual — i.e. prefers a second centroid that corrects in an orthogonal direction (Google SOAR). Vectors within 1e-16 of their centroid get no spill. In practice nearly **every vector is written twice**.
3. **Store (`.clivf`).** Per list: raw float centroid + `c·c` (so a query can be quantized against it), vInt count, then blocks of 16 vectors: 16 **1-bit OSQ codes** (`D/8` bytes, code of `x` relative to the list centroid), then 16× {lower, upper, additionalCorrection: float32; quantizedComponentSum: int16} = 14 B/vector corrections, with delta-coded doc ids per block (`ES920DiskBBQVectorsWriter.java:67-159, 540-548`). `.cenivf`: centroids **int7-quantized against the global centroid** (`writeCentroidsWithoutParents` L422-447); when `numCentroids > 256`, a parent layer is built by running the same hierarchical k-means *over the centroids* with targetSize 16 and both layers are stored (L365-420). `.mivf`: offsets, global centroid. Raw floats stay in the normal `.vec` file for rescoring.
4. **Search (per segment).** `visitRatio = visit_percentage/100`, or if 0/unset `log10(N)²·min(10000, max(num_candidates, 5k)) / N` (`AbstractIVFKnnVectorQuery.java:134-141`). Budget `maxVectorVisited = 2·visitRatio·N` — doubled because SOAR can show a vector twice (`IVFVectorsReader.java:289`).
   - Query → int7 against the global centroid; **flat scan** of all centroids (bulk int7 dot products), or score the parents then stream children of the best parents, keeping a buffer of `visitRatio·numCentroids·(numCentroids/(2·numParents))` children (`ES920DiskBBQVectorsReader.java:83-275`). No HNSW at search time.
   - Open lists nearest-first while `maxVectorVisited > expectedDocs` **or nothing competitive has been collected yet** (so at least one list is always read; `IVFVectorsReader.java:307-318`). Inside a list the query is re-quantized to **4 bits against that list's centroid** and scored asymmetrically (4-bit query × 1-bit codes, bulk of 16 via SIMD); a whole 16-block is skipped if its best possible score cannot beat the current k-th (L537) or if the filter rejects all 16 docs (L516-519).
   - **Filters** are `acceptDocs` bitsets applied inside the posting scan; with a filter the loop keeps opening lists until `min(2·filteredCount·ratioVisited, expected/2)` docs have actually been scored (L319-331) — selective filters make it read more lists rather than return fewer hits.
   - Collector keeps `2·k` candidates per segment (duplicates from SOAR), segments merged to `k` (`AbstractIVFKnnVectorQuery.java:120, 152`).
5. **Rescore.** By default `rescore_vector.oversample = 3`: the IVF is asked for `k' = ceil(3k)` hits, those are re-scored with the raw float vectors and the top `k` returned (`DenseVectorFieldMapper.java:2567-2582`, `RescoreKnnVectorQuery`). `oversample: 0` turns it off; max 10. Rescoring cannot recover a vector whose list was never opened.
6. **Updates / deletes.** There is no incremental assignment: new docs go to a *new* segment which gets its own k-means on flush; deletes are a bitset (`acceptDocs`) at search time; **every merge re-runs k-means over all merged vectors** (`IVFVectorsWriter.java:272-420`). A `force_merge` to one segment is what gives you one global IVF.

## 4. Hyperparameters

| Name | Build / query | Default | What it trades | FAISS analogue |
|---|---|---|---|---|
| `index_options.type: bbq_disk` | build (mapping) | — | selects this format; `element_type` must be `float` in 9.2 | `IndexIVF*` |
| `index_options.cluster_size` | build (new segments / merges) | 384 (64..65536) | **vectors per list**; smaller = more, purer lists = higher recall per vector scanned, more centroids to scan | **`nlist` inverted**: `nlist ≈ N / cluster_size` |
| `index_options.default_visit_percentage` | build-time default, query-overridable | 0 = dynamic | default scan budget as % of the shard's vectors | default `nprobe`, expressed as a fraction of *vectors* not lists |
| `knn.visit_percentage` | query | mapping default | % of vectors to score per shard; ES doubles it internally to cover SOAR copies; 0 → dynamic formula above | `nprobe` ≈ `visit% · N / (100 · cluster_size)` lists |
| `knn.k` | query | — | results; IVF collector keeps `2k` | `k` |
| `knn.num_candidates` | query | `1.5·k` | for `bbq_disk` it only feeds the dynamic budget when `visit_percentage` is unset; **in the top-level `knn` section it becomes the per-shard k, so it also sizes the rescore pool** (`KnnSearchBuilder.java:501`) | none (HNSW `efSearch` vocabulary) |
| `rescore_vector.oversample` | mapping default, query-overridable | 3 (0 = off, else 1 < x ≤ 10) | re-score `ceil(k·oversample)` 1-bit hits with raw floats; raises recall ceiling, costs random reads of `.vec` | `IndexRefineFlat` / `k_factor` |
| `similarity` | build | — | `l2_norm`, `dot_product`, `cosine` (cosine → vectors normalised); score shown is `1/(1+d²)` for L2 | metric |
| `index.compound_format` (index setting) | build | 0.1 | unrelated to recall; set `false` to see `.clivf/.cenivf/.vec` sizes in `_stats` | — |
| *fixed in code* `centroidsPerParentCluster` | — | 16 | parent layer appears when `nlist > 256`; children buffer = `visitRatio·nlist·8` | no analogue (FAISS uses a `quantizer` index, e.g. HNSW, instead) |
| *fixed* `soarLambda` | — | 1.0 | SOAR penalty weight; `−1` disables (used for the parent layer only) | **no analogue** (multiple assignment) |
| *fixed* `MAX_ITERATIONS=6`, `SAMPLES_PER_CLUSTER=64`, `MAXK=128`, split margin 1.34 | — | — | k-means effort and shape | `niter=25`, `max_points_per_centroid=256`, no per-level cap |
| *fixed* code bits | — | 1 bit/dim (+14 B), query 4 bit, centroids int7 | compression vs accuracy | `nbits=1`-ish; no `m` (no PQ) |

## 5. Toy: A..H = 1, 2, 4, 7, 8, 11, 14, 15, "nlist = 3" (`targetSize = 3`), query 9.5

**Why a simulation:** ES rejects `cluster_size: 3` (`"cluster_size must be between 64 and 65536"`) and with 8 ≤ 64 vectors it creates one mean centroid (`HierarchicalKMeans.java:63-77`). `lab/ivf_elasticsearch_toy.py` re-implements the 9.2.0 code path above (including a bit-exact `java.util.Random(42)`) with exact float distances — 1-bit codes are meaningless at `d = 1` (the code is just "above/below the centroid"). Output of `python -I lab/ivf_elasticsearch_toy.py`:

1. `clusterAndSplit(n=8, targetSize=3)`: `k = clamp(int((8+1.5)/3), 2, 128) = 3`; sample `m = min(3·64, 8) = 8` → no sampling.
2. Seeds via `pickInitialCentroids`: start `[1, 2, 4]`, reservoir with `Random(42)` replaces → **`[7, 11, 15]`** (D, F, H).
3. Lloyd: it.1 `[4.4, 11.0, 14.5]` assignments `[0,0,0,0,0,1,2,2]`; it.2 `[3.5, 9.5, 14.5]` `[0,0,0,0,1,1,2,2]`; it.3 `[2.333, 8.667, 14.5]` `[0,0,0,1,1,1,2,2]`; it.4 no change → stop. Sizes `[3, 3, 2]`; none > `1.34·3 = 4.02` so no recursive split, none empty.
4. Final neighbour-aware pass: 3 centroids ≤ 128 → no neighbourhoods, plain Lloyd, already converged.
5. **SOAR** (λ = 1). In 1-D every residual is parallel, so `soar(x,c) = ‖x−c‖² + ‖x−c‖²·‖x−c₁‖²/‖x−c₁‖² = 2‖x−c‖²` — SOAR degenerates to "second-nearest centroid". Spills: A,B,C → c1; D,E → c0; F → c2; G,H → c1.
6. **Posting lists** (primary + SOAR copies):
   - `c0 = 2.333`: {A, B, C} + {D, E} → 5 codes
   - `c1 = 8.667`: {D, E, F} + {A, B, C, G, H} → **8 codes (every vector)**
   - `c2 = 14.5`: {G, H} + {F} → 3 codes
   - 16 codes for 8 vectors = **2.0× storage** of the 1-bit codes (each 1 bit/dim + 14 B).
7. **Query 9.5, k = 3.** Exact answer E(8), F(11), D(7). Centroid scan nearest-first: c1 (|9.5−8.667| = 0.83), c2 (5.0), c0 (7.17).
   - "nprobe = 1" ⇔ `visit_percentage = 50` → `maxVectorVisited = int(2·0.5·8) = 8`; opens **c1** only (8 codes, `expectedDocs = 8`, budget exhausted) → returns **E, F, D — no miss**.
   - "nprobe = 2" ⇔ `visit_percentage = 68.8` → budget 11; opens c1 then **c2** (F, G, H) → E, F, D.
   - **No miss is possible in this toy**: with 3 lists and SOAR, c1 ended up holding all 8 vectors, so one list already answers every query. The script also runs the boundary query **5.5** (exact C, D, E; |5.5−c0| = |5.5−c1| = 3.167, float rounding picks **c1**): "nprobe = 1" opens c1 → C, D, E, no miss. *Plain* IVF (primaries only) would have opened c1 = {D, E, F} and **missed C** — that boundary miss is exactly what SOAR's second assignment buys, at the price of 2× code storage.
8. Rescoring: default `oversample = 3` asks the IVF for `ceil(9)` 1-bit hits, re-scores with raw floats, keeps 3 — irrelevant here (exact distances), but it never reaches a vector in an unopened list.

**Real ES 9.2.0 on the same 8 values** (`dims: 1`, `bbq_disk`, `cluster_size: 64`, force-merged): one centroid (8.0), one list of 8; `knn 9.5, k=3` returns `E 0.3077, F 0.3077, D 0.1379` (score = 1/(1+d²)) at `visit_percentage` 100 **and** 1 — a single list is always opened, so recall is 1.0 by construction.

## 6. Scale: N = 187,043 × 256, 1000 queries, recall@10 vs numpy L2 ground truth

Real ES 9.2.0 (Docker, `-m 4g`, heap 2 g, Apple Silicon M-series, 1 shard, force-merged to **1 segment** so there is one IVF). Data generator = the course's (50 topic centres ± 0.35 noise in 256-d). Queries via the top-level `knn` section, one HTTP request per query; `took` = ES server ms, `wall` = client round-trip ms (sub-ms `took` is integer-rounded by ES).

**Mapping the dials.** `nlist = 4096` is not expressible: that needs `cluster_size ≈ 46`, below the floor of 64. Nearest allowed `cluster_size = 64 → nlist ≈ 2923`; the repeat uses `cluster_size = 183 → nlist ≈ 1022`. `nprobe~` is converted to `visit_percentage = nprobe · cluster_size / N · 100` (ES then doubles the budget for SOAR). Four query configurations, because for `bbq_disk` **three** dials set recall: lists opened (`visit_percentage`), re-scoring on/off (`oversample`), and — in the top-level `knn` API — the pool re-scored (`ceil(num_candidates · oversample)`, since `num_candidates` is the per-shard k there):

- **A** `oversample 0, num_candidates 100` — pure 1-bit ranking, no rescoring
- **B** `oversample 3, num_candidates 15` — ES defaults (`1.5·k`) → 45 re-scored
- **C** `oversample 3, num_candidates 100` → 300 re-scored
- **D** `oversample 3, num_candidates 500` → 1500 re-scored

### `cluster_size = 64` (nlist ≈ 2923)

Build: bulk 10.1 s + force-merge (hierarchical k-means + SOAR + 1-bit coding, single thread) **3.3 s**; store 218.4 MB = `.vec` 191.5 MB raw floats (needed for rescoring) + **`.clivf` 22.9 MB** 1-bit postings + `.cenivf` 1.6 MB centroids. 22.9 MB ≈ 187 k × 2 copies × (32 B code + 14 B corrections) + doc ids + 2923 list headers — i.e. **≈ every vector stored twice** (SOAR), ~8.4× smaller than the floats even so.

| nprobe~ | visit % | A recall | A took/wall ms | B recall | B took/wall | C recall | C took/wall | D recall | D took/wall |
|---:|---:|---:|---|---:|---|---:|---|---:|---|
| 1 | 0.034 | 0.111 | 0.0 / 1.2 | 0.111 | 0.0 / 0.9 | 0.267 | 0.0 / 1.1 | 0.737 | 1.1 / 2.6 |
| 2 | 0.068 | 0.136 | 0.0 / 1.0 | 0.164 | 0.0 / 1.0 | 0.279 | 0.0 / 1.3 | 0.756 | 1.0 / 2.1 |
| 4 | 0.137 | 0.194 | 0.0 / 1.3 | 0.267 | 0.0 / 1.3 | 0.305 | 0.1 / 1.7 | 0.792 | 1.0 / 2.1 |
| 8 | 0.274 | 0.265 | 0.0 / 1.0 | 0.419 | 0.0 / 1.4 | 0.463 | 0.2 / 2.1 | 0.829 | 1.6 / 4.3 |
| 16 | 0.547 | 0.327 | 0.0 / 1.0 | 0.584 | 0.0 / 1.1 | 0.707 | 0.0 / 1.3 | 0.845 | 1.0 / 2.0 |
| 32 | 1.095 | 0.366 | 0.0 / 1.4 | 0.704 | 0.0 / 1.4 | 0.926 | 0.1 / 1.4 | 0.937 | 1.0 / 2.0 |
| **64** | **2.190** | 0.382 | 0.0 / 1.1 | 0.748 | 0.0 / 1.2 | **0.991** | 0.0 / 1.3 | **1.000** | 1.0 / 2.1 |
| 128 | 4.380 | 0.382 | 0.1 / 1.3 | 0.748 | 0.0 / 1.4 | 0.991 | 0.0 / 1.4 | 1.000 | 1.0 / 2.4 |
| 256 | 8.759 | 0.382 | 0.0 / 1.6 | 0.748 | 0.1 / 1.7 | 0.991 | 1.0 / 1.9 | 1.000 | 1.1 / 2.8 |

ES default dial (`visit_percentage` unset → dynamic `log10(N)²·max(100, 50) ≈ 2779` vectors = 1.49 %, `oversample 3`, `num_candidates 100`): **recall 0.991, took 2.0 ms, wall 3.0 ms**.

**Smallest nprobe-equivalent reaching recall ≥ 0.95: 64 (`visit_percentage = 2.19 %`) with rescoring (C or D). Never reached without rescoring (A plateaus at 0.38) or with ES's default `num_candidates` (B plateaus at 0.75).**

### `cluster_size = 183` (nlist ≈ 1022)

Build: bulk 8.9 s + force-merge **2.1 s**; store 214.1 MB (`.clivf` 19.7 MB, `.cenivf` 0.56 MB).

| nprobe~ | visit % | A recall | A took/wall ms | B recall | B took/wall | C recall | C took/wall | D recall | D took/wall |
|---:|---:|---:|---|---:|---|---:|---|---:|---|
| 1 | 0.098 | 0.165 | 0.0 / 0.9 | 0.227 | 0.0 / 1.1 | 0.294 | 0.0 / 1.4 | 0.767 | 1.0 / 1.9 |
| 2 | 0.196 | 0.212 | 0.0 / 1.8 | 0.315 | 0.0 / 1.3 | 0.339 | 0.0 / 1.5 | 0.793 | 1.0 / 1.8 |
| 4 | 0.391 | 0.276 | 0.0 / 1.2 | 0.468 | 0.0 / 1.0 | 0.547 | 0.0 / 1.5 | 0.815 | 1.0 / 1.9 |
| 8 | 0.783 | 0.339 | 0.0 / 1.4 | 0.628 | 0.0 / 1.3 | 0.801 | 0.3 / 1.9 | 0.816 | 1.0 / 2.0 |
| **16** | **1.565** | 0.374 | 0.1 / 2.2 | 0.720 | 0.3 / 2.9 | **0.975** | 0.9 / 3.3 | **0.989** | 1.1 / 3.2 |
| 32 | 3.131 | 0.377 | 0.2 / 2.3 | 0.734 | 0.3 / 3.0 | 0.988 | 0.8 / 3.0 | 1.000 | 1.1 / 3.4 |
| 64 | 6.262 | 0.377 | 1.0 / 3.3 | 0.734 | 0.7 / 3.0 | 0.988 | 0.9 / 3.2 | 1.000 | 1.3 / 3.3 |
| 128 | 12.523 | 0.377 | 0.9 / 3.2 | 0.734 | 0.9 / 3.2 | 0.988 | 0.9 / 3.2 | 1.000 | 1.6 / 3.7 |
| 256 | 25.047 | 0.377 | 1.1 / 3.3 | 0.734 | 1.1 / 3.6 | 0.988 | 1.2 / 3.3 | 1.000 | 2.0 / 3.9 |

Default dial: recall 0.988, took 1.0 ms, wall 2.9 ms. **Smallest nprobe-equivalent for ≥ 0.95: 16 (`visit_percentage = 1.57 %`) with rescoring (C/D)**; A and B never reach it. Same ~1.5–2 % of vectors as with `cluster_size = 64` — the vector budget, not the list count, is what matters, which is why ES exposes `visit_percentage` rather than `nprobe`.

### What was surprising

- **The 1-bit code ceiling is low on this dataset**: recall@10 saturates at **0.38** without rescoring and the plateau is reached at ~1–2 % of vectors visited — beyond that, opening more lists changes nothing because the ranking error is quantization, not coverage. The data is adversarial for 1 bit/dim: all neighbours of a query share its topic centre and differ only by small Gaussian noise, so most 1-bit codes within a list are near-identical. Rescoring is not optional with `bbq_disk`; Elastic ships it on by default for this reason.
- **`num_candidates` silently decides the rescore pool** in the top-level `knn` API (per-shard k = `num_candidates`, `KnnSearchBuilder.java:501`), even though the docs say it is "ignored" for `bbq_disk` when `visit_percentage` is set. ES defaults (`1.5·k = 15` → 45 re-scored) cap at 0.75 here; `num_candidates 100` lifts the same scan to 0.99. In the `query: {knn: …}` DSL, `k` and `oversample` alone set the pool (`k·oversample = 30` → 0.64 at 2.19 %; `oversample 10` → 0.89).
- **Latency is flat**: ~1 ms server time whether 0.03 % or 25 % of 187 k vectors are scanned — 1-bit bulk scoring of ~47 k codes is cheaper than the HTTP round trip; the only visible cost is the rescore pool (D ≈ +1 ms for 1500 raw-float reads).
- Build is fast: ~2–3 s of k-means + coding for 187 k × 256 on one thread, vs. minutes for an HNSW graph of the same size — consistent with Elastic's 11× claim.

## 7. What is specific to Elasticsearch vs plain FAISS IVF

- **Inverse parameterisation**: you set vectors-per-list (`cluster_size`), `nlist` falls out of a recursive k-means that never runs more than 128 centroids per level and only re-splits clusters >134 % of target; FAISS runs one flat k-means with `nlist` fixed and 25 iterations.
- **SOAR double assignment** is always on (λ = 1): ~every vector appears in two lists, so the 1-bit code store is ~2× and ES silently doubles the visit budget; FAISS IVF has single assignment.
- **Quantization is baked into the IVF**: 1-bit OSQ codes relative to the *list* centroid with 14 B of corrections, 4-bit asymmetric query, int7 centroids — closest FAISS reading is "IVF + 1-bit RaBitQ-style codes + `IndexRefineFlat`", not `IVFPQ` (no sub-quantizers, no ADC tables).
- **Budget in vectors, not lists**, with a `num_candidates`-driven dynamic default (`log10(N)²·max(num_candidates, 5k)` vectors) and a filter-aware extension; no `nprobe` integer anywhere.
- **Per-segment indexes, re-clustered on every merge**; no incremental assignment, no list splitting; deletes are bitsets. Recall and latency depend on segment count until you `force_merge`.
- **Rescoring on by default (`oversample 3`)** — and in the top-level `knn` API `num_candidates` is the per-shard k, so it decides how many 1-bit hits get re-scored. Both dials matter for recall; `visit_percentage` alone does not.

Why Elastic positions it as the disk / low-memory alternative to HNSW (their claims, blog 2025-10-23):

- Only centroids (int7) need to be hot; postings are **sequential block reads** of 1-bit codes, bulk-scored off-heap — nothing like an HNSW graph needs to be resident.
- Memory need scales with `nlist`, not `N`; HNSW+BBQ needs the whole graph + codes in page cache or "falls apart" (their words) when RAM is restricted.
- Index build is **~11× faster** in their 1 M-vector test: DiskBBQ 94,075 ms vs HNSW-BBQ 1,054,319 ms, at similar in-memory latency (~4.0 ms vs ~3.4 ms) and recall (91 % vs 92 %) — numbers are Elastic's, not reproduced here.
- Recommended when ≤95 % recall is acceptable and cost matters; HNSW when you need 99 %+ and have the RAM.

## 8. Reproduce

```bash
# venv
python3 -m venv /tmp/venv-ivf-es && /tmp/venv-ivf-es/bin/pip install numpy requests orjson
# Elasticsearch 9.2.0 (needs Docker daemon running)
docker run -d --name es-ivf -m 4g -p 9200:9200 -e discovery.type=single-node \
  -e xpack.security.enabled=false -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  docker.elastic.co/elasticsearch/elasticsearch:9.2.0
curl -s localhost:9200   # wait until it answers
cd modules/05-vector-databases-and-ann
/tmp/venv-ivf-es/bin/python -I lab/ivf_elasticsearch_toy.py            # simulation + real 8-doc run (add --no-es to skip ES)
/tmp/venv-ivf-es/bin/python -I lab/ivf_elasticsearch_scale.py --cluster-sizes 64,183   # ~5 min
# source read
git clone --depth 1 --filter=blob:none --sparse https://github.com/elastic/elasticsearch && cd elasticsearch
git fetch --depth 1 origin tag v9.2.0 && git checkout v9.2.0
git sparse-checkout set server/src/main/java/org/elasticsearch/index/codec/vectors libs/simdvec/src/main/java docs/reference
```
