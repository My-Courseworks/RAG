# IVF in LanceDB (Lance format)

- Flavours: `IVF_FLAT`, `IVF_SQ`, `IVF_PQ` (the default; PQ on **residuals** for L2/cosine), `IVF_RQ` (RaBitQ), `IVF_HNSW_SQ` / `IVF_HNSW_PQ` (HNSW as the in-partition sub-index). Coarse quantizer is k-means (random init, Lloyd's; hierarchical k-means when `num_partitions > 256`).
- Embedded Rust library behind a Python/JS/Rust client; the index is files in the table directory, each partition a contiguous row range, designed for object storage.
- Ran the real thing here: lancedb 0.41.0 on macOS arm64 (no GPU). Toy at d=1 on the real `IVF_FLAT` index; scale test on `IVF_FLAT` and `IVF_PQ`.

## 1. Source read

Repo `https://github.com/lancedb/lance`, commit `e307b5e2cf7d612348f0d59586715e0b3bfd17cc` (shallow clone, 2026-11-15). The `lancedb` client only forwards parameters; the index lives in Lance.

| Stage | File | What it does |
|---|---|---|
| train | `rust/lance-index/src/vector/kmeans.rs` | `KMeansParams` defaults L132-147; random init L227-247; Lloyd's loop L1210-1300; convergence test L335-341; empty-cluster split L250-280; hierarchical gate L1756-1760; sampling `sample_rate * k` L2130-2170 |
| train (sample) | `rust/lance/src/index/vector/utils.rs` | `maybe_sample_training_data` L305-370: scans all rows when `N <= sample_rate * k`, otherwise draws a random sample of that many rows |
| train (cap) | `rust/lance/src/index/vector.rs` | `supported_num_partitions` L658-682 and `vectors_per_partition` L700-716: a quantized index trains at most `N / sample_rate` partitions |
| defaults | `rust/lance-index/src/vector/ivf/builder.rs` | `IvfBuildParams` L20-109 (`max_iters` 50, `sample_rate` 256); `recommended_num_partitions` L194-199 (`N / target_partition_size`, capped at 4096) |
| defaults | `rust/lance-index-core/src/lib.rs` | `target_partition_size` L274-289: 4096 rows for `IVF_FLAT`, 8192 for `IVF_PQ`/`IVF_SQ`, 2^20 for `IVF_HNSW_*` |
| defaults | `rust/lance-index/src/vector/pq/builder.rs` | `PQBuildParams` L29-70 (`num_sub_vectors` 16, `num_bits` 8, `max_iters` 50, `sample_rate` 256); `use_residual` L82-84 (L2 and cosine only) |
| assign | `rust/lance-index/src/vector/ivf/transform.rs` | `PartitionTransformer::transform` L127-160: `compute_partitions_arrow_array` = nearest centroid, one partition per row |
| assign | `rust/lance-index/src/vector/residual.rs` | `compute_residual` L115+: `vector - centroid[partition]` before PQ |
| store | `rust/lance-index/src/vector/ivf/storage.rs` | `IvfModel` L27-40: `centroids`, `offsets[]`, `lengths[]`; `row_range(partition)` L143-147 |
| search (route) | `rust/lance-index/src/vector/ivf.rs` | `find_partitions` L381-392: rank all centroids by f32 distance, keep the first `nprobes` |
| search (how many) | `rust/lance/src/io/exec/knn/adaptive_probe.rs` | Auto probing: profiles L330-370, initial budget L433-470, legacy rule L250-266 |
| search (late) | `rust/lance/src/io/exec/knn.rs` | L2296-2304: with fewer than `k` results and `maximum_nprobes` unmet, probe more partitions |
| search (partition) | `rust/lance/src/index/vector/ivf/v2.rs` | `load_partition_entry` L2593-2607 reads exactly `row_range(partition)`; `search_in_partition` L2964+ |
| search (PQ) | `rust/lance-index/src/vector/pq/storage.rs` | `dist_calculator` L800, `build_distance_table_l2` L999: one ADC table per (query, partition), codes scanned exhaustively |
| rescore | `rust/lance/src/dataset/scanner/vector.rs` | L469-477: with `refine_factor` the ANN node is asked for `k * refine_factor`, then `take` the raw vectors and `flat_knn` re-ranks |
| query params | `rust/lance-index/src/vector.rs` | `ProbeParams::resolve` L141-148; `Query` L154-215 (`minimum_nprobes`, `maximum_nprobes`, `search_effort`, `ef`, `refine_factor`) |

## 2. Algorithm

1. **Sample.** If `N > sample_rate * num_partitions` draw that many random rows, else use every row (`train_kmeans`, kmeans.rs L2150-2160). Default 256 per partition, same constant as FAISS's warning threshold.
2. **Cap.** For a quantized index (`IVF_PQ`, `IVF_SQ`, `IVF_RQ`) the partition count is silently reduced to `min(requested, N / sample_rate)` with a `WARN` log (vector.rs L658-682). `IVF_FLAT` only requires `N >= num_partitions`. At N=187,043 that is 730 partitions, not the 4096 asked for (see scale test).
3. **Train.** `KMeanInit::Random`: `k` distinct training rows become the centroids (kmeans.rs L227-247). No k-means++. Lloyd's for at most `max_iters` (50) or until `|loss_prev - loss| < 1e-4 * loss`. `redos` = 1. Seed `None` -> `SmallRng::from_os_rng()` (L204-210): **not reproducible, and not exposed by lancedb**.
   - `num_partitions > 256` switches to hierarchical k-means (L1756-1760): a tree with fan-out `hierarchical_k = 16`, every node running a small k-means and handing its children a centroid quota proportional to their row share; `refine_iters` global Lloyd's passes afterwards, default 0. Bounds training cost at 4096 partitions; costs a little WCSS at sub-tree boundaries.
   - Empty cluster: after each Lloyd's step every empty centroid is re-seeded as a +/-1/1024 perturbation of the largest cluster's centroid (`split_clusters` L250-280); more than 10% empty logs a `WARN` (L774-791).
   - Cosine is trained as L2 on normalised vectors; dot normalises the centroids after every step (L1246, L984-996).
   - `balance_factor` (default 0, no lancedb surface) adds a cluster-size penalty to the assignment; off by default so partitions are **not** balanced.
4. **Assign.** Each row goes to its single nearest centroid (`compute_partitions`, transform.rs L159). No multi-assignment, no boundary replication, no closure.
5. **Quantize.** `IVF_PQ` with L2/cosine stores PQ codes of the **residual** `x - c[p]` (`use_residual`, pq/builder.rs L82-84); with dot it quantizes the raw vector. `num_sub_vectors` must divide `d` (pq/utils.rs L31); `num_bits` 8 (or 4, then `num_sub_vectors` must be even). One shared codebook for all partitions, trained once on a `256 * 256` sample of residuals. `IVF_FLAT` keeps float32 copies of the vectors inside the index; `IVF_SQ` int8.
6. **Store.** Rows are shuffled into partition order and written so that partition `p` is the contiguous range `offsets[p] .. offsets[p] + lengths[p]` of the index file (`IvfModel`, storage.rs L27-40, L143-147). Two files per index: `index.idx` (centroids, offsets, lengths, metadata) and `auxiliary.idx` (the per-partition payload: row ids plus codes or vectors). One partition = one byte-range read, which is the design point for S3/GCS.
7. **Route.** Every query computes the f32 squared-L2 (or dot) distance to **all** centroids (`find_partitions`, ivf.rs L381-392; a flat scan of the centroid array, no tree or graph over centroids) and sorts them.
8. **How many partitions.** Three regimes (adaptive_probe.rs L42-70):
   - `nprobes(n)` set: `minimum = maximum = n`, policy **Fixed**. The classic FAISS dial.
   - nothing set (lancedb default): `minimum_nprobes = 1`, `maximum_nprobes = None`, `search_effort = 0.5`, policy **Adaptive**. For L2 with `2 <= k <= 10` it opens every partition whose centroid distance is within `(1 + 0.265) x nearest`, clamped to `[6, 24]` partitions (L330-370, L433-470); other k buckets and metrics have their own margin/floor/cap. Then **late search**: if fewer than `k` rows came back (filters, deletions, tiny partitions) it keeps opening partitions in centroid order until it has `k` or runs out (knn.rs L2296-2304).
   - `minimum_nprobes(a).maximum_nprobes(b)`: Adaptive within `[a, b]`.
9. **Scan a partition.** Load the row range (`load_partition_entry`, v2.rs L2593-2607; cached after first use), then exhaustive distance over every row in it: exact f32 for `IVF_FLAT`, int8 for `IVF_SQ`, ADC for `IVF_PQ` (one `256 x num_sub_vectors` lookup table per partition because the codes are residuals to that partition's centroid: pq/storage.rs L999). `IVF_HNSW_*` runs an HNSW walk inside the partition with `ef`. No early termination inside a partition; partitions run concurrently.
10. **Merge and rescore.** Top-k across the probed partitions by index distance. With `refine_factor = r` the index is asked for `k * r` rows, their raw vectors are fetched from the table (`take`) and re-ranked with exact distance (`flat_knn`), top `k` kept (scanner/vector.rs L469-477). Rows in fragments the index has not seen yet (appended after the build) are brute-forced and merged in the same plan (`knn_combined`, L896-920); deleted rows are masked out by the prefilter.
11. **Filters.** `where(...)` with the default `prefilter=True` builds a row-id mask before the ANN node; a probed partition only scores rows in the mask, and late search keeps probing until `k` survivors exist. `postfilter` filters the `k` ANN results and may return fewer.
12. **Updates.** Appends land in new fragments and stay unindexed (`num_unindexed_rows` in `index_stats`) until `optimize()`, which assigns the new rows to the **existing** centroids (no retraining) and appends a new index delta; `create_index(replace=True)` retrains. Partitions are never split or merged by inserts; `optimize` has a split/join plan for partitions that drift past `4x` / below `25%` of `target_partition_size` (builder.rs L3176-3190).

## 3. Hyperparameters (lancedb 0.41.0 Python names)

| Name | When | Default | Trades | FAISS analogue |
|---|---|---|---|---|
| `num_partitions` | build | `None` -> `N / target_partition_size` clamped to `[1, 4096]`; docs say "sqrt(N)" (stale) | more lists: cheaper per-probe scan, more routing cost, more partitions needed per recall | `nlist` |
| `target_partition_size` | build | 4096 (`IVF_FLAT`), 8192 (`IVF_PQ`, `IVF_SQ`), 2^20 (`IVF_HNSW_*`) | the preferred way to size lists; `num_partitions` is documented as deprecated in Lance | none (derives `nlist`) |
| `index_type` / config class | build | `IVF_PQ` | `IvfFlat` exact in-list, `IvfSq` 4x smaller, `IvfPq` 8-32x smaller, `IvfRq` 1-bit RaBitQ, `HnswPq`/`HnswSq` graph in-list | `IndexIVFFlat` / `IndexIVFScalarQuantizer` / `IndexIVFPQ` / `IndexIVFRaBitQ` / none |
| `num_sub_vectors` | build | `d / 16` if divisible, else `d / 8`, else 1 | code bytes per vector (8 bits each); must divide `d` | `m` |
| `num_bits` | build | 8 | 4 halves the code and uses a quantized LUT | `nbits` |
| `max_iterations` | build | 50 | Lloyd's cap for both the IVF and PQ k-means | `ClusteringParameters.niter` (FAISS 25) |
| `sample_rate` | build | 256 | training rows per centroid **and** the hard cap `num_partitions <= N / sample_rate` for quantized indices | `max_points_per_centroid` (same 256) |
| `distance_type` | build | `l2` | `cosine` normalises then trains L2; `dot` disables residual PQ; `hamming` for uint8 | `METRIC_L2` / `METRIC_INNER_PRODUCT` |
| `accelerator` | build | `None` | `"cuda"` / `"mps"` trains k-means in PyTorch; not used here | none |
| `nprobes(n)` | query | unset -> **Adaptive** (see row below); `n` fixes `min = max = n` | partitions scanned vs recall | `nprobe` |
| `minimum_nprobes` / `maximum_nprobes` | query | 1 / `None` (= all) | bounds for Adaptive; `maximum` also bounds late search | none |
| `search_effort(e)` | query | 0.5 | 0 = start at `minimum_nprobes`, 1 = start at all partitions; 0.1 / 0.25 calibrated to 80% / 90% routing recall; any explicit value makes `nprobes(n)` ignored | none |
| `refine_factor(r)` | query | `None` | fetch `k * r` by index distance, re-rank with exact vectors; cost is `k * r` random row reads | `IndexRefineFlat` with `k_factor` |
| `ef(e)` | query | `1.5 * k * refine_factor`, floor `DEFAULT_MIN_EF` | HNSW beam inside each partition (`IVF_HNSW_*` only) | `efSearch` |
| `prefilter` | query | `True` | mask before ANN (late search guarantees `k`) vs filter after | `IDSelector` / none |
| `fast_search` | query | `False` | `True` skips the brute-force pass over unindexed fragments | none |
| `optimize()` | maintenance | manual | indexes appended fragments against existing centroids; `replace=True` rebuild retrains | `add` (no retrain) / rebuild |

No lancedb surface for: k-means seed, `redos`, `balance_factor`, `hierarchical_k`, `refine_iters`, precomputed centroids (`IvfBuildParams.centroids`, Rust/pylance only).

## 4. Toy: A..H = 1, 2, 4, 7, 8, 11, 14, 15 in 3 partitions, real `IVF_FLAT` at d=1

Script: `lab/ivf_lancedb_toy.py`. d=1 is accepted. `IVF_PQ` is not usable here (`num_sub_vectors` must divide 1, and a PQ codebook needs 256 rows).

Caveats read from the code and confirmed by running: the seed is OS entropy and not exposed, so the k-means outcome varies per build; the API does not return centroids, so they are recovered as partition means (exactly what Lloyd's stores) and checked against the index's routing. The script rebuilds until it gets the partitioning below and reports what else it saw. Across 11 builds during this write-up three partitionings appeared:

| Partitioning | Centroids | WCSS | Seen |
|---|---|---|---|
| {A,B,C} {D,E,F} {G,H} | 2.333, 8.667, 14.5 | 13.83 | 6 |
| {A,B,C} {D,E} {F,G,H} | 2.333, 7.5, 13.333 | 13.83 | 3 |
| {A,B} {C,D,E} {F,G,H} | 1.5, 6.333, 13.333 | 17.83 | 2 |

The first two are tied global optima; the third is a worse local optimum that random init with `redos = 1` lands in about one build in five. The steps below are the first partitioning.

1. **Sample.** `N = 8 <= 256 * 3`, so all 8 rows train (`Sample training data: scanning all 8 rows` in the log).
2. **Init.** Three of the eight values are drawn as starting centroids (`kmeans_random_init`).
3. **Lloyd's** converges (tolerance 1e-4 relative) to centroids `2.3333`, `8.6667`, `14.5`; partition 0 = {A,B,C}, 1 = {D,E,F}, 2 = {G,H}. Recovered centroids are Voronoi-consistent with the routing the index performed for all 8 values.
4. **Assignment** (squared distance to each centroid, nearest wins): A 1.78/58.8/182 -> 0; B 0.11/44.4/156 -> 0; C 2.78/21.8/110 -> 0; D 21.8/2.78/56.3 -> 1; E 32.1/0.44/42.3 -> 1; F 75.1/5.44/12.3 -> 1; G 136/28.4/0.25 -> 2; H 160/40.1/0.25 -> 2.
5. **Store.** Two files under `toy.lance/_indices/<uuid>/`: `index.idx` 335 bytes (the `IvfModel`: 3 centroids and, by construction of the shuffle, offsets `[0, 3, 6]` with lengths `[3, 3, 2]`) and `auxiliary.idx` 879 bytes (the 8 float32 vectors plus row ids, written in partition order). Not decoded here; the layout is read from `storage.rs`.
6. **Query 5.5, k=3.** Exact answer: C(2.25), D(2.25), E(6.25). Centroid distances in f32: partition 0 `10.027779`, partition 1 `10.027780`, partition 2 `81.0`: 5.5 is the exact midpoint of 2.333 and 8.667, and f32 rounding decides the order.
   - `nprobes(1)` opens {A,B,C} only -> returns C(2.25), B(12.25), A(20.25). **Misses D and E**: recall@3 = 1/3, and two of the three returned rows are wrong.
   - `nprobes(2)` opens {A,B,C} and {D,E,F} -> C, D, E. Exact.
   - no `nprobes` (Adaptive): L2, k=3 -> margin 0.265, floor 6 partitions > 3 available, so all three open -> exact. On this toy the default is brute force.
7. **Query 9.5, k=3.** Exact: E(2.25), F(2.25), D(6.25). Centroid distances 51.4 / 0.69 / 25.0 -> partition 1 first. `nprobes(1)` opens {D,E,F} -> E, F, D, exact; no miss because the three nearest values share a partition. With the second partitioning ({D,E} alone), `nprobes(1)` returns only **2 rows** for k=3: a Fixed probe count never late-searches, so a small partition under-fills the result.
8. **Plan.** `explain_plan` shows the dial: `ANNIvfPartition: minimum_nprobes=1, maximum_nprobes=Some(1)` for `nprobes(1)`, `minimum_nprobes=1, maximum_nprobes=None` when unset.

## 5. Scale: N = 187,043, d = 256, 1000 queries, recall@10

Script: `lab/ivf_lancedb_scale.py --nlist1024`. L2, data per the common generator (50 Gaussian topics, noise 0.35). One query at a time through the Python client; ms/query includes client overhead (~1 ms). Numbers below are from one run; the OS-seeded k-means moves them by a few 1e-3 between runs.

### Builds

| Index | Partitions requested -> built | Build (s) | Index dir on disk (MB) | Rows |
|---|---|---|---|---|
| `IVF_FLAT` nlist=4096 | 4096 -> 4096 (186 empty) | 0.9 | 196.2 | 187,043 |
| `IVF_PQ` nlist=4096, m=32, 8 bits | 4096 -> **730** | 4.2 | 7.6 | 187,043 |
| `IVF_FLAT` nlist=1024 | 1024 -> 1024 | 1.2 | 193.0 | 187,043 |

Raw table data files: 191.9 MB (float32 payload 191.5 MB). `IVF_FLAT` duplicates every vector into the index (196 MB on top of the table); `IVF_PQ` stores 32 bytes + row id per vector (7.6 MB, 25x smaller).

### Sweep: recall@10 / ms per query (warm cache)

| nprobes | `IVF_FLAT` 4096 | `IVF_PQ` 730, no refine | `IVF_PQ` 730, `refine_factor=10` | `IVF_FLAT` 1024 |
|---|---|---|---|---|
| unset (Adaptive) | 0.584 / 1.5 | 0.236 / 1.5 | 0.694 / 2.3 | **1.000** / 1.4 |
| 1 | 0.047 / 1.4 | 0.090 / 1.3 | 0.134 / 2.1 | 0.106 / 1.3 |
| 2 | 0.086 / 1.4 | 0.129 / 1.3 | 0.239 / 2.2 | 0.198 / 1.3 |
| 4 | 0.163 / 1.4 | 0.170 / 1.3 | 0.386 / 2.1 | 0.355 / 1.3 |
| 8 | 0.275 / 1.5 | 0.208 / 1.4 | 0.570 / 2.2 | 0.608 / 1.4 |
| 16 | 0.450 / 1.4 | 0.236 / 1.5 | 0.694 / 2.3 | 0.923 / 1.4 |
| 32 | 0.682 / 1.4 | 0.236 / 1.8 | 0.694 / 2.6 | **1.000** / 1.5 |
| 64 | 0.901 / 1.5 | 0.236 / 2.2 | 0.694 / 3.0 | 1.000 / 1.6 |
| 128 | 0.946 / 1.7 | 0.236 / 3.1 | 0.694 / 3.9 | 1.000 / 1.9 |
| 256 | **0.966** / 1.9 | 0.236 / 4.8 | 0.694 / 5.6 | 1.000 / 2.5 |

Smallest `nprobes` reaching recall@10 >= 0.95: `IVF_FLAT` 4096: **256** (0.966; a second build of the same data gave 0.923 at 256, so this sits on the edge); `IVF_FLAT` 1024: **32** (and Adaptive, unset); `IVF_PQ` m=32: **never**, with or without `refine_factor=10`.

Surprises, in order of importance:

- **`IVF_PQ` ignored `num_partitions=4096` and built 730** (`N / sample_rate = 187043 / 256`). `analyze_plan` confirms it: `partitions_ranked=730`. No exception, one `WARN` line on stderr.
- **PQ recall plateaus at 0.236 (no refine) and 0.694 (refine 10) from `nprobes = 16` on, identical results all the way to 256 probes.** `analyze_plan` at 256 probes shows `partitions_searched=256, index_comparisons=65 K`, so the partitions were scanned; the ADC top-10 (or top-100 with refine) simply does not change. Routing is not the limit: 16 of 730 partitions already covers the true neighbours (compare `IVF_FLAT` 1024 at 16 probes: 0.923). The limit is the 32-byte code on this generator: every point of a topic sits at nearly the same distance from the query (256-D Gaussian noise of std 0.35, module 4's concentration effect), so neighbour order inside a topic is decided by differences smaller than the PQ reconstruction error. Reranking 100 candidates exactly triples recall but the right 10 are often not even in the ADC top-100. More `nprobes` cannot fix a quantizer problem; more `num_sub_vectors` (64 or 128) or `refine_factor` of 50+ would.
- **Adaptive probing is a recall lottery that depends on `num_partitions`.** With 1024 partitions it opened enough (6..24 within the 26.5% margin) to hit 1.000 at 1.4 ms; with 4096 partitions the same rule gave 0.584, because each topic is split over ~80 partitions and 24 is not enough. The cap of 24 initial probes is calibrated for "well-sized" partitions (`target_partition_size` 4096-8192 rows); asking for 46 rows per partition breaks the calibration.
- **4096 partitions on 187k rows is simply too many for `IVF_FLAT` here**: 46 rows per partition, 186 empty partitions, and recall at any `nprobes` below that of 1024 partitions at a quarter of the probes. Lance's own sizing (`N / 4096` for `IVF_FLAT`) would have chosen 45 partitions; `IVF_PQ`'s `N / 8192` would have chosen 22.
- Latency is flat at ~1.3-1.5 ms up to 32 probes for every index: the Python round trip and plan setup dominate; the partition scan only shows above 64 probes (PQ: +3.4 ms from 16 to 256 probes for 700 ADC scans of ~256 codes each).
- Builds are fast (0.9 s for 4096 flat partitions via hierarchical k-means, 4.2 s for PQ including the codebook) because the whole table fits the training sample: `sample_rate * 4096 = 1,048,576 > N`.

Run-to-run variance from the OS-seeded k-means: three builds of `IVF_FLAT` 4096 gave recall@10 at 256 probes of 0.923, 0.966, 0.966 (and 0.542 / 0.584 / 0.581 for Adaptive); every other cell moved by < 0.01. The script's "partitions built" column prints datafusion's metric verbatim (`4.10 K` for 4096, `730`, `1.02 K` for 1024).

## 6. Specific to LanceDB vs plain FAISS IVF

- **No fixed `nprobe` by default.** Leaving `nprobes` unset gives adaptive probing: open every partition whose centroid is within a calibrated relative margin of the nearest (L2, k<=10: 26.5%, between 6 and 24 partitions), then keep opening until `k` rows exist. FAISS always probes exactly `nprobe`. The old "default `nprobes = 20`" in docs and blogs is gone from this codebase; `nprobes(n)` is still honoured as a Fixed count.
- **Partition cap for quantized indices.** `IVF_PQ`/`IVF_SQ`/`IVF_RQ` train at most `N / sample_rate` partitions (730 here for 4096 requested), with only a `WARN` log. FAISS trains whatever `nlist` you ask for and warns about too few points. `IVF_FLAT` is not capped (4096 built, 186 empty).
- **Hierarchical k-means above 256 partitions**, fan-out 16, no global refinement pass by default. FAISS runs flat Lloyd's over all `nlist` centroids (25 iterations, random init). Lance's init is random too, 50 iterations, 1 redo, OS seed, not exposed.
- **Residual PQ by default for L2 and cosine, raw-vector PQ for dot** (`use_residual`), one shared codebook. FAISS `IndexIVFPQ` uses residuals for every metric by default (`by_residual = true`).
- **Partition = contiguous byte range in a file**, centroids plus offsets/lengths in a tiny `index.idx`; a query costs one range read per probed partition and nothing is required to be in RAM. FAISS IVF lists live in memory (or `OnDiskInvertedLists`).
- **`refine_factor` is a plan transform, not an index type**: the planner asks the ANN node for `k * r`, `take`s the raw vectors from the table files and re-sorts. Appended rows are brute-forced and merged until `optimize()`, which assigns them to existing centroids without retraining; `optimize` can also split or merge partitions that drift from `target_partition_size`.

## 7. Reproduce

```bash
python3 -m venv /tmp/venv-ivf-lancedb && /tmp/venv-ivf-lancedb/bin/pip install "lancedb==0.41.0" pyarrow numpy
cd modules/05-vector-databases-and-ann
/tmp/venv-ivf-lancedb/bin/python -I lab/ivf_lancedb_toy.py /tmp/ivf-lancedb-scratch
/tmp/venv-ivf-lancedb/bin/python -I lab/ivf_lancedb_scale.py /tmp/ivf-lancedb-scratch --nlist1024   # ~5 min on an M-series Mac
```

Source read at `lance` commit `e307b5e2cf7d612348f0d59586715e0b3bfd17cc`: `git clone --depth 1 https://github.com/lancedb/lance`. Lance's log lines (`WARN lance::index::vector] Build Vector Index: training 730 of the 4096 requested IVF partitions`, `partition N is empty, skipping`, `KMeans: more than 10% of clusters are empty`) print to stderr during the builds.
