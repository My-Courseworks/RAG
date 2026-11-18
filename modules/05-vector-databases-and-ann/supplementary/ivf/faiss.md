# IVF in FAISS (the reference implementation)

- **Flavour**: one `IndexIVF` base with pluggable list codecs: `IndexIVFFlat` (raw fp32), `IndexIVFScalarQuantizer` (SQ8/SQ4/SQ6/fp16/bf16/TurboQuant/EDEN), `IndexIVFPQ` (residual PQ, ADC tables), `IndexIVFPQR` (PQ + PQ refinement), `IndexIVFRaBitQ`, `IndexBinaryIVF`, plus FastScan (4-bit SIMD) variants. Coarse quantizer is any `faiss.Index` holding the `nlist` centroids: `IndexFlatL2` (exact) by default, `IndexHNSWFlat` for `nlist >= 65536` (`IVF65536_HNSW32`).
- **Runs here**: yes. `faiss-cpu==1.16.0` wheel on Apple Silicon, 12 OpenMP threads. Source read at tag `v1.16.0`, commit `6ccfd5f9be2283ebc037564bb45c6ba63b71e96a` (same version as the wheel).
- **Real thing ran**: toy and scale tests below are outputs of the real library, not simulations.

## 1. Source read

Repo `https://github.com/facebookresearch/faiss`, commit `6ccfd5f9` (v1.16.0). Paths relative to `faiss/`.

| File | What it does |
|---|---|
| `Clustering.h:22-86` | `ClusteringParameters`: `niter=25`, `nredo=1`, `spherical`, `min_points_per_centroid=39`, `max_points_per_centroid=256`, `seed=1234`, `init_method` (RANDOM / k-means++ / AFK-MC²), `early_stop_threshold`, `use_super_kmeans`. |
| `Clustering.cpp:61-493` | Lloyd's k-means driven by an assignment `Index` (`train_impl`): subsample, random init, `niter` × (search → `compute_centroids` → `split_clusters`), `nredo` keep-best. |
| `impl/ClusteringHelpers.cpp:107-113, 266-330` | Seed handling (`seed<0` → wall clock) and empty-cluster splitting. |
| `IndexIVF.h:30-94, 222-535` | `Level1Quantizer` (quantizer, `nlist`, `cp`, `quantizer_trains_alone`, `clustering_index`) and `IndexIVF` (`nprobe`, `max_codes`, `parallel_mode`, `by_residual`, `invlists`, `direct_map`). |
| `IndexIVF.cpp:44-50` | `Level1Quantizer` ctor **lowers `cp.niter` to 10** for IVF training. |
| `IndexIVF.cpp:88-213` | `train_q1_impl`: the three `quantizer_trains_alone` modes, SuperKMeans branch, HNSW-quantizer link sampling. |
| `IndexIVF.cpp:1543-1576` | `IndexIVF::train`: train coarse quantizer, then (optionally subsampled) residuals → `train_encoder`. |
| `IndexIVF.cpp:318-415` | `add_with_ids` → `quantizer->assign` (1 nearest centroid) → `encode_vectors` → `invlists->add_entry`. |
| `IndexIVF.cpp:433-527` | `search`: `quantizer->search(x, nprobe)` then `search_preassigned`; `parallel_mode` dispatch. |
| `IndexIVF.cpp:529-896` | `search_preassigned`: per query, loop over `nprobe` keys, `scan_one_list` with a heap, `max_codes` early stop, `IDSelectorRange` bound trick. |
| `IndexIVF.cpp:1391-1420` | `remove_ids` (swap-with-last compaction via `DirectMap`), `update_vectors` (re-assign to nearest existing centroid). |
| `IndexIVFFlat.cpp:39-57, 126-150, 173-180` | `by_residual=false`, codes are a `memcpy` of the floats, scanner computes exact L2/IP per code. |
| `IndexIVFPQ.cpp:49-99, 376-470, 525-540` | `by_residual=true`, PQ trained on residuals of `256*ksub` sampled vectors, precomputed-table selection, SIMD scanner. |
| `impl/pq_code_distance/IVFPQ_QueryTables.h:31-99`, `IVFPQScanner_impl.h:168-400` | ADC: `dis0 + sum_m table[m][code[m]]`; polysemous Hamming pre-filter (`polysemous_ht`). |
| `IndexIVFPQR.cpp:45, 119-215` | Refinement: ask IVFPQ for `k*k_factor` (default 4) candidates, re-rank with a second PQ on the 2nd-level residual. |
| `IndexScalarQuantizer.cpp:122-161` | IVF-SQ: `by_residual=true` by default, SQ trained on 100k sampled residuals. |
| `invlists/InvertedLists.h:58-120, 264-447`, `invlists/OnDiskInvertedLists.h:60` | List storage interface; `ArrayInvertedLists` default; on-disk / stacked / masked variants. |
| `index_factory.cpp:162-178, 299-315, 433-438, 1066` | Factory strings: `IVF<n>`, `IVF<n>_HNSW<M>`, `SQ*` names, `PQ<m>+<r>` → IVFPQR, `,RFlat` → `IndexRefine`. |
| `AutoTune.cpp:368-412, 709-777` | `ParameterSpace`: tunables `nprobe` (1,2,4,… `< nlist`), `ht`, `k_factor`, `max_codes` (IMI only), `efSearch`/`quantizer_*`. |

Wiki (fetched 2026-11-15): [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index), [Faster search](https://github.com/facebookresearch/faiss/wiki/Faster-search), [Lower memory footprint](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint), [Index IO, cloning and hyper-parameter tuning](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning).

## 2. The algorithm

**Train** (`IndexIVF::train` → `train_q1_impl` → `Clustering::train_impl`)
1. Guard: `n >= nlist` else throw (`Clustering.cpp:108-114`). If `n > nlist*max_points_per_centroid` (default 256) subsample to exactly that many rows, seeded (`:170-178`). If `n < nlist*min_points_per_centroid` (39) print a stderr WARNING and continue (`:179-187`). If `n == nlist` copy the points as centroids, no iterations (`:189-207`).
2. Init: random permutation of the (sub)sample, first `nlist` rows become centroids; `seed=1234` by default, so training is deterministic given the same data and thread count (`:272-280`; seed <0 → clock, `ClusteringHelpers.cpp:107`). k-means++ / AFK-MC² are opt-in (`init_method`).
3. Iterate `niter` times (**10 for IVF**, `IndexIVF.cpp:49`; 25 is the bare `Clustering` default): add centroids to the assignment index, 1-NN search of all training rows (`Clustering.cpp:338-343`), recompute means (`:400-410`), then `split_clusters`: every empty centroid is replaced by a copy of a large cluster's centroid perturbed by ±1/1024 and its count halved (`ClusteringHelpers.cpp:266-330`). Stop early if the objective stops changing (`Clustering.cpp:452-468`).
4. `nredo > 1`: repeat with a different permutation seed and keep the lowest objective (`:474-492`). `spherical=true` L2-normalises centroids after every update; it is switched on automatically for `METRIC_INNER_PRODUCT` (`IndexIVF.cpp:306-309`).
5. The assignment index *is* the coarse quantizer (mode 0), unless `clustering_index` (e.g. a GPU flat index) is given, or `quantizer_trains_alone=2` (cluster with a flat L2 index, then add centroids to e.g. an HNSW quantizer — this is what `IVF65536_HNSW32` does). With an HNSW quantizer, 1.16 also records sampled level-0 links for training points (`IndexIVF.cpp:62-106`).
6. Train the list codec on `train_encoder_num_vectors()` sampled vectors: 0 (nothing) for Flat, `256*ksub = 65 536` residuals for PQ (`IndexIVFPQ.cpp:97-99`), 100 000 for SQ (`IndexScalarQuantizer.cpp:159-161`). If `by_residual`, those are `x - centroid[assign(x)]` (`IndexIVF.cpp:1560-1570`).

**Assign + store** (`add_with_ids`, `add_core`)
7. `quantizer->assign(n, x)` = single nearest centroid (`IndexIVF.cpp:322`). No multi-assignment, no boundary replication; a `-1` assignment (possible with an approximate HNSW quantizer failing) is counted and dropped (`:372-375, 401-403`).
8. `encode_vectors`: Flat = `memcpy` of the floats (`IndexIVFFlat.cpp:133-134`); PQ/SQ = code of the residual when `by_residual`. `invlists->add_entry(list_no, id, code)` appends to `ArrayInvertedLists` (one `std::vector` of ids and one of codes per list). Lists are never split or merged after training.

**Search** (`search` → `search_preassigned`)
9. `quantizer->search(xq, nprobe)` returns the `nprobe` nearest centroids and their distances (`IndexIVF.cpp:464-470`); `nprobe` is clamped to `nlist` (`:449-450`).
10. For each query, lists are visited **in centroid-distance order**; each list is scanned fully (exact L2/IP for Flat, ADC table lookups for PQ) into a size-`k` max-heap (`:666-790`). Early termination only via `max_codes` (stop once this many codes were scanned, `:787-789`) or, for range search, `max_empty_result_buckets`. No per-list pruning by centroid distance.
11. PQ ADC: `d = ||x - c||² + sum_m T[m][code_m]` with `T` built once per (query, list) from the residual `x - c`; `use_precomputed_table=1` precomputes the `||y_R||² + 2<c, y_R>` term for every (list, m, j) (size `nlist*M*ksub` floats, skipped above 2 GiB, `IndexIVFPQ.cpp:375, 445`). `polysemous_ht>0` first filters codes by Hamming distance to the query's code, computing ADC only for survivors (`IVFPQScanner_impl.h:348-400`).
12. **Rescore** is opt-in, not default: `IndexIVFPQR` (`PQ32+16`) re-ranks `k*k_factor` ADC candidates with a second PQ (`IndexIVFPQR.cpp:132-211`); `IndexRefine` (`...,RFlat` / `Refine(SQ8)`) re-ranks `k*k_factor_rf` with full vectors stored outside the IVF.
13. **Filtering**: an `IDSelector` is applied per code *inside* the list scan (`ResultHandler.h:97`); it does not change which lists are opened, so a selective filter needs a larger `nprobe` (1.16's `ensure_topk_full` lets `max_codes` count only post-filter rows). `IDSelectorRange` on sorted ids binary-searches the id range inside each list instead (`IndexIVF.cpp:557-562, 711-721`).
14. **Updates**: `add` never retrains; new vectors go to the nearest *existing* centroid. `remove_ids` scans every list (or the `DirectMap` hashtable) and compacts by moving the last entry into the hole (`DirectMap.cpp:157-195`). `update_vectors` = remove + re-add (`IndexIVF.cpp:1397-1420`). Drift is the user's problem: retrain and rebuild.

## 3. Hyperparameters

| Name (FAISS) | Build/Query | Default | What it trades | Course vocabulary |
|---|---|---|---|---|
| `nlist` (`IVF<n>`) | build | none; wiki: `4√N`..`16√N` | more lists → smaller lists, cheaper scans, more centroid work and more boundary misses at fixed `nprobe` | `nlist` |
| `nprobe` | query (`index.nprobe` or `SearchParametersIVF`) | **1** | lists opened; linear in time; `nprobe=nlist` = brute force | `nprobe` |
| `cp.niter` | build | **10** (IVF), 25 (bare `Clustering`) | k-means quality vs train time | – |
| `cp.nredo` | build | 1 | re-run k-means, keep best objective | – |
| `cp.max_points_per_centroid` | build | 256 | training subsample cap `nlist*256` | – |
| `cp.min_points_per_centroid` | build | 39 | only a WARNING below `nlist*39` | – |
| `cp.seed` | build | 1234 | determinism (negative = clock) | – |
| `cp.spherical` | build | false (true for IP metric) | normalise centroids → cosine-style clustering | – |
| `cp.init_method` / `use_super_kmeans` / `early_stop_threshold` | build | RANDOM / false / 0 | init quality vs time; SuperKMeans (1.16) is a fast L2 variant | – |
| `quantizer_trains_alone` | build | 0 | 0 = k-means through the quantizer, 1 = quantizer trains itself (IMI), 2 = flat k-means then add to HNSW | – (no analogue elsewhere) |
| `clustering_index` | build | null | run k-means assignment on another index (GPU) | – |
| `by_residual` | build | false (Flat), **true** (PQ, SQ) | encode `x - centroid` → finer codes, needs per-list tables | – |
| `m`, `nbits` (`PQ<m>x<nbits>`) | build | nbits 8 | bytes/vector = `m*nbits/8`; `d % m == 0` | `m`, `nbits` |
| `IndexIVFScalarQuantizer.qtype` | build | `QT_8bit` | SQ8 (1 B/dim), SQ4, SQ6, fp16, bf16, 8bit_direct, TurboQuant `tq*`, `eden*`, `QT_0bit` (centroid only) | SQ |
| `use_precomputed_table` | build (call `precompute_table()`) | 0 (auto off) | memory `nlist*m*256*4` B for faster per-list table build (L2 + residual only) | – |
| `polysemous_ht` (`ht`) | query | 0 (off) | Hamming pre-filter; needs `do_polysemous_training` at build | – |
| `scan_table_threshold` | query | 0 | below this list size compute distances on the fly instead of building tables | – |
| `k_factor` (IVFPQR) / `k_factor_rf` (IndexRefine) | query | 4 / 1 | candidates re-ranked = `k * factor` | rescoring multiplier |
| `max_codes` | query | 0 (unlimited) | stop after this many scanned codes | `ef`-style budget, no exact analogue |
| `max_lists_num`, `ensure_topk_full`, `max_empty_result_buckets` | query (1.16) | 0 / false / 0 | FastScan list cap; soft `max_codes`; range-search stop | – |
| `parallel_mode` | query | 0 | 0 = threads over queries (batch!), 1 = over lists (single query), 2 = both, 3 = finer query split | – |
| `quantizer_efSearch` (`quantizer_params`) | query | HNSW default 16 | accuracy of the coarse search when the quantizer is HNSW | – |
| `direct_map` type | build | NoMap | Array/Hashtable needed for `reconstruct`, `remove_ids` by id, `update_vectors` | – |
| `index_factory` strings | build | – | `IVF4096,Flat`; `IVF4096,PQ96`; `OPQ96,IVF4096,PQ96` (rotation before IVF); `IVF4096_HNSW32,Flat`; `IVF4096,SQ8`; `IVF4096,PQ32+16` (IVFPQR); `IVF4096,PQ32x4fsr,RFlat`; `IVF4096,RaBitQ` | – |

## 4. Toy: A..H = 1, 2, 4, 7, 8, 11, 14, 15, `nlist = 3`, d = 1

Real `IndexIVFFlat(IndexFlatL2(1), 1, 3)`, seed 1234, everything below read back from the index (`lab/ivf_faiss_toy.py`).

1. **Train.** 8 points < `3*39` → FAISS prints `WARNING clustering 8 points to 3 centroids: please provide at least 117 training points` and trains anyway. Random init picks 3 of the 8 values; 10 iterations allowed, converged after 4:

   | iter | objective (Σ sq dist) | imbalance | splits |
   |---|---|---|---|
   | 0 | 107.000 | 1.219 | 0 |
   | 1 | 19.667 | 1.031 | 0 |
   | 2 | 13.833 | 1.031 | 0 |
   | 3 | 13.833 | 1.031 | 0 → early stop |

   `quantizer.reconstruct_n(0, 3)` → centroids **7.5, 13.333, 2.333** (list order 0, 1, 2). Replaying `faiss.Clustering(1, 3)` with `niter=10, seed=1234` on an `IndexFlatL2` gives the same three numbers.
2. **Assign + store** (`invlists.list_size`, `get_ids`, `get_codes`):

   | list | centroid | members |
   |---|---|---|
   | 0 | 7.500 | D=7, E=8 |
   | 1 | 13.333 | F=11, G=14, H=15 |
   | 2 | 2.333 | A=1, B=2, C=4 |

   Voronoi boundaries (centroid midpoints): **4.917** and **10.417**. Note the "natural" split {7, 8, 11} did not happen: 11 is 2.33 from 13.33 but 3.5 from 7.5.
3. **Query q = 9.5, k = 2.** Exact answer: E=8 and F=11 (both sq-dist 2.25). Centroid distances: list 0 → 2.0, list 1 → 3.83, list 2 → 7.17.
   - `nprobe=1`: opens list 0 only, scans 2 codes → returns **E=8 (2.25), D=7 (6.25)**. **Misses F=11**, which sits in list 1 across the 10.417 boundary.
   - `nprobe=2`: opens lists 0 and 1, scans 5 codes → **E=8, F=11** = exact.
   - `nprobe=3`: scans all 8 → exact (this is brute force).
4. **Query q = 5.5, k = 2.** Exact: C=4 and D=7 (2.25 each). Probe order 0 (2.0), 2 (3.17), 1 (7.83). `nprobe=1` → **D=7, E=8**, **misses C=4** (other side of 4.917). `nprobe=2` → exact.
5. **Query q = 9.5, k = 1.** Exact E=8; `nprobe=1` already gets it. Misses need the true neighbour to be in a non-nearest cell, which k=1 at this q avoids.
6. **The dial.** `ParameterSpace().initialize(index)` finds exactly one tunable, `nprobe`, and `set_index_parameters(index, "nprobe=2")` sets it. Sweeping 65 queries on a 0..16 grid (step 0.25), recall@2: `nprobe=1` → **0.908**, misses exactly at q ∈ {4.75..5.75} ∪ {9.25..10.75}, i.e. within ~0.9 of the two boundaries; `nprobe=2` → 1.000.

## 5. Scale test: N = 187 043, D = 256, 1000 queries, k = 10, L2

Shared generator (50 topics, σ = 0.35). Ground truth = numpy brute force (1.7 s). 12 OpenMP threads, batched search (1000 queries per call, `parallel_mode=0`); ms/query = wall time / 1000. Serialized size via `faiss.serialize_index`.

**Build** (train = k-means + codec training, add = assign + encode):

| Index (factory string) | train | add | serialized | B/vector | list sizes min/median/max |
|---|---|---|---|---|---|
| A `IVF4096,Flat` | 5.9 s | 0.6 s | 188.1 MiB | 1054.6 (1024 fp32 + 8 id + 22 amortised coarse centroids) | 1 / 22 / 327 |
| B `IVF4096,PQ32x8` | 13.5 s | 1.1 s | 11.4 MiB | 64.0 (32 code + 8 id + 22 centroids + PQ codebook) | same lists as A |
| C `IVF1024,Flat` | 1.5 s | 0.2 s | 185.1 MiB | 1037.7 | 1 / 192 / 645 |
| D `IVF4096,PQ32x8,RFlat` | 13.6 s | 1.2 s | 194.1 MiB | 1088.0 (B + a full fp32 copy in `IndexFlat`) | same lists as A |

No subsampling happened (`187 043 < 4096*256`), no min-points warning (`> 4096*39 = 159 744`). B's PQ: `M=32, nbits=8, dsub=8, ksub=256`, trained on 65 536 residuals. `use_precomputed_table` auto-selected type 1: a `4096×32×256` fp32 table = 128 MiB in RAM, 11× the codes themselves, rebuilt on load (not in the serialized size).

**Sweep** (`recall@10` vs exact; `%db` = codes scanned / N, from `indexIVF_stats.ndis`):

| nprobe | A recall | A ms/q | A %db | B (PQ32) recall | B ms/q | C (nlist 1024) recall | C ms/q | C %db |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.078 | 0.011 | 0.06 | 0.065 | 0.011 | 0.127 | 0.007 | 0.13 |
| 2 | 0.144 | 0.011 | 0.12 | 0.095 | 0.011 | 0.228 | 0.007 | 0.26 |
| 4 | 0.254 | 0.010 | 0.24 | 0.136 | 0.011 | 0.402 | 0.007 | 0.52 |
| 8 | 0.419 | 0.011 | 0.48 | 0.171 | 0.013 | 0.659 | 0.009 | 1.01 |
| 16 | 0.653 | 0.014 | 0.93 | 0.209 | 0.015 | **0.947** | 0.015 | 1.84 |
| 32 | 0.908 | 0.020 | 1.58 | 0.232 | 0.021 | **1.000** | 0.026 | 3.25 |
| 64 | **0.997** | 0.029 | 1.97 | 0.235 | 0.026 | 1.000 | 0.048 | 6.49 |
| 128 | 1.000 | 0.041 | 3.59 | 0.235 | 0.041 | 1.000 | 0.094 | 12.84 |
| 256 | 1.000 | 0.069 | 7.11 | 0.235 | 0.068 | 1.000 | 0.190 | 25.40 |

- **Smallest `nprobe` for recall@10 ≥ 0.95**: A `IVF4096,Flat` → **64** (1.97 % of the database scanned); C `IVF1024,Flat` → **32** (3.25 % scanned); B `IVF4096,PQ32x8` → **never** (plateau 0.235).
- `ParameterSpace().explore` on A (1.1 s, 12 nprobe values 1..2048, `IntersectionCriterion` = recall@10) returns the Pareto frontier: `nprobe=1..128` kept, `nprobe ≥ 256` dropped as "no better recall, slower" — the same answer as the manual sweep, which is how FAISS expects `nprobe` to be chosen.
- **Surprise 1: `nlist` vs `nprobe` is about the fraction scanned, not the count.** At equal recall ≈ 0.95, 4096 lists need 64 probes (2 % of N) and 1024 lists need 16-32 probes (1.8-3.3 % of N). Lists are very unequal (median 22, max 327 at nlist 4096) because the data has only 50 real clusters: k-means carves each topic into ~80 noise-shaped shards, and a query's true neighbours scatter across many of them — hence recall 0.08 at `nprobe=1`.
- **Surprise 2: PQ32 cannot rank this data, at any `nprobe`.** Recall tracks A up to `nprobe≈4` then flattens at 0.235. Within a topic all points are ~`sqrt(2·256·0.35²) ≈ 7.9` apart with tiny spread (isotropic Gaussian noise, no low-rank structure), so the true top-10 differ from the next 1000 by less than the ADC error of 8-dim subquantizers at 1 byte each. The missing recall is quantization error, not list selection — exactly the case for rescoring:

| D `IVF4096,PQ32x8,RFlat`, `nprobe=64`, `k_factor` | candidates re-ranked | recall@10 | ms/q |
|---|---|---|---|
| 1 | 10 | 0.235 (= B) | 0.033 |
| 2 | 20 | 0.351 | 0.032 |
| 4 (default) | 40 | 0.493 | 0.028 |
| 8 | 80 | 0.648 | 0.031 |
| 16 | 160 | 0.792 | 0.034 |
| 32 | 320 | 0.901 | 0.042 |
| 64 | 640 | **0.968** | 0.047 |

  Rescoring `10*k_factor` PQ candidates with the stored fp32 vectors buys back the recall that `nprobe` could not. On real embeddings (anisotropic, lower intrinsic dimension) PQ32 at d=256 normally reaches 0.9+ without refinement; this generator is close to its worst case.
- Timing caveat: ms/query are batched (1000 queries, 12 threads, `parallel_mode=0`); single-query latency is ~10× higher and is where `parallel_mode=1` applies.

## 6. What is specific to FAISS vs "plain IVF"

- The coarse quantizer is itself an `Index`: swap `IndexFlatL2` for `IndexHNSWFlat` (`IVF65536_HNSW32`) and the centroid search becomes approximate (`quantizer_efSearch` becomes a second dial, and `-1` assignments become possible).
- `niter` is **10** for IVF training, not the 25 that `ClusteringParameters` advertises; the subsample cap `nlist*256` means training cost is bounded regardless of N, and `min_points_per_centroid=39` is only a warning.
- `nprobe` defaults to **1** and is a plain attribute; nothing adapts it per query. Lists are scanned whole, in centroid order, with no distance-based pruning; `max_codes` is the only early stop.
- `by_residual` is on for PQ/SQ and off for Flat; it is what makes IVFPQ at 32 B/vector usable, and what forces the per-list ADC table rebuild (hence `use_precomputed_table` and `scan_table_threshold`).
- Metadata filtering is an `IDSelector` evaluated per code during the scan: it never changes which lists are opened. There is no filtered-HNSW-style graph adaptation; raise `nprobe` or pre-partition into separate indexes.
- Inserts go to the nearest existing centroid for ever; there is no re-clustering, splitting, or balancing. Deletes compact lists by swap-with-last. Rebuild is the maintenance story.
- Rescoring with full vectors is a separate wrapper (`IndexRefine`, `,RFlat`) or a second PQ (`IVFPQR`), both tuned with a `k_factor`; FAISS does not store full vectors alongside PQ codes by default.

## 7. Reproduce

```bash
python3 -m venv /tmp/venv-ivf-faiss && /tmp/venv-ivf-faiss/bin/pip install faiss-cpu==1.16.0 numpy
cd modules/05-vector-databases-and-ann/lab
/tmp/venv-ivf-faiss/bin/python -I ivf_faiss_toy.py      # < 1 s
/tmp/venv-ivf-faiss/bin/python -I ivf_faiss_scale.py    # ~3 min on an M-series Mac, ~1 GiB RAM
```

Source for the citations: `git clone --depth 1 --branch v1.16.0 https://github.com/facebookresearch/faiss` (commit `6ccfd5f9be2283ebc037564bb45c6ba63b71e96a`).
