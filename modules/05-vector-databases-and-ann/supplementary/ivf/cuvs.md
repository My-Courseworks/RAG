# IVF in NVIDIA cuVS (GPU IVF-Flat and IVF-PQ)

- **Flavours:** IVF-Flat (full vectors, interleaved layout) and IVF-PQ (residual PQ, per-subspace or per-cluster codebooks, ADC via a shared-memory lookup table); coarse quantizer is NVIDIA's own *balanced, hierarchical k-means*. Also ships IVF-SQ, IVF-RaBitQ, IVF-binary (not covered here).
- **Where it runs:** CUDA GPUs only. No CPU backend, and PyPI publishes only `manylinux` wheels (`cuvs-cu12`, `cuvs-cu13`).
- **Did the real thing run here?** **No.** This Mac has no NVIDIA GPU; `pip install cuvs-cu12` fails with "No matching distribution found" (the NVIDIA wheel-stub only lists `manylinux_2_28_aarch64/x86_64`), and Docker has no NVIDIA runtime. The code read below is from the real source; the toy is a **SIMULATION** that mirrors the cited lines in numpy; the scale test is **skipped**, no numbers invented.

## 1. Source read

Repo `https://github.com/rapidsai/cuvs`, commit `aac896693db9f45683053bff236379c2449de80d` (branch `release/26.12`, 2026-11-13). Paths relative to the repo root; line numbers are for that commit.

| File | What it implements |
|---|---|
| `cpp/include/cuvs/neighbors/ivf_flat.hpp` L28-80 | `index_params` / `search_params` for IVF-Flat; L180-202 the interleaved list layout doc; L285-294 `veclen` rule |
| `cpp/include/cuvs/neighbors/ivf_pq.hpp` L47-152, L160-214 | IVF-PQ `index_params` (`pq_bits`, `pq_dim`, `codebook_kind`, `codes_layout`, `force_random_rotation`, `max_train_points_per_pq_code`) and `search_params` (`lut_dtype`, `internal_distance_dtype`, `preferred_shmem_carveout`, `coarse_search_dtype`, `max_internal_batch_size`) |
| `cpp/include/cuvs/cluster/kmeans.hpp` L250-302 | `balanced_params`: `n_iters`, `balance_lower_tolerance=0.333`, `balance_upper_tolerance=3.0`, `centroid_offset=0.01`, `donor_selection=SizeSorted` |
| `cpp/src/cluster/detail/kmeans_balanced.cuh` | the coarse quantizer: `build_clusters` L1242-1293 (init + EM), `balancing_em_iters` L1128-1234, `adjust_centers` L963-1080 + kernel L865-915 (the "balancing"), `arrange_fine_clusters` L1297-1354, `build_hierarchical` L1521-1721 (mesoclusters), `calc_centers_and_sizes` L394-496 |
| `cpp/src/cluster/kmeans_balanced.cuh` L69-126 | public `kmeans::fit(balanced_params)` → `build_hierarchical` |
| `cpp/src/neighbors/ivf_flat/ivf_flat_build.cuh` | `build` L392-446 (strided train subsample, fit, extend); `extend` L159-376 (predict labels, optional `adaptive_centers` update, grow lists, interleaved pack kernel L104-156) |
| `cpp/src/neighbors/ivf_flat/ivf_flat_search.cuh` | `search_with_filtering` L319-378 (`n_probes` clamp L335, workspace sizing from `accum_sorted_sizes`), `search_impl` L41-315 (GEMM coarse search L107-190, two-pass interleaved scan L196-290) |
| `cpp/src/neighbors/ivf_flat/detail/jit_lto_kernels/interleaved_scan_impl.cuh` L159 | where the bitset filter is tested per candidate inside the list scan |
| `cpp/src/neighbors/ivf_pq/ivf_pq_build.cuh` | `build` L1241-1400 (random-row subsample with fixed `RngState{137}`, fit, rotation, codebooks), `train_per_subset` L320-408, `train_per_cluster` L411-505, residual computation L81-215, `encode_list_data` L712-800 |
| `cpp/src/neighbors/ivf_pq_index.cu` L350-364, L612-622 | `index_params::from_dataset` heuristic and `calculate_pq_dim` |
| `cpp/src/neighbors/ivf_pq/ivf_pq_search.cuh` | `select_clusters` L60-340 (coarse search in fp32/fp16/int8), `ivfpq_search_worker` L420-520 (LUT + scan + select), checks L900-931 |
| `cpp/src/neighbors/ivf_common.cuh` L32 | `kOutOfBoundsRecord` padding when probed lists hold fewer than `k` rows |
| `python/cuvs/cuvs/neighbors/{ivf_flat,ivf_pq}/*.pyx`, `refine.pyx` L34 | Python surface: `IndexParams`, `SearchParams`, `build`, `search`, `extend`, `save/load`, `refine` |
| `fern/pages/cuvs_bench/param_tuning.md` | NVIDIA's own tuning table (`nlist`, `niter`, `ratio`, `nprobe`, `refine_ratio`, `smemLutDtype`, ...) |

## 2. The algorithm (IVF-Flat; IVF-PQ differences in brackets)

1. **Subsample a training set.** IVF-Flat: `ratio = max(1, N / max(fraction*N, n_lists))`, then takes **every `ratio`-th row** (`copy_matrix` with a row stride; the source literally says `// TODO: a proper sampling`, `ivf_flat_build.cuh` L417-428). [IVF-PQ: proper random `sample_rows` with a **hard-coded seed 137**, `ivf_pq_build.cuh` L1276-1308.] Hard requirement: `n_rows >= n_lists` or `RAFT_EXPECTS` throws (L405); nothing auto-shrinks `n_lists`.
2. **Train the coarse quantizer: balanced hierarchical k-means** (`build_hierarchical`).
   - `n_mesoclusters = min(n_lists, round(sqrt(n_lists)))` (L1538). Run `build_clusters` for that many mesoclusters on the whole training set.
   - `build_clusters` init is **not random**: `label[i] = i mod k`, centres = means of those labels (L1256-1275). No seed anywhere in IVF-Flat training; deterministic for a given row order.
   - EM loop (`balancing_em_iters`): on every iteration after the first, call `adjust_centers`; then E-step `predict` (fused L2-NN against centres, or GEMM + argmin), M-step `calc_centers_and_sizes` (empty cluster → centre becomes 0 via `div_checkzero_op`, L493).
   - **The "balancing"** (`adjust_centers`, SizeSorted mode): sort clusters by size, pair smallest with largest. For every pair where `small < avg*0.333` **or** `large > avg*3.0`, the small cluster's centre is **re-seeded at `donor_centre + 0.01*(donor_point - donor_centre)`**, where the donor point is a pseudo-randomly chosen member of the big cluster (stride = a prime from `kPrimes` indexed by the iteration number, so the choice is reproducible). It is a *re-seeding heuristic*, not a capacity constraint: lists are never forced to equal size, they are nudged until no list is < 1/3 or > 3x the average. Each re-seed can extend the loop by one iteration (`balancing_pullback`=2 for mesoclusters, 5 for the final fine-tune).
   - For inner-product / cosine the centres are L2-normalised every iteration (spherical k-means, L1194-1206) so they do not collapse to zero.
   - `arrange_fine_clusters`: each mesocluster gets a number of fine clusters proportional to its size (at least 1 if non-empty). `build_fine_clusters` runs `build_clusters` inside each mesocluster on at most `2N/n_meso` of its points.
   - **Fine-tune** all `n_lists` centres together on the full training set for only `max(n_iters/10, 2)` iterations (L1664), with the same tolerances.
   - Why GPUs want this: the search kernel assigns one CUDA block per (query, probed list) chunk; wildly different list sizes mean idle SMs and a workspace sized by the *largest* lists (`accum_sorted_sizes` is used exactly for that, `ivf_flat_search.cuh` L339-343). Equal-ish lists = predictable kernel time.
3. **Assign and store.** `extend()` runs `predict` (nearest centre only, single assignment, no boundary replication) in batches of 65 536 rows, grows each list, and packs vectors with an interleaved layout: groups of 32 rows (`kIndexGroupSize`), inside a group `veclen` consecutive dims of row 0, then of row 1, ... (`veclen = 16 bytes / sizeof(T)` if it divides `dim`, else 1). Lists are padded to multiples of 32 (or 1024 unless `conservative_memory_allocation`), so the index is slightly larger than the dataset. Vectors are stored **as-is, not as residuals**. [IVF-PQ: `rotation_matrix @ (x - centre)` residual, then PQ-encode each `pq_len = rot_dim/pq_dim` sub-vector against its codebook; codes are also interleaved (`INTERLEAVED` layout) for coalesced loads.]
4. **Search.** `n_probes = min(params.n_probes, n_lists)`. Coarse step: one cuBLAS GEMM of queries x centres (`-2 q.c + |c|^2`, L107-160), then `select_k` top-`n_probes` per query (L184-190). Fine step: `ivfflat_interleaved_scan` kernel over the probed lists, exact L2/IP/cosine distances per candidate (there is **no early termination**: every row of every probed list is scored), per-block warp-sort top-k, then a second `select_k` merges the per-list results when `n_probes > 1` (L196-290). If the probed lists together contain fewer than `k` rows the output is padded with `kOutOfBoundsRecord` = `uint32/uint64 max`. [IVF-PQ: coarse search can run in fp16/int8 (`coarse_search_dtype`); per (query, list) a LUT of `pq_dim x 2^pq_bits` partial distances is built in shared memory in `lut_dtype` precision, codes are scored by table lookups (ADC), distances accumulated in `internal_distance_dtype`.]
5. **Rescore.** Nothing inside `search`. Refinement is a **separate API**, `cuvs::neighbors::refine` / `cuvs.neighbors.refine(dataset, queries, candidates, k)`: search for `refine_ratio*k`, then recompute exact distances against the original vectors (on GPU or host) and keep `k`. NVIDIA's benchmark configs use `refine_ratio=2` for IVF-PQ; cuVS builds a throw-away IVF-Flat "refinement index" per query batch to do this (`fill_refinement_index`, `ivf_flat_build.cuh` L465-530).
6. **Filtering and updates.** Filtering: `bitset_filter` (one bit per database row) tested per candidate inside the scan (`interleaved_scan_impl.cuh` L159: `valid = vec_id < list_length && sample_filter(...)`), i.e. post-filter inside the probed lists; a selective filter plus small `n_probes` returns fewer than `k` rows, padded with `kOutOfBoundsRecord`. Updates: `extend` appends to the nearest existing list, never splits lists or retrains; with `adaptive_centers=true` the centres are moved to the running mean of their list (so they "drift" with insertion order, `ivf_flat.hpp` L36-47). No per-row delete: the only removal primitive is `ivf_pq::helpers::erase_list` (drop a whole list, `ivf_pq.hpp` L3144); row-level deletion is done with the bitset filter.

## 3. Hyperparameters

| name (C++ / Python) | build or query | default | trades | FAISS analogue |
|---|---|---|---|---|
| `n_lists` | build | 1024 (IVF-PQ `from_dataset`: `sqrt(N)`, or 4 if N < 10 000) | build time + coarse-search cost vs. list size; docs want 1k-10k vectors per list | `nlist` |
| `metric` / `metric_arg` | build | `L2Expanded`, 2.0 | L2 / InnerProduct / Cosine (cosine needs dim > 1) | `METRIC_L2` / `METRIC_INNER_PRODUCT` |
| `kmeans_n_iters` | build | 20 | k-means quality vs build time; fine-tune pass uses `max(n/10, 2)` | `ClusteringParameters.niter` (25) |
| `kmeans_trainset_fraction` | build | 0.5 (`from_dataset`: 1.0 if N < 10k else 0.1) | train-set size; IVF-Flat subsamples by **stride**, IVF-PQ randomly | `max_points_per_centroid` (256) - same purpose, opposite parametrisation |
| `add_data_on_build` | build | true | `false` = train only, populate later with `extend` | `train()` then `add()` |
| `adaptive_centers` | build | false | centres follow inserted data vs. frozen; drift depends on insert order | none |
| `conservative_memory_allocation` | build | false | list padding to 32 instead of 1024 rows; less GPU RAM, slower repeated `extend` | none |
| IVF-PQ `pq_dim` | build | 0 = heuristic (`dim/2` if dim >= 128, rounded down to a multiple of 32; else largest power of two <= dim) | code size + LUT size vs recall; multiples of 32 preferred | `M` |
| IVF-PQ `pq_bits` | build | 8 | 4..8 bits per sub-code; `pq_dim*pq_bits` must be a multiple of 8 | `nbits` |
| IVF-PQ `codebook_kind` | build | `PER_SUBSPACE` | `PER_CLUSTER` = one codebook per IVF list (`n_lists x 2^bits x pq_len`), better fit per list, more memory, slower build | none (FAISS has one shared codebook; `IndexIVFPQ.by_residual` is closest in spirit) |
| IVF-PQ `codes_layout` | build | `INTERLEAVED` | interleaved for coalesced GPU loads vs `FLAT` (contiguous per vector) | none |
| IVF-PQ `force_random_rotation` | build | false | random orthogonal rotation even when `dim % pq_dim == 0` (always applied otherwise) | `OPQ` pre-transform, roughly |
| IVF-PQ `max_train_points_per_pq_code` | build | 256 | codebook train points = `256 * 2^pq_bits` | `max_points_per_centroid` for the PQ sub-quantizer |
| `n_probes` | query | 20 | lists scanned per query; clamped to `n_lists` | `nprobe` |
| IVF-PQ `lut_dtype` | query | fp32 (bench default fp16, CAGRA-build default fp8) | LUT precision = shared-memory footprint vs recall | none (`use_precomputed_table` is a different thing) |
| IVF-PQ `internal_distance_dtype` | query | fp32 | fp16 accumulators = slightly faster, slight recall loss | none |
| IVF-PQ `coarse_search_dtype` | query | fp32 | fp16/int8 GEMM for the centre search; int8 degrades recall unless data is normalised | none |
| IVF-PQ `preferred_shmem_carveout` | query | 1.0 | shared-memory vs L1 split for the search kernel | none |
| IVF-PQ `max_internal_batch_size` | query | 4096 | queries per internal batch; GPU utilisation vs workspace | none |
| `refine_ratio` (bench) / `refine(...)` | query (separate call) | 1 (bench), 2 in NVIDIA's IVF-PQ configs | exact rescoring of `ratio*k` candidates against the original vectors; costs ~25% QPS per NVIDIA | `IndexRefineFlat` |
| balanced k-means `balance_lower_tolerance`, `balance_upper_tolerance`, `centroid_offset`, `donor_selection` | build (only via `cuvs::cluster::kmeans`, **not** exposed through IVF `index_params`) | 0.333, 3.0, 0.01, SizeSorted (IVF-PQ codebooks override to 0.25, 4.0, 0.05, Random) | when a list counts as underfull / overfull and how far from the donor centre the new seed lands | none |

Note on `kmeans_trim_ratio`: it does **not exist** at this commit (grep over `cpp/`, `python/`, `fern/` finds nothing). The balancing knobs are the `balance_*_tolerance` pair above.

## 4. Toy, step by step (SIMULATION, `lab/ivf_cuvs_toy.py`)

Data A..H = 1, 2, 4, 7, 8, 11, 14, 15 (dim = 1 is accepted by cuVS for L2), `n_lists = 3`. No seed: cuVS's k-means init is `label = i mod k`.

**Build with the default `kmeans_trainset_fraction = 0.5`**

1. `trainset_ratio = max(1, 8 // max(4, 3)) = 2` → training rows are every 2nd row: {A=1, C=4, E=8, G=14}.
2. `n_mesoclusters = round(sqrt(3)) = 2`. Init labels [0,1,0,1] → centres [4.5, 9]; one EM step → {A, C} / {E, G}, centres [2.5, 11]; sizes [2, 2] are within [avg/3, 3avg] so `adjust_centers` never fires; 19 more iterations change nothing.
3. Fine clusters per mesocluster, proportional to size: [2, 1]. Mesocluster {1, 4} → centres [1, 4]; mesocluster {8, 14} → centre [11].
4. Fine-tune all three centres on the 4 training points for `max(20/10, 2) = 2` iterations: unchanged, centres **[1, 4, 11]**.
5. `extend` assigns all 8 values to the nearest frozen centre: **list 0 = {A, B}, list 1 = {C, D}, list 2 = {E, F, G, H}** (sizes 2, 2, 4 - the 50% strided subsample never saw B, D, F, H).
6. Query **9.5, k = 3**; exact answer {E=8, F=11, D=7}. Coarse distances to centres [8.5, 5.5, 1.5].
   - `n_probes = 1` opens list 2 → scans 4 vectors → returns {E, F, G=14}: **misses D=7** (it sits in list 1, whose centre 4 is 5.5 away although D itself is 2.5 away).
   - `n_probes = 2` opens lists 2 and 1 → 6 vectors → {E, F, D}: exact.
7. Query **5.5, k = 2**; exact {C, D}. Both live in list 1 (centre 4, distance 1.5): `n_probes = 1` already exact.

**Same build with `kmeans_trainset_fraction = 1.0`** (all 8 rows train)

8. Mesoclusters {A,B,C,D} / {E,F,G,H}, centres [3.5, 12]; fine clusters [2, 1] → [1.5, 5.5] and [12]; fine-tune moves them to **[1.5, 6.33, 13.33]**; lists **{A,B}, {C,D,E}, {F,G,H}**.
9. Query 9.5, k = 3: coarse distances [8, 3.17, 3.83]. `n_probes = 1` opens list 1 → {E, D, C=4}: **misses F=11** (list 2's centre is 0.67 farther than list 1's, while F itself ties E for nearest). `n_probes = 2` → exact.

**What "balanced" changes: Lloyd's vs cuVS on all 8 values, k = 3, identical init** (the `build_clusters` path, also used for mesoclusters and PQ codebooks)

10. Init `i mod 3` → labels [0,1,2,0,1,2,0,1], centres [7.33, 8.33, 7.5]. First E-step empties cluster 2: labels [0,0,0,0,1,1,1,1], centres [3.5, 12, **0**] (empty → 0 via `div_checkzero_op`).
11. **Plain Lloyd's** keeps going from the accidental zero centre: A=1 is pulled to it, then B; it converges to lists **{C,D,E}, {F,G,H}, {A,B}** (centres [6.33, 13.33, 1.5]) - the same partition as step 8. In 1-D with positive data the zero centre happened to be useful; in high dimensions an all-zero centroid is usually just a dead list.
12. **cuVS balancing** at iteration 1: cluster 2 (size 0 < 0.888 = 0.333 x 2.667) is paired with the largest cluster (cluster 1, size 4); donor point = H=15 (first lane hit with stride prime 71); new centre = 12 + 0.01 x (15 - 12) = **12.03**, i.e. a near-duplicate of the big centre, slightly toward H. Next E-step splits the big list: G, H go to the new centre. Two more iterations settle at centres **[2.33, 8.67, 14.5]**, lists **{A,B,C}, {D,E,F}, {G,H}** (sizes 3, 3, 2). One extra iteration is appended because an adjustment happened (`n_iters` 20 → 21).
13. Net effect on this data: both end with sizes 3/3/2 but different partitions; the balanced one puts the boundary between C=4 and D=7 instead of between B=2 and C=4. For query 9.5 the balanced partition holds D, E, F in one list, so `n_probes = 1` would already be exact - a coincidence of this data, not a property of the method.

## 5. Scale

**Skipped**: cuVS cannot execute on this machine (no NVIDIA GPU, no CPU kernels, no macOS wheel). `lab/ivf_cuvs_scale.py` detects the missing library, prints why, and exits without numbers; on a Linux/GPU box it runs the standard N=187 043, D=256 sweep (`n_lists` 4096 and 1024, `n_probes` 1..256, recall@10 and ms/query).

What NVIDIA publishes (theirs, not ours; all figures are chart-only in the posts, so only text-stated facts are repeated):

- "Accelerating Vector Search: NVIDIA cuVS IVF-PQ Deep Dive, Part 1", 2024-07-18, https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/ - DEEP-100M (96-d fp32), H100. IVF-Flat with 100K lists vs IVF-PQ with 50K lists, `pq_dim = 96`, `pq_bits` 5 and 8, refine ratio 2, fp16 LUT and distances: at 10K-query batches IVF-PQ gives "3-4x the number of queries per second of IVF-Flat"; IVF-PQ build is "x2-5" slower; index is "x4-5" smaller (DEEP-1B 360 GiB → 54 GiB, or 24 GiB at ~1.5x slower search). Recall range of interest "0.85-0.99"; refine ratio 2 lifts "from 0.85 to 0.95".
- "... Performance Tuning, Part 2", 2024-07-18, https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-performance-tuning-part-2/ - DEEP-100M: `n_lists` in 10K-50K "likely to yield good performance"; QPS "drops by a factor of three or more" from `pq_dim` 128 to 256-384; refinement ratio 2 "achieves the 0.99 recall" at "~25% cost to the QPS"; LUT in shared memory is "2-5x faster".
- "Accelerating Vector Search: Fine-Tuning GPU Index Algorithms", 2023-09-11, https://developer.nvidia.com/blog/accelerating-vector-search-fine-tuning-gpu-index-algorithms/ - DEEP-100M on H100 vs HNSW on Xeon 8480CL; states that `n_probes = n_lists` is exact and IVF-Flat recall exceeds IVF-PQ at equal probes; no numeric QPS in text.

## 6. What is cuVS-specific vs plain FAISS IVF

- **Balanced hierarchical k-means** as the coarse quantizer: sqrt(n_lists) mesoclusters → fine clusters inside each → short global fine-tune, with underfull (< avg/3) lists re-seeded next to overfull (> 3x avg) ones. FAISS Lloyd's only re-seeds *empty* clusters (by splitting the largest) and trains all `nlist` centres flat. Motivation is GPU load balance, not recall.
- **Deterministic, seed-free init** for IVF-Flat (`label = i mod k`, strided subsample); IVF-PQ uses a *fixed* seed 137. Nothing user-settable; reorder the rows and you get a different index.
- **`adaptive_centers` on `extend`**: optional running-mean update of centres as data is appended (FAISS centroids are frozen after `train`).
- **`codebook_kind = PER_CLUSTER`**: one PQ codebook per IVF list, trained on that list's residuals (FAISS has a single shared codebook).
- **Refinement is an external call** (`refine`) and the search kernel itself never touches full vectors; FAISS wraps the same idea as `IndexRefineFlat`.
- **GPU-shaped storage and knobs**: interleaved 32-row groups with `veclen` chunks, 1024-row list padding, `kOutOfBoundsRecord` padding instead of a smaller result set, `lut_dtype`/`internal_distance_dtype`/`coarse_search_dtype`/`preferred_shmem_carveout`; bitset post-filter inside the scan. `kmeans_trim_ratio` (named in the task brief) does not exist here.

## 7. Reproduce

```bash
# venv (numpy only; cuVS itself is not installable on macOS)
python3 -m venv /private/tmp/claude-501/-Users-aprabhak-Documents-my-github-My-Courseworks-RAG/02c65a2b-b4be-486a-9881-086fb9ba4561/scratchpad/venv-ivf-cuvs
/private/tmp/claude-501/-Users-aprabhak-Documents-my-github-My-Courseworks-RAG/02c65a2b-b4be-486a-9881-086fb9ba4561/scratchpad/venv-ivf-cuvs/bin/pip install numpy
# proof that the real library does not install here:
#   .../venv-ivf-cuvs/bin/pip install --dry-run cuvs-cu12   -> "No matching distribution found for cuvs-cu12"

# source read
git clone --depth 1 https://github.com/rapidsai/cuvs.git   # read at aac896693db9f45683053bff236379c2449de80d

# toy (SIMULATION) and scale (skips on this machine, runs on Linux + NVIDIA GPU with `pip install cuvs-cu12 cupy-cuda12x`)
cd /Users/aprabhak/Documents/my_github/My-Courseworks/RAG
.../venv-ivf-cuvs/bin/python -I modules/05-vector-databases-and-ann/lab/ivf_cuvs_toy.py
.../venv-ivf-cuvs/bin/python -I modules/05-vector-databases-and-ann/lab/ivf_cuvs_scale.py
```
