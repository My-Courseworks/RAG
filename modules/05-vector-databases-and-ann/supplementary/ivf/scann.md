# ScaNN (Google) — IVF as a one-level k-means tree + anisotropic AH

- **Flavour**: single-level k-means tree (`num_leaves` ≈ `nlist`) with brute-force centroid lookup; leaves scored either exactly (`score_brute_force`, float32/int8/bfloat16) or with **asymmetric hashing** = product quantization with a query-aware *anisotropic* encoding loss; optional exact **reorder** of the top `reordering_num_neighbors`. Residual-from-centroid encoding only when the distance is dot product.
- **Where it runs**: CPU only, Linux wheels (`manylinux` x86_64 + aarch64, cp39–cp313). No macOS wheel, no Docker here, source build is Bazel + hours.
- **Did the real thing run here? No.** Everything below is a source read at commit `bb18d9b0cb460fee1a080fa2eeb4109fe3805275` plus a numpy **SIMULATION** of the tree + brute-force-leaf path. The simulation is labelled wherever it appears; AH is described, not simulated.

## 1. Source read

Repo `google-research/google-research`, dir `scann/`, commit `bb18d9b0cb460fee1a080fa2eeb4109fe3805275` (sparse clone, 2026-11-15).

| Stage | File : lines | What it does |
|---|---|---|
| Builder → config proto | `scann/scann_ops/py/scann_builder.py:153-227` (`tree`), `:229-307` (`score_ah`), `:309-322` (`score_brute_force`), `:324-345` (`reorder`) | Python kwargs become a `ScannConfig` text proto; this is where every default lives |
| Search entry | `scann/scann_ops/cc/scann.cc:357-376` | `leaves_to_search > 0` becomes `TreeXOptionalParameters::num_partitions_to_search_override`; without `reorder`, `pre_reorder_nn = final_nn` |
| Training sample | `scann/partitioning/partitioner_factory_base.cc:34-62` | Reservoir-samples `min(expected_sample_size, N)` points, `MTRandom(kDeterministicSeed + 1)` |
| Tree train | `scann/trees/kmeans_tree/kmeans_tree_node.cc:164-269` | One `GmmUtils::ComputeKmeansClustering` call per node; `max_num_levels` defaults to 1 ⇒ flat |
| k-means loop | `scann/utils/gmm_utils.cc:1068-1299` | Init, assignment, convergence test, recompute, re-seed small clusters |
| Init | `gmm_utils.cc:1006-1038` (random), `:926-1004` (k-means++) | Builder default `random_init=True` ⇒ k distinct random sample points |
| Small-cluster re-seed | `gmm_utils.cc:1633-1689` | Cluster below `min_cluster_size` is moved to `big_centre + 1e-7·(random_point − big_centre)` |
| Assign (database) | `scann/partitioning/kmeans_tree_partitioner.cc:124-155`, `:190-293` | Nearest centroid; 2 centroids only with SOAR spilling |
| Assign (query, top-L) | `kmeans_tree_node.cc:129-160`, `kmeans_tree.h:372-451` | All centroid distances, `FastTopNeighbors(max_centers)` ⇒ `FIXED_NUMBER_OF_CENTERS` |
| Residuals + AH train | `scann/base/internal/tree_x_hybrid_factory.cc:357-367`, `:723-733` | Residual path chosen only if `use_residual_quantization`; AH codebooks trained on residuals |
| AH codebooks | `scann/hashes/internal/asymmetric_hashing_impl.cc:42-160` | Plain per-block k-means, `num_clusters_per_block` (16 for `lut16`), 10 iters, seed 1 |
| Anisotropic encode | `asymmetric_hashing_impl.cc:268-273`, `:385-416`, `:438-500` | Parallel/perpendicular residual split, coordinate descent over blocks |
| Leaf scan | `scann/tree_x_hybrid/tree_ah_hybrid_residual.cc:999-1060` | Loop over opened leaves; per-leaf `epsilon = current_topN_epsilon − dist_to_centre` |
| Rescoring | `scann/base/single_machine_base.cc:361-366`, `:805-821` | `ReorderResults` recomputes exact (or int8/bf16) distances for the pre-reorder set |
| Updates | `scann/tree_x_hybrid/mutator.h:951-1000`, `:652-672`, `:576-615` | `AddDatapoint` → nearest centroid, no retrain; incremental mode counts mutations per leaf, splits/reassigns |
| Filtering | `scann/base/search_parameters.h:135-141` | `restricts_enabled()` hard-wired to `false` in OSS; no filter API in the Python layer |

## 2. The algorithm

1. **Sample.** Take `expected_sample_size` (builder: 100 000) points by reservoir sampling; if N is smaller, use all (`partitioner_factory_base.cc:34-42`).
2. **Train centroids.** `GmmUtils::ComputeKmeansClustering` with squared-L2 (the builder hard-codes `partitioning_distance: SquaredL2Distance` regardless of the search metric, `scann_builder.py:210-212`):
   - init: k distinct random sample points (`random_init=True`) or k-means++;
   - `min_cluster_size = min(min_partition_size, sample/k)` (`gmm_utils.cc:1111-1112`);
   - each iteration: nearest-centroid assignment in **float32** (`UNBALANCED_FLOAT32`, `gmm_utils.cc:516-545`), mean per-cluster distance, converged iff every cluster's mean moved ≤ 1e-5 **and** every cluster has ≥ `min_cluster_size` points;
   - otherwise recompute means (`spherical=True` renormalises them) and **re-seed each under-sized cluster next to a random big one** (`gmm_utils.cc:1647-1670`);
   - stop at `training_iterations` (builder 12; proto default 10). Fewer sample points than `num_leaves` is a hard `InvalidArgumentError` (`gmm_utils.cc:1105-1109`), not a silent clamp.
3. **Assign the full dataset.** Every point gets exactly one token = nearest centroid (`TokenForDatapoint`). With `soar_lambda` (dot product only) each point also gets a second, orthogonality-amplified centroid (`kmeans_tree_partitioner.cc:225-242`).
4. **Store.**
   - `score_brute_force()`: raw vectors (float32, or int8/bf16 if `quantize`) grouped by leaf (`TreeXHybridSMMD`).
   - `score_ah(...)`: if `distance_measure == "dot_product"` the vector is first **residualised** `x − c_token` (`ResidualizeToFloat`), AH codebooks are trained on residuals, each residual is encoded to one 4-bit code per 2-dim block (`dimensions_per_block=2`, 16 codewords, `lut16`). For `squared_l2` there is no residual step: AH encodes the raw vector (`scann_builder.py:413-415`, `tree_x_hybrid_factory.cc:723-733`).
5. **Search.** Compute the query's distance to **all** centroids (flat tree = brute force over `num_leaves` centroids; `upper_tree()` adds a second coarse level), keep the `num_leaves_to_search` nearest (`FastTopNeighbors`, `kmeans_tree_node.cc:152-158`). For each opened leaf, in order: scan it (exact, or ADC via a per-query int8 LUT16 table of `16 × num_blocks` entries), add results to one shared top-`pre_reorder_nn` heap, and pass that heap's current k-th distance (minus the query–centroid term in residual mode) into the next leaf as an `epsilon` so the SIMD kernel can skip hopeless candidates (`tree_ah_hybrid_residual.cc:1037-1046`). All opened leaves are visited; there is no stop-early-on-leaf-distance rule.
6. **Rescore** (`reorder(k')`): recompute exact distances for the top k' with the stored float32 (or int8/bf16) vectors and return the top `final_num_neighbors` (`single_machine_base.cc:805-821`).

### What "anisotropic" buys over FAISS PQ (Guo et al., ICML 2020)
- Plain PQ picks the codeword minimising ‖x − x̃‖²; it treats every direction of the error equally.
- ScaNN splits the residual r = x − x̃ into the part **parallel to x** (r∥, along x/‖x‖) and the part perpendicular (r⊥) and minimises `h∥·‖r∥‖² + h⊥·‖r⊥‖²` with `h∥ > h⊥` — because for an inner-product query q, the score error ⟨q, r⟩ is dominated by r∥ when q is close to x, which is exactly the case that decides the top-k.
- The paper's weights: with a relevance threshold T (`anisotropic_quantization_threshold`, meaning "pairs with ⟨q,x⟩ ≥ T·‖x‖ matter"), `h∥ ∝ T²/‖x‖²` and `h⊥ ∝ (1 − T²/‖x‖²)/(d−1)`; the code uses only their ratio η = `ComputeParallelCostMultiplier(T, ‖x‖², d)` (`asymmetric_hashing_impl.cc:268-273`). For unit-norm 256-d vectors and T = 0.2, η ≈ 0.04·255/0.96 ≈ 10.6, i.e. parallel error costs ~10× more than perpendicular error.
- Encoding is per point: start from the min-‖r‖ codeword in each block (= plain PQ), then up to 10 rounds of coordinate descent over blocks, accepting a swap only if it lowers `η·Δ‖r∥‖² + Δ‖r⊥‖²` (`OptimizeSingleSubspace`, `:385-416`; `IndexDatapointNoiseShaped`, `:438-500`).
- Codebooks themselves are **ordinary k-means** (`TrainAsymmetricHashing`, `:42-160`); the proto's `use_noise_shaped_training` is deprecated. So FAISS PQ codebooks + ScaNN encoding would already differ only at encode time.

## 3. Hyperparameters

Build = rebuild to change; Query = per-call. Defaults are the Python builder's (`scann_builder.py`), proto defaults in brackets where they differ.

| Name (builder) | Proto field | Build/Query | Default | Trades | FAISS analogue |
|---|---|---|---|---|---|
| `tree(num_leaves)` | `partitioning.num_children` | Build | required [2] | more leaves = finer pruning, slower centroid scan, smaller leaves | `nlist` |
| `tree(num_leaves_to_search)` | `query_spilling.max_spill_centers` | Build default, **Query** override `search(..., leaves_to_search=L)` | required | recall vs scanned fraction | `nprobe` |
| `tree(training_sample_size)` | `expected_sample_size` | Build | 100 000 | k-means quality vs train time | `max_points_per_centroid`-ish (FAISS caps *per centroid*, ScaNN caps the total) |
| `tree(min_partition_size)` | `min_cluster_size` | Build | 50 [1] | clusters below this are re-seeded; also blocks convergence | none (FAISS tolerates empty lists) |
| `tree(training_iterations)` | `max_clustering_iterations` | Build | 12 [10] | k-means iterations | `cp.niter` (FAISS 25) |
| `tree(spherical)` | `partitioning_type: SPHERICAL` | Build | False | unit-norm centroids for cosine data; input must be unit-norm or it errors | `cp.spherical` |
| `tree(quantize_centroids)` | `query_tokenization_type: FIXED_POINT_INT8` | Build | False | int8 centroids: faster centroid scan, tiny recall loss | none |
| `tree(random_init)` | `single_machine_center_initialization` | Build | True (RANDOM) [KMEANS++] | init quality vs time | FAISS uses random init too |
| `tree(incremental_threshold)` | `incremental_training_config.{fraction,number_of_datapoints}` | Build | None (off) | enables partition reassignment/splitting after upserts | none |
| `tree(avq)` | `partitioning.avq` | Build | None (NaN) | anisotropic *centroids* for dot product; biases centroids toward the direction that matters for MIPS | none |
| `tree(soar_lambda, overretrieve_factor)` | `database_spilling TWO_CENTER_ORTHOGONALITY_AMPLIFIED` | Build | None (off); λ 1.5, factor 2.0 | each point stored in 2 leaves (second chosen to be orthogonal to the first residual); ~2× memory for AH codes, big recall gain at equal `L` | none (closest: multi-assignment in some engines) |
| `upper_tree(num_leaves, num_leaves_to_search)` | `bottom_up_top_level_partitioner` | Build | off | second k-means level over centroids so centroid lookup is not brute force | FAISS `quantizer` = another index (e.g. HNSW); here it is a 2nd k-means |
| `score_ah(dimensions_per_block)` | `projection.num_dims_per_block` | Build | required (docs: 2) | #blocks = d/this; smaller = more codes per vector = more accurate, bigger | `d/m` (i.e. m = d/dims_per_block) |
| `score_ah(hash_type)` | `num_clusters_per_block` + `lookup_type` | Build | `"lut16"` (16 codewords, 4 bits) | `lut256` = 8-bit codes, no SIMD-nibble kernel | `nbits` (4 or 8) |
| `score_ah(anisotropic_quantization_threshold)` | `noise_shaping_threshold` | Build | NaN (plain PQ encoding) | T: lower = more weight on parallel error; 0.2 is the docs' number for unit-norm | none |
| `score_ah(training_sample_size)` | `expected_sample_size` | Build | 100 000 | codebook quality | FAISS trains PQ on the training set given |
| `score_ah(training_iterations)` | `max_clustering_iterations` | Build | 10 | codebook k-means iterations | `cp.niter` of the PQ |
| (auto) residual quantization | `use_residual_quantization` | Build | True iff tree + dot_product | residuals from centroid before AH | FAISS `IVFPQ` always residual (`by_residual=True`) |
| `score_brute_force(quantize)` | `brute_force.fixed_point/bfloat16.enabled` | Build | FLOAT32 | int8/bf16 leaf vectors: 4×/2× less bandwidth | `IndexIVFScalarQuantizer` (SQ8/bf16) |
| `reorder(reordering_num_neighbors)` | `exact_reordering.approx_num_neighbors` | Build default, **Query** override `pre_reorder_num_neighbors` | off | candidates rescored exactly; must exceed k | `IndexRefineFlat` / `k_factor` |
| `reorder(quantize)` | `exact_reordering.fixed_point/bfloat16` | Build | FLOAT32 | int8/bf16 rescoring vectors | `IndexRefine` with SQ |
| `search(final_num_neighbors)` | `num_neighbors` | Query | builder `num_neighbors` | k | `k` |
| `set_n_training_threads(n)` | — | Build | 0 (= all) | train time | `omp_set_num_threads` |
| proto-only: `clustering_seed` | `partitioning.clustering_seed` | Build | 0 (+ `kDeterministicSeed`) | reproducibility; not exposed by the builder, pass a text proto to `create_searcher` | `cp.seed` |
| proto-only: `clustering_convergence_tolerance` | | Build | 1e-5 | | none |
| proto-only: `balancing_type` | | Build | `UNBALANCED_FLOAT32` | `GREEDY_BALANCED` caps leaf sizes | none |
| proto-only: `num_tokenized_branch` | | Build | 1 | multi-branch descent for deep trees | none |
| `autopilot(mode, quantize)` | `autopilot.tree_ah` | Build | off | ScaNN picks `num_leaves`, `L`, AH, reorder from N and d | none |

No FAISS analogue at all: `anisotropic_quantization_threshold`, `soar_lambda`, `avq`, `min_partition_size`, `incremental_threshold`, `autopilot`, `upper_tree`, the per-leaf epsilon propagation. No ScaNN analogue: FAISS `IVFFlat` with `nlist` lists that may be empty (ScaNN re-seeds them), metadata filtering (`IDSelector`), `IndexIVFPQ.polysemous_ht`, RaBitQ/binary flavours.

## 4. Toy — SIMULATION (`lab/ivf_scann_toy.py`)

Data A..H = 1, 2, 4, 7, 8, 11, 14, 15 fed as 2-D `(x, 0)` (ScaNN takes an `(N, D)` array; whether it accepts `D=1` with `score_brute_force` could not be verified here). `num_leaves=3`, builder defaults otherwise, numpy seed 0.

1. `min_cluster_size = min(50, 8 // 3) = 2` (`gmm_utils.cc:1111`).
2. Random init picks points E, H, F ⇒ centres 8, 15, 11.
3. Iter 0 assignment: {A,B,C,D,E}→8, {G,H}→15, {F}→11. Cluster 2 has 1 point < 2 ⇒ **re-seeded** at `centre₀ + 1e-7·(rand − centre₀)` ≈ 4.4 after recompute (`gmm_utils.cc:1647-1670`). This is the ScaNN behaviour FAISS does not have: FAISS would have kept the singleton list.
4. Iter 1: centres 4.4 / 14.5 / 4.4 → sizes 2, 3, 3. Iter 2: 7.5 / 13.33 / 2.33 → sizes 2, 3, 3. Iter 3: unchanged ⇒ **converged** (means moved ≤ 1e-5 and all sizes ≥ 2).
5. Leaves (`datapoints_by_token`): leaf 0 centre 7.5 = {D, E}; leaf 1 centre 13.33 = {F, G, H}; leaf 2 centre 2.33 = {A, B, C}.
6. Query **9.5**, exact top-2 = {E (0.5), F (1.5)}. Distances to centres: 4.0, 14.69, 51.36.
   - `leaves_to_search=1` → open leaf 0 → return **{E, D}**, **miss F** (F sits in leaf 1 whose centre is 3.83 away vs 2.0 for leaf 0, yet F is nearer to the query than D).
   - `leaves_to_search=2` → open leaves 0, 1 → return {E, F} = exact.
7. Query **5.5**, exact top-2 = {C (1.5), D (1.5)}. Distances to centres: 4.0, 61.36, 10.03.
   - `L=1` → leaf 0 → **{D, E}**, **miss C**. `L=2` → leaves 0, 2 → {C, D} = exact.
8. In the real `score_brute_force` path the returned distances are exact, so the only error source is the pruning in step 6/7; with `score_ah` the per-leaf distances would additionally be ADC approximations unless `reorder` is on.

## 5. Scale — SIMULATION (`lab/ivf_scann_scale.py`)

Shared generator (N = 187 043, D = 256, NQ = 1000, K = 10, 50 topic centres, noise 0.35). Simulates `tree(num_leaves, L).score_brute_force()` with builder defaults: 100 000-point training sample, random init, 12 iterations, `min_partition_size=50`. Ground truth = exact L2 top-10. `ms/query` is **numpy cost on one core**, not ScaNN latency; recall is the meaningful column. Apple M-series, Python 3.13, numpy 2.5.4.

**`num_leaves = 4096`** — train 12.0 s (+0.6 s assignment); centroids + inverted lists 5.7 MB, raw vectors 191.5 MB kept for the exact leaf scan. `min_cluster_size = min(50, 100000/4096) = 24`; ~2 850 of 4 096 clusters were below it on every iteration ⇒ 34 106 re-seeds over 12 iterations, never converged; final leaves: min 0, median 11, max 362, **119 empty**.

| L | recall@10 | ms/query (numpy) | candidates/query |
|---|---|---|---|
| 1 | 0.039 | 0.03 | 73 |
| 2 | 0.068 | 0.04 | 125 |
| 4 | 0.121 | 0.06 | 226 |
| 8 | 0.210 | 0.10 | 427 |
| 16 | 0.355 | 0.17 | 825 |
| 32 | 0.590 | 0.32 | 1 602 |
| 64 | 0.903 | 0.62 | 3 068 |
| 128 | **1.000** | 1.19 | 5 935 |
| 256 | 1.000 | 2.51 | 11 723 |

Smallest `num_leaves_to_search` with recall@10 ≥ 0.95: **128** (3.1 % of leaves, ~3.2 % of vectors).

**`num_leaves = 1024`** — train 15.2 s; `min_cluster_size = 50`, ~460 clusters below it per iteration, 5 588 re-seeds; final leaves min 0, median 196, max 721, 3 empty.

| L | recall@10 | ms/query (numpy) | candidates/query |
|---|---|---|---|
| 1 | 0.112 | 0.07 | 245 |
| 2 | 0.191 | 0.10 | 433 |
| 4 | 0.325 | 0.18 | 825 |
| 8 | 0.552 | 0.34 | 1 581 |
| 16 | 0.877 | 0.59 | 3 065 |
| 32 | **1.000** | 1.21 | 5 971 |

Smallest L for ≥ 0.95: **32** (3.1 % of leaves). Same fraction of leaves as at 4096, i.e. recall here is a function of the scanned fraction, as expected for 50 isotropic Gaussian blobs: inside a blob k-means cells carry almost no neighbour information in 256-D, so recall ≈ `L / (leaves per topic)` until a whole topic is covered (82 leaves/topic at 4096, 20 at 1024).

Surprising / worth teaching:
- The course's forced `nlist = 4096` is 10× the ScaNN docs' rule (`num_leaves ≈ sqrt(N)` ≈ 430). With the default 100k training sample that leaves 24 points per centroid, and the default `min_partition_size=50` then forces re-seeding of roughly half the clusters **every** iteration, so the k-means never converges and the final partition is lopsided (median 11, max 362, 119 empty). ScaNN assigns in float32 (`gmm_utils.cc:522`), so the 1e-7 re-seed perturbation is below float32 resolution for coordinates ≳ 0.5 and the re-seeded centre often ends up an exact duplicate that captures nothing — the simulation reproduces this; the real library would behave the same way on this point. Fix in practice: `min_partition_size` small (e.g. 1–5) or `training_sample_size ≥ 50·num_leaves`, or trust `autopilot`.
- Recall is identical at equal *fraction* of leaves (128/4096 = 32/1024), so on this generator `nlist` only moves the per-leaf cost, not the recall curve.

## 6. ScaNN-specific vs plain FAISS IVF

- **Centroid search is brute force** over `num_leaves` centroids in float32 (or int8 with `quantize_centroids`); a second k-means level (`upper_tree`) is the only accelerator — no HNSW-over-centroids option.
- **Trains on a sample** (100k default) and **re-seeds under-sized clusters** every iteration; convergence requires all clusters ≥ `min_cluster_size`. FAISS trains on what you give it and tolerates empty lists.
- **Residuals only for dot product**; for squared-L2, AH encodes raw vectors. FAISS `IVFPQ` is residual for both metrics.
- **Anisotropic encoding** (`anisotropic_quantization_threshold`) and 4-bit LUT16 ADC kernels; codebooks are still plain k-means.
- **SOAR** two-leaf storage and **AVQ** centroids — MIPS-specific recall boosters with no FAISS counterpart.
- **Shared top-N epsilon propagated into every leaf scan** (pruning inside leaves, not early exit across leaves); `reorder` rescoring is built into the same searcher rather than a wrapper index.
- **Updates**: `upsert` tokenizes to the nearest existing centroid with no retraining; with `incremental_threshold`, leaves whose mutation count exceeds `fraction·size` (or an absolute count) and that hold ≥ 200 points are queued for reassignment, and partitions can be split (new centroid = old + 1e-6 perturbation, then reassign) or shrunk; `rebalance()` is a full retrain. **No metadata filtering** in OSS (`restricts_enabled()` returns `false`); filter by over-retrieving and post-filtering.

## 7. Reproduce

```bash
python3 -m venv /tmp/venv-ivf-scann && /tmp/venv-ivf-scann/bin/pip install numpy
# scann itself: only Linux wheels exist ->
#   pip install scann            # works on Linux x86_64/aarch64, Python 3.9-3.13
cd modules/05-vector-databases-and-ann/lab
/tmp/venv-ivf-scann/bin/python -I ivf_scann_toy.py                 # ~0.1 s
/tmp/venv-ivf-scann/bin/python -I ivf_scann_scale.py --nlist 4096  # ~30 s
/tmp/venv-ivf-scann/bin/python -I ivf_scann_scale.py --nlist 1024  # ~45 s
```

Source read: `git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research && git sparse-checkout set scann` → commit `bb18d9b0cb460fee1a080fa2eeb4109fe3805275`.

On Linux, the real equivalents of the two simulations are:

```python
import scann
# pure IVF, exact leaves
s = scann.scann_ops_pybind.builder(xb, 10, "squared_l2").tree(
        num_leaves=4096, num_leaves_to_search=128).score_brute_force().build()
idx, dist = s.search_batched(xq, leaves_to_search=L)          # sweep L
# AH + reorder variant
s2 = scann.scann_ops_pybind.builder(xb, 10, "squared_l2").tree(
        num_leaves=4096, num_leaves_to_search=128
     ).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
```
