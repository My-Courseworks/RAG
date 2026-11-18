# IVF in Knowhere (Milvus / Zilliz)

- **Flavour**: a patched FAISS. One `IvfIndexNode` template wraps `IVF_FLAT`, `IVF_FLAT_CC`, `IVF_SQ8`, `IVF_SQ_CC`, `IVF_PQ`, `IVF_RABITQ`, `IVF_RABITQ_FASTSCAN`, `SCANN` (= IVF-PQ-FastScan + raw-vector reorder) and `BIN_IVF_FLAT`. Coarse quantizer is always a flat k-means codebook (`IndexFlat`), never a tree or HNSW.
- **Where it runs**: inside Milvus query nodes (C++, no pip). On this Mac I ran it through **Milvus Lite 3.2.2** (`pymilvus[milvus_lite]`), which embeds a Milvus binary and therefore a (version-unknown, older) Knowhere.
- **Ran the real thing?** Yes for `IVF_FLAT`, toy and scale. Two caveats found along the way and documented below: Lite silently drops `nlist` on the `MilvusClient` API path, and Lite splits 187k rows into two segments with an independent IVF each. The step-by-step k-means on the 8 raw values is additionally shown as a labelled SIMULATION, because Knowhere exposes no centroids.

## 1. Source read

Repo `https://github.com/zilliztech/knowhere`, commit `560fc876d3ec6b27532b1620443df1cbea6cdffb` (shallow clone, 2026-11-15). Line numbers are from that commit.

| File | What it does |
|---|---|
| `src/index/ivf/ivf_config.h` | Every IVF config field with default, range and build/query scope (`IvfConfig` l.25-68; PQ l.113-178; SCANN l.180-254; SQ l.256-318; RaBitQ l.379-446; RaBitQ-FastScan l.457-519) |
| `src/index/ivf/ivf.cc` l.481-491 | `MatchNlist`: clamps `nlist` to `max(1, N // 39)` when `nlist * 39 > N` |
| `src/index/ivf/ivf.cc` l.549-824 | `TrainInternal`: picks SuperKMeans vs Clustering, builds the FAISS index per variant, `train()` |
| `src/index/ivf/ivf.cc` l.826-862 | `Add`: `index_->add(rows, data)` → FAISS `quantizer->assign` to the single nearest centroid |
| `src/index/ivf/ivf.cc` l.904-1188 | `Search`: one FAISS `search(1, q, k, …)` per query on a thread pool; bitset → `BitsetViewIDSelector`; per-variant `nprobe` / `ensure_topk_full` / `refine_k` handling |
| `src/index/ivf/ivf.cc` l.1249-1460 | `RangeSearch`: `nprobe = nlist` plus `max_empty_result_buckets` early stop |
| `src/index/clustering_config.h` l.21-35 | Global init method (`RANDOM` default, or k-means++) and `early_stop_threshold` |
| `include/knowhere/bitsetview_idselector.h` | `is_member(id) = !bitset.test(id)`: the Milvus filter bitset, as a FAISS `IDSelector` |
| `thirdparty/faiss/faiss/IndexIVF.h` l.68-98 | Knowhere's extended `SearchParametersIVF`: `nprobe`, `max_codes`, `max_lists_num`, `ensure_topk_full`, `max_empty_result_buckets` |
| `thirdparty/faiss/faiss/IndexIVF.cpp` l.430-483, l.655-677 | Probe loop with `max_codes` budget and the `ensure_topk_full` relaxation |
| `thirdparty/faiss/faiss/Clustering.h` l.24-85 | Defaults used unchanged: `niter=25`, `nredo=1`, `min_points_per_centroid=39`, `max_points_per_centroid=256`, `seed=1234`, `init_method=RANDOM` |
| `thirdparty/faiss/faiss/Clustering.cpp` l.104-121 | Subsample to `k*256` training points if larger; only *warn* if fewer than `k*39` |
| `thirdparty/faiss/faiss/SuperKMeans.h` | Knowhere's fast k-means (GEMM on the leading `d'` dims + ADSampling pruning); used when `d >= 32` |
| `thirdparty/faiss/faiss/cppcontrib/knowhere/` | Knowhere's FAISS fork layer: `IndexIVFFlat`/`CC`, `IndexIVFScalarQuantizerCC`, `IndexScaNN`, `IndexRefine`, `IndexCosine`, iterator workspaces |

## 2. The algorithm

1. **Clamp nlist** (`ivf.cc` l.481-491): if `nlist * 39 > N`, `nlist = max(1, N // 39)` with a warning. HEAD only; Lite 3.2.2 does not do this (see section 6).
2. **Normalise for COSINE**: PQ / SQ / RaBitQ normalise the dataset in place and use an inner-product index; `IVF_FLAT` uses `IndexIVFFlatCosine` which stores inverse L2 norms instead (`ivf.cc` l.561-566, l.603-605).
3. **Train the coarse codebook**: FAISS `Clustering` (Lloyd's, random init, `niter=25`, `nredo=1`, `seed=1234`, no spherical) on at most `nlist*256` sampled points; if `d >= 32` and `use_super_kmeans` (default true) it runs `SuperKMeans` instead: iteration 0 is full Lloyd's, later iterations do GEMM on the leading `d'` dimensions and prune the rest with ADSampling bounds (`SuperKMeans.h` l.36-70). Knowhere does not override `niter` or the point limits (grep of `src/index/ivf`: none).
4. **Train the in-list encoder** on the same training data: nothing for FLAT; SQ min/max per dimension (`sq4|sq6|sq8`, or `code_size` 4/6/8/16 for `IVF_SQ_CC`); PQ codebooks (`m` sub-quantizers, `nbits`, clamped by `MatchNbits` l.494-511); RaBitQ with `rbq_bits` ∈ [1,9] bits per dimension; SCANN = PQ-FastScan with `nbits=4` and `m = dim / sub_dim` (`sub_dim=2`) (`ivf.cc` l.679-689).
5. **Assign** (`Add`, l.826-862): FAISS `IndexIVF::add` → `quantizer->assign` = **single nearest centroid**, no multi-assignment, no replication. PQ and RaBitQ encode the **residual** `x - c` (`by_residual` is hard-coded true for RaBitQ: `IndexIVFRaBitQ.h` comment). FLAT and SQ store the vector itself.
6. **Store**: `(nb + nlist) * (code_size + 8)` bytes for FLAT, i.e. codes + 64-bit ids per list + centroids (`Size()`, l.299-301). Optional **refine store**: `refine=true` keeps a second flat copy of every vector in `refine_type` ∈ {`sq6`,`sq8`,`fp16`,`bf16`,`fp32`} for PQ / SQ / RaBitQ (`IndexRefine`).
7. **Search** (`Search`, l.904-1188). For each query, on a pool thread, one FAISS call with Knowhere's `SearchParametersIVF`:
   - `IVF_FLAT`, `IVF_SQ8`, `IVF_PQ`: `nprobe` passed straight through, `max_codes = 0`, exact scan of the `nprobe` nearest lists (`l.1159-1172`). FAISS caps `nprobe` at `nlist` (`IndexIVF.cpp` l.431-432).
   - `IVF_FLAT_CC`, `IVF_SQ_CC` (growing-segment variants) and `SCANN`: if `ensure_topk_full` (default **true** for CC, **false** for SCANN) then `nprobe := nlist` and `max_codes := (nprobe / nlist) * (ntotal - bitset.count())`, so the scan walks lists in centroid order until it has computed that many *unfiltered* distances, never stopping with fewer than `k` candidates (`ivf.cc` l.970-980, l.996-1007; `IndexIVF.cpp` l.480-483, l.658-677).
   - PQ uses ADC lookup tables (upstream FAISS); SCANN uses 4-bit SIMD FastScan tables; RaBitQ quantises the query to `rbq_bits_query` bits (0 = keep fp32 query).
8. **Rescore**: `refine_k` (query-time, float ≥ 1) makes FAISS `IndexRefine` fetch `refine_k * k` candidates from the IVF stage and re-rank them against the refine store (`ivf.cc` l.1043-1051, l.1107-1115, l.1145-1153). SCANN always re-ranks `max(reorder_k, k)` candidates against the raw vectors when `with_raw_data=true` (l.1012-1015).
9. **Range search** opens *all* lists (`nprobe = nlist`) but stops after `max_empty_result_buckets` (default 2) consecutive lists contributed nothing inside the radius (`l.1308-1309`, `IndexIVF.cpp` l.844-854).

## 3. Filtering and updates

- **Filter = bitset pushed into the list scan.** Milvus evaluates the scalar filter first, producing a bitset over the segment's row offsets; Knowhere wraps it as `BitsetViewIDSelector` and FAISS's scanners call `sel->is_member(id)` on every code *before* the distance (`IndexIVFFlat.cpp` l.377; `ResultHandler.h` l.69). The probed lists are still only `nprobe` of them, so a very selective filter can leave fewer than `k` hits unless `ensure_topk_full` widens the scan (CC / SCANN only). Milvus's `max_codes` budget also subtracts `bitset.count()` so filtered-out rows do not eat the budget (`ivf.cc` l.977).
- **pgvector contrast**: pgvector's `ivfflat` returns the `probes` lists' rows to the executor and the `WHERE` clause is applied *after* the index scan, so a selective filter shrinks the result set and may return fewer than `LIMIT` rows (the user bumps `probes` or uses iterative scans). Knowhere evaluates the predicate inside the scan, keeps the heap full of *passing* rows only, and can trade `nprobe` for a code budget.
- **Inserts** never re-train: Milvus writes new rows to a growing segment (brute force, or `IVF_FLAT_CC`/`IVF_SQ_CC` interim indexes whose `ConcurrentDirectMap` lists accept concurrent `add` with `ssize`-row chunks); when the segment seals, a fresh IVF is trained for that segment only. There is **one IVF per sealed segment**, so a collection has `segments × nlist` lists and a query probes `nprobe` lists in each.
- **Deletes** are a bitset too (the delete mask ORed into the filter bitset); rows disappear from results immediately and physically on compaction, which rebuilds the segment's index.
- No list split, no re-balancing, no centroid drift handling: rebuild (compaction) is the only correction.

## 4. Hyperparameters

| Name | Build / query | Default | Trades | FAISS analogue |
|---|---|---|---|---|
| `nlist` | build | 128 (range [1, 65536]) | more lists: cheaper probes, worse boundary recall; clamped to `N // 39` at HEAD | `nlist` |
| `nprobe` | query | 8 (range [1, 65536], capped at `nlist`) | recall vs scanned codes | `nprobe` |
| `use_super_kmeans` | build | true | SuperKMeans (fast, `d >= 32`) vs plain Lloyd's | none (FAISS has only `Clustering`) |
| `use_elkan` | build | true | declared, **not consumed** by `ivf.cc` at this commit (only `data_view_index_config.h` sets it) | none |
| `ensure_topk_full` | query | true (CC), false (SCANN); not applied to plain IVF_FLAT/SQ/PQ | guarantees `k` candidates by scanning all lists under a `max_codes` budget | none; approximates `max_codes` + `nprobe = nlist` |
| `max_empty_result_buckets` | query (range search only) | 2 | early stop vs range-search recall | none |
| `m` | build (PQ) | none; `dim % m == 0` | compression vs ADC accuracy | `M` |
| `nbits` | build (PQ) | 8 (range [1, 24], clamped by `MatchNbits`) | codebook size vs accuracy | `nbits` |
| `sq_type` | build (IVF_SQ) | `SQ8` ∈ {sq4, sq6, sq8} | bytes/dim vs accuracy | `QuantizerType` |
| `code_size` | build (IVF_SQ_CC) | 8 ∈ {4, 6, 8, 16} | same, 16 = fp16 | `QT_fp16` etc. |
| `ssize` | build (CC) | 48 (range [32, 2048]) | growing-list chunk size | none |
| `refine` | build (PQ/SQ/RaBitQ) | false | keep a second, higher-precision copy for re-ranking | `IndexRefine` |
| `refine_type` | build | none ∈ {sq6, sq8, fp16, bf16, fp32/flat} | refine-store bytes vs re-rank accuracy | `IndexRefineFlat` is fp32-only |
| `refine_k` | query | 1 (float ≥ 1) | candidates re-ranked = `refine_k * k` | `k_factor` |
| `rbq_bits` | build (RaBitQ) | 1 (range [1, 9]) | bits per dimension | FAISS RaBitQ `nb_bits` |
| `rbq_bits_query` | query (RaBitQ) | 0 = fp32 query (range [0, 8]) | query-side quantisation speed vs recall | `qb` |
| `reorder_k` | query (SCANN) | none → `k`; must be ≥ `k` | re-ranked candidates | `k_factor` |
| `with_raw_data` | build (SCANN) | true | raw copy for re-ranking vs memory | — |
| `sub_dim` | build (SCANN) | 2 | PQ sub-vector width (`m = dim / sub_dim`, `nbits = 4` fixed) | `M = d / sub_dim` |

Milvus docs (fetched 2026-11-15, `milvus.io/docs/ivf-flat.md` etc.) state `nlist` range [1, 65536], default 128, "we recommend you set a value within this range: [32, 4096]", `nprobe` range [1, nlist] default 8. **No `4*sqrt(N)` rule appears on the current pages**, and `ensure_topk_full` / `max_empty_result_buckets` are not documented there at all; they are Knowhere-level knobs.

## 5. Toy: A..H = 1, 2, 4, 7, 8, 11, 14, 15, `nlist = 3`, query 9.5

Milvus accepts `dim=1` FLOAT_VECTOR, so the values are used as-is. Script: `lab/ivf_knowhere_toy.py`. Knowhere exposes no centroids, so bins are **inferred by probing**: search each value with `nprobe=1` and a large `limit`; the ids returned are exactly the members of that value's nearest list.

**Part 1, real Milvus Lite on the 8 rows (nlist honoured via the ORM API):**

1. Build prints FAISS's own `WARNING clustering 8 points to 3 centroids: please provide at least 117 training points`, i.e. 3 lists were trained on 8 points (HEAD's `MatchNlist` would instead have collapsed this to `nlist = 1`; Lite's bundled Knowhere does not clamp).
2. Lists found by probing: `{A=1, B=2, C=4}` centroid 2.333 · `{D=7, E=8}` centroid 7.5 · `{F=11, G=14, H=15}` centroid 13.333.
3. Exact top-3 for 9.5: `E=8, F=11, D=7`.
4. `nprobe=1` opens the list whose centroid is nearest 9.5 (7.5, distance 2.0 vs 3.833): returns `E=8, D=7` only — **two results for k=3, F=11 missed**, because plain `IVF_FLAT` has no `ensure_topk_full`.
5. `nprobe=2` adds the 13.333 list: `E, F, D`, exact.
6. Reachable rows with `limit=8`: 2 / 5 / 8 at `nprobe` 1 / 2 / 3.

**Part 2, SIMULATION of FAISS `Clustering` on the same 8 values (numpy, seed 1234, random init, Lloyd's):** converges in 6 iterations to centroids `2.333, 8.667, 14.5` with lists `{A,B,C} {D,E,F} {G,H}`; query 9.5 at `nprobe=1` opens the 8.667 list and already returns the exact `E, F, D`. Different random init → different local optimum than Lite found; this is the k-means lottery the course should point at: **the miss in Part 1 depends entirely on where the 7/8 vs 11 boundary landed**.

**Part 3, real Lite with each value replicated 150× (±0.01 jitter, 1200 rows ≥ 3·39):** same three lists as Part 1; exact top-10 for 9.5 is five 8.0s and five 11.0s; `nprobe=1` returns only 8.0s → **recall@10 = 0.50**; `nprobe=2` → 1.00.

## 6. Scale: 187,043 × 256, 1,000 queries, k = 10, Milvus Lite 3.2.2 `IVF_FLAT`

Ground truth: numpy brute-force L2 (≈2 s). Timings include Lite's local-RPC overhead, which dominates: ~0.7 ms/query when 100 queries share one call, ~40 ms for a single-query call, **nearly flat across `nprobe`**, so treat the latency columns as "Lite overhead", not scan cost.

| nprobe | recall@10, nlist=4096 | recall@10, nlist=1024 | ms/q batch (4096) | ms/q single (4096) |
|---|---|---|---|---|
| 1 | 0.1084 | 0.1533 | 1.11 | 46.9 |
| 2 | 0.1964 | 0.2819 | 0.74 | 40.5 |
| 4 | 0.3266 | 0.4822 | 0.74 | 40.4 |
| 8 | 0.5151 | 0.7541 | 0.76 | 40.8 |
| 16 | 0.7322 | **0.9761** | 0.75 | 40.8 |
| 32 | 0.9223 | 1.0000 | 0.76 | 41.1 |
| 64 | **0.9977** | 1.0000 | 0.76 | 40.9 |
| 128 | 1.0000 | 1.0000 | 0.83 | 41.3 |
| 256 | 1.0000 | 1.0000 | 0.81 | 41.5 |

- **Smallest `nprobe` for recall ≥ 0.95: 64 at `nlist=4096`, 16 at `nlist=1024`.** Both are 1/64 of `nlist`, the usual IVF rule of thumb.
- Build: 8.3 s (4096) / 2.9 s (1024) after an 11 s insert+flush. Knowhere `Size()` estimate 188.1 MiB (4096) / 185.1 MiB (1024), of which 182.7 MiB is the raw fp32 vectors; Lite's data directory on disk is 378 / 372 MiB because it also keeps the row store.
- **Surprise 1: two segments, two IVFs.** The build log shows `WARNING clustering 128000 points to 4096 centroids` and `... 59043 points to 4096 centroids`: Lite sealed the data as a 128,000-row segment plus a 59,043-row one and trained `nlist=4096` **on each**. The query therefore probes `nprobe` lists in each of 8,192 lists, and the second segment's k-means ran at 14 points/centroid. `nlist=1024` produced no warning (both segments ≥ 1024·39). This is normal Milvus behaviour (one index per sealed segment); it is why Milvus docs discuss `nlist` per segment, and why recall at a given `nprobe` is lower here than a single-index FAISS run would give.
- **Surprise 2: `MatchNlist` absent.** HEAD would have clamped the 59,043-row segment to `nlist = 1514`; Lite's bundled Knowhere kept 4096. The guard is newer than the Lite binary.
- **Surprise 3: `MilvusClient` drops `nlist` on Lite.** pymilvus 3.0.3's `prepare_index_params().add_index(..., params={"nlist": n})` is flattened to a top-level `nlist` key on the wire; Lite 3.2.2 ignores it and trains the **default 128** (observed as `WARNING clustering 1200 points to 128 centroids` and ~9-row lists). The ORM path `Collection.create_index(field, {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": n}})` sends nested JSON and is honoured (`col.indexes[0].params` echoes it). My first scale pass, run through `MilvusClient`, built in 0.8 s and reached recall 0.95 at `nprobe=4`: a de-facto `nlist=128` curve, discarded. Both lab scripts use the ORM path for this reason and say so.
- Lite's documented index list is `FLAT, BRUTE_FORCE, IVF_FLAT, IVF_SQ8, HNSW, HNSW_SQ, AUTOINDEX` (milvus_lite.md); it nevertheless *accepted* `IVF_PQ`, `IVF_RABITQ`, `SCANN`, `IVF_FLAT_CC`, `IVF_SQ_CC` names on an 8-row collection without error. I did not verify what it built for those.

## 7. Knowhere vs plain FAISS IVF

- **Per-segment indexes**: Milvus trains one IVF per sealed segment and merges per-segment top-k; FAISS has one codebook for the whole corpus. `nlist` is a per-segment number.
- **`MatchNlist` / `MatchNbits`** silently shrink `nlist` to `N // 39` and `nbits` to fit `N` (HEAD); FAISS only warns.
- **SuperKMeans** replaces Lloyd's for `d >= 32` (GEMM on leading dims + ADSampling pruning), default on; FAISS has no such path.
- **Bitset filter inside the scan + `ensure_topk_full`/`max_codes`/`max_lists_num`** search parameters, extended in Knowhere's FAISS fork (`SearchParametersIVF`); FAISS has `sel` and `max_codes` but no "scan every list until k passing candidates" mode. Note `ensure_topk_full` is wired only for `IVF_FLAT_CC`, `IVF_SQ_CC` and `SCANN`, not plain `IVF_FLAT`.
- **`_CC` (concurrent) variants** with `ConcurrentDirectMap` inverted lists for growing segments, and **`IVF_SQ_CC`'s `raw_data_store_prefix`** for a just-in-time index with raw data on disk; FAISS has no concurrent-add IVF.
- **Refine as a first-class option** across PQ / SQ / RaBitQ with `sq6|sq8|fp16|bf16|fp32` refine stores and a float `refine_k`; FAISS `IndexRefine` is fp32-only. **IVF_RABITQ + refine in 5 bullets**: (1) each residual `x - c` is rotated and quantised to `rbq_bits` (default 1) bit per dimension plus two floats, 32× compression at 1 bit; (2) the query is optionally quantised to `rbq_bits_query` bits (0 = keep fp32, best recall); (3) the `nprobe` lists are scanned with RaBitQ's unbiased inner-product estimator, giving an *approximate* ordering; (4) with `refine=true` a second store (`refine_type`, SQ8 recommended by the docs) holds every vector at higher precision; (5) at query time `refine_k * k` candidates from step 3 are re-scored against that store and the true top-k is returned, so recall is set by `refine_k` and storage by `refine_type`. Milvus docs: enable it "if a 0.9+ recall rate is needed".

## 8. Reproduce

```bash
python3 -m venv venv-ivf-knowhere && source venv-ivf-knowhere/bin/activate
pip install "pymilvus[milvus_lite]" numpy          # pymilvus 3.0.3, milvus-lite 3.2.2 on macOS arm64
git clone --depth 1 https://github.com/zilliztech/knowhere.git   # read at 560fc876

cd modules/05-vector-databases-and-ann/lab
python -I ivf_knowhere_toy.py   --db /tmp/knowhere_toy.db          # ~15 s; prints Parts 1-3 above
python -I ivf_knowhere_scale.py --nlist 4096 --db /tmp/knowhere_4096.db   # ~1.5 min incl. ground truth
python -I ivf_knowhere_scale.py --nlist 1024 --db /tmp/knowhere_1024.db
```

Both scripts print FAISS's `WARNING clustering N points to K centroids` lines on stderr: that line is the only place Lite reveals the effective `nlist` and the segment split, so do not silence it.
