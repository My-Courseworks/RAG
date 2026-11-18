# SPANN (Microsoft SPTAG): IVF designed for SSD

- Flavour: **IVF-flat with full-precision vectors on SSD**, centroids ("heads") are real data points kept in RAM behind a SPTAG tree+graph index; lists ("postings") are length-capped and boundary vectors are **replicated into up to `ReplicaCount` lists** under an RNG test. No PQ/SQ by default (an optional OPQ/PQ quantizer exists; not covered).
- Runs as a C++ library/service (Linux/Windows; Python via SWIG). **Not run here**: no Boost, cmake or SWIG on this macOS arm64 host, so no build was attempted (brief: build only if Boost present).
- Everything below is a **code read + a Python SIMULATION** (`lab/ivf_spann_toy.py`) that mirrors the cited lines. Scale test skipped; paper numbers quoted as theirs.

## 1. Source read

Repo `https://github.com/microsoft/SPTAG`, commit `cafdd0abf261a8b26fbbb5ece132a46c823572f5` (2026-09-29, "Parallel bkt with quantization fix (#471)"). Paths under `AnnService/`.

| File | What it implements |
|---|---|
| `inc/Core/SPANN/ParameterDefinitionList.h` | every option + default (`[SelectHead]` L40-67, `[BuildSSDIndex]` L75-220) |
| `src/Core/SPANN/SPANNIndex.cpp` | `SelectHeadAdjustOptions` L733-776, `SelectHeadDynamicallyInternal` L778-824, `SelectHeadDynamically` L826-899, `SelectHeadInternal` L901-1043 (train = head selection); `Index::SearchIndex` L293-395 (query, `MaxDistRatio` prune L324-330); `AddIndex` L1479-1567; `SearchIndexWithFilter` L512-517 |
| `inc/Core/Common/BKTree.h` | balanced k-means: `KmeansAssign` L230-341, `RefineLambda` L136-153, `RefineCenters` L155-207, `InitCenters` L347-369, `TryClustering` L372-443 (centre snapped to data point L421-424), `DynamicFactorSelect` L446-495, `BuildTrees` L579-668 |
| `src/Core/VectorIndex.cpp` | `ApproximateRNG` L1140-1250: posting assignment, RNG replication test L1210-1222 |
| `inc/Core/SPANN/ExtraStaticSearcher.h` | `BuildIndex` L645-1060: candidate heads, posting cut L781-846, page layout `SelectPostingOffset` L1345; `SearchIndex` L260-411 + `ProcessPosting` L120-129 (SSD reads, dedupe, full-vector distance); `ListInfo` L1169-1180 |
| `inc/Core/SPANN/ExtraDynamicSearcher.h` | SPFresh in-place updates: `RNGSelection` L1362-1402, `Append` L1448-1540, `Split` L670-960 (2-means at L782-785), merge L960-1190, `Reassign` L1540-1586, `CheckIsNeedReassign` L410-425 |
| `inc/SSDServing/SSDIndex.h` | optional `Rerank` with full vectors L250-264 (benchmark tool only) |
| `docs/GettingStart.md` L72-102, L130-160 | reference ini files (SIFT1B: `Ratio=0.12`, `ReplicaCount=8`, `PostingPageLimit=3`, `InternalResultNum=64`) |

Head index itself = a normal SPTAG index (`[BuildHead]` section: BKT or KDT trees + relative-neighbourhood graph, `IndexAlgoType` default `KDT`, examples use `BKT`); not re-read here beyond `BKTree.h`.

## 2. Algorithm

1. **Train = pick heads (`SelectHead`)**. Build one BKT over all N vectors: at each node run balanced k-means with `K=BKTKmeansK` (32) on a `SamplesNumber` (1000) subsample, recurse until a node has `<= BKTLeafSize` (8) points (`BKTree.h` L617-664). Balanced k-means = Lloyd's with assignment cost `dist + lambda * count[k]` (L260-261); `lambda = min(adjusted, base^2 / BKTLambdaFactor / n)`; `BKTLambdaFactor=-1` triggers `DynamicFactorSelect`, a sweep over 1e-3..1e3 keeping the lowest count-std (L446-495). Init: 3 random restarts, keep lowest SSE (L347-369). Stop: centre movement `< 1e-3` or 5 iterations without improvement, max 100 (L384-417). Empty cluster -> re-seeded with the farthest point of the largest cluster (L175-181). **Final step snaps every centre onto its closest real data point** (L419-424): heads are vectors, not means.
2. **Choose which tree nodes become heads** (`SelectHeadDynamically`). Target count = `round(Ratio * N)` (or `Count`). Walk the tree bottom-up; a node is emitted as a head when its subtree holds `>= SelectThreshold` points; if `> SplitThreshold`, also emit the `ceil(size/SplitFactor)` biggest children (L800-818). A binary search over `(SelectThreshold, SplitThreshold)` picks the pair whose head count is closest to `Ratio*N` (L848-887). `Ratio` is a target, not exact. `SelectHeadType=Random` is a shuffle instead (L921-931).
3. **Build head index (`BuildHead`)**: SPTAG BKT/KDT + RNG graph over the heads only; stays in RAM, searched approximately with its own `MaxCheck`.
4. **Assign (`BuildSSDIndex`, `ApproximateRNG`)**. For every non-head vector x: search the head index for `InternalResultNum` (64) nearest heads; walk them in distance order; accept head h_i if, for every already accepted h_j, `RNGFactor * d(h_i,h_j) >= d(x,h_i)` (reject when `RNGFactor * nnDist < Dist`, L1217); stop at `ReplicaCount` (8) accepted. Heads are excluded from postings (`ExcludeHead=true`, `ExtraStaticSearcher.h` L741-752): a head is only findable through the head index.
5. **Cut**: `postingSizeLimit = PostingPageLimit * 4096 / (dim*sizeof(T) + 4)`; a longer list keeps its closest `limit` members, dropped replicas are counted (L781-846). `PostingVectorLimit` (118) raises `PostingPageLimit` so at least 118 vectors fit (L784).
6. **Store**: one contiguous byte run per posting: `[int32 id | raw vector] * n`, page-aligned (`SelectPostingOffset`), optional delta-encoding vs head and zstd (`EnableDeltaEncoding`, `EnableDataCompression`). No residuals by default.
7. **Search** (`Index::SearchIndex`). (a) Head index -> `SearchInternalResultNum` (64) nearest heads. (b) `limitDist = d(q,head_0) * MaxDistRatio`; stop opening heads at the first with `dist > limitDist` (only when `limitDist > 0.1`, L329). (c) The heads themselves enter the result set (translated to full ids, L335-336). (d) One async SSD read per opened posting (`BatchReadFileAsync`, L350); every element: dedupe via hash set `m_deduper` (L125), **exact distance on the stored full vector** (L127), push into a size-K heap. (e) Sort. No ADC, no rescoring (vectors are already full precision); `Rerank` exists only in the benchmark harness.
8. **Updates (SPFresh, `Storage=KV|SPDK` only)**: insert -> `RNGSelection` picks `<= ReplicaCount` heads with the same RNG test -> `Append`; posting over `limit + buffer` -> `Split` by 2-means (`KmeansClustering(K=2)` L782-785), the two new centres become new heads in the head index (`AddIndexId` L862), old head deleted, neighbours checked for `Reassign` (L410-425); posting `<= MergeThreshold` (10) -> `Merge` into nearest head with reassign (L996-1179). Delete = tombstone in `m_versionMap`, garbage-collected on the next split (L744-778). With default `Storage=STATIC`, `AddIndex` returns `Fail` ("Only Support KV Extra Update", L1482-1486).
9. **Filters**: `SearchIndexWithFilter` -> `Fail` "Not Support Filter on SPANN Index!" (L512-517). Only the iterative-scan path applies `m_filterFunc` per posting element (`ExtraStaticSearcher.h` L138), i.e. post-filtering inside the posting scan.

## 3. Hyperparameters

Section · name (code field) · build/query · default · what it trades · FAISS analogue.

| Name | B/Q | Default | Trades | FAISS |
|---|---|---|---|---|
| `[SelectHead] Ratio` (`m_ratio`) | B | 0.2 (paper/ini: 0.10-0.12) | heads = RAM + head-search cost vs posting length | `nlist / N` |
| `Count` (`m_headVectorCount`) | B | 0 (= use Ratio) | overrides Ratio | `nlist` |
| `BKTKmeansK` | B | 32 | branching of hierarchical clustering | k in each k-means level (no analogue; FAISS is flat k-means) |
| `BKTLeafSize` | B | 8 | recursion stop; with N <= 8 no clustering happens | - |
| `SamplesNumber` (`m_iSamples`) | B | 1000 | k-means subsample per node | `max_points_per_centroid` (inverse spirit) |
| `BKTLambdaFactor` (`m_fBalanceFactor`) | B | -1 (auto sweep) | balance vs SSE | none (FAISS k-means is unbalanced) |
| `SelectThreshold` / `SplitThreshold` / `SplitFactor` | B | 6 / 25 / 5 (0 = derive from Ratio, L759-775) | which tree nodes become heads | none |
| `TreeNumber` (`m_iTreeNumber`) | B | 1 | BKTs for head selection | none |
| `SelectHeadType` | B | `BKT` (`Random`) | quality vs build time | random centroids |
| `[BuildSSDIndex] InternalResultNum` | B | 64 | candidate heads per vector during assignment | - |
| `ReplicaCount` | B | 8 | recall at boundaries vs SSD bytes | none (multi-assignment) |
| `RNGFactor` | B | 1.0 | how aggressively near-duplicate heads are skipped (`>1` accepts more replicas) | none |
| `PostingPageLimit` | B | 3 pages (12 KB) | bytes per SSD read vs recall of far members | none (lists unbounded) |
| `PostingVectorLimit` | B | 118 | floor on vectors per posting; raises page limit | none |
| `ExcludeHead` | B | true | heads stored only in RAM | none |
| `EnableDeltaEncoding` / `EnableDataCompression` / `ZstdCompressLevel` | B | false / false / 0 | SSD bytes vs CPU | (residual storage, loosely) |
| `Storage` | B | `STATIC` (`KV`, `SPDK`) | enables SPFresh updates | - |
| `SearchInternalResultNum` | Q | 64 | heads considered per query | **`nprobe`** |
| `MaxDistRatio` | Q | 10000 (= off); paper eps2 -> 1.6 / 8.0 | query-aware pruning of far heads | none (`nprobe` is fixed) |
| `SearchPostingPageLimit` | Q | 3 | pages actually read per posting | none |
| `MaxCheck` (SSD section) | Q | 4096 | result-set/dedup capacity; head index has its own `MaxCheck` | `efSearch`-like on the head index |
| `HashTableExponent` | Q | 4 | dedup table size | - |
| `IOThreadsPerHandler`, `NumberOfThreads` | Q/B | 4 / 16 | parallelism | - |
| `MergeThreshold`, `ReassignK`, `DisableReassign`, `BufferLength` | B (SPFresh) | 10 / 0 / false / 3 | update maintenance | none |

No `m`/`nbits` dial: SPANN stores full vectors (PQ only via an optional quantizer file).

## 4. Toy (SIMULATION, `lab/ivf_spann_toy.py`)

Values A..H = 1, 2, 4, 7, 8, 11, 14, 15; d = 1 (SPTAG allows any `Dim`); `Count=3` -> `Ratio=0.375`; seed 0; squared L2.

1. `SelectHeadAdjustOptions`: `BKTKmeansK` 32 -> 3 (capped at head count, L752-757); thresholds stay 6/25/5.
2. **Faithful defaults fail on 8 points**: `BKTLeafSize=8` makes the root a single leaf (8 <= 8), no k-means runs, and `SelectHeadDynamically` can only emit 2 heads from a flat tree (`A, B`, i.e. tree order = arbitrary). The selector is tuned for N >> `SelectThreshold`.
3. With `BKTLeafSize=1` the root runs balanced 3-means. `DynamicFactorSelect` sweep: lambdaFactor 1e-3..1e-2 give counts `[3,2,3]` with heads `{G,E,B}`, 0.1..1e3 give `[2,3,3]` with heads `{H,E,B}`; all have std/avg 0.177, first wins -> `lambdaFactor=0.001`. Centres snapped to data: **heads = B=2, E=8, G=14** (not means 2.33/8.67/14.5). `SelectHeadDynamically` on that tree returns exactly these 3 with `(Select,Split)=(2,15)`. Rule mirrored: root-level balanced k-means + snap-to-data (`BKTree.h` L372-443) and the subtree-size selector (`SPANNIndex.cpp` L778-899).
4. Assignment, `ReplicaCount=2`, `RNGFactor=1.0` (per-vector log in script output):
   - A=1: B accepted (d2=1); E rejected since d2(E,B)=36 < d2(A,E)=49; G rejected. -> 1 copy.
   - C=4: B (4) and E (16) accepted -> 2 copies. D=7: E (1), B (25) -> 2 copies. F=11: E (9), G (9) -> 2 copies.
   - H=15: G accepted; E, B rejected (shadowed by G). -> 1 copy.
   - Rule in 1-D: a point between two heads always passes (d(x,h2) < d(h1,h2)); a point outside both never does. Only boundary vectors get replicated, exactly the paper's intent.
5. Postings (`[id|vector]`, 8 B/elem; limit = 3*4096/8 = 1536, nothing cut): `B:[A,C,D]`, `E:[D,F,C]`, `G:[H,F]`. Heads B,E,G live only in RAM.
6. Plain IVF baseline (Lloyd's, means, single assignment): centroids 2.333 / 8.667 / 14.5; lists `[A,B,C]`, `[D,E,F]`, `[G,H]`.
7. Query 9.5, k=2 (exact E=8, F=11): `SearchInternalResultNum=1` opens E only, scans 3 elements, HIT. `=2` opens E,G, scans 5, HIT. `=2, MaxDistRatio=1.5`: `limitDist = 2.25*1.5 = 3.375`, G (d2=20.25) pruned -> same as 1. Plain IVF `nprobe=1` HIT too (9.5 is not near a plain-IVF boundary).
8. Query 12.0, k=1 (exact F=11): **plain IVF `nprobe=1` MISS -> G=14** (nearest centroid 14.5, F sits in list 1; boundary at 11.58). **SPANN `SearchInternalResultNum=1` HIT**: head G opened, posting `G=[H,F]` contains F because F passed the RNG test for its second head. That is the boundary miss replication removes; plain IVF needs `nprobe=2`. `MaxDistRatio=1.5` prunes E (16 > 6) with no loss.
9. Cost view: SPANN at nprobe=1 scans 2-3 stored vectors + 1 SSD read; replication made the index 8 posting entries for 5 non-head vectors (1.6x). No SPANN miss exists in this toy: with 2 replicas in 1-D every interior point is in both neighbouring postings.

## 5. Scale

**Skipped**: SPTAG not built here (no Boost/cmake/SWIG; brief says build only if Boost present). `lab/ivf_spann_scale.py` keeps the shared generator and the SPANN config to use (`Count=4096`, `ReplicaCount=8`, `PostingPageLimit=3`, dial `SearchInternalResultNum`) and exits with a SKIPPED line. No numbers fabricated.

Paper's own numbers (Chen et al., SPANN, NeurIPS 2021, arXiv 2111.08566), quoted as theirs:
- Abstract: "SPANN is 2x faster than ... DiskANN to reach the same recall quality 90% with same memory cost in three billion-scale datasets. It can reach 90% recall@1 and recall@10 in just around one millisecond with only 32GB memory cost."
- §4.2 settings: "at most 8 closure replicas", "max posting list size to 12KB for byte vectors and 48KB for float vectors", eps1 = 10.0, eps2 = 0.6 (recall@1) / 7.0 (recall@10).
- §4.2.1: heads = "about 10-12% of total vector number", memory "about 32GB for SIFT1B and SPACEV1B ... 60GB for DEEP1B"; Fig. 6(a) SIFT1B: "more than two times faster than DiskANN to reach the 95% recall@1 and recall@10"; Fig. 6(b) SPACEV1B: "DiskANN cannot achieve ... 90% in less than 4ms ... SPANN ... 90% in just around 1ms".
- Mapping to code: eps2 -> `MaxDistRatio = 1 + eps2`; the paper's eps1 closure rule (eq. 2) is **not** in the static build path read here — `ApproximateRNG` only applies the RNG test (eq. in §3.2.2: skip c_ij if `Dist(c_ij, x) > Dist(c_ij-1, c_ij)`) up to `ReplicaCount`.

## 6. SPANN vs plain FAISS IVF (why "IVF for SSD")

- **Heads are data points, not means**, picked by hierarchical balanced k-means + snap; each head distance computed during navigation is also a real candidate (heads enter the result set).
- **Posting length is bounded in bytes** (`PostingPageLimit` pages) so one posting = one bounded SSD read; balance comes from the lambda-penalised k-means, overflow is cut, not spilled.
- **Boundary replication with an RNG test** (`ReplicaCount`, `RNGFactor`): a vector lands in up to 8 postings but only in heads that are in different directions; FAISS assigns once.
- **`nprobe` is a ceiling, `MaxDistRatio` is the dial**: heads farther than `ratio * d(q, head_0)` are skipped per query, so easy queries read fewer pages.
- **Full vectors on disk, no PQ/ADC**: distances in the posting scan are exact, no refine step needed; RAM holds only ~10-12% of vectors (heads) plus their graph.
- **In-place updates by split/merge/reassign (SPFresh)** instead of re-training: a hot posting is 2-means-split into two new heads added to the head index; FAISS IVF never changes its centroids after `train`.

## 7. Reproduce

```bash
python3 -m venv /tmp/venv-ivf-spann && /tmp/venv-ivf-spann/bin/pip install numpy
cd modules/05-vector-databases-and-ann/lab
/tmp/venv-ivf-spann/bin/python -I ivf_spann_toy.py     # simulation, prints sections 4.1-4.9
/tmp/venv-ivf-spann/bin/python -I ivf_spann_scale.py   # prints SKIPPED unless `import SPTAG` works
# source read: git clone --depth 1 https://github.com/microsoft/SPTAG  (commit cafdd0a)
```
