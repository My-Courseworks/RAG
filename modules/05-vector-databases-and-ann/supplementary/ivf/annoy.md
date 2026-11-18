# Annoy: random-projection trees as the partitioning family's tree member

- Flavour: a forest of binary random-projection trees. Each split is a hyperplane through the midpoint of two "means" found by a tiny 2-means; a leaf holds at most `_K = f + 2` items; every candidate is reranked with the exact distance. So it is IVF-Flat-shaped (partition, probe more to recall more, exact rerank) with a tree instead of k-means as the partitioner, and never quantized.
- Runs here: yes, the real library (annoy 1.17.3, `pip` sdist, Python 3.13, macOS arm64) ran both the toy and the 187,043 × 256 sweep. The stock build crashes on this toolchain (see Reproduce); dropping one compiler flag fixes it.
- Contrast with k-means IVF: no training pass, no `nlist` knob (leaf size is fixed by `f`), redundancy from `n_trees` instead of from a wider `nprobe`, and a query budget `search_k` counted in candidate items rather than in lists.

## 1. Source read

Repo `https://github.com/spotify/annoy`, commit `379f744667aba6b40ba3db8a07678df173a88f74` (2025-10-29, "add python 3.13 support"). Everything is in one header; the PyPI 1.17.3 sdist differs from this commit only in DotProduct refactors (diffed).

| What | Where (`src/annoylib.h`) | One line |
|---|---|---|
| Node layout | `Minkowski::Node` 806-812, `Angular::Node` 450-473 | `n_descendants`, (`a` for Euclidean/Manhattan), `children[2]`, `v[f]`; one fixed-size record for items, leaves and splits |
| Leaf capacity | ctor 968-971 | `_s = offsetof(v) + 4f`; `_K = (_s - offsetof(children)) / 4` = `f + 2` for angular and euclidean |
| Train (none) / split | `two_means` 363-403, `Euclidean::create_split` 849-859, `Angular::create_split` 503-511 | pick 2 random items, 200 random-sample mean updates, normal `p - q`, offset through the midpoint |
| Tree build | `_make_tree` 1344-1445, `_split_imbalance` 1337-1342 | recurse until `≤ _K` items; retry a split up to 3 times if the worse side has ≥ 95 %; random coin-flip split if still > 99 % |
| Forest build | `build` 1037-1078, `thread_build` 1260-1297, `AnnoyIndexMultiThreadedBuildPolicy::build` 1528-1560 | `n_trees` roots, each thread seeded `_seed + thread_idx`; `n_trees = -1` builds until nodes ≥ 2 × items; roots copied to the file tail |
| Search | `_get_all_nns` 1447-1507, `Minkowski::margin/side/pq_distance` 815-835 | one max-heap over all roots, priority = min signed margin along the path; stop when `len(candidates) ≥ search_k`; dedupe; exact distance; `partial_sort` top `n` |
| Persist | `save` 1093-1130, `load` 1167-1222, `on_disk_build` 1012-1035 | raw node array; `load` mmaps it read-only and finds roots by scanning the tail for `n_descendants == n_items` |
| Updates | `add_item_impl` 988-1010, `unbuild` 1080-1091 | adds refused after `build`/`load`; `unbuild` throws every tree away |
| Metrics | Angular 450-536, DotProduct 538-705, Hamming 706-804, Manhattan 874-900 | dot reduces to angular via an extra coordinate (Bachrach 2014); Hamming splits on one random bit |
| Python binding | `src/annoymodule.cc` 300-378, 467 | `get_nns_by_item/vector(…, n, search_k=-1, include_distances=False)`, `build(n_trees, n_jobs=-1)` |

## 2. Algorithm

1. Store. `add_item(i, v)` writes item `i` as node `i` (`n_descendants = 1`, raw `v`, no residual). Ids are positions: memory is `max(id) + 1` nodes; gaps are skipped at build (`thread_build` 1283-1285).
2. Split (no training). For the item set at a node, `two_means` picks two random items `p, q`, then 200 times draws a random item and pulls the nearer mean toward it, weighting by that mean's count (`ic * dist` vs `jc * dist`, 388-389) to keep the two sides balanced. The hyperplane normal is `p - q` normalized; Euclidean adds offset `a = -v · (p + q) / 2` so the plane passes through the midpoint (849-859). Angular normalizes first and uses no offset.
3. Assign. Every item goes to `side = (a + v · x > 0)`; an exact zero is a coin flip (819-825). If the bigger side has ≥ 95 % the split is redrawn, up to 3 times; if still > 99 %, items are assigned by coin flips and the normal is zeroed (1397-1420). Each item lands in exactly one leaf per tree.
4. Recurse until a set has `≤ _K = f + 2` items; that set becomes a leaf node whose `children` array (overflowing into `v`) lists the item ids (1353-1370). A one-item subset is the item node itself (1350-1351).
5. Forest. Steps 2-4 repeat `n_trees` times with the same item set and a continuing RNG stream (`Kiss64Random(_seed + thread_idx)`), giving `n_trees` independent partitions. With `n_jobs > 1` trees are split across threads, so the forest depends on the core count unless `n_jobs` is fixed.
6. Search. Push every root with priority `+inf`. Loop: pop the max-priority node. Item → candidate. Leaf → all its items become candidates. Split → compute `margin = a + v · q`; push the near child with priority `min(parent, |margin|)`-style (`pq_distance` 831-835: near child keeps `min(d, margin)`, far child gets `min(d, -margin)`). Because `v` is unit length, `|margin|` is the query's distance to that plane, so a subtree's priority is a lower bound on how far the query is from anything on the other side of the planes it crossed; the far side of a plane that the query barely clears is tried early. Stop when `len(candidates) ≥ search_k` or the heap is empty (1464-1479).
7. Rescore. Sort candidate ids, drop duplicates (the same item reached in several trees), compute the exact metric for each, `partial_sort` the best `n`, return (1481-1506). There is no approximate scan inside a leaf and no quantization anywhere.
8. Filtering: none in the API. Deletes: none. Adds after build: refused (`add_item_impl` 994-997). `unbuild()` drops all trees so items can be added, then `build()` rebuilds every tree from scratch.

## 3. Hyperparameters

| Name | Build / query | Default | Trades | FAISS analogue |
|---|---|---|---|---|
| `f` (dimension) | build | required | also fixes leaf capacity `_K = f + 2` and node size `16 + 4f` bytes (euclidean) | `d` |
| `metric` | build | `'angular'` | angular / euclidean / manhattan / hamming / dot; dot adds a coordinate and reduces to angular | `METRIC_L2` / `METRIC_INNER_PRODUCT` (approx.) |
| `n_trees` (`build(q)`) | build | none; `-1` builds until nodes ≥ 2 × items | more trees = more independent partitions, more build time, bigger file (each tree adds its split and leaf nodes: ≈ 2,000 per tree at N = 187k, f = 256); recall at fixed `search_k` rises only weakly | none. Closest: several IVF indexes with different k-means seeds searched jointly; not the same as `nprobe` |
| `n_jobs` | build | `-1` (all cores) | build speed; changes which trees come out unless fixed | `omp_set_num_threads` (but FAISS results do not depend on it) |
| `set_seed(s)` | build (before `build`) | `1234567890987654321` (Kiss64) | reproducibility only | `cp.seed` |
| `search_k` | query | `n * n_trees` | budget of candidate items gathered across the whole forest before exact rerank; the only recall dial at query time | `nprobe`, but counted in items not lists: `search_k ≈ nprobe × avg list size`, spread over trees by margin |
| `n` | query | required | enters the default `search_k` only | `k` |
| `include_distances` | query | `False` | returns `sqrt` distances (euclidean) or `sqrt(2 - 2cos)` (angular) | always returned |
| `on_disk_build(fn)` | build | off | build into an mmapped file instead of RAM | none (FAISS builds in RAM, `write_index` later) |
| `prefault` (`save`/`load`) | load | `False` | `MAP_POPULATE` read-ahead vs lazy page faults | none |
| `_K = f + 2` (leaf size) | derived, not settable | — | this is the "list size"; the number of leaves per tree ≈ N / (0.5-1) `_K` plays the role of `nlist` and cannot be chosen | `nlist` (implicit) |
| `two_means` 200 steps, 3 split retries at 0.95, random fallback at 0.99, Hamming 20 tries | constants in source | — | balance of the tree | `niter`, `min_points_per_centroid` (no equivalent dial) |

No `m`, `nbits`, residuals or codebooks: Annoy has no compressed form. The index is the raw vectors plus the trees (1.1× to 2.1× the raw bytes below).

## 4. Toy: 8 values, f = 1, euclidean, seed 7

Values A..H = 1, 2, 4, 7, 8, 11, 14, 15. At `f = 1` a node is 20 bytes and `_K = 3`, so a leaf holds ≤ 3 items and 8 items need ≥ 3 leaves per tree. The trees below were parsed from the saved file (`lab/ivf_annoy_toy.py`); the walk is a Python replay of `_get_all_nns` and matched `get_nns_by_vector` on every call. Query `9.5`, `n = 2`; exact answer E(8) and F(11), both at distance 1.5.

`n_trees = 1` (file 280 B = 8 items + 5 tree nodes + 1 root copy):

```
split 13: right iff x < 8.8137
  leaf 8:  {F(11), G(14), H(15)}            # left: x ≥ 8.81
  split 11: right iff x < 5.0088
    leaf 9:  {D(7), E(8)}
    leaf 10: {A(1), B(2), C(4)}
```

Hyperplane thresholds 8.81 and 5.01 are the midpoints `two_means` converged to (not data-chosen medians: 8.81 cuts between E(8) and F(11)).

1. `search_k = 1`: pop root 13 (prio +inf), margin for 9.5 is `-9.5 + 8.8137 = -0.686` → left child (leaf 8) gets priority `min(inf, +0.686) = +0.686`, right child (split 11) gets `-0.686`.
2. Pop leaf 8 → candidates {F, G, H}: 3 collected ≥ budget 1, stop. Rerank exactly: F 1.5, G 4.5, H 5.5. Return F(11), G(14). **Miss: E(8)** sits 0.69 on the other side of the 8.81 plane. Recall@2 = 1/2.
3. `search_k = 2` and `3`: identical. A leaf is added whole, so any budget ≤ 3 is spent by the first leaf; `search_k` only has a visible effect at leaf-size granularity.
4. `search_k = 4`: after leaf 8 (3 candidates < 4) pop split 11 (prio −0.686): margin `-9.5 + 5.0088 = -4.49`, left child leaf 9 keeps `min(-0.686, +4.49) = -0.686`, right child leaf 10 gets −4.49. Pop leaf 9 → {D, E}; 5 ≥ 4, stop. Rerank: E 1.5, F 1.5, D 2.5, G 4.5, H 5.5 → E(8), F(11). Repaired by budget; cost was 5 of 8 items reranked.

`n_trees = 3` (560 B = 8 items + 17 tree nodes + 3 root copies). Trees 0 and 1 are near copies (thresholds 8.814 / 5.009 and 8.815 / 4.904: different random draws, but on 8 points the 200-step 2-means converges to nearly the same pair of means). Tree 2 is different:

```
split 27: right iff x < 7.6596
  split 20: right iff x < 11.9252
    leaf 18: {G(14), H(15)}
    leaf 19: {E(8), F(11)}
  split 23: right iff x < 3.5693
    leaf 21: {C(4), D(7)}
    leaf 22: {A(1), B(2)}
```

5. `search_k = 1`: the three roots pop first (all +inf). Root 25: margin −0.686. Root 26: −0.685. Root 27: `-9.5 + 7.66 = -1.84` → its left child split 20 gets +1.84, the largest priority in the heap, so tree 2 is descended first: in split 20 the margin is `+2.43` (9.5 < 11.93), its right child leaf 19 keeps `min(1.84, 2.43) = 1.84`.
6. Pop leaf 19 → {E, F}: 2 ≥ 1, stop. Rerank: E 1.5, F 1.5 → E(8), F(11). Recall@2 = 2/2 with 2 of 8 items reranked. The second tree repaired the miss because its first plane (7.66) is 1.84 away from the query instead of 0.69, and the heap orders leaves by exactly that margin.
7. `search_k = 2`: same. `search_k = 3`: after leaf 19 (2 < 3) pop leaf 8 from tree 0 (prio +0.686) → {F, G, H}; 5 collected, 4 unique, same answer. `search_k = 4`: identical walk to 3.
8. Default `search_k = n × n_trees = 6` here would add tree 1's leaf 13 {F, G, H} too (duplicates), still 4 unique candidates.

Reading: in Annoy "which bin opens" is decided by signed margin to the nearest plane, pooled over the forest. A single tree's boundary miss is repaired either by spending budget on the next-nearest plane in the same tree (step 4) or by another tree whose planes happen to fall elsewhere (step 6); the forest buys the second option at build time so the query budget can stay small.

## 5. Scale: 187,043 × 256, euclidean, 1000 queries, recall@10 vs brute force

Machine: Apple Silicon, 12 build threads (`n_jobs = -1`), single-threaded Python query loop, timing after a 50-query warm-up of the mmapped file. `_K = 258` items per leaf. Adding the 187,043 items took 1.7 s each time (Python call per item).

| `n_trees` | build s | file MB (× raw) | `search_k` → recall@10 / ms per query |||||||
|---|---|---|---|---|---|---|---|---|---|
| | | | 10 | 100 | 1000 | 3000 | 5000 | 10000 | 50000 |
| 10 | 0.4 | 215.3 (1.12×) | 0.105 / 0.055 | 0.105 / 0.053 | 0.411 / 0.209 | 0.754 / 0.440 | 0.901 / 0.598 | **0.990 / 0.867** | 1.000 / 3.471 |
| 50 | 2.4 | 299.0 (1.56×) | 0.121 / 0.120 | 0.121 / 0.118 | 0.462 / 0.305 | 0.791 / 0.565 | 0.910 / 0.731 | 0.988 / 0.982 | 1.000 / 1.653 |
| 100 | 4.5 | 403.4 (2.11×) | 0.130 / 0.210 | 0.130 / 0.204 | 0.474 / 0.418 | 0.810 / 0.748 | 0.923 / 0.923 | 0.992 / 1.127 | 1.000 / 1.867 |

Smallest `(n_trees, search_k)` reaching recall@10 ≥ 0.95: **`n_trees = 10, search_k = 10000`** (0.990, 0.87 ms/query); it is also the fastest configuration above 0.95. `search_k = 5000` falls just short at every forest size (0.90-0.92), so the crossing is between 5,000 and 10,000 candidates, i.e. 2.7-5.3 % of the items reranked. Brute force ground truth for the 1000 queries took 1.6 s in numpy.

Notes:

- `search_k ≤ 100 ≤ _K`: the walk stops at the first leaf it reaches (≈ 130-258 items, 0.1 % of the data), so recall ≈ 0.10-0.13 and does not move between 10 and 100. `search_k` has no effect below the leaf size, exactly as in the toy.
- `search_k = 10000` (5.3 % of the items reranked) reaches ≈ 0.99 with any forest size. Going from 10 to 100 trees moves recall at a fixed budget by only +0.02 to +0.06 (0.754 → 0.810 at 3000, 0.901 → 0.923 at 5000) while roughly doubling the query time: candidates reached through several trees are duplicates that still count toward `search_k` before dedupe, and every extra root is a heap entry to pop. On this data the budget, not the forest, is the recall dial.
- More trees do help at large budgets: 50,000 candidates cost 3.5 ms with 10 trees but 1.7-1.9 ms with 50-100, because with few trees the walk has to dig into far branches of each tree to fill the budget.
- Build is cheap: no k-means, every split is 200 distance computations plus one pass over the subset. 100 trees in 4.5 s on 12 threads versus a FAISS `nlist = 4096` training pass; the price is the file: 215 MB (1.12× raw), 299 MB (1.56×), 403 MB (2.11×) for 10 / 50 / 100 trees, because every tree stores its ≈ 2000 split/leaf nodes at a full 1040 bytes each (a leaf node is the same size as a 256-d vector).
- Default `search_k = n × n_trees` = 100 / 500 / 1000 here, which gives recall 0.10-0.47 on this data. The default is tuned for Spotify-sized item counts with small `n_trees × n`; set `search_k` explicitly.

## 6. Annoy vs plain FAISS IVF

- Partitioner: hyperplane splits chosen from two random items (plus a 200-step mean pull), not Voronoi cells of trained centroids. Build is O(N log N) per tree with no training set, no `niter`, no "too few points per centroid" warning, and no way to pick `nlist`: leaf size is pinned at `f + 2` by the node layout.
- Redundancy comes from `n_trees` independent partitions, each costing its own split and leaf nodes of `16 + 4f` bytes (≈ 2,000 nodes, 2 MB per tree at N = 187k, f = 256). FAISS has one partition and buys recall with `nprobe`; Annoy can buy it at build time instead, and a boundary miss in one tree is repaired by another tree whose planes fall elsewhere (toy step 6).
- Probe order is a best-bin-first heap over the whole forest keyed by signed margin to the planes crossed; `search_k` is a budget of candidate items pooled over trees, with leaves added whole and duplicates counted. FAISS orders lists by centroid distance and probes exactly `nprobe` lists.
- Always IVF-Flat: candidates are reranked with the exact metric on the stored raw vectors, so there is no `m`/`nbits`, no ADC tables, no memory compression. Index size is 1.1-2.1× the raw vectors here.
- Static by design: `add_item` is refused after `build` or `load`; no delete; `unbuild` + `build` rebuilds every tree. The payoff is a single flat file that `load` mmaps read-only and many processes share, with `on_disk_build` for data that does not fit in RAM. No metadata filtering at all (post-filter with a larger `search_k`).
- Random splits ignore density: a split through two random items of a dense cluster cuts that cluster, so near-duplicate clusters get fragmented across leaves, while k-means would give them cells of their own. `n_trees` is the only remedy.

## 7. Reproduce

```bash
python3 -m venv venv-ivf-annoy && source venv-ivf-annoy/bin/activate
pip install numpy
# the PyPI wheel and a stock sdist build both crash (SIGSEGV / SIGBUS) in _get_all_nns on
# macOS arm64 with Apple clang 17 and Python 3.13 (upstream issue #685). setup.py adds -ffast-math,
# which lets the compiler assume no infinities while the search seeds every root with +inf.
# Rebuild from the sdist with the same flags minus -ffast-math:
ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-fpermissive,-O3,-mcpu=native,-std=c++14,-DANNOYLIB_MULTITHREADED_BUILD,-mmacosx-version-min=10.12" \
  pip install --no-binary :all: --no-cache-dir annoy==1.17.3
cd modules/05-vector-databases-and-ann/lab
python -I ivf_annoy_toy.py --seed 7          # trees, walks, library cross-check; ~1 s
python -I ivf_annoy_scale.py                 # 3 builds + 21 sweeps; ~2 min; writes nothing to the repo
```

`-O2` without `-ffast-math` also works; `-O3 -ffast-math` (upstream default) crashes on this toolchain. The toy's `n_jobs=1` makes the forest depend only on the seed; the scale script uses `n_jobs=-1`, so the exact trees (not the headline numbers) depend on the core count.
