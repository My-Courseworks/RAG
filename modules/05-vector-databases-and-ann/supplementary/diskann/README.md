# Vamana, DiskANN, FreshDiskANN: what was read, what was run

Companion to the module's
[Vamana, DiskANN, FreshDiskANN section](../../README.md#vamana-diskann-freshdiskann-the-graph-that-lives-on-ssd).
That section states conclusions; this page holds the evidence: which implementation ran, the source
lines each claim rests on, every place the code and the papers differ, the full toy log, the full
result tables, and how to reproduce them.

## What ran here, and what did not

| Item | Status |
|---|---|
| Implementation | Microsoft DiskANN, Python package `diskannpy` 0.7.0 (PyPI, released 2024-02-06) |
| Source read | `github.com/microsoft/DiskANN`, tag `0.7.0`, commit `df225d32da81db87596a3ce7c040ad0f78709a7c` (the C++ code line, branch `cpp_main`) |
| Not the code read | The repository's `main` branch, which is a later Rust rewrite and is not what the 0.7.0 wheel contains |
| How it ran | `diskannpy` publishes wheels for Linux x86-64 and Windows only, no macOS wheel and no source distribution. It ran in a `linux/amd64` Docker container (Python 3.11, `numpy` 1.25.0, `hnswlib` 0.8.0) on Apple Silicon, under Docker Desktop's x86-64 emulation |
| In-memory Vamana (`build_memory_index`, `StaticMemoryIndex`) | Ran: toy and scale test |
| Disk index (`build_disk_index`, `StaticDiskIndex`) | Ran: scale test at four PQ sizes |
| Dynamic index (`DynamicMemoryIndex`: `batch_insert`, `mark_deleted`, `consolidate_delete`) | Ran: freshness test |
| hnswlib baseline | Ran, in the same container |
| FreshDiskANN's SSD system (LTI, TempIndexes, StreamingMerge, redo log) | Could not run: not present in the tagged source (`grep -ril` for `StreamingMerge`, `streaming_merge`, `TempIndex`, `merge_insert`, `redo log`, `long term index` over `src/`, `include/`, `python/`, `apps/`, `workflows/` returns nothing) |
| Sharded disk build (the path taken when the data does not fit `build_memory_maximum`) | Read only; the scale test's 187,043 vectors build in one shot |
| SSD latency | Not measured: the container's disk is a virtual disk inside a VM |

Consequences of the emulation: every timing below is slower than native, by an unknown factor. The
two libraries ran in the same container, so their timings are comparable with each other. Recall,
graph sizes, edge lists and read counts do not depend on it.

**MIRROR** is used below for a Python transcription of the library's build and search loops
(`lab/diskann_toy.py`, `lab/diskann_scale.py`). It is a simulation, used only to narrate steps the
library does not log and to count expansions. Wherever it appears, the graph it is checked against,
or walks, is the one the library saved.

## Source citations (tag `0.7.0`, commit `df225d3`)

| Claim | File and lines |
|---|---|
| Defaults: α 1.2, `MAX_OCCLUSION_SIZE` 750, `GRAPH_SLACK_FACTOR` 1.3, `MAX_DEGREE` 64, `BUILD_LIST_SIZE` 100, `SEARCH_LIST_SIZE` 100, `SATURATE_GRAPH` false, `NUM_FROZEN_POINTS_DYNAMIC` 1, `SECTOR_LEN` 4096, `MAX_N_SECTOR_READS` 128, `MAX_GRAPH_DEGREE` 512 | `include/defaults.h:11-32` |
| Entry point of a static build = `calculate_medoid()` | `src/index.cpp:731-736`, called from `link()` at `1292-1295` |
| The *medoid* is the stored point nearest the centroid, first index on ties | `src/in_mem_data_store.cpp:341-390` |
| L2 distance returns the sum of squared differences, no square root | `src/distance.cpp:177-231` |
| GreedySearch: list of capacity `L`, loop while an unexpanded entry remains, expanded nodes collected as the prune pool | `src/index.cpp:790-974` (`iterate_to_fixed_point`); the list itself: `include/neighbor.h:39-135` (`NeighborPriorityQueue`) |
| Returned hop count is never incremented (`hops` stays 0) | `src/index.cpp:878`, `973` |
| A point's own prune: pool = expanded nodes of its search, itself removed; existing out-neighbours are not merged in | `src/index.cpp:977-1052` (`search_for_point_and_prune`) |
| RobustPrune as implemented: rounds at `cur_alpha = 1, ×1.2, …` while `cur_alpha <= alpha`; skip while `occlude_factor > cur_alpha`; factor = `d(p,c) / d(kept,c)`; pool truncated to `maxc` | `src/index.cpp:1055-1148` (`occlude_list`) |
| Sort pool, call `occlude_list`, optional saturation | `src/index.cpp:1158-1196` (`prune_neighbors`) |
| Back-edges: append while the list is shorter than `floor(1.3 × R)`, else re-prune list ∪ {new} | `src/index.cpp:1199-1261` (`inter_insert`) |
| Build: empty graph, ids in order, `omp parallel for schedule(dynamic, 2048)`, one pass, `set_neighbours` overwrites, then a cleanup pass over lists longer than `R` | `src/index.cpp:1269-1370` (`link`); called once from `build_with_data_populated`, `1507-1559` |
| Graph file format read back by the lab scripts | `src/in_mem_graph_store.cpp:200-230` (`save_graph`) |
| Search returns the first `K` of the list | `src/index.cpp:1958-2017` |
| Insert = `search_for_point_and_prune` + `set_neighbours` + `inter_insert` | `src/index.cpp:2856-3006` (`insert_point`) |
| Lazy delete only records the location and drops the tag | `src/index.cpp:3037-3056` (`lazy_delete`) |
| Dynamic search drops entries without a tag (deleted or frozen) from the result, after walking through them | `src/index.cpp:2143-2218` (`search_with_tags`) |
| Consolidation: full scan of all slots; per live node with a deleted out-neighbour, pool = live out-neighbours ∪ live out-neighbours of the deleted ones; at most `R` → keep all, else `occlude_list` | `src/index.cpp:2297-2361` (`process_delete`), `2365-2465` (`consolidate_deletes`) |
| Deleting the start node throws; a dynamic index starts at a frozen point in slot `max_points` | `src/index.cpp:2415-2418`; `61-73`, `738-754` |
| Slots are released only by consolidation | `src/index.cpp:2447` (`release_locations`), `2872` (`reserve_location`) |
| Disk build parameters passed from Python: `R L B M T [pq_disk_bytes]`, no α | `python/src/builder.cpp:12-24` |
| PQ bytes per vector = `floor(budget_bytes / N)`, at least 1, at most the dimension and `MAX_PQ_CHUNKS` 512; 0.25 GB set aside when the budget exceeds 1.25 GB | `src/disk_utils.cpp:1298-1302`, `79-88`; `include/disk_utils.h:38-39`; `include/pq_common.h:6-11` |
| One-shot build if `estimate_ram_usage` fits the build budget, with `saturate_graph = true` and default α | `src/disk_utils.cpp:628-682`; estimate: `include/index.h:35-43` |
| Otherwise: k-means shards, each point to 2 shards, each built at degree `2R/3`, merged | `src/disk_utils.cpp:684-788`; `src/partition.cpp:523-602` |
| On-disk layout: `max_node_len = (width + 1) × 4 + dims × sizeof(T)`; `nnodes_per_sector = 4096 / max_node_len`; multi-sector records when that is 0; sector 0 is metadata | `src/disk_utils.cpp:847-1098` (`create_disk_layout`); `src/pq_flash_index.cpp:98-118` |
| Beam search: up to `beam_width` nearest unexpanded per round, cache lookup first, one read request per uncached node, PQ distances for neighbours, exact distance for each node read, answer = top `k` of the nodes read | `src/pq_flash_index.cpp:1269-1700` (loop `1418-1612`, final sort `1614-1615`, output `1667-1690`) |
| Node cache by BFS from the medoid, capped at 10% of the nodes | `src/pq_flash_index.cpp:351-496` |
| File opened with `O_DIRECT` | `src/linux_aligned_file_reader.cpp:199-202` |
| Python `beam_width` default 2 | `python/src/_static_disk_index.py:110-111` |
| Python search computes `QueryStats` and discards them; `batch_search` passes none | `python/src/static_disk_index.cpp:57-104` |

## Every place the code differs from the papers

Papers: DiskANN (Subramanya et al., NeurIPS 2019), Algorithms 1 to 3 and Sections 2.3 to 3.5;
FreshDiskANN (Singh et al., arXiv:2105.09613), Algorithms 2 to 4 and Section 5.

| # | Topic | Paper | `diskannpy` 0.7.0 | Evidence |
|---|---|---|---|---|
| 1 | Starting graph | Random `R`-regular; the paper says random gives better graphs than empty | Empty | Read `link()`; toy step 1 |
| 2 | Insertion order | Random permutation | Id order, blocks of 2,048 across threads | Read; MIRROR 300/300 |
| 3 | Passes | Two (α = 1, then α) | One | Read; MIRROR 300/300 |
| 4 | α schedule | One α per pass | Rounds inside each prune at 1, 1.2, 1.44, … ≤ α. α = 1.1 behaves as 1.0, 1.3 as 1.2, 2.0 as 1.728 | Ran: identical graphs at 1.0/1.1 and at 1.2/1.3 |
| 5 | Distance in the α test | `d(p,q) = ‖x_p − x_q‖` (Section 1.2) | Squared L2 | Ran: toy step 8 |
| 6 | Tie in the α test | Remove when `α·d(p*,p') ≤ d(p,p')` | Skip only when the ratio is strictly greater than α | Read |
| 7 | Candidate set for a point's own prune | `V ∪ N_out(p)` | `V` only; `set_neighbours` overwrites what back-edges had put there | Read; toy step 5 |
| 8 | Back-edge overflow | Prune when `|N_out(j) ∪ {p}| > R` | Prune when the list already has `floor(1.3R)` entries; cleanup pass at the end | Read; MIRROR 300/300 |
| 9 | Entry point | Medoid | Stored point nearest the centroid | Read; MIRROR 300/300 |
| 10 | Prune pool | All of `V` | Nearest 750 of `V` | Read |
| 11 | Consolidation when the merged candidate set is small | Always RobustPrune | `R` or fewer candidates: keep all, no prune | Read |
| 12 | Disk index and updates | LTI + TempIndexes + StreamingMerge + redo log | Disk index is read-only; none of the rest exists | Searched the source |
| 13 | α values studied for updates | 1, 1.1, 1.2, 1.3 (FreshDiskANN Figure 3) | 1.1 is indistinguishable from 1, and 1.3 from 1.2, by item 4 | Ran (static build) |
| 14 | Record and sector | Vector and list share one 4 KB sector | True only while the record is at most 4,096 B; otherwise whole sectors per record | Read; index header read back |

Library defects met along the way, not paper differences:

- `StaticDiskIndex(num_threads=1)` deadlocks while loading in 0.7.0: `use_medoids_data_as_centroids`
  borrows the only per-thread scratch and `read_nodes` then waits for another. Observed here (all
  threads in `futex_wait`, zero bytes read); fixed upstream later in commits `fd93a04a` and `6f2691c7`.
  The scale script loads with 4 threads and searches with 1.
- The Python docstring says `cache_mechanism=2` readies the cache without filling it; the C++ behind
  it fills it by BFS from the medoid (`python/src/static_disk_index.cpp:23-31`).
- `Index::search` returns a hop count that is never incremented.

## Toy: eight 1-D vectors, `R = 2`, `L = 8`

Output of `lab/diskann_toy.py`. The LIBRARY blocks are neighbour lists parsed from the `ann` file
`diskannpy.build_memory_index` wrote. The narration is MIRROR, asserted equal to the library's graph
before it is printed. Distances in the narration are plain `|a − b|`; ratios are of squared distances.

```text
diskannpy 0.7.0, R=2, L=8, l2, num_threads=1

LIBRARY graph, alpha=1.0: start node E(8), 15 directed edges
  A(1) -> [E, B]
  B(2) -> [A, C]
  C(4) -> [B, D]
  D(7) -> [E, C]
  E(8) -> [D, F]
  F(11) -> [E, G]
  G(14) -> [F, H]
  H(15) -> [G]
  MIRROR reproduces this graph exactly (start node and ordered lists).

LIBRARY graph, alpha=1.1: start node E(8), 15 directed edges
  A(1) -> [E, B]
  B(2) -> [A, C]
  C(4) -> [B, D]
  D(7) -> [E, C]
  E(8) -> [D, F]
  F(11) -> [E, G]
  G(14) -> [F, H]
  H(15) -> [G]
  MIRROR reproduces this graph exactly (start node and ordered lists).

LIBRARY graph, alpha=1.2: start node E(8), 16 directed edges
  A(1) -> [E, B]
  B(2) -> [A, C]
  C(4) -> [B, D]
  D(7) -> [E, C]
  E(8) -> [D, F]
  F(11) -> [E, G]
  G(14) -> [F, H]
  H(15) -> [G, B]
  MIRROR reproduces this graph exactly (start node and ordered lists).

LIBRARY graph, alpha=1.3: start node E(8), 16 directed edges
  A(1) -> [E, B]
  B(2) -> [A, C]
  C(4) -> [B, D]
  D(7) -> [E, C]
  E(8) -> [D, F]
  F(11) -> [E, G]
  G(14) -> [F, H]
  H(15) -> [G, B]
  MIRROR reproduces this graph exactly (start node and ordered lists).

alpha=1.0 vs alpha=1.1: library graphs identical = True (rounds run at 1, 1.2, 1.44, ...; values in between round down)

alpha=1.2 vs alpha=1.3: library graphs identical = True (rounds run at 1, 1.2, 1.44, ...; values in between round down)

alpha=1.0 vs alpha=1.2 (library):
  only at 1.2: ['H->B']
  only at 1.0: []
  threshold on true distance: a farther same-side candidate at b survives a kept neighbour at a when
    code  (squared L2): b^2/(b-a)^2 <= 1.2  <=>  b >= 11.48 a   (sqrt(1.2) = 1.0954)
    paper (L2):         alpha (b-a) >  b       <=>  b >  6.00 a

================ MIRROR narration, alpha=1.0 (graph checked against the library) ================

1. insert A(1)
   search: expand E (d=7), out-list [], new: [] -> list [E7*]  (* = expanded)
   visited set, sorted by distance to A: [E 7]
   prune round alpha=1.00: keep E
   A -> [E]
   back-edge E -> A: room, appended; E -> [A]
   graph: A>E  E>A

2. insert B(2)
   search: expand E (d=6), out-list [A], new: [A] -> list [A1 E6*]  (* = expanded)
   search: expand A (d=1), out-list [E], new: [] -> list [A1* E6*]  (* = expanded)
   visited set, sorted by distance to B: [A 1, E 6]
   prune round alpha=1.00: keep A
   prune round alpha=1.00:   E vs kept A: d(B,E)^2 / d(A,E)^2 = 36/49 = 0.7347  -> not behind
   prune round alpha=1.00: keep E (ratio 0.7347 <= 1.00)
   B -> [A, E]
   back-edge A -> B: room, appended; A -> [E, B]
   back-edge E -> B: room, appended; E -> [A, B]
   graph: A>EB  B>AE  E>AB

3. insert C(4)
   search: expand E (d=4), out-list [A, B], new: [A, B] -> list [B2 A3 E4*]  (* = expanded)
   search: expand B (d=2), out-list [A, E], new: [] -> list [B2* A3 E4*]  (* = expanded)
   search: expand A (d=3), out-list [E, B], new: [] -> list [B2* A3* E4*]  (* = expanded)
   visited set, sorted by distance to C: [B 2, A 3, E 4]
   prune round alpha=1.00: keep B
   prune round alpha=1.00:   A vs kept B: d(C,A)^2 / d(B,A)^2 = 9/1 = 9.0000  -> behind B
   prune round alpha=1.00:   E vs kept B: d(C,E)^2 / d(B,E)^2 = 16/36 = 0.4444  -> not behind
   prune round alpha=1.00: keep E (ratio 0.4444 <= 1.00)
   C -> [B, E]
   back-edge B -> C: list full [A, E], re-prune [A 1, C 2, E 6] from B
      round alpha=1.00: keep A
      round alpha=1.00:   C vs kept A: 4/9 = 0.4444  -> not behind
      round alpha=1.00:   E vs kept A: 36/49 = 0.7347  -> not behind
      round alpha=1.00: keep C (ratio 0.4444 <= 1.00)
      round alpha=1.00:   E vs kept C: 36/16 = 2.2500  -> behind C
      B -> [A, C]
   back-edge E -> C: list full [A, B], re-prune [C 4, B 6, A 7] from E
      round alpha=1.00: keep C
      round alpha=1.00:   B vs kept C: 36/4 = 9.0000  -> behind C
      round alpha=1.00:   A vs kept C: 49/9 = 5.4444  -> behind C
      E -> [C]
   graph: A>EB  B>AC  C>BE  E>C

4. insert D(7)
   search: expand E (d=1), out-list [C], new: [C] -> list [E1* C3]  (* = expanded)
   search: expand C (d=3), out-list [B, E], new: [B] -> list [E1* C3* B5]  (* = expanded)
   search: expand B (d=5), out-list [A, C], new: [A] -> list [E1* C3* B5* A6]  (* = expanded)
   search: expand A (d=6), out-list [E, B], new: [] -> list [E1* C3* B5* A6*]  (* = expanded)
   visited set, sorted by distance to D: [E 1, C 3, B 5, A 6]
   prune round alpha=1.00: keep E
   prune round alpha=1.00:   C vs kept E: d(D,C)^2 / d(E,C)^2 = 9/16 = 0.5625  -> not behind
   prune round alpha=1.00:   B vs kept E: d(D,B)^2 / d(E,B)^2 = 25/36 = 0.6944  -> not behind
   prune round alpha=1.00:   A vs kept E: d(D,A)^2 / d(E,A)^2 = 36/49 = 0.7347  -> not behind
   prune round alpha=1.00: keep C (ratio 0.5625 <= 1.00)
   prune round alpha=1.00:   B vs kept C: d(D,B)^2 / d(C,B)^2 = 25/4 = 6.2500  -> behind C
   prune round alpha=1.00:   A vs kept C: d(D,A)^2 / d(C,A)^2 = 36/9 = 4.0000  -> behind C
   D -> [E, C]
   back-edge E -> D: room, appended; E -> [C, D]
   back-edge C -> D: list full [B, E], re-prune [B 2, D 3, E 4] from C
      round alpha=1.00: keep B
      round alpha=1.00:   D vs kept B: 9/25 = 0.3600  -> not behind
      round alpha=1.00:   E vs kept B: 16/36 = 0.4444  -> not behind
      round alpha=1.00: keep D (ratio 0.3600 <= 1.00)
      round alpha=1.00:   E vs kept D: 16/1 = 16.0000  -> behind D
      C -> [B, D]
   graph: A>EB  B>AC  C>BD  D>EC  E>CD

5. insert E(8)
   search: expand E (d=0), out-list [C, D], new: [C, D] -> list [E0* D1 C4]  (* = expanded)
   search: expand D (d=1), out-list [E, C], new: [] -> list [E0* D1* C4]  (* = expanded)
   search: expand C (d=4), out-list [B, D], new: [B] -> list [E0* D1* C4* B6]  (* = expanded)
   search: expand B (d=6), out-list [A, C], new: [A] -> list [E0* D1* C4* B6* A7]  (* = expanded)
   search: expand A (d=7), out-list [E, B], new: [] -> list [E0* D1* C4* B6* A7*]  (* = expanded)
   visited set, sorted by distance to E: [D 1, C 4, B 6, A 7]
   prune round alpha=1.00: keep D
   prune round alpha=1.00:   C vs kept D: d(E,C)^2 / d(D,C)^2 = 16/9 = 1.7778  -> behind D
   prune round alpha=1.00:   B vs kept D: d(E,B)^2 / d(D,B)^2 = 36/25 = 1.4400  -> behind D
   prune round alpha=1.00:   A vs kept D: d(E,A)^2 / d(D,A)^2 = 49/36 = 1.3611  -> behind D
   E -> [D]
   graph: A>EB  B>AC  C>BD  D>EC  E>D

6. insert F(11)
   search: expand E (d=3), out-list [D], new: [D] -> list [E3* D4]  (* = expanded)
   search: expand D (d=4), out-list [E, C], new: [C] -> list [E3* D4* C7]  (* = expanded)
   search: expand C (d=7), out-list [B, D], new: [B] -> list [E3* D4* C7* B9]  (* = expanded)
   search: expand B (d=9), out-list [A, C], new: [A] -> list [E3* D4* C7* B9* A10]  (* = expanded)
   search: expand A (d=10), out-list [E, B], new: [] -> list [E3* D4* C7* B9* A10*]  (* = expanded)
   visited set, sorted by distance to F: [E 3, D 4, C 7, B 9, A 10]
   prune round alpha=1.00: keep E
   prune round alpha=1.00:   D vs kept E: d(F,D)^2 / d(E,D)^2 = 16/1 = 16.0000  -> behind E
   prune round alpha=1.00:   C vs kept E: d(F,C)^2 / d(E,C)^2 = 49/16 = 3.0625  -> behind E
   prune round alpha=1.00:   B vs kept E: d(F,B)^2 / d(E,B)^2 = 81/36 = 2.2500  -> behind E
   prune round alpha=1.00:   A vs kept E: d(F,A)^2 / d(E,A)^2 = 100/49 = 2.0408  -> behind E
   F -> [E]
   back-edge E -> F: room, appended; E -> [D, F]
   graph: A>EB  B>AC  C>BD  D>EC  E>DF  F>E

7. insert G(14)
   search: expand E (d=6), out-list [D, F], new: [D, F] -> list [F3 E6* D7]  (* = expanded)
   search: expand F (d=3), out-list [E], new: [] -> list [F3* E6* D7]  (* = expanded)
   search: expand D (d=7), out-list [E, C], new: [C] -> list [F3* E6* D7* C10]  (* = expanded)
   search: expand C (d=10), out-list [B, D], new: [B] -> list [F3* E6* D7* C10* B12]  (* = expanded)
   search: expand B (d=12), out-list [A, C], new: [A] -> list [F3* E6* D7* C10* B12* A13]  (* = expanded)
   search: expand A (d=13), out-list [E, B], new: [] -> list [F3* E6* D7* C10* B12* A13*]  (* = expanded)
   visited set, sorted by distance to G: [F 3, E 6, D 7, C 10, B 12, A 13]
   prune round alpha=1.00: keep F
   prune round alpha=1.00:   E vs kept F: d(G,E)^2 / d(F,E)^2 = 36/9 = 4.0000  -> behind F
   prune round alpha=1.00:   D vs kept F: d(G,D)^2 / d(F,D)^2 = 49/16 = 3.0625  -> behind F
   prune round alpha=1.00:   C vs kept F: d(G,C)^2 / d(F,C)^2 = 100/49 = 2.0408  -> behind F
   prune round alpha=1.00:   B vs kept F: d(G,B)^2 / d(F,B)^2 = 144/81 = 1.7778  -> behind F
   prune round alpha=1.00:   A vs kept F: d(G,A)^2 / d(F,A)^2 = 169/100 = 1.6900  -> behind F
   G -> [F]
   back-edge F -> G: room, appended; F -> [E, G]
   graph: A>EB  B>AC  C>BD  D>EC  E>DF  F>EG  G>F

8. insert H(15)
   search: expand E (d=7), out-list [D, F], new: [D, F] -> list [F4 E7* D8]  (* = expanded)
   search: expand F (d=4), out-list [E, G], new: [G] -> list [G1 F4* E7* D8]  (* = expanded)
   search: expand G (d=1), out-list [F], new: [] -> list [G1* F4* E7* D8]  (* = expanded)
   search: expand D (d=8), out-list [E, C], new: [C] -> list [G1* F4* E7* D8* C11]  (* = expanded)
   search: expand C (d=11), out-list [B, D], new: [B] -> list [G1* F4* E7* D8* C11* B13]  (* = expanded)
   search: expand B (d=13), out-list [A, C], new: [A] -> list [G1* F4* E7* D8* C11* B13* A14]  (* = expanded)
   search: expand A (d=14), out-list [E, B], new: [] -> list [G1* F4* E7* D8* C11* B13* A14*]  (* = expanded)
   visited set, sorted by distance to H: [G 1, F 4, E 7, D 8, C 11, B 13, A 14]
   prune round alpha=1.00: keep G
   prune round alpha=1.00:   F vs kept G: d(H,F)^2 / d(G,F)^2 = 16/9 = 1.7778  -> behind G
   prune round alpha=1.00:   E vs kept G: d(H,E)^2 / d(G,E)^2 = 49/36 = 1.3611  -> behind G
   prune round alpha=1.00:   D vs kept G: d(H,D)^2 / d(G,D)^2 = 64/49 = 1.3061  -> behind G
   prune round alpha=1.00:   C vs kept G: d(H,C)^2 / d(G,C)^2 = 121/100 = 1.2100  -> behind G
   prune round alpha=1.00:   B vs kept G: d(H,B)^2 / d(G,B)^2 = 169/144 = 1.1736  -> behind G
   prune round alpha=1.00:   A vs kept G: d(H,A)^2 / d(G,A)^2 = 196/169 = 1.1598  -> behind G
   H -> [G]
   back-edge G -> H: room, appended; G -> [F, H]
   graph: A>EB  B>AC  C>BD  D>EC  E>DF  F>EG  G>FH  H>G

Cleanup pass: no list is longer than R, nothing to do

================ MIRROR narration, alpha=1.2 (graph checked against the library) ================

1. insert A(1)
   search: expand E (d=7), out-list [], new: [] -> list [E7*]  (* = expanded)
   visited set, sorted by distance to A: [E 7]
   prune round alpha=1.00: keep E
   A -> [E]
   back-edge E -> A: room, appended; E -> [A]
   graph: A>E  E>A

2. insert B(2)
   search: expand E (d=6), out-list [A], new: [A] -> list [A1 E6*]  (* = expanded)
   search: expand A (d=1), out-list [E], new: [] -> list [A1* E6*]  (* = expanded)
   visited set, sorted by distance to B: [A 1, E 6]
   prune round alpha=1.00: keep A
   prune round alpha=1.00:   E vs kept A: d(B,E)^2 / d(A,E)^2 = 36/49 = 0.7347  -> not behind
   prune round alpha=1.00: keep E (ratio 0.7347 <= 1.00)
   B -> [A, E]
   back-edge A -> B: room, appended; A -> [E, B]
   back-edge E -> B: room, appended; E -> [A, B]
   graph: A>EB  B>AE  E>AB

3. insert C(4)
   search: expand E (d=4), out-list [A, B], new: [A, B] -> list [B2 A3 E4*]  (* = expanded)
   search: expand B (d=2), out-list [A, E], new: [] -> list [B2* A3 E4*]  (* = expanded)
   search: expand A (d=3), out-list [E, B], new: [] -> list [B2* A3* E4*]  (* = expanded)
   visited set, sorted by distance to C: [B 2, A 3, E 4]
   prune round alpha=1.00: keep B
   prune round alpha=1.00:   A vs kept B: d(C,A)^2 / d(B,A)^2 = 9/1 = 9.0000  -> behind B
   prune round alpha=1.00:   E vs kept B: d(C,E)^2 / d(B,E)^2 = 16/36 = 0.4444  -> not behind
   prune round alpha=1.00: keep E (ratio 0.4444 <= 1.00)
   C -> [B, E]
   back-edge B -> C: list full [A, E], re-prune [A 1, C 2, E 6] from B
      round alpha=1.00: keep A
      round alpha=1.00:   C vs kept A: 4/9 = 0.4444  -> not behind
      round alpha=1.00:   E vs kept A: 36/49 = 0.7347  -> not behind
      round alpha=1.00: keep C (ratio 0.4444 <= 1.00)
      round alpha=1.00:   E vs kept C: 36/16 = 2.2500  -> behind C
      B -> [A, C]
   back-edge E -> C: list full [A, B], re-prune [C 4, B 6, A 7] from E
      round alpha=1.00: keep C
      round alpha=1.00:   B vs kept C: 36/4 = 9.0000  -> behind C
      round alpha=1.00:   A vs kept C: 49/9 = 5.4444  -> behind C
      E -> [C]
   graph: A>EB  B>AC  C>BE  E>C

4. insert D(7)
   search: expand E (d=1), out-list [C], new: [C] -> list [E1* C3]  (* = expanded)
   search: expand C (d=3), out-list [B, E], new: [B] -> list [E1* C3* B5]  (* = expanded)
   search: expand B (d=5), out-list [A, C], new: [A] -> list [E1* C3* B5* A6]  (* = expanded)
   search: expand A (d=6), out-list [E, B], new: [] -> list [E1* C3* B5* A6*]  (* = expanded)
   visited set, sorted by distance to D: [E 1, C 3, B 5, A 6]
   prune round alpha=1.00: keep E
   prune round alpha=1.00:   C vs kept E: d(D,C)^2 / d(E,C)^2 = 9/16 = 0.5625  -> not behind
   prune round alpha=1.00:   B vs kept E: d(D,B)^2 / d(E,B)^2 = 25/36 = 0.6944  -> not behind
   prune round alpha=1.00:   A vs kept E: d(D,A)^2 / d(E,A)^2 = 36/49 = 0.7347  -> not behind
   prune round alpha=1.00: keep C (ratio 0.5625 <= 1.00)
   prune round alpha=1.00:   B vs kept C: d(D,B)^2 / d(C,B)^2 = 25/4 = 6.2500  -> behind C
   prune round alpha=1.00:   A vs kept C: d(D,A)^2 / d(C,A)^2 = 36/9 = 4.0000  -> behind C
   D -> [E, C]
   back-edge E -> D: room, appended; E -> [C, D]
   back-edge C -> D: list full [B, E], re-prune [B 2, D 3, E 4] from C
      round alpha=1.00: keep B
      round alpha=1.00:   D vs kept B: 9/25 = 0.3600  -> not behind
      round alpha=1.00:   E vs kept B: 16/36 = 0.4444  -> not behind
      round alpha=1.00: keep D (ratio 0.3600 <= 1.00)
      round alpha=1.00:   E vs kept D: 16/1 = 16.0000  -> behind D
      C -> [B, D]
   graph: A>EB  B>AC  C>BD  D>EC  E>CD

5. insert E(8)
   search: expand E (d=0), out-list [C, D], new: [C, D] -> list [E0* D1 C4]  (* = expanded)
   search: expand D (d=1), out-list [E, C], new: [] -> list [E0* D1* C4]  (* = expanded)
   search: expand C (d=4), out-list [B, D], new: [B] -> list [E0* D1* C4* B6]  (* = expanded)
   search: expand B (d=6), out-list [A, C], new: [A] -> list [E0* D1* C4* B6* A7]  (* = expanded)
   search: expand A (d=7), out-list [E, B], new: [] -> list [E0* D1* C4* B6* A7*]  (* = expanded)
   visited set, sorted by distance to E: [D 1, C 4, B 6, A 7]
   prune round alpha=1.00: keep D
   prune round alpha=1.00:   C vs kept D: d(E,C)^2 / d(D,C)^2 = 16/9 = 1.7778  -> behind D
   prune round alpha=1.00:   B vs kept D: d(E,B)^2 / d(D,B)^2 = 36/25 = 1.4400  -> behind D
   prune round alpha=1.00:   A vs kept D: d(E,A)^2 / d(D,A)^2 = 49/36 = 1.3611  -> behind D
   E -> [D]
   graph: A>EB  B>AC  C>BD  D>EC  E>D

6. insert F(11)
   search: expand E (d=3), out-list [D], new: [D] -> list [E3* D4]  (* = expanded)
   search: expand D (d=4), out-list [E, C], new: [C] -> list [E3* D4* C7]  (* = expanded)
   search: expand C (d=7), out-list [B, D], new: [B] -> list [E3* D4* C7* B9]  (* = expanded)
   search: expand B (d=9), out-list [A, C], new: [A] -> list [E3* D4* C7* B9* A10]  (* = expanded)
   search: expand A (d=10), out-list [E, B], new: [] -> list [E3* D4* C7* B9* A10*]  (* = expanded)
   visited set, sorted by distance to F: [E 3, D 4, C 7, B 9, A 10]
   prune round alpha=1.00: keep E
   prune round alpha=1.00:   D vs kept E: d(F,D)^2 / d(E,D)^2 = 16/1 = 16.0000  -> behind E
   prune round alpha=1.00:   C vs kept E: d(F,C)^2 / d(E,C)^2 = 49/16 = 3.0625  -> behind E
   prune round alpha=1.00:   B vs kept E: d(F,B)^2 / d(E,B)^2 = 81/36 = 2.2500  -> behind E
   prune round alpha=1.00:   A vs kept E: d(F,A)^2 / d(E,A)^2 = 100/49 = 2.0408  -> behind E
   F -> [E]
   back-edge E -> F: room, appended; E -> [D, F]
   graph: A>EB  B>AC  C>BD  D>EC  E>DF  F>E

7. insert G(14)
   search: expand E (d=6), out-list [D, F], new: [D, F] -> list [F3 E6* D7]  (* = expanded)
   search: expand F (d=3), out-list [E], new: [] -> list [F3* E6* D7]  (* = expanded)
   search: expand D (d=7), out-list [E, C], new: [C] -> list [F3* E6* D7* C10]  (* = expanded)
   search: expand C (d=10), out-list [B, D], new: [B] -> list [F3* E6* D7* C10* B12]  (* = expanded)
   search: expand B (d=12), out-list [A, C], new: [A] -> list [F3* E6* D7* C10* B12* A13]  (* = expanded)
   search: expand A (d=13), out-list [E, B], new: [] -> list [F3* E6* D7* C10* B12* A13*]  (* = expanded)
   visited set, sorted by distance to G: [F 3, E 6, D 7, C 10, B 12, A 13]
   prune round alpha=1.00: keep F
   prune round alpha=1.00:   E vs kept F: d(G,E)^2 / d(F,E)^2 = 36/9 = 4.0000  -> behind F
   prune round alpha=1.00:   D vs kept F: d(G,D)^2 / d(F,D)^2 = 49/16 = 3.0625  -> behind F
   prune round alpha=1.00:   C vs kept F: d(G,C)^2 / d(F,C)^2 = 100/49 = 2.0408  -> behind F
   prune round alpha=1.00:   B vs kept F: d(G,B)^2 / d(F,B)^2 = 144/81 = 1.7778  -> behind F
   prune round alpha=1.00:   A vs kept F: d(G,A)^2 / d(F,A)^2 = 169/100 = 1.6900  -> behind F
   G -> [F]
   back-edge F -> G: room, appended; F -> [E, G]
   graph: A>EB  B>AC  C>BD  D>EC  E>DF  F>EG  G>F

8. insert H(15)
   search: expand E (d=7), out-list [D, F], new: [D, F] -> list [F4 E7* D8]  (* = expanded)
   search: expand F (d=4), out-list [E, G], new: [G] -> list [G1 F4* E7* D8]  (* = expanded)
   search: expand G (d=1), out-list [F], new: [] -> list [G1* F4* E7* D8]  (* = expanded)
   search: expand D (d=8), out-list [E, C], new: [C] -> list [G1* F4* E7* D8* C11]  (* = expanded)
   search: expand C (d=11), out-list [B, D], new: [B] -> list [G1* F4* E7* D8* C11* B13]  (* = expanded)
   search: expand B (d=13), out-list [A, C], new: [A] -> list [G1* F4* E7* D8* C11* B13* A14]  (* = expanded)
   search: expand A (d=14), out-list [E, B], new: [] -> list [G1* F4* E7* D8* C11* B13* A14*]  (* = expanded)
   visited set, sorted by distance to H: [G 1, F 4, E 7, D 8, C 11, B 13, A 14]
   prune round alpha=1.00: keep G
   prune round alpha=1.00:   F vs kept G: d(H,F)^2 / d(G,F)^2 = 16/9 = 1.7778  -> behind G
   prune round alpha=1.00:   E vs kept G: d(H,E)^2 / d(G,E)^2 = 49/36 = 1.3611  -> behind G
   prune round alpha=1.00:   D vs kept G: d(H,D)^2 / d(G,D)^2 = 64/49 = 1.3061  -> behind G
   prune round alpha=1.00:   C vs kept G: d(H,C)^2 / d(G,C)^2 = 121/100 = 1.2100  -> behind G
   prune round alpha=1.00:   B vs kept G: d(H,B)^2 / d(G,B)^2 = 169/144 = 1.1736  -> behind G
   prune round alpha=1.00:   A vs kept G: d(H,A)^2 / d(G,A)^2 = 196/169 = 1.1598  -> behind G
   prune round alpha=1.20: keep B (ratio 1.1736 <= 1.20)
   prune round alpha=1.20:   A vs kept B: d(H,A)^2 / d(B,A)^2 = 196/1 = 196.0000  -> behind B
   H -> [G, B]
   back-edge G -> H: room, appended; G -> [F, H]
   back-edge B -> H: list full [A, C], re-prune [A 1, C 2, H 13] from B
      round alpha=1.00: keep A
      round alpha=1.00:   C vs kept A: 4/9 = 0.4444  -> not behind
      round alpha=1.00:   H vs kept A: 169/196 = 0.8622  -> not behind
      round alpha=1.00: keep C (ratio 0.4444 <= 1.00)
      round alpha=1.00:   H vs kept C: 169/121 = 1.3967  -> behind C
      B -> [A, C]
   graph: A>EB  B>AC  C>BD  D>EC  E>DF  F>EG  G>FH  H>GB

Cleanup pass: no list is longer than R, nothing to do

================ query traces on the alpha=1.0 library graph ================

query 1.6, L=2, k=1: start at E(8)
   hop 1: expand E (d=6.4), out-list [D, F], unseen [D, F], admitted [D] -> list [D5.4 E6.4*]
   hop 2: expand D (d=5.4), out-list [E, C], unseen [C], admitted [C] -> list [C2.4 D5.4*]
   hop 3: expand C (d=2.4), out-list [B, D], unseen [B], admitted [B] -> list [B0.4 C2.4*]
   hop 4: expand B (d=0.4), out-list [A, C], unseen [A], admitted [A] -> list [B0.4* A0.6]
   hop 5: expand A (d=0.6), out-list [E, B], unseen [], admitted [] -> list [B0.4* A0.6*]
   stop: every entry in the list is expanded. 5 expansions, 6 distance computations of 8.
   answer: ['B']
   library answer: ['B'], distances [0.15999998152256012] (squared L2)  == MIRROR

query 14.4, L=2, k=1: start at E(8)
   hop 1: expand E (d=6.4), out-list [D, F], unseen [D, F], admitted [D, F] -> list [F3.4 E6.4*]
   hop 2: expand F (d=3.4), out-list [E, G], unseen [G], admitted [G] -> list [G0.4 F3.4*]
   hop 3: expand G (d=0.4), out-list [F, H], unseen [H], admitted [H] -> list [G0.4* H0.6]
   hop 4: expand H (d=0.6), out-list [G], unseen [], admitted [] -> list [G0.4* H0.6*]
   stop: every entry in the list is expanded. 4 expansions, 5 distance computations of 8.
   answer: ['G']
   library answer: ['G'], distances [0.15999969840049744] (squared L2)  == MIRROR

query 3.2, L=1, k=1: start at E(8)
   hop 1: expand E (d=4.8), out-list [D, F], unseen [D, F], admitted [D] -> list [D3.8]
   hop 2: expand D (d=3.8), out-list [E, C], unseen [C], admitted [C] -> list [C0.8]
   hop 3: expand C (d=0.8), out-list [B, D], unseen [B], admitted [] -> list [C0.8*]
   stop: every entry in the list is expanded. 3 expansions, 5 distance computations of 8.
   answer: ['C']
   library answer: ['C'], distances [0.6399999260902405] (squared L2)  == MIRROR

query 3.2, L=2, k=1: start at E(8)
   hop 1: expand E (d=4.8), out-list [D, F], unseen [D, F], admitted [D] -> list [D3.8 E4.8*]
   hop 2: expand D (d=3.8), out-list [E, C], unseen [C], admitted [C] -> list [C0.8 D3.8*]
   hop 3: expand C (d=0.8), out-list [B, D], unseen [B], admitted [B] -> list [C0.8* B1.2]
   hop 4: expand B (d=1.2), out-list [A, C], unseen [A], admitted [] -> list [C0.8* B1.2*]
   stop: every entry in the list is expanded. 4 expansions, 6 distance computations of 8.
   answer: ['C']
   library answer: ['C'], distances [0.6399999260902405] (squared L2)  == MIRROR

================ query traces on the alpha=1.2 library graph ================

query 1.6, L=2, k=1: start at E(8)
   hop 1: expand E (d=6.4), out-list [D, F], unseen [D, F], admitted [D] -> list [D5.4 E6.4*]
   hop 2: expand D (d=5.4), out-list [E, C], unseen [C], admitted [C] -> list [C2.4 D5.4*]
   hop 3: expand C (d=2.4), out-list [B, D], unseen [B], admitted [B] -> list [B0.4 C2.4*]
   hop 4: expand B (d=0.4), out-list [A, C], unseen [A], admitted [A] -> list [B0.4* A0.6]
   hop 5: expand A (d=0.6), out-list [E, B], unseen [], admitted [] -> list [B0.4* A0.6*]
   stop: every entry in the list is expanded. 5 expansions, 6 distance computations of 8.
   answer: ['B']
   library answer: ['B'], distances [0.15999998152256012] (squared L2)  == MIRROR

query 14.4, L=2, k=1: start at E(8)
   hop 1: expand E (d=6.4), out-list [D, F], unseen [D, F], admitted [D, F] -> list [F3.4 E6.4*]
   hop 2: expand F (d=3.4), out-list [E, G], unseen [G], admitted [G] -> list [G0.4 F3.4*]
   hop 3: expand G (d=0.4), out-list [F, H], unseen [H], admitted [H] -> list [G0.4* H0.6]
   hop 4: expand H (d=0.6), out-list [G, B], unseen [B], admitted [] -> list [G0.4* H0.6*]
   stop: every entry in the list is expanded. 4 expansions, 6 distance computations of 8.
   answer: ['G']
   library answer: ['G'], distances [0.15999969840049744] (squared L2)  == MIRROR

query 3.2, L=1, k=1: start at E(8)
   hop 1: expand E (d=4.8), out-list [D, F], unseen [D, F], admitted [D] -> list [D3.8]
   hop 2: expand D (d=3.8), out-list [E, C], unseen [C], admitted [C] -> list [C0.8]
   hop 3: expand C (d=0.8), out-list [B, D], unseen [B], admitted [] -> list [C0.8*]
   stop: every entry in the list is expanded. 3 expansions, 5 distance computations of 8.
   answer: ['C']
   library answer: ['C'], distances [0.6399999260902405] (squared L2)  == MIRROR

query 3.2, L=2, k=1: start at E(8)
   hop 1: expand E (d=4.8), out-list [D, F], unseen [D, F], admitted [D] -> list [D3.8 E4.8*]
   hop 2: expand D (d=3.8), out-list [E, C], unseen [C], admitted [C] -> list [C0.8 D3.8*]
   hop 3: expand C (d=0.8), out-list [B, D], unseen [B], admitted [B] -> list [C0.8* B1.2]
   hop 4: expand B (d=1.2), out-list [A, C], unseen [A], admitted [] -> list [C0.8* B1.2*]
   stop: every entry in the list is expanded. 4 expansions, 6 distance computations of 8.
   answer: ['C']
   library answer: ['C'], distances [0.6399999260902405] (squared L2)  == MIRROR
```

MIRROR validation (`python -I lab/diskann_toy.py --validate 300`): 300 random builds, 6 to 40 distinct
integer points in one or two dimensions, `R` in {2, 3, 4, 6, 8}, `L` in {R, R+1, 2R, 10, 50}, α in
{1.0, 1.1, 1.2, 1.3, 1.44, 1.5, 2.0}, one thread.

```text
validate: 300/300 random builds reproduced exactly by MIRROR
```

Other `R` at the same eight values, read from the library (probe run, not in the lab script):

| `R` | α | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 1.0 | E, B | A, E, C | B, E, D | E, C | D, F | E, G | F, H | G |
| 3 | 1.2 | E, B | A, C | B, E, D | E, C | D, F | E, G | F, H | G, B |
| 4 | 1.0 | E, B | A, E, C | B, E, D | E, C | D, F | E, G | F, H | G |
| 4 | 1.2 | E, B | A, E, C, H | B, E, D | E, C | D, F | E, G | F, H | G, B |

With `R ≥ 3`, B and C can keep their early edge to the entry point E as well (B loses it again at `R = 3`, α = 1.2).

## Scale test: in-memory Vamana and hnswlib

187,043 × 256-d, 50-topic Gaussian mixture (`rng(0)`, spread 0.35), 1,000 queries, recall@10 against
brute-force L2: the generator of `../ivf/README.md`. Single-thread ms per query, best of three passes.
*MIRROR expansions* counts neighbour lists read per query by walking the library's saved graph with
MIRROR's search over the first 200 queries; the last column confirms MIRROR's top-10 equals the
library's on all 200.

```text
N=187043 D=256 NQ=1000 K=10; brute-force L2 ground truth in 2.3 s; cpus=12

Vamana in-memory R=32 L_build=100 alpha=1.2: build 13.2 s (all cores), graph file 23.5 MiB = 132 B/node, degree mean 32.0 max 32 (max_observed 32), start node 144819
   L   recall@10   ms/query(1 thread)   MIRROR expansions/query   MIRROR distance comps/query   MIRROR top-10 == library
   10     0.4602        0.080               17.0                     484                      200/200
   20     0.6411        0.129               27.4                     727                      200/200
   30     0.7430        0.190               37.5                     934                      200/200
   50     0.8580        0.244               57.0                     1264                      200/200
   75     0.9181        0.316               82.0                     1600                      200/200
  100     0.9435        0.379               106.7                     1866                      200/200
  150     0.9657        0.476               156.5                     2272                      200/200
  200     0.9739        0.569               206.4                     2563                      200/200

Vamana in-memory R=64 L_build=100 alpha=1.2: build 26.0 s (all cores), graph file 45.6 MiB = 256 B/node, degree mean 62.9 max 64 (max_observed 64), start node 144819
   L   recall@10   ms/query(1 thread)   MIRROR expansions/query   MIRROR distance comps/query   MIRROR top-10 == library
   10     0.6595        0.153               15.7                     851                      200/200
   20     0.8157        0.228               25.1                     1220                      200/200
   30     0.8954        0.293               34.6                     1521                      200/200
   50     0.9569        0.387               54.2                     1987                      200/200
   75     0.9837        0.494               79.1                     2404                      200/200
  100     0.9917        0.589               104.0                     2698                      200/200
  150     0.9967        0.662               153.9                     3083                      200/200
  200     0.9988        0.799               203.9                     3318                      200/200

Vamana in-memory R=32 L_build=100 alpha=1.0: build 8.3 s (all cores), graph file 14.9 MiB = 84 B/node, degree mean 19.9 max 32 (max_observed 32), start node 144819
   L   recall@10   ms/query(1 thread)   MIRROR expansions/query   MIRROR distance comps/query   MIRROR top-10 == library
   10     0.4514        0.078               17.3                     452                      200/200
   20     0.6330        0.132               27.8                     686                      200/200
   30     0.7344        0.170               37.0                     866                      200/200
   50     0.8521        0.241               56.4                     1179                      200/200
   75     0.9200        0.312               81.4                     1501                      200/200
  100     0.9500        0.388               106.1                     1758                      200/200
  150     0.9756        0.483               155.8                     2149                      200/200
  200     0.9853        0.558               205.6                     2430                      200/200
```

```text
N=187043 D=256 NQ=1000 K=10; brute-force L2 ground truth in 2.2 s; cpus=12

hnswlib M=16 ef_construction=200: build 7.5 s (all cores), index file minus vectors 26.5 MiB = 149 B/node
  ef   recall@10   ms/query(1 thread)
   10     0.4486        0.062
   20     0.6138        0.096
   30     0.7110        0.132
   50     0.8270        0.184
   75     0.9030        0.244
  100     0.9398        0.290
  150     0.9725        0.344
  200     0.9845        0.405

hnswlib M=32 ef_construction=200: build 8.9 s (all cores), index file minus vectors 49.3 MiB = 276 B/node
  ef   recall@10   ms/query(1 thread)
   10     0.5596        0.094
   20     0.7345        0.146
   30     0.8282        0.185
   50     0.9230        0.262
   75     0.9664        0.326
  100     0.9842        0.409
  150     0.9952        0.507
  200     0.9976        0.546
```

Reading it:

- Expansions per query are `L` plus 4 to 8 in every configuration. α = 1.2 against 1.0 at `R = 32`
  changes them by under 1.
- α = 1.2 at `R = 32` saturates every list (mean degree 32.0); α = 1.0 leaves the mean at 19.9.
- hnswlib's file minus the raw vectors is 149 B per node at `M = 16`; Vamana at `R = 32` is 132 B.

## Scale test: the disk index

Same corpus. `build_disk_index(R = 64, L = 100)`, α at the library default of 1.2 (not settable from
Python), `search_memory_maximum` chosen so the library derives 32, 64, 128 or 256 PQ bytes per vector.
*4KB reads per query* is the growth of `read_bytes` in `/proc/self/io` across the 1,000-query batch,
divided by 4,096 and by 1,000; the index file is opened with `O_DIRECT`, so every node read reaches
the block layer. The millisecond column is on a VM's virtual disk and is indicative only.

```text
N=187043 D=256 NQ=1000 K=10; brute-force L2 ground truth in 2.0 s; cpus=12

DiskANN disk index R=64 L_build=100, 32 PQ bytes per vector in RAM: build 129.6 s (all cores)
   ann_disk.index                          255,381,504 B
   ann_mem.index.data                      191,532,040 B
   ann_metadata.bin                                 32 B
   ann_pq_compressed.bin                     5,985,384 B
   ann_pq_pivots.bin                           267,420 B
   ann_sample_data.bin                      19,189,768 B
   ann_sample_ids.bin                           74,968 B
   ann_vectors.bin                         191,532,040 B
   on-disk node = 1284 B, 3 nodes per 4,096 B sector; SSD file 243.6 MiB; PQ codes in RAM 5.71 MiB (32 B/vector); raw vectors 182.7 MiB
   cached nodes = 0
      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)
      10     2     0.0678        1.401               19.6
      20     2     0.1058        1.309               31.1
      30     2     0.1334        1.292               41.7
      50     2     0.1749        2.135               62.4
      75     2     0.2216        2.060               88.5
     100     2     0.2583        3.178               114.6
     200     2     0.3670        5.782               216.2
      50     1     0.1699        1.981               61.0
      50     4     0.1857        1.147               66.9
      50     8     0.2030        1.011               78.0
   cached nodes = 10000
      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)
      10     2     0.0678        0.679               12.6
      20     2     0.1058        0.775               22.8
      30     2     0.1334        1.017               32.3
      50     2     0.1749        1.486               51.3
      75     2     0.2216        2.221               75.5
     100     2     0.2583        2.599               99.9
     200     2     0.3670        5.509               195.6
      50     1     0.1699        2.096               50.9
      50     4     0.1857        1.090               53.0
      50     8     0.2030        0.882               57.0

DiskANN disk index R=64 L_build=100, 64 PQ bytes per vector in RAM: build 92.3 s (all cores)
   ann_disk.index                          255,381,504 B
   ann_mem.index.data                      191,532,040 B
   ann_metadata.bin                                 32 B
   ann_pq_compressed.bin                    11,970,760 B
   ann_pq_pivots.bin                           267,548 B
   ann_sample_data.bin                      19,015,688 B
   ann_sample_ids.bin                           74,288 B
   ann_vectors.bin                         191,532,040 B
   on-disk node = 1284 B, 3 nodes per 4,096 B sector; SSD file 243.6 MiB; PQ codes in RAM 11.42 MiB (64 B/vector); raw vectors 182.7 MiB
   cached nodes = 0
      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)
      10     2     0.2114        1.154               20.0
      20     2     0.3053        1.866               30.2
      30     2     0.3667        1.482               39.6
      50     2     0.4521        2.836               58.7
      75     2     0.5311        2.417               83.4
     100     2     0.5917        3.143               108.4
     200     2     0.7335        6.349               208.4
      50     1     0.4468        2.630               56.7
      50     4     0.4636        1.614               63.7
      50     8     0.4785        1.280               75.1
   cached nodes = 10000
      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)
      10     2     0.2114        0.417               13.2
      20     2     0.3053        0.672               22.6
      30     2     0.3667        0.825               31.3
      50     2     0.4521        1.262               49.0
      75     2     0.5311        2.207               71.9
     100     2     0.5917        2.771               95.2
     200     2     0.7335        5.179               188.2
      50     1     0.4468        2.050               48.6
      50     4     0.4636        1.158               50.4
      50     8     0.4785        0.915               53.9

DiskANN disk index R=64 L_build=100, 128 PQ bytes per vector in RAM: build 150.4 s (all cores)
   ann_disk.index                          255,381,504 B
   ann_mem.index.data                      191,532,040 B
   ann_metadata.bin                                 32 B
   ann_pq_compressed.bin                    23,941,512 B
   ann_pq_pivots.bin                           267,804 B
   ann_sample_data.bin                      19,036,168 B
   ann_sample_ids.bin                           74,368 B
   ann_vectors.bin                         191,532,040 B
   on-disk node = 1284 B, 3 nodes per 4,096 B sector; SSD file 243.6 MiB; PQ codes in RAM 22.83 MiB (128 B/vector); raw vectors 182.7 MiB
   cached nodes = 0
      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)
      10     2     0.5746        1.138               18.1
      20     2     0.7591        1.368               27.1
      30     2     0.8507        1.232               36.5
      50     2     0.9310        1.781               55.8
      75     2     0.9608        3.216               80.4
     100     2     0.9733        3.487               105.4
     200     2     0.9832        7.351               205.2
      50     1     0.9293        2.129               54.0
      50     4     0.9345        1.425               59.9
      50     8     0.9392        1.346               70.4
   cached nodes = 10000
      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)
      10     2     0.5746        0.433               11.8
      20     2     0.7591        0.794               20.1
      30     2     0.8507        1.096               28.6
      50     2     0.9310        1.719               46.2
      75     2     0.9608        2.438               68.7
     100     2     0.9733        2.898               91.5
     200     2     0.9832        6.285               183.2
      50     1     0.9293        2.164               46.0
      50     4     0.9345        1.249               46.8
      50     8     0.9392        1.080               49.3

DiskANN disk index R=64 L_build=100, 256 PQ bytes per vector in RAM: build 230.0 s (all cores)
   ann_disk.index                          255,381,504 B
   ann_mem.index.data                      191,532,040 B
   ann_metadata.bin                                 32 B
   ann_pq_compressed.bin                    47,883,016 B
   ann_pq_pivots.bin                           268,316 B
   ann_sample_data.bin                      19,275,784 B
   ann_sample_ids.bin                           75,304 B
   ann_vectors.bin                         191,532,040 B
   on-disk node = 1284 B, 3 nodes per 4,096 B sector; SSD file 243.6 MiB; PQ codes in RAM 45.66 MiB (256 B/vector); raw vectors 182.7 MiB
   cached nodes = 0
      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)
      10     2     0.6828        1.247               17.9
      20     2     0.8170        1.314               26.9
      30     2     0.8865        1.780               36.2
      50     2     0.9448        2.793               55.6
      75     2     0.9682        3.618               80.3
     100     2     0.9768        3.874               105.2
     200     2     0.9836        8.562               205.1
      50     1     0.9431        2.806               53.9
      50     4     0.9461        1.835               59.7
      50     8     0.9501        1.661               69.7
   cached nodes = 10000
      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)
      10     2     0.6828        0.555               11.7
      20     2     0.8170        0.984               19.8
      30     2     0.8865        1.258               28.3
      50     2     0.9448        1.968               46.0
      75     2     0.9682        2.946               68.5
     100     2     0.9768        3.721               91.3
     200     2     0.9836        7.497               182.9
      50     1     0.9431        2.272               45.9
      50     4     0.9461        1.531               46.5
      50     8     0.9501        1.506               48.7
```

Reading it:

- Reads per query are `L` plus 5 to 16 with no cache at `W = 2`. 10,000 cached nodes (5% of the
  graph) remove 6 to 22 of them.
- Recall is set by the PQ size, not by the graph: the same graph walked with exact distances in RAM
  reaches 0.992 at `L = 100`.
- 256 bytes on 256 dimensions is one byte per dimension, which is scalar quantization in all but name.
- Wider beams trade more reads for fewer round trips.

## Freshness test

`lab/diskann_fresh.py`: `DynamicMemoryIndex`, `R = 32`, `L = 100`, 100,000 live vectors from the same
generator, built by `batch_insert` into an empty index. Twenty cycles of: mark 5,000 random live
vectors deleted, `consolidate_delete()`, insert 5,000 vectors never seen before. Recall@10 against
brute force over the vectors live at that moment. Three delete schedules (`--seed 1`, `2`, `3`); within
a schedule the deletes and inserts are the same for both α.

Columns: mean recall@10 at five search `L`; *worst-topic* is the lowest per-topic recall@10 at
`L = 50`; the *pending deletes* block is measured between `mark_deleted` and `consolidate_delete`:
deleted ids returned at `L = 10` and at `L = 50`, and result rows at `L = 10` whose distances are not
ascending; then seconds for the consolidation and for the 5,000 inserts.

```text
# schedule 1 (--seed 1, the default)
diskannpy DynamicMemoryIndex freshness test: 20 cycles, D=256, NQ=1000, K=10, cpus=12

alpha=1.2  R=32 L_build=100  n=100000  5000 deleted + 5000 inserted per cycle (5%)  built by batch_insert in 5 s
 cycle  L=10    L=15    L=20    L=30    L=50    worst-topic(L=50) | pending deletes: dead@L10 dead@L50 unsorted-rows@L10 | consolidate-s insert-s
   0   0.6084  0.7173  0.7868  0.8730  0.9471     0.907
   1   0.6146  0.7129  0.7829  0.8674  0.9414     0.886            |              5        0        373           |       1.5        0.3
   2   0.6145  0.7152  0.7818  0.8660  0.9404     0.869            |              7        0        415           |       1.4        0.3
   3   0.6245  0.7222  0.7857  0.8654  0.9391     0.900            |              5        0        356           |       1.3        0.3
   4   0.6257  0.7221  0.7883  0.8677  0.9402     0.907            |              6        0        388           |       1.3        0.3
   5   0.6173  0.7199  0.7837  0.8689  0.9415     0.887            |              5        0        411           |       1.2        0.3
   6   0.6182  0.7199  0.7848  0.8668  0.9394     0.887            |             15        0        409           |       1.3        0.3
   7   0.6204  0.7209  0.7839  0.8682  0.9400     0.907            |              3        0        429           |       1.4        0.3
   8   0.6268  0.7204  0.7847  0.8681  0.9376     0.892            |              4        0        401           |       1.3        0.3
   9   0.6206  0.7155  0.7833  0.8707  0.9395     0.891            |              1        0        405           |       1.2        0.3
  10   0.5961  0.6978  0.7643  0.8487  0.9172     0.021            |              4        0        395           |       1.3        0.3
  11   0.6056  0.7005  0.7639  0.8452  0.9141     0.062            |             12        0        434           |       1.4        0.3
  12   0.6012  0.7001  0.7678  0.8518  0.9178     0.108            |              6        0        406           |       1.3        0.3
  13   0.6074  0.7002  0.7667  0.8473  0.9184     0.158            |              3        0        397           |       1.4        0.3
  14   0.6068  0.7004  0.7640  0.8473  0.9173     0.208            |              5        0        396           |       1.3        0.3
  15   0.5993  0.6958  0.7640  0.8430  0.9174     0.263            |              8        0        424           |       1.3        0.3
  16   0.6052  0.7058  0.7675  0.8491  0.9235     0.308            |              1        0        391           |       1.3        0.3
  17   0.6069  0.7014  0.7651  0.8480  0.9244     0.321            |              5        0        396           |       1.2        0.2
  18   0.6083  0.7023  0.7679  0.8507  0.9240     0.338            |              3        0        387           |       1.2        0.3
  19   0.6138  0.7041  0.7705  0.8540  0.9259     0.404            |              4        0        410           |       1.3        0.3
  20   0.6068  0.7024  0.7683  0.8524  0.9267     0.421            |              3        0        404           |       1.3        0.3
  change cycle 0 -> last: L=10: -0.0016  L=15: -0.0149  L=20: -0.0185  L=30: -0.0206  L=50: -0.0204

alpha=1.0  R=32 L_build=100  n=100000  5000 deleted + 5000 inserted per cycle (5%)  built by batch_insert in 2 s
 cycle  L=10    L=15    L=20    L=30    L=50    worst-topic(L=50) | pending deletes: dead@L10 dead@L50 unsorted-rows@L10 | consolidate-s insert-s
   0   0.5902  0.6930  0.7615  0.8514  0.9336     0.888
   1   0.5717  0.6693  0.7409  0.8316  0.9179     0.867            |              8        0        369           |       0.3        0.2
   2   0.5709  0.6636  0.7306  0.8250  0.9140     0.853            |              9        0        435           |       0.2        0.2
   3   0.5695  0.6613  0.7291  0.8197  0.9105     0.867            |             13        0        438           |       0.3        0.2
   4   0.5721  0.6662  0.7303  0.8176  0.9092     0.860            |              4        0        387           |       0.2        0.2
   5   0.5605  0.6541  0.7270  0.8175  0.9116     0.859            |              6        0        420           |       0.2        0.2
   6   0.5579  0.6546  0.7265  0.8148  0.9097     0.847            |              7        0        399           |       0.2        0.2
   7   0.5647  0.6600  0.7228  0.8136  0.9109     0.859            |             11        0        432           |       0.2        0.2
   8   0.5514  0.6493  0.7199  0.8134  0.9097     0.858            |              1        0        389           |       0.2        0.2
   9   0.5589  0.6536  0.7208  0.8175  0.9078     0.848            |              2        0        415           |       0.2        0.2
  10   0.5482  0.6470  0.7209  0.8135  0.9062     0.847            |              4        0        389           |       0.2        0.2
  11   0.5516  0.6459  0.7181  0.8120  0.9050     0.853            |              4        0        409           |       0.2        0.2
  12   0.5490  0.6512  0.7213  0.8143  0.9031     0.852            |              3        0        410           |       0.2        0.2
  13   0.5557  0.6550  0.7204  0.8143  0.9024     0.860            |              6        0        386           |       0.2        0.2
  14   0.5574  0.6508  0.7223  0.8086  0.9013     0.850            |              4        0        406           |       0.2        0.2
  15   0.5559  0.6500  0.7206  0.8071  0.8993     0.814            |              7        0        411           |       0.2        0.2
  16   0.5442  0.6414  0.7120  0.8057  0.9003     0.833            |              2        0        400           |       0.2        0.2
  17   0.5480  0.6505  0.7162  0.8067  0.9019     0.817            |              2        0        404           |       0.2        0.2
  18   0.5599  0.6509  0.7187  0.8138  0.9049     0.839            |              4        0        387           |       0.2        0.2
  19   0.5488  0.6479  0.7177  0.8138  0.9068     0.828            |              2        0        394           |       0.2        0.2
  20   0.5501  0.6433  0.7189  0.8118  0.9076     0.876            |              4        0        395           |       0.2        0.2
  change cycle 0 -> last: L=10: -0.0401  L=15: -0.0497  L=20: -0.0426  L=30: -0.0396  L=50: -0.0260

# schedule 2 (--seed 2)
diskannpy DynamicMemoryIndex freshness test: 20 cycles, D=256, NQ=1000, K=10, cpus=12

alpha=1.2  R=32 L_build=100  n=100000  5000 deleted + 5000 inserted per cycle (5%)  built by batch_insert in 5 s
 cycle  L=10    L=15    L=20    L=30    L=50    worst-topic(L=50) | pending deletes: dead@L10 dead@L50 unsorted-rows@L10 | consolidate-s insert-s
   0   0.6117  0.7200  0.7867  0.8719  0.9452     0.907
   1   0.6164  0.7158  0.7816  0.8640  0.9389     0.888            |              7        0        394           |       1.5        0.3
   2   0.6130  0.7128  0.7789  0.8633  0.9389     0.900            |             14        0        420           |       1.3        0.3
   3   0.6223  0.7163  0.7812  0.8651  0.9363     0.900            |              8        0        379           |       1.3        0.3
   4   0.6215  0.7170  0.7832  0.8673  0.9367     0.904            |              2        0        415           |       1.3        0.3
   5   0.6206  0.7211  0.7867  0.8709  0.9384     0.892            |             18        0        410           |       1.4        0.3
   6   0.6274  0.7196  0.7868  0.8693  0.9366     0.892            |             35        0        403           |       1.4        0.3
   7   0.6222  0.7164  0.7807  0.8675  0.9372     0.888            |              2        0        371           |       1.6        0.4
   8   0.6219  0.7138  0.7822  0.8677  0.9366     0.888            |              7        0        434           |       1.3        0.3
   9   0.6260  0.7198  0.7869  0.8690  0.9386     0.904            |             11        0        457           |       1.3        0.3
  10   0.6309  0.7262  0.7932  0.8680  0.9387     0.896            |              1        0        446           |       1.4        0.3
  11   0.6320  0.7283  0.7897  0.8684  0.9385     0.891            |             28        0        378           |       1.3        0.3
  12   0.6252  0.7258  0.7902  0.8690  0.9382     0.891            |              4        0        385           |       1.3        0.3
  13   0.6251  0.7261  0.7882  0.8687  0.9406     0.874            |              9        0        402           |       1.3        0.3
  14   0.6109  0.7144  0.7796  0.8599  0.9333     0.843            |              3        0        403           |       1.4        0.3
  15   0.6231  0.7198  0.7897  0.8660  0.9349     0.874            |              4        0        420           |       1.3        0.3
  16   0.6252  0.7194  0.7862  0.8633  0.9337     0.887            |              3        0        402           |       1.3        0.3
  17   0.6260  0.7191  0.7867  0.8655  0.9387     0.887            |              2        0        408           |       1.3        0.3
  18   0.6163  0.7112  0.7761  0.8626  0.9391     0.879            |              4        0        393           |       1.4        0.3
  19   0.6295  0.7231  0.7835  0.8663  0.9403     0.891            |              4        0        386           |       1.2        0.3
  20   0.6080  0.7068  0.7806  0.8624  0.9374     0.887            |              5        0        408           |       1.3        0.3
  change cycle 0 -> last: L=10: -0.0037  L=15: -0.0132  L=20: -0.0061  L=30: -0.0095  L=50: -0.0078

alpha=1.0  R=32 L_build=100  n=100000  5000 deleted + 5000 inserted per cycle (5%)  built by batch_insert in 3 s
 cycle  L=10    L=15    L=20    L=30    L=50    worst-topic(L=50) | pending deletes: dead@L10 dead@L50 unsorted-rows@L10 | consolidate-s insert-s
   0   0.5912  0.6936  0.7623  0.8515  0.9333     0.888
   1   0.5690  0.6659  0.7326  0.8216  0.9112     0.660            |             17        0        399           |       0.3        0.2
   2   0.5702  0.6662  0.7353  0.8268  0.9151     0.868            |              3        0        424           |       0.2        0.2
   3   0.5664  0.6645  0.7293  0.8203  0.9112     0.853            |              3        0        390           |       0.2        0.2
   4   0.5697  0.6632  0.7275  0.8210  0.9096     0.856            |              4        0        421           |       0.2        0.2
   5   0.5640  0.6647  0.7280  0.8229  0.9101     0.870            |             11        0        377           |       0.2        0.2
   6   0.5597  0.6518  0.7182  0.8142  0.9066     0.855            |              7        0        408           |       0.2        0.2
   7   0.5678  0.6616  0.7301  0.8243  0.9106     0.850            |              0        0        396           |       0.2        0.2
   8   0.5660  0.6633  0.7340  0.8246  0.9082     0.845            |              2        0        417           |       0.2        0.2
   9   0.5646  0.6621  0.7314  0.8245  0.9116     0.840            |              8        0         54           |       0.2        0.2
  10   0.5596  0.6571  0.7230  0.8175  0.9078     0.860            |              2        0        440           |       0.2        0.2
  11   0.5624  0.6602  0.7317  0.8222  0.9069     0.865            |              6        0        285           |       0.2        0.2
  12   0.5606  0.6571  0.7286  0.8210  0.9071     0.852            |              3        0        356           |       0.2        0.2
  13   0.5575  0.6574  0.7237  0.8137  0.9027     0.800            |              6        0        411           |       0.2        0.2
  14   0.5461  0.6482  0.7228  0.8118  0.9011     0.807            |              9        0        409           |       0.2        0.2
  15   0.5555  0.6481  0.7226  0.8176  0.9044     0.835            |              4        0        409           |       0.2        0.2
  16   0.5474  0.6533  0.7220  0.8148  0.9035     0.853            |              1        0        423           |       0.2        0.2
  17   0.5435  0.6463  0.7205  0.8126  0.9052     0.852            |              5        0        405           |       0.2        0.2
  18   0.5430  0.6423  0.7117  0.8051  0.9037     0.846            |              9        0        405           |       0.2        0.2
  19   0.5548  0.6489  0.7165  0.8110  0.9040     0.853            |              3        0        351           |       0.2        0.2
  20   0.5554  0.6520  0.7181  0.8099  0.9020     0.837            |              5        0        414           |       0.2        0.2
  change cycle 0 -> last: L=10: -0.0358  L=15: -0.0416  L=20: -0.0442  L=30: -0.0416  L=50: -0.0313

# schedule 3 (--seed 3)
diskannpy DynamicMemoryIndex freshness test: 20 cycles, D=256, NQ=1000, K=10, cpus=12

alpha=1.2  R=32 L_build=100  n=100000  5000 deleted + 5000 inserted per cycle (5%)  built by batch_insert in 5 s
 cycle  L=10    L=15    L=20    L=30    L=50    worst-topic(L=50) | pending deletes: dead@L10 dead@L50 unsorted-rows@L10 | consolidate-s insert-s
   0   0.6085  0.7152  0.7829  0.8706  0.9450     0.892
   1   0.6190  0.7203  0.7850  0.8667  0.9422     0.888            |              7        0        404           |       1.6        0.3
   2   0.6199  0.7199  0.7857  0.8667  0.9390     0.892            |              8        0        407           |       1.4        0.3
   3   0.6152  0.7211  0.7832  0.8674  0.9393     0.884            |              7        0        409           |       1.4        0.3
   4   0.6267  0.7238  0.7867  0.8655  0.9391     0.856            |             13        0        444           |       1.4        0.3
   5   0.6246  0.7233  0.7895  0.8705  0.9412     0.900            |             12        0        395           |       1.4        0.3
   6   0.6236  0.7180  0.7851  0.8672  0.9391     0.887            |              1        0        381           |       1.4        0.3
   7   0.6236  0.7195  0.7882  0.8656  0.9387     0.904            |             10        0        393           |       1.5        0.3
   8   0.6294  0.7208  0.7896  0.8663  0.9431     0.913            |              5        0        398           |       1.5        0.3
   9   0.6234  0.7213  0.7807  0.8641  0.9402     0.885            |              3        0        405           |       1.4        0.3
  10   0.6220  0.7209  0.7846  0.8654  0.9397     0.887            |              4        0        388           |       1.4        0.3
  11   0.6274  0.7180  0.7807  0.8650  0.9404     0.900            |              2        0        404           |       1.4        0.3
  12   0.6227  0.7192  0.7834  0.8655  0.9400     0.893            |              5        0        409           |       1.5        0.3
  13   0.6219  0.7181  0.7830  0.8655  0.9379     0.900            |              4        0        401           |       1.4        0.3
  14   0.6166  0.7107  0.7770  0.8629  0.9326     0.871            |              2        0        415           |       1.4        0.3
  15   0.6165  0.7128  0.7796  0.8699  0.9376     0.869            |             47        0        378           |       1.4        0.3
  16   0.6202  0.7193  0.7858  0.8683  0.9374     0.879            |              3        0        399           |       1.5        0.3
  17   0.6272  0.7231  0.7875  0.8652  0.9368     0.879            |              7        0        455           |       1.4        0.3
  18   0.6226  0.7217  0.7875  0.8674  0.9414     0.900            |              4        0        417           |       1.4        0.3
  19   0.6238  0.7256  0.7886  0.8678  0.9383     0.900            |              5        0        382           |       1.4        0.3
  20   0.6173  0.7200  0.7851  0.8657  0.9369     0.887            |              5        0        424           |       1.5        0.3
  change cycle 0 -> last: L=10: +0.0088  L=15: +0.0048  L=20: +0.0022  L=30: -0.0049  L=50: -0.0081

alpha=1.0  R=32 L_build=100  n=100000  5000 deleted + 5000 inserted per cycle (5%)  built by batch_insert in 3 s
 cycle  L=10    L=15    L=20    L=30    L=50    worst-topic(L=50) | pending deletes: dead@L10 dead@L50 unsorted-rows@L10 | consolidate-s insert-s
   0   0.5918  0.6965  0.7632  0.8518  0.9334     0.888
   1   0.5706  0.6662  0.7338  0.8290  0.9193     0.874            |             12        0        420           |       0.4        0.2
   2   0.5658  0.6652  0.7361  0.8242  0.9159     0.859            |              7        0        415           |       0.2        0.2
   3   0.5715  0.6652  0.7356  0.8243  0.9172     0.873            |              7        0        425           |       0.2        0.2
   4   0.5693  0.6679  0.7356  0.8247  0.9115     0.856            |             32        0        443           |       0.2        0.2
   5   0.5698  0.6609  0.7315  0.8251  0.9127     0.862            |              8        0        401           |       0.2        0.2
   6   0.5669  0.6653  0.7359  0.8247  0.9112     0.854            |              5        0        388           |       0.2        0.2
   7   0.5602  0.6622  0.7287  0.8195  0.9081     0.864            |              8        0        441           |       0.2        0.2
   8   0.5495  0.6531  0.7221  0.8142  0.9091     0.860            |              6        0        380           |       0.2        0.2
   9   0.5603  0.6513  0.7210  0.8133  0.9093     0.850            |              8        0        421           |       0.2        0.2
  10   0.5552  0.6543  0.7226  0.8150  0.9070     0.856            |              2        0        376           |       0.2        0.2
  11   0.5555  0.6519  0.7227  0.8167  0.9072     0.856            |              6        0        410           |       0.2        0.2
  12   0.5585  0.6520  0.7198  0.8140  0.9060     0.858            |              6        0        391           |       0.2        0.2
  13   0.5549  0.6531  0.7228  0.8121  0.9047     0.814            |              4        0        437           |       0.2        0.2
  14   0.5473  0.6433  0.7107  0.8063  0.8986     0.633            |             52        0        410           |       0.2        0.2
  15   0.5529  0.6493  0.7190  0.8122  0.9021     0.814            |              2        0        391           |       0.2        0.2
  16   0.5525  0.6516  0.7246  0.8129  0.9009     0.807            |              2        0        386           |       0.2        0.2
  17   0.5525  0.6489  0.7202  0.8082  0.9008     0.813            |             64        0        427           |       0.2        0.2
  18   0.5598  0.6542  0.7215  0.8144  0.9056     0.837            |             32        0        419           |       0.2        0.2
  19   0.5554  0.6524  0.7169  0.8137  0.9020     0.847            |              5        0        390           |       0.2        0.2
  20   0.5544  0.6533  0.7202  0.8132  0.9036     0.855            |             56        0        426           |       0.2        0.2
  change cycle 0 -> last: L=10: -0.0374  L=15: -0.0432  L=20: -0.0430  L=30: -0.0386  L=50: -0.0298
```

Reading it:

- **Change in recall@10 at `L = 50` over 20 cycles.** α = 1.2: −0.020, −0.008, −0.008. α = 1.0:
  −0.026, −0.031, −0.030. At `L = 20`: α = 1.2: −0.019, −0.006, +0.002; α = 1.0: −0.043, −0.044, −0.043.
- **Schedule 1 at α = 1.2 loses a topic.** At cycle 10 the worst topic's recall drops from 0.891 to
  0.021 in one cycle and climbs back by about 0.04 per cycle. The mean drops by 0.022, which is one
  topic in fifty. An earlier run of the same schedule, made before the diagnostics were added, shows
  the same step at the same cycle (0.9394 to 0.9176), so it follows from which vectors were deleted,
  not from thread timing.
- **α = 1.0 shows milder events of the same kind**: worst topic 0.660 at cycle 1 of schedule 2, and
  0.633 at cycle 14 of schedule 3.
- **Not investigated**: which deletion cut the topic off. One untested explanation: at α = 1.2 every
  list is already full of near neighbours, and `occlude_list` fills nearest first and stops at `R`,
  so when a node that bridged into another topic is deleted, the far candidates inherited from it are
  never reached.
- **Pending deletes.** `dead@L50` is 0 in all 120 cycles. At `L = 10 = k`, 401 of 1,000 rows on
  average have out-of-order distances (range 54 to 457), and 0 to 64 deleted ids appear. A ten-entry
  list contains at least one deleted node with probability 1 − 0.95¹⁰ = 0.40. `search_with_tags`
  writes only the live entries and returns how many it wrote; the binding allocates the output
  arrays uninitialised and ignores that count, so the unwritten tail holds whatever was in memory.
- **Slots are reused.** Capacity was 105,000 for 100,000 live vectors, and 20 × 5,000 inserts all
  succeeded, so consolidation released the deleted slots.
- **Cost on 12 emulated cores.** Consolidation about 1.3 s at α = 1.2 and 0.2 s at α = 1.0; 5,000
  inserts 0.2 to 0.3 s. Schedule 2 partly overlapped another job, so read timings from schedule 1.
- **Differences from the paper's experiment.** FreshDiskANN's Figures 2 and 3 start from a statically
  built index and re-insert the same points they deleted, so the corpus is unchanged from cycle to
  cycle. This test builds by insertion and inserts new points, so the corpus drifts.

## Reproduce

```bash
# 1. image (diskannpy has no macOS wheel; on Apple Silicon this runs under x86-64 emulation)
mkdir diskann-docker && cd diskann-docker
cat > Dockerfile <<'EOF'
FROM --platform=linux/amd64 python:3.11-slim
RUN apt-get update -qq && apt-get install -y -qq g++ libgomp1 > /dev/null && rm -rf /var/lib/apt/lists/*
RUN pip install -q --no-cache-dir diskannpy==0.7.0 "numpy<2" && pip install -q --no-cache-dir hnswlib==0.8.0
EOF
docker build --platform linux/amd64 -t diskann-lab .

# 2. run from modules/05-vector-databases-and-ann/
LAB="$PWD/lab"
docker volume create diskann-lab-work      # index files; a bind mount may refuse O_DIRECT
docker run --rm --platform linux/amd64 -v "$LAB":/lab:ro diskann-lab python -I /lab/diskann_toy.py
docker run --rm --platform linux/amd64 -v "$LAB":/lab:ro diskann-lab python -I /lab/diskann_toy.py --validate 300
docker run --rm --platform linux/amd64 -v "$LAB":/lab:ro -v diskann-lab-work:/work diskann-lab \
    python -I /lab/diskann_scale.py --work /work --stage all
docker run --rm --platform linux/amd64 -v "$LAB":/lab:ro diskann-lab python -I /lab/diskann_fresh.py
docker volume rm diskann-lab-work

# 3. source that was read
git clone https://github.com/microsoft/DiskANN.git && cd DiskANN && git checkout 0.7.0   # df225d3
```

Determinism: the toy and the validation are deterministic (one build thread). The scale and freshness
tests build with all cores, so graphs, and recall in the third decimal, vary between runs; PQ training
samples differ between builds as well.
