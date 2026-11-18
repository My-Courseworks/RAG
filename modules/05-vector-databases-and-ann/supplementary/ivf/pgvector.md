# pgvector `ivfflat` — IVF inside Postgres

- Flavour: **IVF-Flat only**. k-means centroids (one per list), full `float4` vectors stored uncompressed in the list pages, exact distance inside each probed list. No PQ/SQ/residual coding of any kind.
- Runs anywhere Postgres runs. Here: a **real** PostgreSQL 16.2 + pgvector **0.6.2** bundled by the `pgserver` wheel (no Docker needed on this Mac). Source read at HEAD (0.9.0-dev) — the ivfflat core is unchanged since 0.6.x except `ivfflat.iterative_scan` / `ivfflat.max_probes` (added 0.8.0), which the bundled 0.6.2 does **not** have, so those two are documented from source only, not run.
- Toy and scale numbers below are measured, not simulated. k-means seeding is not SQL-controllable, so the toy reports the distribution over 25 rebuilds.

## 1. Source read

Repo `https://github.com/pgvector/pgvector`, commit `10a73eceefe033ac944f91782c553b5fc67eccb9` (2026-11-13, `vector.control` default_version 0.8.7, CHANGELOG 0.9.0 unreleased). Clone: shallow, `src-pgvector/pgvector/`.

| File | Role |
|---|---|
| `src/ivfflat.h` | Constants: `IVFFLAT_DEFAULT_LISTS 100`, `MIN 1`, `MAX 32768`, `DEFAULT_PROBES 1` (L61-64); `IVFFLAT_MAX_DIM 2000` (L42); on-disk `IvfflatListData {startPage, insertPage, Vector center}` (L261-266). |
| `src/ivfflat.c` | Registers the `lists` reloption (L51-52) and the GUCs `ivfflat.probes`, `ivfflat.iterative_scan`, `ivfflat.max_probes` (L54-65); planner cost model `ratio = probes/lists` (L138). |
| `src/ivfbuild.c` | Build: ANALYZE-style block + reservoir sampling (`SampleRows`, L121-144), sample count rule (L437-446), "little data" NOTICE (L457-463), nearest-centroid assignment of every row into a tuplesort keyed by list id (`AddTupleToSort`, L150-210), list pages written in list order (`InsertTuples`, L263-322), parallel heap scan for the assign phase (L612+). |
| `src/ivfkmeans.c` | k-means++ init (L23-89), Elkan's accelerated Lloyd (L214-452; hard cap 500 iterations L314; stop on zero reassignments L449), empty cluster → random centre (L193-196), random centres when zero samples (L109-128), spherical variant via angular distance for ip/cosine (comment L206-211). |
| `src/ivfscan.c` | Query: `GetScanLists` keeps the `maxProbes` nearest centroids in a pairing heap then qsorts them (L66-139); `GetScanItems` reads `probes` lists per batch, pushes every (exact distance, tid) into a tuplesort (L182-253); `ivfflatbeginscan` resolves `probes`/`maxProbes` (L343-417); `ivfflatgettuple` pulls from the sort and, when exhausted and `listIndex < maxProbes`, fetches the next batch of lists — that is the whole iterative-scan mechanism (L487-493). |
| `src/ivfinsert.c` | Insert: scan the list pages, pick the nearest centroid (`FindInsertPage`, L20-66), append tuple to that list's `insertPage` (chaining a new page when full). No retrain, no split. |
| `src/ivfutils.c` | `maintenance_work_mem` check for the k-means arrays (L125-133); per-type max dims: `vector` 2000, `halfvec` 4000, `bit` 64000 (L419, L438, L454); `IvfflatCheckNorm` rejects zero-norm vectors under cosine (L93-96). |
| `src/ivfvacuum.c` | Bulk delete removes dead tuples from list pages; centroids are never recomputed. |
| `sql/vector.sql` | Opclasses: `vector_l2_ops` (scan proc `vector_l2_squared_distance`, k-means proc `l2_distance`), `vector_ip_ops` and `vector_cosine_ops` (scan proc `vector_negative_inner_product`, k-means proc `vector_spherical_distance` = `acos(a·b)/π`, k-means norm proc `vector_norm`; cosine additionally has a storage norm proc) (L406-425). Plus `halfvec_*_ops`, `bit_hamming_ops`. |
| `README.md` | Guidance: build *after* loading data; `lists ≈ rows/1000` up to 1M rows, `sqrt(rows)` above; `probes ≈ sqrt(lists)`; filtering is applied *after* the index scan; iterative scans (0.8.0+) (L334-342, L380-386, L424-550). |

## 2. The algorithm

1. **Sample.** `numSamples = max(lists × 50, 10000)`, capped at the table's max possible tuples (`ivfbuild.c` L437-446). Rows are picked with Postgres' ANALYZE block sampler + Vitter reservoir (L121-144), seeded from the backend PRNG. If `samples < lists` → `NOTICE: ivfflat index created with little data … This will cause low recall` (L457-463), but the build proceeds (empty clusters get random centres, `ivfkmeans.c` L193-196).
2. **Train.** k-means++ seeding (`InitCenters`), then Elkan's triangle-inequality k-means, ≤ 500 iterations, early exit when no point changes cluster. Distance for L2 ops is plain L2 (not squared); for ip/cosine ops it is angular `acos(a·b)/π` on L2-normalised samples and centroids re-normalised after every mean step (`ComputeNewCenters` → `NormCenters`) — i.e. spherical k-means. The Elkan lower-bound matrix is `samples × lists` floats and must fit in `maintenance_work_mem` (`ivfutils.c` L125-133). Zero samples (empty table) → uniform-random centres in [0,1)^d.
3. **Assign.** Full table scan (parallel workers allowed); each row goes to its single nearest centroid by the opclass distance (`AddTupleToSort` L173-183). Under `vector_cosine_ops` the stored copy is L2-normalised first and zero-norm rows are silently skipped (L161-168). No multi-assignment, no boundary replication.
4. **Store.** Metapage (block 0) → list pages (block 1…: `startPage`, `insertPage`, full `center` vector per list) → entry pages: one chain of 8 KB pages per list holding `(full vector, heap TID)` index tuples. Vectors are stored as-is (no residuals, no quantisation). Index size ≈ table vector size.
5. **Search.** Query is normalised for cosine ops; every centroid is scored (linear scan of the list pages, `GetScanLists`), the `probes` nearest lists are chosen; each tuple in those lists gets an exact distance via the opclass scan proc (`vector_l2_squared_distance` etc.) and is pushed into a `tuplesort` ordered by distance; Postgres then pulls tuples in order until `LIMIT` is satisfied. No early termination inside a list, no ADC tables, no rescoring step (nothing to rescore — vectors are exact).
6. **Iterative scan (0.8.0+, not in 0.6.2).** With `ivfflat.iterative_scan = relaxed_order`, `maxProbes = min(lists, max(ivfflat.max_probes, probes))`. The first batch scans `probes` lists; if the executor keeps asking for tuples after the sort is drained (because `WHERE` rejected most of them), the next `probes` lists are scanned and sorted as a fresh batch (`ivfflatgettuple` L487-493). Batches are not merged, hence "relaxed" order: a later batch may yield a closer vector than one already returned. There is no `strict_order` mode for ivfflat (only `off` / `relaxed_order`, `ivfflat.c` L38-42).
7. **Insert / delete.** Insert = nearest existing centroid, append (`ivfinsert.c`). Delete/VACUUM = remove tuples. Centroids never move after build → the README's "create the index after loading data" rule; `REINDEX` is the only retrain.

## 3. Hyperparameters

| Name | When | Default | Trades | FAISS analogue |
|---|---|---|---|---|
| `lists` (reloption, `WITH (lists = n)`) | build | 100 (min 1, max 32768) | more lists → smaller lists → faster probes, but more samples (`lists×50`), bigger Elkan matrix, more centroid scoring per query, lower recall at fixed `probes` | `nlist` |
| `ivfflat.probes` (GUC, `SET`/`SET LOCAL`) | query | 1 (min 1, max 32768; silently capped at `lists`) | recall vs latency; `= lists` is exact search | `nprobe` |
| `ivfflat.iterative_scan` (GUC, 0.8.0+) | query | `off` | `relaxed_order` keeps opening lists until `LIMIT` rows survive the `WHERE` filter; results may be slightly out of order | no analogue (FAISS has no post-filter loop) |
| `ivfflat.max_probes` (GUC, 0.8.0+) | query | 32768 | upper bound on lists opened by an iterative scan; `max(max_probes, probes)` is used, capped at `lists` | no analogue |
| opclass (`vector_l2_ops` / `vector_ip_ops` / `vector_cosine_ops` / `halfvec_*` / `bit_hamming_ops`) | build | `vector_l2_ops` is `DEFAULT` | metric; picks the k-means distance (L2 vs spherical) and whether stored vectors are normalised | `METRIC_L2` / `METRIC_INNER_PRODUCT` (cosine = normalise + IP) |
| `maintenance_work_mem` | build | 64 MB (Postgres) | must hold `samples × lists × 4 B` for Elkan; `lists=4096` on this dataset needs ≈ 3.1 GB or the build errors out | no analogue (FAISS trains in RAM unconditionally) |
| `max_parallel_maintenance_workers` | build | 2 | parallelises the assign/sort phase only; k-means is single-threaded | `omp_set_num_threads` (loosely) |
| `work_mem` | query | 4 MB | tuplesort of all scanned candidates spills to disk above this | no analogue |
| `enable_seqscan`, `LIMIT`, `ORDER BY … <-> q` | query | — | the planner may skip the index entirely when `probes/lists` is large or there is no `ORDER BY`; the index is never used without an `ORDER BY` on the distance operator | no analogue (FAISS has no planner) |

No `m` / `nbits`: there is no quantiser. Sample size (`lists × 50`, floor 10000), 500 iterations, k-means++ init and the random seed are all hard-coded; FAISS exposes them via `ClusteringParameters` (`niter=25`, `max_points_per_centroid=256`, `seed`).

## 4. Toy: A..H = 1, 2, 4, 7, 8, 11, 14, 15 as `vector(1)`, `lists = 3`

Script: `lab/ivf_pgvector_toy.py` (real Postgres; centroids parsed from index block 1 since pgvector has no SQL introspection; bins verified by `probes = 1` self-queries).

1. `CREATE TABLE toy (id text, v vector(1))`, 8 rows. pgvector allows `d = 1` for `vector_l2_ops` (ip/cosine require `d > 1`, `ivfbuild.c` L366-369).
2. `CREATE INDEX … USING ivfflat (v vector_l2_ops) WITH (lists = 3)`. Samples: `max(3×50, 10000)` capped at max tuples → all 8 rows. `lists = 10` on the same 8 rows emits `NOTICE: ivfflat index created with little data` (checked); `lists = 3` does not.
3. k-means++ + Elkan. The seed is Postgres' per-backend PRNG, not `setseed()`, so 25 rebuilds gave three different local optima:

   | Count /25 | Partition | Centroids (read from disk) |
   |---|---|---|
   | 14 | {1,2,4} {7,8,11} {14,15} | 2.333, 8.667, 14.5 |
   | 8 | {1,2,4} {7,8} {11,14,15} | 2.333, 7.5, 13.333 |
   | 3 | {1,2} {4,7,8} {11,14,15} | 1.5, 6.333, 13.333 |

   The rest of the walk-through uses the most common one.
4. Lists: `0 → {A,B,C} = {1,2,4}` centroid 2.333; `1 → {D,E,F} = {7,8,11}` centroid 8.667; `2 → {G,H} = {14,15}` centroid 14.5.
5. Query `q = 9.5`, top-3. Centroid distances: list 1 → 0.833, list 2 → 5.0, list 0 → 7.167. Exact answer: 8 (1.5), 11 (1.5), 7 (2.5).
   - `SET ivfflat.probes = 1` → opens list 1 → returns 8, 11, 7. **Exact.** 9.5 lies between 8 and 11 but both are in the same list.
   - `probes = 2` → lists 1, 2 → same answer.
6. Query `q = 5.5`, top-3. Centroid distances: list 0 → 3.167, list 1 → 3.167 (an exact tie), list 2 → 9.0. Exact answer: 4 (1.5), 7 (1.5), 8 (2.5).
   - `probes = 1` → the tie goes to list 0 (the heap only replaces on strictly smaller distance, `ivfscan.c` L115) → returns **4 (1.5), 2 (3.5), 1 (4.5)**. **Miss: 7 and 8**; recall@3 = 1/3. The nearest *centroid* is not the list holding the nearest *vectors*.
   - `probes = 2` → lists 0 and 1 → 7, 4, 8. Exact.
7. With the second-most-common partition ({7,8} is its own list, centroid 7.5), `q = 9.5` at `probes = 1` opens {7,8} (distance 2.0 vs 3.833 to 13.333) and misses 11. Which boundary bites depends on where k-means happened to land — the course lesson that the approximation error of IVF is a property of the partition, not of the query.

## 5. Scale: 187,043 × 256, 1000 queries, recall@10

Script: `lab/ivf_pgvector_scale.py`. `COPY … FORMAT BINARY` via `pgvector`'s psycopg adapter, `maintenance_work_mem = '8GB'`, `max_parallel_maintenance_workers = 4`, `enable_seqscan = off` so every probe setting really hits the index. Client-side timing, single connection, warm cache. Table: 191 MB of vectors.

| lists | probes | recall@10 | ms/query |
|---|---|---|---|
| 187 | 1 | 0.5575 | 0.43 |
| 187 | 2 | 0.8322 | 0.69 |
| 187 | 4 | 0.9930 | 1.07 |
| 187 | 8 | 1.0000 | 1.96 |
| 187 | 16 | 1.0000 | 3.54 |
| 187 | 32 | 1.0000 | 7.11 |
| 187 | 64 | 1.0000 | 16.73 |
| 187 | 128 | 1.0000 | 35.16 |
| 4096 | 1 | 0.0753 | 0.24 |
| 4096 | 2 | 0.1353 | 0.29 |
| 4096 | 4 | 0.2428 | 0.30 |
| 4096 | 8 | 0.4087 | 0.44 |
| 4096 | 16 | 0.6396 | 0.65 |
| 4096 | 32 | 0.8939 | 0.93 |
| 4096 | 64 | 0.9967 | 1.13 |
| 4096 | 128 | 1.0000 | 1.94 |
| 4096 | 256 | 1.0000 | 3.52 |
| 1024 | 1 | 0.1856 | 0.20 |
| 1024 | 2 | 0.3282 | 0.26 |
| 1024 | 4 | 0.5643 | 0.43 |
| 1024 | 8 | 0.8500 | 0.71 |
| 1024 | 16 | 0.9931 | 0.91 |
| 1024 | 32 | 1.0000 | 1.47 |
| 1024 | 64 | 1.0000 | 2.75 |
| 1024 | 128 | 1.0000 | 5.39 |
| 1024 | 256 | 1.0000 | 11.20 |

Builds (wall clock incl. sampling, k-means, parallel assign, page writes):

| lists | build | index size | note |
|---|---|---|---|
| 187 (README rule: rows/1000) | 1.8 s | 209.6 MiB | 10,000 samples |
| 4096 (brief's target) | 62.7 s | 228.2 MiB | `lists×50 = 204,800 > N` → every row is a sample; Elkan matrix 187,043 × 4096 floats ≈ 2.9 GB (fails with default `maintenance_work_mem = 64MB`) |
| 1024 | 4.3 s | 213.6 MiB | 51,200 samples |

- Smallest `probes` with recall ≥ 0.95: **lists=187 → probes=4** (0.993); **lists=4096 → probes=64** (0.9967); **lists=1024 → probes=16** (0.9931).
- Index size ≈ table size at every `lists` (flat storage; each tuple is the 1,032-byte vector + TID + header). Compare FAISS `IndexIVFFlat`: same bytes, pgvector additionally pays 8 KB page overhead.
- At the 0.95 recall point the three configurations cost about the same (1.07 / 0.91 / 1.13 ms): `probes × N/lists` ≈ 4,000 rows scanned in each case, so `lists` mostly moves the floor (scoring 4096 centroids ≈ 0.24 ms at `probes=1`) and the build cost (1.8 s vs 62.7 s), not the recall/latency frontier. Below the knee, `lists=4096` at the README's `probes ≈ sqrt(lists) = 64` lands right on 0.9967 — the rule of thumb holds here.
- Filtering (query 0, `WHERE topic = …` then `LIMIT 10`): filtering on the query's **own** topic returns 10 rows at every `probes`; filtering on a **different** topic returns **0 rows** at `lists=187, probes ≤ 4`, at `lists=1024, probes ≤ 16`, and at `lists=4096, probes ≤ 64` — the probed lists hold only the query's own cluster, and the `WHERE` is applied after the index yields TIDs. This is the post-filter starvation the README warns about; `ivfflat.iterative_scan = relaxed_order` (0.8.0+) is the fix. On the bundled 0.6.2 `SET ivfflat.iterative_scan` errors with `invalid configuration parameter name … "ivfflat" is a reserved prefix`, so it is documented from `ivfscan.c` only, not measured.
- The dataset has 50 well-separated topics, so any `lists ≥ 50` partitions mostly along topic lines and recall saturates fast; the `lists=187` curve reaching 1.0 at `probes=8` says more about the generator than about pgvector.

## 6. pgvector-specific vs plain FAISS IVF

- No quantiser at all: ivfflat is `IndexIVFFlat` and nothing else. Compression in pgvector means `halfvec` (2 B/dim) or `bit` + Hamming, chosen at the column/opclass level, not an IVF option.
- Training knobs are hidden: sample size `max(lists×50, 10000)`, k-means++ init, Elkan, 500 iterations, and the random seed are not user-settable; FAISS exposes all of them.
- Memory for training is gated by `maintenance_work_mem` and the Elkan `samples × lists` float matrix — `lists=4096` needs ~3 GB here; FAISS just allocates.
- Everything is MVCC/WAL: inserts land in the nearest existing list and are visible transactionally; `VACUUM` reclaims; no retrain until `REINDEX`. FAISS `add` is the same nearest-centroid rule but has no delete path without `IDSelector`.
- Search is a full `tuplesort` of all candidates in the probed lists (no heap of size k), the planner decides whether to use the index at all (`probes/lists` ratio in the cost model), and `WHERE` filters are applied after the index yields TIDs; `iterative_scan`/`max_probes` (0.8.0+) re-open more lists when a filter starves the `LIMIT`. FAISS has `nprobe` only; filtering is the caller's problem.
- Cosine = normalise-on-write + spherical k-means + inner-product scan; zero-norm vectors are silently excluded from a cosine index.

## 7. Reproduce

```bash
# Python 3.12 (pgserver 0.1.4 has no 3.13 wheel). pgserver bundles PostgreSQL 16.2 + pgvector 0.6.2.
python3.12 -m venv venv-ivf-pgvector
venv-ivf-pgvector/bin/pip install numpy pgserver 'psycopg[binary]' pgvector

cd modules/05-vector-databases-and-ann/lab
venv-ivf-pgvector/bin/python -I ivf_pgvector_toy.py --builds 25          # ~1 min
venv-ivf-pgvector/bin/python -I ivf_pgvector_scale.py --lists 187 4096 1024  # see build times above
```

Both scripts create a throwaway Postgres data directory under `$TMPDIR` (override with `--pgdata`). To run the `iterative_scan` demo you need pgvector ≥ 0.8.0, e.g. `docker run -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg17`, and to point the script at it (not done here; no Docker on this machine).
