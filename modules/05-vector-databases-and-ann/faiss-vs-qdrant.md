# FAISS vs Qdrant: indexing and quantization

What each one actually does, as read from source and docs during M5. **FAISS** is a library (in-process,
no metadata, no server); **Qdrant** is an engine (server, payloads, filters, persistence).

- **Source pinned:** FAISS v1.16.0 (commit `6ccfd5f9`); Qdrant commit `95d89e5` (2026-11-13).
- **Grading:** everything here was read in code or docs unless marked *inferred* or *not run*.

## Indexing

| | FAISS | Qdrant |
|---|---|---|
| **Dense index families** | flat, IVF (`IVFFlat`, `IVFPQ`, `IVFScalarQuantizer`, `IVFRaBitQ`, fast-scan), HNSW (`HNSWFlat`, `HNSWSQ`, `HNSWPQ`, `HNSWRaBitQ`), IVF with an HNSW coarse quantizer (`IVF262144_HNSW32`) | **HNSW only**: *"Qdrant currently only uses HNSW as a dense vector index"*; plus a full scan the planner falls back to (`full_scan_threshold`) |
| **IVF** | yes, the reference implementation | none |
| **Why HNSW** | one option among many | docs: compatible with filtering during search, and among the most accurate and fastest in public benchmarks |
| **Sparse vectors** | not covered here | exact index the docs call *"similar to the inverted index"* |
| **Placement** | RAM by default; disk is opt-in (`OnDiskInvertedLists`, memory-mapped load) | HNSW config has `on_disk` and memory-tier options |

## Quantization: what each offers

| Technique | FAISS | Qdrant |
|---|---|---|
| **Scalar** | `QT_8bit`, `QT_4bit`, `QT_6bit`, `QT_8bit_uniform`, fp16 / bf16, direct | `int8` only |
| **Binary** | binary indexes (`IndexBinaryFlat`, …) over bit codes; in M5 the sign bits were made with numpy | 1-bit, 1.5-bit, 2-bit |
| **Product** | `IndexPQ`, `IndexIVFPQ`, fast-scan, OPQ, refine | `product` |
| **RaBitQ** | `IndexRaBitQ`, `IndexIVFRaBitQ`, `IndexHNSWRaBitQ`, 1–9 bits | none; borrows RaBitQ's bit-transposed query scoring for binary |
| **TurboQuant** | `QT_*_tq` types (2026) | `encoded_vectors_tq.rs`; listed in its docs as a fourth method (not studied in M5) |
| **Fixed-table (Lloyd–Max for N(0,1))** | `QT_*_eden`, `QT_*_tqmse` | none |

## Scalar quantization

| | FAISS `QT_8bit` | Qdrant `int8` |
|---|---|---|
| **Range** | min and max per dimension | one min and max over every value of every dimension |
| **Levels** | 256 (`step = range / 255`) | 128 (`step = range / 127`, clamped 0…127) |
| **Encode / decode** | `floor`, decode at `+0.5` (bin centre) | `round`, decode at `min + code × step` (no offset) |
| **Extremes** | the max decodes half a step high | the global max decodes exactly |
| **Query** | stays float | quantized with the same rule |
| **Score** | affine decode fused into the SIMD kernel | integer dot product of code arrays + precomputed correction terms |
| **Bytes per vector** | `d` | `d` padded to a multiple of 16, + 4 B (one float32 of correction terms); 1,540 B at 1,536-d |
| **Bounds options** | `RangeStat`: min/max (default), mean ± std, quantiles, `RS_optim` (broken in practice by float32 accumulation) | `quantile`: one pool from 5,000 sampled vectors; the cut is counted in vectors, so it trims far less than the docs' "1%" |
| **Toy (8 × 4-d)** | worst error 0.0125 | step 0.0567, 2.3–3.2× coarser per dim; worst error 0.0276 (*Python replication, not run*) |

## Binary quantization

| | FAISS | Qdrant |
|---|---|---|
| **Encoding** | you supply bit codes | 1-bit: `v > 0`; 2-bit: z-score per dimension, levels at ±2/3 σ; 1.5-bit: 2-bit with adjacent dims sharing the second bit |
| **Query** | bit codes, Hamming distance | `SameAsStorage` (default), `Scalar4bits`, `Scalar8bits` |
| **Rescore by default** | no built-in rescoring; add it yourself | on |

## Product quantization

| | FAISS `IndexPQ` | Qdrant `product` |
|---|---|---|
| **User chooses** | `m` (chunks) and `nbits` (2^`nbits` centroids per chunk) | `compression`: `x4`, `x8`, `x16`, `x32`, `x64` |
| **Chunk size** | `dsub = d / m`; refuses unless `d % m == 0` | `get_bucket_size`: 1, 2, 4, 8, 16 dims; the last chunk may be shorter |
| **Centroids per chunk** | 2^`nbits` | 256, hard-coded (`CENTROIDS_COUNT`) |
| **Training data** | `n < k` refused; `n = k` copies points; below 39 per centroid warns (a 2017 test artefact); above 256 per centroid subsamples (seeded, settable) | ≤ 256 vectors: the vectors become the centroids (lossless); more: k-means on a random 10,000-vector sample, ≤ 100 iterations |
| **Encode distance** | squared L2 | Euclidean, whatever the collection metric |
| **Decode** | `reconstruct` / `sa_decode` | none |
| **Score** | per-query table, float query (ADC) | per-query table using the collection's metric, float query |
| **Bytes per vector** | `m × nbits / 8` | `⌈d / chunk_size⌉`, nothing else |
| **Rescore by default** | via `IndexRefine` / `RFlat` when asked | off |

## Order: quantize before or after building the index?

| Index | Order | Structure built from |
|---|---|---|
| **FAISS IVF-PQ** | k-means bins → PQ on residuals → per vector: assign bin, encode residual | bins from originals; contents are codes |
| **FAISS HNSW-SQ / HNSW-PQ** | train quantizer → encode into storage → insert into graph | **codes** (`storage_distance_computer`) |
| **Qdrant HNSW** | quantize the segment → build the graph with the quantized vectors | **codes**; PQ and TurboQuant read each inserted point's original once as the query; GPU build with TurboQuant falls back to fp16 originals |
| *DiskANN, for contrast* | PQ first → Vamana graph | originals; PQ only steers search |

- **Qdrant has no switch** to build on originals while quantization is on; changing the quantization config discards the old graph.

## Search and rescoring

| | FAISS | Qdrant |
|---|---|---|
| **Oversample + rescore** | explicit: `IndexRefine` / `RFlat` with `k_factor` | `oversampling` × top from the quantized walk, then rescore with originals if `rescore` |
| **Rescore default** | off unless configured | scalar off, PQ off, binary on, TurboQuant depends on bits |
