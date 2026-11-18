# Reconstruction error vs recall

Can a quantizer's reconstruction error, `‖x − x̂‖²` averaged over the corpus, decide which quantizer
to use? Sometimes it can compare two of them; it cannot replace recall.

## When reconstruction error is comparable

- **Same corpus,** measured on held-out vectors the quantizer was not trained on.
- **Same measure:** mean error per vector, not a raw sum, so corpus size drops out; relative error
  `‖x − x̂‖² / ‖x‖²` if comparing across datasets.
- **Same bytes per vector:** more bytes always means less error.
- **Both must reconstruct a vector:**
  - **Yes:** FAISS scalar and PQ (`reconstruct` / `sa_decode`); Qdrant scalar, implicitly
    (`min + step × code`).
  - **No:** binary (its "decode" is ±1 per dimension, no scale); Qdrant PQ (no decode, and its
    centroids are not exposed).
  - **Misleading:** FAISS RaBitQ. Its source warns the decode is *"heavily geared towards maintaining
    the IP, not L2"*.
  - **Degenerate:** Qdrant PQ on 256 vectors or fewer is lossless, so its error is 0.

## Measured: FAISS scalar vs Qdrant scalar, same corpus

187,043 × 256-d; ranges fit on 150,000 vectors, error measured on the other 37,043; recall@10 on 1,000
queries, each method scored its own way (FAISS: float query vs decoded codes; Qdrant: query quantized
too). Qdrant's rows replicate `encoded_vectors_u8.rs` in Python; Qdrant itself was not run.

| Corpus | Method | Error per vector | recall@10 | 4× oversample + rescore |
|---|---|---|---|---|
| 50-topic mixture | FAISS `QT_8bit` | 0.0164 | 0.956 | 1.000 |
| | Qdrant `int8` | 0.131 (8.0×) | 0.865 | 1.000 |
| embedding-like | FAISS `QT_8bit` | 0.0000182 | 0.982 | 1.000 |
| | Qdrant `int8` | 0.000372 (20.5×) | 0.911 | 1.000 |

- **Why the gap:**
  - **Half the levels:** 128 instead of 256 doubles the step; error grows with the step squared, so ~4×.
  - **One shared range:** every dimension gets the widest dimension's step. The low-variance
    dimensions of the embedding-like corpus suffer most, hence its extra ~5×.
- **Here error and recall agree:** same family, same idea, so the better reconstruction also ranks
  better.
- **Rescoring erases the gap:** both reach 1.000. Qdrant does not rescore scalar quantization by
  default; `rescore` must be turned on, and the originals kept somewhere.
- **The choice is still an engine choice:** FAISS's quantizer cannot run inside Qdrant.

## Measured: worse reconstruction, better recall

Embedding-like corpus, same setup:

| Method | Bytes | Reconstruction error | recall@10 |
|---|---|---|---|
| PQ32x8 | 32 B | **0.0545** | 0.253 |
| RaBitQ 1-bit, float query | 40 B | 0.1085 (2× worse) | **0.307** |

## Why the two can disagree

Recall asks only whether the true neighbours stay ahead of the rest. That depends on errors in the
**distances**, measured against the gaps between neighbours, not on how far each rebuilt vector is from
its original.

- **How the query is scored:** a float query (asymmetric) carries error on one side only; a quantized
  query (symmetric) carries it on both.
- **Which direction the error points:** error that leaves inner products alone does not change the
  ranking. ScaNN's anisotropic quantization accepts more total error to cut the part along each vector.
- **Shared errors cancel:** a shift common to every vector moves all distances together.
- **Where the error lands:** error in dimensions that do not separate close neighbours costs nothing.
- **What the estimator does:** RaBitQ estimates distances with a correction factor instead of
  decoding, so its rebuilt vector was never the point.

## Verdict

- **Reconstruction error:** a diagnostic, fair within one family at equal bytes.
- **Recall@10 against exact search,** same queries, same bytes per vector: the measure to decide on.
  It needs no decode and no access to internals, so it works across every quantizer and vendor.
- **For LogiTrack:** rerun the comparison on a sample of the real 1,536-d vectors; the gap depends on
  how unevenly variance spreads across dimensions.

## Reproduce

Scripts in [`lab/`](lab/), run from that folder with `python -I` (needs `numpy`, `faiss-cpu`):

- [`sq_faiss_vs_qdrant.py`](lab/sq_faiss_vs_qdrant.py): the FAISS vs Qdrant scalar table.
- [`recon_vs_recall.py`](lab/recon_vs_recall.py): the PQ vs RaBitQ table (imports the corpora from
  [`quant_compare.py`](lab/quant_compare.py)).
