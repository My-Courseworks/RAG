# Embedding under constraints — plan

> $\color{red}\text{to be discussed}$ — a draft plan, to revisit after a full read of the [M4 README](README.md). Nothing here is built or run.

## Goal
A **written plan**, not code yet, for a lab that tests the **embedding stage only**. It shows how parser choice and configuration, and embedding model choice, affect embedding under constraints we simulate on purpose. It's modelled on [practising-scale-under-constraints.md](../../practising-scale-under-constraints.md) and expands that file's EMB layer into a full study.

## Scope
- **In:** parsed nodes → embedding → vectors, plus what producing them costs and whether they came out right.
- **Out:**
  - Retrieval of any kind: no retriever, no top-k, no recall@k.
  - ANN index internals (M5), reranking, generation.
  - Parsing quality for its own sake. Parsers are varied only to see their effect on embedding.

## What varies
1. **Prose parsers, sensible configs vs pathological ones**
   - `SentenceSplitter(512, 50)` as the baseline.
   - `SentenceWindowNodeParser`: a sensible window against an oversized one. **The expected result is that embedding stays flat while storage grows.** That contrast is the lesson.
   - `HierarchicalNodeParser`:
     - **Sensible:** `[2048, 512, 128]`
     - **Too deep:** many levels and tiny leaves.
     - **Too shallow:** one or two levels, close to flat.
2. **Non-prose parsers:** markdown, JSON, code and HTML, each fed its own source type.
3. **Embedding models**
   - General-purpose vs specialised: code models for code, a legal model for carrier agreements where one exists.
   - API vs local.
   - Formats without a specialised model (JSON, markdown, HTML): the serialisation before embedding becomes the variable instead, for example a path or heading prefix.

## Simulated constraints
Using the scale file's kit (K1–K7):
- **The mock provider in front of a real or local model** injects 429s, latency, timeouts and per-request token caps, even though the real API would handle a laptop-sized load.
- Fault injectors, RAM and CPU caps, and quotas scaled down to match the smaller corpus.

## What gets measured
- **Integrity:**
  - Every node has exactly one vector.
  - Nodes truncated silently at the model's max input length.
  - Correct dimension, no NaN or zero vectors.
  - Determinism across retries.
  - Vectors still aligned with their nodes after async retries.
- **Resources:** embedded tokens, wall-clock time under the limits, retries, dollars at real prices, vector memory.
- **Optional intrinsic quality** (decision still open): pairwise gold-vs-distractor cosine, drift from a full-precision reference, near-duplicate share. No index, no ranking.

## Format
- The scale file's structure: shrink the system while holding the key ratios fixed, inject the constraint, make a napkin prediction, measure, explain the gap, name what breaks next.
- A prediction-log entry per experiment, and a mapping back to [year-one scale](../../syllabus.md#corpus-scale).
- The full experiment matrix listed in the plan, with each cell marked as baseline, one-factor-at-a-time, or a chosen interaction.

## Status and dependencies
- **Waits on a read and discussion of M4.** M4 decides which embedding knobs (dimensions, truncation, model choice, batch size) the lab is built around.
- **Open decision:** whether to include the intrinsic quality checks. Recommended yes, because the model comparison needs some quality signal.
- **Housekeeping:** the scale file's EMB-01 still cites M14's old 3.74B tokens and 62 h, now 3.61B and about 60 h.
