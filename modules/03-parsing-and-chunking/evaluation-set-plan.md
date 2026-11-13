# An evaluation set that can actually tell chunkers apart — plan

> $\color{red}\text{to be discussed}$ — a plan, not yet run. It replaces the invalid quality columns
> in M3's worked example ([README](README.md#worked-example-two-chunkers-one-corpus-measured)) with
> numbers that can separate one chunker from another, on documents like the ones production sees.

## Remember these five things

1. **Label spans, not documents.** A chunker changes *which span* comes back, so only span-level
   labels can see the difference.
2. **Label every correct source.** If six tables hold the answer, all six are gold, not just one.
3. **The question must match its document.** A question about "clause 5.6, liability" needs a
   clause 5.6 about liability.
4. **Split the queries, not the corpus.** Tune on development queries; score the winner once on test
   queries.
5. **Use real, messy PDFs.** A markdown corpus with `#` headings hides every parsing problem.

## What's wrong with the current set

Checked on the seed-1337 corpus from `labs/generate_corpus.py`:

| Problem | Evidence |
|---|---|
| **Labels are per document** | `build_queries` writes only `gold_doc_ids`; no offsets |
| **One gold document among dozens of lookalikes** | Each carrier-agreement question's gold is one of about 63 agreements with the same carrier (46 to 78) |
| **Clause text carries no topic** | Clause bodies are random words from a 149-word vocabulary |
| **Questions don't match their gold document** | Only 24 of 80 gold documents contain the clause asked about; in 5 of 80 the topic matches that clause's heading |
| **Other correct answers count as misses** | 79 of 80 rate questions name a lane and container found in several tables (median 6), but only one is gold |
| **Too few queries** | The frozen 60 have only 24 that apply to the worked example; at n = 60 the 95% interval is about ±0.12 |
| **Too easy** | Markdown files with explicit headings: nothing scanned, no tables crossing pages, no figures |

## What a valid set needs

### Labels
- **Each question records every span that answers it:** `doc_id`, `version`, `start_char_idx`,
  `end_char_idx`, and the page and bounding box once the source is a PDF.
- **Mark which spans are required together** for a complete answer (a rate plus the rule that
  changes it, a policy plus its exception).
- **Unanswerable questions** have an empty gold set, and are checked to be truly unanswerable.

### Strata
Report every metric per stratum, never only in aggregate:
- **By source type:** carrier agreements, rate tables, manuals, and so on.
- **By region kind:** native prose, OCR'd prose, native table, scanned table, figure text.
- **By question type:**
  - lexical (exact identifiers);
  - semantic (paraphrase);
  - numeric lookups;
  - **a row of a table crossing pages, on a page without the header;**
  - **prose that refers to table rows;**
  - multi-hop across documents;
  - unanswerable.

### Size
- **For a ±0.10 interval at recall ≈ 0.7:** about 80 test questions per stratum
  (n ≈ 1.96² × 0.21 / 0.10²).
- **For ±0.05:** about 320 per stratum.
- **Compare configurations pairwise on the same questions** (a paired bootstrap, or McNemar's test).
  Paired tests need fewer questions than comparing two separate intervals.

### Splits
- **Development set:** compare configurations, tune chunk sizes and thresholds.
- **Test set:** frozen. Run once per decision, never tuned against.
- **Split by document,** so tuning can't learn one document's quirks from its development questions
  and be scored on the same document's test questions.
- **Stratify the split,** so each stratum has development and test questions.

## Two corpora

### A · A fixed generator (cheap, fast iteration, still synthetic)
- **Clause text that carries its topic:** a clause on liability talks about liability.
- **Questions generated from a real clause,** recording its offsets as the gold span.
- **Unique identifiers in questions** (agreement id, effective date, tariff id), or **every**
  matching span recorded as gold.
- **Rendered to PDF**, then made messy on purpose:
  - some pages rasterized and noised (scanned);
  - some pages part native, part image;
  - rate tables running many pages, with the header only on the first;
  - figures with captions.
- Every transformation is recorded, so each gold span still maps to page and bounding box.

### B · Real documents (what the decision has to rest on)
- **Contracts: CUAD,** 510 commercial contracts with expert span annotations across 41 clause
  categories. Real span-level ground truth. ⚠️ Confirm the licence (believed CC BY 4.0) before use.
- **Carrier tariffs:** published rate schedules and demurrage terms from carriers' websites, labelled
  by hand. ⚠️ Check each site's terms of use.
- **Scanned variants:** print and scan, or rasterize, a sample of both, so OCR'd regions are real.

### Labelling real documents
- **Written guidelines:** what counts as an answer span, and what counts as complete.
- **Two annotators per question;** disagreements are settled by a third.
- **Measure agreement** (span overlap F1 between annotators). Low agreement means the guidelines,
  not the annotators, need fixing.
- **Questions are written without seeing any chunker's output,** so no configuration is favoured.

## Metrics

| Metric | Definition | Why |
|---|---|---|
| **Span recall@k** | A gold span counts as found if the top-k retrieved text covers at least X% of it (X fixed before running, e.g. 80%) | What a chunker actually changes |
| **Complete-set recall@k** | Every span required together is found | Rates plus the rules that change them |
| **Document recall@k** | Secondary only | Comparable with older numbers |
| **Answer correctness** | Numeric: exact match against the gold value. Prose: an LLM judge against the gold spans, spot-checked by a person | The generator's actual answer |
| **Unanswerable handling** | The system declines instead of answering | Hallucination on empty gold sets |

## Checks that the set is valid, before trusting any result
- **Oracle check:** a retriever that returns the gold spans scores 1.0.
- **Random check:** random retrieval scores about what chance predicts.
- **Label audit:** re-label a random 10% blind; agreement must clear the bar set in the guidelines.
- **Sensitivity check:** two clearly different configurations (say 128- vs 2048-token chunks) must
  produce a difference larger than the interval. If they don't, the set can't separate chunkers.

## Running it

**Not now.** Running happens only after corpus A or B passes the validity checks.

| Item | Choice |
|---|---|
| Embedding | `text-embedding-3-small`, real API, the account's real tier |
| Configurations | The six rows of M3's worked example, plus the gate-free pipeline from M3's decision |
| Answer generation and judge | A real LLM, chosen at run time and recorded |
| Vector store | Qdrant server, as in [node-storage-measurement-plan.md](node-storage-measurement-plan.md) |
| Rate limits | Real backoff, logged; vectors stored so re-runs cost nothing |
| Cost | Corpus A at the pilot's size: a few dollars to embed six configurations. Corpus B is much smaller. The LLM judge is the larger cost, so estimate it from the question count before running. |

## Deliverables
- The corpus, its labels, the guidelines and the annotator agreement.
- Development and test splits, frozen, with checksums.
- A replacement for the worked example's quality columns: per stratum, with 95% intervals and
  paired comparisons.
- A short write-up per finding: what was predicted, what was measured, and which findings in M3
  survived.
