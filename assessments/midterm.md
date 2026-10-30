# Mid-term assessment — M1 through M11

> **Covers:** M1 (why RAG exists) · M2 (pipeline and the F1–F8 taxonomy) · M3 (parsing and chunking) ·
> M4 (embeddings) · M5 (vector databases and ANN) · M6 (metadata, filters, multi-tenancy) ·
> M7 (sparse, dense, hybrid) · M8 (index structures beyond flat vectors) · M9 (ranking, fusion,
> reranking) · M10 (query understanding) · M11 (context assembly, prompting, temperature).
> **Format:** closed-book, 3 hours, written. Paper and pen or a plain text editor — no retrieval
> system, no documentation, no search. Every question is answerable from reasoning and arithmetic
> you can do at the desk.
> **Marking:** 100 points total. Section A 40 (10 × 4) · Section B 36 (4 × 9) · Section C 24 (3 × 8).
> **What the paper tests:** whether you can *choose* and *localize*. Recall of definitions earns
> nothing. A correct answer names a mechanism, a number, and a condition under which you would
> change your mind.
>
> **The domain spine.** All problems use LogiTrack, the freight forwarder and customs broker from
> M1: a 3,000-document pilot corpus (~13.5M words) growing to 480,000 documents, 8,500 queries/day,
> P95 ≤ 2.5 s, ≤ $18,000/month (≈ 7.06¢ per query), recall@10 ≥ 0.85, index freshness ≤ 15 minutes,
> 40 jurisdictions, 12 clients. Its corpus spans carrier service agreements (PDF), customs manuals
> (PDF/DOCX per jurisdiction), tariff rate tables (XLSX), dangerous-goods rules (cross-referenced
> PDF), service bulletins (dated email/HTML), and tickets plus an ops wiki (short markdown). The
> 60-query evaluation set from M1 is frozen and stratified into nine strata: clause lookup, numeric
> tariff, jurisdiction/date, lexical identifier, multi-hop cross-reference, freshness-sensitive,
> multi-document aggregate, unanswerable, adversarial.

---

## Section A — conceptual reasoning (10 questions × 4 points)

Answer in 150–250 words each. Each question expects a mechanism, not a definition.

**A1.** LogiTrack needs per-client access control on retrieved evidence, a 15-minute freshness bound,
and a citation for every claim. Explain why these three requirements jointly make retrieval the only
viable mechanism, and then state the one thing fine-tuning would still legitimately contribute to
this system.
*Rubric (4):* 1 for naming that weights cannot be partitioned per client; 1 for naming freshness as
an update-latency property of a store rather than of a model; 1 for naming provenance as a property
of the retrieved span; 1 for a correct fine-tuning complement (behaviour, format, routing) rather
than a restatement of retrieval.

**A2.** A colleague argues that long-context models make LogiTrack's retrieval pipeline unnecessary.
State the *condition* under which they would be right for a specific query class, the two
measurements that would tell you the condition holds, and the reason the condition cannot hold for
the corpus as a whole.
*Rubric (4):* 1 for the condition being document-scoped and fitting the window; 1 for cost-per-query
and latency as the qualifying measurements; 1 for citing effective-capacity degradation at lengths
below the technical maximum; 1 for the corpus-scale argument (a window holds a single-digit
percentage of ~18.0M tokens, and 160× less of the target corpus).

**A3.** For each symptom, name the failure ID and the owning pipeline stage. (a) The correct node is
retrieved at rank 8 and the answer used rank 1. (b) The answer is a half-sentence whose other half
sits in the node ranked 9. (c) The retrieved evidence is right, and the answer quotes a number from
it that does not appear in the context. (d) A cited agreement was revised last week and the answer
reflects the previous revision.
*Rubric (4):* 1 per item — F2/rank (rerank stage), F3/chunk (parse–chunk), F5 or F6 (assemble vs
generate, with the distinction stated), F7/ingest.

**A4.** LogiTrack's carrier agreements total 9M words. You must choose `chunk_size` and
`chunk_overlap` for them. State which of retrieval precision, retrieval recall, and answer
completeness you are sacrificing, compute the token-volume multiplier for your chosen overlap, and
state the cost of your choice on a full re-embed at an illustrative $0.02 per 1M tokens.
*Rubric (4):* 1 for an explicit named sacrifice; 1 for correct use of 1/(1 − O/C); 1 for arithmetic
consistent with stated corpus size and overlap; 1 for connecting the re-embed cost to a model-change
event rather than to steady-state spend.

**A5.** You must choose between two embedding models for LogiTrack: a 768-dimension model with
strong general benchmarks, and a 1024-dimension model with a documented weakness on rare tokens.
State your decision, the metric you will decide it on (be specific about the stratum), the storage
implication of your choice at 3.1M nodes, and the condition that would reverse it.
*Rubric (4):* 1 for a decision stated in the imperative; 1 for a stratum-specific metric (the lexical
identifier stratum, not aggregate recall); 1 for storage arithmetic at 3.1M nodes; 1 for a reversal
condition that is a measurement.

**A6.** LogiTrack's filters are 40 jurisdictions × 12 clients × document date. Explain the
cardinality trap for filtered ANN search, then decide between pre-filtering and post-filtering for a
query scoped to *one* client and *one* jurisdiction. Justify with the failure mode your choice
creates.
*Rubric (4):* 1 for explaining that a highly selective pre-filter can leave too few candidates to
search (or destroy recall); 1 for explaining that post-filtering can return fewer than k results
after ranking; 1 for a decision consistent with the selectivity described; 1 for naming the resulting
failure ID (F1 for over-filtering; F1 or F2 for under-filtering).

**A7.** Explain why the queries "clause 14.2(b)" and "UN3480" defeat dense-only retrieval. Then
choose between reciprocal rank fusion and weighted score fusion for combining BM25 with dense
retrieval, and justify your choice in terms of the scores being combined.
*Rubric (4):* 1 for the mechanism (rare identifiers carry little distributional semantics; the
embedding of an identifier is close to many neighbours); 1 for identifying score incomparability as
the problem; 1 for a choice that does not require calibrated comparable scores (RRF) or a stated
calibration step; 1 for naming a tuning plan for any weight used.

**A8.** For each query class, name the index structure you would use and one thing you concede by
using it: (a) "What is the demurrage window in our Maersk agreement?" (b) "Summarize how DG rules
changed across all 40 jurisdictions this year." (c) "Which carrier agreement governs the annex
referenced in §4.1 of the DG rules?"
*Rubric (4):* 1 for (a) flat vector over structure-aware chunks, conceding nothing structural but
paying index memory; 1 for (b) a summary- or community-summarization index, conceding staleness and
extraction cost; 1 for (c) a graph or recursive-retrieval path, conceding ingest-time extraction
cost and node count; 1 for a consistent statement of what is conceded in each case.

**A9.** LogiTrack retrieves 40 candidates for a 4,000-token answer budget and must cut to 6. Justify
the candidate-set size of 40, name the artifact you are buying with a reranker, and state the
measurement that tells you whether the reranker earned its latency.
*Rubric (4):* 1 for the recall@k → precision@k split (the retriever's job is to not lose the answer);
1 for candidate size as a recall/cost tradeoff rather than a quality knob; 1 for nDCG@k or the rank of
the first relevant node as the measurement; 1 for stating the latency paid against the 2.5 s budget.

**A10.** Two LogiTrack tasks: (i) extract a surcharge figure verbatim from a rate table; (ii)
synthesize a one-paragraph summary of DG exposure across 12 carriers. Choose a temperature for each,
explain why a temperature of 0 is not determinism, and describe the evaluation protocol that would
let you re-derive your choices on your own system rather than trusting them.
*Rubric (4):* 1 for ~0 for extraction and ~0.2–0.4 for synthesis, with a reason tied to the sampling
distribution; 1 for the non-determinism explanation (provider nondeterminism, batching, seed
availability); 1 for a sweep protocol with repeated runs; 1 for naming answer variance and
faithfulness as the measured quantities.

---

## Section B — design and decision problems (4 problems × 9 points)

Produce the artefact named. These are marked on the decision, the price, and the reversal condition —
in that order.

**B1 (9). The architecture ADR — RAG vs fine-tuning vs long-context.**
Write the decision for LogiTrack's document Q&A: a one-sentence imperative decision; the mechanism by
which your choice works; the rejected alternatives with reasons that quantify (not "too expensive");
the per-query allocation of the 7.06¢ ceiling across query embedding, search, rerank and generation;
and the metric-and-threshold reversal condition.
*Strong answer:* states which stratum the chosen architecture *cannot* serve (the multi-document
aggregate stratum, by construction); gives an allocation summing to the ceiling; names the excluded
architecture and the cost line that excludes it; gives a reversal condition with a metric, a
threshold, and a duration.
*Weak answer:* lists pros and cons; rejects fine-tuning because "it is expensive"; no allocation; a
reversal condition that is an opinion ("if quality is poor").

**B2 (9). The chunking ADR.**
For LogiTrack's carrier agreements (9M words, clause-numbered PDFs) and rate tables (XLSX), state the
answer unit for each source, choose a parser class and justify it by the structure present after
extraction, set `chunk_size`/`chunk_overlap` with the token-volume arithmetic, and state the
acceptance test and revert condition.
*Strong answer:* treats the two sources as different problems (structure recovery for the tables, not
a chunk-size parameter); shows 1/(1 − O/C) and the resulting node count; states which of precision /
recall / completeness is conceded; gives a revert threshold on a named metric at a named k.
*Weak answer:* chooses one chunk size for both sources; asserts overlap is "good practice"; no
arithmetic; no revert condition.

**B3 (9). The metadata schema and isolation model.**
Design the metadata schema that makes LogiTrack's filtered retrieval work across 40 jurisdictions and
12 clients, choose an isolation strategy (shared index + filter, namespace, collection, or
DB-per-tenant), and state the specific leak scenario you are defending against plus the test that
proves the defence works.
*Strong answer:* names the fields, their types, and which are filterable versus merely stored;
chooses an isolation model consistent with the client count and its blast radius; names a concrete
leak scenario (a query in client A's session retrieving client B's clause); describes a test that
*fails* when the defence is removed.
*Weak answer:* a field list with no filterability distinction; "we'll use filters"; a leak scenario
that is a category ("data leakage") rather than a sequence.

**B4 (9). The query-transformation budget.**
LogiTrack has a 2.5 s P95 budget. Choose which of query rewriting, decomposition, HyDE, multi-query
expansion and routing you will ship, price each in tokens and milliseconds, state which stratum each
buys, and say what you drop when the arithmetic does not fit.
*Strong answer:* prices each transformation separately and sums them against the budget; ties each to
a stratum (routing to jurisdiction resolution; decomposition to the multi-hop stratum; rewriting to
colloquial queries); drops at least one and names the stratum left exposed; notes that transformations
are F1 remedies that cannot fix a structural gap.
*Weak answer:* ships all of them; no per-transformation pricing; no statement of what is left
unaddressed.

---

## Section C — diagnostic problems (3 problems × 8 points)

Each problem gives a failing pipeline and its traces. Localize the failure to a stage, name the
failure ID, justify the localization, and state the experiment that would confirm it. Diagnosis and
justification carry the marks; the correct label alone earns 2.

**C1 (8). The query that never matched.**
Traces:
- Query: *"What is the DG surcharge for lithium batteries on the Rotterdam–Singapore lane?"*
- Retrieved top-5 (node, score): rate-table row (commodity rate, lane match) 0.81 · rate-table row
  (commodity rate, adjacent lane) 0.79 · wiki page on DG handling 0.74 · customs manual excerpt 0.72
  · rate-table row (commodity rate, same lane, other container) 0.70.
- The corpus contains the correct answer in a rate-table row keyed on **UN3480**; that row exists in
  the index and is intact, and it does not appear in the top-50.
- The generator's answer cites the retrieved rate-table row and quotes **€0**, which is present in
  the cited node's text.

Localize the failure, name the ID, and justify why the retrieval of a node containing the cited number
does not make this an F6 failure. Then state the experiment that distinguishes your diagnosis from a
pure F1 diagnosis, and say what each outcome would tell you to change.
*Strong answer:* localizes to stage 4 (embed) as F4 — user vocabulary ("lithium batteries") versus
corpus vocabulary ("UN3480") — while noting the retrieval was *correct* given the query embedding and
that F1 is a consequence rather than the cause; recognizes the answer as faithful to its context (so
not F6) and verifiable-and-wrong; proposes the discriminating experiment (re-run with a lexical
retriever, or with the identifier in the query) where success confirms F4 and failure would indicate a
missing or mis-indexed node (F1, stage 5).

**C2 (8). The clause that was amputated.**
Traces:
- Query: *"What is the demurrage liability window in our Maersk agreement?"*
- The source clause reads: *"Demurrage liability shall accrue from the fourth (4th) calendar day
  following discharge, and shall be extinguished on the fourteenth (14th) calendar day, after which
  the carrier's liability for delay shall cease."*
- Node 41 (rank 1, score 0.88) contains the clause text up to "...extinguished on the fourteenth
  (14th) calendar day". Node 42 (rank 9, score 0.79) contains "after which the carrier's liability
  for delay shall cease."
- The answer states the liability window is fourteen days, citing node 41.

Localize the failure, name the ID, and justify why a reranker would not have fixed it. State what the
retrieved-node text alone would have shown an engineer had they looked only at node 41, and what
architectural change you would make rather than a configuration change.
*Strong answer:* localizes to stage 3 (chunk) as F3 with the boundary as the cause; notes the
generator was faithful to an amputated span; explains that a reranker over a candidate set containing
both halves produces a *better-ordered* set but an answer that still needs both halves in the context
(so the fix is expansion, not ranking); identifies the readable signature (a sentence with no
terminator); recommends a context-expansion structure (sentence-window with metadata replacement, or
hierarchical/auto-merging) rather than a size change.

**C3 (8). The fact that was in the context and not in the answer.**
Traces:
- Query: *"Summarize our liability exposure across the top three carriers for demurrage."*
- Retrieved and assembled: 10 nodes, 9,200 tokens. The relevant clause for carrier 2 is node 7 of 10
  and occupies tokens ~6,100–6,700 of the assembled context.
- Answer: covers carriers 1 and 3 in detail, states that carrier 2's agreement "does not specify a
  demurrage window", with a citation to node 2 (a general terms node).
- The generator's context window is 16,000 tokens. No truncation occurred. A faithfulness score of
  0.93 was recorded for this answer.

Localize the failure, name the ID, and explain how a 0.93 faithfulness score is consistent with a
wrong answer. Then state the measurements that would distinguish F5 from F6 in this case, and say
which single change you would make first and why.
*Strong answer:* localizes to stage 8 (assemble) as F5 in the first instance — the evidence was
retrieved and present but positioned mid-context, where use is least reliable — while explicitly
keeping F6 open and stating the discriminating measurement (re-score with the answer span moved to
the start or end of the context, and separately with the context reduced to the 3 relevant nodes);
notes the faithfulness judge scored the *cited* node (a general terms node from which the claim is
indeed unsupported-but-plausible) or accepted the negative claim as consistent with absent evidence —
i.e. faithfulness-as-judged does not detect an omission; infers that the highest-leverage change is
reordering plus reducing the context (a fix to both F5 and the token budget) rather than a prompt
rewrite, since the evidence never reached a usable position.

---

## Answer rubric — what a strong and a weak answer look like

The paper is marked on five dimensions, applied across all three sections.

| Dimension | Strong answer | Weak answer |
|---|---|---|
| **Mechanism** | Explains *why* a failure happens at a specific stage, in terms of what that stage does to the data | Names the stage or the technique and stops |
| **Quantification** | Puts a number on the decision: tokens, milliseconds, cents, recall at a k, storage at a node count | Uses adjectives ("more expensive", "slower", "better recall") |
| **Reversal condition** | States the metric, the threshold and the duration that would change the decision | States a preference, or "if quality drops" |
| **Stage attribution** | Localizes a symptom to one stage by ascending failure ID, and says what would falsify the localization | Lists several plausible causes and picks the most familiar |
| **Honest uncertainty** | Names the stratum, capability or scenario that the answer does *not* cover, and what it would cost to cover it | Presents the answer as complete |

Two failure patterns cost the most marks and are worth naming explicitly. **Tuning a structural
problem** — proposing `top_k`, chunk size or a prompt change for a failure that no parameter can
reach (the multi-document aggregate stratum, a cross-tenant leak) — is marked as wrong even when the
change would improve the aggregate metric. And **answering a diagnosis question with a remedy** —
skipping the localization to jump to a fix — loses the diagnostic marks even if the fix is correct,
because the marks are for the reasoning path.

---

## Marking guide

| Section | Items | Points each | Total |
|---|---|---|---|
| **A** — conceptual reasoning | 10 | 4 | **40** |
| **B** — design and decision problems | 4 | 9 | **36** |
| **C** — diagnostic problems | 3 | 8 | **24** |
| | | | **100** |

**Section A marking.** Each question's four rubric points are independent. A question answered with a
correct decision and no mechanism earns 1. A question answered with an unsupported mechanism earns 0
on the mechanism point but may still earn the quantification and reversal points if they are present
and consistent with the (wrong) decision — consistency is marked, not agreement with the instructor's
preference.

**Section B marking.** Per problem: decision stated in the imperative, 2 · price or arithmetic, 3 ·
rejected alternative with a quantified reason, 2 · reversal condition as a measurement, 2. A design
problem with no arithmetic cannot exceed 6.

**Section C marking.** Per problem: correct failure ID, 2 · correct owning stage, 2 · justification
that distinguishes the diagnosis from the adjacent failure (F1 vs F4, or F5 vs F6), 3 · the
confirming experiment, 1. A diagnosis with the right ID and no justification scores 2 of 8.

**Grade bands.** 85+ — every decision priced and reversible; diagnoses localized with falsifiable
tests. 70–84 — sound decisions, thin arithmetic or a missing reversal condition in two or more
places. 55–69 — correct use of the taxonomy and the vocabulary, but decisions asserted rather than
priced. Below 55 — recall of terminology without mechanism, or remedies offered in place of
diagnoses.
