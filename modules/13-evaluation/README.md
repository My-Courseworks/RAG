# M13 · Evaluation of Retrieval and Generation

> **Module question:** How do you know it got better, and how do you keep it from getting worse?
> **Cross-cutting threads:** Failure modes · Tradeoff ledger · LlamaIndex at a glance
> **Domain spine:** LogiTrack — growing the M1 sixty-query set into 400 stratified queries and making it the merge gate
> **Prerequisites:** [M11 — Context Assembly, Prompting, and the Temperature Dial](../11-context-assembly-temperature/README.md), [M12 — Grounding, Citations, and Attribution](../12-grounding-and-citations/README.md)

---

## Opening scene — four months of improvements, no change in quality

LogiTrack's team ships fast. Every change is benchmarked against the 60 queries frozen in M1, and
every change wins: hybrid retrieval (+0.06 recall@10), a cross-encoder reranker (+0.04 nDCG@10),
query rewriting for colloquial questions (+0.09 recall@10 there), hierarchical chunking (+0.05
recall@10), `compact` instead of `refine` synthesis (fewer calls, flat quality). The dashboard is a
wall of green arrows, and the weekly review reports that retrieval is "basically solved".

In the same period the escalation rate to human brokers is flat, and M12's hand-run citation audit
still finds ungrounded claims in three of every forty answers. Nothing is broken. The evaluation is,
in three ways:

1. **The 60 queries were sampled from what the pilot could already answer** — written in M1 by the
   engineer who built the first index, against the documents that index retrieved well. Every
   stratum the system is bad at (numeric tariff extraction, cross-reference questions,
   *unanswerable* questions) is absent or represented by two cases.
2. **Retrieval metrics were reported as the outcome.** The metric that tracks the compliance
   incident — ungrounded claims — was never computed. A 0.06 recall gain costing 0.03 citation
   precision is a net loss on the failure mode that mattered, and the harness cannot express that
   trade.
3. **The gate was a dashboard.** Numbers were read, discussed, and released anyway, because no
   threshold was written down and no stratum could block. A metric that cannot fail a merge has
   already been overruled.

The team did the hard part: they built an evaluation set and ran it weekly. What they built was a
scoreboard for the work already done, not a test suite for the work not yet done.

> **Failure mode (the module in one line):** treating evaluation as a score to report rather than
> as **a gate with a noise floor, a stratified sampling frame, and the same maintenance obligations
> as any other test suite** — so the harness optimises the queries it happens to contain, rewards
> changes that make the digits move, and cannot block anything.

---

## Retrieval metrics, and precisely what each one ignores

Five standard metrics are in play, and LlamaIndex computes all of them over a ground-truth node
set: the framework's notebook constructs the evaluator with
`metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]` and passes it through
`RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)` [LI-EVR2]; the usage-pattern
page demonstrates the same call with `["mrr", "hit_rate"]` [LI-EVR]. The guide summarises the family
the same way: "given a dataset of questions and ground-truth rankings, we can evaluate retrievers
using ranking metrics like mean-reciprocal rank (MRR), hit-rate, precision, and more" [LI-EV].

Notation: for query $q$, $R_q$ is the ground-truth relevant set, $A_q(k)$ the top-$k$ retrieved
set, and $\mathrm{rel}_q(i) \in \{0,\dots,g\}$ the graded relevance at rank $i$.

$$\text{Recall@}k = \frac{|R_q \cap A_q(k)|}{|R_q|} \qquad\qquad \text{Precision@}k = \frac{|R_q \cap A_q(k)|}{k}$$

$$\text{MRR} = \frac{1}{|Q|}\sum_{q} \frac{1}{\mathrm{rank}_q(\text{first relevant})} \qquad \text{AP}_q = \frac{1}{|R_q|}\sum_{i \le k,\; \mathrm{rel}_q(i) > 0} \text{Precision}_q@i$$

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\mathrm{rel}_q(i)} - 1}{\log_2(i+1)} \qquad\qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

Each is blind to something structural, and the blind spots are what make a headline number dangerous:

| Metric | Rewards | Structurally blind to | The degenerate strategy it invites |
|---|---|---|---|
| **Recall@k** | Finding relevant nodes *anywhere* in the top $k$ | Ordering within the top $k$; the cost of the irrelevant slots that came with them | Retrieve more — raise $k$ until recall looks acceptable and let the reranker sort it out (→ M9) |
| **Precision@k** | A clean top $k$ | Ordering within the top $k$; and it is **capped by the ground truth**: if $|R_q| = 3$ and $k = 10$, precision@10 $\le 0.3$ at any system quality | Retrieve fewer — shrink $k$, discarding recall into a different column |
| **MRR** | The *first* relevant node near rank 1 | Everything at rank 2 and beyond: one relevant chunk at rank 1 and forty relevant chunks in the top 10 score identically | Optimise rank 1 exclusively; nothing penalises losing the other 39 |
| **MAP** | Early saturation of recall, averaging precision at each relevant position | Graded relevance (binary only); and it is dominated by **where recall saturates** — 6 of 10 relevant nodes outside your $k$ contribute zero but still divide $\text{AP}_q$ | Tune for the first few hits; or compute AP over the *full* ranking and report a number for a ranking the query engine never consumes |
| **nDCG@k** | Graded gain with a positional discount | The absolute number of relevant nodes, because IDCG normalisation divides $|R_q|$ out — nDCG 0.9 on a one-chunk ground truth and on a fifty-chunk one print identically | Report nDCG alone and never notice recall collapsing on thin-ground-truth queries |

Two deserve spelling out, because they produce confidently wrong conclusions.

**nDCG is two assumptions wearing one number.** The gain term $2^{\mathrm{rel}} - 1$ asserts a
grade-3 node is worth exactly **seven** grade-1 nodes. The discount $1/\log_2(i+1)$ asserts rank 10
is worth $1/\log_2 11 \approx 0.29$ of rank 1 — an attention-decay model with no basis in your UI,
where a reader may or may not scroll. Switching the gain function from exponential to linear changes
which of two retrieval configs wins whenever the grade distributions differ. State both assumptions
or nDCG is a preference wearing a decimal point.

**MAP measures a ranking you do not ship.** $\text{AP}_q$ sums precision at *every* relevant
position in the ranking you scored, while a RAG pipeline consumes the top-$k$ its context budget
allows (→ M11, M18). Computing AP over a 100-node ranking and feeding 6 nodes to the synthesizer
means 94% of the number describes a decision the system never makes. The only defensible form is
$\text{AP@}k$ at the same $k$ the query engine uses. The framework's notebook computes all six
metrics at whatever $k$ the retriever was built with (`similarity_top_k=2`) [LI-EVR2], which makes
`similarity_top_k` a *metric-defining* parameter, not just a runtime one.

LogiTrack's headline retrieval metric is therefore **recall@10**, whose failure mode (F1 · retrieval
miss) is unbounded — a chunk never retrieved cannot be recovered downstream — while its blind spot
(ordering) is M9's reranker job, measured separately with nDCG@10.

> **LlamaIndex at a glance:** `RetrieverEvaluator` evaluates a retriever against a query plus
> ground-truth node ids — `retriever_evaluator.evaluate(query="query", expected_ids=["node_id1",
> "node_id2"])` — and the batch path is `await retriever_evaluator.aevaluate_dataset(qa_dataset)`,
> which the guide notes "should run much faster than you trying to call `.evaluate` on each query
> separately" [LI-EVR]. Per-query results carry a `metric_vals_dict` for building your own table
> [LI-EVR2].

---

## The ground truth is a node set, and that decision moves the metric

`RetrieverEvaluator` takes `expected_ids`, not spans [LI-EVR]. Your annotations are span-level
(M3 annotated each query with its answer's character offsets). Something must convert spans to node
ids, and the rule is a measurement decision with a larger effect than most retrieval changes:

| Span → node rule | A node counts as relevant when | Effect |
|---|---|---|
| Any intersection | It overlaps the answer span at all | Inflates $|R_q|$ with boundary neighbours, deflating precision@k and inflating recall@k |
| Containment | It contains the *entire* answer span | Deflates $|R_q|$; under chunking that splits the answer (F3), recall@k is 0 for a query the pipeline answers correctly |
| Overlap fraction | Overlap $\ge \tau$ of the answer span, for a stated $\tau$ | Tunable and honest; adds a second parameter that must be frozen with the query set |
| Answer-bearing | A human marks the node(s) that carry the answer | Most faithful and most expensive; the only rule that survives a re-chunk unchanged, because a human re-marks |

The rule must be **frozen with the query set** and re-derived whenever chunking changes (→ M3),
because a re-chunk invalidates the node ids whether or not the question changed. This is the first
place evaluation acquires a maintenance obligation a software test does not have: the test's
fixtures are stable; yours are derived from a corpus that is being re-ingested (→ M14).

---

## Generation metrics are four failure axes, not one score

Both flagship frameworks say the generation side is several dimensions at once. RAGAS frames
evaluation as "the ability of the retrieval system to identify relevant and focused context
passages, the ability of the LLM to exploit such passages in a faithful way, or the quality of the
generation itself", and presents "a suite of metrics which can be used to evaluate these different
dimensions without having to rely on ground truth human annotations" [RAGAS23]. ARES names three in
its own vocabulary: "evaluating RAG systems along the dimensions of context relevance, answer
faithfulness, and answer relevance" [ARES23].

| Axis | The question | Fails independently when | Ground truth needed? |
|---|---|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? | The generator adds parametric content (→ M12's opening scene) | No |
| **Answer relevance** | Does the answer address the question asked? | The system answers a nearby question, or refuses when it could answer | No |
| **Context precision** | Is the retrieved context relevant to the query? | Retrieval noise buries the right chunk (F2 → M9) | No |
| **Context recall** | Does the retrieved context contain what is needed? | Retrieval misses (F1 → M3, M7), or assembly truncates (F5 → M11) | Usually yes, by reference to a gold answer |

That last column is the operational fact the framework states plainly: "Note that many of these
current evaluation modules do *not* require ground-truth labels. Evaluation can be done with some
combination of the query, context, response, and combine these with LLM calls" [LI-EV]. It is also
the source of the field's biggest disagreement, because reference-free does not mean cheap: the
label was replaced by an LLM call whose reliability is now the thing you must measure.

Why four axes rather than one score is not stylistic — they dissociate, and the dissociation is the
diagnosis:

| | Context recall **high** | Context recall **low** |
|---|---|---|
| **Faithfulness high** | The good case: grounded, complete, correct | The dangerous case. Faithfulness reports **1.0** while the required evidence was never retrieved; the system either refuses validly or answers fluently from pretraining — and M12 showed the second outcome is the one that carries citations and ships |
| **Faithfulness low** | Fluently contradicts good evidence. Loud, rare, caught in a demo | Broken twice over; surfaces as nonsense or an abstention |

Answer relevance cuts across both: a system can have recall 1.0, faithfulness 1.0, and answer
relevance 0, because it retrieved the right evidence and answered a different question. **No single
number can be in all four of those places at once**, and a dashboard reporting one of them as
"quality" has silently chosen which failure mode it does not care about. For LogiTrack the choice is
explicit: report all four, and let faithfulness be the one that must never regress — an unfaithful
answer is the compliance incident (→ M12), while a context-recall miss is a coverage loss a broker
notices immediately.

### The evaluator classes, and which axis each one touches

| Primitive | Documented behaviour | Axis | Interface |
|---|---|---|---|
| `FaithfulnessEvaluator` | "evaluates if the answer is faithful to the retrieved contexts (in other words, whether if there's hallucination)" [LI-EVU] | Faithfulness | `evaluate_response(response=response)`, or `evaluate(response=..., contexts=[node.get_content()])` per node |
| `RelevancyEvaluator` | "evaluates if the retrieved context and the answer is relevant and consistent for the given query"; "requires the `query` to be passed in, in addition to the `Response` object" [LI-EVU] | Answer relevance **and** context relevance, jointly | `evaluate_response(query=query, response=response)` |
| `CorrectnessEvaluator` | "Whether the generated answer matches that of the reference answer given the query (requires labels)" [LI-EV] | Correctness | Needs a reference answer |
| `SemanticSimilarityEvaluator` | "Whether the predicted answer is semantically similar to the reference answer (requires labels)" [LI-EV] | Correctness, softly | Needs a reference answer |
| `RetrieverEvaluator` | Retrieval metrics over `expected_ids`, via `from_metric_names([...])` [LI-EVR] | Context recall (partially) | `evaluate(query, expected_ids)`, `aevaluate_dataset(qa_dataset)` |
| `BatchEvalRunner` | Runs "a set of evaluators across many questions" with `workers=8` [LI-EVU] | All of the above at scale | `await runner.aevaluate_queries(query_engine, queries=questions)` |

Row two contains an ambiguity that bites. The guide describes `RelevancyEvaluator` as measuring
whether "the retrieved context **and** the answer is relevant and consistent for the given query"
[LI-EVU] — a *conjunction*, one result over two variables — while RAGAS and ARES keep the two
separate [RAGAS23][ARES23]. The framework's answer is to evaluate per source node with
`contexts=[source_node.get_content()]`, turning the conjunction into one result per node [LI-EVU],
but the guide's terminology does not distinguish the two axes, so build the split explicitly.
Every output is an `EvaluationResult` with `passing`, `score`, and `feedback`, and "different
evaluators may populate a subset of the result fields" [LI-EVU] — so do not assume `score` exists.

> **Tradeoff (the ledger entry M13-1).** *Decision:* make faithfulness the non-regressing gate
> metric and treat context recall, nDCG@10, and cost as reported-with-margin metrics.
> *Costs:* the gate becomes blind to slow recall erosion that never crosses the faithfulness
> threshold, and one axis carries the release decision alone. *Accepted because:* the axes are not
> equally expensive to get wrong — an unfaithful answer is an unbounded compliance cost, a recall
> miss a bounded coverage cost the M19 online signals surface within a day. *Reverses if:* the
> dominant query class is exploratory and an abstention costs as much as a wrong answer, at which
> point recall becomes the unbounded axis and the gate inverts.

---

## LLM-as-a-judge: what it agrees with, and the three biases you must price

Because most generation axes are reference-free [LI-EV], and because the guide states the mechanism
plainly — "This uses a 'gold' LLM (e.g. GPT-4) to decide whether the predicted answer is correct in
a variety of ways" [LI-EV] — the generation half of your harness rests on a judge. That makes the
judge a *component under test*, not an oracle. MT-Bench both licenses and bounds the practice:
"We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and
self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate
some of them", and it reports that "strong LLM judges like GPT-4 can match both controlled and
crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement
between humans" [JUDGE23].

Self-preference has the clearest mechanism. Wataoka and colleagues introduce "a novel quantitative
metric to measure the self-preference bias", report that "GPT-4 exhibits a significant degree of
self-preference bias", and locate the cause: "LLMs assign significantly higher evaluations to
outputs with lower perplexity than human evaluators, regardless of whether the outputs were
self-generated. This suggests that the essence of the bias lies in perplexity and that the
self-preference bias exists because LLMs prefer texts more familiar to them" [SELFJUDGE23]. That
last sentence changes harness design: the fix is not merely a judge from a different model family —
the mechanism is *familiarity*, so any judge favours whatever prose most resembles its own training
distribution. Tune your generator on the judge's corpus and you have built the bias in on purpose.

| Bias | Documented shape | Mitigation | Residual risk |
|---|---|---|---|
| **Position** | Preference depends on which candidate is shown first [JUDGE23] | Score both orderings and average; discard pairs where the orderings disagree | Doubles judge cost; the disagreement rate becomes its own metric |
| **Verbosity** | Longer answers preferred [JUDGE23] | Rubrics scoring *claims present or absent* rather than overall quality; a length-control instruction; report the answer-length distribution alongside the score | An enumerating rubric approaches a hand-check and loses the cost advantage |
| **Self-preference** | Judge favours lower-perplexity text; the effect tracks familiarity, not authorship [JUDGE23][SELFJUDGE23] | Cross-family judge, **and** periodic re-derivation of judge-human agreement on *your* corpus | Every judge has a familiarity profile; the only real control is a human-anchored sample |
| **Limited reasoning** | Weak on tasks needing chains of inference [JUDGE23] | Route multi-hop and numeric-extraction claims to correctness-with-reference or human review | Those are exactly the classes where judging is hardest and matters most |

The rule: **calibrate the judge, then gate on the judge.** Calibration means a stratified sample of
the golden set, hand-labelled, with judge-human agreement reported as a number you maintain —
invalidated the moment you change judge model, judge prompt, or generator. That is the eval set's
own maintenance obligation applied to the instrument.

> **Tradeoff (the ledger entry M13-2).** *Decision:* use an LLM judge for faithfulness,
> recalibrated against a hand-labelled sample at every judge or generator change, and never for the
> release-blocking decision on the numeric-extraction stratum. *Costs:* a recurring hand-labelling
> budget and a two-tier harness where the compliance stratum is label-measured and the rest is
> judge-measured. *Accepted because:* a judge whose agreement with humans is unmeasured is a scoring
> function with unknown bias, and gating a release on it is gating on an unvalidated artifact.
> *Reverses if:* a judge is validated on your corpus at agreement high and stable enough across
> versions to be pinned — at which point the hand-labelling drops to a slow audit.

---

## The synthetic-query trap, quantified

The framework offers synthetic generation as the shortcut that makes all of this affordable:
`generate_question_context_pairs(nodes, llm=llm, num_questions_per_chunk=2)`, which "uses the LLM
to auto-generate questions from each context chunk" and returns an `EmbeddingQAFinetuneDataset`
"containing `queries`, `relevant_docs`, and `corpus`" [LI-EVR]; and
`RagDatasetGenerator.from_documents(...)` followed by `generate_questions_from_nodes()`, documented
under "Question Generation" as creating "a fully automated evaluation flow over your data" [LI-EVU].

Both work by the same mechanism, and that mechanism is the trap: **the query is generated *from* a
chunk, and that chunk is simultaneously declared the ground truth.** The evaluation set is produced
by a function whose relevant-doc label is an input, not an annotation.

Why that inflates scores, in arithmetic checkable in-file. The generator is instructed to produce a
question answerable only from chunk $c$, so the question's content derives from $c$. Write the
retrieval score as surface overlap $\phi(q,c)$ plus semantic proximity $\psi(q,c)$; because $q$ was
conditioned on $c$, both are maximised at $c$ by construction. The retrieved rank of $c$ is then
bounded by the number of chunks closer to $q$ than $c$ is — for a paraphrase of $c$, essentially the
near-duplicate population of $c$, which M3 measured as non-empty because overlap creates
near-duplicates and which M14 exists to shrink. Two consequences, and they are different problems:

1. **Absolute inflation.** A random retriever over $|C|$ chunks with single-chunk ground truth scores
   $\text{recall@}10 = 10/|C|$. As the pipeline improves, synthetic recall@10 climbs toward 1.0 and
   saturates, so the metric stops discriminating exactly where you need it. This is the ceiling
   effect, and it is the direct cause of the opening scene: five green arrows, because the harness
   had no headroom left to detect anything but the changes it was built around.
2. **Variance collapse.** A change that moves real-query recall@10 by 0.06 moves synthetic recall@10
   far less, because the source chunk sits at rank 1 before and after. The harness does not merely
   inflate the level; it *shrinks the differences you are trying to gate on*, so an A/B gate
   calibrated on synthetic queries under-fires.

The framework's own two-step structure exposes the fix: the retriever guide presents "Dataset
generation" and "Retrieval Evaluation" as separate steps [LI-EV], and offers manual curation as the
first-class alternative — "You can manually curate a retrieval evaluation dataset of questions +
node id's" [LI-EVR]. Manual curation is the only path whose ground truth is not derived from the
retrieval target.

| Use | Synthetic generation | Because |
|---|---|---|
| Bootstrapping a first harness for a new corpus | ✅ | A ceiling-limited set still catches gross breakage |
| Smoke-testing that an ingestion change did not destroy the index | ✅ | Direction, not magnitude, is what you need |
| Training-data construction (→ M15's pseudo-labelling) | ✅ | Overlap with the retrieval target is the *point* there |
| Calibrating a release threshold or an SLO | ❌ | The number is inflated by an unknown amount |
| Choosing between two configs that differ by < 0.05 | ❌ | The variance in the difference is below the ceiling gap |
| Detecting that recall is *eroding* over months | ❌ | Saturation hides erosion until it becomes a cliff |

> **Tradeoff (the ledger entry M13-3).** *Decision:* hand-curate the 400-query golden set with
> span-level ground truth; allow synthetic generation only for smoke tests and training data.
> *Costs:* an ongoing annotation budget — roughly two engineer-weeks to build, a few hours a month to
> maintain — and a set that ages with the corpus (→ M14, M20). *Accepted because:* a synthetic set
> cannot gate a release, and a set that cannot gate is a scoreboard; the opening scene is the price
> of not paying this one. *Reverses if:* synthetic queries are generated *without* conditioning on
> the retrieval target — from a held-out document set the retriever cannot see — which removes the
> label leak but also removes the guarantee that a supporting chunk exists.

---

## Building and maintaining the golden set: stratification, sizing, decay

Growing 60 unstratified queries to 400 is a **sampling frame** exercise. Strata are chosen so each
contains a failure mode you intend to detect and block.

| Stratum | Queries | Failure mode it guards | Metric that must move |
|---|---|---|---|
| Known-item / clause lookup ("clause 14.2(b)", "UN3480") | 90 | F1 · retrieval miss on exact identifiers (→ [M7](../07-sparse-dense-hybrid/README.md)) | recall@10 |
| Numeric extraction (tariff rates, DG surcharges) | 70 | F6 · unfaithful synthesis; verbatim correctness (→ M11, M12) | faithfulness, exact-match |
| Comparative, cross-carrier | 50 | F3 · answers spanning documents | context recall |
| Aggregate / "across all 12 carriers" | 40 | No top-$k$ answer exists at all (→ M8, M16) | abstention correctness |
| Multi-hop / cross-reference ("see §4.1 of the CSA") | 40 | F3 · chunk-boundary loss across documents (→ M8) | nDCG@10 |
| Vague / colloquial | 50 | F4 · representation mismatch (→ M10) | recall@10 vs baseline |
| **Unanswerable — must abstain** | 40 | Ungrounded answers, scope confusion (→ M12, M17) | abstention precision, must be 100% |
| Out-of-scope jurisdiction / adversarial | 20 | F9 · poisoned context, F10 · scope leak (→ M21) | leak rate, must be 0 |

Two thirds of the frame is decided by asking *what should be blocked*, not *what is common*. The
unanswerable stratum is the one an unstratified 60-query set never contains — nobody writing
queries from their own corpus writes one the corpus cannot answer — and it is the only stratum
measuring the outcome the opening scene's compliance incident produced.

**Sizing.** For a proportion metric, $\text{SE} = \sqrt{p(1-p)/n}$. At $p = 0.8$, $n = 400$:

$$\text{SE} = \sqrt{\frac{0.8 \times 0.2}{400}} = 0.02 \quad\Rightarrow\quad \text{95\% CI half-width} \approx 1.96 \times 0.02 \approx 0.039$$

So 400 queries resolve a **~4-point** system-level difference. Push the same arithmetic into a
stratum — 8 strata, ~50 each:

$$\text{SE} = \sqrt{\frac{0.16}{50}} = 0.057 \quad\Rightarrow\quad \text{95\% CI half-width} \approx 0.11$$

**This is the most consequential arithmetic in the module: 400 queries support a system-level
statement and cannot support a per-stratum one.** Either size the gated strata separately, or
report per-stratum results as directional evidence and never as thresholds. Corollary for the noise
floor: six metrics across eight strata is up to 48 comparisons per merge, so at $\alpha = 0.05$
expect roughly $0.05 \times 48 \approx 2.4$ false alarms per merge if you gate on all of them. Gate
on two headline metrics and report the rest.

**Decay.** An evaluation set rots in four ways, each with an owner:

| Decay mode | Symptom | Maintenance action |
|---|---|---|
| **Corpus drift** | A query's ground-truth chunk no longer exists, or its clause was superseded (→ M14) | Re-derive span→node mappings on every re-chunk; retire queries whose source was hard-deleted |
| **Distribution drift** | The query mix moved; the stratum counts did not (→ M19) | Re-sample stratum proportions from the last 30 days of real queries, quarterly |
| **Contamination** | Your engineers have seen the 400 queries often enough to tune against them | Hold out a rotating 20% never used for tuning; rotate annually |
| **Judge drift** | Judge model or prompt changed, silently rescaling every threshold | Recalibrate judge-human agreement; version the gate threshold with the judge |

That last row is the trap in soft-gated harnesses: a threshold is calibrated against an instrument.
Change the instrument without recalibrating and every threshold becomes meaningless while continuing
to produce plausible numbers.

### The harness mindset: an evaluation set is a test suite

| Test-suite obligation | Eval-set equivalent for LogiTrack |
|---|---|
| An owner | A named eval owner per stratum; the unanswerable stratum is owned by the compliance lead, not the ML engineer |
| Versioning | Query set, corpus snapshot, index snapshot, and judge version ship as one versioned tuple; a run records all four |
| Fixtures are maintained | Span→node mappings re-derived whenever chunking changes; retiring a query is a reviewed change, not a deletion |
| A flake budget | Judge variance quantified and published; judge temperature 0; judge outputs cached so re-runs are byte-identical |
| Assertions are reviewed | A new case needs a written failure narrative and a stratum; every production incident adds a case within the sprint (→ [M20](../20-failure-modes-and-drift/README.md)) |
| A fast subset in CI | 60-query smoke gate on every pull request; the full 400 + judge axes nightly |
| Failing tests fixed or deleted, never muted | No `xfail` analogue: a red gate blocks or escalates to a decision with written acceptance |

The last line separates the opening scene from a working harness. An eval set that can be explained
away is a scoreboard; one whose failures require a written decision is a gate.

The shape of the loop, with the two feedback edges that make it a harness rather than a report:

```mermaid
flowchart TD
    C[Corpus snapshot<br/>pinned by version] --> IX[Index snapshot]
    G[Golden set 400<br/>stratified, versioned,<br/>span-level ground truth] --> IX
    IX --> R{Retrieval eval}
    IX --> N{Generation eval<br/>per axis}
    R --> M1[recall@k, nDCG@k, MRR, AP@k]
    N --> M2[faithfulness, answer relevance,<br/>context precision, context recall]
    N --> M3[correctness, semantic similarity<br/>requires labels]
    M1 --> CAL[Judge calibration<br/>hand-labelled sample<br/>agreement rate]
    M2 --> CAL
    M3 --> CAL
    CAL --> GT{Gate vs pinned baseline}
    GT -->|pass| SH[Merge]
    GT -->|block| BL[Block + per-stratum report]
    BL --> NC[New case written from the failure] --> G
    CAL -->|judge or generator changed| RC[Thresholds invalid:<br/>recalibrate before gating] --> GT
    DR[Corpus drift<br/>superseded sources] -->|re-derive span-to-node maps| G
```

---

## Wiring it as a merge gate rather than a dashboard

A gate is a function from a diff to a boolean, and it needs four things a dashboard does not.

| Element | Requirement | What goes wrong without it |
|---|---|---|
| **Frozen target** | Gate against a *pinned index snapshot*, not the live index, so a code change is not compared to a corpus that moved underneath it (→ M14) | A parser regression looks like a retrieval improvement |
| **Noise floor** | State the margin: at $n=400$ differences under ~4 points are unresolvable, so a 2-point gate fires on noise | Flaky gates; the team learns to re-run until green |
| **Failure semantics** | Two gate types: an **absolute SLO** (known-item recall@10 ≥ 0.85, LogiTrack's M1 number) and a **no-regression margin** against the pinned baseline | Only one type: an absolute gate cannot catch erosion while you are above it, and a relative gate ratchets quality downward one accepted regression at a time |
| **Per-stratum guards** | Strata that must never degrade *at all*: unanswerable abstention precision = 1.0, scope-leak rate = 0 | A system-level mean hides a total collapse in a small stratum |
| **Cost budget** | Judge calls dominate cost; predict them before spending | The full harness becomes too expensive to run per merge, so it stops running — a gate too slow to run is not a gate |

Both harness costs are estimable before spending anything. LlamaIndex documents a token-prediction
path built on `MockLLM(max_tokens=256)` and `MockEmbedding(embed_dim=1536)` wired through
`Settings.llm` and `Settings.embed_model`, with `TokenCountingHandler` reading out
`total_embedding_token_count`, `prompt_llm_token_count`, and `completion_llm_token_count` — the
documented purpose being "to predict token usage of LLM and embedding calls ... during 1) index
construction, and 2) index querying, before any respective LLM calls are made" [LI-EVC]. M18's cost
model leans on exactly this.

> **LlamaIndex at a glance:** the batch runner is `BatchEvalRunner`, constructed as
> `BatchEvalRunner({"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
> workers=8)` and driven by `await runner.aevaluate_queries(vector_index.as_query_engine(),
> queries=questions)` [LI-EVU]. Two details make it a harness component: the evaluator dict lets you
> name the axes yourself, which is how the conjunction problem gets fixed in practice; and `workers`
> is your concurrency lever against rate limits — the notebook's own run shows the framework
> retrying through `RateLimitError: Error code: 429` several times [LI-EVR2].

---

## Worked example: every system from M3 to M12, on one table

The lab is the harness itself: build it once, then score every architecture the course has produced
against the 400-query set. Retrieval columns from `RetrieverEvaluator` with
`["recall", "ndcg", "mrr", "ap"]`; generation columns from `FaithfulnessEvaluator` and per-node
`RelevancyEvaluator` through `BatchEvalRunner`; cost from `TokenCountingHandler`
[LI-EVR][LI-EVU][LI-EVC].

| System (owner) | recall@10 | nDCG@10 | Faithfulness | Context precision | Context recall | Abstention correct | $/query | p95 ms |
|---|---|---|---|---|---|---|---|---|
| M3 `TokenTextSplitter(1024, 20)` | 0.58 | 0.44 | 0.71 | 0.62 | 0.55 | 0.31 | 0.009 | 1.4 s |
| M3 `SentenceSplitter(512, 50)` | 0.69 | 0.53 | 0.74 | 0.71 | 0.63 | 0.38 | 0.010 | 1.4 s |
| M3 `HierarchicalNodeParser([2048,512,128])` | 0.78 | 0.61 | 0.76 | 0.68 | 0.72 | 0.41 | 0.014 | 1.7 s |
| M7 hybrid, BM25 + dense, RRF | 0.83 | 0.64 | 0.74 | 0.66 | 0.77 | 0.44 | 0.012 | 1.6 s |
| M9 + `SentenceTransformerRerank(top_n=6)` | 0.83 | **0.79** | 0.79 | **0.88** | 0.77 | 0.44 | 0.013 | 1.9 s |
| M10 + query rewriting on vague stratum | 0.87 | 0.80 | 0.79 | 0.87 | 0.80 | 0.46 | 0.016 | 2.2 s |
| M11 temperature 0 vs 0.2 (5 runs) | 0.87 | 0.80 | 0.79 ±0.02 | 0.87 | 0.80 | 0.46 | 0.016 | 2.2 s |
| M12 + node-level citations (`citation_chunk_size=512`) | 0.87 | 0.80 | 0.81 | 0.87 | 0.80 | 0.46 | 0.018 | 2.3 s |
| M12 + span attribution + abstention policy | 0.87 | 0.80 | **0.93** | 0.87 | 0.80 | **0.71** | 0.021 | 2.4 s |

Read it as the gate would, not as a leaderboard:

1. **The reranker is the best retrieval purchase and it is bought on nDCG, not recall.** Row 4 →
   row 5 moves recall@10 by **zero** and nDCG@10 by 0.15 while context precision rises 0.22. A gate
   watching only recall@10 calls the reranker a no-op — the worked proof that one metric's blind
   spot is another's job.
2. **Retrieval improvements stop helping faithfulness.** Rows 3–6 take recall@10 from 0.78 to 0.87
   while faithfulness stays inside 0.74–0.79: the evidence got better, the generator did not get
   more faithful. Only row 9, which changes *grounding* rather than retrieval, moves faithfulness —
   by 0.12. That is the measurement-spine version of M12's argument, and why the gate has two
   headline metrics.
3. **Abstention correctness and faithfulness move together, and both cost money.** The last row is
   the only one where the compliance axis clears, and it is the most expensive and slowest. Whether
   it fits the 2.5 s P95 and \$18k ceiling is M18's question — and the harness has made it a *cost*
   question with a known quality price rather than a taste argument.
4. **The temperature row is flat because the query class is extraction.** Five runs at temperature
   0.2 gave ±0.02 faithfulness on this mix. M11's lab finds a larger spread on the synthesis
   stratum; the point is that the harness can only answer that question if a synthesis stratum
   exists, which is why the frame has one.

**LogiTrack's decision after M13:** the 400-query stratified set is the merge gate; recall@10 and
faithfulness gate, with nDCG@10, context precision, context recall, abstention correctness,
$/query, and p95 reported-with-margin; the unanswerable and scope-leak strata gate at their own
strict thresholds; the gate runs against a pinned index snapshot on every merge and against live
ingest nightly.

---

## Design exercise

> *Paper-based. Think, then write. Deliver a decision, not a survey.*

**Task.** Write the evaluation ADR for a system you own, or for LogiTrack. The deliverable is a gate
specification: what is measured, on what sample, against what threshold, and what happens when it
fails.

1. **Write the sampling frame before the queries.** For each stratum, state the failure mode it
   guards and the metric that must move when that failure mode is present. A stratum with no metric
   that would move is not a stratum, it is a hope.
2. **Do the sizing arithmetic and state what you can claim.** Compute the 95% CI half-width for your
   total sample and your smallest gated stratum at your expected $p$; write down the smallest
   difference your gate can resolve; then resize or delete any stratum whose CI half-width exceeds
   the effect you want to detect.
3. **Set the noise floor and count your comparisons.** State how many gated comparisons run per
   merge and the expected false-alarm count at your chosen $\alpha$. If it exceeds 1, reduce the
   number of gated metrics rather than raising $\alpha$ — a gate nobody trusts is worse than a
   narrow one.
4. **Choose the ground-truth representation and freeze it.** Node ids or spans, the span→node rule,
   the $k$ for every $@k$ metric, and the grading scheme if you use nDCG. State the two nDCG
   assumptions you are making and what would change if you swapped them.
5. **Calibrate the judge or refuse to use one.** For any gated judge-scored axis, state the judge
   model and prompt version, the hand-labelled sample size, the measured agreement, and the
   recalibration cadence. If you cannot afford the sample, report the axis rather than gating it.
6. **Specify the failure path.** For each gated metric: which strata can block, who is paged, the
   documented override, and — the one that matters — what new query gets added to the set as a
   result of the failure.
7. **Price the harness.** Estimate judge calls, tokens, and wall-clock for the full run and the PR
   subset using `MockLLM`/`MockEmbedding` prediction rather than a guess [LI-EVC]. If the full run
   cannot happen per merge, say which metrics are nightly-only and how a nightly regression is
   allowed to affect a release.

**Why this exercise matters.** Evaluation is the only module whose output every other module uses,
and the one most often built as an afterthought because it ships no user-visible feature. The
senior-engineer version of the work is not "compute recall@10" — it is deciding, in writing, which
failures may block a release, how much noise the gate must tolerate, and what it costs to find out.
The opening scene is what a competent team produces without that written decision. Note what the
exercise forces you to give up: the comfort of a single quality score. Four axes, a judge you must
calibrate, and a set that decays are the price of being able to answer the module's question at all.

---

## Where the literature disagrees

- **"LLM judges agree with humans well enough to gate a release."** MT-Bench reports that "strong
  LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving
  over 80% agreement, the same level of agreement between humans" [JUDGE23] — which, if it held for
  your task, would justify gating. The TREC 2024 RAG Track's support-assessment study found
  something materially weaker on exactly this axis: "for 56% of the manual from-scratch assessments,
  human and GPT-4o predictions match perfectly (on a three-level scale), increasing to 72% in the
  manual with post-editing condition", while also reporting that "an independent human judge
  correlates better with GPT-4o than a human judge, suggesting that LLM judges can be a reliable
  alternative for support assessment" [TRECSUP25]. The two papers measure different things —
  *pairwise preference between chatbots* versus *a three-level ground-truth-support judgment* — and
  that is the disagreement in one line: preference agreement transfers poorly to veridical
  assessment. **What to change:** calibrate the judge on *your* task, not the task a benchmark
  validated it for, and treat 56%-on-a-3-level-scale as the prior. On a three-level scale, 44%
  disagreement means judge and human land in different buckets nearly half the time — an acceptable
  *ranking* signal and a dangerous *gate*.
- **"Synthetic query generation gives you a valid evaluation set."** The SIGIR 2024 study
  investigates "whether it is possible to use LLMs to construct fully synthetic test collections by
  generating not only synthetic judgments but also synthetic queries", and concludes: "Our
  experiments indicate that using LLMs it is possible to construct synthetic test collections that
  can reliably be used for retrieval evaluation" — while itself naming the hazard, analysing "the
  potential risks of bias such test collections may exhibit towards LLM-based models" [STC24].
  **This module's argument is narrower and does not contradict it**, which is the honest way to
  report it: the framework's *default* mechanism — a query conditioned on a chunk, with that same
  chunk written in as ground truth [LI-EVR][LI-EVU] — is not the construction the study evaluates.
  **What to change:** if you use synthetic generation, reproduce the studied construction
  (independent queries, generated *judgments*), not the convenience function, and report which one
  you used.
- **"Reference-free evaluation means you do not need labels."** RAGAS presents "a suite of metrics
  which can be used to evaluate these different dimensions without having to rely on ground truth
  human annotations" [RAGAS23], and the guide says the same for many of its evaluators [LI-EV].
  ARES, published weeks later for the same purpose, takes the opposite position: "To mitigate
  potential prediction errors, ARES utilizes a small set of human-annotated datapoints for
  prediction-powered inference (PPI)", across "eight different knowledge-intensive tasks in KILT,
  SuperGLUE, and AIS", and it "accurately evaluates RAG systems while using only a few hundred human
  annotations during evaluation" [ARES23]. **What to change:** read "reference-free" as
  "reference-free at scale, label-anchored at calibration" — RAGAS's contribution is removing the
  per-example label, ARES's is proving that a few hundred labels buy a documented bias correction.
  That is a budget, not a philosophical conflict, and Entry M13-2 is its engineering form.
- **"A public benchmark tells you how your system will do."** CRAG evaluates its own 4,409
  question-answer pairs over "five domains and eight question categories" and reports that "most
  advanced LLMs achieve <=34% accuracy on CRAG, adding RAG in a straightforward manner improves the
  accuracy only to 44%", and that "state-of-the-art industry RAG solutions only answer 63% of
  questions without any hallucination" — while also finding "much lower accuracy in answering
  questions regarding facts with higher dynamism, lower popularity, or higher complexity" [CRAG24].
  The framework guide adds the contamination counterpoint for a different benchmark: "HotpotQA is
  evaluated on a Wikipedia corpus. LLMs, especially GPT4, tend to have memorized information from
  Wikipedia relatively well. Hence, the benchmark is not particularly good for evaluating retrieval
  + rerank systems with knowledgeable models like GPT4" [LI-CWE]. **What to change:** use public
  benchmarks for model *selection*, never for release gating. CRAG's 34%-to-44% gap describes a
  general web corpus with high-dynamism questions; LogiTrack's corpus is 3,000 private freight
  documents no model has memorised — which cuts both ways, because no benchmark has ever measured a
  system like yours either. The framework guide agrees in its own words: BEIR [`[BEIR21]`, surveyed
  in M4] is "more helpful when you have a unique model that you want to evaluate", best used to
  check "whether and by how much its performance degrades on a diverse set of domains" — a drift
  instrument, not a score [LI-CWE]. The community's answer has been to standardize the *submission
  interface* rather than the corpus: the TREC 2024 RAG Track shipped a reusable framework, a chosen
  collection (MS MARCO V2.1), and "standardized the I/O definitions which assist the end user" so
  systems can be compared even when they share no corpus with yours [RAGNAROK24].
- **"There is a best retrieval metric."** This module reports five, defines each, and refuses to
  rank them; the framework's notebook computes `["hit_rate", "mrr", "precision", "recall", "ap",
  "ndcg"]` as a set [LI-EVR2]. The real disagreement is not which metric is correct but whether a
  single headline number is acceptable — and the arithmetic above is the argument: each metric's
  blind spot is another's job, $\text{AP}$ measured past your context budget scores a ranking you do
  not ship, and nDCG's gain and discount are assumptions you must state. **What to change:** name the
  metric, the $k$, the ground-truth rule, and the gain function in the gate specification, and treat
  any unqualified "our nDCG is 0.8" as an unverifiable claim.

---

## Sources

- [Evaluating](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/) — `[LI-EV]` · owns the response/retrieval evaluation split, the evaluator taxonomy, the reference-free claim, and question generation as a workflow.
- [Usage Pattern (Response Evaluation)](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/usage_pattern/) — `[LI-EVU]` · owns `BaseEvaluator`, `EvaluationResult`'s `passing`/`score`/`feedback`, `FaithfulnessEvaluator`, `RelevancyEvaluator`, `RagDatasetGenerator`, and `BatchEvalRunner`.
- [Usage Pattern (Retrieval)](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/usage_pattern_retrieval/) — `[LI-EVR]` · owns `RetrieverEvaluator.from_metric_names`, `generate_question_context_pairs`, `EmbeddingQAFinetuneDataset`, and `aevaluate_dataset`.
- [Retrieval Evaluation](https://developers.llamaindex.ai/python/examples/evaluation/retrieval/retriever_eval/) — `[LI-EVR2]` · owns the six-metric `metrics` list, `metric_vals_dict` per query, the `similarity_top_k=2` example, and the observed 429 retries under batch load.
- [Cost Analysis — Usage Pattern](https://developers.llamaindex.ai/python/framework/understanding/evaluating/cost_analysis/usage_pattern/) — `[LI-EVC]` · owns `MockLLM`/`MockEmbedding` token prediction and `TokenCountingHandler`'s counters.
- [Component Wise Evaluation](https://developers.llamaindex.ai/python/framework/optimizing/evaluation/component_wise_evaluation/) — `[LI-CWE]` · owns the isolate-the-component argument, BEIR as a domain-shift instrument, and the HotpotQA memorization caveat.
- *Ragas: Automated Evaluation of Retrieval Augmented Generation* — Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert · arXiv:2309.15217 (2023) · [arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217) — `[RAGAS23]` · owns the reference-free RAG metric suite and the dimension decomposition.
- *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems* — Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia · arXiv:2311.09476 (NAACL 2024) · [arxiv.org/abs/2311.09476](https://arxiv.org/abs/2311.09476) — `[ARES23]` · owns context relevance / answer faithfulness / answer relevance as judged dimensions and prediction-powered inference with a small human-labelled set.
- *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica · arXiv:2306.05685 (NeurIPS 2023 Datasets and Benchmarks) · [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685) — `[JUDGE23]` · owns position, verbosity and self-enhancement bias, limited reasoning, and the >80% human-agreement result.
- *Self-Preference Bias in LLM-as-a-Judge* — Koki Wataoka, Tsubasa Takahashi, Ryokan Ri · arXiv:2410.21819 (NeurIPS 2024 Safe Generative AI Workshop) · [arxiv.org/abs/2410.21819](https://arxiv.org/abs/2410.21819) — `[SELFJUDGE23]` · owns the quantitative self-preference metric and the perplexity explanation of the bias.
- *Support Evaluation for the TREC 2024 RAG Track: Comparing Human versus LLM Judges* — Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin · arXiv:2504.15205 (SIGIR 2025) · [arxiv.org/abs/2504.15205](https://arxiv.org/abs/2504.15205) — `[TRECSUP25]` · owns the human-vs-GPT-4o support-agreement rates on the TREC RAG Track submissions.
- *Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track* — Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghaddam, Eric Zhang, Ryan Nguyen, Daniel Campos, Nick Craswell, Jimmy Lin · arXiv:2406.16828 (2024) · [arxiv.org/abs/2406.16828](https://arxiv.org/abs/2406.16828) — `[RAGNAROK24]` · owns the TREC 2024 RAG Track's reusable framework, the MS MARCO V2.1 collection choice, and standardized RAG I/O.
- *CRAG — Comprehensive RAG Benchmark* — Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang, Lei Chen, Nicolas Scheffer, Yue Liu, Nirav Shah, Rakesh Wanga, Anuj Kumar, Wen-tau Yih, Xin Luna Dong · arXiv:2406.04744 (NeurIPS 2024 Datasets and Benchmarks) · [arxiv.org/abs/2406.04744](https://arxiv.org/abs/2406.04744) — `[CRAG24]` · owns the 4,409-question factual QA benchmark, its five domains / eight question categories, and the accuracy and hallucination figures. · **M16** owns the survey of this dataset's benchmark use.
- *Synthetic Test Collections for Retrieval Evaluation* — Hossein A. Rahmani, Nick Craswell, Emine Yilmaz, Bhaskar Mitra, Daniel Campos · arXiv:2405.07767 (SIGIR 2024) · [arxiv.org/abs/2405.07767](https://arxiv.org/abs/2405.07767) — `[STC24]` · owns the fully-synthetic test-collection feasibility result and the LLM-bias risk it identifies in such collections.
- *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* — owned by **M4**, surveyed there; cited here only as the instrument the framework guide recommends for detecting post-fine-tuning domain shift [LI-CWE] — `[BEIR21]`. This entry is a cross-reference, not a survey: **M4** owns the
  bibliography entry, the author list, and the paper link.

**Unsupported (synthetic, no located source):** the LogiTrack corpus, the 60- and 400-query sets,
the eight stratum counts, the four decay modes' symptoms, the opening scene's narrative, and every
number in the worked-example table are fiction or constructed illustration, not empirical findings.
The five tradeoff-ledger entries are design decisions argued in this file. All statistics are
derived in-file from $\text{SE} = \sqrt{p(1-p)/n}$ at the stated $n$ and $p$: they are arithmetic
about sampling, not measurements of any system, and the expected-false-alarm count
($\alpha \times{}$ comparisons) assumes independent comparisons, which stratified metrics on a
shared query set are not — treat it as an order-of-magnitude argument for narrowing the gate, not a
calibrated error rate. The claim that synthetic query generation *saturates* recall@10 and *shrinks
measurable differences* is a mechanism argued here from the fact that the generator conditions on
the retrieval target; the cited sources report relative effects and none reports the specific
ceiling effect or variance collapse quantified above. The claim that a gate threshold is invalidated
by a judge change is argued here; no cited source measures threshold drift under a judge swap. Two
items are deliberately **left unassigned**:

- **The reserved key `TREC23` — the course outline's reserved key for "the TREC RAG track / comprehensive RAG
  benchmark" — could not be attached to a real paper without fabricating one.** The TREC RAG Track's
  first edition was TREC **2024**, not 2023 — its framework and baselines are `[RAGNAROK24]`
  [arxiv.org/abs/2406.16828](https://arxiv.org/abs/2406.16828), and its support-assessment study is
  `[TRECSUP25]` [arxiv.org/abs/2504.15205](https://arxiv.org/abs/2504.15205). The comprehensive RAG
  benchmark is `[CRAG24]`, a KDD Cup 2024 dataset and explicitly *not* the TREC track. Rather than
  date a 2024 artifact 2023, the slot is reported empty here.
- **`RelevancyEvaluator`'s "answer relevance vs context relevance" split is not documented as a
  split.** The class measures whether "the retrieved context and the answer is relevant and
  consistent for the given query" [LI-EVU] — one result over two variables — while `[RAGAS23]` and
  `[ARES23]` treat the two as separate dimensions. This module asserts the split must be built
  explicitly in the harness; that is an inference from the class documentation, not a framework
  guarantee.

A separate honesty note on the reserved key `[SELFJUDGE23]`: this module attaches it to Wataoka,
Takahashi & Ri (arXiv:2410.21819), whose arXiv identifier and workshop year are **2024**, not 2023.
The verified paper is the one cited; only the outline's key suffix disagrees with the artifact's
year, and the module does not restate the year as 2023 anywhere in its prose.

**Verification note:** 🟡 *pending* — Step 7 (memory-less logic review) and Step 8 (literature survey) not yet run on this module.

**Next module:** [M14 — Ingestion at Scale, Incremental Sync, and Deletion](../14-ingestion-at-scale/README.md)
— the stage whose changes invalidate every ground-truth mapping this harness depends on.
