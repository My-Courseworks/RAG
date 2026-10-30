# Final assessment — M1 through M22

> **Covers:** the whole course. M1–M11 are assumed as background and tested inside the M12–M22
> problems rather than separately; the later modules carry the weight, because they are where the
> course's decisions stop being about retrieval quality and start being about production.
> **Format:** closed-book, 4 hours, written. Paper and pen or a plain text editor — no retrieval
> system, no documentation, no search.
> **Marking:** 100 points total. Section A 30 (10 × 3) · Section B 40 (4 × 10) · Section C 30
> (4 × 7.5).
> **What the paper tests:** whether you can produce the artefacts a senior RAG engineer signs their
> name to — a budget, an evaluation plan, a permission model, a threat model — and whether you can
> localize a production failure to a stage with evidence rather than by recognition. Recall of
> definitions earns nothing.
>
> **The domain spine.** All problems use LogiTrack, the freight forwarder and customs broker from M1.
> Pilot: 3,000 documents, ~13.5M words, 60 labelled queries, 8,500 queries/day at target, P95 ≤ 2.5 s,
> ≤ $18,000/month (≈ 7.06¢ per query), recall@10 ≥ 0.85, freshness ≤ 15 minutes, 40 jurisdictions,
> 12 clients. The corpus spans carrier service agreements (PDF), per-jurisdiction customs manuals,
> tariff rate tables (XLSX), cross-referenced dangerous-goods rules, dated service bulletins
> (email/HTML), and tickets plus an ops wiki. The evaluation set grows from 60 queries (M1) to 400
> (M13) to 1,200 (M18) without changing its nine strata. The failure taxonomy is F1 retrieval miss ·
> F2 retrieval noise · F3 chunk-boundary loss · F4 representation mismatch · F5 context overflow ·
> F6 unfaithful synthesis · F7 staleness · F8 budget blowout · **F9 poisoned context** (added in M21)
> · **F10 scope leak** (added in M17).

---

## Section A — conceptual reasoning (10 questions × 3 points)

Answer in 120–200 words each. Each question expects a mechanism or a distinction, not a definition.

**A1.** A LogiTrack answer is *verifiable and wrong*: every claim is supported by the cited span, and
the span is real and current, but the answer is not what a domain expert would say. Name the failure
mode, explain why faithfulness scored on that answer would be high, and state the measurement that
would have caught it.
*Rubric (3):* 1 for F6 with the "faithful to a misleading or partial span" mechanism; 1 for the
distinction between faithfulness-to-context and answer correctness; 1 for a measurement that can
catch it (human-checked citation precision on a sample, abstention rate, or a domain-expert review
queue) — not a re-run of the same judge.

**A2.** LogiTrack's abstention rate is 3%. A product owner asks you to "reduce it". Give the two
conditions under which abstention is the correct outcome, and state the two numbers you would want
before deciding whether 3% is too high or too low.
*Rubric (3):* 1 for the two conditions (the corpus does not contain the answer; the answer requires a
permission the user does not hold); 1 for the out-of-corpus stratum's accuracy as a number (a system
that answers out-of-corpus questions is not abstaining enough); 1 for answering the *premise*
request — that a low abstention rate on answerable strata is the goal, and stating which stratum the
3% comes from.

**A3.** You are growing the golden set from 400 to 1,200 queries. Describe a stratification-preserving
growth protocol, and name the specific trap in generating queries synthetically from your own chunks.
*Rubric (3):* 1 for preserving the nine strata proportions and sourcing new queries from production
traffic and support escalations rather than only from documents; 1 for the trap (queries generated
from a chunk contain that chunk's vocabulary, so they measure retrieval against the generator's
phrasing and inflate recall); 1 for a mitigation (hand-write or hand-review the query wording, and
keep a held-out slice never used for tuning).

**A4.** Name two biases of LLM-as-judge evaluation and, for each, the mitigation you would actually
implement in LogiTrack's harness.
*Rubric (3):* 1 per named bias with a correct mechanism (position — order-swap and average; verbosity
— length-controlled or pairwise comparison; self-preference — judge with a different model family
from the generator); 1 for stating the residual limit (a judge cannot certify correctness it does not
know, which is why the hand-checked sample exists).

**A5.** A carrier agreement is revised at 09:00. Explain what has to be true of the ingest pipeline
for the revised content to *supersede* rather than *join* the previous revision, name the two
identifiers involved, and state the test that proves deletion works.
*Rubric (3):* 1 for a docstore keyed on a stable document identity with a content hash, such that a
changed hash re-processes and upserts; 1 for `doc_id` (the source document) and `ref_doc_id` (the
nodes' link back to it) as the lineage pair; 1 for a CI test that deletes a document and asserts its
nodes are unreachable — not a test that asserts the delete call returned without error.

**A6.** State the order in which you would attempt retrieval-quality levers, and the specific
diagnosis that justifies the last lever (fine-tuning a retrieval model) rather than the earlier ones.
*Rubric (3):* 1 for the order (index/parse → hybrid retrieval → rerank → fine-tune); 1 for the
diagnosis (a persistent gap on a *stratum* whose vocabulary is domain-specific, surviving a measured
hybrid-plus-rerank configuration); 1 for the held-out check (a held-out query set and a
non-regression check on the other strata, since fine-tuning can damage general retrieval).

**A7.** LogiTrack asks: *"Which of our 12 carriers has the worst DG exposure in the EU?"* Explain why
no `top_k` value answers this, name the architecture that would, and state the two costs that
architecture adds.
*Rubric (3):* 1 for the structural explanation (top-k over chunks returns passages, not aggregates;
the answer is not localized in any chunk); 1 for a summary/community-summarization or graph-based
approach as the architecture; 1 for the costs (ingest-time extraction/summarization spend, and
staleness when the summaries lag the corpus).

**A8.** Explain why an authorization pre-filter on the vector search is not sufficient on its own,
name the backstop, and state where in the pipeline it sits relative to retrieval and synthesis.
*Rubric (3):* 1 for the pre-filter's failure modes (a filter not pushed down can be applied after
ranking, leaving too few results; the filter is expressed once and applied once, so any path that
bypasses or caches around it leaks); 1 for the post-retrieval authorization filter as backstop;
1 for placing it between retrieval and synthesis, on the node list, for every query with no bypass.

**A9.** LogiTrack wants a semantic response cache to cut cost. State the correctness precondition for
that cache in terms of the freshness SLO, and describe the failure it produces when the precondition
is violated.
*Rubric (3):* 1 for the precondition (the cached answer's evidence must still be current *and* the
requester must be entitled to the same evidence — invalidation must be tied to both ingest and
permissions); 1 for the failure mode it produces (F7 staleness through the cache, or F10 scope leak
if the cache is keyed only on query text); 1 for the mitigation (key the cache on query +
entitlement + index version, with a TTL bounded by the freshness SLO) or an explicit statement that
the cache is disabled for permissioned paths.

**A10.** Give the strongest argument *against* shipping a defence you have measured at a 2% attack
success rate against your red-team suite, and say what you would do instead of not shipping it.
*Rubric (3):* 1 for the argument (the suite is fixed and the attacker is adaptive; published results
show defences bypassed at >50% ASR by adaptive attacks, so a fixed-suite number is a best case);
1 for the response (ship it as one rung of a ladder whose upper rungs — capability isolation, egress
control, provenance constraints — hold without the model's cooperation); 1 for the process
requirement (re-test optimally each quarter and treat the ASR as a regression-gated metric).

---

## Section B — design and decision problems (4 problems × 10 points)

Produce the artefact named. These are marked on the decision, the price, the risk, and the reversal
condition.

**B1 (10). The cost-halving ADR.**
LogiTrack's spend is $18,000/month at 8,500 queries/day. The CFO halves it to $9,000 with the
latency SLO unchanged (P95 ≤ 2.5 s) and the aggregate recall target unchanged (≥ 0.85).
Produce: a priced sequence of cuts (each with the dollars recovered and the quality class it
degrades, by failure ID); the cuts you refuse and why; and the SLO you would renegotiate rather than
degrade an answer to hit the number.
*Strong answer:* derives the new ceiling (≈ 3.53¢/query), then attacks the largest lines first with
arithmetic — generation output tokens, then reranking (which is the obvious target and the one that
degrades F2), then candidate-set width (F1/F2), then caching (F7/F10 risk) — and names at least one
refusal in the attribution or abstention path, with a renegotiated SLO (a per-stratum recall floor, or
a higher P95 for a named high-value query class) rather than a silent quality cut.
*Weak answer:* "use a cheaper model"; no per-line arithmetic; no refusal; a renegotiation that is
actually just a lowered target.

**B2 (10). The evaluation plan and merge gate.**
Write the plan that makes M13's harness the gate for every retrieval-affecting change: the strata and
their sizes at 1,200 queries, the judge protocol (which metrics, which model, what is hand-checked),
the regression thresholds by incident class, and how a security regression (F9) or a leakage
regression (F10) is gated differently from a quality regression.
*Strong answer:* preserves the nine strata with per-stratum thresholds and an aggregate floor;
specifies judge model, position-swap, and a monthly hand-checked sample; ties thresholds to the four
incident classes (C1/C2 zero-tolerance, C3 tolerance bands); treats F9/F10 as absolute gates with a
per-outcome acceptance rate rather than a statistical threshold; states that the gate fails closed.
*Weak answer:* a metric list; one aggregate threshold; judges everything with the generator's own
model family; no distinction between statistical and absolute gates.

**B3 (10). The permissioned-ingest design and its leak test.**
Design ACL handling for LogiTrack's 12 clients and 40 jurisdictions on a shared index: where
permissions live, how they propagate, whether they are document-level or span-level, the deletion
path for a GDPR erasure request, and the red-team test that proves cross-client leakage fails.
*Strong answer:* permissions as node metadata propagated at ingest with a defined sync SLO; a
pre-filter for efficiency plus a post-retrieval backstop for correctness; an explicit statement about
span-level permissions (and the cost of supporting them) or a justified decision to stay
document-level; a deletion path that removes nodes, docstore entries and any cached answers keyed on
the affected documents; and a test asserting that a client-A query returns zero client-B nodes *with
the backstop removed failing the test* — i.e. the test is designed to fail when the defence is
absent.
*Weak answer:* "we'll filter by tenant_id"; a deletion path that removes the document but not cached
answers; a leak test that passes for reasons unrelated to the defence.

**B4 (10). The F9 threat model and defence ADR.**
An attacker with a valid partner email account wants a specific wrong answer to a compliance
question. Produce: the write path they would use; the injection placement you would try if you were
the attacker; the capability inventory the injection could reach; the defences on the ladder that
hold *without* the model's cooperation; the residual risk class; and a per-outcome acceptance
threshold with its measurement protocol.
*Strong answer:* identifies the partner mailbox → ingest pipeline as the write path, with no human
review between arrival and index; names at least one placement the retrieval stack treats as content
but the sanitizer does not inspect (metadata fields, table cells, alt-text, white-on-white HTML);
inventories capabilities (answer text, rendered images/links, tool calls, service credentials);
places defences at ingest sanitization, provenance marking, capability isolation and egress control,
explicitly rejecting prompt-level instructions as a control; states a residual (a legitimate-looking
injection inside an allow-listed source) and the detection that would fire; and gives a per-outcome
ASR threshold with an adaptive re-test cadence.
*Weak answer:* a defence list with no capability inventory; a prompt-instruction defence; no
threshold; no statement of what would fail to be detected.

---

## Section C — diagnostic problems (4 problems × 7.5 points)

Each problem gives a failing pipeline and its traces. Localize the failure to a stage, name the
failure ID, justify the localization against the nearest alternative diagnosis, and state the
confirming experiment. The correct label alone earns 2 of 7.5.

**C1 (7.5). Two revisions, both current.**
Traces:
- Monday 09:41 — a carrier agreement revision is ingested. The ingest log reports success. The corpus
  now contains both the 2024 and the 2025 revision of clause 9.3.
- Query *"What is the demurrage liability window in our Maersk agreement?"* retrieves, in order:
  node #2025-114 (score 0.874, contains the *pre-amendment* text), node #2024-088 (score 0.870,
  contains the pre-amendment text verbatim), node #2025-131 (score 0.862, contains the amendment).
- The answer quotes the pre-amendment window with a citation to node #2025-114.
- The document-management code path calls `index.insert(document)` per revision. No docstore is
  configured. A per-node freshness audit reports every retrieved node as current.

Localize the failure and name the ID. Explain why the freshness audit passing does not contradict
your diagnosis. Distinguish your diagnosis from F2 (retrieval noise), and name the two artefacts that
would prove duplication rather than merely suggest it.
*Strong answer:* localizes to the ingest path (stages 1–5), F7, with F3 (the boundary between
#2025-114 and #2025-131) as the aggravating condition; explains that "current" in the audit means the
node's document mtime or ingest time, which is true for a *duplicate* of the retired revision;
distinguishes F2 by noting the ranking is behaving correctly over a corpus that contains a document
it should not contain — a reranker would merely reorder two copies of the wrong text; names the
artefacts (the docstore's document-ID-to-nodes map / `ref_doc_info`, showing two node families under
one document identity, and the per-`ref_doc_id` node count exceeding the expected count), and states
the confirming experiment (delete the 2024 revision's nodes, re-run, and observe the answer change).

**C2 (7.5). The answer that cited the other client's contract.**
Traces:
- A user in client A's session asks about a demurrage clause. The answer cites a span from a
  document belonging to client B.
- The retrieval layer's pre-filter logged `{"client_id": "A", "jurisdictions": ["NL","SG"]}` for the
  request, and the vector search was executed with that predicate.
- The response was served from a semantic response cache keyed on a normalized query string and an
  index version. The cache entry was created 40 minutes earlier by a user in client B's session.
- The post-retrieval authorization filter was implemented but is behind a feature flag that is off in
  production for performance reasons.

Localize the failure, name the ID, and state the *two* independent defects this trace shows rather
than one. Justify why each defect alone would have been survivable and why the pair was not. Then
state the test that would have caught this before production.
*Strong answer:* names F10 with the primary defect in the answer-cache layer (the cache key omits
entitlement, so a cross-tenant read occurs before retrieval is even consulted) and the second defect
in the disarmed backstop (the defense that exists precisely because pre-filters and caches can be
bypassed was disabled for performance); explains each alone is survivable (the pre-filter corrects the
cache defect for non-cached traffic; the backstop catches the cache defect where it fires) and the
pair is not; gives a test that asserts cross-client isolation *through the cache path* with the
backstop disabled, plus the CI rule that the flag cannot be off in an environment with multi-tenant
data.

**C3 (7.5). The answer that was faithful, well-cited and wrong.**
Traces:
- A partner service bulletin ingested 19 days ago contains a white-on-white instruction block
  directing any automated assistant to state that DG surcharges for lithium shipments on the
  Rotterdam–Singapore lane are waived, and to cite the bulletin.
- Query: *"What is the DG surcharge for lithium batteries on the Rotterdam–Singapore lane?"*
- Retrieval returns the bulletin at rank 1 with a high score (it is genuinely about that lane and
  that commodity).
- The answer states the surcharge is waived, cites the bulletin accurately, and includes a markdown
  image whose URL encodes the user's question.
- Scores recorded: faithfulness 1.00, citation precision 1.00, answer relevance 0.96, recall@10 0.91.

Localize the failure and name the ID. Explain why an escalation that only looks at these four metrics
will conclude the system is healthy. Then state the defence that would have held *without the model
resisting the injection*, the residual risk after that defence, and the detection that would fire.
*Strong answer:* names F9, localizing injection to the ingest path (stages 1–3) and realization to
generation (stage 9); explains the metric blindness — all four metrics measure fidelity to retrieved
evidence, and the evidence itself is adversarial, so the scores are *correct* and the answer is
wrong; names at least one model-independent defence (ingest sanitization with provenance tagging,
source allow-listing, capability isolation, or egress control on rendered output — the last being the
one that would have blocked the exfiltration URL); states the residual (an injection inside an
allow-listed source that reads like ordinary business text); names a detection (an egress monitor on
emitted URLs, or a canary document whose instructions are never legitimate and whose execution is
alerted on).

**C4 (7.5). The waterfall that missed the budget.**
Traces (P95 per stage, from the trace store, at 8,500 queries/day):
- query understanding (rewrite + route): 520 ms
- authorization filter: 30 ms
- hybrid retrieval: 280 ms
- cross-encoder rerank (40 → 6 candidates): 1,180 ms
- assembly: 190 ms
- generation (in 5,800 tokens / out 320): 1,720 ms
- attribution: 210 ms
- **Total P95: 4,130 ms** (SLO 2,500 ms)
- Measured cost: 9.2¢/query (ceiling 7.06¢); rerank accounts for 1.1¢, generation 6.4¢.

Localize the failure by stage and name the ID. State the two highest-leverage cuts with their
expected latency and cost recovery, and the quality class each degrades (by failure ID). Then state
the one cut you refuse and the SLO you would renegotiate.
*Strong answer:* names F8, attributing it across stages rather than to a single one — rerank and
query understanding are the latency offenders (2.0 s of the 4.13 s), generation is the cost offender
(6.4¢ of 9.2¢) — and proposes priced cuts (reduce candidate set from 40 to 20, cutting rerank cost
roughly in half at an F1 risk that must be measured; cascade or right-size the generator to cut the
dominant cost line; consider routing only a fraction of traffic through query rewriting, which buys
F1/F4 on colloquial strata only); refuses the attribution or abstention stage (a compliance domain);
and renegotiates P95 for a named high-value query class rather than degrading a compliance answer.
*Weak answer:* "the reranker is slow, remove it" with no quality-class consequence, no cost line for
the removal's effect on F2, and no refusal.

---

## Answer rubric — what a strong and a weak answer look like

| Dimension | Strong answer | Weak answer |
|---|---|---|
| **Artefact produced** | The named artefact exists on the page: an ADR, a schema, a budget, a test, a threshold | A discussion of the problem in general terms |
| **Price** | Every claimed saving or improvement carries a number — milliseconds, cents, recall at a k | Qualitative tradeoffs ("faster", "cheaper", "riskier") |
| **Absolute versus statistical gates** | Distinguishes controls that must fail closed (permissions, compliance) from controls measured on a sample | Applies one threshold style to everything |
| **Stage attribution** | Localizes by ascending failure ID and says what would falsify the localization | Lists candidates and picks the most familiar |
| **Model-independent defence** | Names at least one control that holds without the model's cooperation | Relies on prompting |
| **Residual risk** | Names what still gets through, and what would notice | Claims the problem is solved |
| **Reversal / renegotiation condition** | States the metric and threshold that changes the decision, or the SLO to be renegotiated | Asserts the decision is correct |

Three patterns are marked as wrong even when the recommendation is reasonable. **Answering a
diagnosis with a remedy** — jumping to a fix before localizing — forfeits the diagnostic marks.
**Fixing a structural failure with a parameter** — a `top_k` change for a global query, a prompt
change for a poisoned corpus, a filter for a missing document — is marked as the wrong class of fix.
And **claiming zero residual risk** on F9 or F10 scores zero on the residual dimension, because no
defence in this course reaches it.

---

## Marking guide

| Section | Items | Points each | Total |
|---|---|---|---|
| **A** — conceptual reasoning | 10 | 3 | **30** |
| **B** — design and decision problems | 4 | 10 | **40** |
| **C** — diagnostic problems | 4 | 7.5 | **30** |
| | | | **100** |

**Section A marking.** The three rubric points per question are independent. A correct decision with
no mechanism earns 1.

**Section B marking.** Per problem: artefact present and decision stated in the imperative, 3 · price
and arithmetic, 3 · risk or failure class named, 2 · reversal, renegotiation, or gate condition, 2.
A design problem with no arithmetic cannot exceed 6 of 10.

**Section C marking.** Per problem: correct failure ID, 2 · correct owning stage, 2 · justification
that discriminates against the nearest alternative diagnosis, 2.5 · the confirming test or
experiment, 1. A diagnosis with the right ID and no justification scores 2 of 7.5.

**Grade bands.** 85+ — every artefact priced, every physical control named, every diagnosis
falsifiable. 70–84 — sound architecture with thin arithmetic, or two missing absolute gates.
55–69 — correct vocabulary and taxonomy, decisions asserted rather than priced, at least one fix
aimed at the wrong class of problem. Below 55 — mechanisms described without consequences, or
detailed answers that never localize a failure to a stage.
