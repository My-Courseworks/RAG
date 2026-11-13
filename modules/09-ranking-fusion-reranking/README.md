# M9 · Ranking, Fusion, and Reranking

> **Module question:** How do you reorder a candidate set so the top-k is actually the best-k?
> **Cross-cutting threads:** Failure modes · Tradeoff ledger · LlamaIndex at a glance
> **Domain spine:** LogiTrack — 40 candidates retrieved, 6 that fit a 4,000-token answer budget
> **Prerequisites:** [M7 — Sparse, Dense, and Hybrid Retrieval](../07-sparse-dense-hybrid/README.md) · [M3 — Corpus Acquisition, Parsing, and the Chunking Decision](../03-parsing-and-chunking/README.md)

---

## Opening scene — the chunk that was ranked ninth

Return to M3's opening scene, because M9 is where it actually gets fixed.

The demurrage clause reads, verbatim: **"Demurrage liability shall accrue from the fourth (4th)
calendar day following discharge, and shall be extinguished on the fourteenth (14th) calendar day,
after which the carrier's liability for delay shall cease."** The chunk boundary fell between
"extinguished on the fourteenth (14th) calendar day" and "after which the carrier's liability for
delay shall cease." The retriever returned the first chunk. The generator, holding a sentence that
ends without a terminator, treated *fourteen* as the liability **window** rather than its
**endpoint**.

M3 wrote: "The second half — the sentence that would have disambiguated — was in the index, ranked
9th, below the top-k cut."

Stop on that. The evidence **was retrieved**. It was not a retrieval miss (F1). The candidate set
was correct, and the ordering was wrong: the disambiguating chunk sat at rank 9 while the
misleading one sat at rank 1, and the application's top-k was 6. Every component behaved correctly
and the answer was still wrong, because **the number that decides which chunks reach the model is
a cut, and the cut is a ranking decision nobody made deliberately.**

This is the distinction the whole module turns on. M3's fix — better chunking — reduces how often
meaning gets severed. M9's fix is orthogonal and much cheaper: **decide how many candidates to
retrieve, decide how to score them, and decide where in the prompt they land.** Rank 9 → rank 1 is
a reordering problem, and reordering a candidate set you already paid for costs a fraction of
re-chunking and re-embedding a corpus.

> **Failure mode (the module in one line):** treating the top-k cut as a *retrieval parameter*
> rather than as the **ranking decision that determines what the model is allowed to reason about**
> — so a correct chunk ranked 9th is as invisible as a chunk that was never retrieved, and the
> system reports the wrong number with a real citation attached.

---

## The retriever/reranker split

The framework names the split cleanly: "A reranker takes the top-`k` nodes returned by an initial
retriever (usually a dense vector search, keyword search, or a hybrid of the two) and re-orders
them based on a stronger, and typically slower, measure of query-document relevance. Reranking is
one of the highest-leverage knobs in a RAG pipeline: **initial retrievers are tuned for recall over
a large corpus, and a reranker lets you recover precision before nodes reach the LLM.**" [[LI-RERANK]](#src-li-rerank)

That is the whole contract. Two quantities, two stages, two different optimizations:

| Stage | Optimizing | Cost per query | Failure if wrong |
|---|---|---|---|
| **Retriever** (→ M5, M7) | recall@N — get the answer *into* the candidate set | one ANN or BM25 lookup, ~10–100 ms | F1 · retrieval miss — invisible, unrecoverable |
| **Fusion** (→ M7) | a merged candidate set from heterogeneous scorers | arithmetic, negligible | candidate ranking noise |
| **Reranker** (this module) | precision@k — get the answer *into the top k* | N scorer evaluations, linear in N | F2 · retrieval noise — a confident wrong answer |
| **Reorder** (this module) | *positional* quality within the final set | arithmetic, negligible | F5 · context overflow, mid-prompt evidence loss |
| **Budget cut** (→ M11) | fit the evidence into the answer budget | truncation | silent evidence loss at the end |

```mermaid
flowchart LR
    A[Retriever\nrecall@N, N≥40] --> B[Fusion\nM7]
    B --> C[Reranker\nprecision@n]
    C --> D[Reorder\nposition within k]
    D --> E[Budget cut\n→ M11]
    C -.->|"drop here and F2 survives"| F[Nodes reach the LLM]
```

Two properties of this pipeline decide every parameter in it.

**The reranker is a Pareto instrument, not a quality instrument.** It cannot add a document the
retriever missed. It can only reorder what is present. That is why the candidate-set size N is
M9's most consequential number and why N is set by the *recall* curve (M5's territory), not by the
reranker's preference. If recall@40 is 0.94 and recall@6 is 0.71, then N=40 buys you 0.23 of
headroom, and the reranker's job is to convert it. If recall@40 is 0.74, no reranker gets you to
0.94, and M9 is the wrong module to be reading.

**Reranker cost is linear in N; recall gain is not.** At N=10 → 40, a cross-encoder does 4× the
work. The recall curve above that range flattens. So there is an interior optimum for N, set by
the ratio of your latency budget (→ M18) to the flat part of your recall curve — and it is a
measurable number, not a default.

The framework exposes the mechanism as a **node postprocessor**, and the placement is explicit:
node postprocessors "are most commonly applied within a query engine, after the node retrieval step
and before the response synthesis step", and they take a list of `NodeWithScore` objects and
"transform, filter, or re-order them before the LLM sees them" [[LI-NP](#src-li-np), [LI-NPP](#src-li-npp)]. The base class is
`BaseNodePostprocessor`, whose single abstract method is `_postprocess_nodes(self, nodes:
List[NodeWithScore], query_bundle: Optional[QueryBundle]) -> List[NodeWithScore]` [[LI-NP]](#src-li-np). That
signature is the module's entire API surface: everything M9 discusses is a function from a list of
scored nodes to a list of scored nodes.

Two operational facts from that signature. First, `postprocess_nodes` "can take in either a
`query_str` or `query_bundle` (`QueryBundle`), though not both" [[LI-NP]](#src-li-np) — which means the position
of a postprocessor in the list matters, because a postprocessor that rewrites the query (→ M10)
and one that scores against the query are order-dependent. Second, postprocessors **compose in
list order**, and "all node post-processors will be applied during each query" [[LI-NP]](#src-li-np). A
reranker followed by a similarity cutoff is a different pipeline from a similarity cutoff followed
by a reranker — in the second case the cutoff filters on *retrieval* scores and the reranker never
sees the nodes it would have rescued.

> **LlamaIndex at a glance:** reranking is configured in one kwarg on the query engine —
> `index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])` — where "the
> retriever fetches a wider set (`similarity_top_k=10`) and the reranker prunes it down to the best
> `top_n=3` before the LLM sees them" [[LI-RERANK]](#src-li-rerank). The two numbers, `similarity_top_k` and
> `top_n`, are the candidate-set decision and the precision decision respectively, and they are
> the only two knobs in this module that appear in a typical config.

> **Tradeoff (the ledger entry M9-1).** *Decision:* retrieve a wide candidate set (N ≈ 40–100) and
> rerank down to a small top-n, rather than retrieving exactly the n chunks you will send.
> *Costs:* reranker latency linear in N, plus a reranker model to host or pay for per query; if the
> reranker is an LLM, this dominates the latency budget (→ M18). *Accepted because:* recall is
> bounded upstream and precision is fixable downstream — a candidate that was never retrieved
> cannot be recovered by any downstream stage, while a candidate that was retrieved badly ranked
> can. *Reverses if:* your recall@n is already ≈ recall@N (the retrieval is easy), in which case
> the reranker reorders a set whose top-n was already correct and you are paying latency for
> nothing.

---

## Cross-encoders: the scoring function that cannot be precomputed

The cross-encoder is the reference reranker, and its defining property is an architectural one.

Nogueira & Cho describe it as "a simple re-implementation of BERT for query-based passage
re-ranking", and report that their system "is the state of the art on the TREC-CAR dataset and the
top entry in the leaderboard of the MS MARCO passage retrieval task, outperforming the previous
state of the art by 27% (relative) in MRR@10" [[CE19]](#src-ce19). The 27% relative gain is the reason reranking
became standard practice; it is also a number from 2019, on MS MARCO, against a 2019 baseline.

The mechanism is why it is slow and why it is good. A **bi-encoder** (→ M4) embeds query and
document independently, so document vectors can be computed offline and compared with one dot
product. A **cross-encoder** feeds the query and the passage through the transformer *together*,
with cross-attention between them, and emits a single relevance score. There is no document vector.
Nothing can be precomputed. Scoring N candidates costs N full forward passes, and the score cannot
be reused for a different query.

That is the entire trade. The model sees query and document jointly — which is exactly the signal a
bi-encoder's fixed-size bottleneck destroyed — and pays for it with per-query, per-candidate
compute. It is why reranking is the classic *late* stage: run the cheap, precomputable, recall-tuned
stage over millions of documents, then the expensive, precise stage over tens.

The framework's reranker stack in practice:

| Postprocessor | Backing model | Documented positioning | Watch out for |
|---|---|---|---|
| `SentenceTransformerRerank` | cross-encoder via `sentence-transformers` | "No API key, local, recommended default" [[LI-NPP]](#src-li-npp) | **The constructor's default `model` is `cross-encoder/stsb-distilroberta-base`**, which "was trained for semantic textual similarity rather than passage re-ranking"; pass an MS MARCO model or you are using the wrong scorer [[LI-NPP]](#src-li-npp) |
| `CohereRerank` / `JinaRerank` / `VoyageAIRerank` / `MixedbreadAIRerank` | hosted | "Hosted API, minimal setup" [[LI-NPP]](#src-li-npp) | per-query cost and a network hop in the latency budget |
| `LLMRerank` / `RankGPTRerank` / `RankLLMRerank` | an LLM judges relevance | "Highest quality, latency tolerant" [[LI-NPP]](#src-li-npp) | "latency tolerant" is a load-bearing phrase (→ M18) |
| `ColbertRerank` | ColBERT v2 late interaction | "Token-level late interaction … useful for narrow technical terminology" [[LI-NPP]](#src-li-npp) | a second index/storage story (below) |
| `ColPaliRerank` | vision-language over document images | multimodal | out of scope for this course (§1) |

The default-model warning deserves emphasis because it is the single most common way to ship a
reranker that does nothing. `SentenceTransformerRerank(model="cross-encoder/stsb-distilroberta-base",
top_n=3)` is a syntactically perfect, semantically wrong configuration: `stsb-distilroberta-base` was
trained on semantic textual similarity, a symmetric sentence-pair task, not on the asymmetric
query-passage relevance task you are performing. The framework's recommended alternatives are
`cross-encoder/ms-marco-MiniLM-L6-v2` for speed and `Qwen/Qwen3-Reranker-0.6B` for "stronger
multilingual quality" [[LI-NPP]](#src-li-npp). Note also that the two doc pages in this module show different
MiniLM variants — the node-postprocessor page's example uses `cross-encoder/ms-marco-MiniLM-L2-v2`
while the rerankers guide uses `cross-encoder/ms-marco-MiniLM-L6-v2` — and the page itself flags the
relationship ("shown above is the smaller `L2` variant") [[LI-NPP](#src-li-npp), [LI-RERANK](#src-li-rerank)]. Neither is wrong;
neither is a default you should inherit without knowing which you got.

> **Multimodal inputs** ($\color{red}\text{to be discussed}$) — a text reranker scores what parsing made of the page, not the page.
> - A rate table or a scanned, stamped agreement page reaches the reranker as serialized or OCR text from [M3](../03-parsing-and-chunking/README.md); bad serialization means bad scores.
> - MS MARCO rerankers learned from Bing web passages ([MS MARCO](https://arxiv.org/abs/1611.09268)); table rows and OCR text are off that distribution, so test them.
> - `ms-marco-MiniLM-L6-v2` caps query + passage at 512 tokens ([tokenizer config](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/main/tokenizer_config.json)); token-heavy tables can be cut before the answering row.
> - A figure is ranked by its caption or vision-model description; the reranker cannot check that text against the figure.
> - Page-image reranking (`ColPaliRerank`) stays deferred ([syllabus §1](../../syllabus.md#1-scope-knobs-frozen)).

The cross-encoder family has moved past BERT classification. Nogueira, Jiang & Lin show that a
pretrained sequence-to-sequence model can be adapted to ranking by generating relevance labels as
"target words" and "interpreting the underlying logits of these target words … as relevance
probabilities", reporting results "at least on par with previous classification-based models" that
"can surpass them with larger, more-recent models" — and, importantly for domain work,
"significantly outperform[ing] an encoder-only model in a data-poor regime" [[MONOT5-20]](#src-monot5-20). That last
finding is the one that should interest you if your corpus is 3,000 documents and your labels are
60 queries: the data-poor regime is your regime, and **M15** owns what to do about it.

> **LlamaIndex at a glance:** `SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L6-v2",
> top_n=3)` then `postprocessor.postprocess_nodes(nodes)` — or, in a query engine, the same object
> in `node_postprocessors` [[LI-NPP]](#src-li-npp). The two required knobs are `model` (get this right; the default
> is the wrong task) and `top_n` (this is your precision decision, and it is also your budget
> decision because `top_n` nodes are what M11 will assemble).

> **Tradeoff (the ledger entry M9-2).** *Decision:* pay N cross-encoder forward passes per query to
> recover precision from a recall-optimized candidate set. *Costs:* a GPU-hosted model or a paid
> API in the critical path, with cost and latency linear in N — the only stage in the pipeline
> whose cost the *query* controls rather than the *corpus*. *Accepted because:* it is the cheapest
> available intervention that changes what the model sees, it requires no re-indexing, no
> re-embedding, and no corpus migration, and it can be added and removed by a config change.
> *Reverses if:* the candidate set's top-n ordering was already correct (measure first: if
> nDCG@10 is flat with and without the reranker, you bought latency for nothing), or if N must grow
> so large to capture recall that N forward passes no longer fit the latency budget.

---

## Late interaction: paying for tokens instead of pairs

ColBERT is the design that refuses the bi-encoder/cross-encoder dichotomy. Khattab & Zaharia
introduce "a late interaction architecture that independently encodes the query and the document
using BERT and then employs a cheap yet powerful interaction step that models their fine-grained
similarity. By delaying and yet retaining this fine-granular interaction, ColBERT can leverage the
expressiveness of deep LMs while simultaneously gaining the ability to pre-compute document
representations offline." The reported result is that "ColBERT's effectiveness is competitive with
existing BERT-based models (and outperforms every non-BERT baseline), while executing two
orders-of-magnitude faster and requiring four orders-of-magnitude fewer FLOPs per query" [[COLBERT20]](#src-colbert20).

The mechanism is the point: instead of one vector per document, ColBERT keeps **one vector per
token**, precomputes the document's token matrix offline, and at query time computes a
token-level maximum-similarity aggregation (MaxSim) between query tokens and document tokens. So
the document side *is* precomputable — restoring the bi-encoder's economics — while the query-document
interaction is retained, restoring a measure of the cross-encoder's precision. The cost moves from
query-time compute to **storage**: one vector per token is orders of magnitude more index bytes
than one vector per chunk, which is a storage-and-memory decision (→ M5, M18), not a latency one.

In the reference stack this is a reranker: `ColbertRerank(top_n=5, model="colbert-ir/colbertv2.0",
tokenizer="colbert-ir/colbertv2.0", keep_retrieval_score=True)` — documented as using "Colbert V2
model as a reranker to rerank documents according to the fine-grained similarity between query
tokens and passage tokens", and recommended "when the query and document contain narrow technical
terms" [[LI-NPP]](#src-li-npp). That last clause is a precise fit for this course's domain: `UN3480`, clause
identifiers, tariff codes, and jurisdiction abbreviations are exactly the "narrow technical
terminology" where token-level matching beats a pooled sentence embedding.

Note the `keep_retrieval_score=True` flag. It is a small API detail with a real consequence: the
reranker overwrites node scores with its own relevance scores, and preserving the retriever's score
alongside is what lets you reason about the two stages separately later (→ M19's tracing). Without
it, your logs show one score and you cannot tell whether a bad answer came from a bad candidate or
from a bad reordering.

> **Tradeoff (the ledger entry M9-3).** *Decision:* use late interaction (token-level vectors)
> rather than a cross-encoder when the corpus's retrieval difficulty is concentrated in narrow
> technical terminology. *Costs:* token-level index storage — orders of magnitude more vectors than
> chunk-level embeddings — plus a second artifact to build, persist, and keep in sync with the
> corpus (→ M14). *Accepted because:* the document side is precomputable offline, so the query-time
> cost stays inside a latency budget that a cross-encoder's N forward passes may not fit; and the
> precision gain is concentrated exactly where dense bi-encoders lose (rare tokens). *Reverses if:*
> the storage bill exceeds the latency bill you were avoiding — at which point a smaller
> cross-encoder over the same candidate set is the better trade, and the measurement is bytes
> against milliseconds (→ M18).

---

## LLM rerankers: the quality ceiling and its price

The fourth reranker family replaces the scorer with a generative model. The framework positions
these unambiguously: `LLMRerank`, `RankGPTRerank`, and `RankLLMRerank` are for "highest quality,
latency is not critical" [[LI-RERANK]](#src-li-rerank), and "these use an LLM itself to judge relevance" [[LI-NPP]](#src-li-npp).
`LLMRerank` is documented as asking "the LLM to return the relevant documents and a score of how
relevant they are" and returning "the top N ranked nodes" [[LI-NPP]](#src-li-npp). `RankGPTRerank` is a
RankGPT-agent wrapper, and `RankLLMRerank(model="rank_zephyr", top_n=5)` uses the `rank_llm`
package's instruction-tuned listwise rerankers [[LI-NPP]](#src-li-npp).

The claim that motivates them comes from Sun et al.: "our experiments reveal that properly
instructed LLMs can deliver competitive, even superior results to state-of-the-art supervised
methods on popular IR benchmarks", and they further report that a "distilled 440M model outperforms
a 3B supervised model on the BEIR benchmark" [[RANKGPT23]](#src-rankgpt23). That paper also does the honest thing the
literature usually skips: it builds **NovelEval**, a test set "based on the latest knowledge and
aiming to verify the model's ability to rank unknown knowledge", explicitly "to address concerns
about data contamination of LLMs" [[RANKGPT23]](#src-rankgpt23). Read that as a warning about every LLM-reranker
benchmark number you encounter, including this one.

Two structural costs distinguish LLM rerankers from cross-encoders, and neither is latency:

1. **Listwise prompting changes the cost model.** A cross-encoder scores N candidates
   independently and can be batched with no loss. A listwise LLM reranker (RankGPT, RankZephyr)
   must put *multiple candidates into one prompt* to compare them, which makes the prompt grow with
   N and forces sliding-window schemes when N is large. The candidate-set size decision and the
   reranker's prompt budget become the same decision.
2. **The reranker becomes a system with an instruction surface.** An LLM reranker is steered by a
   prompt, and that prompt is text in the same context window as the documents it is ranking. A
   retrieved document can therefore influence its own ranking — the instruction/data boundary that
   **M21** owns, arriving inside the ranking stage rather than the generation stage.

> **LlamaIndex at a glance:** `LLMRerank(top_n=2, service_context=service_context)` then
> `postprocessor.postprocess_nodes(nodes)` — the documented signature, with `top_n` as the only
> required decision [[LI-NPP]](#src-li-npp). `RankGPTRerank(top_n=3, llm=OpenAI(model="gpt-4o-mini"))` and
> `RankLLMRerank(model="rank_zephyr", top_n=5)` are the documented alternatives [[LI-NPP]](#src-li-npp). No
> framework doc page in this module states a latency figure for any of them; "latency tolerant"
> [[LI-RERANK]](#src-li-rerank) is the entire guidance, and **that is a documentation gap you must close with your
> own measurement** before putting one in a 2.5 s budget.

> **Tradeoff (the ledger entry M9-4).** *Decision:* use an LLM reranker when the ranking judgement
> requires reasoning that a cross-encoder cannot perform — comparing candidates against each other,
> or applying a criterion stated in the query. *Costs:* the largest per-query latency of any stage
> except synthesis, a prompt that grows with N, an instruction surface a retrieved document can
> influence (→ M21), and a second context window to budget (→ M18). *Accepted because:* a
> cross-encoder scores each candidate in isolation against the query, so it cannot express
> *comparative* or *criterion-conditioned* relevance at all; if your queries say "which of these
> supersedes the other", no pairwise scorer answers it. *Reverses if:* a cross-encoder measured on
> your own data lands within noise of the LLM reranker — the disagreement below suggests this
> happens more often than the positioning implies.

---

## Diversity, position, and the two ordering problems

Reranking optimizes *relevance ordering*. Two other ordering problems must be solved separately,
and both have dedicated primitives.

### Diversity: MMR, and the score-scale trap

Maximal marginal relevance was introduced by Carbonell & Goldstein for "reordering documents" in
the context of producing summaries, and it does one thing: select the next item by combining its
relevance to the query with its *dissimilarity* to items already selected [[MMR98]](#src-mmr98). The framework
documents the algorithm it implements verbatim:

$$\text{MMR} = \arg\max_{d_i \in D \setminus R}\left[\lambda \cdot Sim_1(d_i, q) - (1-\lambda) \cdot \max_{d_j \in R} Sim_2(d_i, d_j)\right]$$

"Here, D is the set of all candidate documents, R is the set of already selected documents, q is
the query, $Sim_1$ is the similarity function between a document and the query, and $Sim_2$ is the
similarity function between two documents." The parameter λ is exposed as `mmr_threshold`: "If
mmr_threshold is close to 1, more emphasis is put on relevance, while a mmr_threshold close to 0
puts more emphasis on diversity." [[LI-MMR]](#src-li-mmr)

**Where MMR lives in the reference stack is not where the course syllabus says it lives.** There is
no `MMRPostprocessor`: a full-text search of the framework documentation for that identifier returns
**zero results**, and `postprocessor.mmr` appears in neither the node-postprocessor module list nor
the rerankers guide [[LI-NPP](#src-li-npp), [LI-RERANK](#src-li-rerank)]. MMR is exposed as a **vector-store query mode**:
`index.as_query_engine(vector_store_query_mode="mmr")`, with the λ parameter passed through
`vector_store_kwargs={"mmr_threshold": 0.2}` [[LI-MMR]](#src-li-mmr). That placement is architecturally
significant — MMR is a *retrieval* mode, so it competes with top-k retrieval rather than composing
after a reranker, and you cannot put a cross-encoder in front of it and then apply MMR to the
reranked set without dropping to the vector-store layer yourself.

The trap, documented in the same page and easy to miss: "the node score will be scaled with the
threshold and will additionally be penalized for the similarity to previous nodes. As the threshold
goes to 1, the scores will become equal and similarity to previous nodes will be ignored, turning
off the impact of MMR." [[LI-MMR]](#src-li-mmr) The page's own numbers show how violent the rescaling is — the
same query, same corpus, three configurations:

| Configuration | Node scores |
|---|---|
| No MMR | `[0.3877, 0.3816]` |
| MMR, `mmr_threshold = 0.8` | `[0.0775, -0.3161]` |
| MMR, `mmr_threshold = 0.2` | `[0.3102, 0.1845]` |

Source: the MMR retrieval notebook's score printout [[LI-MMR]](#src-li-mmr). Note the **negative** scores. Every
downstream threshold you calibrated on a non-MMR pipeline — M4's near-duplicate threshold, a
`SimilarityPostprocessor(similarity_cutoff=0.75)` [[LI-NP]](#src-li-np), a relevance gate in your evaluation
harness — is now operating on a different scale, and `similarity_cutoff=0.75` applied to
`[-0.3161, ...]` silently discards **everything**. Turning on MMR is a change to your scoring
function's units, and nothing in the API says so.

### Position: `LongContextReorder` and the lost-in-the-middle effect

The last ordering problem is not *which* nodes make the cut but *where in the prompt* they land.
The framework states the effect and cites the source: "Models struggle to access significant details
found in the center of extended contexts. A study observed that the best performance typically
arises when crucial data is positioned at the start or conclusion of the input context.
Additionally, as the input context lengthens, performance drops notably, even in models designed for
long contexts." [[LI-NPP]](#src-li-npp) That study is **M9's claim to own**, and the primary source is Liu, Lin,
Hewitt, Paranjape, Bevilacqua, Petroni & Liang, who "analyze the performance of language models on
two tasks that require identifying relevant information in their input contexts: multi-document
question answering and key-value retrieval", and report the finding directly: "we observe that
performance is often highest when relevant information occurs at the beginning or end of the input
context, and significantly degrades when models must access relevant information in the middle of
long contexts, even for explicitly long-context models." Their conclusion is broader than a
prompt-formatting tip: performance degradation "indicat[es] that current language models do not
robustly make use of information in long input contexts." [[LITM23]](#src-litm23)

`LongContextReorder` is the mechanical response: "This module will re-order the retrieved nodes,
which can be helpful in cases where a large top-k is needed." [[LI-NPP]](#src-li-npp) It exists because a
reranker produces a *ranked list*, and a naive assembler pastes that list in rank order — putting
rank 1 first and rank n last, with everything important-between in the middle where the literature
says it is least accessible. The fix is to place the highest-ranked evidence at the *ends* of the
assembled context. **M11 owns the assembly-time application** of this finding and does not
re-survey the paper; M9's job is to establish the effect and to note that it makes the *final*
ordering of the top-n a separate, cheap, and separately-testable decision.

> **LlamaIndex at a glance:** `LongContextReorder()` takes no arguments and is applied as
> `node_postprocessors=[LongContextReorder()]` [[LI-NPP]](#src-li-npp). Its placement in the postprocessor list is
> the design decision: it must run *after* the reranker (or it reorders an unranked list) and
> *before* synthesis. Ordering the list as `[reranker, LongContextReorder]` is the configuration;
> ordering it as `[LongContextReorder, reranker]` is a no-op that looks like an improvement in a
> diff.

> **Tradeoff (the ledger entry M9-5).** *Decision:* spend the top-n slots on distinctive evidence
> (MMR) and place the strongest evidence at the context's edges (`LongContextReorder`), rather than
> filling the budget with the top-n most similar chunks in rank order. *Costs:* MMR changes the
> node score scale so every downstream threshold must be re-calibrated [[LI-MMR]](#src-li-mmr), and reordering
> separates prompt position from rank, so a rank-correlated metric (nDCG@k) no longer describes
> what the model actually saw. *Accepted because:* five near-duplicate chunks occupying five of six
> slots is a slot-allocation failure, and mid-prompt evidence is measurably harder for the model to
> use [[LITM23]](#src-litm23) — both are losses that no amount of retrieval quality recovers. *Reverses if:*
> measurement shows your chunks are already diverse (then MMR costs a re-calibration for nothing)
> or that your model is position-insensitive on your prompts (the honest negative result — see
> below).

---

## Fusing across stages, not across retrievers

The module title says "Fusion", so the boundary with M7 needs stating precisely.

**M7 owns fusing retrieval scores**, and does not re-survey it here: reciprocal rank fusion and
weighted score fusion across BM25, dense, and learned-sparse retrievers are M7's claim, and the
score-incomparability problem between those scorers is M7's argument. If you have not read M7, the
one-sentence version is that scores from different retrievers are not on a common scale, which is
why rank-based fusion is more robust than score-based fusion.

**M9 owns fusing *stages*** — and it is a different problem with the same shape. After a reranker
runs, you hold two scores per node: the retriever's score and the reranker's score. They are not
comparable, they were produced by different functions, and the reranker has *overwritten* the first
unless you asked it not to (`keep_retrieval_score=True` on `ColbertRerank` [[LI-NPP]](#src-li-npp) preserves it).
Three defensible policies:

| Policy | Mechanism | Consequence |
|---|---|---|
| **Replace** | the reranker's score becomes the node score | Simple; the retriever's signal is gone, so you cannot diagnose a bad candidate set from a bad reordering (→ M19) |
| **Preserve both** | keep the retrieval score in metadata, rerank score in `score` | Diagnosable; requires discipline not to confuse the two fields later |
| **Blend** | weighted sum of normalized retrieval and rerank scores | Recovers the retriever's signal — and reintroduces exactly the score-incomparability problem M7 warns about, now between two stages |

The honest default is **preserve both**. It costs one metadata field and it is the only policy that
keeps M19's stage-localization question answerable.

---

## Worked example: 40 candidates → 6 for a 4,000-token answer budget

LogiTrack's answer budget is 4,000 tokens of evidence (→ M11), and at M3's `SentenceSplitter(512,
50)` that is roughly **6 chunks** of ~600 tokens each. The known-item subset of the 180-query set
gives the candidate-set curve and the reranker comparison on one page:

| N (`similarity_top_k`) | recall@N | nDCG@10 | Reranker wall-clock |
|---|---|---|---|
| 6 (the M3 baseline cut) | 0.71 | 0.58 | — |
| 20 | 0.86 | 0.58 | — |
| **40** | **0.94** | 0.58 | — |
| 100 | 0.95 | 0.58 | — |

`nDCG@10` is flat because it is computed on the retrieved list, which the reranker has not touched
yet. That flatness is the point: **N buys recall, and nothing else.** Then the same N=40 candidate
set through five configurations, with the same 6-slot output budget:

| Configuration | nDCG@10 | Answer accuracy | P50 added latency | P95 added latency | Cost/query |
|---|---|---|---|---|---|
| No reranker (top-6 of 40) | 0.58 | 0.71 | — | — | $0 |
| `SentenceTransformerRerank` (`ms-marco-MiniLM-L6-v2`) | **0.79** | **0.88** | 74 ms | 118 ms | self-hosted |
| `SentenceTransformerRerank` (`Qwen3-Reranker-0.6B`) | 0.82 | 0.90 | 310 ms | 470 ms | self-hosted |
| `ColbertRerank` (`colbertv2.0`) | 0.81 | 0.89 | 96 ms | 152 ms | +2.1 GB index |
| `CohereRerank` (`rerank-v3.5`) | 0.80 | 0.89 | 180 ms | 260 ms | per-query API |
| `LLMRerank` (`gpt-4o-mini`, `top_n=6`) | 0.84 | **0.92** | 1,340 ms | 2,180 ms | per-query API, high |

Read the four findings that decide LogiTrack's architecture:

1. **The cheapest reranker captures most of the gain.** `ms-marco-MiniLM-L6-v2` moves nDCG 0.58 →
   0.79 and accuracy 0.71 → 0.88 for 74 ms of P50. The `Qwen3` cross-encoder and the LLM reranker add
   0.03 and 0.05 nDCG respectively and cost 4× and 18× the latency. This is the shape to expect and
   the reason entry M9-2's revert condition is a measurement rather than a preference.
2. **The LLM reranker does not fit the budget.** M1 fixed P95 end-to-end at 2.5 s. At 2,180 ms P95
   *for the ranking stage alone*, `LLMRerank` consumes 87% of the entire end-to-end budget before
   retrieval, synthesis, or network overhead. This is not a tuning problem; it is an architecture
   exclusion, and it is why the framework's "latency tolerant" [[LI-RERANK]](#src-li-rerank) is a warning rather than a
   feature.
3. **N=40 is the knee, and N=100 is not.** recall@40 = 0.94 and recall@100 = 0.95: 60% more
   candidate-set compute (and 2.5× the reranker bill) for 0.01 of recall. The candidate-set
   decision has an interior optimum and it is set by the recall curve's flattening point, not by the
   reranker's preference.
4. **ColBERT's cost is storage, not latency.** 96 ms P50 — faster than the Cohere API — bought with
   2.1 GB of token-level vectors. The decision is a bytes-against-milliseconds trade (→ M18), and
   at LogiTrack's scale it is the wrong side of the trade; at [480k documents](../../syllabus.md#corpus-scale) (M14) it becomes a
   different question.

**LogiTrack's decision after M9:** `SentenceTransformerRerank` with an MS MARCO cross-encoder,
`similarity_top_k=40` → `top_n=6`, plus `LongContextReorder` after the reranker, with the retrieval
score preserved in node metadata. `LLMRerank` is deferred and explicitly out of the 2.5 s budget;
`ColbertRerank` is deferred pending the storage question at M14's scale; MMR is deferred until a
duplicate-chunk measurement shows it is needed — because the score-rescaling trap means enabling it
invalidates every threshold calibrated so far.

> **Multimodal inputs** ($\color{red}\text{to be discussed}$) — the averages above can hide a table and scan gap.
> - Slice the reranker comparison by source format: prose PDFs vs rate tables (XLSX/HTML) vs scanned pages.
> - If table queries lag, fix serialization in [M3](../03-parsing-and-chunking/README.md) before swapping rerankers.
> - Per-slice metrics and their gates belong to [M13](../13-evaluation/README.md).

---

## Design exercise

> *Paper-based. Think, then write.*

**Task.** Write the ranking ADR for a retrieval system you own — or use LogiTrack's N=40 → n=6
configuration if you have none. Deliver a decision with a latency number per stage, not a
comparison of reranker families.

1. **State the two budgets you are dividing.** The candidate-set size N and the answer budget in
   tokens. Compute n from the budget and your chunk size; do not choose n by taste. If your N/n
   ratio is below ~3, you have not bought a reranker, you have bought an expensive sort.
2. **Plot recall@N and find the knee.** For N ∈ {6, 20, 40, 100}, measure recall@N on your labelled
   set. Choose N at the point where the curve flattens — not at the largest value your store
   tolerates. If the curve has not flattened by your latency limit, your problem is the retriever
   (→ M5, M7), not the reranker, and M9 cannot fix it.
3. **Choose the scorer and justify it by *what signal the query needs*.** Independence-scored
   pairwise relevance → cross-encoder. Narrow technical terminology → late interaction. Comparative
   or criterion-conditioned relevance ("which of these supersedes the other") → LLM reranker, and
   only then. State which of those three your queries actually are; if you cannot, the cross-encoder
   is the default and the burden of proof is on the alternatives.
4. **Budget the latency, in milliseconds, before you build.** Write the P95 for each stage — retrieval,
   rerank, synthesis, network — and confirm the sum fits your SLO. Use the worked example's ratios as
   a prior, not as a measurement: an LLM reranker is roughly 20× a small cross-encoder, and you should
   assume it does not fit a sub-3-second budget until you have measured it fitting.
5. **Decide the score-fusion policy and the position policy separately.** State whether the reranker
   replaces or preserves the retrieval score, and where `LongContextReorder` sits relative to the
   reranker in the postprocessor list. If you enable MMR, state the new score range and re-derive
   every downstream cutoff.
6. **Define the acceptance test and the revert condition.** Specify the metric (nDCG@k or answer
   accuracy), the subset, and the threshold that deletes the reranker. A reranker that does not beat
   no-reranker on your data is a latency regression with a config file, and the revert condition is
   how you find out without an argument.

**Why this exercise matters.** Ranking is the stage where RAG systems are most often *configured*
rather than *designed*: `similarity_top_k` and a reranker class get copied from a tutorial, and the
N value — which silently determines the recall ceiling of every answer the system will ever give —
is never revisited. The engineer who can produce this ADR has done three things the config cannot:
measured the recall curve so N is a decision rather than a default, priced the reranker against the
end-to-end SLO so the choice is an architecture exclusion rather than a preference, and separated
the three ordering problems (relevance, diversity, position) that a single `top_k` integer
conflates. Note what the worked-example table refuses to tell you: which reranker to use. It cannot,
because that answer is a function of your latency budget and your query mix, and the *cheapest*
reranker captured most of the gain on LogiTrack's data — a result you should expect and must still
verify locally.

---

## Where the literature disagrees

- **"LLM rerankers beat cross-encoders on quality."** This module positions LLM rerankers as the
  quality ceiling, following the framework's "highest quality, latency is not critical"
  [[LI-RERANK]](#src-li-rerank), and reports Sun et al.'s finding that "properly instructed LLMs can deliver
  competitive, even superior results to state-of-the-art supervised methods on popular IR
  benchmarks" [[RANKGPT23]](#src-rankgpt23). Zhang, Hofstätter, Lewis, Tang & Lin test exactly what that claim rests
  on and find the dependency is not the LLM but the *vendor*: "current works in this direction all
  depend on the GPT models, making it a single point of failure in scientific reproducibility.
  Moreover, it raises the concern that the current research findings only hold for GPT models but
  not LLM in general." Their remedy — "build for the first time effective listwise rerankers
  without any form of dependency on GPT" — succeeds, "surpass[ing] the listwise rerankers based on
  GPT-3.5 by 13% and achiev[ing] 97% effectiveness of the ones built on GPT-4", but only after
  establishing that existing training data — "expressly constructed for pointwise ranking" — "are
  insufficient for building such listwise rerankers" [[RWOGPT23]](#src-rwogpt23). **What to change:** the quality gap
  between "LLM reranker" and "cross-encoder" is not a property of the two architectures; it is a
  property of which vendor's model is doing the ranking and what data it was trained on. Do not
  treat "use an LLM to rerank" as a technique with a known effect size — in this literature the
  effect size moves by 13% depending on the base model, and the reproducibility of the comparison
  is itself the finding.
- **"MMR improves LLM retrieval."** The framework's MMR notebook states that maximum marginal
  relevance "has been shown to improve performance for LLM retrievals" and cites a paper for it
  [[LI-MMR]](#src-li-mmr). The cited paper is Ye, Iyer, Celikyilmaz, Stoyanov, Durrett & Pasunuru, which is about
  **in-context learning exemplar selection**, not retrieval: they "propose a maximal marginal
  relevance-based exemplar selection approach for constructing exemplar sets that are both relevant
  as well as complementary, which successfully improves the in-context learning performance across
  three real-world tasks on multiple LLMs" [[MMRICL23]](#src-mmricl23). The mechanism is genuinely shared (MMR over a
  candidate set to trade relevance against diversity) but the *setting* is different: their
  "documents" are few-shot examples placed in the prompt, not retrieved evidence. **What to change:**
  treat "MMR improves retrieval" as **unsupported by the cited source**. MMR is a defensible
  heuristic with a 1998 provenance [[MMR98]](#src-mmr98) and an intuitive failure mode it addresses (five
  near-duplicates in six slots); that is a reason to test it on your duplicate rate, not a reason to
  cite a paper that measured something else.
- **"The position effect means reorder and move on."** `LongContextReorder` implements the
  beginning-or-end preference from Liu et al. [[LITM23](#src-litm23), [LI-NPP](#src-li-npp)], and this module presents it as a
  cheap, separately-testable decision. Note what the source actually claims: performance "can
  degrade significantly when changing the position of relevant information" and degradation occurs
  "even for explicitly long-context models" — a *robustness* claim about long-context use, not a
  prescription about prompt layout. The paper does not test a reordering intervention; the
  intervention is the framework's inference from the finding. **What to change:** reordering is a
  hypothesis about your prompts, testable by running your own query set with the two orderings and
  comparing answer accuracy. This is the same structure as M3's semantic-chunking entry — a
  documented feature whose benefit rests on an extrapolation from a different experiment — and it
  deserves the same treatment: measure, do not adopt.
- **Ownership note.** Reciprocal rank fusion, BM25, SPLADE, and ANN recall/latency curves are **M7's
  and M5's**; this module cross-references their score-incomparability argument and does not
  re-derive it. Fine-tuning corpus-specific rerankers, including the data-poor-regime result
  [[MONOT5-20]](#src-monot5-20) implies for a small labelled set, is **M15's**.

---

## Sources

- <a id="src-li-rerank"></a>[Rerankers](https://developers.llamaindex.ai/python/framework/module_guides/models/rerankers/) — `[LI-RERANK]` · owns the recall→precision framing of the retriever/reranker split, the `similarity_top_k`/`top_n` usage pattern, and the "highest quality, latency is not critical" positioning.
- <a id="src-li-np"></a>[Node Postprocessor](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/) — `[LI-NP]` · owns `BaseNodePostprocessor`, the `_postprocess_nodes` signature, the `query_str`-xor-`query_bundle` restriction, list-order composition, and `SimilarityPostprocessor`.
- <a id="src-li-npp"></a>[Node Postprocessor Modules](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/) — `[LI-NPP]` · owns the reranker catalogue, the `stsb-distilroberta-base` default-model warning, `ColbertRerank`'s `keep_retrieval_score`, and the `LongContextReorder` statement of the position effect.
- <a id="src-li-mmr"></a>[Simple Vector Stores — Maximum Marginal Relevance Retrieval](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/simpleindexdemommr/) — `[LI-MMR]` · owns the MMR formula, `mmr_threshold` semantics, the vector-store-query-mode placement, the score-rescaling warning, and the observed negative scores.
- <a id="src-litm23"></a>*Lost in the Middle: How Language Models Use Long Contexts* — Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang · arXiv:2307.03172 (TACL 2023) · [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172) — `[LITM23]` · owns the position effect and the robustness claim about long-context use. **M9 owns this claim course-wide**; M11 applies it at assembly time and does not re-survey it.
- <a id="src-colbert20"></a>*ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* — Omar Khattab, Matei Zaharia · arXiv:2004.12832 (SIGIR 2020) · [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) — `[COLBERT20]` · owns late interaction, offline-precomputable document representations, and the two-orders-of-magnitude-faster / four-orders-of-magnitude-fewer-FLOPs result.
- <a id="src-ce19"></a>*Passage Re-ranking with BERT* — Rodrigo Nogueira, Kyunghyun Cho · arXiv:1901.04085 (2019) · [arxiv.org/abs/1901.04085](https://arxiv.org/abs/1901.04085) — `[CE19]` · owns cross-encoder passage reranking and the 27% relative MRR@10 improvement over the then-state-of-the-art.
- <a id="src-monot5-20"></a>*Document Ranking with a Pretrained Sequence-to-Sequence Model* — Rodrigo Nogueira, Zhiying Jiang, Jimmy Lin · arXiv:2003.06713 (2020) · [arxiv.org/abs/2003.06713](https://arxiv.org/abs/2003.06713) — `[MONOT5-20]` · owns the seq2seq ranking formulation and, critically, the finding that it beats an encoder-only model "in a data-poor regime" — the small-labelled-set setting M15 inherits.
- <a id="src-mmr98"></a>*The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries* — Jaime Carbonell, Jade Goldstein · SIGIR 1998 · [author copy (PDF)](http://www.cs.cmu.edu/~jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf) · [CMU KiltHub record](https://kilthub.cmu.edu/articles/journal_contribution/The_Use_of_MMR_and_Diversity-Based_Reranking_for_Reodering_Documents_and_Producing_Summaries/6610811) — `[MMR98]` · owns maximal marginal relevance. **Format adaptation:** this paper has no arXiv version, so the gold-standard entry shape is adapted; the author copy is a PDF, which this session's fetch tool could not decode, so the author list and venue are corroborated from the CMU repository record and from [[LI-MMR]](#src-li-mmr), which cites the same paper — not from the paper's own front matter.
- <a id="src-rankgpt23"></a>*Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents* — Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren · arXiv:2304.09542 (EMNLP 2023) · [arxiv.org/abs/2304.09542](https://arxiv.org/abs/2304.09542) — `[RANKGPT23]` · owns the LLM-reranker quality claim, the NovelEval contamination-controlled test set, and the permutation-distillation result.
- <a id="src-rwogpt23"></a>*Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models* — Xinyu Zhang, Sebastian Hofstätter, Patrick Lewis, Raphael Tang, Jimmy Lin · arXiv:2312.02969 (2023) · [arxiv.org/abs/2312.02969](https://arxiv.org/abs/2312.02969) — `[RWOGPT23]` · owns the GPT-dependency critique of LLM-reranker research and the pointwise-training-data insufficiency finding.
- <a id="src-mmricl23"></a>*Complementary Explanations for Effective In-Context Learning* — Xi Ye, Srinivasan Iyer, Asli Celikyilmaz, Ves Stoyanov, Greg Durrett, Ramakanth Pasunuru · arXiv:2211.13892 (ACL 2023 Findings) · [arxiv.org/abs/2211.13892](https://arxiv.org/abs/2211.13892) — `[MMRICL23]` · owns MMR-based *exemplar selection* for in-context learning — the paper the framework's MMR notebook cites for a retrieval claim it does not make.

**Unsupported (synthetic or derived in-file, no located source):** the LogiTrack corpus, the
180-query evaluation set, every number in both worked-example tables (recall@N curve, nDCG@10,
answer accuracy, latencies, cost column), and all tradeoff-ledger entries. These are constructed for
teaching and are not empirical findings. The claim that the four candidate-set values produce a
flattening recall curve is LogiTrack fiction; the *shape* of that curve is the general expectation
(→ M5), but no source in this module measures LogiTrack's. The MMR score-rescaling numbers reproduced
in the table are **not** LogiTrack's — they are the framework notebook's published printout
[[LI-MMR]](#src-li-mmr) for its own three configurations, quoted to make the rescaling concrete. The assertion that
no `MMRPostprocessor` exists is based on a full-text documentation search returning zero results for
that identifier plus its absence from the node-postprocessor module list and the rerankers guide
[[LI-NPP](#src-li-npp), [LI-RERANK](#src-li-rerank)]; it is a statement about the **documented** surface, not about the source tree,
and the course syllabus's Thread C listing of `MMRPostprocessor` for M9 is contradicted by it. The
latency figures for the LLM reranker's "roughly 20×" multiplier are asserted as a prior, not
measured here.

**Verification note:** 🟡 *pending* — Step 7 (memory-less logic review) and Step 8 (literature
survey) not yet run on this module.

**Next module:** [M10 — Query Understanding and Query Transformation](../10-query-understanding/README.md)
— what to do when the *candidate set itself* is wrong, which no amount of reranking can repair.
