# Retrieval-Augmented Generation — A Senior Engineer's Course

A structured self-study curriculum that takes a working software or ML engineer to the level where
they can **design, defend, measure, and operate** a production retrieval-augmented generation
system — and pass a senior RAG engineer interview on the strength of the decisions they can
justify, not the vocabulary they can recite.

**22 modules · 22 labs · 1 capstone · 2 assessments · 1 tooling map.**

---

## What makes this course different

Most RAG material is a list of techniques: here is chunking, here is hybrid search, here is
reranking. That material teaches you to *recognize* a RAG system. It does not teach you to *own*
one, because it never tells you what each technique costs, when it stops paying, or what breaks
when you get it wrong.

This course is built the other way round, on three structures:

- **A running domain spine.** One invented company, LogiTrack, whose corpus is
  hierarchical legal prose, cross-referenced regulations, machine-hostile tables, dated bulletins,
  and multi-tenant permissions — growing by 160× over twelve months under a fixed budget and a
  hard latency SLO. Every module's worked example advances it, so concepts compound instead of
  restarting.
- **A failure taxonomy.** Eight named failure modes (F1–F8) introduced in
  [M2](modules/02-rag-pipeline-end-to-end/README.md) and extended to ten. Every module opens on
  the failure it prevents and every decision maps back to a failure. Engineering is the practice
  of naming your failures in advance.
- **A tradeoff ledger.** Every module contributes entries of the exact shape *decision · cost ·
  why accepted · what reverses it*. The ledger is the capstone's decision record.

Every claim is grounded: stack and API claims against the LlamaIndex docs, conceptual and
scientific claims against primary literature with full author lists and links that open the paper.
Where the literature disagrees with the module, the module says so and quotes it. Where coverage
is thin, it says that too.

---

## Who this is for

**You are ready if** you can write Python comfortably, have shipped a service that talks to an
LLM API, and are comfortable with dot products and cosine similarity. No prior information
retrieval background is assumed — BM25, nDCG, and HNSW are all built from first principles.

**You will get the most out of it if** you have a corpus you actually care about, because every
design exercise asks you to apply the decision to your own system, not to recall a definition.

**Not for you if** you want a framework tutorial. This is a systems course that uses a framework.

---

## The reference stack

**LlamaIndex (Python framework)** is the single reference stack. Every API and feature claim is
traceable to a page in its [framework documentation](https://developers.llamaindex.ai/python/framework/),
and version-sensitive statements carry their version. Model backends are reached through a
LiteLLM-style router so that swapping providers is configuration rather than a rewrite.

The point is not to teach you LlamaIndex. It is that a senior engineer must be able to read a
framework's actual documentation and distinguish what it guarantees from what a blog post
asserted — and this course models that discipline on every page.

---

## The modules

### Part I — Foundations
| # | Module | The decision it earns you |
|---|---|---|
| M1 | [Why RAG Exists and How to Think About It](modules/01-why-rag-exists/README.md) | RAG vs fine-tuning vs long context, with a revert condition |
| M2 | [The RAG Pipeline End to End](modules/02-rag-pipeline-end-to-end/README.md) | The failure taxonomy and where you instrument first |

### Part II — Ingestion
| # | Module | The decision it earns you |
|---|---|---|
| M3 | [Corpus Acquisition, Parsing, and the Chunking Decision](modules/03-parsing-and-chunking/README.md) | The chunking ADR, with cost arithmetic — *gold standard module* |
| M4 | [Embeddings and Vector Space](modules/04-embeddings-and-vector-space/README.md) | The embedding model choice, budgeted |
| M5 | [Vector Databases and ANN Indexes](modules/05-vector-databases-and-ann/README.md) | The store selection, by cost model and licence |
| M6 | [Metadata, Filters, and Multi-Tenancy](modules/06-metadata-filters-multitenancy/README.md) | The metadata contract and isolation model |

### Part III — Indexing and Retrieval
| # | Module | The decision it earns you |
|---|---|---|
| M7 | [Sparse, Dense, and Hybrid Retrieval](modules/07-sparse-dense-hybrid/README.md) | Fusion strategy and the α that fits your query mix |
| M8 | [Index Structures Beyond Flat Vectors](modules/08-index-structures-beyond-flat/README.md) | Which non-vector index earns its complexity |
| M9 | [Ranking, Fusion, and Reranking](modules/09-ranking-fusion-reranking/README.md) | The candidate-set size, priced in latency |
| M10 | [Query Understanding and Query Transformation](modules/10-query-understanding/README.md) | Which transformations ship |

### Part IV — Generation
| # | Module | The decision it earns you |
|---|---|---|
| M11 | [Context Assembly, Prompting, and the Temperature Dial](modules/11-context-assembly-temperature/README.md) | Temperature per task, with measured variance |
| M12 | [Grounding, Citations, and Attribution](modules/12-grounding-and-citations/README.md) | Citation policy and the abstention policy |

### Part V — Measurement
| # | Module | The decision it earns you |
|---|---|---|
| M13 | [Evaluation of Retrieval and Generation](modules/13-evaluation/README.md) | The regression gate that blocks a merge |

### Part VI — Advanced and Scale
| # | Module | The decision it earns you |
|---|---|---|
| M14 | [Ingestion at Scale, Incremental Sync, and Deletion](modules/14-ingestion-at-scale/README.md) | Freshness design and the re-embed budget |
| M15 | [Fine-Tuning Embeddings and Rerankers for a Domain](modules/15-finetuning-retrieval-models/README.md) | When tuning beats better retrieval plumbing |
| M16 | [Advanced Architectures: GraphRAG, Self-RAG, Agentic Retrieval](modules/16-advanced-architectures/README.md) | Quality gain per added dollar |

### Part VII — Production
| # | Module | The decision it earns you |
|---|---|---|
| M17 | [Multi-Tenant, Permissioned, and Regulated RAG](modules/17-multitenant-permissioned/README.md) | The authorisation backstop that fails closed |
| M18 | [Cost Engineering and Latency Budgets](modules/18-cost-and-latency/README.md) | The budget ADR — what you cut and what you refuse to |
| M19 | [Observability, Tracing, and Online Evaluation](modules/19-observability/README.md) | The dashboard that catches a drop before a user does |
| M20 | [Failure Modes, Drift, and Continuous Improvement](modules/20-failure-modes-and-drift/README.md) | Triage from symptom to stage |
| M21 | [Security, Prompt Injection, and Adversarial Retrieval](modules/21-security-prompt-injection/README.md) | A measured attack success rate, before and after |
| M22 | [Architecture Review and the Senior Engineer's Playbook](modules/22-architecture-review/README.md) | The architecture, defended under challenge |

---

## How to work through it

1. **Read [syllabus.md](syllabus.md) first.** It holds the domain spine, the failure taxonomy,
   the cross-cutting threads, and the ownership map. Modules assume you know what is in it.
2. **One module at a time, in order.** Cross-references are dense and forward-looking; the
   course is a spine, not a menu.
3. **Do the design exercise before reading the next module.** The exercises are where learning
   lands, and each one feeds the [capstone](capstone/README.md).
4. **Run the lab.** Reading about a recall/precision tradeoff and measuring one are different
   activities.
5. **Take the [midterm](assessments/midterm.md) after M11** and the
   [final](assessments/final.md) after M22.
6. **Keep the ledger.** Write every tradeoff entry into your own decision record as you go.
   At M22 you will read it back as your architecture.

### Suggested pace

| Track | Pace | Duration |
|---|---|---|
| Intensive | 1 module/day + lab | ~5 weeks |
| Working engineer | 3 modules/week | ~8 weeks |
| Deep + capstone-first | 1 module/week, capstone in parallel | ~6 months |

---

## Supporting material

| File | What it holds |
|---|---|
| [syllabus.md](syllabus.md) | The blueprint: scope, spine, module sketches, failure taxonomy, ownership map, verification protocol |
| [capstone/README.md](capstone/README.md) | The end-to-end system, its eight deliverables, milestones, and rubric |
| [maps/vector-db-map.md](maps/vector-db-map.md) | The vector DB and tooling landscape with **verified** licences, cost models, and honest flagging of the unverified |
| [assessments/midterm.md](assessments/midterm.md) | Mid-term after M11 |
| [assessments/final.md](assessments/final.md) | Final after M22 |
| `discussions/` | Durable answers to questions raised while working through the course |

---

## Verifying this course yourself

Two checkers ship with the course. They are the mechanical half of the verification protocol in
[syllabus §9](syllabus.md); the judgment half (the memory-less logic review and the literature
survey) cannot be automated and is documented per module.

```bash
python3 tools/check_tree.py            # anatomy, link resolution, citation graph, status table
python3 tools/check_cross_module.py    # key-format uniformity, ownership map, re-survey detection
```

- **`check_tree.py`** asserts every module carries the full anatomy in order, every relative link
  resolves, every inline citation has a bibliography entry and vice versa, and the syllabus status
  table matches the modules on disk.
- **`check_cross_module.py`** asserts one key format across every bibliography, and that a claim
  owned by one module is **not** re-surveyed in another — the failure the ownership map in
  syllabus §6 exists to prevent.

Both currently report **zero hard failures across all 22 modules.**

The lab corpus is also reproducible and is what the worked-example arithmetic is derived from:

```bash
python3 labs/generate_corpus.py --out ./data --seed 1337 --docs 3000 --queries 400
```

---

## Honesty notes

- **LogiTrack is fiction.** Its corpus, numbers, queries, and worked-example tables carry no
  citation burden and are flagged as synthetic in every module that uses them. Any *real-world*
  number embedded in an example — an embedding dimension, a benchmark score, a spec limit — does
  carry a source.
- **Version-sensitive claims carry versions.** RAG tooling moves fast; a claim true at the
  authoring baseline may not be true at yours. Each module names its baseline where it matters.
- **Disagreement is reported, not smoothed.** Every module has a *Where the literature disagrees*
  section when the literature actually disagrees, quoting the sources rather than paraphrasing
  them.
- **Unsupported claims are labelled.** Where a claim has no locatable source, it sits in an
  explicit unsupported bucket rather than being attached to an invented reference.
- **Modules are verified, not merely written.** Each module's Verification note records the
  memory-less logic review and the literature survey it passed, with rounds and residual findings.
  See [syllabus §9](syllabus.md) for the protocol.
