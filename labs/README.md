# The LogiTrack lab harness

The course's labs all run against one synthetic corpus, generated deterministically from a seed.
That is deliberate: if every learner has the same corpus, then every measured number in a module
is comparable, and a disagreement about a result is a disagreement about method rather than about
data.

---

## Honest status

**The labs are written to be runnable but were not executed during authoring.** The authoring
sandbox had no `llama-index-core`, no embedding provider credentials, and no vector store
running, so no lab output in any module is a real measurement — they are illustrative numbers
clearly marked as synthetic fiction in each module's *Unsupported* bucket.

What this means for you:

- Every lab is a specification you should **run and replace with your own output.** The moment
  you run one, the module's illustrative table becomes your measurement, and the exercise becomes
  real.
- If a lab fails against a newer `llama-index-core` than the baseline pinned in
  `requirements.txt`, that is expected drift in a fast-moving framework — fix the call, and note
  the version. That is itself the M4/M22 skill of reading docs rather than memory.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # or point LiteLLM at any provider / a local Ollama or vLLM
```

The labs are written against a LiteLLM-style router so that the backend is a configuration
change. Nothing in the course requires a specific model vendor.

---

## Generate the corpus

```bash
python generate_corpus.py --out ./data --seed 1337 --docs 3000
```

The generator produces the six LogiTrack source types described in
[syllabus §2](../syllabus.md), with the properties that make the course's problems real:

| Source type | Deliberately included |
|---|---|
| Carrier service agreements | Numbered clauses, cross-references, and answers that **straddle clause boundaries** |
| Customs procedure manuals | Per-jurisdiction variants with overlapping text (a tenancy and dedup problem) |
| Tariff and rate tables | Real grid structure written to XLSX, so a naive text extract destroys row/column affiliation |
| DG rules | Forward and backward cross-references between documents (a multi-hop problem) |
| Ops wiki / tickets | Short high-signal text, as the contrast case for chunk size |
| Service bulletins | Dated and versioned, with supersession chains (a freshness problem) |

It also emits `data/questions.jsonl` — the labelled query set, stratified into lexical-only,
semantic-only, multi-hop, numeric-extraction, and unanswerable classes, with span-level gold
answers. That file **is** the course's evaluation spine: M1 freezes its first 60 queries, M13
grows it to 400 and makes it the regression gate, M18 attaches it to CI, M22 reports on it.

> Because the queries are generated alongside the corpus, they are synthetic and their labels are
> exact by construction. That makes them a fair way to *compare* configurations — and a poor way
> to estimate absolute production quality. **M13 owns the circularity trap this creates**
> (queries derived from the same text you will retrieve inflate scores); read it before
> treating a lab's absolute number as a forecast.

---

## Layout

```
labs/
├── requirements.txt
├── generate_corpus.py      # deterministic LogiTrack corpus + labelled queries
├── eval_harness.py         # shared metric computation (recall@k, nDCG, MRR, faithfulness)
└── README.md
```

Per-module labs live beside their module under `modules/NN-slug/lab/` and import the shared
harness from here, so a metric is defined once and every module measures the same quantity.

---

## Running a lab

Each module's README names its lab and the decision that lab informs. The general shape:

```bash
cd modules/03-parsing-and-chunking/lab
python run.py --data ../../../labs/data --config configs/baseline.yaml
```

Labs print a comparison table. **Replace the module's illustrative table with your output** and
state in your ledger what changed — that record, not the lab, is the deliverable.
