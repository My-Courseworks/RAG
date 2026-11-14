# Practising scale under constraints — RAG

A laptop cannot hold LogiTrack's month-12 system, but it can be made to fail the way that system
fails. This doc lists the places in the RAG pipeline where you can make a resource limit bind on
purpose (54 injection points, from the corpus generator to the cost meter), the fixtures that do the
binding, and the arithmetic that maps each laptop result back to production. Each experiment ends in
a number you predicted, a number you measured, and a fix whose signature you can see in the curve.
That is the shape of answer interviews ask for, and it sticks where reading theory does not: *I
made this bottleneck happen, I measured where it appeared, this fix moved it, and this is what broke
next.*

> **Status.** Everything below is a **proposal**. The lab kit has not been built and no experiment
> has been run. Snippets were checked against official docs or upstream source (see
> [Appendix B](#appendix-b)), and none has been executed. Numbers marked as predictions are napkin
> arithmetic, not measurements. ⚠️ marks a claim that could not be verified; test it before you
> rely on it.

## Contents

1. [Why scale is learnable without production](#why-scale)
2. [The method](#the-method)
3. [The constraint map](#the-constraint-map)
4. [The lab kit](#the-lab-kit)
5. [Injection points, by layer](#injection-points)
   - [GEN · corpus generation](#layer-gen) · [PARSE · parsing and chunking](#layer-parse) ·
     [EMB · embedding](#layer-emb) · [VEC · vector store](#layer-vec) ·
     [FILT · filters, tenancy, permissions](#layer-filt) · [HYB · sparse and hybrid](#layer-hyb) ·
     [RRK · reranking](#layer-rrk) · [QRY · query understanding](#layer-qry) ·
     [LLM · context and generation](#layer-llm) · [CIT · grounding and citations](#layer-cit) ·
     [CACHE · caching](#layer-cache) · [ING · ingestion, sync, deletion](#layer-ing) ·
     [SRV · serving and traffic](#layer-srv) · [EVAL · the evaluation harness](#layer-eval) ·
     [OBS · observability](#layer-obs) · [SEC · security](#layer-sec) · [COST · cost](#layer-cost) ·
     [OPS · recovery](#layer-ops)
6. [The lab ladder](#the-lab-ladder)
7. [From experiment to interview answer](#interview)
8. [What this cannot teach](#cannot-teach)
9. [Appendix A — what the research found in the course itself](#appendix-a)
10. [Appendix B — verification status](#appendix-b)

---

<a id="why-scale"></a>
## Why scale is learnable without production

**"Scale" is not a size. It is the point where some resource limit binds.** A 23 GB index is not
"big". It is big relative to 16 GB of RAM, a 1M-tokens-per-minute provider quota, or a 15-minute
freshness window. Move the limit and the same phenomenon appears at a hundredth of the data. The
phenomenon follows the **ratio** of demand to limit, not the absolute numbers. So you do not need
production hardware. You need to shrink the limit by the same factor you shrank the data.

You already have the napkin numbers for LogiTrack's month-12 corpus (the course's running fictional
company; see [syllabus §2](syllabus.md)):

| Quantity | Month 12 | Arithmetic |
|---|---|---|
| Documents | 480,000 | 3,000 × 160 |
| Words | ≈ 2.2B | 480k × 4,513 (the generator's measured mean) = 2.17B |
| Tokens | ≈ 2.9B | 2.17B ÷ 0.75 words/token |
| Chunks @ 512 tokens | ≈ 5.6M | 2.9B ÷ 512 |
| float32 @ 1,024-d | ≈ 23 GB | 5.6M × 4,096 B |
| int8 / binary | ≈ 5.8 GB / 0.7 GB | ÷ 4 / ÷ 32 |
| Queries | 8,500/day ≈ 0.1 QPS | 8,500 ÷ 86,400 |

Add one assumption for the busy hour (⚠️ invented: 70% of traffic in 8 hours, with bursts at 3×):
8,500 × 0.7 ÷ 28,800 s × 3 ≈ **0.62 QPS**. By Little's law (L = λW, items in the system = arrival
rate × time each spends there), 0.62 QPS × 2.5 s ≈ **1.5 requests in flight**. LogiTrack's own
servers almost never queue on query traffic. What binds instead is:

1. **Corpus-side memory and build time:** 23 GB of float32, HNSW graph builds, BM25 rebuilds.
2. **Shared provider limits:** tokens per minute (TPM) and requests per minute (RPM), drawn on by
   query, ingest, eval and judge traffic alike.
3. **Bursts and backfills:** partner bulk uploads, re-embeds, re-parses, revocation storms.

So most experiments here are about memory, rate limits and replaying time, and few are about raw QPS.

### A worked miniaturisation

M5 ([Vector Databases and ANN](modules/05-vector-databases-and-ann/README.md)) describes an index that
"does not fit": 184 GB of vectors on a 64 GB box. The quantity that decides the behaviour is memory
pressure:

$$P_{mem} = \frac{\text{resident bytes the index needs}}{\text{RAM available to the process}} = \frac{184}{64} \approx 2.9$$

On the laptop, take 300k chunks of 1,024-d float32: 300,000 × 4,096 B = **1.23 GB**. To get the same
P_mem of 2.9 you need 1.23 GB ÷ 2.9 ≈ **425 MB** of RAM for vectors, plus the store's own baseline
(measure it with an empty collection; it does not scale with data). That gives a container of
roughly 425 MB + baseline. Inside it, an mmap-backed index thrashes the page cache in the same way
and for the same reason as the 64 GB box. The reverse mapping holds too. The miniature is
5.6M ÷ 300k = **18.7×** smaller than month-12 production, so a 2 GB container for 300k vectors
behaves like a 2 × 18.7 ≈ 37 GB production node (with 1.6× headroom).

The ratio does not carry everything. HNSW search touches about log N nodes, so a 300k graph needs
fewer hops than a 5.6M graph and the production cliff arrives slightly earlier than the miniature
predicts. Every injection point below states the ratio it preserves and, under **Where the
miniature lies**, what it gives up.

---

<a id="the-method"></a>
## The method

Five habits. Each comes with one concrete example from this course.

| # | Habit | What it means | Example here |
|---|---|---|---|
| 1 | **Shrink the machine instead of growing the data** | Cap RAM, CPU, IOPS, TPM, context window and freshness budget until the small dataset is "too big" | 300k vectors in a 425 MB container reproduces M5's 184 GB-on-64 GB case ([VEC-01](#vec-01)) |
| 2 | **Hold the ratio fixed** | Pick the dimensionless ratio that governs the phenomenon and keep its production value | 3k-doc corpus (19.95M embedded tokens) against a mock quota of 5,530 TPM takes the same 60 virtual hours as 3.61B tokens at 1M TPM ([EMB-01](#emb-01)) |
| 3 | **Measure curves, not points** | Sweep the constraint and plot the metric; the knee is the finding, and a single point tells you nothing | p99 and page-cache refaults vs container RAM from 4 GB down to 256 MB ([VEC-01](#vec-01)); recall vs filter selectivity ([FILT-01](#filt-01)) |
| 4 | **Fake the provider and break things on purpose** | An OpenAI-compatible mock with your own limits, latency shape and faults; toxiproxy; SIGKILL | A 50% 5xx brown-out for 5 virtual minutes shows retry amplification ([SRV-03](#srv-03)); killing ingest workers exposes random node ids ([ING-02](#ing-02)) |
| 5 | **Predict before you measure** | Write the napkin number and a range first; the gap between prediction and measurement is where you learn | "A 15-minute poll gives freshness p95 ≈ 0.95 × 15 + t_proc", so it can only meet a 15-minute p95 if processing takes < 45 s ([ING-01](#ing-01)) |

### The ratios to hold fixed

Almost every experiment preserves one of these. Production values are LogiTrack's (course fiction
plus the napkin above); laptop values are the proposed miniature.

| Ratio | Production value | Laptop value | How you set it |
|---|---|---|---|
| **Memory pressure** P_mem = working set ÷ RAM | 2.9 (M5's 184 GB on 64 GB); ≈ 0.6 for 23 GB float32 on a 37 GB node | 1.23 GB in 425 MB (2.9) or in 2 GB (0.6) | `mem_limit` + `memswap_limit` equal; subtract the measured baseline RSS |
| **Data scale factor** | 5.6M chunks | 300k chunks (18.7× smaller) | generator `--docs ≈ 25,600` (11.7 chunks/doc) |
| **Utilisation** ρ = λ × service time ÷ servers, at the bottleneck | reranker ≈ 0.12 at busy hour (0.6 QPS × 0.4 core-s ÷ 2 cores, ⚠️ illustrative service time) | same 0.12 | choose λ and `--cpus` together; if you compress time by s, compress service time by s too |
| **Selectivity** p_c (fraction of corpus a filter admits) and p_n (fraction of a query's true neighbours it admits) | 1/40 jurisdictions × 1/12 clients = 0.21% | 0.21% (fractions are N-independent) | `--jurisdictions 40 --clients 12`; scale any *absolute* planner threshold by ÷ 18.7 |
| **Budget ratio** work ÷ quota | 3.61B tokens ÷ 1M TPM = 60 h | 19.95M ÷ 5,530 TPM = 60 virtual h | mock provider `TPM`, `RPM`, `SPEED` |
| **Freshness** poll period ÷ SLO, ingest ρ | 15 min ÷ 15 min; 2,640 events/day ≈ 1.8/min | same, in virtual time | replayer `SPEED`; mock limits in virtual time; population sample φ with capacity × φ |
| **Burst ÷ index size** | 2,000 docs × 11.7 = 23,400 vectors into 5.6M = 0.4% | 1,250 vectors into 300k | burst sized by fraction, not count |
| **Context fill** evidence tokens ÷ window | 6 × 600 = 3,600 of a 10.4k prompt ≈ 35% | 1,430 of a 4,096 window | `num_ctx` / `-c` on a small local model |
| **Cache** capacity ÷ distinct intents, TTL × per-intent arrival rate | unknown U; TTL ≤ 15 min | U ÷ 10, C ÷ 10, same virtual TTL | Redis `maxmemory`; Zipf load generator |
| **Identifier rarity** df ÷ N per identifier class | UN3480 in ~40 of 187,043 chunks at pilot | same df/N | generator `--identifier-density` |
| **Tenant overhead** T × c ÷ total memory | 12 tenants, negligible | sweep T to 1,000 | measure c from an empty collection |
| **Sampling** λ × stratum share × sample rate | 8,500 × 4% × 1% = 3.4 samples/day | same, in virtual time | replay `SPEED`, sampler rate |
| **Rescore IO** random reads per query ÷ volume IOPS | 30 reads at the volume's IOPS | same IOPS cap | `--device-read-iops` (Linux only) |
| **Fault rate** kills per 1,000 docs, 5xx share, omitted-filter share | unknown | sweep 0.1–5% | chaos scripts, mock `P_5XX` |

Anything that scales with log N, a fixed per-process overhead, or hardware topology does **not**
follow these ratios. Those are what the "lies" fields list.

---

<a id="the-constraint-map"></a>
## The constraint map

The index of the doc. Rows are injection points grouped by pipeline stage, columns are the aspects
they exercise. ● = primary, ○ = secondary. Every ID links to its section.

Failure modes referenced throughout are the course's taxonomy ([syllabus §6](syllabus.md)):
**F1** retrieval miss · **F2** retrieval noise (right chunk buried) · **F3** chunk-boundary loss ·
**F4** representation mismatch (user and corpus vocabulary never meet) · **F5** context overflow ·
**F6** unfaithful synthesis · **F7** staleness · **F8** budget blowout · **F9** poisoned context
(M21) · **F10** scope leak (M17).

| Stage | ID | Injection point | Mem | Lat | Thru | Cost | Fresh | Quality | Correct/idem | Isolation/sec |
|---|---|---|---|---|---|---|---|---|---|---|
| Corpus gen | [GEN-01](#gen-01) | Generator memory, streaming and per-doc seeding | ● | | ○ | | | | ○ | |
| | [GEN-02](#gen-02) | Prose entropy and metadata–text correlation | | | | | | ● | | |
| | [GEN-03](#gen-03) | Revision, near-duplicate and boilerplate density | ○ | | | | ● | ● | ● | |
| | [GEN-04](#gen-04) | Event timeline and replay speed | | | ○ | | ● | | ○ | |
| | [GEN-05](#gen-05) | Gold-label construction and validators | | | | | | ● | ● | |
| Parse/chunk | [PARSE-01](#parse-01) | Parser throughput and version backfill | | | ● | ○ | ● | ○ | ● | |
| | [PARSE-02](#parse-02) | Extraction corruption | | ○ | | ○ | | ● | ● | |
| | [PARSE-03](#parse-03) | Node-count explosion vs in-memory stores | ● | ○ | | | | ○ | | |
| Embedding | [EMB-01](#emb-01) | Provider TPM/RPM in bulk ingest and re-embed | | | ● | ○ | ● | | ○ | |
| | [EMB-02](#emb-02) | Self-hosted embedding on a CPU budget | | ● | ● | ● | | ○ | | |
| | [EMB-03](#emb-03) | Mixed-model index from a killed migration | | | | | | ● | ● | |
| Vector store | [VEC-01](#vec-01) | RAM-to-index pressure and quantization | ● | ● | | ○ | | ● | | |
| | [VEC-02](#vec-02) | Disk IOPS under mmap and rescore | ○ | ● | ○ | | | | | |
| | [VEC-03](#vec-03) | CPU cap and the brute-force crossover | | ● | ● | | | ○ | | |
| | [VEC-04](#vec-04) | HNSW build time and build-time dials | ● | | ● | | ● | ● | | |
| | [VEC-05](#vec-05) | Ingest-while-serving interference | | ● | ● | | ● | | | |
| | [VEC-06](#vec-06) | Query-time ANN dials and the noise floor | | ● | | | | ● | | |
| Filters/tenancy | [FILT-01](#filt-01) | Filter selectivity and post-filter collapse | | ● | | | | ● | | ○ |
| | [FILT-02](#filt-02) | Tenant count, skew, noisy neighbour | ● | ● | | ○ | | ● | | ● |
| | [FILT-03](#filt-03) | Permission-change propagation | | | ○ | | ● | | ● | ● |
| | [FILT-04](#filt-04) | Scope-leak fault injection | | | | | | | ○ | ● |
| Sparse/hybrid | [HYB-01](#hyb-01) | BM25 memory, query cost, rebuild under churn | ● | ● | | | ● | | | |
| | [HYB-02](#hyb-02) | Identifier rarity, IDF drift, stemmer faults | | | | | | ● | | |
| Rerank | [RRK-01](#rrk-01) | Reranker CPU × candidate N × concurrency | | ● | ● | ○ | | ● | | |
| | [RRK-02](#rrk-02) | Hosted reranker latency, timeouts, degraded mode | | ● | | ○ | | ○ | | |
| Query | [QRY-01](#qry-01) | Fan-out tail amplification | | ● | ● | ● | | ○ | | |
| | [QRY-02](#qry-02) | Router misroute and the mixture-P95 trap | | ● | | ● | | ● | | |
| Context/generation | [LLM-01](#llm-01) | Context window, `num_output`, tokenizer caps | | ○ | | ○ | | ● | ● | |
| | [LLM-02](#llm-02) | Provider TTFT and shared TPM at the busy hour | | ● | ● | ● | | | | |
| | [LLM-03](#llm-03) | Nondeterminism and silent alias drift | | | | | | ● | ● | |
| | [LLM-04](#llm-04) | Agent-loop token blow-up | | ● | | ● | | ○ | | ○ |
| | [LLM-05](#llm-05) | LLM work at ingest (graph, enrichment) | | | ● | ● | ● | ○ | ● | ○ |
| Grounding | [CIT-01](#cit-01) | Citation and node-id stability | | | | | | ○ | ● | |
| Caching | [CACHE-01](#cache-01) | Semantic cache capacity, TTL, threshold | ● | ● | | ● | ○ | ● | | ● |
| | [CACHE-02](#cache-02) | Invalidation on supersede, delete, model change | | | | | ● | | ● | ● |
| Ingest/sync | [ING-01](#ing-01) | Freshness under replay: poll vs push, bursts | | | ● | | ● | | ○ | |
| | [ING-02](#ing-02) | Worker crash mid-ingest: idempotency | | | ○ | ○ | | ○ | ● | |
| | [ING-03](#ing-03) | Duplicate, out-of-order and lost events | | | | | ● | | ● | |
| | [ING-04](#ing-04) | Deletion and right-to-be-forgotten completeness | | | | | ○ | | ● | ● |
| | [ING-05](#ing-05) | `num_workers` × per-process RSS | ● | | ● | | | | | |
| | [ING-06](#ing-06) | Disk, inode and file-count limits | ● | | | | ○ | | ● | |
| Serving | [SRV-01](#srv-01) | Open-loop load and the queueing knee | | ● | ● | | | | | |
| | [SRV-02](#srv-02) | Network RTT, bandwidth, payload size | | ● | ○ | | | | | |
| | [SRV-03](#srv-03) | Timeouts, retry storms, circuit breaking | | ● | ● | ○ | | | | |
| Eval harness | [EVAL-01](#eval-01) | Eval-set size, A/A tests, noise floor | | | | | | ● | | |
| | [EVAL-02](#eval-02) | Injected label noise and circularity | | | | | | ● | | |
| | [EVAL-03](#eval-03) | Mock judge: variance, bias, quota, wall-clock | | | ● | ● | | ● | | |
| Observability | [OBS-01](#obs-01) | Trace pipeline backpressure and sampling | ○ | | ● | ○ | | ○ | | ● |
| | [OBS-02](#obs-02) | Planted regressions: detection latency | | | | | ● | ● | ● | ○ |
| | [OBS-03](#obs-03) | Query drift injection | | | | | | ● | | |
| Security | [SEC-01](#sec-01) | Poisoning dose-response and placement | | | | | ○ | ○ | | ● |
| | [SEC-02](#sec-02) | Egress and capability surface | | | | | | | | ● |
| Cost | [COST-01](#cost-01) | Token/cost meter and waterfall reconciliation | | ● | | ● | | ○ | | |
| Recovery | [OPS-01](#ops-01) | Snapshot, restore and rollback RTO | | | ● | | ● | | ○ | |

**Column coverage (● count):** memory 9 · latency 20 · throughput 17 · cost 9 · freshness 14 ·
quality 21 · correctness 16 · isolation 9. Memory, cost and isolation are the thinnest columns. FILT-02/03/04,
CACHE-01/02 and ING-04 are where M17's warning becomes something you can measure: "almost every
performance feature you build is an authorization bypass unless you deliberately make it not one."

---
<a id="the-lab-kit"></a>
## The lab kit

> **Proposed, not built, not executed.** None of the files below exists in the repo. Each snippet
> was written against official docs or upstream source as checked on 2026-11-12 (the tool table in
> [Appendix B](#appendix-b)), and none has been run. Treat them as reviewed drafts. Pin every image
> and package version when you build them.

Seven fixtures cover all 54 injection points. Most experiments need two or three.

| Fixture | What it gives you | Used by |
|---|---|---|
| K1 · constraint profiles | RAM, CPU, IO, file and disk caps as compose overrides | VEC-*, PARSE-01/03, EMB-02, ING-05/06, RRK-01, HYB-01 |
| K2 · mock provider | OpenAI-compatible embeddings, chat and rerank with token buckets in virtual time, a latency model, faults, alias drift and per-consumer metering | EMB-01, LLM-*, QRY-*, RRK-02, SRV-*, EVAL-03, COST-01 |
| K3 · fault injectors | toxiproxy toxics, netem, SIGKILL, disk-full, noisy neighbour, parse corruption, ACL and filter drops | PARSE-02, FILT-02/03/04, ING-02/06, RRK-02, SRV-02/03 |
| K4 · load driver | open-loop k6 arrival rates (plus a closed-loop Locust twin, used only to show coordinated omission) | SRV-01, RRK-01, CACHE-01, VEC-05 |
| K5 · virtual clock + replayer | one clock shared by every process; `events.jsonl` replayed at speed s with dup, reorder and drop | GEN-04, ING-*, FILT-03, CACHE-02, OBS-02/03 |
| K6 · metrics collector | `docker stats`, cgroup v2 counters, mock meter, Qdrant/Redis/k6/OTel scrapes, the standard plots | all |
| K7 · prediction log | one page per experiment plus a CSV that tracks your calibration | all |

### K1 · Constraint profiles (compose overrides)

```yaml
# constraints/compose/base.yml
services:
  qdrant:    { image: qdrant/qdrant:<pinned>, ports: ["6333:6333"], volumes: ["qdrant:/qdrant/storage"] }
  pg:        { image: pgvector/pgvector:<pinned>, environment: { POSTGRES_PASSWORD: x }, ports: ["5432:5432"] }
  redis:     { image: redis:7, command: ["redis-server", "--maxmemory", "64mb", "--maxmemory-policy", "allkeys-lru"] }
  toxiproxy: { image: ghcr.io/shopify/toxiproxy, ports: ["8474:8474", "26333:26333", "28080:28080"] }   # ⚠️ image path from its README, not re-checked
  mockprov:  { build: ../mockprov, ports: ["8080:8080"], env_file: ../mockprov/profiles/default.env }
volumes: { qdrant: {} }
```

```yaml
# constraints/compose/profiles/ram-3x.yml — reproduce "184 GB on a 64 GB box" (P_mem ≈ 2.9) at 300k chunks
services: { qdrant: { mem_limit: 425m, memswap_limit: 425m, cpus: 2 } }   # add your measured baseline RSS
# profiles/ram-fit.yml        (≈ 37 GB production node):  mem_limit: 2g
# profiles/cpu-1.yml:                                      cpus: 1
# profiles/cpu-prod-scale.yml (N × 18.7 emulation):        cpus: 0.054  (+ shorter CPU period; see VEC-03)
# profiles/io-cloud.yml       (Linux only):                blkio_config: { device_read_iops: [{ path: /dev/nvme0n1, rate: 3000 }] }
# profiles/tenant-1000.yml:                                one collection per tenant, same mem_limit as the shared case
```

Run with `docker compose -f base.yml -f profiles/ram-3x.yml up -d`. Sweep a limit without
restarting (which would lose the page cache) using `docker update --memory 1g --memory-swap 1g <ctr>`
or `--cpus`. Check that the limit applied with `docker inspect`: whether `deploy.resources.limits`
is honoured outside Swarm is reported inconsistently ⚠️, which is why the profiles use
service-level keys.

Caveats that change results:

- Equal `--memory` and `--memory-swap` means no swap. Under cgroup v2 (Docker Desktop included)
  `--oom-kill-disable` is silently discarded.
- `--cpus=1.5` is a CFS quota (`--cpu-period=100000 --cpu-quota=150000`). It throttles in whole
  periods, so at small quotas latency comes in steps. Shorten `--cpu-period`.
- Block-IO throttles (`--device-read-bps`, `--device-read-iops`) apply to real block devices only.
  On Docker Desktop for Mac, bind-mounted folders go through virtiofs and are not throttled, and the
  VM device path (`/dev/vda`) is ⚠️ unverified. Verify with
  `dd if=/data/bigfile of=/dev/null bs=1M count=200 iflag=direct` before trusting a throttle.
- On a Mac, `resource.setrlimit(RLIMIT_AS, ...)` exists but enforcement is unreliable ⚠️. Use a
  container. On a Linux host, `systemd-run --scope -p MemoryMax=2G -p CPUQuota=100% cmd` caps a
  non-Docker process.
- Docker Desktop specifics (inferred, not from official docs): every container shares one Linux VM
  whose RAM caps `--memory` (default 50% of host RAM, 1 GB swap); "disk" reads may be served by
  macOS's page cache beneath the VM, so cold reads look too fast. **Recommendation:** toxiproxy for
  network faults, container caps for RAM and CPU, and a Linux host (or a Lima/Colima/OrbStack VM with
  its own disk) for anything that throttles IO.

### K2 · Mock provider (OpenAI-compatible)

Reproduces provider **limits and latency shapes** deterministically. Embeddings are hash vectors,
which makes them valid for throughput, memory and limit experiments and meaningless for retrieval
quality.

```python
# constraints/mockprov/app.py   —   uvicorn app:app --port 8080
import asyncio, base64, hashlib, math, os, random, time
from collections import defaultdict
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

def E(k, d): return type(d)(os.getenv(k, d))
RPM, TPM, SPEED = E("RPM", 3000.0), E("TPM", 400000.0), E("SPEED", 1.0)        # SPEED: virtual-time compression
TOK_BURST, MAX_REQ_TOK = E("TOK_BURST", -1.0), E("MAX_REQ_TOKENS", 300000)      # bucket depth; per-request cap
TTFT_BASE, TTFT_PER_1K = E("TTFT_BASE_MS", 150.0), E("TTFT_PER_1K", 35.0)
DECODE_MS = E("DECODE_MS", 1.28)   # M18 prints "500 tokens at 5.2 ms/token = 640 ms"; 640/500 = 1.28 (Appendix A)
SIGMA, P_TAIL, TAIL_MULT = E("LAT_SIGMA", 0.3), E("P_TAIL", 0.0), E("TAIL_MULT", 10.0)
P_5XX, P_HANG, NOISE, SWAP_AT = E("P_5XX", 0.0), E("P_HANG", 0.0), E("NOISE", 0.0), E("SWAP_AT", -1.0)
DIM, OUT_TOK, CONTENT = E("DIM", 1024), E("OUT_TOKENS", 500), os.getenv("MOCK_CONTENT", "")
rng, T0, app = random.Random(E("SEED", 0)), time.monotonic(), FastAPI()
METER = defaultdict(lambda: defaultdict(int))
def vnow(): return (time.monotonic() - T0) * SPEED                               # virtual seconds
def ntok(s): return max(1, len(s) // 4)

class Bucket:                                     # token bucket, refilled in VIRTUAL time
    def __init__(s, per_min, cap=None):
        s.rate = per_min; s.cap = s.lvl = cap if cap and cap > 0 else per_min; s.t = vnow()
    def wait(s, n):
        t = vnow(); s.lvl = min(s.cap, s.lvl + (t - s.t) * s.rate / 60); s.t = t
        return 0.0 if s.lvl >= n else (n - s.lvl) * 60 / s.rate                 # virtual seconds to wait
REQ, TOK = Bucket(RPM), Bucket(TPM, TOK_BURST)

async def gate(n, who):
    if n > MAX_REQ_TOK or n > TOK.cap:                                          # can never succeed: reject, don't 429
        return JSONResponse({"error": {"message": "request too large", "type": "invalid_request_error"}}, 400)
    w = max(REQ.wait(1), TOK.wait(n))
    if w > 0:
        METER[who]["429"] += 1
        return JSONResponse({"error": {"message": "rate limited", "type": "rate_limit_error",
                                       "code": "rate_limit_exceeded"}}, 429,
                            {"retry-after": str(max(1, math.ceil(w / SPEED))),
                             "x-ratelimit-remaining-requests": str(int(REQ.lvl)),
                             "x-ratelimit-remaining-tokens": str(int(TOK.lvl))})
    REQ.lvl -= 1; TOK.lvl -= n
    if rng.random() < P_5XX: return JSONResponse({"error": {"message": "overloaded", "type": "server_error"}}, 503)
    if rng.random() < P_HANG: await asyncio.sleep(600)                          # exercises client timeouts
    return None

async def latency(ms):
    ms *= math.exp(rng.gauss(0, SIGMA)) * (TAIL_MULT if rng.random() < P_TAIL else 1)
    await asyncio.sleep(ms / 1000 / SPEED)

@app.post("/v1/embeddings")
async def embeddings(req: Request):
    b = await req.json(); xs = b["input"] if isinstance(b["input"], list) else [b["input"]]
    n, who = sum(ntok(str(x)) for x in xs), req.headers.get("x-consumer", "default")
    if (r := await gate(n, who)): return r
    await latency(20 + 0.02 * n); METER[who]["embed_tokens"] += n
    def vec(x):
        v = np.random.default_rng(int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:8], "little")).standard_normal(DIM)
        v = (v / np.linalg.norm(v)).astype("<f4")
        return base64.b64encode(v.tobytes()).decode() if b.get("encoding_format") == "base64" else v.tolist()
    return {"object": "list", "model": b.get("model", "mock-embed"), "usage": {"prompt_tokens": n, "total_tokens": n},
            "data": [{"object": "embedding", "index": i, "embedding": vec(x)} for i, x in enumerate(xs)]}

@app.post("/v1/chat/completions")                 # non-streaming only
async def chat(req: Request):
    b, who = await req.json(), req.headers.get("x-consumer", "default")
    n_in = ntok(" ".join(str(m.get("content", "")) for m in b["messages"]))
    n_out = min(b.get("max_tokens") or OUT_TOK, OUT_TOK)
    if (r := await gate(n_in + n_out, who)): return r
    await latency(TTFT_BASE + TTFT_PER_1K * n_in / 1000 + DECODE_MS * n_out)
    swapped = SWAP_AT >= 0 and vnow() > SWAP_AT                                 # silent alias remap (LLM-03)
    text = CONTENT or f"The surcharge is {'3.15' if rng.random() < NOISE else '3.5'}%."
    if swapped: text = "Based on the provided documents, " + text.lower() + " Please verify with the carrier."
    METER[who]["in"] += n_in; METER[who]["out"] += n_out
    return {"id": f"mock-{rng.getrandbits(32):x}", "object": "chat.completion", "created": int(time.time()),
            "model": "mock-frontier-2" if swapped else b.get("model", "mock-frontier-1"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}}

@app.post("/v1/rerank")                           # Cohere/Jina-like shape; adapt to your client
async def rerank(req: Request):
    b, who = await req.json(), req.headers.get("x-consumer", "rerank")
    docs = b["documents"]
    if (r := await gate(sum(ntok(str(d)) for d in docs), who)): return r
    await latency(60 + 3 * len(docs))
    return {"results": [{"index": i, "relevance_score": 1 / (1 + i)} for i in range(len(docs))][: b.get("top_n", len(docs))]}

@app.get("/metrics")
def metrics(): return {"virtual_s": vnow(), "meter": METER}
```

Wiring:

- LLM: `OpenAILike(model="mock-frontier-1", api_base="http://localhost:8080/v1", api_key="x",
  is_chat_model=True, context_window=16000, timeout=30, max_retries=3)` from
  `llama-index-llms-openai-like`. ⚠️ The package is not in `labs/requirements.txt`; check the
  constructor parameters at your pinned version.
- Embeddings: `OpenAIEmbedding(model="text-embedding-3-small", api_base="http://localhost:8080/v1",
  api_key="x", embed_batch_size=100)`. The mock ignores the model name.
- LiteLLM: `model="openai/mock-frontier-1", api_base=...`. Router knobs used below:
  `model_list[].litellm_params.{rpm,tpm}`, `num_retries`, `timeout`, `allowed_fails`,
  `cooldown_time`, `fallbacks=[{"frontier": ["cheap"]}]`, `routing_strategy`. Use this mock rather
  than LiteLLM's `mock_testing_rate_limit_error` (⚠️ seen in one proxy example only).
- Consumer tagging for [COST-01](#cost-01) goes through a default-headers option (`X-Consumer:
  query|ingest|eval|judge`). LiteLLM has `extra_headers`; ⚠️ check the parameter name on
  LlamaIndex's OpenAI classes.

Notes:

- The OpenAI Python SDK may request `encoding_format="base64"` for embeddings by default ⚠️. The mock
  handles both formats.
- Concurrency reorders RNG draws, so the mock is seeded but **not** deterministic run to run under
  concurrency. Real providers behave the same way (M11).
- `TOK_BURST` sets the bucket depth. Real providers allow undocumented bursts; the default depth is
  one minute of quota. A request larger than the depth can never pass, so the mock returns 400
  instead of looping on 429. [EMB-01](#emb-01) explains why this matters when you scale TPM down.
- Provider headers being imitated: OpenAI `x-ratelimit-{limit,remaining,reset}-{requests,tokens}`
  plus `Retry-After` on a 429. OpenAI embeddings accept ≤ 2,048 inputs, ≤ 8,192 tokens each and
  ≤ 300,000 tokens per request.

For physical caps rather than simulated ones, run a small local model: Ollama (`OLLAMA_NUM_PARALLEL`,
`OLLAMA_MAX_QUEUE`, `OLLAMA_CONTEXT_LENGTH`, request `options.num_ctx`; the macOS app needs
`launchctl setenv`; `num_thread` is ⚠️ not in current docs) or `llama-server -c 4096 -np 2 -t 4`
(continuous batching is on by default).

### K3 · Fault injectors

| Script | Does | For |
|---|---|---|
| `chaos/kill_workers.sh` | SIGKILL a random ingest worker every 20–60 s (`restart: on-failure` in compose) | [ING-02](#ing-02) |
| `chaos/toxics.sh` | toxiproxy presets: `lan`, `cross-az`, `internet`, `brownout`, `hang-2pct`, `reset-peer` | [RRK-02](#rrk-02), [SRV-02](#srv-02), [SRV-03](#srv-03), [ING-02](#ing-02) |
| `chaos/netem.sh` | `tc qdisc ... netem delay/loss/rate/duplicate/reorder` (Linux, needs `NET_ADMIN`) | [SRV-02](#srv-02) |
| `chaos/disk_full.sh` | tmpfs-backed data volume of fixed size | [ING-06](#ing-06) |
| `chaos/noisy_neighbour.sh` | `stress-ng --vm 1 --vm-bytes 1G --vm-keep` sibling container | [FILT-02](#filt-02) |
| `chaos/corrupt_parse.py` | page garble and table flatten at parse time | [PARSE-02](#parse-02) |
| `chaos/drop_acl.py` | strip `acl` from x% of nodes in a backfill | [FILT-03](#filt-03) |
| `chaos/drop_filter.py` | drop `filters` on x% of engine constructions | [FILT-04](#filt-04) |
| `chaos/alias_swap.env` | `SWAP_AT=<virtual s>` for the mock | [LLM-03](#llm-03) |

```bash
# chaos/kill_workers.sh
while true; do sleep $((20 + RANDOM % 40))
  docker kill --signal=SIGKILL $(docker ps -q -f name=ingest-worker | shuf -n1); done
```

```bash
# chaos/toxics.sh (excerpt) — toxiproxy works at the TCP layer, so it is the portable choice on a Mac
toxiproxy-cli create -l 0.0.0.0:26333 -u qdrant:6333 qdrant
toxiproxy-cli create -l 0.0.0.0:28080 -u mockprov:8080 rerank
toxiproxy-cli toxic add -t latency -a latency=20 -a jitter=5 qdrant            # cross-az
toxiproxy-cli toxic add -t bandwidth -a rate=2000 qdrant                       # KB/s
toxiproxy-cli toxic add -t timeout -a timeout=3000 --tox 0.02 rerank           # 2% of connections hang 3 s
toxiproxy-cli toxic add -t reset_peer -a timeout=0 --tox 0.01 qdrant           # ambiguous acks (⚠️ attribute name)
```

### K4 · Load driver (open loop)

```js
// constraints/load/k6/query_open.js     k6 run -e BASE=http://localhost:8000 query_open.js
import http from 'k6/http';
const qs = open('./queries.txt').split('\n').filter(Boolean);
export const options = {
  scenarios: { busy_hour: { executor: 'ramping-arrival-rate', startRate: 36, timeUnit: '1m',
    preAllocatedVUs: 20, maxVUs: 400,
    stages: [{ target: 36, duration: '5m' }, { target: 72, duration: '5m' },
             { target: 144, duration: '5m' }, { target: 288, duration: '5m' }] } },   // 1x,2x,4x,8x of 0.6 QPS
  thresholds: { 'http_req_duration{route:query}': ['p(95)<2500'], dropped_iterations: ['count==0'] },
};
export default function () {
  const q = qs[Math.floor(Math.random() * qs.length)];
  http.post(`${__ENV.BASE}/query`, JSON.stringify({ q, tenant: 'CL001' }),
            { headers: { 'Content-Type': 'application/json' }, tags: { route: 'query' }, timeout: '30s' });
}
```

Open loop means arrivals do not wait for responses, so a slow system builds a backlog the way
production does. `dropped_iterations` counts arrivals k6 could not start because every VU was
busy: the system fell behind. Keep a Locust twin (`wait_time = constant_throughput(...)`, which is
closed-loop) only to demonstrate coordinated omission in [SRV-01](#srv-01). Calling Locust's
behaviour coordinated omission is this doc's analysis; Locust's docs do not use the term. `wrk2
-t2 -c50 -d60s -R200 --latency URL` is an HTTP-edge alternative that measures latency from the
intended send time.

### K5 · Virtual clock and event replayer

```python
# constraints/replay/clock.py — ONE clock for every process (compose passes CLOCK_START and SPEED to all)
import os, time
EPOCH = float(os.getenv("VIRTUAL_EPOCH", "1735689600"))          # 2025-01-01T00:00:00Z
SPEED = float(os.getenv("SPEED", "1"))
START = float(os.environ.setdefault("CLOCK_START", str(time.time())))   # must be passed in, or each process starts its own clock
def now() -> float:  return EPOCH + (time.time() - START) * SPEED
def sleep_virtual(s): time.sleep(s / SPEED)
```

```python
# constraints/replay/replayer.py — emits events at virtual time, with dup / reorder / drop faults
import heapq, json, random, time
from clock import now
def replay(path, sink, dup=0.0, drop=0.0, reorder_s=0.0, seed=0):
    rng, buf = random.Random(seed), []
    def release(upto):
        while buf and buf[0][0] <= upto:
            t, _, ev = heapq.heappop(buf)
            while now() < t: time.sleep(0.001)
            sink(ev, emitted_at=now())                       # log emitted_at for freshness accounting
    for line in open(path):                                  # {"t","op","doc_id","logical_id","version","supersedes","acl"}
        ev = json.loads(line)
        if rng.random() < drop: continue                     # lost notification (ING-03)
        j = rng.uniform(0, reorder_s); heapq.heappush(buf, (ev["t"] + j, rng.random(), ev))
        if rng.random() < dup: heapq.heappush(buf, (ev["t"] + j + rng.uniform(0, reorder_s), rng.random(), ev))
        release(ev["t"])
    release(float("inf"))
```

**Freshness prober (`replay/prober.py`).** For each event, poll the serving path (filter
`doc_id == X AND version == v`, or a unique per-version marker token that generator v2 emits) until
the new version is visible. For supersede and delete events, poll until the predecessor is *absent*.
Record `t_visible − t_event` in virtual seconds as two series, new content and retired content
(M14's "two numbers, not one").

Why an explicit clock service: libfaketime (`FAKETIME="+0 x60"`) can be bypassed by Python through
vDSO, and the in-process libraries do not cross process boundaries. `freezegun.freeze_time(...,
tick=True)` also freezes `monotonic`; `time_machine.travel(..., tick=True)` does not. Use those two
for unit tests only. The mock provider runs its limits and latency in the same virtual time
(`SPEED`), so arrival and service compress together and ρ is preserved.

### K6 · Metrics collector

```bash
# constraints/metrics/collect.sh — 1 Hz container + cgroup snapshots (cgroup v2 namespaced paths)
while true; do ts=$(date +%s)
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.BlockIO}}' | sed "s/^/$ts,/" >> out/stats.csv
  for c in qdrant pg; do
    docker exec $c sh -c 'cat /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.stat' 2>/dev/null \
      | grep -E '^(oom_kill|anon|file|workingset_refault_file) ' | sed "s/^/$ts,$c,/" >> out/cgroup.csv
  done
  curl -s localhost:8080/metrics >> out/mockprov.jsonl; echo >> out/mockprov.jsonl
  sleep 1; done
```

- `docker stats` MEM USAGE subtracts `inactive_file`, so it understates the page cache an mmap store
  is using. Read `memory.stat` `file` as well.
- cgroup v2 has no plain `workingset_refault` key; use `workingset_refault_file`. `memory.high`
  throttles and never OOM-kills; `memory.max` kills, and `memory.events` `oom_kill` counts it.
- Also scrape Qdrant's Prometheus `/metrics`, `redis-cli INFO stats` (`keyspace_hits`,
  `keyspace_misses`, `evicted_keys`), `k6 --out json=...`, and OTel spans (Phoenix or Langfuse) for
  per-stage timings.

`metrics/report.py` produces the three standard plots: **knee curves** (metric vs constraint, log
x-axis), **p99 time series overlaid on events** (bursts, kills, brown-outs), and **true end-to-end
P95 vs the sum of stage P95s** (percentiles do not add; see [COST-01](#cost-01)).

### K7 · Prediction log

```markdown
<!-- constraints/predictions/E{NN}-{slug}.md -->
# E{NN} · {title}              ID: {GEN-01…OPS-01}   Module(s): {M#}   Failure(s): {F#}   Date: {yyyy-mm-dd}
## Hypothesis (one sentence, names a stage and a metric)
## Ratio preserved / ratio given up        e.g. P_mem = 3.3 preserved; HNSW hop count (log N) not preserved
## Napkin prediction                        value ± range, with the arithmetic shown
## Confidence (0–100%) and what would surprise me
## Setup                                    profile files, generator flags + seed, versions (pin!), commit
## Measured                                 value ± CI (n, bootstrap B), plot path
## Delta and explanation                    if |delta| > range: which assumption was wrong?
## What breaks next                         the next binding constraint, predicted
## Ledger impact                            which M#-n entry this confirms / reverses (decision · cost · why · reverses-if)
```

`predictions/log.csv`, one row per experiment:

```
id, date, point_id, predicted, low, high, measured, ci_low, ci_high, within_range, surprise_note
```

Your calibration score is the fraction of rows within range, tracked over time. The "Ledger impact"
line ties each result to the course's tradeoff ledger: every module's entries have the shape
*decision · cost · why accepted · what reverses it*, labelled `M#-n` (for example M14-3,
"continuous incremental sync with a per-document verification read").

### Proposed directory layout

```
RAG/
├── practising-scale-under-constraints.md      ← this doc
├── constraints/
│   ├── compose/base.yml
│   ├── compose/profiles/{ram-fit,ram-3x,cpu-1,cpu-prod-scale,io-cloud,tenant-1000}.yml
│   ├── mockprov/{app.py,Dockerfile,profiles/{default,ingest-backfill,brownout,judge,drift}.env}
│   ├── replay/{clock.py,replayer.py,prober.py}
│   ├── load/k6/{query_open.js,queries.txt}   load/locust/closed_loop.py
│   ├── chaos/{kill_workers.sh,toxics.sh,netem.sh,disk_full.sh,noisy_neighbour.sh,corrupt_parse.py,drop_acl.py,drop_filter.py,alias_swap.env}
│   ├── metrics/{collect.sh,report.py}
│   ├── experiments/E{NN}-{slug}/{run.py,config.yaml,README.md}   ← one per injection point actually run
│   └── predictions/{TEMPLATE.md,log.csv,E{NN}-*.md}
└── labs/
    ├── generate_corpus.py          ← v1, kept so existing module numbers stay reproducible
    ├── generate_corpus_v2.py       ← fact table, per-doc seed, shards, spans, events.jsonl, the GEN knobs
    └── eval_harness.py             ← listed in labs/README.md; does not exist today
```

---
<a id="injection-points"></a>
## Injection points, by layer

Each point uses one template. **Knob** is what you turn. **Exercises** is the phenomenon it makes
appear. **Course** gives the modules and concepts it tests. **Inject** shows how, with a snippet only
when the kit does not already cover it. **Miniature math** maps laptop to production. **Predict
first** is the number to write in the prediction log before you run. **Measure / Pass if** names
the curve and the bar. **Fixes to try** lists each fix with its expected signature in the
measurement and its cost. **Where the miniature lies** says what does not transfer.

---

<a id="layer-gen"></a>
### Layer GEN · corpus generation

The first thing that fails to scale is the harness. The current
[`labs/generate_corpus.py`](labs/generate_corpus.py) (v1) was re-run at seed 1337 while this doc was
written, so the following are counts of its output, not predictions. All 3,000 docs are `.md`. The
filler prose is drawn uniformly from a **28-word** list, which gives **253 distinct alphabetic words**
across 13.5M words (the same 253 in a 300-doc run). There are no spans and no revisions (9
bulletin supersession links), and 20 jurisdictions where the spine says 40. All 80 unanswerable
queries are one string. 56 of 80 lexical queries name a subclause that does not exist in their gold
doc. Every doc is held in a Python list until the end of the run. The points below turn v1's
defects into the knobs of a **generator v2**:

| v2 knob (proposed flag) | Point | Default to match the spine |
|---|---|---|
| streaming sharded output, `seed_i = H(global_seed, i)` | [GEN-01](#gen-01) | always on |
| `--vocab --zipf --jur-corr --length-profile` | [GEN-02](#gen-02) | realistic, c = 0.8 |
| `--revision-rate --revision-delta --boilerplate-share` | [GEN-03](#gen-03) | 0.1 / 0.03 / 0.3 |
| `--events --speed --burst --op-mix` → `events.jsonl` | [GEN-04](#gen-04) | 1,307 new + 1,333 changed per day |
| fact table → docs → questions, spans, multi-gold, `validate` | [GEN-05](#gen-05) | always on |
| `--formats pdf,docx,xlsx,html,md` | [PARSE-01](#parse-01) | the syllabus format table |
| `--jurisdictions 40 --clients 12 --tenant-dist zipf:1.2` | [FILT-01](#filt-01), [FILT-02](#filt-02) | 40 / 12 / uniform |
| `--acl` principals and `acl_change` events | [FILT-03](#filt-03) | on |
| `--identifier-density` (`14.2(b)`-style clause ids, UN codes) | [HYB-02](#hyb-02) | realistic df/N |
| `--poison k --placement` | [SEC-01](#sec-01) | off |

<a id="gen-01"></a>
### GEN-01 · Generator memory, streaming and per-doc seeding

**Knob:** RAM available to the generator; `--docs`. · **Exercises:** a "load everything, then
process" batch job dying at a size nobody tested; the same shape that later breaks
`SimpleVectorStore`, `SimpleDocumentStore` and `rank_bm25`. · **Course:** M14 (batch vs streaming,
deterministic ids).

**Inject:** run v1 under a cap and sweep `--docs` 3k → 10k → 30k → 60k → 100k, recording max RSS
(`/usr/bin/time -v` on Linux) and the exit code (137 = OOM-killed):

```bash
docker run --rm --memory=1g --memory-swap=1g -v $PWD:/w -w /w python:3.12-slim \
  python labs/generate_corpus.py --out /tmp/d --docs 60000
```

**Miniature math:** v1 appends every `Doc` to a list before writing anything. At the measured mean
of 4,513 words × ~8.5 bytes ≈ 38 KB of text per doc on disk. Most docs contain an em dash, so
CPython stores them at 2 bytes per character: ≈ **77 KB of string per doc** in RAM, and
1 GB ÷ 77 KB ≈ **13k docs** before OOM, ignoring interpreter overhead. At 480k docs the resident
text alone is 480k × 77 KB ≈ **37 GB**. A full-scale v1 run cannot finish on a 16 GB laptop,
whatever the RAG stack does.

**Predict first:** "At `--memory=1g`, v1 dies at ___ docs; RSS per doc ≈ ___ KB; v2 peak RSS at
480k ≈ ___ MB."

**Measure:** max RSS and wall-clock vs `--docs`. RSS should be linear for v1 and flat for v2. ·
**Pass if:** v2 generates 480k docs under `--memory=512m`, and doc #1234 is byte-identical between a
3k run and a 480k run.

**Fixes to try:**
- Stream each doc to `shard-NNNN/` as it is generated → RSS flat in N (→ no cost).
- Per-doc seeding, `seed_i = H(global_seed, i)` → doc *i* identical at any corpus size, and shards can
  be generated in parallel (→ "same seed, same corpus" must be redefined, because v1's single RNG
  stream is gone).
- Fact table first ([GEN-05](#gen-05)) → gold labels become joins (→ a schema to maintain).

**Where the miniature lies:** nowhere much. This is a real failure at real scale. Python string
overhead differs between versions, so do not quote 77 KB/doc as a constant.

<a id="gen-02"></a>
### GEN-02 · Prose entropy and metadata–text correlation

**Knob:** filler vocabulary size V and Zipf exponent; doc-length distribution per source type; the
correlation c ∈ [0, 1] between metadata (jurisdiction, client, carrier) and body vocabulary; share
of short docs. · **Exercises:** BM25 IDF becoming meaningful, embeddings forming clusters, and the
gap between p_n and p_c (a filter that admits 2.5% of the corpus but 35% of a query's neighbours).
· **Course:** M3 (doc length), M4 (similarity distributions, the "cone effect" where all embeddings
sit in a narrow cone), M5 (quantization error depends on the data), M6 (p_n vs p_c), M7 (dense vs
BM25 split); F2, F4.

**Inject:** replace v1's `_para()` (uniform draws from the 28-word list) with a grammar over the
fact table: `--vocab 5000 --zipf 1.1 --jur-corr 0.0..1.0 --length-profile realistic|uniform`.

**Miniature math:** nothing to scale. This knob is about realism. Target the **distinct terms per 1M
tokens** and type/token ratio of real legal prose. Heaps' law, V(n) ≈ K·n^β with β ≈ 0.4–0.6 for
English, is ⚠️ unverified as a target; calibrate on a real contract corpus. v1 has 253 words over
13.5M, which is effectively β ≈ 0. The generated median doc is 2,796 words, while M3 builds its
argument on a median "under 1,000 words".

**Predict first:** "With v1 text, M6's jurisdiction filter shows p_n/p_c ≈ ___ (expect ≈ 1). With
c = 0.8 it is ≈ ___."

**Measure:** distinct terms and type/token ratio; BM25 IDF histogram; cosine histogram of random
pairs (M4 expects a 0.6–0.8 floor); measured p_n per filter axis vs c. Then rerun M7's five-shape
comparison at c = 0 and c = 0.8. · **Pass if:** p_n ≈ p_c at c = 0 and p_n/p_c > 5 at c = 0.8;
otherwise the knob is broken.

**Fixes to try:** none. GEN-02 is the *precondition* for FILT-01/02, HYB-01/02, VEC-01/06 and
SEC-01. M6's correlated-filter result and M7's dense-vs-BM25 contrast cannot appear on v1 text at
all.

**Where the miniature lies:** templated text embeds into tighter clusters than real contracts, so ANN
looks *easier* than production. Calibrate absolute recall once against a public corpus (a BEIR legal
or finance set) before trusting it.

<a id="gen-03"></a>
### GEN-03 · Revision, near-duplicate and boilerplate density

**Knob:** share of docs that revise an earlier doc; % of text changed per revision; boilerplate
shared across jurisdictions. · **Exercises:** M14's opening, "two versions of clause 4.2, both
live"; near-duplicates eating top-k slots; dedup thresholds that do not transfer across embedding
models. · **Course:** M3 (overlap creates near-duplicates), M4 (threshold calibration), M9 (MMR,
maximal marginal relevance: diversify the top-k), M14 (supersession), M20 (nodes-per-`ref_doc_id`
audit); F2, F7.

**Inject:** `--revision-rate 0.1 --revision-delta 0.03 --boilerplate-share 0.3`. M14's example
revision is "97% the same text". v1 has none: carrier agreements are always `supersedes=None`.

**Miniature math:** hold the **duplicate occupancy of the top-k**. With r live versions per logical
doc and k = 10, near-duplicate occupancy ≈ min(k, r × hits). That does not depend on N, so 3k docs
with r = 2 waste the same slots as 480k with r = 2. M14 quotes a corpus "1.9× its stated size"
without supersession.

**Predict first:** "With no docstore, after replaying ___ revisions the corpus is ___× its logical
size. With two versions 0.004 apart in score, the old one wins ___% of runs."

**Measure:** near-duplicate share of top-10 (cos > τ); live versions per logical id; answer flip rate
across repeated runs when two versions score within 0.004 (M20 and final-exam C1, "two revisions
both current, 0.874 vs 0.870"). · **Pass if:** no logical id has more than one live version, and
the flip rate on revised clauses is 0.

**Fixes to try:**
- Docstore + stable `doc_id` per logical doc (ledger entry M14-1) → live versions = 1 (→ history
  lost).
- Supersession link + delete job → retired content gone within the retirement budget, which M14
  sets at 11 min (→ a reconciliation job that must itself be idempotent).
- Effective-date filter → correct only if every query path applies it; break one path
  ([FILT-04](#filt-04)) and watch it fail.
- MMR (`vector_store_query_mode="mmr"`) → duplicate share of top-k falls (→ the score scale changes
  and similarity cutoffs break, M9).

**Where the miniature lies:** real revisions renumber clauses and move tables. Templated revisions
keep the layout, so dedup looks easier than it is.

<a id="gen-04"></a>
### GEN-04 · Event timeline and replay speed

**Knob:** arrival process (Poisson vs bursty partner uploads of 2,000 docs); op mix (create, revise,
delete, `acl_change`, duplicate); per-source clock offsets; replay speed s (60× to 8,760×);
population sample φ. · **Exercises:** time as a resource. A year of corpus change becomes an hour,
so freshness, backlog and detection-latency experiments can run on a laptop. · **Course:** M14
(staleness window), M20 (drift), M22 (freshness row, "sustained ingest must exceed the peak arrival
rate by 2×").

**Inject:** v2 emits `events.jsonl` (`{"t","op","doc_id","logical_id","version","supersedes","acl"}`)
in virtual time, in generation order, with no faults. The replayer (K5) adds dup, reorder and drop
at replay time, so one timeline serves every fault experiment. Validators: versions increase per
`logical_id` in `t` order; every revise or delete references a logical id created earlier; daily
counts match the configured rates.

**Miniature math:**
- Production arrival: (480,000 − 3,000) ÷ 365 ≈ **1,307 new docs/day** plus M18's 40,000
  changes/month ≈ **1,333/day**, about 2,640 events/day ≈ 1.8/min.
- Time compression: 12 months at s = 8,760 is one real hour, which is **133 new docs/s**. At ~6k
  tokens each that is 800k tokens/s of embedding, beyond any local embedder.
- Population sampling: replay a fraction φ of logical docs and scale ingest capacity (workers, TPM)
  by φ as well, which preserves ρ = arrival × service time ÷ capacity. Choose
  φ ≤ (your measured embed tokens/s, [EMB-02](#emb-02)) ÷ 800k.

**Predict first:** "At s = 8,760 my pipeline needs φ ≤ ___ to keep up. A 2,000-doc burst at φ = 0.05
is ___ docs."

**Measure:** emitted vs intended event rate, and replayer lag (how late events are released) vs s.
· **Pass if:** replayer lag stays below 1% of the freshness SLO in virtual time at the chosen s.

**Fixes to try:** lower s, or lower φ → replayer lag falls (→ a longer run, or fewer absolute docs,
which breaks index-size effects: run the memory experiments separately at full miniature size).

**Where the miniature lies:** the virtual clock removes real jitter (cron drift, source-system
notification delays, GC pauses). Add a detection-lag distribution taken from your source systems.

<a id="gen-05"></a>
### GEN-05 · Gold-label construction and validators

**Knob:** how labels are built (derived from generated text, as in v1, or joined from a fact table,
as in v2); multi-gold allowed or not; spans emitted or not; validator strictness. · **Exercises:**
measuring the instrument before trusting it. A wrong gold label is indistinguishable from a
retrieval miss. · **Course:** M13 (golden set, circularity trap), M1-1 (the frozen stratified
60-query set), [labs/README.md](labs/README.md) (which promises "span-level gold answers" and labels
"exact by construction").

**Inject:** run a label audit over `questions.jsonl`. At seed 1337, v1 gives (counted while writing
this doc):

| Class (80 each) | Defect | Count |
|---|---|---|
| lexical | names a subclause absent from the gold doc (queries draw 1–6; docs have 1–3) | 56/80 |
| lexical | fails "subclause exists AND clause topic matches" | 80/80 |
| multi-hop | gold doc has no entry for the queried UN code (13 do not mention it at all) | 24/80 |
| semantic | gold is 1 of ~121 same-title wiki docs (research estimate, not re-counted) | — |
| numeric | query matches ~5.9 docs with ~5.8 distinct rates (research estimate, not re-counted) | — |
| unanswerable | all 80 are the same string, so effective n = 1 | 1 distinct |

v2 builds entities → facts → docs → questions, so every gold label is a join, multi-gold is listed
where several docs state the same fact, and spans are char offsets into the emitted text.

**Miniature math:** none. Label error is a rate and independent of N. On a strict criterion v1's
lexical + multi-hop error is (80 + 24) ÷ 160 = 65%, or (56 + 24) ÷ 160 = 50% on the
subclause-exists test alone. A "~35%" figure for v1 needs its criterion stated.

**Predict first:** "On v1, ___% of lexical gold labels are wrong; on v2, ___."

**Measure:** validator failures per class; span → text round-trip; doc *i* byte-identical across
corpus sizes. · **Pass if:** 0 validator failures on v2, and every span resolves to its answer
string.

**Fixes to try:**
- Fact-table-first generation → natural label error → 0 (→ the schema has to grow with every new
  query class).
- Validators in CI on the generator → a label regression fails the build (→ generator changes slow
  down).
- An adjudication sample with inter-rater agreement, for the classes a fact table cannot express →
  measured residual error (→ human hours).

**Where the miniature lies:** synthetic labels, even correct ones, are cleaner than human ones.
Injected noise ([EVAL-02](#eval-02)) and an adjudicated real-query slice are still needed.

---

<a id="layer-parse"></a>
### Layer PARSE · parsing and chunking

<a id="parse-01"></a>
### PARSE-01 · Parser throughput and the parser-version backfill

**Knob:** CPU cores for parsing (`--cpus`); parser version; pages per doc; output formats. ·
**Exercises:** a parser upgrade as a *backfill*, not a deploy; one that silently re-chunks an
unchanged corpus. · **Course:** M3 (extract → structure → split), M14 (`IngestionCache` is keyed on
content + transformation, not transformation *version*; "bump the cache collection name"), M20
(parser version on every node; the `parse-1.7.0` incident), M22 ("layers ordered by cost of
reversal"); F3, F7.

**Inject:** generator v2 emits real PDF/DOCX/XLSX/HTML. v1 emits only `.md`, although the syllabus
format table says PDF and the labs README says XLSX. Parse with `pypdf`/`openpyxl` under
`docker run --cpus=1` and record pages/s. Then "ship" a changed splitter as `parse-1.7.0` for 5% of
sources, as in M20.

**Miniature math:** carrier agreements are 25% of 480k = 120k docs at ~80 pages = **9.6M pages**.
At R = 20 pages/s/core (⚠️ placeholder; measure it), a full re-parse is 9.6M ÷ 20 = 480k core-s =
**133 core-hours**. The 3k corpus has 718 agreements ≈ 57k pages, so the factor to production is
120,000 ÷ 718 ≈ 167. A parser change also changes chunk text, which forces a re-embed
([EMB-01](#emb-01)) and re-derivation of eval node ids ([CIT-01](#cit-01)).

**Predict first:** "pypdf parses ___ pages/s on one core; a 480k re-parse is ___ core-hours ≈ ___
days on 8 cores."

**Measure:** pages/s/core; extrapolated core-hours; `IngestionCache` hit rate before and after the
bump; shift in the nodes-per-`ref_doc_id` distribution. · **Pass if:** the stale-cache bug
reproduces without the fix and disappears with it, and a "parser changed under an unchanged corpus"
alert fires within one ingest cycle.

**Fixes to try:**
- Version every transformation and namespace the cache by it → cache hit rate drops to 0 for
  affected docs instead of silently serving old output (→ a full re-run of those docs).
- Rolling re-parse into a second index + alias swap → golden-set recall unchanged during the
  migration (→ 2× storage while it runs).
- Parser-change alert (M20) → fires within one cycle (→ one more alert to own).

**Where the miniature lies:** generated PDFs are clean (no scans, font subsets or OCR), so real
parse rates are lower and more variable. Font-subset garbage is a separate injection
([PARSE-02](#parse-02)).

<a id="parse-02"></a>
### PARSE-02 · Extraction corruption (garbled pages, flattened tables)

**Knob:** fraction of pages garbled; table extraction mode (structured vs flattened). ·
**Exercises:** M19's "invisible incident": P95 falls, cost falls, faithfulness *rises*, and recall
on one stratum (the dangerous-goods, "DG", queries) collapses from 0.88 to 0.41. · **Course:** M3
(three table options), M19 (ingest assertions; the controls table), M13 (per-stratum recall), M20
(triage tree: "one stratum → F3"); F3, F6 → F1.

**Inject:** a transform at the parse stage (`chaos/corrupt_parse.py`):

```python
def corrupt(text, frac, rng):          # M19: pages 44–63 became plausible garbage
    if rng.random() < frac: return "".join(rng.sample(text, len(text)))
    return text
def flatten_table(md):                 # M3: lose row/column affiliation
    return " ".join(md.replace("|", " ").split())
```

**Miniature math:** hold the **share of a stratum's gold evidence** that is corrupted. M19 garbled 20
pages of one agreement, which held the DG stratum's evidence. In miniature, corrupt the docs that
hold that stratum's gold until the same fraction of gold evidence is lost, whatever N is.

**Predict first:** "Corrupting 20% of DG gold docs drops DG recall@10 from ___ to ___. The ingest
assertion catches ___% of them. Faithfulness moves by ___ (which sign?)."

**Measure:** detection latency per control, in virtual hours, on the K5 replay. · **Pass if:** the
per-document ingest assertion detects within one ingest cycle, and faithfulness sampling is shown
to miss.

**Fixes to try:**
- Per-document ingest assertion (text length per page, glyph coverage, language confidence) →
  detection < 15 min, ingest fails closed (→ about a day to build plus a 30-run baseline).
- Nightly per-stratum recall → ~12 h detection (→ the stratified golden set).
- Margin alert (`margin_top1_top2`, the score gap between first and second result) → should move
  before recall. M19 claims this without measuring it; test it.
- Faithfulness sampling → **does not fire**, because faithfulness improves.

**Where the miniature lies:** a random shuffle is easier to detect than real mojibake. Also plant
subtle corruption: swapped digits, a dropped table column.

<a id="parse-03"></a>
### PARSE-03 · Node-count explosion vs in-memory stores

**Knob:** node parser config; RAM; store type (`SimpleVectorStore`/`SimpleDocumentStore` vs a
server). · **Exercises:** framework defaults that hold everything in memory and persist one JSON
blob, falling over long before the vector math. Index memory follows node count, not token count.
· **Course:** M3 (expansion "paid in memory rather than quality"), M8 (docstore on the query path),
M5 (memory); F3, F8.

**Inject:** build M3's three configs under `--memory=2g --memory-swap=2g`:
`SentenceSplitter(512, 50)`, `HierarchicalNodeParser([2048, 512, 128])`,
`SentenceWindowNodeParser(window_size=3)`. Persist and reload the `StorageContext`; record peak
RSS, persist size and load time.

**Miniature math:** M3's pilot counts are 40,703 / 187,043 leaf vectors (4.6×) / 615,445 (15×; stored text ~7×
embedded). × 160 gives **6.5M / 29.9M / 98.5M** vectors at month 12. v1's `_para()` emits one
"sentence" of up to 680 words (280–680 in agreements, DG rules and customs manuals), so the window parser yields far fewer than 615k nodes on v1 (the
research estimates ~72k ⚠️ not re-run). Realistic sentence lengths need v2. Resolved 2026-11-13: M3 now embeds
leaves only — 187,043 vectors out of 243,340 hierarchical nodes (the old 217,495 split every level from the document).

**Predict first:** "Hierarchical at 300k leaf chunks needs ___ GB of docstore + ___ GB of vectors.
`SimpleDocumentStore` load time at 300k ≈ ___ s."

**Measure:** nodes, vectors, payload bytes, docstore bytes; load time vs N; retrieval p95 with the
docstore on the query path. · **Pass if:** the chosen config's projected 480k footprint fits the
memory tier in your M5 ADR.

**Fixes to try:**
- Server-backed docstore and vector store (Redis/Mongo/Postgres) → RSS flat in N (→ one more store
  in backup and restore).
- Embed leaves only → vectors = leaf count (→ parents fetched at query time).
- Window via `MetadataReplacementPostProcessor` → vectors = sentence count with no extra vectors,
  payload ~7× (→ payload memory).

**Where the miniature lies:** JSON (de)serialisation is CPU-bound and laptop CPUs are fast, so
absolute load times understate a small cloud VM. The ratio between configs transfers.

---

<a id="layer-emb"></a>
### Layer EMB · embedding

<a id="emb-01"></a>
### EMB-01 · Provider TPM/RPM during bulk ingest and re-embed

**Knob:** provider TPM, RPM, max tokens per request, `embed_batch_size`, client concurrency
(`num_workers`, `arun`), 429 handling (`Retry-After`). · **Exercises:** a re-embed as a throughput
and scheduling project that shares quota with production; unbounded `arun` producing retry storms;
concurrency past the limit buying nothing. · **Course:** M4 (`embed_batch_size` default 10 "can be
too small"), M14 (rate-limit levers, backfill as a project, dual-write + alias swap), M15 ("a
rollback path that takes nine hours is not a rollback path"), M22; F7, F8.

**Inject:** point `OpenAIEmbedding(api_base="http://localhost:8080/v1")` at K2. Sweep
`embed_batch_size` ∈ {10, 100, 2,048} × `num_workers` ∈ {1, 4, 16}.

**Miniature math:** wall-clock **T = max(tokens ÷ TPM, requests ÷ RPM)**. Concurrency is not in the
formula.
- Production: 3.61B tokens (M14's full re-embed with 1.25× overlap) at 1M TPM = 3,610 min = **60 h**;
  at 5M TPM, 12 h. M15's nine-hour rollback needs 3.61B ÷ 540 min ≈ **6.7M TPM**. M14's own
  "9 h" nightly rebuild implies the same ≈ 6.7M TPM sustained, which the module never states.
- RPM binds only when `batch × tokens per node < TPM ÷ RPM`. At 1M TPM and 3,000 RPM (⚠️ illustrative
  tier), TPM ÷ RPM = 333 tokens. With ~500-token chunks RPM never binds, even at batch 1.
- Dollars: 3.61B × $0.01/1M = **$36**, and $72 at $0.02. M14 printed $37,400 and $74,800, a 1000×
  error, fixed on 2026-11-13 ([Appendix A](#appendix-a)). The constraint is wall-clock, not money.
- Two ways to miniaturise; pick one per question:
  1. **Same limits, smaller work.** The 3k corpus embeds ~19.95M tokens (M3's `SentenceSplitter(512,
     50)`). At the production 1M TPM that is 20 real minutes. Multiply by 3.61B ÷ 19.95M = 181 to
     get 60 h. This keeps the per-request cap honest. Use it for throughput curves.
  2. **Same work ÷ quota.** Set TPM_mock = 1M × 19.95M ÷ 3.61B ≈ **5,530 TPM** and the miniature takes
     the same **60 virtual hours** (60 real minutes at `SPEED=60`). This shows how the backfill
     shares quota with live ingest over the same virtual duration. You must then scale the
     per-request cap by the same factor: 300,000 × 0.00553 ≈ 1,660 tokens ≈ 3 chunks per request.
     That pushes you toward RPM-bound behaviour production would not see.

**Predict first:** "Mode 1: 4 → 16 workers changes throughput by ___% and 429s by ___×. Mode 2: with
batch 100, every request is rejected as too large: true or false?"

**Measure:** tokens/min vs concurrency (it should plateau at TPM); 429 count; retry amplification;
live-ingest freshness p95 during the backfill (with [ING-01](#ing-01)). · **Pass if:** throughput
≥ 95% of TPM with < 1% of requests 429'd, and the live freshness SLO holds during the backfill.

**Fixes to try:**
- Larger batches, up to 2,048 inputs / 300k tokens per request → RPM stops binding and the curve
  flattens at the TPM line (→ one bad doc fails 2,047 others).
- Client-side token-bucket pacing → 429 rate ≈ 0 at equal throughput (→ code to own).
- A separate quota or project for backfill vs live ingest → live freshness p95 unchanged during the
  backfill (→ procurement).
- Rolling migration over days with a dual index and a gate before the swap (→ 2× storage).

**Where the miniature lies:** real providers use token buckets with undocumented burst allowances
and limits per model and per org; the mock is one clean bucket. Anthropic rounds remaining counts to
the nearest thousand. Real per-request latency grows with batch size; model it with `latency(20 +
0.02·n)` or measure it.

<a id="emb-02"></a>
### EMB-02 · Self-hosted embedding on a CPU budget ("$0", priced in hours)

**Knob:** `--cpus`, `torch.set_num_threads`, model size (bge-small 384-d, a 768-d model,
Qwen3-Embedding-0.6B), max sequence length, Matryoshka (MRL) truncation via
`HuggingFaceEmbedding(truncate_dim=...)`. · **Exercises:** self-hosting turning dollars into CPU
hours and wall-clock; local models saturating the machine before a provider would; query embedding
eating M18's 45 ms waterfall line. · **Course:** M4 (storage/compute/re-embed triple; MRL "up to
14×"; bge-small's 512-token max length as a disqualifier), M14, M18 (the embedding cache "can never
pay for itself on cost", only on latency); F4, F8.

**Inject:** `docker run --cpus=2 -e OMP_NUM_THREADS=2 ... python embed_bench.py --model
BAAI/bge-small-en-v1.5 --n 5000`. `llama-index-embeddings-huggingface` is **missing from
`labs/requirements.txt`**.

**Miniature math:** measure tokens/s/core = r. Full re-embed = 3.61B ÷ r ÷ 3,600 core-hours. At a
placeholder r = 2,000 tokens/s/core (⚠️ measure yours), that is ≈ 500 core-hours. For ingest, hold
**core-hours per day of corpus change**: ~2,640 docs/day × ~6k tokens ≈ 16M tokens/day ÷ r. For
query latency the number transfers directly, since one 60-token query on one core is the same work
in production.

**Predict first:** "bge-small does ___ tokens/s on 2 cores; the re-embed is ___ core-days;
truncating to 256-d loses ___ recall@10."

**Measure:** tokens/s vs cores (find where extra threads stop helping); query-embed p95; recall@10
vs truncation dim on M4's 60-query in-domain set. · **Pass if:** query-embed p95 ≤ 45 ms on the
capped budget.

**Fixes to try:**
- MRL 1,024 → 256 → storage 4× smaller, faster search (→ recall loss on tail queries; measure it).
- Smaller model → throughput up, recall down (→ measured on the in-domain set).
- Query-embedding cache → 45 ms → ~5 ms on hits (→ $144/month pod in M18, justified by latency only).
- Hosted for backfill, local for queries → **invalid**: two embedding spaces ([EMB-03](#emb-03)).

**Where the miniature lies:** Apple M-series and AVX-512 laptop cores are not cloud vCPUs. Calibrate
one run on the target instance before extrapolating core-hours. Long laptop runs throttle
thermally, which bends the throughput curve.

<a id="emb-03"></a>
### EMB-03 · Mixed-model index from a killed migration

**Knob:** fraction of the corpus re-embedded with model B when the migration is interrupted. ·
**Exercises:** "an index with two embedding spaces in it, which is the worst of all states" (M14). ·
**Course:** M4 (models "not interchangeable"; thresholds do not transfer), M14 (dual-write, atomic
cutover, cache namespaced per model), M20 (a model change is "never a hot-swap"); F4, F7.

**Inject:** start an in-place re-embed with no alias swap, `docker kill` the job at 30% and 60%, then
query with model A and with model B.

**Miniature math:** the migrated fraction is the knob, and it does not depend on N. 40% migrated at
3k behaves like 40% migrated at 480k.

**Predict first:** "At 50% migrated, recall@10 querying with A = ___, with B = ___."

**Measure:** recall@10 vs migrated fraction for each query model. · **Pass if:** with the fix, recall
stays inside the noise band ([EVAL-01](#eval-01)) for the whole migration.

**Fixes to try:**
- Dual-write + alias swap → recall flat; killing and retrying is safe (→ 2× storage during the
  migration).
- `embed_model_id` on every node, filtered at query time → a mismatch becomes visible instead of
  silent (→ one more mandatory filter).

**Where the miniature lies:** nowhere material. This one transfers exactly.

---

<a id="layer-vec"></a>
### Layer VEC · vector store: memory, disk, CPU, build

<a id="vec-01"></a>
### VEC-01 · RAM-to-index pressure: memory tiers and quantization

**Knob:** container RAM; vector placement (RAM / mmap); HNSW placement; quantization (none / int8 /
binary); rescore on or off; oversampling. · **Exercises:** page-cache thrash when mmap'd vectors
exceed RAM; a p99 *cliff*, not a slope; OOM-kill when vectors are forced into RAM; M22's "the memory
tier changes somewhere around 2M nodes". · **Course:** M5 (quantization ladder: int8 4×, binary 32×,
PQ 64×; oversample + rescore; ledger entry M5-2, which reverses "if originals are on slow storage and
rescore dominates latency"), M4 (bytes = 4·D·N), M22 (capacity table); F1 (silent recall loss from
quantization), F8.

**Inject:** K1 `ram-3x` / `ram-fit` profiles, then:

```bash
curl -X PUT localhost:6333/collections/chunks -H 'Content-Type: application/json' -d '{
  "vectors": {"size": 1024, "distance": "Cosine", "on_disk": true},
  "hnsw_config": {"m": 16, "ef_construct": 100, "on_disk": false},
  "quantization_config": {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": true}}}'
# Qdrant >= 1.19 replaced on_disk / always_ram with a memory mode (old flags deprecated, not removed):
#   "vectors": {..., "memory": "cold"},  "quantization_config": {"scalar": {..., "memory": "pinned"}}
curl -X POST localhost:6333/collections/chunks/points/query -H 'Content-Type: application/json' -d '{
  "query": [/* 1024 floats */], "limit": 10,
  "params": {"hnsw_ef": 128, "quantization": {"rescore": true, "oversampling": 2.0}}}'
```

To test 1M *real* vectors without paying to embed, load `KShivendu/dbpedia-entities-openai-1M`
(1M ada-002 1,536-d vectors, MIT licence, on Hugging Face).

**Miniature math:** P_mem = resident bytes needed ÷ (container RAM − baseline RSS). **Measure the
baseline first** with an empty collection. 300k × 4,096 B = 1.23 GB vs 22.9 GB in production
(18.7×). A 2 GB container ≈ a 37 GB node; a **425 MB** container (+ baseline) reproduces M5's 184 GB
on 64 GB (P = 2.9). int8 codes are 307 MB, so with originals on mmap the *quantized* working set
fits in ~512 MB while the originals do not, which is exactly M5-2's design. Qdrant's sizing rule of
thumb (vectors × dim × 4 B × 1.5) gives 1.84 GB at 300k and 34.4 GB at 5.6M.

**Predict first:** "float32 in RAM OOMs at ___ MB; mmap p99 doubles at ___ MB; int8 + rescore holds
p99 < 50 ms down to ___ MB."

**Measure:** recall@10 vs exact (`"params": {"exact": true}`) on 1,000 sampled queries; p50/p99 at a
fixed load; `memory.stat` `file` and `workingset_refault_file`; `memory.events` `oom_kill`; block IO.
Plot p99 and refaults/s vs container RAM from 4 GB down to 256 MB and find the knee. · **Pass if:**
the chosen config holds recall ≥ 0.95 vs exact with p99 inside its waterfall allocation (M18
line 5: 120 ms for hybrid retrieval) at the production P_mem.

**Fixes to try:**

| Config | Expected signature | Cost |
|---|---|---|
| float32 in RAM | fast until P > 1, then `oom_kill` increments | RAM |
| float32 mmap (`on_disk` / `cold`) | survives; p99 rises with refaults/s | disk-IO-bound latency |
| int8 in RAM + rescore from mmap | p50 near in-RAM; p99 tracks rescore page faults | recall vs exact −0.01 to −0.03 (M5's table) |
| binary + rescore, oversampling 2–4 | smallest RAM; recall depends on dims and centering (poor at 384-d) | oversampling × rescore IO |

**Where the miniature lies:** the fixed process overhead does not scale down, so subtract it. HNSW
hops grow ~log N, so per-query page touches are *fewer* in miniature and the real cliff arrives a
little earlier. On Docker Desktop "disk" may be the macOS page cache; run on Linux or combine with
[VEC-02](#vec-02).

<a id="vec-02"></a>
### VEC-02 · Disk bandwidth and IOPS caps under mmap and rescore

**Knob:** read IOPS and bandwidth on the store's data volume. · **Exercises:** "rescoring can
significantly decrease search speed if originals are on disk" (Qdrant, via M5); M5-2's reversal
condition; cold-start latency after a restart (M5 quotes a Turbopuffer cold p90 of 1,214 ms). ·
**Course:** M5-2; M18 (latency as the binding budget); F8.

**Inject (Linux, real block device):**

```bash
docker run --device-read-iops /dev/nvme0n1:500 --device-read-bps /dev/nvme0n1:20mb ... qdrant/qdrant
echo "259:0 riops=500 rbps=20971520" > /sys/fs/cgroup/<grp>/io.max        # cgroup v2 directly
```

**Miniature math:** a rescore reads k × oversampling random 4 KB pages (1,024-d float32). At k = 10,
oversampling 3, that is 30 reads per query: **60 ms** at 500 IOPS, 6 ms at 5,000 IOPS (⚠️ a typical
cheap cloud SSD; check your volume type). Hold **rescore reads per query ÷ volume IOPS** at the
production value. The IO cost does not depend on N, so this miniature transfers well.

**Predict first:** "At 500 IOPS and oversampling 3, p99 = ___ ms; turning rescore off saves ___ ms
and costs ___ recall."

**Measure:** p99 vs IOPS cap (sweep 100 → 10,000); rescore share of query time (traces);
time-to-warm after restart. · **Pass if:** p99 ≤ 120 ms at the production volume's IOPS.

**Fixes to try:**
- Lower oversampling → IO falls linearly (→ recall loss, measured).
- Pin originals in RAM for the hot tenant only (→ RAM for one tenant).
- Warm-up after restart (`pg_prewarm` for pgvector, or a warm-up query set) → cold p99 converges to
  warm p99 within N minutes (→ slower restarts).
- Skip rescore for non-compliance traffic (→ quantized-score recall on that traffic).

**Where the miniature lies:** the throttle may not bind at all on Docker Desktop (virtiofs is not a
block device). Reads already in the page cache bypass the throttle: drop caches
(`echo 3 > /proc/sys/vm/drop_caches`) or pair the IO cap with a small memory cap.

<a id="vec-03"></a>
### VEC-03 · CPU cap and the brute-force crossover

**Knob:** CPU quota; N; exact vs ANN. · **Exercises:** "the pilot does not need an ANN index" (M5),
and the N at which ANN becomes mandatory. · **Course:** M5 (exact search is O(N·D); "choose the
index for the corpus you will have"), M4 (distance cost linear in D, so MRL helps); F8.

**Inject:**

```python
import faiss, numpy as np
faiss.omp_set_num_threads(1)
xb = np.load("vecs.npy").astype("float32"); faiss.normalize_L2(xb)
flat = faiss.IndexFlatIP(xb.shape[1]); flat.add(xb)
hnsw = faiss.IndexHNSWFlat(xb.shape[1], 16, faiss.METRIC_INNER_PRODUCT); hnsw.add(xb); hnsw.hnsw.efSearch = 64
```

Run under `docker run --cpus=1`, and under `--cpu-period=10000 --cpu-quota=500` (0.05 CPU in 10 ms
periods) to emulate a larger N. FAISS is not in `labs/requirements.txt`.

**Miniature math:** exact search scans N·D·4 bytes per query and is memory-bandwidth bound. 22.9 GB at
~20 GB/s (⚠️ machine-dependent) ≈ **1.1 s per query** on one core, which breaks a 2.5 s budget once
anything else is added. 1.23 GB ≈ 60 ms. To make 300k behave like 5.6M on latency, give it 1/18.7 of
a core (`--cpus ≈ 0.054`). M5's own endpoint (29.9M × 1,536) is 46.0G multiply-accumulates per query.

**Predict first:** "Flat over 300k × 1,024 takes ___ ms on one core; HNSW (ef = 64) ___ ms; flat
crosses 120 ms at N ≈ ___."

**Measure:** latency vs N for flat and HNSW at fixed CPU on a log-log plot; read off where flat
crosses 120 ms. · **Pass if:** the extrapolation, checked against the 1M dbpedia set, predicts the
measured 1M latency within its error bar.

**Fixes to try:**
- HNSW → latency ~flat in N (→ recall < 1, and [VEC-04](#vec-04)'s build cost).
- Smaller D via MRL → exact latency ∝ D (→ recall, [EMB-02](#emb-02)).
- Pre-filter to a small subset, then exact (M6) → exact recall on small admitted sets (→ only works
  at low selectivity).

**Where the miniature lies:** a CFS quota is not a slower core. With small quotas a 20 ms task stalls
for whole periods, so p99 shows steps; shorten `--cpu-period`. A CPU quota also does not cut memory
bandwidth, and Apple silicon's bandwidth is far above a cloud vCPU's.

<a id="vec-04"></a>
### VEC-04 · HNSW build time and build-time dials (and pgvector `maintenance_work_mem`)

**Knob:** `m`, `ef_construct`/`ef_construction`, cores, `maintenance_work_mem`,
`max_parallel_maintenance_workers`. · **Exercises:** build dials you cannot change without a rebuild
("an under-built graph caps recall permanently", ledger entry M5-1); rebuilds that take hours at
scale; pgvector builds slowing sharply once the graph leaves memory. · **Course:** M5-1 (generous
`m`/`ef_construction`, tune only `ef_search`), M5 (dials table: "m: memory ~linear; No — rebuild"),
M14 (rebuild vs incremental; CleANN), M22 (capacity model); F1, F7, F8.

**Inject:**

```sql
-- pgvector 0.8.x, inside --cpus=2 --memory=2g
SET maintenance_work_mem = '64MB';            -- default; then '1GB'
SET max_parallel_maintenance_workers = 1;     -- default 2; then 3
\timing on
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
-- watch for: NOTICE: hnsw graph no longer fits into maintenance_work_mem after N tuples
```

For Qdrant, time upsert-complete → `status: green` in `GET /collections/chunks` per (`m`,
`ef_construct`).

**Miniature math:** HNSW build is ~O(N log N × ef_construct). Measure at 30k, 100k and 300k, fit
N log N, and extrapolate to 5.6M: 18.7 × log(5.6M)/log(300k) ≈ 18.7 × 1.23 ≈ **23×** the 300k build
time. Compare that with the freshness budget: M5 says "rebuild-every-15-minutes at 184 GB is an
outage, not a configuration". For the pgvector cliff, hold graph bytes ÷ `maintenance_work_mem`:
graph ≈ N × (vector bytes + m × 2 × 8 B) ⚠️ approximate, so 300k × ~4.3 KB ≈ 1.3 GB and 64 MB
spills almost at once. The NOTICE's tuple count tells you where.

**Predict first:** "pgvector HNSW at 300k: ___ min at 64 MB vs ___ min at 1 GB; extrapolated 5.6M
build: ___ h."

**Measure:** build seconds vs N and `m`; recall vs exact at fixed `ef_search` per (`m`,
`ef_construct`); recall drift after 10/30/50% churn by upsert and delete. · **Pass if:** an
extrapolated production rebuild time, with its error bar, is written into the ADR.

**Fixes to try:**
- Incremental upsert instead of rebuild → index stays green, recall vs exact stable as N grows (this
  tests CleANN's "quality degrades as updates happen" claim) (→ eventual compaction).
- Build offline, swap alias (→ 2× storage, the build still takes hours).
- Raise `maintenance_work_mem` → NOTICE disappears, build time falls by a measurable factor (→ RAM
  during builds).

**Where the miniature lies:** extrapolating N log N over 18.7× is about one decade, so the error bars
matter. Laptop cores are faster than cloud vCPUs; measure one point on the target instance.

<a id="vec-05"></a>
### VEC-05 · Ingest-while-serving: optimizer and indexing interference

**Knob:** concurrent upsert rate during query load; the CPU cap both share; Qdrant
`optimizers_config.indexing_threshold` (segments below it are brute-forced); pgvector inserts into a
live HNSW. · **Exercises:** query p99 spiking during ingest bursts ("freshness bought with latency");
recently upserted points in unindexed segments that are slower or not yet visible (M14:
"index-write lag… some stores index asynchronously"). · **Course:** M14 (staleness contributors),
M5 (FreshDiskANN), M22 (2× ingest headroom), M18 (the 190 ms unallocated line); F7, F8.

**Inject:** K4 open-loop queries at the busy-hour rate while K5 pushes a burst of new docs ("partner
bulk upload"), all under `cpus: 2`.

**Miniature math:** hold **burst work ÷ steady CPU headroom**. A production burst of 2,000 docs ×
11.7 = 23,400 vectors into 5.6M is 0.4% growth. The same 0.4% of 300k is **1,250 vectors**. Pushing
23,400 into 300k (7.8%) exaggerates optimizer work ~19×. Size the burst by fraction, and pick the CPU
cap so steady-state query utilisation matches the plan (e.g. ρ = 0.3).

**Predict first:** "A 0.4% burst at `cpus: 2` raises query p99 from ___ to ___ ms; draining takes
___ min; rate-limiting ingest to ___ upserts/s keeps p99 flat with freshness p95 at ___ min."

**Measure:** query p99 time series over upsert rate and optimizer status; write-to-visible latency
(upsert → first query that returns the point); backlog drain time. · **Pass if:** p99 within budget
and burst freshness p95 ≤ 15 virtual minutes.

**Fixes to try:**
- Separate ingest CPU (cpuset or a replica) → query p99 unchanged (→ idle capacity).
- Token-bucket the ingest worker → p99 flat while the burst drains slower and freshness p95 rises.
  That is the trade.
- Defer index build during bulk load (Qdrant's bulk-upload advice; ⚠️ exact current key unverified)
  → fast load followed by an index-build CPU spike.

**Where the miniature lies:** production can isolate query replicas from the ingest leader, while
single-node Qdrant on a laptop always shares CPU. State whether your production design separates
them.

<a id="vec-06"></a>
### VEC-06 · Query-time ANN dials and the recall-vs-exact noise floor

**Knob:** `hnsw_ef` / `hnsw.ef_search` / `efSearch`; `ivfflat.probes` / `nprobe`; index rebuild seed;
build threads. · **Exercises:** "approximate search does not fail loudly" (M5; its opening recall
loss of 0.83 → 0.71); the noise floor an index-drift detector needs (M20: "returned node-ID set for a
fixed query"). · **Course:** M5 (ef "changeable per query"; a CI check against exact on 1,000 vectors
per rebuild), M20, M13; F1.

**Inject:** sweep ef ∈ {16, 32, 64, 128, 256, 512}; pgvector `SET ivfflat.probes` ∈ {1, 4, 16, 64}
with `lists = √N`. Rebuild the same index twice with a multi-threaded build and compute Jaccard@10
of the result sets for fixed queries.

**Miniature math:** the ef needed for a fixed recall grows slowly with N (roughly log N). Build the
recall-vs-ef curve at 30k, 100k and 300k, read how the ef for 0.95 moves per decade, and extrapolate
one decade (18.7×). For IVF hold **vectors per list** constant (`lists ∝ N`) so the probe fraction
transfers.

**Predict first:** "ef for recall 0.95 is ___ at 30k and ___ at 300k, so ___ at 5.6M; Jaccard@10
between two rebuilds = ___."

**Measure:** recall@10 vs exact and p99 vs ef (the Pareto curve); Jaccard@10 between rebuilds (the
noise floor). · **Pass if:** the chosen ef gives ≥ 0.95 vs exact at the projected N with p99 inside
its allocation.

**Fixes to try:**
- Per-query ef, raised for compliance strata (→ latency on those strata).
- Rebuild check vs exact in CI → a recall drop beyond the noise floor blocks the swap (→ CI minutes).

**Where the miniature lies:** random vectors have no cluster structure and need far higher ef than
real embeddings. Use v2 embeddings or dbpedia-1M, never the mock's hash vectors.

---
<a id="layer-filt"></a>
### Layer FILT · metadata filters, multitenancy, permissions

<a id="filt-01"></a>
### FILT-01 · Filter selectivity sweep and the post-filter collapse

**Knob:** p_c ∈ {50%, 10%, 2.5%, 0.2%, 0.05%}; filter–vector correlation (GEN-02's `--jur-corr`);
execution path (post-filter, post-filter with over-fetch, pre-filter + exact, filter-aware HNSW);
planner thresholds. · **Exercises:** M6's cliff (post-filter recall 0.97 → 0.31 at p_n = 0.08); "an
index-wide recall@10 is not the recall any single tenant experiences". · **Course:** M5-3 (require
filter-aware traversal), M6-2/M6-3 (classify filter axes; measure p_n per axis), M6 (Qdrant planner,
pgvector iterative scans); F1 (over-filtering), F10 (missing filter).

**Inject:**
- Generator: `--jurisdictions 40 --clients 12` (v1 has 20 jurisdictions), plus a `tag` field with
  controlled cardinality.
- Qdrant: payload index (`PUT /collections/chunks/index` with
  `{"field_name": "jurisdiction", "field_schema": "keyword"}`), then vary
  `hnsw_config.full_scan_threshold`. Its unit is KB, and "1 KB = 1 vector of size 256", so at
  1,024-d float32 (4 KB per vector) a 10,000 KB threshold ≈ 2,500 vectors.
- pgvector 0.8+:

```sql
SET hnsw.ef_search = 40;                  -- default; "10% match → ~4 rows" (pgvector README, via M6)
SET hnsw.iterative_scan = relaxed_order;
SET hnsw.max_scan_tuples = 20000;         -- default; try 1000 / 100000
SELECT id FROM chunks WHERE client_id = 'CL001' ORDER BY embedding <=> $1 LIMIT 10;
```

**Miniature math:** expected post-filter survivors = k × p_n, so over-fetch k_fetch ≈ k ÷ p_n (M6).
These are fractions and hold at any N: 1/40 × 1/12 = 0.21% at 3k or at 480k. The **absolute count
admitted** does not hold: 0.2% of 300k is 600 vectors, while 0.2% of 5.6M is 11,200. With a planner
brute-force threshold of ≈ 2,500 vectors, the miniature falls back to exact search and production
does not. **Scale the threshold by ÷ 18.7** (≈ 134 vectors ≈ 535 KB), or the miniature will look
better than production. Also note that 38.4% of v1 docs are `client_id="ALL"` (all DG, customs and
bulletins). A strict `client = X` filter therefore excludes the DG corpus, and the real filter is
`client IN (X, ALL)` with p_c ≈ 0.384 + 0.616/12 ≈ 44%, not 8.3%.

**Predict first:** "Post-filter recall@10 at p_n = 0.03 ≈ ___ (ceiling ≈ 0.03). Iterative scan with
`max_scan_tuples = 20000` recovers to ___ at 300k and to ___ at 5.6M, because ___."

**Measure:** p_c and measured p_n per profile; recall@10 against **exact search restricted to the
filtered set**; p99 per path; plot recall vs p_n per path. · **Pass if:** every scope axis reaches
≥ 0.90 vs filtered exact at the production p_n. Expect M6's published post-filter numbers to fail
the k × p_n ceiling ([Appendix A](#appendix-a)).

**Fixes to try:**

| Fix | Signature | Cost |
|---|---|---|
| pre-filter + exact under the threshold | recall = 1 vs filtered exact; latency ∝ admitted count | brute force at mid selectivity |
| filter-aware HNSW (payload index, `payload_m`) | recall ≈ 0.9+ across the sweep | build cost, extra links |
| pgvector iterative scan | recall recovers until `max_scan_tuples` binds, then cliffs | "a scan budget rather than a filter" |
| over-fetch k ÷ p_n | recall recovers; candidate list grows to 125–1,000 and lands on the reranker ([RRK-01](#rrk-01)) | rerank latency |

**Where the miniature lies:** the planner thresholds above. v1 text gives p_n ≈ p_c on every axis, so
M6's "jurisdiction is correlated" result needs GEN-02.

<a id="filt-02"></a>
### FILT-02 · Tenant count, tenant skew and the noisy neighbour

**Knob:** number of tenants (12 → 64 → 1,000); size distribution (uniform vs Zipf, e.g. one tenant at
40% and a 1% tail); isolation model (shared collection + `is_tenant` index, collection per tenant,
Postgres partition per tenant); per-tenant query-rate skew. · **Exercises:** per-collection fixed
overhead ("each collection carries its own resource overhead", Qdrant); Qdrant Cloud's
1,000-collection cap and Milvus's 64-database default; "vectors from one tenant can affect recall
(and speed) for other tenants" (pgvector); small tenants starved by big ones. · **Course:** M6-4
(shared index + mandatory filter), M6 isolation profiles, M17-1 (shared index + the backstop, a
post-retrieval authorization check, vs per-tenant), M17-6 (recall penalty for narrow principals);
F10, F1. M17 never uses the term noisy neighbour and has no per-tenant fairness treatment.

**Inject:** `--tenant-dist zipf:1.2 --tenants 64` (v1 draws clients uniformly). Qdrant tenant index:

```bash
curl -X PUT localhost:6333/collections/chunks/index -H 'Content-Type: application/json' \
  -d '{"field_name":"tenant","field_schema":{"type":"keyword","is_tenant":true}}'
```

Compare with T collections under the same `mem_limit: 2g`. Noisy neighbour: the big tenant bulk-upserts
and queries heavily while you measure a small tenant's p99, with `chaos/noisy_neighbour.sh` in a
sibling container.

**Miniature math:** total memory = N_vectors × v + T × c, where c is the fixed cost per collection
(segments, an HNSW graph, quantization codebooks). Hold the **overhead share** T·c ÷ total. Measure c
directly from an empty collection and compute the crossover T* = N·v ÷ c at both sizes: it arrives
18.7× sooner in the 300k miniature. M6/M17's "12× storage" for per-tenant indexes holds only if docs
are duplicated per tenant. With 38.4% shared `ALL` docs it is 0.616 + 12 × 0.384 ≈ **5.2×** vectors.

**Predict first:** "Collection-per-tenant at 1,000 tenants uses ___× the RAM of the shared collection
at 300k; worst-tenant recall on a shared index without `is_tenant` is ___."

**Measure:** RSS and disk vs T per model; per-tenant recall@10 vs filtered exact, reporting the
**minimum** over tenants; small-tenant p99 during the big tenant's burst. · **Pass if:** worst-tenant
recall ≥ 0.90 and its p99 within budget.

**Fixes to try:**
- Shared collection + `is_tenant` → memory ≈ 1×, small-tenant recall stays high (→ one blast radius).
- Collection per tenant → isolation, overhead growing as T·c (→ caps on collection count).
- Big tenants isolated, long tail shared (Qdrant's guidance for "a smaller number of larger
  tenants") → both properties (→ two code paths).
- Per-tenant rate limits and admission control → small-tenant p99 flat during the big tenant's burst
  (→ a fairness policy to own).

**Where the miniature lies:** per-collection overhead on a laptop filesystem may differ from cloud
storage. The tenant query mix is invented; take the real Zipf from logs before trusting T*.

<a id="filt-03"></a>
### FILT-03 · Permission-change propagation: revocation storms and fail-closed ACLs

**Knob:** ACL grant/revoke rate; replay speed; propagation path (IdP claim TTL, node re-emit, cache
invalidation); fraction of nodes missing `acl`. · **Exercises:** permission drift (M20: "documents
are current. They are just no longer permitted."); a stale semantic-cache entry served after a
revoke (final exam C2: an entry created 40 minutes earlier by another client); the backstop that
"denies everything" when `acl` is missing at scale. · **Course:** M17 ("a cache is an authorization
decision you made earlier, on behalf of a different user"; "an ACL change is a content change"), M20
(permission drift, "sync SLO"), M22 scorecard ("sync SLO is informal"); F10. No module states a
revocation SLO.

**Inject:** add `acl_change` (document-side) and `principal_revoke` (user-side) events to the GEN-04
timeline, replay at `SPEED=600`, and include a storm of 1,000 revocations in one virtual minute.
Separately, drop `acl` from x% of nodes in a backfill (`chaos/drop_acl.py`) and run with and without:

```python
class RequireACL(TransformComponent):           # M17: fail closed at ingest
    def __call__(self, nodes, **kw):
        for n in nodes:
            if "acl" not in n.metadata: raise ValueError(f"missing acl on {n.ref_doc_id}")
        return nodes
```

**Miniature math:** time to enforcement = detection lag + re-emit lag + cache TTL + token/claim TTL.
These are durations and do not depend on N. What scales is **revocation burst ÷ ingest capacity**:
1,000 doc-side changes × 11.7 nodes = 11,700 node re-emits, compared with the steady capacity you
measure in [ING-01](#ing-01). A pure ACL change on unchanged text may not change the
`IngestionCache` key (content + transformation), so the nodes must be actively re-emitted.

**Predict first:** "With ACL-keyed cache purge off, ___% of post-revoke queries in the first 15
virtual minutes hit a stale entry; a 1,000-doc storm drains in ___ virtual min."

**Measure:** revocation → enforcement p95/p99 (virtual minutes); leaked answers after revoke (must be
0); storm drain time; % of queries returning "no authorized evidence" during the `acl`-missing
backfill. · **Pass if:** 0 leaks and propagation p99 ≤ a stated revocation SLO (propose ≤ 15 min).

**Fixes to try:**
- Query-time authorization against live claims (the backstop) → revocation effective on the next
  request regardless of index state (→ narrow principals lose candidates, FILT-01/02).
- ACL hash in the cache key, and purge on ACL change → zero cross-principal hits after a revoke (→
  hit rate, [CACHE-01](#cache-01)).
- Fail-closed ingest (`RequireACL`) → a missing `acl` becomes a coverage miss, never a leak (→
  ingest stalls on bad sources).

**Where the miniature lies:** a real IdP's token TTL and claim caching sit outside your system. Model
them as a parameter, not a measurement.

<a id="filt-04"></a>
### FILT-04 · Scope-leak fault injection: forgotten filters, inferred filters, extra engines

**Knob:** probability a request path omits `filters`; probability the auto-retriever's LLM infers the
wrong `client`; number of places that construct a query engine (main, `SubQuestionQueryEngine`
sub-engines, agent tool, graph retriever). · **Exercises:** "a permission bug does not appear in your
quality metrics. It appears only in your red-team report." Leaks via cache, trace store and
community summaries. · **Course:** M6 (`filters` is optional; auto-retrieval is "not an
authorization mechanism"), M17 ("the number of query-engines in your codebase is the number of
places the backstop can be forgotten"; its 12-vector leak table), M21, final exam C2 (backstop behind
a flag "off for performance reasons"); F10.

**Inject:** wrap engine construction and drop the filter on 0.1–1% of calls (`chaos/drop_filter.py`).
Make the `VectorIndexAutoRetriever`'s LLM the mock with `MOCK_CONTENT` returning another client id.
Run M17's 12-vector red-team script against A (filter only), B (backstop only), C (both).

**Miniature math:** leak probability per query = p_omit × P(cross-tenant node in top-k | no filter),
and the second term ≈ 1 − (own-tenant share)^k. Neither depends on N; it depends on k, so the
miniature is exact. At 1% omission and 8,500 queries/day that is up to 85 unfiltered answers a day.

**Predict first:** "At p_omit = 1%, A leaks on ___ per 1,000 queries; B and C on ___. Recall differs
by ___."

**Measure:** leaked nodes per 1,000 queries per configuration; recall and latency per configuration.
These will be identical, which is the lesson. · **Pass if:** C has 0 leaks across 10k randomized
requests.

**Fixes to try:**
- Unconditional backstop on every path through one factory → leaks = 0 at any p_omit (→ "one extra
  pass over ≤ 40 nodes", ~20 ms in M18).
- A CI leak test that fails when the backstop is removed (final exam B3) → the test turns red when
  you delete the postprocessor (→ a test to maintain).

**Where the miniature lies:** nowhere for leaks. The ranking and timing side channels (M17 vectors
11–12) need real traffic statistics.

---

<a id="layer-hyb"></a>
### Layer HYB · sparse and hybrid retrieval

<a id="hyb-01"></a>
### HYB-01 · BM25 index memory, query cost and rebuild time under churn

**Knob:** RAM; implementation (`rank_bm25`, pure Python, vs `bm25s`, which LlamaIndex's
`BM25Retriever` uses with a `Stemmer.Stemmer("english")`); N; churn rate. · **Exercises:** the sparse
index as a second ingest artifact that becomes the freshness bottleneck (M7-1); a library that is
fine at pilot and O(N) per query at scale. · **Course:** M7-1, M7-4 (cost of always-run hybrid), M14
(staleness contributors); F7, F8. M7 flags the BM25 freshness cost "and leaves it unpriced".

**Inject:** build both at 30k / 100k / 300k chunks under `--memory=1g`; record RSS, build time,
persist size, query p95. Replay 1% churn per 15 virtual minutes and rebuild each cycle, since BM25
"must be rebuilt when the corpus changes" (M7).

**Miniature math:** `rank_bm25.get_scores` scores every document, so query cost is O(N) with Python
objects per doc; `bm25s` uses sparse matrices. Hold **rebuild time ÷ freshness window**. If a 300k
rebuild takes 30 s, production at ~linear cost takes 18.7 × 30 s ≈ **9.4 min**, which is 62% of the
15-minute window before embedding has started.

**Predict first:** "rank_bm25 at 300k: RSS ___ GB, query p95 ___ ms (bm25s ___ ms); extrapolated
rebuild at 5.6M ___ min."

**Measure:** RSS vs N; query p95 vs N (log-log slope ≈ 1 for rank_bm25); rebuild time vs N; share of
the freshness budget. · **Pass if:** the BM25 update path uses ≤ 2 min of the 15-minute budget at
the projected N.

**Fixes to try:**
- Store-native sparse (Qdrant sparse vectors, Postgres `tsvector`/GIN, OpenSearch) → incremental
  updates, no full rebuild (→ a second engine's scoring differs from BM25's).
- Segment-and-merge (rebuild only the delta shard, merge results) → rebuild ∝ delta (→ cross-segment
  IDF skew).
- `bm25s` over `rank_bm25` → query p95 sub-linear in N (→ a new dependency).

**Where the miniature lies:** vocabulary grows sub-linearly with N (Heaps' law), so postings memory
per doc falls at scale. On v1's 253-word vocabulary this experiment means nothing; use GEN-02 text.

<a id="hyb-02"></a>
### HYB-02 · Identifier rarity, IDF drift and stemmer faults

**Knob:** document frequency of identifiers (clause ids like `14.2(b)`, UN codes, tariff codes) at
df ∈ {1, 10, 100, 1,000}; identifier-query share (5 / 15 / 30%); ingest vs query stemmer mismatch.
· **Exercises:** dense retrieval missing identifier queries (M7: dense 0.38 vs BM25 0.96 on
identifiers), a gap that widens or narrows with scale by identifier class; a stemmer mangling codes
silently. · **Course:** M7 (IDF, k1/b; "a regex over clause IDs and UN numbers is a two-line
feature"), M4 ("an identifier has no semantics to embed"), M15 (F4 in the weights vs the query side);
F4, F1.

**Inject:** v2 `--identifier-density`. v1 contains **zero** `\d+\.\d+\([a-z]\)` identifiers
(re-counted at seed 1337) and 10 UN codes, each in hundreds of DG docs. Fault: build `BM25Retriever`
with the English stemmer at ingest and a different one at query time.

**Miniature math:** IDF = log(N ÷ df). At pilot, UN3480 in ~40 of 187,043 chunks gives
ln(4,676) ≈ 8.5 (M7). If occurrences grow with N (every new DG doc mentions UN3480), df/N and IDF are
constant. If the identifier is rare (one contract's clause), df stays fixed while N grows, so IDF
rises. Hold **df/N per identifier class**.

**Predict first:** "Dense recall on df = 1 identifiers is ___, on df = 1,000 ___; a stemmer mismatch
drops BM25 identifier recall from ___ to ___."

**Measure:** recall@10 on the identifier stratum vs df and N, per retriever. · **Pass if:** the
identifier stratum is ≥ 0.9 at the projected df/N.

**Fixes to try:**
- Always-run hybrid with RRF (reciprocal rank fusion) → identifier stratum near the BM25 level (→
  hybrid cost on every query). LlamaIndex's `QueryFusionRetriever` hardcodes RRF k = 60 (src, per the
  research; ⚠️ not re-checked), so M7's "k = 2 → 0.78" row is not what the shipped config runs.
- Regex identifier router → identifier queries go sparse-only and skip the dense call (→ a
  classifier to maintain).
- Identifier-preserving tokenizer → the stemmer-fault drop disappears (→ custom analysis chain).

**Where the miniature lies:** real identifiers are messy ("UN 3480", "UN-3480"). Inject variants.

---

<a id="layer-rrk"></a>
### Layer RRK · ranking and reranking

<a id="rrk-01"></a>
### RRK-01 · Reranker CPU budget × candidate N × concurrency (and silent truncation)

**Knob:** `--cpus`, `torch.set_num_threads`, candidate N ∈ {6, 20, 40, 100}, `CrossEncoder`
`batch_size`, `max_length` ∈ {128, 256, 512}, concurrent queries. · **Exercises:** "reranker cost is
linear in N; recall gain is not"; the only stage "whose cost the *query* controls rather than the
*corpus*" (M9-2); over-fetch from FILT-01 landing here; the tail of a long chunk silently ignored
past `max_length`. · **Course:** M9-1, M9-2, M18 (rerank line; M18 cut 5, "rerank only on
low-margin queries"), M22 ("first cut under pressure"); F2, F8.

**Inject:**

```python
import torch, time; torch.set_num_threads(2)
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", max_length=512, device="cpu")
t = time.perf_counter(); ce.predict([(q, c) for c in cands[:40]], batch_size=16); print(time.perf_counter() - t)
```

In LlamaIndex use `SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L6-v2", top_n=6,
device="cpu", keep_retrieval_score=True)`. **Never the default model**, which is
`cross-encoder/stsb-distilroberta-base`, an STS model rather than a passage ranker (src). Truncation
probe: put the disambiguating clause beyond token 512 in a 600-token chunk.

**Miniature math:** cost ∝ N × pair length. Define s(N) as core-seconds per query; then
ρ = λ × s(N) ÷ cores. With s(40) = 0.4 core-s (⚠️ placeholder; measure it) on 2 cores at λ = 0.6 QPS,
ρ = 0.6 × 0.4 ÷ 2 = **0.12**, so no queueing. At M22's 4× load ρ = 0.48. Add M10's multi-query
fan-out (×4 rerank calls) and ρ ≈ 1.9, which is **unstable**. This transfers exactly once s(N) is
measured on the target CPU and λ is set to the same multiple. M9's 74 ms P50 for 40 candidates does
not state its hardware.

**Predict first:** "MiniLM-L6 on 2 CPU cores scores 40 × 512-token pairs in ___ ms (M9 prints 74 ms);
end-to-end p95 crosses 2.5 s at ___ QPS without multi-query and ___ with it."

**Measure:** s(N) p50/p95 vs cores; nDCG@10 and recall@N vs N; end-to-end p95 vs arrival rate at
fixed N (find the ρ → 1 knee); truncation loss. · **Pass if:** rerank p95 ≤ its waterfall
allocation (M18 line 7: 180 ms) at busy-hour λ × 2.

**Fixes to try:**
- N at the recall@N knee (M9: recall@40 0.94, recall@100 0.95) → recall flat past the knee while
  latency stays linear (→ the knee moves with filters).
- Margin-gated rerank (M18 cut 5: 40% of queries) → mean rerank cost −60%, p95 unchanged if the
  gating targets the tail (→ a margin threshold to calibrate).
- Smaller or quantized model; ⚠️ an ONNX backend exists in sentence-transformers ≥ 3 (check) → s(N)
  roughly halves (→ nDCG loss, measured).
- GPU → s(N) flat until the batch saturates (→ $/hour, a new M18 line).

**Where the miniature lies:** laptop CPUs (wide SIMD, big caches) beat small cloud vCPUs; calibrate
once. A GPU changes the curve's shape entirely.

<a id="rrk-02"></a>
### RRK-02 · Hosted reranker (or any remote stage): latency, 429s, timeouts, degraded mode

**Knob:** added RTT and jitter, 429 rate and timeout rate on the rerank hop; client timeout τ;
fallback policy. · **Exercises:** "per-query cost and a network hop in the latency budget" (M9 on
hosted rerankers); a dependency's tail becoming your tail; graceful degradation. · **Course:** M9
(the "no reranker" row as degraded mode), M18 line 7 (180 ms rerank API), M20's C2 failure-budget
class ("P95 > 2.5 s for > 30 min"); F2, F8.

**Inject:** route the rerank client through toxiproxy to K2's `/v1/rerank`:
`toxics.sh internet` (latency 180 ms, jitter 80) plus `hang-2pct` (2% of connections hang 3 s).

**Miniature math:** with timeout τ and fallback to retrieval order, stage p99 ≈ min(τ, remote p99)
and accuracy = (1 − f) × acc_rerank + f × acc_none, where f is the fallback rate. M9 gives 0.88
with MiniLM and 0.71 without, so each 1% of fallbacks costs 0.17 points. Both are rates, so N does
not matter.

**Predict first:** "With 2% 3-second hangs and no timeout, end-to-end p99 = ___; with τ = 400 ms it
is ___ and accuracy drops ___."

**Measure:** stage and end-to-end p95/p99; fallback rate; accuracy under degradation. · **Pass if:**
end-to-end P95 ≤ 2.5 s with the remote at 2% timeouts.

**Fixes to try:**
- Tight timeout + fall back to retrieval order → p99 capped at τ, accuracy −f × 0.17 (→ silent
  quality dips unless f is on a dashboard).
- Fall back to a local MiniLM → a smaller dip (→ CPU, [RRK-01](#rrk-01)).
- Hedged request after the p95 (⚠️ a pattern, not a vendor feature) → p99 down (→ ~5% more calls).

**Where the miniature lies:** toxiproxy draws per connection, but real provider tails are correlated
(incidents). Add a 10-minute brown-out in which every request is slow.

---

<a id="layer-qry"></a>
### Layer QRY · query understanding

<a id="qry-01"></a>
### QRY-01 · Fan-out tail amplification (sub-questions, multi-query, per-claim attribution)

**Knob:** per-call latency distribution (lognormal plus a heavy tail, e.g. 2% at 10×); fan-out N
(`num_queries`, sub-questions, attribution claims); `use_async`; provider RPM. · **Exercises:**
"sub-questions multiply, they do not divide" (M16); M7's `num_queries=4` taking p99 from 31 ms to
640 ms. · **Course:** M7-4 (`num_queries` "silently inserts a generation call"), M10 ("trees
multiply"; 1 + N + 1 LLM calls), M16 (`use_async` is "a latency decision disguised as a parameter"),
M12-2 (per-claim attribution, one call per claim), M18 (concurrency must carry the principal); F8.

**Inject:** K2 with `LAT_SIGMA=0.6 P_TAIL=0.02 TAIL_MULT=10 RPM=120 SPEED=1`.
`QueryFusionRetriever(..., num_queries=4)` vs 1;
`SubQuestionQueryEngine.from_defaults(query_engine_tools=..., use_async=True|False)` for
N ∈ {1, 2, 3, 5}; M12 attribution with 1 → 20 claims.

**Miniature math:**
- Serial: latency = sum of calls, so P95 ≈ N × single-call P95 when tails are correlated.
- Parallel: latency = max over N, so P(request slow) = 1 − (1 − p)^N. At p = 2%: N = 3 → 5.9%,
  N = 5 → 9.6%. Both exceed the 5% a P95 can absorb.
- Calls multiply too: RAG-Fusion 4 + 1, sub-questions 1 + N + 1, so RPM saturates at RPM ÷ calls per
  query QPS.
- These are properties of the distribution, independent of corpus size, so the miniature is exact.

M10's own trace is a check on the serial assumption: its children sum to 70.44 s inside a 66.49 s
query, so they cannot have run strictly in sequence ([Appendix A](#appendix-a)).

**Predict first:** "With p = 2% tails, N = 3 async has p95 = ___ ms, serial ___ ms; at RPM = 120 the
system 429s above ___ QPS with RAG-Fusion on."

**Measure:** p50/p95/p99 vs N, async vs serial; calls per query; 429 rate at fixed QPS. ·
**Pass if:** the decomposed path's p95 fits its SLO class (sync, or async per ledger entry M18-4).

**Fixes to try:**
- Async with a per-branch timeout and a partial answer → p95 ≈ the timeout (→ completeness, measured).
- Decompose only the strata that need it (M10 classifier) → mixture P95 holds if the expensive share
  is < 5% ([QRY-02](#qry-02)).
- Cap N (→ recall on multi-part questions).
- Batch claims into one attribution call → 20 calls → 1 (→ a longer prompt, [LLM-01](#llm-01)).

**Where the miniature lies:** the mock draws independently, but real tails are correlated across
concurrent calls from one client (shared pool, same backend), so independent draws flatter parallel
fan-out. Add a correlated-slowdown mode.

<a id="qry-02"></a>
### QRY-02 · Router misclassification and the mixture-P95 trap

**Knob:** misroute rate r (0 / 5 / 10 / 20%); expensive-path share f; selector type; router failure
(malformed or empty selection). · **Exercises:** misroutes sending cheap queries to expensive engines
and expensive ones to cheap engines ("a confident, well-cited, wrong answer", M10-3); "a router with
no fallback converts a classification failure into an error page". · **Course:** M10-3, M16-2
(reverses "if the router's misclassification rate exceeds the accuracy delta"), M18-3 (misroute pays
4.5×); F1, F8.

**Inject:** wrap the M10 defect classifier / `RouterQueryEngine` selector to return a random route
with probability r; have the mock return malformed JSON to `PydanticSingleSelector` 1% of the time;
set f by sampling strata in the load mix.

**Miniature math:** if the expensive path alone exceeds 2.5 s, mixture P95 ≤ 2.5 s requires
f + r_cheap→expensive < ~5%. M10's own sample has multi-part 13 + multi-hop 8 = 21 of 50 = **42%**
of queries on paths adding 2.1–2.6 s; the module calls this share 16%. "Neither can run on more than
a minority of traffic" understates it, because for P95 "minority" must mean under 5%. All fractions,
so they transfer exactly.

**Predict first:** "With f = 10% at +2.1 s, end-to-end P95 = ___ (above 2.5 s?); the largest f that
keeps P95 ≤ 2.5 s is ___."

**Measure:** end-to-end P95 vs f and r (heatmap); per-stratum recall vs r; error rate under router
faults. · **Pass if:** the sync class P95 ≤ 2.5 s at the measured r.

**Fixes to try:**
- Move the expensive path out of the sync SLO class (M18-4: async deep-analysis mode) → sync P95
  restored (→ a second product surface).
- Trained classifier instead of an LLM selector → router latency in ms and r measurable (→ labelled
  data).
- Fallback route on empty or malformed selection → error rate → 0 (→ some answers from the wrong
  path).

**Where the miniature lies:** the stratum mix is invented. Take f from 100–200 real queries, as the
M10/M16 exercises ask.

---

<a id="layer-llm"></a>
### Layer LLM · context assembly and generation

<a id="llm-01"></a>
### LLM-01 · Context-window, `num_output` and tokenizer caps

**Knob:** context window (2k / 4k / 8k / 16k); `num_output` (64–1,024); evidence chunk count; the
tokenizer used for budgeting vs the model's real one; response mode (`compact`, `refine`,
`tree_summarize`). · **Exercises:** F5's silent truncation of the lowest-ranked evidence; `compact`
turning into a refine chain (≈ ⌈tokens ÷ B⌉ calls); `refine` overwriting a correct early answer; an
answer cut off by a small `num_output`, "indistinguishable from a model failure" (M11-1); lost in the
middle. · **Course:** M11 ("`context_window` is an input limit, not a total"; mode table; M11-1,
M11-2), M9 (`LongContextReorder`), M1 (effective vs technical capacity), midterm C3 (evidence at
tokens ~6,100–6,700 of 16k); F5, F6.

**Inject:** a small local model makes the cap physical: Ollama with `"options": {"num_ctx": 4096}` or
`llama-server -m model.gguf -c 4096 -np 2`. Set `Settings.context_window=4096` and
`Settings.num_output=256`. Count LLM calls per response mode with `TokenCountingHandler` and
`CallbackManager` events. Tokenizer fault: leave `cl100k` token counting in place while generating
with a Llama-family model.

**Miniature math:** hold **evidence tokens ÷ window**. M18's 10.4k-token prompt carries ~6 × 600 =
3,600 evidence tokens ≈ 35%; in a 4,096 window the same ratio is ~1,430 tokens. M1's long-context
argument is window ÷ corpus: 1M ÷ 18.0M ≈ 5.6% (M1 prints 5.3%). Reproduce it with a 4k window over
~72k tokens of corpus (4,096 ÷ 72k ≈ 5.7%, roughly four or five v1 carrier agreements at ~15.8k
tokens each). Tokenizer error grows
with chunk count (M11).

**Predict first:** "At `num_ctx` 4,096 with 8 chunks, `compact` makes ___ LLM calls; with `cl100k`
budgeting on a Llama tokenizer the prompt overflows on ___% of queries."

**Measure:** LLM calls per query by mode; truncation rate; accuracy vs needle position (1..6) and
window; over/under-fill tokens vs chunk count under the tokenizer mismatch. · **Pass if:** zero
truncated answers on the golden set, and accuracy position-invariant within the noise band.

**Fixes to try:**
- Reserve `num_output` and budget in the model's own tokenizer → truncation and
  "context length exceeded" errors → 0 (→ fewer evidence tokens).
- Fill by value, then reorder (`[reranker, LongContextReorder]`) → accuracy up when the needle is
  mid-context (→ one more postprocessor).
- Response mode per query class (M11-2) → call count matches the formula (→ per-class config).

**Where the miniature lies:** small local models are far more position-sensitive than frontier
models. That is fine for *seeing* the mechanism and wrong for *sizing* it.

<a id="llm-02"></a>
### LLM-02 · Generation provider: TTFT vs prompt size, decode, shared TPM at the busy hour

**Knob:** the mock's latency model, TTFT = a + b × input tokens and decode = c × output tokens;
lognormal noise; one org-wide TPM shared by query, ingest, eval and judge traffic; 5xx and hang
probabilities; client timeout and retries. · **Exercises:** the query path queuing on the
**provider's** limits rather than your servers; the nightly eval or a backfill starving daytime
traffic; TTFT and decode behaving "as one coupled quantity under load" (M18, citing DistServe). ·
**Course:** M18 (latency waterfall lines 9–10; "size the batch for your cost budget and verify
latency at peak utilisation"), M13 (`workers` "is your concurrency lever against rate limits"), M22
(latency is the binding constraint); F8.

**Inject:** K2 with `TTFT_BASE_MS=150 TTFT_PER_1K=35 DECODE_MS=1.28 TPM=400000 RPM=3000 P_5XX=0.005
P_HANG=0.002`, then run an eval job ([EVAL-03](#eval-03)) on the same key at the same time. TTFT for
a 10.4k-token prompt is 150 + 35 × 10.4 ≈ 514 ms, close to M18's 520 ms.

**Miniature math:**
- M18 line 10 prints "500 output tokens at 5.2 ms/token = 640 ms", but 500 × 5.2 = **2,600 ms**.
  640 ms is 1.28 ms/token, or 123 tokens at 5.2 ms. At 2,600 ms, lines 9–10 alone take 3.1 s, over
  the whole 2.5 s budget ([Appendix A](#appendix-a)). The kit's `DECODE_MS` default is 1.28 so the
  mock reproduces the waterfall as printed. Run it at 5.2 once to watch the budget break.
- Busy hour: 0.6 QPS × 10.4k tokens × 60 s ≈ **374k input TPM**. The nightly golden-set run (400
  queries × 3 judge axes = 1,200 calls × ~4k tokens = 4.8M tokens) squeezed into 15 minutes adds
  **320k TPM** on the same quota: 694k demand against a 400k limit.
- Hold **peak TPM demand ÷ TPM limit**. That ratio, not QPS, sets the 429 rate. To keep the run
  cheap on a laptop, scale demand and limit by the same factor.

**Predict first:** "With eval running at noon, ___% of user requests 429; halving the chunk count
saves ___ ms of TTFT."

**Measure:** 429 rate; end-to-end P95 with and without the concurrent eval; TTFT vs prompt size;
frontier TPM by consumer (`X-Consumer`). · **Pass if:** busy-hour P95 ≤ 2.5 s with the eval running.

**Fixes to try:**
- Separate keys or projects for eval, backfill and live → live 429s → 0 (→ quota procurement).
- Schedule eval off-peak behind a client token bucket (→ slower feedback).
- Prompt diet (fewer or shorter chunks) → TTFT and cost fall with input tokens (→ recall, F5).
- Cascade to a cheap tier (M18's 62/38 split) → frontier TPM falls (→ escalated queries pay twice,
  in money and latency).
- Streaming → better perceived latency; TTFT becomes the user-facing number (→ total P95 unchanged).

**Where the miniature lies:** the latency model is one you chose. It cannot reproduce prefill/decode
interference or batching on the provider's GPUs. To *see* the coupling, run `llama-server -np 4` and
watch per-request decode slow as slots fill; its absolute numbers will not transfer.

<a id="llm-03"></a>
### LLM-03 · Nondeterminism, silent alias drift and the canary noise floor

**Knob:** batch-dependent variance at T = 0; temperature and top-p; a provider-side alias remap at
virtual time T; canary set size. · **Exercises:** model drift under a pinned identifier being "silent
by construction" (M20-2); an alert threshold "set by taste, not derivation" (M22 scorecard);
"`temperature=0` is a variance reducer, not a determinism switch" (M11). · **Course:** M11 (variance
protocol, worst-case exact match), M20 (vendor model events; M20-2 nightly shadow canary), M13 (noise
floor), midterm A10; F6.

**Inject:** K2 with `NOISE=0.02` (the payload flips "3.5" to "3.15" 2% of the time) and
`SWAP_AT=86400` (after one virtual day, answer style and the `model` field change). Locally, run
llama.cpp or Ollama with a fixed `seed` while varying concurrent slots (`-np`) to reproduce "another
user's request can change your logits" (M11).

**Miniature math:** the null distribution of divergence comes from R repeated runs with no change;
the alert threshold is its 99th percentile, and detection power depends on canary size n. At 1,200
queries over 9 strata ≈ 133 per stratum, SE = √(0.85 × 0.15 ÷ 133) ≈ 3.1 points per run, so a drop
measured against an unpaired baseline run has SE ≈ √2 × 3.1 ≈ 4.4 points. M20's C2 threshold of 5
points is only ~1.1 SE of that difference, and its C3 band of 2–5 points sits inside the noise.

**Predict first:** "Over 200 canary queries × 5 runs at T = 0 with noise 0.02, ___% of queries give
more than one distinct payload; detecting a 3-point shift needs ≥ ___ canary queries."

**Measure:** distinct payloads per query across R = 5 at T = 0 (M11 "Class A"; its table shows
values that are impossible for R = 5, see [Appendix A](#appendix-a)); null-divergence p99; detection
latency after the swap. · **Pass if:** the swap is detected within 24 virtual hours with ≤ 1 false
alarm per month.

**Fixes to try:**
- Threshold derived from the null distribution → false-alarm rate ≈ the α you chose, swap caught in
  one nightly run (→ canary compute).
- Pin explicit model versions and log `model_id` on every trace (→ forced migrations on deprecation).
- Validator + prompt-keyed cache for extraction ("get determinism from a validator and a cache", M11)
  (→ cache invalidation, [CACHE-02](#cache-02)).

**Where the miniature lies:** real provider nondeterminism has unknown structure, and the mock's
Gaussian is a stand-in. The method (null distribution → threshold) transfers; the threshold does not.

<a id="llm-04"></a>
### LLM-04 · Agent loops: cumulative-token blow-up and loop budgets

**Knob:** max steps, max cumulative tokens, max wall-clock; per-call latency; transcript growth per
step; tool result size. · **Exercises:** M16's agent at 3–5× tokens and 4–8× P95; P95 of 11.4 s,
"4.6× over the 2.5 s budget"; $53,634/month if every query were agentic. · **Course:** M16-2 (≤ 15%
routed share with a ceiling and a deterministic fallback), M18-4 (agent outside the SLO class), M21
(an agent turns F9 into an action problem); F8, F10.

**Inject:** a `FunctionAgent` or `ReActAgent` on the mock, scripted to call the retrieval tool k times
with a growing transcript. Record tokens per call. Compare a 5-step cap with a 40k cumulative-token
cap.

**Miniature math:** each call re-sends the transcript, so call n costs ≈ base + n·Δ and n steps cost
n·base + Δ·n(n+1)/2, which is **quadratic**. With base = 1.2k and Δ = 2k: 5 steps = 6k + 30k = 36k
tokens; 3 steps = 3.6k + 12k = 15.6k. Cutting 5 → 3 steps saves 57%, not 40%. Corpus-independent, so
exact.

**Predict first:** "A 5-step agent with 1.2k base and 2k Δ uses ___ tokens; cutting to 3 steps saves
___%."

**Measure:** tokens vs step; P95 vs mean steps; ceiling-trip rate; quality at each cap. ·
**Pass if:** p99 cost per query ≤ the routed-path budget.

**Fixes to try:**
- Cumulative-token ceiling + deterministic fallback → hard cost bound per query; revert if the trip
  rate exceeds 5% (M16 exercise) (→ answers truncated at the ceiling).
- Transcript compaction (⚠️ a design pattern) → per-step cost roughly flat (→ lost context).

**Where the miniature lies:** scripted step counts are not real agent behaviour. Run the real agent
on the mock with realistic tool outputs to get the step distribution.

<a id="llm-05"></a>
### LLM-05 · LLM work at ingest: graph extraction, enrichment, summaries

**Knob:** provider RPM/TPM for ingest; `num_workers`; `max_paths_per_chunk`; extraction failure rate;
`raise_on_error`; temperature. · **Exercises:** LLM-bound, rate-limited index builds; partial graphs
from errors that are "swallowed and logged"; "a rebuild may produce a different graph" (M8). ·
**Course:** M8-5 (an LLM call per chunk), M16-1 ($26,199 build; amortisation), M6-1 (enrichment token
bill), M17 (community summaries need intersection ACLs), M14 (derived structures:
rebuild-or-tombstone); F8.

**Inject:** `PropertyGraphIndex.from_documents(docs, kg_extractors=[SimpleLLMPathExtractor(llm=mock,
max_paths_per_chunk=10, num_workers=4)])` against K2 at `RPM=60 TPM=100000 P_5XX=0.03` (M16's "3% of
extractions silently failed"). Also run `SummaryExtractor` / `QuestionsAnsweredExtractor` in an
`IngestionPipeline` (M6 enrichment).

**Miniature math:** calls = chunks. M16: 4,032,000 calls, 4,935M input + 726M output tokens.
T = max(calls ÷ RPM, tokens ÷ TPM). At an illustrative 10,000 RPM and 2M TPM (⚠️ assumed tier):
max(403 min, 5,661M ÷ 2M = 2,830 min) ≈ **47 h**, TPM-bound. M8's "at `num_workers=4` wall clock is
not the problem; the token bill is" fails whenever a limit binds. The 3k miniature (~40k chunks) at
limits scaled by 40k ÷ 4.03M (≈ 100 RPM, ≈ 20k TPM) gives the same ~47 virtual hours.

**Predict first:** "At the scaled limits, extracting 40k chunks takes ___ virtual hours regardless of
`num_workers` above ___; with 3% swallowed failures, ___ of 12 carriers lose at least one DG
relation."

**Measure:** wall-clock vs workers (flat beyond the binding limit); tokens; entities missing after 3%
failures; triple-set Jaccard between two builds. · **Pass if:** the extrapolated build fits the
maintenance window and coverage loss is reported.

**Fixes to try:**
- Scope the graph to the subset that needs it (M8: DG + carrier agreements) → calls ∝ subset (→
  global questions outside it degrade).
- Schema-constrained extraction (`SchemaLLMPathExtractor(..., strict=True)`) → fewer output tokens (→
  a schema to maintain).
- `raise_on_error=True` + retry queue → missing-entity rate → 0 (→ a stalled build on poison input).
- Incremental extraction for changed docs only (M18 cut 1) → daily calls ∝ delta (→ entity coverage
  decays until a rebuild).

**Where the miniature lies:** mocked extraction output is not a graph. The mock measures throughput,
cost and failure handling only; measure quality with a small local LLM on a few hundred chunks.

---

<a id="layer-cit"></a>
### Layer CIT · grounding and citations

<a id="cit-01"></a>
### CIT-01 · Citation and node-id stability under re-chunk and re-ingest

**Knob:** re-chunk (1,024 → 512); re-ingest with the default random node ids vs deterministic ids;
`citation_chunk_size`. · **Exercises:** audit records and eval ground truth pointing at nodes that no
longer exist; "a delete targeting node ids from last week's log hits nodes that no longer exist"
(M14). · **Course:** M12-4 (carry `ref_doc_id` and a stable external id), M13 (the span → node rule is
re-derived on every re-chunk), M14 (deterministic ids are a precondition for parallel ingest), M17
(audit record fields); F7.

**Inject:** store citations keyed on `node.id_` from run 1. Re-ingest the same docs (the framework
default, "by default, the node ids are set to random uuids", M14) or re-chunk, then resolve the
stored citations.

**Miniature math:** none. With random ids the dangling fraction is 100% at any N. With ids derived
from (doc_id, offsets) it is 0% for unchanged text.

**Predict first:** "After a plain re-ingest, ___% of stored citations resolve."

**Measure:** dangling-citation rate; eval node-id mappings broken. · **Pass if:** 0 dangling for
unchanged docs.

**Fixes to try:**
- Deterministic node ids = H(doc_id, char_start, char_end, parser_version) → dangling 0 on unchanged
  spans (→ ids change whenever the parser does, by design).
- Span-level citations (char offsets) → citations survive re-chunking. This needs v2 spans
  ([GEN-05](#gen-05)); v1 has none (→ offset bookkeeping).

**Where the miniature lies:** nowhere.

---
<a id="layer-cache"></a>
### Layer CACHE · caching

<a id="cache-01"></a>
### CACHE-01 · Semantic response cache: capacity, TTL, threshold and query popularity

**Knob:** Redis `maxmemory` and `maxmemory-policy` (default `noeviction`); TTL; similarity threshold
τ; query popularity (Zipf α over U distinct intents); key composition (query, tenant, ACL hash,
`index_version`). · **Exercises:** the miss-ratio curve vs memory; M18's value cliff ("the knee is a
cliff, not a plateau"); evictions erasing the hit rate; a wrong answer "served fast and confidently
with citations attached". · **Course:** M18-2 (τ = 0.88, keyed on tenant + ACL hash, false-hit rate
measured, not assumed), M17 (the cache as a leak path), final exam A9 (key = query + entitlement +
index version, TTL ≤ freshness SLO), M19 (`cache_outcome` on traces); F7, F8, F10.

**Inject:**

```bash
docker run -d --name cache --memory=128m redis:7 redis-server --maxmemory 64mb --maxmemory-policy allkeys-lru
redis-cli INFO stats | grep -E 'keyspace_hits|keyspace_misses|evicted_keys'
```

The load generator draws queries from Zipf(α) over U paraphrase clusters. Plant **near-miss pairs**,
similar wording with different answers (this contract vs that one, the NL rate vs the DE rate), so
the false-hit rate is measurable.

**Miniature math:** hit rate depends on **capacity ÷ distinct intents (C/U)**, on α, and on **TTL ×
arrival rate per intent**. Hold all three. Production is 255k queries/month; the miniature is 25.5k
queries replayed at `SPEED=30` with the same α, U ÷ 10, C ÷ 10 and the same virtual TTL. The
false-hit rate f(τ) is a property of the embedding and the query pairs, not of N, so it transfers if
the near-miss share matches production. M18's value function is V(τ) = h × Q × c_gen − f × Q ×
c_wrong − C_infra, with c_gen = $0.0679, c_wrong = $12 (fiction) and C_infra = $144.

**Predict first:** "At 64 MB the cache holds ___ entries at hit rate ___; at τ = 0.88 the false-hit
rate on near-miss pairs is ___; the ACL hash in the key costs ___ points of hit rate."

**Measure:** hit rate vs `maxmemory` (the miss-ratio curve); false-hit rate vs τ on planted pairs;
V(τ) from measured h and f; `evicted_keys`. · **Pass if:** measured f ≤ 2× the assumed f at the
chosen τ (M18-2's reversal trigger). Before trusting M18's τ table, note that its hit rate *rises*
with τ while the text calls raising τ "loosening" ([Appendix A](#appendix-a)). Your measurement
settles which way it goes.

**Fixes to try:**

| Fix | Signature | Cost |
|---|---|---|
| larger cache (C/U ↑) | hit rate climbs the miss-ratio curve, then flattens | RAM ($144/month pod in M18) |
| stricter τ | false hits fall faster than hits | savings |
| tenant + ACL hash + `index_version` in the key | cross-tenant hits = 0; hit rate falls by the shared fraction | fewer hits |
| TTL ≤ 15 min | stale-hit window ≤ freshness SLO | hit rate on slow-changing content |

**Where the miniature lies:** U and α are invented; the real repetition rate is unknown (M18's
MeanCache figure of ~31% describes one service's workload). LLM-generated paraphrases are more alike
than real users' rewordings, which inflates hits.

<a id="cache-02"></a>
### CACHE-02 · Cache invalidation on supersession, deletion and model change

**Knob:** which events purge which cache (response cache, `IngestionCache`, embedding cache, rerank
cache); purge latency; TTL. · **Exercises:** capstone challenge 2, where the old version "must stop
being retrievable within 15 minutes, *including from your caches*"; M14's "deleting from the index
is not deleting the document" (`IngestionCache` is not reached by `delete_ref_doc`, so a re-ingest of
a deleted doc "can be served entirely from cache"). · **Course:** M14 (deletion inventory,
supersession lag), M18 (three caching layers), M17 (the right-to-be-forgotten chain includes
caches), final exam A9; F7, F10.

**Inject:** in the K5 replay, emit `revise` and `delete` events for docs that back currently cached
answers. Query them immediately and every virtual minute afterwards.

**Miniature math:** stale-serve window = min(TTL, purge latency), a duration independent of N. Share
of cached answers affected per day = (docs changed per day ÷ docs) × cached answers per doc. Hold the
change fraction: 2,640 ÷ 480k = 0.55% per day.

**Predict first:** "Without doc-keyed purge, a superseded answer stays served for ___ min after the
event (= TTL)."

**Measure:** time from supersede event to the last stale cached answer (p99, virtual); stale serves
per 1,000 queries. · **Pass if:** p99 ≤ 15 virtual minutes, and 0 stale serves after the purge.

**Fixes to try:**
- Cache entries carry their `ref_doc_id`s and `index_version`, purged by doc on change → stale serves
  → 0 after the purge latency (→ a reverse index from doc to entries).
- TTL ≤ 15 min as a backstop → worst-case staleness bounded (→ hit rate).
- Namespace the `IngestionCache` by transformation and model version (M14) → no stale embeddings
  after a model change (→ a cold cache after every change).

**Where the miniature lies:** nowhere material. This is a correctness property.

---

<a id="layer-ing"></a>
### Layer ING · ingestion, sync, freshness, deletion

<a id="ing-01"></a>
### ING-01 · Accelerated event replay: freshness under polling vs push, bursts and backlog

**Knob:** replay speed s; arrival process (Poisson vs bursty); detection mode (poll every P minutes
vs push/webhook); ingest capacity (workers, provider TPM); per-doc processing time t_proc. ·
**Exercises:** "staleness is a latency, not a boolean", with its five additive contributors (M14);
detection lag dominating once ingest is fast; backlog growth when ρ > 1; retired content lagging
behind new content (M14: 8 min vs 11 min). · **Course:** M14 (staleness window, state machine,
ledger entry M14-3, worked-example rows), M22 (freshness row; 2× rule), M21 (faster ingest means
poison lands faster), M16-3 (nightly community summaries up to 24 h stale); F7.

**Inject:** K5 replays GEN-04's `events.jsonl` at speed s into a source directory or queue; ingest
polls or subscribes; the prober records two freshness series. Every component reads K5's clock, and
the mock provider's limits run in virtual time, so arrival and service compress together.

**Miniature math:**
- Hold ρ = λ_events × t_proc ÷ workers. Production λ ≈ 1.8 events/min on average.
- Polling: lag ~ U(0, P) + t_proc, so p95 ≈ 0.95P + t_proc. With P = 15 min, p95 ≤ 15 min needs
  t_proc < 45 s, and p99 ≤ 15 needs t_proc < 9 s. M22's "14 min p99" with a 15-minute ingest cycle
  is therefore impossible ([Appendix A](#appendix-a)).
- Bursts: 2,000 docs × ~6k tokens = 12M tokens = 12 min of embedding at 1M TPM, so the burst's tail
  breaks freshness unless capacity is ~2× peak (M22).
- Population sampling (GEN-04): replay a fraction φ while scaling capacity by φ. ρ is preserved.

**Predict first:** "Poll P = 15 min, t_proc = 70 s, so p95 = ___ (verdict on the SLO?); a 2,000-doc
burst at TPM ___ drains in ___ min."

**Measure:** freshness p50/p95/p99 for new content and **separately** for retired content (M14: "two
numbers, not one"); backlog depth over time; burst drain time; tokens per day. · **Pass if:** both
series are ≤ 15 min p95 including the burst.

**Fixes to try (M14's designs, with their expected signatures):**

| Design | Expected signature |
|---|---|
| nightly rebuild | staleness p95 ≈ 0.95 × 24 h + the 9 h 20 m rebuild ≈ 32 h (M14 prints 12 h); tokens/day = full corpus |
| hourly incremental, no vector store attached | new-content lag ~60 min; **supersession never happens** |
| continuous + docstore + vector store | p95 in minutes; supersession implicit (history lost) |
| continuous + supersession links + verification read | new ≈ 8 min, retired ≈ 11 min (M14), plus verification reads |
| push instead of poll | the 0.95P term disappears |

**Where the miniature lies:** the virtual clock has no cron drift, notification delay or GC pauses;
add a detection-lag distribution from your sources. Population sampling breaks anything that depends
on absolute counts (VEC-01, VEC-04), so run memory experiments separately at full miniature size.

<a id="ing-02"></a>
### ING-02 · Worker crash mid-ingest: idempotency and duplicate creation

**Knob:** kill rate; kill point (after embed before upsert; mid-batch; after upsert before the
docstore write); node-id scheme; docstore attached or not. · **Exercises:** "a retried batch produces
different node ids and duplicates rather than deduplicating" (M14); "a continuous worker with no
idempotency re-ingests on every restart"; a partially failed run that "leaves the index in a state no
one can describe". · **Course:** M14 (two invariants: idempotent keyed work and an observable
transaction boundary; M14-1, M14-3), M20 (nodes-per-`ref_doc_id` audit); F7, F2 (duplicates eat
top-k slots).

**Inject:** `chaos/kill_workers.sh`, or in-process `if rng.random() < P_KILL:
os.kill(os.getpid(), signal.SIGKILL)` at named checkpoints, with `restart: on-failure`. For
ambiguous acks without process death, add toxiproxy `reset_peer` on the vector-store connection.

**Miniature math:** expected duplicates ≈ kills × in-flight batch size × P(kill after upsert, before
ack), which depends on batch size and kill rate, not N. Hold **kills per 1,000 docs**. Production
failure rates are unknown, so sweep 0.1–5%.

**Predict first:** "With random UUIDs, batch 64 and one kill per 500 docs, ingesting 25k docs creates
___ duplicate nodes."

**Measure:** total vectors vs expected; docs with an unexpected node count; near-identical vector
pairs (cos > 0.999); retry token spend. · **Pass if:** after 1,000 random kills the invariants hold:
"index size = Σ expected nodes" and "no `ref_doc_id` has two live versions".

**Fixes to try:**
- Deterministic ids + upsert semantics → duplicates = 0 at any kill rate (→ an id scheme stable
  across parser and model changes).
- Per-document verification read (`Indexed → Verified` in M14's state machine) → partial docs are
  quarantined, never half-served (→ about 2× ingest wall-clock, M14-3).
- Docstore + vector store attached (without a vector store the pipeline "can only check for and
  remove duplicate inputs") → hash-unchanged docs are skipped on retry, so retry cost falls (→ a
  docstore to back up).

**Where the miniature lies:** if anything it understates production, where network partitions cause
ambiguous acks with no process death.

<a id="ing-03"></a>
### ING-03 · Duplicate, out-of-order, late and lost events

**Knob:** duplicate rate; reorder window (virtual seconds); lost-event rate; delete-before-create;
revise-before-create. · **Exercises:** "a revised contract with a new filename carries no
supersession signal" (M14); "F7's most common form is a delete that silently did nothing" (M20); an
older revision arriving after the newer one and winning under last-write-wins on arrival time. ·
**Course:** M14 (four operations: insert, update, dedup, supersede; "every path re-queries"), M20
(triage tree, F7 branch), final exam C1; F7.

**Inject:** replayer `dup=0.05, reorder_s=300, drop=0.01`. Drops model the source system never
notifying you.

**Miniature math:** all rates, so N-independent. Lost events need a **reconciliation sweep** with
period R. The expected staleness of a lost change is ≈ R ÷ 2, so R is the knob that bounds it.

**Predict first:** "With last-write-wins on arrival and a 5-minute reorder window, ___% of revised
docs end with the *older* version live."

**Measure:** live-version violations; stale-serve duration for lost events; invariant failures. ·
**Pass if:** 0 violations at 5% dup / 5-minute reorder, and lost-event staleness ≤ R.

**Fixes to try:**
- Version-ordered upsert (compare `version` or `effective_date`, not arrival time) → reordering
  leaves the newest version live (→ every source must carry a version).
- Idempotent handling keyed on (doc_id, version) → duplicates are no-ops (→ a processed-keys store).
- Periodic full reconciliation (source listing vs docstore) → lost events healed within R (→ a full
  listing every R).
- Tombstones for deletes that arrive before creates (→ tombstone retention).

**Where the miniature lies:** real sources have anomalies the replayer will not invent, such as clock
skew between source systems. Add per-source clock offsets in GEN-04.

<a id="ing-04"></a>
### ING-04 · Deletion and right-to-be-forgotten completeness across stores

**Knob:** which stores the delete job reaches (vector store, docstore, index store,
`IngestionCache`, response cache, rerank cache, graph/summaries, traces, eval dumps, backups); the
`delete_from_docstore` flag. · **Exercises:** "the default is the non-compliant option" (M14); "you
cannot delete what you cannot enumerate" (M17); tree indexes that do not support deletion; an ANN
tombstone that may not be physical removal. · **Course:** M14-2 (deletion as an explicit, idempotent,
verified job over an enumerated store inventory), M17 (lineage must reach generated artifacts), final
exam B3; F7, F10.

**Inject:** ingest a canary doc with a unique token (`CANARY-7f3a`). Push it through every path:
query it, cache the answer, trace it, include it in a nightly eval dump, build a community summary.
Then run `index.delete_ref_doc(doc_id)` with the **default** `delete_from_docstore=False`, then with
`True`, and grep every store for the token.

**Miniature math:** none; this is a correctness property. The scale is in the *inventory*, which
grows with architecture features (each cache, replica or derived structure adds a row), not with N.

**Predict first:** "After the default `delete_ref_doc`, the canary survives in ___ of ___ stores."

**Measure:** stores still holding the token after each delete variant; time to complete. ·
**Pass if:** 0 live stores hold the token and the proof list is generated automatically.

**Fixes to try:**
- Inventory-driven delete job with a verification read per store → canary count 0 in every live store;
  backups age out in N days (→ the inventory must be maintained as the architecture grows).
- Never log node text (M17-3 / M19-1 retention split) → traces never contain the token (→ harder
  debugging).

**Where the miniature lies:** physical removal inside vector stores (segment compaction) and backup
expiry take real time that the replay clock can hide. Measure compaction lag in wall time.

<a id="ing-05"></a>
### ING-05 · Ingest parallelism memory: `num_workers` × per-process RSS

**Knob:** `pipeline.run(documents=..., num_workers=k)` for k ∈ {1, 2, 4, 8}; container memory; a local
embedding model in-process vs remote. · **Exercises:** "local embedding models saturate the machine
before the provider does"; scaling ingest by workers running out of RAM before CPU. · **Course:** M14
(levers table: `multiprocessing.Pool`, "process-per-worker memory multiplies"); F7, F8.

**Inject:** `--memory=4g`, an in-process HuggingFace embedding model, and `memory.events` `oom_kill`
watched as k rises.

**Miniature math:** RSS ≈ base + k × (model weights + batch working set), since each pool process
holds its own copy. OOM at k* = (RAM − base) ÷ per-worker size. Independent of N.

**Predict first:** "Each worker adds ___ MB; OOM at k = ___ under 4 GB."

**Measure:** RSS vs k; docs/min vs k; the OOM point. · **Pass if:** the chosen k leaves 30% RAM
headroom.

**Fixes to try:**
- Embedding as a separate service (one model copy, batched) → RSS flat in k (→ a service to run).
- `arun` with bounded concurrency (→ one process's CPU).
- Smaller per-worker batches (→ more requests, [EMB-01](#emb-01)).

**Where the miniature lies:** fork copy-on-write may share model pages until they are written, so
results depend on the start method (`fork` vs `spawn`; macOS defaults to `spawn`).

<a id="ing-06"></a>
### ING-06 · Disk capacity, inode and file-count limits

**Knob:** volume size, inode count, files per directory, `ulimit -n`. · **Exercises:** a disk that
fills during an optimizer merge or snapshot, leaving a partially applied upsert batch; too many open
files at high segment counts; `SimpleDirectoryReader` slowly walking 480k files in one directory. ·
**Course:** M14 (`Indexed → Verified`, which exists for exactly the "pipeline said OK, store rejected
it" class), M19 (ingest assertions); F7.

**Inject:**

```bash
docker run --tmpfs /data:rw,size=300m ...                         # disk full mid-ingest
docker run --ulimit nofile=256:256 ...                            # open segment files / sockets
truncate -s 2G img && mkfs.ext4 -N 20000 img && sudo mount -o loop img /mnt/small   # Linux: few inodes
```

**Miniature math:** v1 writes one file per doc into one directory: 480k files plus 480k manifest
rows at month 12. Hold **free space ÷ largest compaction**, because an HNSW rebuild or segment merge
temporarily needs extra space.

**Predict first:** "With 300 MB of disk, ingest fails at ___ chunks; without verification, ___ docs
are reported indexed but return nothing."

**Measure:** docs reported ingested vs docs actually queryable after the disk fills. · **Pass if:**
zero docs in `Indexed` that fail the verification query.

**Fixes to try:**
- Per-document verification read → failed docs land in `Quarantined` and the predecessor stays live
  (→ ~2× ingest wall-clock).
- Sharded output directories (GEN-01) (→ none).
- Disk-headroom alert at 2× the largest compaction (→ idle disk).

**Where the miniature lies:** tmpfs is RAM-backed and fails instantly, while real disks degrade
slowly (latency rises before ENOSPC).

---

<a id="layer-srv"></a>
### Layer SRV · serving and query traffic

<a id="srv-01"></a>
### SRV-01 · Open-loop load, coordinated omission and the queueing knee (M22 Challenge B)

**Knob:** arrival rate λ (multiples of the busy-hour rate); burstiness (Poisson vs on/off);
load-generator model (open vs closed); capacity of the bottleneck stage (CPU-capped reranker,
provider TPM). · **Exercises:** M22 Challenge B, the failure-under-load scenario ("at 4× query
volume, p99 latency goes to 6 seconds and the error rate rises"); the knee as ρ → 1; "the same
architecture meets 2.5 s at 40% utilisation and misses it at 75%" (M18). · **Course:** M18 (the
unallocated line as queueing; "verify latency at peak utilisation"), M22 ("if your answer contains
'I'd restart the service', the answer is wrong"), M19 (per-stage spans locate the bottleneck); F8.
No module teaches load-testing method, open vs closed loop, or coordinated omission.

**Inject:** K4's `ramping-arrival-rate` 1× → 2× → 4× → 8×, against a Locust closed-loop twin at the
same nominal rate. Bottleneck: rerank container at `cpus: 1` ([RRK-01](#rrk-01)) and the mock's TPM
at the production ratio ([LLM-02](#llm-02)).

**Miniature math:** busy hour ≈ 0.6 QPS, L = λW ≈ 1.5 in flight. Queueing delay grows like ρ ÷ (1 −
ρ) (M/M/1; Kingman's formula for general service times). M18 attributes this growth to Little's law,
which only says L = λW. Hold ρ at the bottleneck: if production's reranker is at ρ = 0.12 at 1×,
choose λ_mini and the CPU cap so ρ_mini = 0.12 at 1×. Then 4× is 0.48, and 4× with retries or fan-out
passes 1. **Coordinated omission:** a closed-loop generator stops sending while it waits, so its p99
understates latency by the backlog it never created.

**Predict first:** "The knee is at λ = ___ × busy hour; Locust reports p99 ___ ms while k6 reports
___ ms at the same nominal rate."

**Measure:** p50/p95/p99 and error rate vs λ (the hockey stick); k6 `dropped_iterations`; per-stage
queue time from spans. · **Pass if:** P95 ≤ 2.5 s at 2× busy-hour λ, and a documented λ_max.

**Fixes to try:**
- Find and scale the bottleneck stage (traces) → the knee moves right in proportion (→ that stage's
  cost).
- Admission control / load shedding (503 + `Retry-After` above λ_max) → admitted-request p99 flat,
  error rate explicit (→ rejected users).
- Degrade gracefully: skip rerank or verification above a ρ threshold (M18's conditional
  verification) → p99 flat, quality dips on a known stratum.
- Restart → the queue drains briefly and refills at the same ρ. Run it to show why it is wrong.

**Where the miniature lies:** the curve transfers only if the miniature's bottleneck is the same stage
as production's; with a mock provider, the provider bottleneck is whatever you configured. Loopback
networking hides connection-pool limits, so cap pool sizes explicitly.

<a id="srv-02"></a>
### SRV-02 · Network latency, bandwidth and payload size between services

**Knob:** RTT and jitter (app → vector DB, app → provider); bandwidth; payload (`with_payload`,
`with_vector`, top-k, chunk text size). · **Exercises:** a managed or serverless vector DB over the
internet (M5: Turbopuffer cold p90 1,214 ms, upserts p90 248 ms); payload bloat from returning
vectors you do not need; sequential round trips dominating the budget. · **Course:** M5
(object-storage-native engines), M18 (per-stage allocations), M9 (hosted reranker "network hop"); F8.

**Inject:** `toxics.sh cross-az` and `bandwidth rate=2000` on the Qdrant proxy, or on Linux `tc qdisc
add dev eth0 root netem delay 20ms 5ms distribution normal rate 20mbit` (needs `NET_ADMIN`). Query
with `"with_payload": true, "with_vector": true` vs ids only, at limit 40.

**Miniature math:** 40 candidates × (4.3 KB text + 4 KB vector) ≈ 332 KB per query (M18's sizes). At
20 Mbit/s ≈ 2.5 MB/s that is **133 ms** of transfer; ids only is ~1 KB. Sequential remote hops
(embed → search → rerank → generate → verify ≈ 5) cost ≈ 5 × RTT. Independent of N, so exact.

**Predict first:** "`with_vector=true` at limit 40 over 20 Mbit adds ___ ms."

**Measure:** stage latency vs RTT and bandwidth; bytes per query. · **Pass if:** network share ≤ 10%
of P95.

**Fixes to try:**
- Return ids and scores, fetch text for the top-n only → transfer ~12× smaller at top-6 (332 KB →
  ~27 KB) (→ a second fetch).
- Co-locate, or parallelise independent hops (dense ∥ BM25, M18 line 5) (→ placement constraints).
- Connection reuse / keep-alive / gRPC → the handshake is paid once (→ pool sizing).

**Where the miniature lies:** toxiproxy does not simulate packet loss or TCP retransmit tails; netem
does, on Linux.

<a id="srv-03"></a>
### SRV-03 · Timeouts, retries and retry storms (with circuit breaking)

**Knob:** client timeout; retry count r; backoff and jitter; retry budget; circuit-breaker thresholds
(LiteLLM `allowed_fails`, `cooldown_time`, `fallbacks`). · **Exercises:** retry storms (M14:
"unbounded concurrency produces 429s and retry storms"); Challenge B's rising error rate; a fallback
model answering in a different quality regime. · **Course:** M14, M18 (the 190 ms unallocated line is for
"queueing delay, retries, GC headroom"), M20 (C2 class: P95 > 2.5 s for > 30 min), M22 Challenge B; F8. No module covers
retry budgets or circuit breakers.

**Inject:** a K2 brown-out profile raising `P_5XX` from 0 to 50% for 5 virtual minutes. LiteLLM Router
with `num_retries=3` vs 0, with and without `fallbacks=[{"frontier": ["cheap"]}]`, `allowed_fails=3`,
`cooldown_time=30`.

**Miniature math:** offered-load amplification = (1 − p^(r+1)) ÷ (1 − p) for failure probability p and
r retries. At p = 0.5, r = 3: (1 − 0.0625) ÷ 0.5 = **1.875×**. As p → 1 it tends to r + 1 = **4×**.
Retries without a budget turn a brown-out into an outage. A rate property, so exact.

**Predict first:** "During a 50% brown-out with r = 3 and no budget the provider sees ___× normal
traffic; recovery after it heals takes ___."

**Measure:** offered vs admitted calls; amplification; recovery time after the brown-out; P95 during
and after. · **Pass if:** amplification ≤ 1.2× and recovery ≤ 1 virtual minute after the provider
heals.

**Fixes to try:**
- Exponential backoff with jitter, honouring `Retry-After` → retries de-synchronise and 429 bursts
  flatten (→ longer tail for retried requests).
- Retry budget (e.g. ≤ 10% extra calls) → amplification capped at ~1.1× (→ more user-visible errors
  during the brown-out).
- Circuit breaker / cooldown + fallback → fast-fail during the brown-out, quality dip measured on the
  fallback (→ a second model's quality bar).

**Where the miniature lies:** real 429 policies also penalise sustained overage; the mock refills
instantly.

---
<a id="layer-eval"></a>
### Layer EVAL · the evaluation harness itself

<a id="eval-01"></a>
### EVAL-01 · Eval-set size, A/A tests and the noise floor

**Knob:** golden-set size n (30 / 60 / 100 / 200 / 400 / 1,200); per-stratum n; paired vs unpaired
comparison; repeats. · **Exercises:** "at n=400 differences under ~4 points are unresolvable, so a
2-point gate fires on noise — the team learns to re-run until green" (M13); 48 comparisons × α = 0.05
= 2.4 false alarms per merge; worked-example deltas across the course that sit inside their own noise
band (M1 n = 60, M7 strata of 8–20, M10 strata of 5–24, M12's 40 answers). · **Course:** M13 (sizing,
multiple comparisons, gate types), M20 (C2/C3 thresholds), M22-2 (per-stratum gate), M15 (DG stratum
0.79 vs 0.85 vs 0.86); F1–F6.

**Inject:** an A/A run, the *same* config twice with a reseeded ANN rebuild ([VEC-06](#vec-06)) and
T = 0.2 generation or judge, then bootstrap-resample:

```python
import numpy as np
def paired_boot(a, b, B=10000, rng=np.random.default_rng(0)):
    d = np.asarray(a) - np.asarray(b); idx = rng.integers(0, len(d), (B, len(d)))
    return np.percentile(d[idx].mean(1), [2.5, 97.5])
```

**Miniature math:** n is the knob and corpus N is irrelevant. At n = 400, p = 0.85, the 95%
half-width is 1.96 × √(0.1275 ÷ 400) = **0.035**. The unpaired minimum detectable effect at 80% power
is ≈ 2.8 × √(2 × 0.1275 ÷ 400) = **0.071**. A 50-query stratum has a half-width of 0.10. The rule of
three says that zero failures in n trials bounds the rate at ≤ 3/n (95%), so "abstention must be
100%" checked on 40 queries only bounds failures at ≤ 7.5%.

**Predict first:** "A/A on 60 queries swings recall by ±___; a 2-point gate fires on ___% of A/A
runs."

**Measure:** empirical A/A false-alarm rate per gate; CI width vs n. · **Pass if:** ≤ 1 false alarm
per 20 merges at the chosen gate.

**Fixes to try:**
- Paired bootstrap on the same queries → CI narrower than unpaired by about √(1 − correlation) (→
  both configs run on every query).
- Gate on the lower CI bound, not the point estimate (→ slower to accept real wins).
- Fewer gated metrics (→ less coverage).
- Grow the strata that gate (→ annotation hours).

**Where the miniature lies:** synthetic queries have lower variance than real ones (M13's "variance
collapse"), so A/A on a synthetic set understates the real noise.

<a id="eval-02"></a>
### EVAL-02 · Injected label noise and the circularity trap

**Knob:** gold-label flip rate ε (0 / 10 / 20 / 35 / 50%); ambiguity (gold is one of many valid docs);
synthetic-query share and how it is conditioned (on the target chunk vs on held-out docs). ·
**Exercises:** configuration rank inversions (BM25 vs dense vs hybrid) caused by label noise rather
than by the systems; synthetic-set inflation ("an A/B gate calibrated on synthetic queries
under-fires", M13). · **Course:** M13 (the circularity trap: queries generated from the text you
retrieve inflate scores; ledger entry M13-3, hand-curated gold), M1-1, M15 (a synthetic positive is
"a leak by construction"); F1, F2. Also [labs/README.md](labs/README.md), which makes the synthetic
set the regression gate that M13-3 forbids.

**Inject:** flip gold doc ids to random docs at rate ε, deterministically by seed, and also
preferentially inside the hardest decile. Separately generate a circular synthetic set with
`generate_question_context_pairs(nodes, llm=local_llm, num_questions_per_chunk=2)` and compare it
with a held-out-doc set. GEN-05 measures v1's *natural* label error; this point injects controlled
error.

**Miniature math:** observed delta ≈ (1 − ε) × true delta, so the required n grows by ≈ 1 ÷ (1 − ε)².
At ε = 0.35 that is 2.4×: 400 queries behave like ~170, and the 0.035 half-width at n = 400 becomes
an effective resolution of about 5.4 points (0.035 ÷ 0.65).

**Predict first:** "At ε = 0.35, hybrid's 0.15 advantage over dense shrinks to ___; the chance the two
configs swap rank on a 60-query subset is ___."

**Measure:** recall attenuation vs ε; rank-inversion probability between config pairs vs ε; the
synthetic-vs-held-out gap. · **Pass if:** the rank order of M7's configs is stable at the v2 set's
audited noise rate.

**Fixes to try:**
- v2 fact-table gold ([GEN-05](#gen-05)) → natural error ≈ 0 (→ schema work).
- An adjudication pass on a sample with inter-rater agreement (→ human hours).
- Held-out-doc synthetic generation → inflation shrinks (→ no guarantee a supporting chunk exists,
  as M13-3's reversal condition notes).

**Where the miniature lies:** random flips are kinder than real label errors, which correlate with
hard queries.

<a id="eval-03"></a>
### EVAL-03 · Mock judge: variance, bias, rate limits and harness wall-clock

**Knob:** judge flip probability; position, verbosity and self-preference bias; judge RPM/TPM;
`BatchEvalRunner(workers=k)`; judge cache on or off; golden-set size × runs. · **Exercises:** gate
decisions flipping on judge noise; the judge competing with live traffic for TPM; "judge calls
dominate cost; predict them before spending" (M13). · **Course:** M13 (four judge biases; "calibrate
the judge, then gate on the judge"; flake budget; T = 0 + cache for byte-identical re-runs; "a gate
too slow to run is not a gate"), M13-2 (never judge-gate the numeric stratum), M19 (online
RAGChecker too expensive); F6.

**Inject:** a judge route on K2 that returns structured verdicts with P(pass) = base + bias terms +
noise (`profiles/judge.env`: `TPM=200000 RPM=500`). Point `FaithfulnessEvaluator(llm=mock_judge)`
and `BatchEvalRunner(evaluators, workers=8)` at it.

**Miniature math:** harness wall-clock ≈ max(calls ÷ RPM, tokens ÷ TPM). Nightly run: 400 queries × 3
judge axes = 1,200 calls × ~4k tokens = 4.8M tokens; at 200k TPM that is **24 min**, or 48 min with
a position-swap pass. The calls cap gives 1,200 ÷ 500 = 2.4 min, so TPM binds. M18 prices the whole
eval line at $121.28/month: small in dollars, large in wall-clock and quota. Independent of N.

**Predict first:** "The full nightly eval at 200k TPM takes ___ min; with 5% judge noise a gate at
Δ = 0.02 flips on ___% of A/A runs."

**Measure:** gate-flip rate vs judge noise; harness wall-clock vs workers (plateau at the binding
limit); 429s. · **Pass if:** the PR gate runs in ≤ 5 min and flips on ≤ 5% of A/A runs.

**Fixes to try:**
- Cache judge outputs → byte-identical re-runs, cost → 0 on unchanged pairs (→ cache invalidation
  when the judge prompt changes).
- A 60-query PR subset and the 400-query set nightly → PR gate ≤ 5 min (→ weaker PR gate).
- Calibrate the judge against a hand-labelled sample → agreement number reported, fewer gate flips
  (→ labelling hours).

**Where the miniature lies:** real judge biases have unknown sizes. Reproduce the *shape* (M13 cites
TREC 2024: 56% perfect human/judge match on a 3-level scale) and calibrate on your own labels.

---

<a id="layer-obs"></a>
### Layer OBS · observability

<a id="obs-01"></a>
### OBS-01 · Trace pipeline under backpressure: queue limits, drops and sampling

**Knob:** exporter queue size, batch size and schedule; collector CPU/RAM; network to the collector;
head sampling rate; tail sampling policy; span size (node ids, scores, node text). · **Exercises:**
observability that fails under exactly the load you need it for; lost `index_version` joins; head
sampling at 1% making a 4% stratum nearly invisible. · **Course:** M19 (span schema,
`index_version`/`cache_outcome`, "audit which spans your integration actually emits", M19-1
retention split), M2-2 (retain the full per-query trace), M17 (the trace store is leak vector 10); F1,
F2, F7. M19 does not cover trace sampling, backpressure or trace volume.

**Inject:** Python OTel SDK env vars (standard in the spec): `OTEL_BSP_MAX_QUEUE_SIZE=256`,
`OTEL_BSP_SCHEDULE_DELAY=5000`, `OTEL_BSP_MAX_EXPORT_BATCH_SIZE=64`. Collector container at `cpus:
0.25`, `mem_limit: 128m`, with `memory_limiter` and `batch` processors and an exporter
`sending_queue.queue_size` (⚠️ verify the names against your collector version). Toxiproxy latency
and bandwidth on the OTLP port; Phoenix or Langfuse as the backend.

**Miniature math:** hold **trace bytes/s ÷ collector capacity**. M18 budgets ~30 KB per trace ×
255k/month ≈ 7.65 GB-month (M18 prints 7.3). A RETRIEVE span with 40 node ids and scores (plus node text, the sensitive
field) varies ~10× in size, so drops concentrate in the largest spans, which are exactly the
RETRIEVE spans that diagnose F1/F2. Sampling: 8,500 queries/day × 4% stratum × 1% head sampling ≈
**3.4 traces/day**.

**Predict first:** "At 1% head sampling the DG stratum yields ___ traces/day; the collector at 0.25
CPU starts dropping at ___ spans/s."

**Measure:** span drop rate vs QPS; drop rate by span type; join success (`query_id`,
`index_version`); detection latency for the [OBS-02](#obs-02) incident under each sampling policy. ·
**Pass if:** 0 audit-record loss, and diagnostic traces exist for ≥ 95% of failing queries.

**Fixes to try:**
- Tail sampling on low-margin, thumbs-down and error traces → diagnostic traces kept at ~5% of volume
  (→ a stateful collector tier).
- Split small structured audit events from large debug spans → audit completeness 100% while debug
  traces drop (→ two pipelines).
- Hash node text at the span boundary → span size and leak surface shrink (→ harder debugging).

**Where the miniature lies:** production runs horizontally scaled collectors. The ratio and the
*shape* of failure transfer; the absolute drop point does not.

<a id="obs-02"></a>
### OBS-02 · Planted regressions: detection latency per control

**Knob:** failure type and severity; sampling rates; the set of controls. · **Exercises:** M19's
"time to detect … 45 hours"; M20's "twenty-nine hours between the ingest and the first signal";
health metrics flat or improving while quality collapses. · **Course:** M19 (controls table: ingest
assertion < 15 min, nightly stratum ~12 h, margin ~6 h), M20 (triage tree; failure budget classes
from C1, compliance, zero per quarter, "stop the line", to C4, individual answers, tracked), M22 scorecard (its diagnosability row cites this test); F3, F6,
F7, F10. The syllabus's M20 sketch specifies a lab that will "inject four planned failures"; it was
never written.

**Inject:** in the K5 replay at `SPEED=100`, plant (1) a parser-version swap on 5% of agreements
([PARSE-01](#parse-01)); (2) a disabled delete path / no docstore → two live revisions
([GEN-03](#gen-03), [ING-03](#ing-03)); (3) a provider alias swap ([LLM-03](#llm-03)); (4) an ACL
source change without sync ([FILT-03](#filt-03)); plus M19's 20 garbled DG pages
([PARSE-02](#parse-02)). Controls: ingest assertions, nightly per-stratum recall, margin alert, query
drift, 1% faithfulness sampling, canary divergence, leak test.

**Miniature math:** detection latency = time until a control's sample includes enough affected
queries. Affected samples/day = λ × s × q (stratum share s, sampling rate q): at 8,500 × 4% × 1% =
3.4/day, a statistically meaningful signal takes days. Hold λ·s·q in virtual time.

**Predict first:** "Faithfulness sampling detects the garbled-PDF incident after ___ (never?); the
margin alert leads recall by ___ h."

**Measure:** detection latency (virtual hours) per failure × control; false alarms per virtual week.
· **Pass if:** every planted failure is detected within 24 virtual hours by at least one paging
control, and the triage tree localises each to the right stage.

**Fixes to try:** the controls are the fixes. The signature is the per-control detection-latency
table, which turns M19's asserted table into a measured one.

**Where the miniature lies:** planted failures are clean; real incidents are mixed and partial. Add
one unplanned compound failure (the alias swap during a parser change).

<a id="obs-03"></a>
### OBS-03 · Query drift injection: new jurisdictions, vocabulary and strata

**Knob:** drifted share of the query mix (1 / 5 / 20%); drift type (the 20 jurisdictions v1 lacks; new
UN codes; a new client; colloquial rewording); detection window. · **Exercises:** query drift (M19,
M20); "new client or jurisdiction → *Create the stratum*" (M19); a golden set measuring "the old
system as the corpus quadruples" (M19-4). · **Course:** M19 (detector table: centroid shift in
hours–days), M20 (drift taxonomy; re-stratify the golden set); F1, F4.

**Inject:** the load generator shifts the mix over virtual weeks. Detectors: query-embedding centroid
shift; KS/PSI on query length; new-token rate; MMD on PCA components.

**Miniature math:** samples needed to detect a shift of size δ at power 0.8 scale with ~1/δ². Days to
detect = samples ÷ (λ × sampling rate). Independent of N.

**Predict first:** "A 5% mix of queries about unseen jurisdictions is detected after ___ queries."

**Measure:** detection latency vs mix fraction; false alarms on stable weeks. · **Pass if:** 5% drift
is detected within 7 virtual days.

**Fixes to try:** drift detector + golden-set loop → the new stratum is created within one review
cycle and gets its own recall number (→ annotation for every new stratum).

**Where the miniature lies:** real drift is gradual and seasonal, and step changes are easier to
detect.

---

<a id="layer-sec"></a>
### Layer SEC · security

<a id="sec-01"></a>
### SEC-01 · Poisoning dose-response: k poisoned docs vs corpus size N, and placement

**Knob:** poisoned docs k ∈ {1, 2, 5, 10, 20}; corpus size N ∈ {3k, 30k, 300k chunks}; placement
(body, footnote, hidden HTML, table cell, metadata `subject`, filename, alt text, a previously deleted
doc); top_k; reranker on/off; defence rung. · **Exercises:** "a better-ranked injection is a more
reliable injection"; "faithfulness to a poisoned span is *compliance* with it" (the poisoned answer
scores 1.0 faithfulness and 1.0 citation precision). · **Course:** M21 (F9; defence ladder; 7-step
red team; "a defence that reports only the first is reporting its best case"), M17 (leak vector 8),
M20 (canary documents); F9, F10.

**Inject:** v2 `--poison k --target-query "<q>" --placement table_cell`. v1 has no HTML or XLSX output
and its wiki metadata is only `{"topic": ...}`, so M21's placements cannot be built today. Give the
payload a benign, measurable goal (emit a canary token, or cite the poisoned doc), never a real
exfiltration endpoint.

**Miniature math:** attack success rate (ASR) is governed by P(the poison wins a top-k slot for its
target query). For a query-targeted poison that depends on absolute k and similarity and is **nearly
independent of N** (M21's [POISON23] source: "five texts in millions"). This is the one security
experiment where the miniature is no more optimistic than production. Test it: plot ASR vs k at
three N; flat lines confirm.

**Predict first:** "ASR at k = 5 is ___ at N = 3k and ___ at 300k; the sanitizer removes ___% of
poisons and ___% of legitimate rate rows."

**Measure:** ASR per outcome with ≥ 60 trials per cell (0/60 bounds ASR ≤ 5% at 95%; M21's 0/10
bounds it only at ≤ 30%); sanitizer false-positive rate and the numeric-stratum recall delta; an
adaptive-attacker curve (ASR vs attacker rewrite iterations). · **Pass if:** cross-tenant outcome ASR
= 0/60 under the adaptive attacker, and wrong-answer ASR upper bound < 5%.

**Fixes to try:**

| Rung | Expected signature | Cost |
|---|---|---|
| provenance marking (spotlighting) | ASR ↓ on body placements, uneven on metadata | tokens |
| ingest sanitizer | ASR ↓, **and legitimate docs dropped** (M21: 2 of 11 drops were real rate rows) → F1 on the numeric stratum | recall |
| provenance allow-list | ASR → 0 for non-allow-listed sources | recall |
| capability isolation + egress allow-list | outcome impact → 0 even when the injection "works" | UX friction |

**Where the miniature lies:** templated docs make poisons stand out. Realistic poison text needs
GEN-02's realistic prose.

<a id="sec-02"></a>
### SEC-02 · Egress and capability surface: exfiltration sink and tool calls

**Knob:** renderer behaviour (fetches images or not); URL allow-list; tool set (read-only vs a fake
`send_email`); approval gate. · **Exercises:** "citations are an egress channel"; "the severity of F9
is bounded by what the generator can reach, not by how good the injection is". · **Course:** M21-2
(sandboxed renderer), M21 ladder rows 7–9, MCP security guidance ("MUST NOT accept any tokens that
were not explicitly issued"); F9. Supply-chain note: `SentenceTransformerRerank` passes
`trust_remote_code=True` to `CrossEncoder` by default (src).

**Inject:** a local canary server counting requests and query strings (`python -m http.server 9999
2>&1 | tee egress.log`); a minimal markdown renderer that fetches images; a fake tool that logs its
arguments.

**Miniature math:** rates per outcome, independent of N. These are design properties, not scale
effects.

**Predict first:** "Without the allow-list, ___ of 60 poisoned answers trigger an egress fetch."

**Measure:** egress requests per 100 poisoned queries; tool calls carrying attacker strings. ·
**Pass if:** 0 and 0.

**Fixes to try:** egress allow-list → the canary server receives 0 requests whatever the ASR (→
broken legitimate links); read-only tools → action ASR → 0 (→ features you cannot ship).

**Where the miniature lies:** nowhere for the mechanism. Real renderers (Slack unfurls, email
clients) have more channels; enumerate them.

---

<a id="layer-cost"></a>
### Layer COST · cost accounting

<a id="cost-01"></a>
### COST-01 · Token and cost meter with per-query budgets; waterfall reconciliation

**Knob:** price table (dated assumptions); per-query budget cap (7.06¢, or 3.53¢ for final exam B1's
$9k); cache hit rate; cascade share; routed share. · **Exercises:** "the invoice that arrived before
the incident" ($41,600 vs $18,000); "six additions … six multipliers" (M18); two lines are 95.1% of
committed spend; the embedding cache that "can never pay for itself on cost". · **Course:** M18
(waterfall method; M18-1 reserve; cut list and refused-cut list; cascade), M16 (amortisation), M1-3
(cost and latency SLOs pinned from day one); F8.

**Inject:** K2 meters tokens per `model` and per `X-Consumer`. In-process, use `TokenCountingHandler`
(counts are cumulative until `reset_counts()`), and `MockLLM(max_tokens=256)` /
`MockEmbedding(embed_dim=1024)` to *predict*. Replay one virtual day of 8,500 queries with the strata
mix at `SPEED=100`.

**Miniature math:** monthly $ = Σ over lines of volume × price (M18's method). Hold the **per-query
token vectors**, which are N-independent apart from top_k × chunk size, and the traffic mix; then one
miniature day × 30 is the month. Check: M18's committed $15,596.95 ÷ 255,000 = $0.0612/query and
$15,596.95 ÷ 30 ≈ $520/day. **Percentiles do not add:** M18's waterfall and final exam C4 both sum
per-stage P95s, and the sum of stage P95s is not the end-to-end P95. Compute the composed P95 from
whole traces.

**Predict first:** "The day's metered spend is ___ (vs M18's model at $520/day); sum of stage P95s =
___ vs true end-to-end P95 = ___."

**Measure:** predicted (`MockLLM`) vs metered tokens; ¢/query by stratum; the latency waterfall from
traces, with **true end-to-end P95 next to the sum of stage P95s**. · **Pass if:** metered ¢/query ≤
budget at P95 ≤ 2.5 s with quality inside the noise band.

**Fixes to try:** apply M18's cuts one at a time and measure each saving's *quality* cost on the
golden set. Expected signatures: routing removes frontier tokens from 85% of traffic; caching takes
26% of queries off the path; margin-gated rerank cuts rerank calls 60%. The refused cuts (backstop,
eval) should show ≈ $0 and > 0 ms, which is why a dollars-only table would cut them.

**Where the miniature lies:** prices are assumptions (M18 flags every one). Provider-side
prefix-cache discounts and minimum cacheable lengths differ, and the mock cannot know them.

---

<a id="layer-ops"></a>
### Layer OPS · recovery

<a id="ops-01"></a>
### OPS-01 · Snapshot, restore and full-rebuild RTO under IO caps

**Knob:** disk read/write bandwidth; snapshot size; restore path (snapshot vs re-embed vs re-upsert
from stored vectors). · **Exercises:** "roll back to last known-good index" (M20's C1 response);
M14's "one-way rollback"; M15's nine-hour rollback. · **Course:** M14 (alias swap; retention window
for the old index), M20 (C1 stop-the-line), M15-5; F7, F8.

**Inject:** Qdrant snapshot API (`POST /collections/chunks/snapshots`; ⚠️ verify the restore
endpoint for your version); `pg_dump`/`pg_restore` for pgvector; under `--device-write-bps` / `io.max`
caps on Linux.

**Miniature math:** RTO ≈ snapshot bytes ÷ bandwidth + index rebuild time (if the snapshot excludes
the graph). Production: 23 GB vectors + payload + graph ≈ 35–45 GB; at 200 MB/s that is
175–225 s ≈ 3–4 min of IO, plus [VEC-04](#vec-04)'s rebuild. Re-embedding from source instead is
60 h at 1M TPM ([EMB-01](#emb-01)), so keeping vectors, not just text, is what makes recovery
possible at all. Hold bytes ÷ bandwidth.

**Predict first:** "Restore from snapshot at 50 MB/s takes ___ min; re-embedding takes ___ h."

**Measure:** RTO per path. · **Pass if:** rollback RTO ≤ the C1 response target.

**Fixes to try:**
- Keep the previous index live behind an alias → rollback is a pointer flip, RTO ≈ seconds (→ 2×
  storage).
- Store vectors alongside the source → a rebuild skips embedding (→ storage).

**Where the miniature lies:** cloud volume bandwidth and IOPS vary with burst credits, and laptop NVMe
is far faster. Throttle it.

---
<a id="the-lab-ladder"></a>
## The lab ladder

Climb one rung at a time. Each rung adds one piece of the kit and reuses everything below it. Rule:
do not start a rung until the previous rung's prediction log contains at least one **wrong**
prediction that you have explained. The pace column assumes the course's "working engineer" track
(about 3 modules a week) and is only a suggestion.

| Rung | Pace | Kit added | Experiments | Modules | Why this order |
|---|---|---|---|---|---|
| **0 · Python only, day 1** | week 1 | K7 prediction log, bootstrap helper | [GEN-01](#gen-01) on v1, [GEN-05](#gen-05) label audit of v1, [EVAL-01](#eval-01) A/A on v1 labels, [VEC-03](#vec-03) FAISS flat vs HNSW at 30k | M1, M2, M13 (preview) | No services. It teaches that the harness and the labels fail first, which is why v2 comes before everything else |
| **0.5 · Generator v2** | week 1–2 | fact table, per-doc seeds, sharded streaming, spans, multi-gold, `events.jsonl`, format emitters, validators | GEN-01 … GEN-05 implemented; rerun GEN-01 and GEN-05 (natural error should be ≈ 0); [GEN-02](#gen-02) calibration | M3, M13 | Every later rung depends on v2's realism and labels |
| **1 · One capped vector store** | weeks 2–3 | K1 profiles, K6 collector | [PARSE-03](#parse-03), [VEC-01](#vec-01), [VEC-02](#vec-02) (Linux), [VEC-03](#vec-03), [VEC-04](#vec-04), [VEC-06](#vec-06), [PARSE-01](#parse-01) throughput, at 30k → 100k → 300k chunks | M3, M4, M5 | Memory and build curves: the first knees |
| **2 · Filters, tenants, hybrid** | weeks 3–4 | payload and tenant indexes, pgvector as a second engine, BM25 | [FILT-01](#filt-01), [FILT-02](#filt-02), [FILT-04](#filt-04), [HYB-01](#hyb-01), [HYB-02](#hyb-02), [GEN-03](#gen-03) | M6, M7, M8 | Selectivity and tenancy fractions, nearly all N-independent |
| **3 · Mock provider online** | weeks 4–5 | K2 mock provider, toxiproxy (K3), LiteLLM router | [EMB-01](#emb-01), [EMB-02](#emb-02), [EMB-03](#emb-03), [RRK-01](#rrk-01), [RRK-02](#rrk-02), [QRY-01](#qry-01), [QRY-02](#qry-02), [LLM-01](#llm-01), [LLM-02](#llm-02), [LLM-03](#llm-03) | M4, M9, M10, M11 | Rate limits and latency distributions, which is where the query side actually queues |
| **4 · The instrument and advanced paths** | weeks 5–6 | mock judge route, `TokenCountingHandler` | [CIT-01](#cit-01), [EVAL-01](#eval-01) (full), [EVAL-02](#eval-02), [EVAL-03](#eval-03), [LLM-04](#llm-04), [LLM-05](#llm-05) | M12, M13, M15, M16 | The eval harness's own cost and noise, measured before it judges anything else |
| **5 · Time** | weeks 6–7 | K5 clock + replayer + prober, K3 chaos scripts | [GEN-04](#gen-04), [ING-01](#ing-01) … [ING-06](#ing-06), [FILT-03](#filt-03), [CACHE-02](#cache-02), [PARSE-01](#parse-01) backfill, [PARSE-02](#parse-02), [OPS-01](#ops-01) | M14, M17, M20 | Freshness, idempotency and deletion: where the 480k story binds |
| **6 · A production week in an hour** | weeks 7–8 | K4 open-loop load, OTel collector, Redis cache, cost meter | [VEC-05](#vec-05), [SRV-01](#srv-01), [SRV-02](#srv-02), [SRV-03](#srv-03), [CACHE-01](#cache-01), [COST-01](#cost-01), [OBS-01](#obs-01), [OBS-02](#obs-02), [OBS-03](#obs-03) | M18, M19, M20, M22 | Composition: the waterfall, the knee, detection latency |
| **7 · Adversarial** | week 8 | poison generator, egress sink, fake tools | [SEC-01](#sec-01), [SEC-02](#sec-02), FILT-04 rerun | M21, M17 | Needs v2 placements; could move up to just after rung 2 |
| **8 · Full 480k replay** | capstone | everything | the three modes below | M22, [capstone](capstone/README.md) | Fills M22's capacity table with measured numbers |

**Fixture dependencies.** Every rung needs K6 and K7. GEN-02 realism is a precondition for FILT-01/02,
HYB-01/02, VEC-01/06 and SEC-01. GEN-04's timeline is a precondition for every rung-5 experiment and
for OBS-02/03. K2 is a precondition for anything that calls a provider. The one ordering trap: do not
run VEC-05 before VEC-04, because you need the build cost to size the burst.

**Rung 8 needs three honest modes, because one laptop cannot do all of it at once.**

1. **Scale-only mode** (memory, build, ingest machinery at true count). v2 streams all 480k docs
   (≈ 2.2B words, ~18 GB on disk at v1's 8.5 bytes/word; check [ING-06](#ing-06) first). Vectors come from the mock's hash
   embeddings or from replicated, jittered real vectors. It says nothing about retrieval quality.
2. **Quality mode** (a sampled slice). A 5% population sample (~24k docs, ~280k chunks) with real
   embeddings, the timeline thinned by φ = 0.05 and provider limits scaled by φ. It preserves ρ and
   the quality-relevant fractions (selectivity, tenant shares, duplicate density).
3. **Time mode.** 12 virtual months at `SPEED=8760` (one real hour) for the event and query
   timelines, with the mock provider in virtual time. Throughput check first: 480k docs a year at
   8,760× is 133 docs/s, or ~800k tokens/s of embedding. No local embedder keeps up (⚠️ measure
   yours in [EMB-02](#emb-02); likely days of CPU), so run with
   φ ≤ (your embed tokens/s) ÷ 800k.

**Deliverable:** M22's capacity table, with each cell replaced by a measured value ± interval and the
mode it came from.

---

<a id="interview"></a>
## From experiment to interview answer

The pattern is **numbers → bottleneck → fix → what breaks next**. Interviewers ask scale questions to
see whether you can find the binding constraint and reason about the second-order effect of
removing it. A measured experiment gives you all four parts, plus the arithmetic that shows you did
not memorise them. Fill the placeholders from your prediction log. Where your measurement disagrees
with the course, say so: that is the strongest version of the answer.

1. **Memory tier ([VEC-01](#vec-01)).** "I reproduced our month-12 index (5.6M × 1,024-d, 23 GB) as
   300k vectors in a `<your container size>` container, holding RAM ÷ index at `<your P_mem>`. p99
   went from `<p99 before>` to `<p99 at the knee>` ms, with page-cache refaults at `<refaults/s>`.
   int8 in RAM with originals on mmap and 2× oversampling held recall vs exact at `<recall>` and p99
   at `<p99>` in `<RAM %>` of the RAM. Next to break is rescore IO: below `<IOPS>` IOPS rescoring
   dominates, so I'd watch that before buying RAM."
2. **Re-embed as a throughput project ([EMB-01](#emb-01)).** "A full re-embed is 3.61B tokens. At
   $0.01–0.02 per 1M that is $36–72, so cost is irrelevant. At `<TPM>` TPM it is `<hours>` hours of
   wall-clock, and throughput plateaued at `<tokens/min>` whatever the worker count. Batching to
   `<inputs>` per request took RPM out of the picture, and a separate backfill quota held live
   freshness p95 at `<minutes>`. What breaks next is rollback: a `<hours>`-hour re-embed means
   rollback must be an alias flip, so we pay 2× storage during migrations."
3. **Filter selectivity ([FILT-01](#filt-01)).** "With client ∧ jurisdiction admitting `<p_c>`% of
   vectors, post-filtered HNSW kept `<survivors>` of the true top-10, which matches k × p_n.
   Filter-aware traversal restored `<recall>` at `<p99>` ms; pgvector iterative scans recovered
   until `max_scan_tuples` bound at `<N>`. I also had to scale the planner's brute-force threshold
   by 1/18.7, or the miniature looked better than production. Next is small-tenant recall, so I gate
   on the worst tenant, not the index average."
4. **Reranker CPU ([RRK-01](#rrk-01)).** "Scoring 40 candidates with MiniLM-L6 took `<ms>` on 2 CPU
   cores, against the course's 74 ms. At `<multiple>`× busy-hour load, utilisation reached `<ρ>` and
   p95 broke 2.5 s. Picking N at the recall@N knee (`<N>`) and reranking only low-margin queries cut
   rerank CPU `<percent>` with nDCG within `<delta>`. Next is over-fetch from selective filters,
   which pushes N back up, so filter strategy and N are one decision."
5. **Fan-out and mixture P95 ([QRY-01](#qry-01), [QRY-02](#qry-02)).** "With a 2% heavy tail per
   LLM call, three parallel sub-questions made `<percent>`% of requests slow, matching 1 − 0.98³ ≈
   5.9%, enough to break P95 alone. Routing decomposition to `<share>`% of traffic (it has to be
   under 5%) and moving the rest to an async path restored sync P95 to `<seconds>` s. Next is router
   error: at `<misroute %>` misroutes the cheap path answers global questions confidently and wrong."
6. **Freshness under replay ([ING-01](#ing-01)).** "Replaying 12 virtual months at `<speed>`×
   showed that a 15-minute poll can't meet a 15-minute p95: we measured `<p95>` min against the
   0.95 × P + t_proc prediction of `<predicted>`. Push plus a per-document verification read gave
   `<new>` min for new content and `<retired>` min for retired content, and a 2,000-doc burst
   drained in `<minutes>`. Next is source-system detection lag, outside our pipeline, now
   `<percent>`% of the budget."
7. **Idempotency under crashes ([ING-02](#ing-02)).** "Killing workers `<kills>` times per 1,000
   docs created `<duplicates>` duplicate nodes, because node ids default to random UUIDs.
   Deterministic ids from (doc_id, offsets, parser version) plus a verification read took
   duplicates to 0 for `<percent>`% more ingest wall-clock. Next is ambiguous acks on network
   resets, which I test with toxiproxy `reset_peer`."
8. **Semantic cache economics ([CACHE-01](#cache-01)).** "On a Zipf(`<α>`) query mix the cache hit
   `<percent>`% at `<MB>` MB. On planted near-miss pairs the false-hit rate went from `<f low>` to
   `<f high>` across the threshold sweep. With c_wrong = $`<cost>`, value peaked at τ = `<τ>` and
   went negative past `<τ>`. Keying on tenant + ACL hash cost `<points>` points of hit rate and
   removed all cross-tenant hits. Next is invalidation: without doc-keyed purges a superseded clause
   stayed cached for the full TTL of `<minutes>` min."
9. **The instrument ([EVAL-01](#eval-01), [EVAL-02](#eval-02), [GEN-05](#gen-05)).** "A/A runs on our
   60-query set swung recall by ±`<swing>`, so module-level wins of +0.04–0.09 were inside noise.
   Auditing the generated labels found `<percent>`% wrong. Label noise of `<ε>` meant ~`<factor>`×
   more queries to detect a 5-point change. I moved the gate to a paired-bootstrap lower bound on a
   fact-table-derived set. Next is per-stratum power: strata under `<n>` queries can't gate."
10. **Poisoning ([SEC-01](#sec-01)).** "Planting k = `<k>` poisoned docs gave ASR `<x>`/60 at 3k chunks
    and `<y>`/60 at 300k, so ASR barely moved with corpus size: it is governed by whether the poison
    wins a top-k slot. Provenance marking cut ASR to `<a>`; the sanitizer cut it to `<b>` but dropped
    `<percent>`% of legitimate rate rows, an F1 cost. Only egress allow-listing took exfiltration to
    0/60. Next is the adaptive attacker: after `<iterations>` rewrites ASR climbed back to `<c>`, so
    the number we report is the adaptive one."
11. **Load and Challenge B ([SRV-01](#srv-01), [SRV-03](#srv-03)).** "At our busy hour only ~1.5
    requests are in flight (0.6 QPS × 2.5 s), so the 4× knee wasn't our servers. It was shared
    provider TPM plus retries amplifying load by `<factor>`×. A retry budget and a separate eval
    quota held P95 at `<seconds>` s at 4×. A restart only drains the queue until it refills at the
    same ρ, so the fix was admission control at `<QPS>` QPS."

---

<a id="cannot-teach"></a>
## What this cannot teach

| Limit | Why the miniature misses it | What to do instead |
|---|---|---|
| **Real traffic shape** | Zipf exponents, diurnal and seasonal cycles, burst correlation and the query mix are all invented here, and the cache (CACHE-01), router (QRY-02) and drift (OBS-03) results depend on them directly | Replay a real query log as soon as one exists; until then report results as a function of α, f and the burst factor, not as single numbers |
| **Correlated provider incidents** | The mock draws latency and faults independently per request; real brown-outs hit every request from a client at once | Add a brown-out mode, and treat any parallel fan-out result as optimistic |
| **Frontier-model behaviour** | Small local models are far more position-sensitive, and less consistent, than frontier models. The mock's Gaussian noise is not the provider's nondeterminism | Use local models to *see* mechanisms (LLM-01, LLM-03); size them with a small paid canary on the real model |
| **Effects that scale with log N or absolute counts** | HNSW hop count, planner thresholds, segment counts and per-collection overhead do not follow the ratio | Each point lists these under "Where the miniature lies"; extrapolate them separately, with error bars |
| **Hardware topology** | Laptop cores, memory bandwidth and NVMe are faster than cloud vCPUs and volumes; burst credits, NUMA and cloud noisy neighbours are absent; laptops throttle thermally | Calibrate one point per curve on the target instance type |
| **Docker Desktop's VM** | Shared VM RAM, macOS page cache under "disk", unthrottled virtiofs | Use a Linux host or a VM with its own disk for IO experiments |
| **Months-long effects** | Index fragmentation, compaction backlogs, slow label drift and cost creep take real months; the virtual clock compresses arrivals but not physical compaction | Measure compaction in wall time; re-run the golden-set audit on a schedule |
| **Real document mess** | Scanned PDFs, OCR noise, font subsets and inconsistent identifiers ("UN 3480") are cleaner in generated docs | Calibrate once against a public messy corpus; inject variants (PARSE-02, HYB-02) |
| **Multi-region and replication** | One node per store; no cross-region consistency, failover or replica lag | Out of scope for a laptop; treat replica lag as one more staleness contributor in ING-01 |
| **Organisation and process** | Quota procurement, change management, on-call, legal sign-off on deletion, the conversation with finance in M18's opening | The ledger entries and the ADRs are where these live; experiments only supply the numbers |
| **Real attackers** | The adaptive-attacker curve uses your own rewrites; real attackers have more time and better placements | Report the adaptive ASR, and treat 0/60 as a bound (≤ 5%), not as zero |
| **GPU serving dynamics** | Prefill/decode interference and batching on provider GPUs are invisible through an API and only roughly visible with `llama-server -np` | Cite DistServe's mechanism (M18); do not derive latency from tokens/s |

---

<a id="appendix-a"></a>
## Appendix A — what the research found in the course itself

Each line was re-checked against the module text for this doc. Line numbers are approximate
pointers into each file. "Re-counted" means v1 was re-run at seed 1337 with `--docs 3000 --queries
400` (and 300 docs for the vocabulary check) and its output counted.

### A.1 Infrastructure (blocking)

1. **No lab code exists.** There is no `modules/*/lab/` and no `labs/eval_harness.py`, though
   [labs/README.md](labs/README.md) (~line 78) lists it. The syllabus (~line 19) says labs live in
   `labs/NN-slug/`; labs/README (~line 82) says `modules/NN-slug/lab/`. All 22 modules end in
   "Verification note: 🟡 pending".
2. **Missing dependencies.** [labs/requirements.txt](labs/requirements.txt) lacks
   `llama-index-embeddings-huggingface` (M4's local models), `llama-index-llms-openai-like` (the
   mock wiring), `faiss` and `pymilvus`.

### A.2 Generator vs claims (re-counted at seed 1337)

3. **Formats.** All 3,000 docs are `.md`, while the [syllabus](syllabus.md) (~lines 41–48) lists
   PDF/DOCX/XLSX/HTML and labs/README (~line 54) says rate tables are "written to XLSX".
4. **Labels.** There are no spans (only `gold_doc_ids`), against labs/README's "span-level gold
   answers" and "exact by construction" (~lines 60–65). Lexical queries: 56/80 name a nonexistent
   subclause, and 80/80 fail "subclause exists AND topic matches". Multi-hop: 24/80 gold docs have no
   entry for the queried UN code. All 80 unanswerable queries are one string.
5. **Coverage.** 20 jurisdictions vs the spine's 40 (syllabus ~line 53). 38.4% of docs are
   `client_id="ALL"`. Five query classes assigned round-robin, vs M1's nine strata
   ([M1](modules/01-why-rag-exists/README.md) ~lines 95–105) and M13's eight
   ([M13](modules/13-evaluation/README.md) ~line 321).
6. **Missing features the modules use.** No revisions (only 9 bulletin supersession links), no ACLs,
   no tenant skew, no `14.2(b)`-style identifiers (0 matches), no UN3090 and no "state of charge"
   (both used in [M10](modules/10-query-understanding/README.md) and
   [M15](modules/15-finetuning-retrieval-models/README.md)). Straddle templates print "3th" (1,936
   times). The DG "reduced to {m} percent" template always increases the surcharge (m ∈ 11–21 > n ∈
   3–9). The rate-table title carrier and header carrier are drawn independently.
7. **Prose.** The filler list has 28 words. The corpus has 253 distinct alphabetic words in 13.5M
   (the same 253 at 300 docs). Median doc length is 2,796 words, vs M3's "median … under 1,000 words"
   ([M3](modules/03-parsing-and-chunking/README.md) ~lines 226, 326).

### A.3 Scale numbers that disagree across modules

8. **Re-embed dollars off by 1000×.** *M14 part fixed 2026-11-13: now 3.61B tokens → $36 / $72 / $470, and the
   "five-to-six-figure" claim is replaced by a wall-clock argument; M22 and the capstone are still open.*
   [M14](modules/14-ingestion-at-scale/README.md) ~lines 292–294:
   3.74B tokens × $0.01/1M = **$37.40**, not $37,400; likewise $74.80 (not $74,800) and $486.20 (not
   $486,200). The "five-figure-to-six-figure project" (~line 298) follows from the error. ~Line 374's
   "five-figure monthly line item" for nightly rebuilds is $1.1k/month at $0.01 (3.74B × 30 × $0.01/1M
   = $1,122). [M22](modules/22-architecture-review/README.md) ~line 341 escalates it to "a seven-figure
   event", and [capstone](capstone/README.md) challenge 1 (~line 95), "your 12-month ingest will cost
   more than the annual budget in one re-embedding", is false at the course's own rates. The real
   constraint is wall-clock: ~60 h at 1M TPM.
9. **Endpoint vector count.** 5.04M chunks ([M18](modules/18-cost-and-latency/README.md) ~line 124),
   6.6M nodes (M22 ~line 223, matching `SentenceSplitter(512, 50)`), ~29.9M
   ([M5](modules/05-vector-databases-and-ann/README.md) ~line 58, hierarchical leaves only since 2026-11-13; was 34.8M). M22's
   reference architecture uses hierarchical chunking (~line 154) while its capacity table uses the
   flat-splitter count.
10. **Dimension.** 1,024 (M18 storage), 1,536 (M5), and M4's chosen `text-embedding-3-large` at 3,072
    with a dimension sweep ([M4](modules/04-embeddings-and-vector-space/README.md) ~line 402).
11. **Corpus words at month 12.** *Fixed 2026-11-13: one definition in
    [syllabus § Corpus scale](syllabus.md#corpus-scale) (2.17B words, 2.89B tokens); M3, M14 and the capstone link to it.*
    Was: 2.2B (M22 ~line 222), 2.24B (M14 ~line 280, from 14M rather than
    13.5M), 3.1B (capstone ~line 25; M3 ~line 271). 480k × 4,513 = 2.17B.
12. **Ingest rate.** M14 (~line 360): ~600 changed + ~4,000 new wiki pages a day, but 4,000/day alone
    is 1.46M a year, three times the 480k total. M18: 40,000 changed docs/month. M22 (~line 224):
    12k–22k docs/month. Growth from 3k to 480k implies ≈ 1,307 new docs/day.
13. **Freshness.** M22 ~line 377 passes "11 min p50, 14 min p99" with an ingest path that runs "every
    15 min" (~line 121). A 15-minute cycle alone puts p99 at ≈ 0.99 × 15 = 14.85 min before any
    processing. ~Line 230's "4 minutes of slack" is measured against the p50, not the p99 SLO.
    The same arithmetic applies to M14's worked-example table (~line 364), which gives the nightly
    rebuild a 12 h staleness P95: a 24 h cycle alone puts p95 at ≈ 0.95 × 24 = 22.8 h, before the
    9 h 20 m rebuild.

### A.4 Arithmetic and method errors worth teaching

14. **Decode line (new in this doc).** M18 ~line 84: "500 output tokens at 5.2 ms/token | 640 ms".
    500 × 5.2 = 2,600 ms. As printed, 640 ms is 1.28 ms/token. At 2,600 ms the generation lines alone
    (520 + 2,600 ms) exceed the 2.5 s budget. The "46% of the budget" text (~line 111) matches 640 ms.
    The research catalogue carried the 640 ms figure without noticing.
15. **Percentiles summed.** M18's waterfall allocates per-stage P95s that sum to the 2,500 ms SLO.
    [Final exam](assessments/final.md) C4 (~lines 256–272) sums stage P95s to "Total P95: 4,130 ms",
    and its rubric's "2.0 s of the 4.13 s" for rerank + query understanding is 1,180 + 520 = 1.70 s.
16. **Inverted cache threshold.** M18's τ table (~lines 251–256) has hit rate *rising* with τ (0.03
    at 0.80, 0.71 at 0.94) while the text (~line 258) calls 0.88 → 0.90 "loosening" and says "as τ
    falls you are admitting pairs". Either the table's direction is inverted or τ is a distance.
17. **Little's law.** M18 ~line 339: "Little's law governs it: wait time grows with utilisation".
    Little's law is L = λW; the growth of waiting time with utilisation is a queueing result (M/M/1,
    Kingman).
18. **Batching (analysis).** M18 ~line 337 says the 190 ms unallocated line "is where batching
    lives". At ~1.5 requests in flight, LogiTrack's own servers have nothing to batch, so this can
    only be the provider's batching.
19. **M10 fan-out.** ~Line 97 says latency is "the sum along the critical path", but the quoted
    trace (~lines 259–264) has children summing to 70.44 s inside a 66.49 s query, so they
    overlapped. ~Line 452 calls the expensive engines' share "the two smallest (16%)". The table's
    strata (~line 424) give multi-part + multi-hop = 21/50 = 42%; the two smallest strata are 13/50 =
    26%. For P95, "a minority of traffic" (~line 455) has to mean < 5%.
20. **M6 recall ceilings.** Post-filter recall vs filtered exact is bounded by ≈ p_n (survivors
    k × p_n out of k), but [M6](modules/06-metadata-filters-multitenancy/README.md) ~lines 325–328
    report 0.86 / 0.31 / 0.09 / 0.04 at p_n = 0.35 / 0.08 / 0.03 / 0.01, which is 2.5–4× the ceiling.
21. **Per-tenant storage multiplier.** M6 (~lines 248, 388, 491) and
    [M17](modules/17-multitenant-permissioned/README.md) (~line 111) say per-tenant indexes multiply
    storage by 12. With 38.4% shared `ALL` docs it is 0.616 + 12 × 0.384 ≈ 5.2×, unless docs are
    duplicated per tenant.
22. **M9.** ~Line 451: N 40 → 100 is "60% more candidate-set compute" in the same sentence as "2.5×
    the reranker bill"; 100/40 is +150%. The 74 ms P50 for MiniLM-L6 (~line 434) does not state its
    hardware.
23. **M7.** ~Line 353: "identifier and numeric queries cap out below 0.6 for every fusion
    configuration". The table (~lines 312–315) shows identifier at 0.94/0.91 under fusion; the
    sub-0.6 strata are numeric (0.41/0.40) and multi-hop (0.58/0.55).
24. **Statistics.** M5's worked example reports p99 from the 60-query set (~line 286), where p99 is
    essentially the maximum. M21's red-team cells are out of 10 trials
    ([M21](modules/21-security-prompt-injection/README.md) ~line 403), and 0/10 bounds ASR only at
    ≤ 30%. M13's "must be 100%" / "must be 0" at n = 40 / 20 (~lines 308–309) cannot be certified (rule
    of three: ≤ 7.5% / ≤ 15%). M13's "~4-point" resolution (~line 320) is a one-proportion CI
    half-width, while the unpaired difference needed at 80% power is ~8 points
    (2.8 × √(2 × 0.16 ÷ 400) ≈ 0.079 at M13's p = 0.8).
25. **M15.** ~Line 360: "−0.03 nDCG@10 did not breach LogiTrack's ≤2-point tolerance". 0.03 is 3
    points.
26. **M13 vs labs README.** labs/README (~line 62) makes the synthetic `questions.jsonl` "the
    regression gate". M13-3 (~line 288) says "a synthetic set cannot gate a release".
27. **M11 impossible values.** The Class A table (~lines 470–476) reports R = 5 repeats with
    "distinct payloads" of 6 and 7 and worst-case exact match of 0.83, 0.67 and 0.50. Per query, five runs
    give at most 5 distinct payloads and exact-match values in steps of 0.2.
28. **M1.** ~Line 227: a 1M-token window holds "5.3%" of an 18.0M-token corpus; 1 ÷ 18.0 = 5.6%.
29. **M19.** "Tuesday, 14:07 to Thursday, 09:20" (~line 21) is 43 h 13 min, not the "45 hours" of
    ~line 39. 1,180 chunks from 20 pages (~line 18) is 59 chunks per page.
30. **M16.** ~Line 321: a $26,199 build against $6.72/day of queries is 3,899 days ≈ 10.7 years, not
    "20 years". The table's "4.03B" index-build tokens (~line 306) matches neither its own extraction
    line (4,935M in + 726M out ≈ 5.66B, ~line 153) and looks like the 4.03M call count.
31. **M20.** ~Line 349: "twenty-nine hours between the ingest and the first signal". Mon 09:41 →
    Tue 14:08 is 28 h 27 min.
32. **M8.** "At `num_workers=4` wall clock is not the problem; the token bill is" (~line 358) holds only
    while no provider limit binds ([LLM-05](#llm-05)).

### A.5 Module text confirmed; framework side not re-checked

`llama-index` is not installed in this environment, so the framework half of these rests on the
research catalogue's reading of upstream source:

33. M7's decision ships `QueryFusionRetriever(mode="reciprocal_rerank")` (~line 355) and quotes the
    k = 2 row (0.78); the research found RRF k = 60 hardcoded in LlamaIndex (k = 60 gives 0.75 in M7's
    own sweep).
34. M9 ~line 270 shows `LLMRerank(top_n=2, service_context=service_context)`; the research found no
    `service_context` parameter in current source.
35. M9 ~line 436 gives `ColbertRerank` "+2.1 GB index"; the research found that postprocessor encodes
    candidates at query time, so the index cost describes ColBERT-as-retriever.

### A.6 Dropped or narrowed

- The research's "29-word filler list" is 28 words (corrected above).
- The research's leaf-only node count for hierarchical chunking (~168k at pilot, 26.9M at month 12)
  is not in any module and was not re-derived, so it is dropped from A.3.
- The research's "semantic gold is 1 of ~121 same-title docs" and "numeric queries match ~5.9 docs"
  were not re-counted; GEN-05 carries them as research estimates.
- The research's claim that the sentence-window parser yields ~72k nodes on v1 was not re-run (it
  needs `llama-index`); PARSE-03 marks it ⚠️.

### A.7 Topics the course never covers (this doc supplies them)

Load-testing method (open vs closed loop, coordinated omission, arrival processes); queueing,
utilisation and admission control; provider quota shared across query, ingest, eval and judge;
retry budgets and circuit breakers; trace sampling and backpressure; a revocation SLO; noisy
neighbours and per-tenant fairness; bulk-load vs incremental index builds; restore and rollback RTO;
thermal and CPU-quota artifacts in laptop benchmarks; Docker Desktop's VM caveats; small local models
as a way to make context caps physical; public precomputed embedding sets (dbpedia-1M) for testing
scale without paying to embed.

---

<a id="appendix-b"></a>
## Appendix B — verification status

**None of the snippets in this doc has been executed.** The kit is unbuilt. Every "Predict first"
blank is yours to fill, and every number not attributed to the course or to the re-counted v1
output is napkin arithmetic.

| What | Checked against | How |
|---|---|---|
| Course claims (module, syllabus, labs, capstone, assessments text) | the files in this repo | M14 and M18 read in full for this doc; every Appendix A line re-checked by targeted reads of the cited lines; the research catalogue read all modules |
| v1 generator behaviour (Appendix A.2, GEN layer) | `labs/generate_corpus.py` at seed 1337 | re-run outside the repo with a counting script (formats, vocabulary, ALL share, label defects, identifiers, "3th") |
| Docker, cgroup v2, compose, systemd, netem, toxiproxy, Pumba, libfaketime / freezegun / time-machine, stress-ng, k6, Locust, wrk2 | official docs (2026-11-12, via the research catalogue) | flags and keys in the kit tables |
| Qdrant, pgvector 0.8.x, FAISS, Redis, Ollama, llama-server, vLLM, OpenAI / Anthropic rate-limit headers, OpenAI embedding request caps | official docs or upstream source (via the research catalogue) | names, defaults and units used in the points |
| LlamaIndex classes marked (src): `SentenceTransformerRerank` defaults, `QueryFusionRetriever` RRF k, `LLMRerank`, `ColbertRerank` | upstream source, via the research catalogue | not re-checked here (`llama-index` not installed) |
| Mock provider, replayer, clock, collector, k6 script | written against the APIs above | **not run**. Two changes from the research version: a `TOK_BURST` / `MAX_REQ_TOKENS` 400 path, and a `DECODE_MS` default of 1.28 (Appendix A, item 14) |

**⚠️ ledger (verify before relying on it):**

- Docker Desktop for Mac block-device throttling (the `/dev/vda` path) and whether throttles bind at
  all.
- Whether compose `deploy.resources` limits apply outside Swarm.
- The toxiproxy image path, and the `reset_peer` toxic's attribute name.
- Qdrant: `optimizers_config.max_optimization_threads`; the current bulk-upload advice (disable or
  defer indexing); the snapshot restore endpoint.
- OTel collector config keys (`memory_limiter`, `sending_queue.queue_size`, tail sampling) for your
  collector version.
- The OpenAI Python SDK's default `encoding_format="base64"` for embeddings; the default-headers
  option name on LlamaIndex's OpenAI classes.
- `llama-index-llms-openai-like` / `OpenAILike` constructor parameters at your pinned version.
- Ollama `num_thread` (not in current docs).
- LiteLLM `mock_testing_rate_limit_error` (seen in one proxy example only).
- LlamaIndex `DocstoreStrategy.UPSERTS_AND_DELETE` failing to delete docs that share a content hash
  (PR #22712; merge status unknown).
- The Heaps'-law β target for realistic legal prose; calibrate on a real corpus.
- The HNSW graph-size approximation in VEC-04's `maintenance_work_mem` arithmetic.
- The sentence-transformers ONNX backend for faster CPU reranking.
- Illustrative numbers: 5,000 IOPS for a cheap cloud SSD; provider tiers of 3,000 RPM (embeddings)
  and 10,000 RPM / 2M TPM (generation); 20 pages/s/core parse rate; 2,000 tokens/s/core local
  embedding; 0.4 core-s rerank service time; 20 GB/s memory bandwidth; the busy-hour shape.
- macOS `RLIMIT_AS` enforcement (assume it does not work).
- Node counts not re-run: the sentence-window parser on v1 text; whether M3's hierarchical count
  includes non-leaf nodes.
