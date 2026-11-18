# Index decision diagram — options to compare

Draft for choosing what goes in M5's recap section, "Putting it together: choosing an index". Not linked from the README.

## Option A — current (decision tree)

One path per question; family, placement, compression and filtering are mixed at the same level.

```mermaid
flowchart TD
    A[Vectors to index] --> B{Raw vectors<br/>fit in RAM?}
    B -->|Yes| C{Filters selective<br/>and frequent?}
    B -->|No| D[Quantize and/or go to disk]
    C -->|No| E[HNSW graph: m / ef_construct / ef]
    C -->|Yes| F[Filter-aware graph: ACORN-style traversal]
    D --> G{Scarcest resource?}
    G -->|Memory| H[IVF + PQ codes: nlist / nprobe / m]
    G -->|Latency on SSD| I[Disk-resident graph: Vamana / DiskANN family]
    G -->|Bounded error| J[Graph or IVF + RaBitQ codes]
    E --> K[Measure recall@10 vs exact search vs p99]
    F --> K
    H --> K
    I --> K
    J --> K
    K --> L{Regression acceptable?}
    L -->|Yes| M[Ship; re-measure as the corpus grows]
    L -->|No| N["Raise ef or nprobe, add rescoring, or change family (orange-bordered boxes)"]
    classDef family stroke:#e8590c,stroke-width:3px
    class E,F,H,I family
```

## Option B — four independent choices (layers)

An index is one pick from each layer. Only layer 1 is the family.

```mermaid
flowchart TD
    S{"Exact search fits RAM<br/>and meets p99?"} -->|Yes| X["Exact search: no index<br/>(LogiTrack pilot)"]
    S -->|No| L1

    subgraph L1["1 · Family: how to skip most vectors"]
        A1["Graph (HNSW)<br/>dials: m, ef_construction, ef"]
        A2["Partitions (IVF)<br/>dials: nlist, nprobe"]
    end
    subgraph L2["2 · Placement: where it lives"]
        B1["RAM: index fits"]
        B2["SSD: it does not<br/>(e.g. DiskANN graph)"]
    end
    subgraph L3["3 · Compression: how each vector is stored"]
        C1["float32 (none)"]
        C2["int8 · 4×"]
        C3["binary / RaBitQ · 32×"]
        C4["PQ · up to 64×"]
    end
    subgraph L4["4 · Filter handling"]
        D1["Pre-filter"]
        D2["Filter-aware walk (ACORN)<br/>for mid-selectivity filters"]
        D3["Post-filter: fails at<br/>mid selectivity"]
    end

    L1 --> L2 --> L3 --> L4 --> K["Measure recall@10 vs exact,<br/>p99, memory"]
    K --> Q{Meets target?}
    Q -->|Yes| M[Ship; re-measure as the corpus grows]
    Q -->|No| N["Fix cheapest first:<br/>1. raise ef / nprobe (no rebuild)<br/>2. add rescoring (no rebuild)<br/>3. change compression (rebuild, same dials)<br/>4. change family (rebuild, new dials)"]

    classDef family stroke:#e8590c,stroke-width:3px
    class A1,A2 family
```

## How they differ

| | Option A | Option B |
|---|---|---|
| What it teaches | Which path to take, question by question | That an index is four independent picks |
| "Family" | Mixed with placement and compression | Only layer 1, orange-bordered |
| Decision criteria | Explicit (fits RAM? filters selective?) | Short hints inside each option |
| Remediation | One box, no order | Cheapest-first ladder, marked rebuild or not |
| LogiTrack's pilot | Not shown | "Exact search: no index" exit at the top |
