#!/usr/bin/env python3
"""
Generate the LogiTrack synthetic corpus for the RAG course labs.

Deterministic given --seed. Produces the six source types described in syllabus §2, each with
the structural properties the course's problems depend on, plus a stratified, span-labelled
query set.

This is course fiction: the corpus is invented, and no citation burden attaches to its content.
It exists so every learner measures against the same data.

Usage:
    python generate_corpus.py --out ./data --seed 1337 --docs 3000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary. Deliberately includes domain jargon, exact identifiers (part
# numbers / clause IDs / UN codes) that defeat dense retrieval, and near-
# duplicate phrasing across jurisdictions that creates a dedup problem.
# ---------------------------------------------------------------------------

JURISDICTIONS = [
    "NL", "DE", "SG", "US", "GB", "BE", "FR", "IT", "ES", "PL",
    "AE", "JP", "KR", "AU", "CA", "BR", "IN", "ZA", "TR", "MX",
]

CARRIERS = [
    "Maersk", "MSC", "CMA CGM", "Hapag-Lloyd", "ONE", "Evergreen",
    "COSCO", "HMM", "Yang Ming", "ZIM", "PIL", "Wan Hai",
]

CONTAINERS = ["20ft", "40ft", "40ft HC", "45ft HC", "20ft reefer", "40ft reefer"]

COMMODITIES = [
    "general cargo", "temperature-controlled pharma", "lithium-ion batteries",
    "industrial solvents", "automotive parts", "textiles", "fresh produce",
    "machinery", "hazardous waste", "medical devices",
]

DG_UN_CODES = [
    "UN3480", "UN3481", "UN1263", "UN1993", "UN3077", "UN3082",
    "UN1789", "UN3264", "UN1203", "UN1950",
]

PORTS = [
    "Rotterdam", "Singapore", "Hamburg", "Antwerp", "Felixstowe", "Jebel Ali",
    "Shanghai", "Busan", "Los Angeles", "Long Beach", "Santos", "Mumbai",
    "Durban", "Istanbul", "Veracruz", "Tokyo",
]

CLAUSE_TOPICS = [
    "Demurrage and Detention", "Freight Charges and Surcharges", "Liability and Limitation",
    "Force Majeure", "Dangerous Goods", "Temperature Control", "Customs Compliance",
    "Termination", "Insurance", "Governing Law and Jurisdiction", "Payment Terms",
    "Documentation Requirements", "Subcontracting", "Claims and Time Bars",
]

# Sentences engineered so that the ANSWER straddles a plausible chunk boundary:
# the operative number sits in the first half, the disambiguating qualifier in the
# second. This is what makes F3 (chunk-boundary loss) reproducible instead of anecdotal.
STRADDLE_SENTENCES = [
    (
        "Demurrage liability shall accrue from the {n}th ({n}th) calendar day following discharge, "
        "and shall be extinguished on the {m}th ({m}th) calendar day, "
        "after which the carrier's liability for delay shall cease entirely"
    ),
    (
        "The free time allowed for container usage shall be {n} ({n}) calendar days from "
        "discharge, provided that such free time shall not extend beyond the {m}th ({m}th) day "
        "following the completion of customs formalities"
    ),
    (
        "The dangerous goods surcharge for commodities classified under {code} shall be "
        "{n} percent of the applicable base freight rate, unless the shipper has furnished a "
        "valid multimodal dangerous goods declaration, in which case the surcharge shall be "
        "reduced to {m} percent"
    ),
    (
        "Claims for cargo damage must be notified to the carrier in writing within {n} days of "
        "delivery, failing which the claim shall be deemed waived, save that where the damage was "
        "not reasonably discoverable on delivery the period shall be {m} days from the date "
        "discovery could reasonably have been made"
    ),
]


@dataclass
class Doc:
    doc_id: str
    source_type: str
    title: str
    jurisdiction: str
    client_id: str
    effective_date: str
    version: str
    supersedes: str | None
    text: str
    metadata: dict = field(default_factory=dict)


def _date(rng: random.Random) -> str:
    return f"{rng.randint(2021, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _para(rng: random.Random, words: int) -> str:
    """Neutral filler prose of roughly `words` words."""
    vocab = (
        "carrier shipper consignee freight container vessel voyage port terminal customs "
        "clearance documentation manifest bill lading clause provision regulation compliance "
        "obligation liability indemnity warranty notice period charge tariff schedule"
    ).split()
    return " ".join(rng.choice(vocab) for _ in range(words)) + "."


# ---------------------------------------------------------------------------
# Generators, one per source type
# ---------------------------------------------------------------------------

def gen_carrier_agreement(rng: random.Random, i: int) -> Doc:
    """Hierarchical, clause-numbered legal prose. Contains cross-refs and straddle answers."""
    carrier = rng.choice(CARRIERS)
    jur = rng.choice(JURISDICTIONS)
    lines = [f"# Master Service Agreement — {carrier} / LogiTrack ({jur})", ""]
    lines.append("## 1. Definitions and Interpretation")
    lines.append(_para(rng, rng.randint(370, 680)))
    lines.append("")

    clause_no = 2
    for topic in rng.sample(CLAUSE_TOPICS, k=min(len(CLAUSE_TOPICS), rng.randint(8, 14))):
        lines.append(f"## {clause_no}. {topic}")
        for sub in range(1, rng.randint(2, 4)):
            lines.append(f"### {clause_no}.{sub}")
            lines.append(_para(rng, rng.randint(320, 650)))
            # Inject a cross-document reference (creates the M8 multi-hop problem)
            if rng.random() < 0.35:
                other = rng.choice(CLAUSE_TOPICS)
                tgt = rng.randint(2, 14)
                lines.append(
                    f"The provisions of this clause are subject to clause {tgt}.{rng.randint(1, 3)} "
                    f"({other}) of the applicable customs procedure manual."
                )
            # Inject the boundary-straddling answer sentence (creates F3)
            if rng.random() < 0.45:
                tpl = rng.choice(STRADDLE_SENTENCES)
                lines.append(
                    tpl.format(
                        n=rng.randint(3, 9), m=rng.randint(11, 21), code=rng.choice(DG_UN_CODES)
                    ) + "."
                )
            lines.append("")
        clause_no += 1

    lines.append(f"## {clause_no}. Governing Law and Jurisdiction")
    lines.append(f"This Agreement shall be governed by the laws of {jur}.")
    return Doc(
        doc_id=f"csa-{i:05d}",
        source_type="carrier_agreement",
        title=f"Master Service Agreement — {carrier} ({jur})",
        jurisdiction=jur,
        client_id=rng.choice([f"CL{n:03d}" for n in range(1, 13)]),
        effective_date=_date(rng),
        version=f"v{rng.randint(1, 9)}",
        supersedes=None,
        text="\n".join(lines),
        metadata={"carrier": carrier, "clauses": clause_no},
    )


def gen_customs_manual(rng: random.Random, i: int) -> Doc:
    """Per-jurisdiction prose with heavy overlap across jurisdictions (dedup problem)."""
    jur = rng.choice(JURISDICTIONS)
    lines = [f"# Customs Procedure Manual — {jur}", ""]
    lines.append("## 1. Scope and Application")
    # Shared boilerplate across all jurisdictions -> near-duplicate documents
    lines.append(
        "This manual sets out the procedures applicable to the import, export, and transit of "
        "goods through the customs territory, and applies to all consignments handled by "
        "LogiTrack or its authorised subcontractors."
    )
    lines.append("")
    for sec, topic in enumerate(
        ["Entry Summary", "Valuation", "Origin", "Duty Relief", "Post-Clearance Audit",
         "Transit", "Warehousing", "Penalties"], start=2
    ):
        lines.append(f"## {sec}. {topic}")
        for sub in range(1, rng.randint(2, 4)):
            lines.append(_para(rng, rng.randint(285, 585)))
            if rng.random() < 0.25:
                tpl = rng.choice(STRADDLE_SENTENCES)
                lines.append(
                    tpl.format(
                        n=rng.randint(3, 9), m=rng.randint(11, 21), code=rng.choice(DG_UN_CODES)
                    ) + "."
                )
            lines.append("")
    return Doc(
        doc_id=f"cpm-{i:05d}",
        source_type="customs_manual",
        title=f"Customs Procedure Manual — {jur}",
        jurisdiction=jur,
        client_id="ALL",
        effective_date=_date(rng),
        version=f"v{rng.randint(1, 5)}",
        supersedes=None,
        text="\n".join(lines),
        metadata={"jurisdiction": jur},
    )


def gen_rate_table(rng: random.Random, i: int) -> Doc:
    """A real grid. Written as markdown table so a naive extract destroys row affiliation."""
    lanes = [(rng.choice(PORTS), rng.choice(PORTS)) for _ in range(rng.randint(6, 14))]
    header = ("Origin|Destination|Container|Base Rate USD|Free Days|DG Surcharge %|"
              "Valid From|Valid To")
    lines = [f"# Tariff Rate Schedule — {rng.choice(CARRIERS)} ({rng.choice(JURISDICTIONS)})", ""]
    lines.append(header)
    lines.append("|".join(["---"] * 8))
    rows = []
    for o, d in lanes:
        for cont in rng.sample(CONTAINERS, k=rng.randint(1, 3)):
            row = {
                "origin": o, "destination": d, "container": cont,
                "base_rate": rng.randrange(1200, 3600, 10),
                "free_days": rng.randint(3, 12),
                "dg_surcharge": round(rng.uniform(1.5, 9.0), 1),
                "valid_from": _date(rng), "valid_to": _date(rng),
            }
            rows.append(row)
            lines.append(
                f"{o}|{d}|{cont}|{row['base_rate']}|{row['free_days']}|"
                f"{row['dg_surcharge']}|{row['valid_from']}|{row['valid_to']}"
            )
    return Doc(
        doc_id=f"trf-{i:05d}",
        source_type="rate_table",
        title=f"Tariff Rate Schedule — {rng.choice(CARRIERS)}",
        jurisdiction=rng.choice(JURISDICTIONS),
        client_id=rng.choice([f"CL{n:03d}" for n in range(1, 13)]),
        effective_date=_date(rng),
        version=f"v{rng.randint(1, 4)}",
        supersedes=None,
        text="\n".join(lines),
        metadata={"rows": rows, "row_count": len(rows)},
    )


def gen_dg_rules(rng: random.Random, i: int) -> Doc:
    """Cross-referenced regulation text -> multi-hop retrieval."""
    jur = rng.choice(JURISDICTIONS)
    lines = [f"# Dangerous Goods Handling Rules — {jur}", ""]
    for idx, code in enumerate(rng.sample(DG_UN_CODES, k=rng.randint(5, 9)), start=1):
        lines.append(f"## {idx}. Class {code}")
        lines.append(_para(rng, rng.randint(280, 520)))
        # Forward and backward cross-references
        other = rng.choice(DG_UN_CODES)
        lines.append(
            f"Consignments bearing {code} shall additionally comply with the segregation "
            f"requirements set out in the entry for {other} in this manual, and with the "
            f"documentation requirements in the applicable customs procedure manual."
        )
        tpl = rng.choice(STRADDLE_SENTENCES)
        lines.append(
            tpl.format(n=rng.randint(3, 9), m=rng.randint(11, 21), code=code) + "."
        )
        lines.append("")
    return Doc(
        doc_id=f"dgr-{i:05d}",
        source_type="dg_rules",
        title=f"Dangerous Goods Handling Rules — {jur}",
        jurisdiction=jur,
        client_id="ALL",
        effective_date=_date(rng),
        version=f"v{rng.randint(1, 6)}",
        supersedes=None,
        text="\n".join(lines),
        metadata={"jurisdiction": jur},
    )


def gen_wiki_ticket(rng: random.Random, i: int) -> Doc:
    """Short, high-signal text — the contrast case for chunk size."""
    topic = rng.choice([
        "How to file a post-clearance amendment",
        "Reefer temperature excursion procedure",
        "Bill of lading correction request",
        "DG declaration rejection handling",
        "Demurrage dispute escalation path",
        "Carrier portal access reset",
    ])
    steps = [
        f"{n}. {_para(rng, rng.randint(8, 20))}" for n in range(1, rng.randint(3, 6))
    ]
    text = f"# {topic}\n\n" + "\n".join(steps)
    if rng.random() < 0.4:
        tpl = rng.choice(STRADDLE_SENTENCES)
        text += "\n\n" + tpl.format(
            n=rng.randint(3, 9), m=rng.randint(11, 21), code=rng.choice(DG_UN_CODES)
        ) + "."
    return Doc(
        doc_id=f"wki-{i:05d}",
        source_type="wiki_ticket",
        title=topic,
        jurisdiction=rng.choice(JURISDICTIONS),
        client_id=rng.choice([f"CL{n:03d}" for n in range(1, 13)]),
        effective_date=_date(rng),
        version="v1",
        supersedes=None,
        text=text,
        metadata={"topic": topic},
    )


def gen_bulletin(rng: random.Random, i: int) -> Doc:
    """Dated bulletins with a supersession chain -> the freshness problem (F7)."""
    jur = rng.choice(JURISDICTIONS)
    text = (
        f"# Service Bulletin {i:04d} — {jur}\n\n"
        f"Effective immediately. {_para(rng, rng.randint(200, 380))}\n\n"
        f"{_para(rng, rng.randint(200, 380))}\n"
    )
    tpl = rng.choice(STRADDLE_SENTENCES)
    text += "\n" + tpl.format(
        n=rng.randint(3, 9), m=rng.randint(11, 21), code=rng.choice(DG_UN_CODES)
    ) + "."
    return Doc(
        doc_id=f"bul-{i:05d}",
        source_type="service_bulletin",
        title=f"Service Bulletin {i:04d} — {jur}",
        jurisdiction=jur,
        client_id="ALL",
        effective_date=_date(rng),
        version="v1",
        supersedes=None,
        text=text,
        metadata={"bulletin_no": i},
    )


GENERATORS = [
    ("carrier_agreement", gen_carrier_agreement, 0.25),
    ("customs_manual", gen_customs_manual, 0.15),
    ("rate_table", gen_rate_table, 0.12),
    ("dg_rules", gen_dg_rules, 0.15),
    ("wiki_ticket", gen_wiki_ticket, 0.25),
    ("service_bulletin", gen_bulletin, 0.08),
]


# ---------------------------------------------------------------------------
# Query generation
#
# NOTE (read M13 before trusting these): the queries are derived from the same
# text the retriever will search, which is exactly the circularity trap M13
# describes. They are valid for COMPARING configurations and invalid as an
# estimate of absolute production quality.
# ---------------------------------------------------------------------------

def build_queries(rng: random.Random, docs: list[Doc], n: int) -> list[dict]:
    queries: list[dict] = []
    by_type: dict[str, list[Doc]] = {}
    for d in docs:
        by_type.setdefault(d.source_type, []).append(d)

    def pick(t: str) -> Doc:
        return rng.choice(by_type[t])

    classes = ["lexical", "semantic", "multi_hop", "numeric", "unanswerable"]
    for qid in range(n):
        cls = classes[qid % len(classes)]
        if cls == "lexical":
            d = pick("carrier_agreement")
            m = rng.randint(1, 6)
            clause = rng.randint(2, 12)
            q = f"What does clause {clause}.{m} of the {d.metadata.get('carrier','carrier')} agreement say about {rng.choice(CLAUSE_TOPICS).lower()}?"
            gold = [d.doc_id]
        elif cls == "semantic":
            d = pick("wiki_ticket")
            q = f"How do I handle {d.title.lower()}?"
            gold = [d.doc_id]
        elif cls == "multi_hop":
            d = pick("dg_rules")
            code = rng.choice(DG_UN_CODES)
            q = (f"Which segregation requirements apply to a consignment bearing {code}, and "
                 f"which manual states them?")
            gold = [d.doc_id]
        elif cls == "numeric":
            d = pick("rate_table")
            row = rng.choice(d.metadata["rows"])
            q = (f"What is the base rate for a {row['container']} from {row['origin']} to "
                 f"{row['destination']}?")
            gold = [d.doc_id]
        else:
            q = ("What is LogiTrack's policy on transporting live animals in refrigerated "
                 "containers?")
            gold = []  # deliberately unanswerable
        queries.append({
            "query_id": f"q{qid:05d}",
            "class": cls,
            "query": q,
            "gold_doc_ids": gold,
            "answerable": cls != "unanswerable",
            "note": "synthetic; derived from generated text (see M13 circularity caveat)",
        })
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./data", type=Path)
    ap.add_argument("--seed", default=1337, type=int)
    ap.add_argument("--docs", default=3000, type=int)
    ap.add_argument("--queries", default=400, type=int)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out: Path = args.out
    (out / "docs").mkdir(parents=True, exist_ok=True)

    weights = [w for _, _, w in GENERATORS]
    docs: list[Doc] = []
    counter = 0
    for idx in range(args.docs):
        _, fn, _ = rng.choices(GENERATORS, weights=weights, k=1)[0]
        counter += 1
        docs.append(fn(rng, counter))

    # Build a supersession chain for bulletins (the F7 freshness problem).
    bulletins = [d for d in docs if d.source_type == "service_bulletin"]
    bulletins.sort(key=lambda d: d.effective_date)
    for prev, cur in zip(bulletins, bulletins[1:]):
        if prev.jurisdiction == cur.jurisdiction:
            cur.supersedes = prev.doc_id

    manifest = []
    for d in docs:
        (out / "docs" / f"{d.doc_id}.md").write_text(d.text, encoding="utf-8")
        rec = asdict(d)
        rec.pop("text", None)
        manifest.append(rec)

    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest), encoding="utf-8"
    )
    queries = build_queries(rng, docs, args.queries)
    (out / "questions.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in queries), encoding="utf-8"
    )

    words = sum(len(d.text.split()) for d in docs)
    print(f"wrote {len(docs)} docs -> {out/'docs'}")
    print(f"wrote {len(manifest)} manifest rows -> {out/'manifest.jsonl'}")
    print(f"wrote {len(queries)} queries -> {out/'questions.jsonl'}")
    print(f"corpus words: {words:,} (~{words/1e6:.2f}M)")
    by_type: dict[str, int] = {}
    for d in docs:
        by_type[d.source_type] = by_type.get(d.source_type, 0) + 1
    print("by type:", json.dumps(by_type))


if __name__ == "__main__":
    main()
