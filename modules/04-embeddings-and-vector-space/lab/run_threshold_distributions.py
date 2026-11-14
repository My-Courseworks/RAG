"""Measure M4's "Worked: setting a similarity threshold from two distributions" on real open models.

Builds ~200 LogiTrack-style clause chunks from templates (deterministic given --seed), derives three
labelled pair sets from them, scores every pair with each model (cosine on L2-normalized embeddings,
local CPU, no API), and prints the score distributions, the cutoff each model implies, and what happens
when one model's cutoff is carried over to another.

Pair sets:
  duplicates      the same clause with meaning-preserving differences only: a re-upload under a new
                  filename and page, the same boilerplate under another carrier's agreement header, or
                  re-wrapped lines with typographic punctuation normalized. Never a byte copy.
  hard negatives  the same chunk with exactly ONE figure changed (a percentage, USD amount, day count
                  or date); every other byte identical.
  random pairs    two different chunks.

Usage (from this folder):
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_threshold_distributions.py                     # seed 0, e5 + BGE + raw BERT
    python run_threshold_distributions.py --seed 1            # another corpus and pair draw
    python run_threshold_distributions.py --models e5 bge --examples
    python run_threshold_distributions.py --models e5 bge ada002 te3small te3large   # paid OpenAI API, opt-in

The OpenAI models read the key from OPENAI_API_KEY, else ~/.config/openai_key, and cache embeddings
outside the repo (--cache), so reruns don't re-bill.

Why not labs/generate_corpus.py: about 94% of its words are filler drawn from a 30-word vocabulary
("clearance consignee schedule notice ..."), so its chunks are word salad, and its real clause text
comes from four sentence templates. Chunks of either would make the three pair sets say nothing
about clauses. The templates below are clause-shaped and carry the figures a hard negative changes.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModel, AutoTokenizer

# --- models -------------------------------------------------------------------------------------
# prefix and pooling follow each model card; see lab/README.md for the quotes.
MODELS = {
    "e5": dict(name="intfloat/e5-base-v2", prefix="query: ", pooling="mean"),  # card: "query: " for symmetric tasks
    "bge": dict(name="BAAI/bge-base-en-v1.5", prefix="", pooling="cls"),        # card: no instruction for passages
    "bert": dict(name="bert-base-uncased", prefix="", pooling="mean"),          # untrained for similarity
    # Paid OpenAI models: opt-in only, never in the default --models. usd_per_1m = list price per 1M tokens.
    "ada002": dict(name="text-embedding-ada-002", api="openai", usd_per_1m=0.10),
    "te3small": dict(name="text-embedding-3-small", api="openai", usd_per_1m=0.02),
    "te3large": dict(name="text-embedding-3-large", api="openai", usd_per_1m=0.13),
}
OPEN_MODELS = [k for k, spec in MODELS.items() if "api" not in spec]
OPENAI_URL = "https://api.openai.com/v1/embeddings"
OPENAI_KEY_FILE = Path("~/.config/openai_key").expanduser()
DEFAULT_CACHE = Path("~/.cache/rag-course/m4-openai-embeddings.sqlite").expanduser()

# --- corpus vocabulary (course fiction) ----------------------------------------------------------
CARRIERS = ["Maersk", "MSC", "CMA CGM", "Hapag-Lloyd", "ONE", "Evergreen", "COSCO", "HMM",
            "Yang Ming", "ZIM", "PIL", "Wan Hai"]
JURISDICTIONS = ["NL", "DE", "SG", "US", "GB", "BE", "FR", "AE", "JP", "KR", "IN", "BR"]
PORTS = ["Rotterdam", "Singapore", "Hamburg", "Antwerp", "Felixstowe", "Jebel Ali", "Shanghai",
         "Busan", "Long Beach", "Santos", "Mumbai", "Durban"]
CONTAINERS = ["20ft", "40ft", "40ft HC", "45ft HC", "20ft reefer", "40ft reefer"]
UN_CODES = [("UN3480", "lithium-ion batteries"), ("UN1263", "paint and paint-related material"),
            ("UN1993", "flammable liquids, n.o.s."), ("UN3077", "environmentally hazardous substances"),
            ("UN1789", "hydrochloric acid"), ("UN1203", "motor spirit")]
COMMODITIES = ["temperature-controlled pharma", "fresh produce", "frozen seafood", "vaccines",
               "cut flowers", "dairy products"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def _date(rng):
    return f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(2025, 2027)}"


def _usd(v):
    return f"{v:,}"


def _lane(rng):
    a, b = rng.sample(PORTS, 2)
    return f"{a}–{b}"  # en dash, as typeset contracts write lanes


# Each template returns (clause title, body). Every body carries at least one figure of the four
# kinds a hard negative may change, and at least one typographic character (’ or –) so the
# "reformatted" duplicate changes tokens, not just whitespace.
def t_dg_surcharge(rng):
    code, desc = rng.choice(UN_CODES)
    p = rng.choice([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0])
    return "Dangerous Goods Surcharge", (
        f"For consignments classified under {code} ({desc}), the Carrier shall levy a dangerous goods "
        f"surcharge of {p}% of the applicable base freight rate. Where the Shipper has lodged a valid "
        f"multimodal dangerous goods declaration no later than {rng.choice([24, 48, 72])} hours before "
        f"the documentation cut-off, the surcharge shall be reduced to {p - 1.0}%. The Carrier’s "
        f"acceptance of a booking does not waive any segregation requirement.")


def t_free_time(rng):
    d, k = rng.randint(4, 14), rng.randint(3, 7)
    a = rng.randrange(60, 200, 5)
    return "Free Time and Demurrage", (
        f"Free time for {rng.choice(CONTAINERS)} containers discharged at {rng.choice(PORTS)} shall be "
        f"{d} calendar days from the day of discharge. Thereafter demurrage accrues at USD {a} per "
        f"container per day for days {d + 1}–{d + k}, and at USD {a + rng.randrange(40, 120, 5)} per "
        f"container per day from day {d + k + 1} until the container is gated out.")


def t_detention(rng):
    d = rng.randint(5, 21)
    return "Detention", (
        f"The Merchant may retain the Carrier’s equipment free of charge for {d} days from gate-out. "
        f"Equipment not returned empty to the designated depot within that period incurs detention "
        f"of USD {rng.randrange(45, 160, 5)} per unit per day. Detention is invoiced monthly and is "
        f"not subject to the free-time extension in clause {rng.randint(3, 9)}.{rng.randint(1, 4)}.")


def t_claims(rng):
    d, m = rng.choice([3, 5, 7, 10, 14]), rng.choice([6, 9, 12])
    return "Claims and Time Bars", (
        f"Notice of loss of or damage to the goods must be given to the Carrier in writing within {d} "
        f"days of delivery, failing which delivery is prima facie evidence of receipt in good order. "
        f"The Carrier’s liability is discharged unless suit is brought within {m} months of delivery, "
        f"or of the date the goods should have been delivered.")


def t_liability(rng):
    return "Limitation of Liability", (
        f"The Carrier’s liability for loss of or damage to the goods shall not exceed USD "
        f"{_usd(rng.randrange(500, 2500, 50))} per package or USD {rng.choice([2, 3, 4, 5])}.{rng.randint(0, 9)}0 "
        f"per kilogram of gross weight, whichever is higher. The limit does not apply where the nature and "
        f"value of the goods were declared before shipment and inserted in the transport document. "
        f"Liability for delay is limited to {rng.choice([100, 150, 200])}% of the freight payable.")


def t_payment(rng):
    d = rng.choice([14, 21, 30, 45, 60])
    p = rng.choice([1.0, 1.5, 2.0])
    return "Payment Terms", (
        f"Invoices are payable in full within {d} days of the invoice date, without set-off or "
        f"deduction. Overdue amounts bear interest at {p}% per month from the due date until payment. "
        f"Where an invoice remains unpaid {rng.choice([30, 45, 60])} days after the due date, the "
        f"Carrier may exercise a lien over the Merchant’s goods in its custody.")


def t_reefer(rng):
    t = rng.choice([-25, -18, 2, 4, 5, 8, 13])
    return "Temperature Control", (
        f"Refrigerated containers carrying {rng.choice(COMMODITIES)} shall be maintained at a set point of "
        f"{t} °C with a tolerance of ±{rng.choice([0.5, 1.0, 2.0])} °C. Any excursion lasting longer than "
        f"{rng.choice([30, 60, 90])} minutes shall be reported to the Shipper within {rng.choice([2, 4, 6])} "
        f"hours. The Carrier’s liability for temperature loss is limited to {rng.choice([50, 75, 100])}% "
        f"of the invoice value of the affected cargo.")


def t_baf(rng):
    return "Bunker Adjustment Factor", (
        f"A bunker adjustment factor of USD {rng.randrange(120, 680, 10)} per TEU applies to all "
        f"shipments on the {_lane(rng)} lane with effect from {_date(rng)}. The factor is reviewed every "
        f"{rng.choice([30, 60, 90])} days against the average bunker price at {rng.choice(PORTS)} and "
        f"changes only when that price moves by more than {rng.choice([5, 8, 10])}%.")


def t_pss(rng):
    return "Peak Season Surcharge", (
        f"A peak season surcharge of USD {_usd(rng.randrange(300, 1500, 50))} per FEU applies to "
        f"bookings on the {_lane(rng)} lane loaded between {_date(rng)} and {_date(rng)}. The Carrier "
        f"shall give no less than {rng.choice([14, 21, 30])} days’ written notice of any change to the "
        f"surcharge amount or period.")


def t_termination(rng):
    d, r = rng.choice([30, 60, 90, 180]), rng.choice([14, 21, 30])
    return "Termination", (
        f"Either party may terminate this Agreement for convenience on {d} days’ written notice to the "
        f"other. Either party may terminate with immediate effect where the other commits a material "
        f"breach that is not remedied within {r} days of written notice specifying the breach. "
        f"Termination does not affect shipments already in transit.")


def t_insurance(rng):
    return "Insurance", (
        f"The Merchant shall insure the goods for not less than {rng.choice([100, 110, 120])}% of their "
        f"CIF value against all risks of physical loss or damage. Any deductible above USD "
        f"{_usd(rng.randrange(1000, 10000, 500))} per occurrence shall be borne by the Merchant. "
        f"Certificates of insurance shall be produced within {rng.choice([5, 7, 10])} days of the Carrier’s request.")


def t_customs_bond(rng):
    return "Customs Bond and Entry", (
        f"Goods entered for transit shall be covered by a customs bond of not less than USD "
        f"{_usd(rng.randrange(25000, 150000, 5000))}. The entry summary must be filed within "
        f"{rng.choice([3, 5, 10])} working days of arrival, and the declarant’s records retained for "
        f"{rng.choice([3, 5, 7])} years. A late filing penalty of {rng.choice([1, 2, 5])}% of the duty due applies.")


def t_mqc(rng):
    return "Minimum Quantity Commitment", (
        f"The Shipper commits to tender a minimum of {rng.randrange(200, 3000, 50)} TEU in each contract "
        f"year beginning {_date(rng)}. Any shortfall below {rng.choice([80, 85, 90])}% of that commitment "
        f"incurs a fee of USD {rng.randrange(50, 300, 10)} per TEU not tendered, invoiced within "
        f"{rng.choice([30, 45])} days of the contract year’s end.")


def t_gri(rng):
    return "General Rate Increase", (
        f"The Carrier may apply a general rate increase of up to {rng.choice([5, 7.5, 10, 12.5, 15])}% to "
        f"the base rates in Annex B with effect from {_date(rng)}, provided it gives the Shipper "
        f"{rng.choice([14, 30, 45])} days’ written notice. A Shipper that objects may withdraw affected "
        f"lanes such as {_lane(rng)} without penalty.")


def t_vgm(rng):
    return "Verified Gross Mass", (
        f"The Shipper shall submit the verified gross mass of each packed container no later than "
        f"{rng.choice([12, 24, 48])} hours before vessel loading. Containers exceeding "
        f"{_usd(rng.randrange(26000, 30500, 100))} kg are subject to an overweight surcharge of USD "
        f"{rng.randrange(150, 600, 25)} per container, and may be rolled to the next sailing at the Shipper’s cost.")


def t_cutoff(rng):
    return "Terminal Cut-off Change", (
        f"With effect from {_date(rng)}, the full-container cut-off at {rng.choice(PORTS)} moves to "
        f"{rng.choice([8, 12, 16, 18])}:00 local time, {rng.choice([24, 36, 48])} hours before scheduled "
        f"departure. Containers gated in after the cut-off will be rolled, and storage of USD "
        f"{rng.randrange(30, 120, 5)} per day will apply from the Carrier’s next available sailing.")


TEMPLATES = [t_dg_surcharge, t_free_time, t_detention, t_claims, t_liability, t_payment, t_reefer,
             t_baf, t_pss, t_termination, t_insurance, t_customs_bond, t_mqc, t_gri, t_vgm, t_cutoff]
CUSTOMS_TEMPLATES = {t_customs_bond}
BULLETIN_TEMPLATES = {t_cutoff}


def make_source(rng, template):
    """Return a dict describing the document the chunk came from: header fields only."""
    if template in CUSTOMS_TEMPLATES:
        jur, rev = rng.choice(JURISDICTIONS), rng.randint(1, 6)
        return dict(kind="customs", jur=jur, rev=rev, page=rng.randint(2, 60))
    if template in BULLETIN_TEMPLATES:
        return dict(kind="bulletin", no=rng.randint(1, 9999), page=1)
    return dict(kind="msa", carrier=rng.choice(CARRIERS), jur=rng.choice(JURISDICTIONS),
                rev=rng.randint(1, 9), page=rng.randint(2, 80))


def header(src, filename=None):
    if src["kind"] == "customs":
        title = f"Customs Procedure Manual — {src['jur']}, rev. {src['rev']}"
        default = f"customs_manual_{src['jur']}_rev{src['rev']}.pdf"
    elif src["kind"] == "bulletin":
        title = f"Service Bulletin SB-{src['no']:04d}"
        default = f"SB-{src['no']:04d}.pdf"
    else:
        title = f"Master Service Agreement — {src['carrier']} / LogiTrack ({src['jur']}), v{src['rev']}"
        default = f"MSA_{src['carrier'].replace(' ', '-')}_{src['jur']}_v{src['rev']}.pdf"
    return f"{title} | {filename or default} | p. {src['page']}"


def render(src, clause_no, title, body, width=92, filename=None):
    return f"{header(src, filename)}\n\n{clause_no} {title}\n" + textwrap.fill(body, width)


def make_chunks(n, rng):
    """n chunks, each a dict with the fields needed to re-render it with meaning-preserving changes."""
    chunks = []
    for i in range(n):
        template = TEMPLATES[i % len(TEMPLATES)] if i < len(TEMPLATES) else rng.choice(TEMPLATES)
        title, body = template(rng)
        chunk = dict(src=make_source(rng, template), clause=f"{rng.randint(2, 18)}.{rng.randint(1, 5)}",
                     title=title, body=body)
        chunk["text"] = render(chunk["src"], chunk["clause"], chunk["title"], chunk["body"])
        chunks.append(chunk)
    return chunks


# --- duplicates: meaning-preserving variants -----------------------------------------------------
TYPOGRAPHY = str.maketrans({"’": "'", "–": "-", "—": "-", "±": "+/-"})


def reupload(chunk, rng):
    """Same page, new filename and page number (re-uploaded under another name or in another bundle)."""
    src = dict(chunk["src"], page=max(1, chunk["src"]["page"] + rng.choice([-3, -1, 1, 2, 5, 11])))
    stem = re.sub(r"\.pdf$", "", header(chunk["src"]).split(" | ")[1])
    filename = rng.choice([f"{stem} (1).pdf", f"{stem}_FINAL.pdf", f"scan_{rng.randint(10000, 99999)}.pdf",
                           f"{stem}_signed.pdf", f"Copy of {stem}.pdf"])
    return render(src, chunk["clause"], chunk["title"], chunk["body"], filename=filename)


def other_carrier(chunk, rng):
    """The same boilerplate clause in another carrier's agreement: only the header differs."""
    carrier = rng.choice([c for c in CARRIERS if c != chunk["src"]["carrier"]])
    src = dict(chunk["src"], carrier=carrier, jur=rng.choice(JURISDICTIONS), page=rng.randint(2, 80))
    return render(src, chunk["clause"], chunk["title"], chunk["body"])


def reformatted(chunk, rng):
    """Same header, body re-extracted: line breaks moved, typography normalized, clause title upper-cased."""
    body = chunk["body"].translate(TYPOGRAPHY)
    title = chunk["title"].upper() if rng.random() < 0.5 else chunk["title"]
    text = render(chunk["src"], chunk["clause"], title, body, width=rng.choice([60, 72, 120, 10_000]))
    return text.replace(". ", ".  ") if rng.random() < 0.5 else text


def make_duplicates(chunks, n, rng):
    pairs = []
    for chunk in rng.sample(chunks, n):
        kinds = [reupload, reformatted] + ([other_carrier] if chunk["src"]["kind"] == "msa" else [])
        kind = rng.choice(kinds)
        variant = kind(chunk, rng)
        assert variant != chunk["text"]
        pairs.append((chunk["text"], variant, kind.__name__))
    return pairs


# --- hard negatives: exactly one figure changed --------------------------------------------------
# Each group captures only the number, so the swap leaves every other byte (including the line
# breaks textwrap put between "USD" and its amount) untouched.
FIGURE = re.compile(
    r"\b(?P<date>\d{1,2})(?=\s+(?:" + "|".join(MONTHS) + r")\s+\d{4})"
    r"|USD\s+(?P<usd>\d{1,3}(?:,\d{3})+|\d+)(?![.,]\d)"
    r"|(?P<pct>\d+(?:\.\d+)?)%"
    r"|\b(?P<days>\d+)(?=\s+(?:calendar\s+|working\s+)?days)")


def changed_figure(kind, s, rng):
    """Return a different value for one figure, in the same format."""
    if kind == "date":                                     # the day of the month
        return str((int(s) - 1 + rng.choice([1, 2, 7, 14])) % 28 + 1)
    if kind == "usd":
        v = int(s.replace(",", ""))
        step = max(5, round(v * rng.choice([0.1, 0.2, 0.25]) / 5) * 5)
        return _usd(v - step if v > step and rng.random() < 0.5 else v + step)
    if kind == "pct":
        v, decimals = float(s), len(s.split(".")[1]) if "." in s else 0
        step = rng.choice([0.5, 1.0]) if decimals else rng.choice([1, 2, 5])
        return f"{v - step if v > step and rng.random() < 0.5 else v + step:.{decimals}f}"
    v, step = int(s), rng.choice([1, 2, 3, 7])             # a day count
    return str(v - step if v > step and rng.random() < 0.5 else v + step)


def make_hard_negatives(chunks, n, rng):
    pairs = []
    for chunk in rng.sample(chunks, n):
        text = chunk["text"]
        body_start = text.index("\n\n") + 2                  # never touch the header
        matches = [m for m in FIGURE.finditer(text) if m.start() >= body_start]
        m = rng.choice(matches)
        kind = m.lastgroup
        start, end = m.span(kind)
        new = text[:start] + changed_figure(kind, m.group(kind), rng) + text[end:]
        assert new != text
        pairs.append((text, new, kind))
    return pairs


def make_random_pairs(chunks, n, rng):
    return [tuple(chunks[i]["text"] for i in rng.sample(range(len(chunks)), 2)) + ("random",)
            for _ in range(n)]


# --- scoring -------------------------------------------------------------------------------------
@torch.no_grad()
def embed(texts, spec, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(spec["name"])
    model = AutoModel.from_pretrained(spec["name"]).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer([spec["prefix"] + t for t in texts[i:i + batch_size]], max_length=512,
                          padding=True, truncation=True, return_tensors="pt")
        hidden = model(**batch).last_hidden_state
        if spec["pooling"] == "cls":
            pooled = hidden[:, 0]
        else:
            mask = batch["attention_mask"][..., None].to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        out.append(F.normalize(pooled.double(), p=2, dim=1))  # float64: hard negatives sit near 1.0
    config_path = Path(try_to_load_from_cache(spec["name"], "config.json"))  # .../snapshots/<commit>/
    return torch.cat(out).numpy(), config_path.parent.name[:10]


class ModelUnavailable(Exception):
    """The API says the model doesn't exist or is retired; the run records it and carries on."""


def openai_key():
    """OPENAI_API_KEY if set, else the bare key in ~/.config/openai_key. Never printed or logged."""
    key = os.environ.get("OPENAI_API_KEY") or OPENAI_KEY_FILE.read_text(encoding="utf-8")
    return key.strip()


def post_embeddings(model, inputs, key, attempts=5):
    """One POST /v1/embeddings call, retried on 429 and 5xx. Errors report only status and error type."""
    body = json.dumps({"model": model, "input": inputs, "encoding_format": "float"}).encode()
    for attempt in range(attempts):
        request = urllib.request.Request(OPENAI_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            try:
                error = json.load(err).get("error") or {}
            except ValueError:
                error = {}
            kind = error.get("code") or error.get("type") or "unknown"
            if err.code == 404 or kind in ("model_not_found", "model_deprecated", "deprecated"):
                raise ModelUnavailable(f"HTTP {err.code}, {kind}") from None
            if (err.code == 429 or err.code >= 500) and attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenAI embeddings call failed: HTTP {err.code}, {kind}") from None
        except urllib.error.URLError as err:
            raise RuntimeError(f"OpenAI embeddings call failed: {type(err.reason).__name__}") from None


def embed_openai(texts, spec, cache_path, batch_size=512, post=post_embeddings):
    """Embed via the OpenAI REST API, reading and writing a (model, sha256(text)) -> vector cache.

    Returns (vectors, label, tokens billed this run). Only uncached texts are sent.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cache_path)
    db.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, sha TEXT, vec BLOB, PRIMARY KEY (model, sha))")
    shas = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    cached = dict(db.execute(f"SELECT sha, vec FROM emb WHERE model = ? AND sha IN ({','.join('?' * len(shas))})",
                             [spec["name"], *shas]).fetchall())
    missing = sorted({sha: t for sha, t in zip(shas, texts) if sha not in cached}.items())
    tokens = 0
    if missing:
        key = openai_key()
        for i in range(0, len(missing), batch_size):           # the API allows up to 2048 inputs per call
            batch = missing[i:i + batch_size]
            reply = post(spec["name"], [t for _, t in batch], key)
            tokens += reply["usage"]["total_tokens"]
            for item in sorted(reply["data"], key=lambda d: d["index"]):
                vec = np.asarray(item["embedding"], dtype=np.float64)
                cached[batch[item["index"]][0]] = vec.tobytes()
                db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
                           (spec["name"], batch[item["index"]][0], vec.tobytes()))
            db.commit()
    db.close()
    vectors = np.stack([np.frombuffer(cached[sha], dtype=np.float64) for sha in shas])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, f"api ({len(texts) - len(missing)} cached, {len(missing)} sent)", tokens


def score_pairs(pair_sets, spec, cache_path):
    """Score every pair; returns (scores by set, model revision or cache label, tokens billed)."""
    texts = sorted({t for pairs in pair_sets.values() for a, b, _ in pairs for t in (a, b)})
    if spec.get("api") == "openai":
        vectors, revision, tokens = embed_openai(texts, spec, cache_path)
    else:
        (vectors, revision), tokens = embed(texts, spec), 0
    index = {t: i for i, t in enumerate(texts)}
    scores = {name: np.array([float(vectors[index[a]] @ vectors[index[b]]) for a, b, _ in pairs])
              for name, pairs in pair_sets.items()}
    return scores, revision, tokens


def module_cutoff(hard_negatives, cap=0.01):
    """The module's rule: the lowest value with <= cap of hard negatives strictly above it."""
    return float(np.sort(hard_negatives)[::-1][int(cap * len(hard_negatives))])


def balance_cutoff(duplicates, hard_negatives):
    """The candidate cutoff that maximizes (share of duplicates above) - (share of hard negatives above)."""
    candidates = np.concatenate([duplicates, hard_negatives])
    gain = [(duplicates > c).mean() - (hard_negatives > c).mean() for c in candidates]
    return float(candidates[int(np.argmax(gain))])


def naive_cutoff(duplicates):
    """What a team sets when it checks duplicates against random pairs only: the duplicates' 5th percentile."""
    return float(np.percentile(duplicates, 5))


def summary_rows(s):
    """The printed table for one model: (label, value, format) rows, in print order."""
    dup, hn, rnd = s["duplicates"], s["hard_negatives"], s["random"]
    cut, bal, naive = module_cutoff(hn), balance_cutoff(dup, hn), naive_cutoff(dup)
    gap = np.percentile(dup, 5) - np.percentile(hn, 99)
    return [
        ("random pairs, p5", np.percentile(rnd, 5), ".6f"),
        ("random pairs, median", np.median(rnd), ".6f"),
        ("hard negatives, median", np.median(hn), ".6f"),
        ("hard negatives, p99", np.percentile(hn, 99), ".6f"),
        ("duplicates, p5", np.percentile(dup, 5), ".6f"),
        ("duplicates, median", np.median(dup), ".6f"),
        ("gap (dup p5 - hardneg p99)", gap, "+.6f"),
        ("clean gap (dup min > hardneg max)?", "yes" if dup.min() > hn.max() else "no", ""),
        ("hard negs above dup median", (hn > np.median(dup)).mean(), ".1%"),
        ("CUTOFF (<=1% hard negs above it)", cut, ".6f"),
        ("  dups caught", (dup > cut).mean(), ".1%"),
        ("  hard negs dropped", (hn > cut).mean(), ".1%"),
        ("balance cutoff (max caught - dropped)", bal, ".6f"),
        ("  dups caught", (dup > bal).mean(), ".1%"),
        ("  hard negs dropped", (hn > bal).mean(), ".1%"),
        ("naive cutoff (dup p5, no hard negs)", naive, ".6f"),
        ("  dups caught", (dup > naive).mean(), ".1%"),
        ("  hard negs dropped", (hn > naive).mean(), ".1%"),
    ]


def print_medians_by_kind(title, pairs, scores, models, set_name):
    print(f"\n{title}, median score by kind:")
    for kind in sorted({kind for _, _, kind in pairs}):
        mask = np.array([k == kind for _, _, k in pairs])
        print(f"  {kind:<14} n={mask.sum():<4}" + "".join(
            f"{m:>9} {np.median(scores[m][set_name][mask]):.6f}" for m in models))


def print_carry_over(label, cutoff_rule, scores, models):
    print(f"\nCarry-over of the {label}: cutoff from one model applied to another's scores")
    print("  (from cutoff -> to: dups caught / hard negs dropped)")
    for src in models:
        cut = cutoff_rule(scores[src])
        for dst in models:
            if dst != src:
                d, h = scores[dst]["duplicates"], scores[dst]["hard_negatives"]
                print(f"  {src:<8} {cut:.6f} -> {dst:<8}: {(d > cut).mean():6.1%} / {(h > cut).mean():6.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=200)
    parser.add_argument("--n-dups", type=int, default=120)
    parser.add_argument("--n-hardnegs", type=int, default=120)
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=OPEN_MODELS,
                        help="ada002, te3small and te3large call the paid OpenAI API")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="OpenAI embedding cache (keep it outside the repo)")
    parser.add_argument("--examples", action="store_true", help="print one pair of each kind")
    args = parser.parse_args()
    transformers.logging.set_verbosity_error()   # hides bert-base-uncased's unused-MLM-head report
    transformers.logging.disable_progress_bar()

    rng = random.Random(args.seed)
    chunks = make_chunks(args.n_chunks, rng)
    pair_sets = {"duplicates": make_duplicates(chunks, args.n_dups, rng),
                 "hard_negatives": make_hard_negatives(chunks, args.n_hardnegs, rng),
                 "random": make_random_pairs(chunks, args.n_random, rng)}
    print(f"seed {args.seed} | {len(chunks)} chunks | "
          + " | ".join(f"{k} {len(v)}" for k, v in pair_sets.items())
          + f" | torch {torch.__version__} | transformers {transformers.__version__}")
    if args.examples:
        for name, pairs in pair_sets.items():
            a, b, kind = pairs[0]
            print(f"\n--- {name} ({kind}) ---\n{a}\n~~~\n{b}")
        print()

    scores, tables, billed = {}, {}, {}
    for key in args.models:
        spec = MODELS[key]
        try:
            scores[key], revision, billed[key] = score_pairs(pair_sets, spec, args.cache)
        except ModelUnavailable as err:
            print(f"UNAVAILABLE {key:<8} {spec['name']}: {err}; skipped", flush=True)
            continue
        tables[key] = summary_rows(scores[key])
        print(f"scored {key:<8} {spec['name']} @ {revision}", flush=True)
    models = list(scores)
    if not models:
        return

    print(f"\n{'':<40}" + "".join(f"{k:>10}" for k in models))
    for i, (label, _, fmt_spec) in enumerate(tables[models[0]]):
        print(f"{label:<40}" + "".join(f"{format(tables[k][i][1], fmt_spec):>10}" for k in models))

    print_medians_by_kind("Duplicates", pair_sets["duplicates"], scores, models, "duplicates")
    print_medians_by_kind("Hard negatives", pair_sets["hard_negatives"], scores, models, "hard_negatives")
    print_carry_over("CUTOFF", lambda s: module_cutoff(s["hard_negatives"]), scores, models)
    print_carry_over("naive cutoff", lambda s: naive_cutoff(s["duplicates"]), scores, models)

    paid = [k for k in models if "usd_per_1m" in MODELS[k]]
    if paid:
        print("\nOpenAI tokens billed this run (cached texts cost nothing):")
        for k in paid:
            cost = billed[k] * MODELS[k]["usd_per_1m"] / 1e6
            print(f"  {k:<8} {billed[k]:>8,} tokens  ${cost:.5f}")
        total = sum(billed[k] * MODELS[k]["usd_per_1m"] / 1e6 for k in paid)
        print(f"  total    {sum(billed[k] for k in paid):>8,} tokens  ${total:.5f}")

if __name__ == "__main__":
    main()
