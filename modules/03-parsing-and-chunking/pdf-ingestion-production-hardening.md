# PDF ingestion that survives production

> $\color{red}\text{to be discussed}$ — how to turn messy, untagged PDFs into text, tables and
> links that RAG can trust, at the scale of a multi-customer document engine. Linked from M3's
> reader section ([README](README.md#the-structure-aware-path-parsing-before-chunking)).

## Remember these five things

1. **The unit of work is the region, not the page.**
   - One page can hold native prose, a scanned paragraph, and the first rows of a table.
   - Every decision (native or OCR, which engine, which checks) is made per region.
2. **Pages are processed alone; documents are assembled whole.**
   - First a parallel pass over pages, then passes over the whole document.
   - Anything that crosses a page boundary (a table, a sentence, a reference) is handled at document
     level.
3. **Every table and figure gets an id, and every reference resolves to an id or a condition.**
   - Resolution waits until the whole document is known, because prose can mention a table 40 pages
     before it appears.
4. **Rows go to SQL, prose goes to the vector index, and links connect the two in both directions.**
5. **Check numbers as if money depends on them, because it does.**
   - An OCR'd 8 read as 0 in a rate is a billing error, not a fuzzy answer.

## The case this design must survive

An LTL carrier's pricing agreement, like the ones a carrier such as XPO sends its customers:

- **One rate table about 100 pages long.**
  - Columns: lane (service centre → service centre), NMFC class, weight break, and the rate in $ per
    100 lb.
  - It's long because every lane is a row.
- **Prose rules that refer to a handful of rows.**
  - Discounts, FAK (freight-all-kinds) class overrides, minimum charges, lane exceptions.
- **Mixed sources:**
  - some table pages are native text and some are scanned;
  - some prose pages are partly native and partly scanned, **and the scan continues the sentence**;
  - the table's first page also has prose above the table.

**The rule for this doc:** if a step breaks on this case, the step is wrong.

## 1 · Tools

| Tool | What it is | Use it for |
|---|---|---|
| [Docling](https://docling-project.github.io/docling/) | Layout model, TableFormer for tables, OCR; CPU; MIT | The default for every region |
| [MinerU](https://arxiv.org/pdf/2509.22186) | Layout, table and formula pipeline; GPU | Re-reading suspect table rows and hard layouts |
| [olmOCR](https://arxiv.org/html/2502.18443) | Fine-tuned 7B vision-language model; GPU; Apache-2.0 | Standalone scanned pages with no structure crossing pages |

- **Every engine's output is converted into one format:**
  - regions, each with a page number and a bounding box;
  - tables as **cell grids**, not markdown;
  - for every span, its source (native or OCR) and its OCR confidence.
- **Why one format:**
  - a table read half by Docling (a cell grid) and half by olmOCR (markdown) can't be stitched back
    together;
  - without bounding boxes, a row can't be cited.
- **The limit on olmOCR:** it doesn't appear to return coordinates (confirm on the version you use).
  So it reads only standalone scanned pages, never a segment of a table that crosses pages.
- **ColPali** is a *retriever* (page images to vectors), not a converter. It's deferred in the
  syllabus.
- Every ranking and throughput figure here comes from the projects' own reports. **Measure on your
  own pages before choosing.**

## 2 · The shape: one map over pages, then three passes over the document

```
Pass 1 · per page, in parallel, no state shared between pages
  split the page into regions ─► choose a source per region (native or OCR) ─► extract
  output: prose spans, table segments, figures, candidate references

Pass 2 · assemble the document
  reading order across pages ─► join paragraphs across seams ─► stitch table segments
  ─► attach each table's intro text ─► register every table and figure

Pass 3 · resolve references, once every object is registered
  candidate references ─► object ids or conditions, checked against real table values

Pass 4 · split by content, chunk, link
  rows ─► SQL · prose ─► chunks ─► vector index · links ─► chunk metadata + a reverse link table
```

- **Why one `for page in doc.pages` loop can't do it:**
  - a reference can come before its table (a forward reference);
  - a table crosses 100 pages;
  - a sentence crosses from a native region into a scanned one.
- **Could a single loop work anyway?** Only by keeping unresolved references in a pending list and
  resolving them at the end. That's pass 3 under another name.

## 3 · Pass 1 — per page, per region

### 3.1 Split the page into regions
- The layout model returns regions with bounding boxes:
  - prose, table, picture, caption, footnote, page header and footer.

### 3.2 Choose a source for each region
Read the PDF's own data **inside the region's bounding box** (PyMuPDF or pypdfium2, milliseconds per
page):

| Signal inside the region | Verdict |
|---|---|
| Characters ≈ 0 | Scanned |
| An image covers the region and there's little text | Scanned |
| Invisible text drawn over an image | A scanner already OCR'd it; check its quality |
| `(cid:123)` glyphs, `U+FFFD`, a high share of non-words | Broken text layer, so treat it as scanned |
| Normal characters and no image under them | Native |

- Docling already OCRs only image regions above an area threshold (`bitmap_area_threshold`), unless
  you force full-page OCR (`force_full_page_ocr`). Confirm both names for your version.

### 3.3 Route by region kind × source

| Region | Native | Scanned |
|---|---|---|
| **Prose** | Read the text layer | Text OCR |
| **Table** | Table structure model on the text layer | **Table recognition on the image**, not plain text OCR |
| **Picture** | — | Classify it first: a logo, signature or stamp becomes metadata; a chart or diagram stays a figure |
| **Header, footer, page number** | Drop it; keep the page number as metadata | Same |

- **Scanned pages from many documents share one GPU queue,** so the GPU stays full.

### 3.4 Collect candidate references
- Record the raw strings and their positions only: "Table 3", "Appendix B", "the rates below",
  "class 85 and above".
- **Don't resolve them yet.** Their targets may be on a page that hasn't been processed.

## 4 · Pass 2 — assemble the document

### 4.1 Reading order
- Order the regions across columns and across pages into one stream.
- A wrong order interleaves columns into fake sentences that no splitter can repair.

### 4.2 Join paragraphs across seams
A seam is where one region ends and the next one continues the same text, such as native text
running into a scanned region.
- **Join them when:**
  - the first region ends mid-sentence (no closing punctuation, a hyphen at the end);
  - the next region starts mid-sentence (lowercase, no heading);
  - the two are adjacent in reading order.
- **Join first, chunk second.** Otherwise the splitter cuts exactly at the seam.
- Log every join with its reason, so a wrong join can be found later.

### 4.3 Stitch the table
- **Match segments on content, not coordinates:**
  - the column count;
  - the header text, fuzzily, because OCR may read "Class" as "C1ass";
  - the column types (code, number, currency).
  - Why not coordinates: a scanned page is skewed and scaled, so its x-positions won't line up with
    native pages.
- **Clean each seam:**
  - drop repeated header rows (keep one, use it to confirm the columns match);
  - join a row whose cells wrap onto the next page.
- **Check continuity at every seam:**
  - the keys keep running in order (class 50 → 500, lanes in sequence);
  - no row is missing or duplicated at the boundary.
- **If stitching fails at page *k*:** keep the parts as flagged segments. Never merge them silently.

### 4.4 Attach each table's intro text
- The prose just above a table often scopes it ("rates below apply to zone 3, effective …").
- Store it as the table's context (zone, effective date), **and** index it as prose.

### 4.5 Register every object
For every table and figure, record:
- an **id**;
- its **label** ("Table 3", "Appendix B", "Schedule A"), usually from its caption;
- its **page span**;
- for tables: the columns, which of them are keys, and the **key values that actually occur**.

## 5 · Pass 3 — resolve references

| Kind | How it's found | Exact? |
|---|---|---|
| **The PDF's own links** | Link annotations and bookmarks in born-digital PDFs | Exact, and free when present |
| **Labels in prose** | "Table 3", "Appendix B", "Exhibit 2", "see page 42", matched against the registry | Exact once the label matches |
| **Positional words** | "the rates below", "the following schedule": the nearest table in reading order | A guess, so flag it |
| **Keys mentioned in prose** | "class 85 and above", "lanes from Illinois", a service-centre code, from a rule-based or schema-constrained LLM extractor | Only after checking against the table's real values |
| **Footnote markers** | `*`, `†`, `(a)` in a cell, matched to the footnote text | Exact |

- **Reject keys that don't exist** in the table. An extractor will invent plausible lane codes.
- **A link is a condition, not a list of rows.**
  - A rule covers "all lanes from Illinois", not 3,000 row ids.
  - Store `refers_to: {table: T, where: {origin_state: IL}}`.
- **Unresolved references stay as dangling links,** each with its reason.
- **References to other documents** ("base rates per tariff XPO-CZ-2024"):
  - keep a **registry per tenant**;
  - re-resolve dangling links when that document arrives (M14).

## 6 · Pass 4 — split by content, chunk, link

### 6.1 Where each kind of content goes

| Store | Contents |
|---|---|
| **SQL** | Table rows: keys, values, `table_id`, effective date, and each row's source, confidence, page and bounding box |
| **Vector index** | Prose chunks, plus **one summary node per table** (what it covers, its columns, key ranges and page span), so a question can be routed to the table |
| **Link table** | Object → chunks, and footnote marker → footnote text |

### 6.2 Chunk, then attach links
- **Links live on spans before chunking.** Each chunk inherits the links of the spans it contains.
- **Each chunk records:**
  - its **page span** and **a bounding box per span**, since one joined paragraph can cover two pages
    and two sources;
  - whether each span is native or OCR, and its confidence.
- **Keep links out of the embedded text:** add `refers_to` to `excluded_embed_metadata_keys` (and
  usually `excluded_llm_metadata_keys`).
- **Why the reverse link table:** a chunk's metadata only points outward. "Every rule that applies to
  table T" needs the reverse direction.

### 6.3 Account for every region
- Every region ends up in exactly one place: prose, a table, a figure, metadata, or dropped page
  furniture.
- Any region left over means the document failed.

## 7 · Checking tables

### 7.1 Warning signs, before extraction
These decide **which extra checks run**, not whether to re-extract:
- no ruling lines (a borderless table);
- headers spanning rows or columns, or merged cells;
- rotated or nested tables;
- scanned segments;
- cells holding several lines of text.

**A table crossing a page break is not a warning sign.** It triggers stitching (4.3). Every engine
reads one page at a time, so re-extracting the table doesn't fix it.

### 7.2 Validation, after stitching

| Check | Applies to | A failure looks like |
|---|---|---|
| **Ragged rows** | All segments, **after seam rows are joined** | The column count changes from row to row |
| **Text coverage** | Native segments | The PDF's characters inside the table region ≠ the characters in the cells |
| **OCR confidence** | Scanned segments (they have no text layer to compare) | Low-confidence cells, especially digits |
| **Type check** | All | A numeric column won't parse as numbers |
| **Monotonicity** | Rate tables | Rates should rise with class within a lane and fall as weight breaks get heavier |
| **Continuity** | Every seam | A missing, duplicated or out-of-order key |
| **Empty-cell ratio** | All | Far more blanks than expected (some blanks are genuine, like a lane with no service) |

- Use **generic checks** plus **rules each tenant can configure.** A multi-customer engine can't hard-code
  one industry's domain rules.
- **Watch for digit swaps:** 0 and 8, 1 and 7, 5 and 6, and dropped decimal points.

### 7.3 When a row fails
1. **Re-read only the suspect rows** with a second engine (MinerU or a vision model), into the same
   cell-grid format.
   - Measure agreement between the two readings with a cell-level metric such as GriTS or TEDS.
   - Running two extractors over all 100 pages would double the cost for nothing.
2. **Still failing → quarantine the row.**
   - Serve the rows that pass.
   - **A flagged row is never returned as an exact rate.**
3. **Human review per row or segment, not per table,** within the compliance limits in section 10.

## 8 · Why the links exist: what happens at query time

**Question:** "What does 3,200 lb of class 85 from Chicago Heights to Dallas cost under our
agreement?"

| Step | What happens |
|---|---|
| 1 · **Retrieve prose** | Finds "72% off tariff T's base rates", "FAK class 70 for classes 50–85", "minimum charge $X" |
| 2 · **Apply the rules** | **The prose changes the key:** class 85 is rated as **70** |
| 3 · **Look up the rows** | `lookup_rate(tariff_id, origin, dest, class=70, weight=3200)`, plus the next weight break up |
| 4 · **Calculate** | A tool applies the discount, deficit-weight rating and the minimum charge. The LLM does no arithmetic. |
| 5 · **Answer with citations** | Cites the prose clauses **and** the table rows (page and bounding box), M12 |

- **The order is the point.** SQL run on the raw question fetches the class-85 row: a real number,
  correctly read, and the wrong answer.
- **Pull in every rule whenever the table is hit,** however each one ranks. When an answer touches
  table T, the retrieval tool attaches every prose chunk linked to T.
- **Two kinds of table tool:**
  - a **typed lookup** (fixed parameters, tenant filter always enforced) for rate questions;
  - free **text-to-SQL** only for analysis across the table ("which Chicago lanes have class-70 rates
    above $Y?").
- **Guardrails on the answer:**
  - every dollar figure must cite a row returned by the lookup tool;
  - the lookup's keys must reflect any rule retrieved (FAK, lane exceptions);
  - the rules and the rows must share one agreement version and effective date.
- **Workflow or agent:**
  - known question shapes ("quote this shipment") run a **fixed workflow** of the steps above;
  - open-ended questions ("why did this invoice differ from the quote?") go to an **agent** with
    the same tools;
  - a router chooses between them (M10). The agent side is M16.

## 9 · Scale

Assumes about 500 words per page and every document a PDF, so these are upper bounds.

| Tool | Pilot, ≈ 27k pages | Year one, ≈ 4.3M pages |
|---|---|---|
| Docling, ~1–3 s per page on a CPU without OCR | Under 1 h on 16 cores, ≈ $1 | ≈ 2,400 CPU-hours, ≈ 37 h on 64 vCPUs, ≈ $100 |
| olmOCR, ~$190 per million pages | ≈ 2 GPU-hours, ≈ $5 | ≈ 310 H100-hours, ≈ $820 |
| MinerU, ~$596 per million pages (olmOCR paper's comparison) | ≈ $16 | ≈ $2.6k |

- **Conversion can cost more than embedding.** At year one it's about $72 to embed, against about
  $820 to convert everything with olmOCR.
- **The cost depends on the mix of regions, not the page count.**
  - Most regions stay on Docling.
  - A tariff with many scanned table pages sends far more regions to the GPU.
  - **Measure the mix of region kinds on real documents** before trusting any total.
- **Triage** costs about 10 ms per page: about 12 CPU-hours for year one.
- **Caching:**
  - cache **each page's or region's output**, keyed by **tenant + content hash + engine version**;
  - **never cache an assembled document as final.** Assembly depends on neighbouring pages, so re-run
    passes 2–4 whenever any input or engine changes.

## 10 · What production needs on top

| Area | What production needs |
|---|---|
| **Evaluation** | A labelled set of regions per document type. Measure mis-routes in both directions, wrong seam joins, stitching failures and wrong reference resolutions, plus what each error costs. Until then, every threshold here is a guess. |
| **Hostile input** | Customer PDFs can be malformed, decompression bombs, carry JavaScript or embedded files, be encrypted, or run to 10,000 pages. Parse in a sandbox with time, memory and page limits per job, and a quarantine queue. |
| **Text layers can lie** | Invisible text that doesn't match the image is a prompt-injection route (M21). Check the text layer against OCR **only where it's suspicious** (invisible text, text over an image) or on a random sample. Checking every region would make OCR cost explode. |
| **Caches per tenant** | A cache shared across tenants lets a cache hit reveal that another tenant holds the same document. Scope every cache to the tenant (M17). |
| **Fairness between tenants** | One tenant's huge upload starves the GPU queue. Use quotas per tenant, fair scheduling, and separate interactive and bulk lanes. |
| **Two levels of jobs** | Page and region jobs: idempotent, retried, run in parallel. One assembly job per document, which waits for all its pages and has a timeout. |
| **Whole-document state** | A document is **complete, or flagged as a whole.** Indexing pages one by one would break stitching and links. |
| **Versions per region and row** | A stitched table can mix rows from several engines. Record engine and model versions per region and row. An upgrade means a planned backfill that re-runs assembly for each affected document. |
| **Human review** | Staff reading customer contracts needs consent, access control and an audit trail. The same applies to external vision-model APIs, so models are often self-hosted. |
| **Deletion and residency** | A deletion request must reach the original, region outputs, caches, SQL rows, vectors, link tables and the review queue. Regional documents stay in their region. |
| **Forms and signatures** | Signatures, initials, checkboxes, form fields, handwriting. When the platform generated the document, use its own data instead of re-parsing the PDF. |
| **Service levels and observability** | Latency per page and per document, queue depth, cost per tenant, the mix of region kinds over time, and alerts on quality drift (M19). |
| **The numbers** | Measure throughput and cost per region on real documents. Section 9's figures come from vendor reports. |

## 11 · As an interview answer

- **Mid-level:** cheap triage first, escalate only what needs it, validate tables after extraction.
- **Senior:** adds regions instead of pages, document-level assembly, reference resolution and
  links, and row-level quarantine. In other words, sections 2–8.
- **Staff:** adds section 10 unprompted: hostile input, tenant isolation including caches, measured
  evaluation, two levels of jobs, versioning and backfills, and compliance limits on review and
  external APIs.
- **Two kinds of failure, both serious:**
  - a wrong number in a rate is a billing error;
  - a leak between tenants or a compliance breach can end the customer relationship.

## Sources

- olmOCR: <https://arxiv.org/html/2502.18443>
- MinerU2.5: <https://arxiv.org/pdf/2509.22186>
- Docling layout models: <https://arxiv.org/pdf/2509.11720>
- Docling documentation: <https://docling-project.github.io/docling/>
- READoc benchmark: <https://arxiv.org/pdf/2409.05137>
- PDF-extraction overview and benchmarks: <https://github.com/dantetemplar/pdf-extraction-agenda>
