# Hierarchy and window beyond prose

> $\color{red}\text{to be discussed}$ — an idea and some test findings, not yet built.
> Back to [M3 parsers](README.md#the-parsers-and-when-each-one-earns-its-place) · test plan in [testing-sentence-window-vs-hierarchical.md](testing-sentence-window-vs-hierarchical.md) · auto-merging internals in [M8](../08-index-structures-beyond-flat/README.md#structure-at-index-time-recursive-retrieval-and-auto-merging).

## The idea

Hierarchical and sentence window are really **expansion policies**, not parsers. They can be applied to any parser's output. That splits the problem into two separate choices:

- **Where boundaries go:** decided by the parser (sentence, markdown heading, HTML tag, JSON path, syntax tree).
- **How you widen at query time:** decided by the policy.
  - **Hierarchical:** climb to the parent when enough siblings match.
  - **Window:** pull in the N units before and after the hit.

`AutoMergingRetriever` only reads the PARENT/CHILD links in the docstore. It doesn't care how the tree was built.

## Hierarchical, applied per format

| Format | Leaf | Mid level | Top level |
|---|---|---|---|
| Markdown | A sentence or bullet | The full content of an h3 | The full content of an h2 or h1 |
| Table | A **row with its column names** (not a cell) | — | The whole table |
| HTML | A `p` or `li` | A `section` / heading block | The page or top-level section |
| JSON | A scalar `path: value` | An object | An array element or record |
| Code | A function or method | A class | A module |

## Window, applied per format

| Format | Unit | Window = N before and after |
|---|---|---|
| Markdown | A bullet or sentence | The neighbouring bullets or sentences |
| Table | A row | The N rows before and after, **plus the header row** |
| HTML | An element (`p`, `li`, `tr`) | The neighbouring elements |
| Code | A line | The lines before and after |
| JSON | A key | The other keys of the same object |

## What LlamaIndex actually does today (observed on llama-index-core 0.14.25)

- **`HierarchicalNodeParser` accepts any parser for each level** through `node_parser_map`.
  - The test put markdown sections on top and `SentenceSplitter(64)` leaves underneath.
  - **The heading tree is lost.** `### B1` and `### B2` came out as siblings of `## B`, not its children.
  - Each level just re-splits the level above, so you get "section → sentences" in two levels, not h1 → h2 → h3.
- **`MarkdownNodeParser` doesn't link sections to each other.**
  - Each node holds only its own heading's text. The `## B` node contains "b text", not B1 and B2.
  - So "the full content of an h2" isn't a node anywhere.
  - Links are SOURCE, PREVIOUS and NEXT only. There's no PARENT or CHILD.
  - PREVIOUS and NEXT do exist, so a **section-level window** is possible today.
- **`JSONNodeParser` flattens; it doesn't build a hierarchy.** Two rate rows came out as **one node**:
  ```
  rates port Rotterdam
  rates size 20ft
  rates usd 42
  rates port Singapore
  ...
  ```
  - The array index is dropped, so you can't tell which row "42" belongs to.
- **Code:** `CodeHierarchyNodeParser` lives in the separate `llama-index-packs-code-hierarchy` package. It is not tested here; this is from memory.
  - It builds module → class → function nodes with parent and child links.
  - It replaces each child's body in the parent with a pointer comment, so code isn't stored twice.
- **Elsewhere (from memory, not verified):**
  - Haystack's `HierarchicalDocumentSplitter` is size-based.
  - Docling's `HierarchicalChunker` keeps the heading path as metadata, but doesn't link nodes for merging.

## Format-specific traps

- **Markdown:** text before the first h3 inside an h2 has to belong to the h2 node itself, or it ends up with no parent.
- **Tables:**
  - A cell alone is a bad leaf: "42" embeds as noise.
  - A row-window without the header row loses the column meaning.
- **HTML:** cosmetic nesting (`div` wrappers, layout tags) makes a deep tree with no meaning. Only meaningful tags (`section`, `h*`, `table`, `li`) should become levels.
- **JSON:**
  - Keep the index in each leaf's path (`rates[0].usd: 42`) so the row identity survives.
  - Often you shouldn't chunk records at all. Metadata filters or a structured query beat embeddings for them.
- **Code:**
  - Single lines are poor *retrieval* units: `return x + offset` embeds as nothing (the same as case G in the test plan).
  - A line window helps *generation* only once the hit is found, so embed at function level and expand by lines.
  - Neighbours by reference (callers, callees, imports) are often worth more than adjacent lines. That's a graph index (→ M8, M10).

## Traps a structural tree has that a size-based tree doesn't

- **Uneven fan-out breaks the fixed ratio.** Size-based trees have about 5 children per parent, so "more than 0.5" means something consistent there. Structural trees don't:
  - A section with **1** child merges on every hit.
  - A section with **20** children almost never merges.
  - Use a per-parent rule (for example, at least k children, or a ratio scaled by fan-out), or cap the fan-out.
- **The top level has no size limit.** An h1 or a module can be 30k tokens, and merging into it blows the context budget.
  - You still need size caps: structure picks the boundaries, a token limit caps them.
  - That is the "hybrid" idea, which as I recall Docling's `HybridChunker` also uses.
- **Leaves can be too small.** Bullets, rows and lines embed poorly by themselves.
  - Prepend the ancestor path to the leaf's embedded text: `Maersk › 14. Demurrage › 14.2 Liability › …`.
  - That is the same fix as the `header_path` finding in the M3 `MarkdownNodeParser` collapsible.
- **Parents store their children's text again.** That costs storage at every level. The code-hierarchy pointer trick above avoids it.

## Traps a window has outside prose

- **A window crosses structural boundaries.** Rows from the next table, bullets from the next list, or lines from the next function all add noise (case F in the test plan).
  - Clip the window at the enclosing structure: same list, same table, same function.
- **N is in units, and units vary in size.** Three rows, three bullets and three lines are very different token counts.
  - Cap the window in tokens, not only in units.

## Next step

- Build the ~30-line markdown tree with real PARENT and CHILD links, run `AutoMergingRetriever` on it, and measure the fan-out problem with numbers.
- Add structure-aware variants to the test plan: a structural hierarchy, and a window clipped at structure.
