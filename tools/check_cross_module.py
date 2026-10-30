#!/usr/bin/env python3
"""
Course-wide cross-module checks (syllabus §9 step 3 and §11).

1. KEY FORMAT: every bibliography uses the frozen gold-standard shape --
   `- <title-or-link> — <provenance> — `[KEY]` · owns <claim>.`
2. OWNERSHIP: a reserved key owned by module A must not be RE-SURVEYED in module B.
   Heuristic: if a non-owner module contains a full bibliographic entry (author list +
   arXiv id) for a key it does not own, that is a re-survey.
3. CONTRADICTION: scan "Where the literature disagrees" sections for cross-module
   claims about the same subject so a human can check they do not conflict.
"""

from __future__ import annotations
import re, pathlib, collections

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULES = sorted(ROOT.glob("modules/*/README.md"))

# Owner map from syllabus §6 Thread D (key -> owning module number)
OWNERS = {
    "RAG20": 1, "LONGCONTEXT23": 1, "INPUTLEN24": 1,
    "DPR20": 4, "BEIR21": 4, "SENTBERT19": 4, "MTEB22": 4, "MRL22": 4,
    "HNSW16": 5, "PQ11": 5,
    "BM25": 7, "ROB94": 7, "SPLADE21": 7, "RRF09": 7,
    "COLBERT20": 9, "CE19": 9, "LITM23": 9, "MMR98": 9,
    "HYDE22": 10, "RAGFUS23": 10, "SELFASK22": 10, "STEPBACK23": 10,
    "RAGAS23": 13, "ARES23": 13, "JUDGE23": 13, "RAGNAROK24": 13,
    "SELFRAG23": 16, "CRAG24": 16, "GRAPHRAG24": 16, "MODRAG24": 16,
    "INJECT23": 21, "POISON23": 21,
}

def module_num(p: pathlib.Path) -> int:
    return int(p.parent.name.split("-")[0])


def bib_entries(body: str):
    """Yield each bibliography entry as one joined string (bullet + continuation lines)."""
    cur = None
    for line in body.splitlines():
        if line.startswith("- "):
            if cur is not None:
                yield cur
            cur = line
        elif cur is not None and (line.startswith("  ") or not line.strip()):
            cur += " " + line.strip()
        elif cur is not None:
            yield cur
            cur = None
    if cur is not None:
        yield cur

fail, warn, info = [], [], []

# ---------- 1. key format ----------
ENTRY = re.compile(r"^- .*?—\s*`\[[A-Za-z0-9\-]+\]`\s*·\s*owns\s")
for p in MODULES:
    text = p.read_text(encoding="utf-8")
    if "## Sources" not in text:
        fail.append(f"{p.parent.name}: no ## Sources")
        continue
    body = text.split("## Sources", 1)[1]
    # bibliography ends at the first non-bullet, non-continuation block (usually the
    # Unsupported / derived-notes prose), which is not part of the key system
    cut = len(body)
    seen_bullet = False
    pos = 0
    for line in body.splitlines(keepends=True):
        pos += len(line)
        if line.startswith("- "):
            seen_bullet = True
        elif seen_bullet and line.strip() and not line.startswith("  "):
            cut = pos - len(line)
            break
    body = body[:cut]
    # collect bibliography bullets = consecutive "- " lines at the start of the block
    for b in bib_entries(body):
        has_key = re.search(r"`\[[A-Za-z0-9\-]+\]`", b)
        has_owns = "· owns" in b
        # Allowed variants in the frozen format:
        #  (a) keyed entry with "· owns ..."
        #  (b) stack-doc / tooling bullet carrying `[KEY]` without an ownership clause
        #  (c) cross-reference entry that names the owning module
        is_stackdoc = bool(re.search(r"`\[[A-Za-z0-9\-]+\]`", b)) and (
            "developers.llamaindex.ai" in b or "docs." in b or "github.com" in b
            or "pricing" in b.lower() or ".ai/" in b or ".com/" in b or ".io/" in b)
        is_xref = bool(re.search(r"owns this (claim|finding|survey)|owned by \*\*M|Owned by \*\*M|M\d+ owns", b))
        if not (has_key or is_stackdoc or is_xref):
            fail.append(f"{p.parent.name}: entry matches no permitted format -> {b[:78]}")
        elif has_key and not has_owns and not is_xref and not is_stackdoc:
            fail.append(f"{p.parent.name}: keyed entry missing '· owns ...' -> {b[:78]}")
        if "http" not in b:
            warn.append(f"{p.parent.name}: entry has no link -> {b[:70]}")

# ---------- 2. re-survey detection ----------
for p in MODULES:
    n = module_num(p)
    text = p.read_text(encoding="utf-8")
    body = text.split("## Sources", 1)[1] if "## Sources" in text else ""
    for key, owner in OWNERS.items():
        if owner == n:
            continue
        for line in bib_entries(body):
            if f"`[{key}]`" not in line:
                continue
            # a full entry for a key you do not own, with no ownership marker = re-survey
            if (re.search(r"arXiv:\d{4}\.\d{4,5}", line)
                    and not re.search(r"owns this (claim|finding|survey)|owned by \*\*M|Owned by \*\*M|M\d+ owns", line)
                    and "owns the survey" not in line
                    and "cross-reference" not in line):
                fail.append(
                    f"{p.parent.name}: RE-SURVEYS [{key}] owned by M{owner} "
                    f"(full entry with arXiv id, no ownership marker)"
                )
            break

# ---------- 3. contradiction cross-check ----------
topics = {
    "long context": r"long[- ]context",
    "semantic chunking": r"semantic chunking",
    "hybrid search": r"hybrid (search|retrieval)",
    "reranking": r"rerank",
    "temperature": r"temperature",
    "fine-tuning": r"fine[- ]tun",
    "graph": r"\bGraphRAG\b|knowledge graph|property graph",
    "LLM-as-judge": r"LLM[- ]as[- ]a?[- ]judge|LLM judge",
    "injection": r"prompt injection",
}
sections = collections.defaultdict(list)
for p in MODULES:
    t = p.read_text(encoding="utf-8")
    m = re.search(r"## Where the literature disagrees(.*?)(?=\n## |\Z)", t, re.S)
    if m:
        sections[p.parent.name] = m.group(1)
for topic, pat in topics.items():
    hits = [name for name, sec in sections.items() if re.search(pat, sec, re.I)]
    if len(hits) > 1:
        info.append(f"topic '{topic}' contested in: {', '.join(sorted(hits))}")

print(f"modules                : {len(MODULES)}")
print(f"modules w/ disagreements: {len(sections)}")
print(f"hard failures          : {len(fail)}")
print(f"warnings               : {len(warn)}")
if fail:
    print("\n--- FAILURES ---")
    for f in fail: print("  ✗", f)
if warn:
    print("\n--- WARNINGS ---")
    for w in warn[:20]: print("  !", w)
if info:
    print("\n--- CROSS-MODULE TOPICS TO EYEBALL (not errors) ---")
    for i in info: print("  ·", i)
