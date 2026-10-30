#!/usr/bin/env python3
"""
Course-tree integrity checker (syllabus §11 "verify the whole tree").

Checks, per module README:
  1. Required anatomy blocks present, in order.
  2. Every relative markdown link resolves to a real file.
  3. Citation graph is one-to-one: every inline `[KEY]` has a Sources entry and vice versa.
  4. Verification note present and states a status.
  5. Failure-mode IDs referenced are within the defined taxonomy (F1-F10).
Also: syllabus status table vs reality, and orphan-file detection.

Exit code 1 if any hard check fails.
"""

from __future__ import annotations
import re, sys, pathlib, collections

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULES = sorted(ROOT.glob("modules/*/README.md"))

REQUIRED = [
    ("# M", re.compile(r"^# M\d+ · ", re.M)),
    ("Module question header", re.compile(r"\*\*Module question:\*\*")),
    ("Cross-cutting threads", re.compile(r"\*\*Cross-cutting threads:\*\*")),
    ("Domain spine", re.compile(r"\*\*Domain spine:\*\*")),
    ("Prerequisites", re.compile(r"\*\*Prerequisites:\*\*")),
    ("Opening scene", re.compile(r"^## Opening scene", re.M)),
    ("Failure mode line", re.compile(r"\*\*Failure mode \(the module in one line\):\*\*")),
    ("LlamaIndex at a glance", re.compile(r"\*\*LlamaIndex at a glance:\*\*")),
    ("Tradeoff ledger entry", re.compile(r"Tradeoff \(the ledger entry M\d+-\d+\)")),
    ("Worked example", re.compile(r"^## Worked example", re.M)),
    ("Design exercise", re.compile(r"^## Design exercise", re.M)),
    ("Sources", re.compile(r"^## Sources", re.M)),
    ("Verification note", re.compile(r"\*\*Verification note:\*\*")),
    ("Next module", re.compile(r"\*\*Next module:\*\*|^\*\*Course complete", re.M)),
]

fail: list[str] = []
warn: list[str] = []


def check_anatomy(path: pathlib.Path, text: str) -> None:
    for name, pat in REQUIRED:
        if not pat.search(text):
            fail.append(f"{path.parent.name}: MISSING anatomy block -> {name}")
    # ordering of the big sections
    order = ["## Opening scene", "## Worked example", "## Design exercise", "## Sources"]
    pos = [text.find(h) for h in order]
    present = [(h, p) for h, p in zip(order, pos) if p >= 0]
    for (h1, p1), (h2, p2) in zip(present, present[1:]):
        if p1 > p2:
            fail.append(f"{path.parent.name}: section out of order: {h1} after {h2}")


def check_links(path: pathlib.Path, text: str) -> int:
    n = 0
    for m in re.finditer(r"\[([^\]]+)\]\((\.\.?/[^)#\s]+|/Users/[^)#\s]+)", text):
        target = m.group(2)
        if target.startswith("/Users"):
            t = pathlib.Path(target)
        else:
            t = (path.parent / target).resolve()
        if not t.exists():
            fail.append(f"{path.parent.name}: DEAD LINK '{m.group(1)}' -> {target}")
        else:
            n += 1
    return n


def check_citations(path: pathlib.Path, text: str) -> None:
    """One-to-one citation graph.

    The bibliography is the FIRST contiguous run of "- " bullets after "## Sources".
    Everything else -- the Unsupported prose AND any trailing blockquotes (verification
    note, next-module link) -- is body and may legitimately cite keys.
    """
    lines = text.splitlines()
    sidx = next((i for i, l in enumerate(lines) if l.startswith("## Sources")), None)
    if sidx is None:
        fail.append(f"{path.parent.name}: no ## Sources heading")
        return

    bib_idx = set()
    started = False
    for i in range(sidx + 1, len(lines)):
        l = lines[i]
        if l.startswith("- "):
            started = True
            bib_idx.add(i)
        elif l.startswith("  ") and started:
            bib_idx.add(i)                    # continuation of the previous bullet
        elif not l.strip():
            continue                          # blank lines inside/after the block
        elif started:
            break                             # first non-bullet block: bibliography over

    tail = "\n".join(lines[i] for i in sorted(bib_idx))
    body = "\n".join(l for i, l in enumerate(lines) if i not in bib_idx)

    KEY = r"(?:[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)"
    inline = set(re.findall(r"\[(" + KEY + r")\](?![(\w])", body))
    entries = set(re.findall(r"`\[([A-Za-z0-9\-]+)\]`", tail))

    for k in sorted(inline - entries):
        fail.append(f"{path.parent.name}: CITATION with no entry -> [{k}]")
    for k in sorted(entries - inline):
        warn.append(f"{path.parent.name}: Sources entry not cited inline -> [{k}]")


def check_failure_ids(text: str, name: str) -> None:
    ids = set(re.findall(r"\bF(\d+)\b", text))
    for i in sorted(ids, key=int):
        if int(i) > 10 or int(i) < 1:
            warn.append(f"{name}: failure id F{i} outside defined taxonomy F1-F10")


def check_verification_note(text: str, name: str) -> None:
    m = re.search(r"\*\*Verification note:\*\*(.{0,200})", text, re.S)
    if not m:
        return
    note = m.group(1)
    if not re.search(r"✅|🟡|⬜|pending|in progress|round", note):
        warn.append(f"{name}: verification note states no status")


def main() -> int:
    total_links = 0
    for p in MODULES:
        t = p.read_text(encoding="utf-8")
        check_anatomy(p, t)
        total_links += check_links(p, t)
        check_citations(p, t)
        check_failure_ids(t, p.parent.name)
        check_verification_note(t, p.parent.name)

    # status table truth
    syl = (ROOT / "syllabus.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (M\d+) \| (.+?) \| (⬜ planned|🟡 written|✅ verified) \|.*\|$", syl, re.M)
    declared = {m: st for m, _, st in rows}
    if len(rows) != 22:
        fail.append(f"syllabus: expected 22 status rows, found {len(rows)}")
    for i, p in enumerate(MODULES, start=1):
        mid = f"M{i}"
        if mid not in declared:
            fail.append(f"syllabus: no status row for {mid}")

    # orphans: any .md under modules/ that isn't a README.md, plus stray files
    for f in ROOT.glob("modules/**/*"):
        if f.is_file() and f.name != "README.md" and "lab" not in f.parts:
            warn.append(f"orphan file in modules tree: {f.relative_to(ROOT)}")

    print(f"modules scanned      : {len(MODULES)}")
    print(f"resolving links      : {total_links}")
    print(f"hard failures        : {len(fail)}")
    print(f"warnings             : {len(warn)}")
    if fail:
        print("\n--- FAILURES ---")
        for f in fail:
            print("  ✗", f)
    if warn:
        print("\n--- WARNINGS ---")
        for w in warn:
            print("  !", w)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
