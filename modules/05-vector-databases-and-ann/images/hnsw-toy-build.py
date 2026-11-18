"""Render hnsw-toy-build.gif: a slow, step-by-step HNSW build over eight 1-D vectors, m = 2.

Regenerate with (matplotlib + pillow needed):
    python hnsw-toy-build.py [--out hnsw-toy-build.gif] [--frames-dir some/dir]

--frames-dir additionally writes every frame as PNG (frame-00.png ...) for inspection.
The insertion steps are hard-coded to match the module README's worked example.
"""

import argparse
import io
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from PIL import Image  # noqa: E402

# ----------------------------------------------------------------------------- data
VALUE = {"A": 1, "B": 2, "C": 4, "D": 7, "E": 8, "F": 11, "G": 14, "H": 15}
TOP_LAYER = {"A": 0, "B": 0, "C": 1, "D": 0, "E": 2, "F": 0, "G": 1, "H": 0}

# One entry per insertion. edges: list of (u, v, layer). walk: visited (node, layer) in order.
# jump: optional (from, to, layer) long move to highlight. dist: (node, layer, label) hints on the walk frame.
# rejected: (node, layer, label) candidates the selection heuristic drops, shown crossed out on the link frame.
# Ground truth: hnswlib, M=2, L2, with the levels pinned to TOP_LAYER. Its neighbour-selection rule keeps the
# nearest candidate and keeps a further one only if it is closer to the new vector than to any kept neighbour;
# in 1-D that rejects every "skip-one" edge, so layer 0 ends up a plain chain.
STEPS = [
    dict(
        node="A",
        appear="Step 1 · insert A (value 1), top layer 0 — the graph is empty",
        walk=[],
        edges=[],
        linked="A sits alone on layer 0 → A becomes the entry point · no edges to add",
    ),
    dict(
        node="B",
        appear="Step 2 · insert B (value 2), top layer 0",
        walk=[("A", 0)],
        dist=[("A", 0, "d=1")],
        walk_caption="walk: start at entry point A (L0) · candidates A(1)",
        edges=[("B", "A", 0)],
        linked="candidates A(1) → keep A; nothing to reject → link B–A on layer 0",
    ),
    dict(
        node="C",
        appear="Step 3 · insert C (value 4), top layer 1 → higher than entry point A → C becomes entry point",
        walk=[("A", 0)],
        dist=[("B", 0, "d=2"), ("A", 0, "d=3")],
        walk_caption="L1: C is alone, nothing to link · L0: walk from A · candidates B(2), A(3)",
        edges=[("C", "B", 0)],
        rejected=[("A", 0, "behind B")],
        linked="candidates B(2), A(3) → keep B; A is closer to B (1) than to C (3): behind B, rejected → link C–B on layer 0",
    ),
    dict(
        node="D",
        appear="Step 4 · insert D (value 7), top layer 0",
        walk=[("C", 1), ("C", 0)],
        dist=[("C", 0, "d=3"), ("B", 0, "d=5"), ("A", 0, "d=6")],
        walk_caption="walk: start at C(L1), no neighbours there → slide down C's pole to layer 0 · candidates C(3), B(5), A(6)",
        edges=[("D", "C", 0)],
        rejected=[("B", 0, "behind C"), ("A", 0, None)],
        linked="candidates C(3), B(5), A(6) → keep C; B is closer to C (2) than to D (5): behind C, rejected (A likewise) → link D–C on layer 0",
    ),
    dict(
        node="E",
        appear="Step 5 · insert E (value 8), top layer 2 → new highest layer → E becomes entry point",
        walk=[("C", 1), ("C", 0)],
        dist=[("C", 1, "d=4"), ("D", 0, "d=1"), ("C", 0, "d=4"), ("B", 0, "d=6"), ("A", 0, "d=7")],
        walk_caption="L2: E alone · L1: only C is there, C(4) · slide down C's pole to layer 0 · candidates D(1), C(4), B(6), A(7)",
        edges=[("E", "C", 1), ("E", "D", 0)],
        rejected=[("C", 0, "behind D")],
        linked="L1: link E–C · L0: candidates D(1), C(4), … → keep D; C is closer to D (3) than to E (4): behind D, rejected → link E–D on layer 0",
    ),
    dict(
        node="F",
        appear="Step 6 · insert F (value 11), top layer 0",
        walk=[("E", 2), ("E", 1), ("E", 0)],
        dist=[("E", 1, "d=3"), ("C", 1, "d=7"), ("E", 0, "d=3"), ("D", 0, "d=4")],
        walk_caption="walk: start at E(L2), slide down E's pole to L1: d(F,C)=7 > d(F,E)=3, stay at E; slide down E's pole to L0 · candidates E(3), D(4)",
        edges=[("F", "E", 0)],
        rejected=[("D", 0, "behind E")],
        linked="candidates E(3), D(4) → keep E; D is closer to E (1) than to F (4): behind E, rejected → link F–E on layer 0",
    ),
    dict(
        node="G",
        appear="Step 7 · insert G (value 14), top layer 1",
        walk=[("E", 2), ("E", 1), ("E", 0)],
        dist=[("E", 1, "d=6"), ("C", 1, "d=10"), ("F", 0, "d=3"), ("E", 0, "d=6")],
        walk_caption="walk: start at E(L2), slide down E's pole to L1 · L1 candidates E(6), C(10) · slide down E's pole to L0 · L0 candidates F(3), E(6)",
        edges=[("G", "E", 1), ("G", "F", 0)],
        rejected=[("C", 1, "behind E"), ("E", 0, "behind F")],
        linked="L1: keep E; C is closer to E (4) than to G (10): rejected → link G–E · L0: keep F; E is closer to F (3) than to G (6): rejected → link G–F",
    ),
    dict(
        node="H",
        appear="Step 8 · insert H (value 15), top layer 0",
        walk=[("E", 2), ("E", 1), ("G", 1), ("G", 0)],
        jump=("E", "G", 1),
        dist=[("E", 1, "d=7"), ("G", 1, "d=1"), ("G", 0, "d=1"), ("F", 0, "d=4")],
        walk_caption="walk: E(L2), slide down E's pole to L1: d(H,G)=1 < d(H,E)=7 → hop across to G (the one long jump); slide down G's pole to L0 · candidates G(1), F(4)",
        edges=[("H", "G", 0)],
        rejected=[("F", 0, "behind G")],
        linked="candidates G(1), F(4) → keep G; F is closer to G (3) than to H (4): behind G, rejected → link H–G on layer 0",
    ),
]

FINAL_CAPTION = "Entry point E. Layer 1 = {C, E, G}, Layer 2 = {E}. Layer 0: a chain of 7 edges; m = 2 was never reached."

EXPECTED_L0 = {frozenset(e) for e in ["AB", "BC", "CD", "DE", "EF", "FG", "GH"]}
EXPECTED_L1 = {frozenset("CE"), frozenset("EG")}

# ----------------------------------------------------------------------------- style
W_IN, H_IN, DPI = 10.0, 6.5, 100
LAYER_Y = {0: 0.0, 1: 1.15, 2: 2.3}
BAND_HALF = 0.43
ARC_H = 0.27  # peak height of an edge that skips over a node
NODE_R_PT = 11  # node radius in points

C_BAND = "#f8f9fa"
C_BAND_EDGE = "#dee2e6"
C_NODE = "#ffffff"
C_NODE_EDGE = "#343a40"
C_NEW = "#339af0"
C_ENTRY = "#e8590c"
C_EDGE = "#868e96"
C_EDGE_NEW = "#1c7ed6"
C_WALK = "#1c7ed6"
C_JUMP = "#c2255c"
C_COPY = "#adb5bd"
C_REJECT = "#868e96"
C_REJECT_FILL = "#e9ecef"
C_TEXT = "#212529"
C_MUTED = "#868e96"

plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


# ----------------------------------------------------------------------------- drawing
def edge_path(x1, x2, y, skips_node):
    if not skips_node:
        return MplPath([(x1, y), (x2, y)], [MplPath.MOVETO, MplPath.LINETO])
    xm = (x1 + x2) / 2
    # quadratic Bézier peaks at half the control-point offset
    return MplPath([(x1, y), (xm, y + 2 * ARC_H), (x2, y)], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])


def skips_over(u, v, layer, present):
    lo, hi = sorted((VALUE[u], VALUE[v]))
    return any(lo < VALUE[n] < hi for n in present if TOP_LAYER[n] >= layer)


def draw_frame(present, edges, entry, *, new=None, new_edges=(), walk=(), jump=None, dist=(), rejected=(), caption="", title=None):
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.01, 0.19, 0.98, 0.73])
    ax.set_xlim(-2.4, 16.9)
    ax.set_ylim(-0.95, LAYER_Y[2] + BAND_HALF + 0.12)
    ax.axis("off")

    # bands
    for layer, y in LAYER_Y.items():
        ax.add_patch(Rectangle((-0.6, y - BAND_HALF), 17.3, 2 * BAND_HALF, facecolor=C_BAND, edgecolor=C_BAND_EDGE, lw=1, zorder=0))
        ax.text(-2.2, y, f"Layer {layer}", ha="left", va="center", fontsize=12, color=C_TEXT, fontweight="bold")
    ax.text(-0.4, LAYER_Y[0] + BAND_HALF - 0.06, "Layer 0 — all 8 vectors; in 1-D the selection rule leaves a chain", ha="left", va="top", fontsize=8.5, color=C_MUTED, style="italic", zorder=1)

    # ruler under layer 0
    ry = -BAND_HALF - 0.22
    ax.plot([0, 16], [ry, ry], color=C_MUTED, lw=1, zorder=1)
    for v in range(17):
        ax.plot([v, v], [ry, ry + (0.07 if v % 2 == 0 else 0.04)], color=C_MUTED, lw=1)
        if v % 2 == 0:
            ax.text(v, ry - 0.08, str(v), ha="center", va="top", fontsize=8, color=C_MUTED)
    ax.text(16.6, ry - 0.02, "value", ha="left", va="center", fontsize=8, color=C_MUTED, style="italic")

    # dotted copy lines joining copies of the same vector across layers
    for n in present:
        if TOP_LAYER[n] > 0:
            ax.plot([VALUE[n]] * 2, [LAYER_Y[0], LAYER_Y[TOP_LAYER[n]]], ls=":", lw=1.1, color=C_COPY, zorder=1)

    # edges
    new_set = {(frozenset((u, v)), l) for u, v, l in new_edges}
    for u, v, l in edges:
        is_new = (frozenset((u, v)), l) in new_set
        p = edge_path(VALUE[u], VALUE[v], LAYER_Y[l], skips_over(u, v, l, present))
        ax.add_patch(PathPatch(p, fill=False, lw=3 if is_new else 1.6, edgecolor=C_EDGE_NEW if is_new else C_EDGE, zorder=2 if is_new else 1.5, capstyle="round"))

    # walk: vertical "fire pole" slides (same node, layer n → n-1) and horizontal hops (different nodes, same layer)
    walk = list(walk)
    for (n1, l1), (n2, l2) in zip(walk, walk[1:]):
        is_jump = jump is not None and (n1, n2, l1) == jump
        if n1 == n2 and l2 == l1 - 1:
            # paint the pole segment solid blue underneath the arrow so the descent reads as "down the pole"
            ax.plot([VALUE[n1]] * 2, [LAYER_Y[l1], LAYER_Y[l2]], color=C_WALK, lw=5, alpha=0.25, solid_capstyle="butt", zorder=3)
            ax.text(VALUE[n1] + 0.3, (LAYER_Y[l1] + LAYER_Y[l2]) / 2, "pole", ha="left", va="center", fontsize=8.5, color=C_WALK, style="italic", zorder=5)
        ax.annotate(
            "",
            xy=(VALUE[n2], LAYER_Y[l2]),
            xytext=(VALUE[n1], LAYER_Y[l1]),
            arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.35", color=C_JUMP if is_jump else C_WALK, lw=3.2 if is_jump else 2.4, shrinkA=NODE_R_PT + 7, shrinkB=NODE_R_PT + 7, connectionstyle="arc3,rad=-0.25" if is_jump else "arc3,rad=0"),
            zorder=4,
        )
        if is_jump:
            ax.text((VALUE[n1] + VALUE[n2]) / 2, LAYER_Y[l1] + 0.33, "jump: d=1 beats d=7", ha="center", va="bottom", fontsize=10, color=C_JUMP, fontweight="bold", zorder=5)

    # distance hints; a node a pole descends from gets its label beside the pole, not under the arrow
    pole_tops = {a for a, b in zip(walk, walk[1:]) if a[0] == b[0] and b[1] == a[1] - 1}
    for n, l, label in dist:
        if (n, l) in pole_tops:
            ax.text(VALUE[n] - 0.3, LAYER_Y[l] - 0.2, label, ha="right", va="top", fontsize=8.5, color=C_WALK, zorder=5, fontweight="bold")
        else:
            ax.text(VALUE[n], LAYER_Y[l] - 0.2, label, ha="center", va="top", fontsize=8.5, color=C_WALK, zorder=5, fontweight="bold")

    # nodes (every copy)
    visited = set(walk)
    rejected_at = {(n, l): label for n, l, label in rejected}
    for n in present:
        for l in range(TOP_LAYER[n] + 1):
            x, y = VALUE[n], LAYER_Y[l]
            if (n, l) in visited:
                ax.scatter([x], [y], s=(2 * NODE_R_PT + 12) ** 2, facecolor="none", edgecolor=C_WALK, lw=2.2, zorder=5)
            is_new = n == new
            is_entry = n == entry
            is_rej = (n, l) in rejected_at
            ax.scatter([x], [y], s=(2 * NODE_R_PT) ** 2, facecolor=C_NEW if is_new else (C_REJECT_FILL if is_rej else C_NODE), edgecolor=C_ENTRY if is_entry else (C_REJECT if is_rej else C_NODE_EDGE), lw=3.2 if is_entry else 1.4, zorder=6)
            ax.text(x, y, n, ha="center", va="center", fontsize=11, fontweight="bold", color="white" if is_new else (C_REJECT if is_rej else C_TEXT), zorder=7)
            if is_rej:
                # cross the candidate out; the label says which kept neighbour it hides behind
                r = 0.19
                ax.plot([x - r, x + r], [y - r * 0.55, y + r * 0.55], color=C_REJECT, lw=2.4, zorder=8, solid_capstyle="round")
                ax.plot([x - r, x + r], [y + r * 0.55, y - r * 0.55], color=C_REJECT, lw=2.4, zorder=8, solid_capstyle="round")
                if rejected_at[(n, l)]:
                    ax.text(x, y - 0.2, rejected_at[(n, l)], ha="center", va="top", fontsize=8, color=C_REJECT, style="italic", zorder=8)
        if n == new:  # candidates carry d= labels in the same slot; the new node is never a candidate
            ax.text(VALUE[n], LAYER_Y[0] - 0.2, f"value {VALUE[n]}", ha="center", va="top", fontsize=8.5, color=C_NEW, fontweight="bold", zorder=7)

    # legend (top right, small)
    lx, ly = 12.0, LAYER_Y[2] + 0.08
    ax.scatter([lx], [ly], s=120, facecolor=C_NODE, edgecolor=C_ENTRY, lw=2.4, zorder=6)
    ax.text(lx + 0.35, ly, "entry point", va="center", fontsize=8.5, color=C_MUTED)
    ax.scatter([lx + 2.6], [ly], s=120, facecolor=C_NEW, edgecolor=C_NODE_EDGE, lw=1, zorder=6)
    ax.text(lx + 2.95, ly, "being inserted", va="center", fontsize=8.5, color=C_MUTED)

    fig.text(0.5, 0.955, title or "HNSW build · 8 one-dimensional vectors · m = 2", ha="center", va="center", fontsize=15, fontweight="bold", color=C_TEXT)
    fig.text(0.5, 0.10, "\n".join(textwrap.wrap(caption, 92)), ha="center", va="center", fontsize=12.5, color=C_TEXT, linespacing=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def title_frame():
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor="white")
    fig.text(0.5, 0.58, "HNSW build", ha="center", va="center", fontsize=34, fontweight="bold", color=C_TEXT)
    fig.text(0.5, 0.46, "8 vectors, m = 2", ha="center", va="center", fontsize=22, color=C_TEXT)
    fig.text(0.5, 0.33, "what hnswlib builds for these 8 values\n1-D vectors, so position on the ruler = value. Each step: the new vector appears → walk from the entry point →\nkeep the nearest candidate, keep a further one only if it is closer to the new vector than to any neighbour already kept.", ha="center", va="center", fontsize=12.5, color=C_MUTED, linespacing=1.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------------- assembly
def build_frames():
    frames, durations = [], []

    def add(img, ms):
        frames.append(img)
        durations.append(ms)

    add(title_frame(), 2500)

    present, edges, entry = [], [], None
    for step in STEPS:
        n = step["node"]
        present = present + [n]
        if entry is None or TOP_LAYER[n] > TOP_LAYER[entry]:
            entry = n
        add(draw_frame(present, edges, entry, new=n, caption=step["appear"]), 2000)
        if step["walk"]:
            add(draw_frame(present, edges, entry, new=n, walk=step["walk"], jump=step.get("jump"), dist=step.get("dist", ()), caption=step["walk_caption"]), 2500)
        edges = edges + step["edges"]
        add(draw_frame(present, edges, entry, new=n, new_edges=step["edges"], rejected=step.get("rejected", ()), caption=step["linked"]), 2500)

    add(draw_frame(present, edges, entry, caption=FINAL_CAPTION), 4000)

    l0 = {frozenset((u, v)) for u, v, l in edges if l == 0}
    l1 = {frozenset((u, v)) for u, v, l in edges if l == 1}
    assert l0 == EXPECTED_L0, l0 ^ EXPECTED_L0
    assert l1 == EXPECTED_L1, l1 ^ EXPECTED_L1
    assert not any(l == 2 for _, _, l in edges)
    assert entry == "E"
    return frames, durations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_suffix(".gif")))
    ap.add_argument("--frames-dir", default=None)
    args = ap.parse_args()

    frames, durations = build_frames()
    if args.frames_dir:
        d = Path(args.frames_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(d / f"frame-{i:02d}.png")

    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames, {sum(durations) / 1000:.1f} s, {Path(args.out).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
