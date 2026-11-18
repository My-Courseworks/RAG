"""Annoy toy: 8 one-dimensional values, random-projection trees, search_k walk.

Builds the real Annoy index (euclidean, f=1) with n_trees = 1 and 3 at a fixed
seed, saves it, parses the on-disk node layout straight from the file (Annoy has
no public tree walk), prints every split threshold and leaf, then replays
`_get_all_nns` step by step for query 9.5 at search_k = 1, 2, 3 and checks the
replay against the library's own `get_nns_by_vector`.

Node layout for the Euclidean metric (src/annoylib.h, struct Minkowski::Node):
    int32 n_descendants | float32 a | int32 children[2] | float32 v[f]
so a node is 16 + 4*f bytes; with f = 1 that is 20 bytes and
_K = (20 - offsetof(children)=8) / 4 = 3 items fit in a leaf.

Run:  python -I ivf_annoy_toy.py [--seed S]
"""
import argparse
import heapq
import os
import struct
import tempfile

from annoy import AnnoyIndex

LABELS = "ABCDEFGH"
VALUES = [1.0, 2.0, 4.0, 7.0, 8.0, 11.0, 14.0, 15.0]
F = 1
NODE_HEADER = 16  # n_descendants(4) + a(4) + children[2](8)
NODE_SIZE = NODE_HEADER + 4 * F
K = (NODE_SIZE - 8) // 4  # max items that fit in a leaf node


def read_nodes(path):
    raw = open(path, "rb").read()
    assert len(raw) % NODE_SIZE == 0, len(raw)
    nodes = []
    for off in range(0, len(raw), NODE_SIZE):
        n_desc, a, c0, c1 = struct.unpack_from("<ifii", raw, off)
        v = struct.unpack_from("<%df" % F, raw, off + NODE_HEADER)
        nodes.append({"n_desc": n_desc, "a": a, "children": (c0, c1), "v": v, "raw": raw[off:off + NODE_SIZE]})
    return nodes


def leaf_items(node):
    # 2 <= n_descendants <= K: the children array overflows into v and holds item ids
    return list(struct.unpack_from("<%di" % node["n_desc"], node["raw"], 8))


def name(i):
    return "%s(%g)" % (LABELS[i], VALUES[i])


def print_tree(nodes, nid, n_items, depth=0, is_root=False):
    pad = "   " * depth
    if nid < n_items:
        print("%s- item %s" % (pad, name(nid)))
        return
    nd = nodes[nid]
    if nd["n_desc"] <= K and not is_root:
        print("%s- leaf node %d: {%s}" % (pad, nid, ", ".join(name(i) for i in leaf_items(nd))))
        return
    v0, a = nd["v"][0], nd["a"]
    thr = -a / v0
    sign = ">" if v0 > 0 else "<"
    print("%s- split node %d: v=%+.4f a=%+.4f  => right (children[1]) iff x %s %.4f" % (pad, nid, v0, a, sign, thr))
    print_tree(nodes, nd["children"][0], n_items, depth + 1)
    print_tree(nodes, nd["children"][1], n_items, depth + 1)


def replay_query(nodes, roots, n_items, q, n, search_k, verbose=True):
    """Python replay of AnnoyIndex::_get_all_nns (annoylib.h, Euclidean/Minkowski)."""
    if search_k == -1:
        search_k = n * len(roots)
    heap = []  # max-heap on priority -> store negatives
    for r in roots:
        heapq.heappush(heap, (-float("inf"), r))
    nns = []
    step = 0
    while len(nns) < search_k and heap:
        neg, i = heapq.heappop(heap)
        d = -neg
        nd = nodes[i]
        step += 1
        if nd["n_desc"] == 1 and i < n_items:
            nns.append(i)
            if verbose:
                print("     pop  item %-6s prio=%+.3f -> candidate" % (name(i), d))
        elif nd["n_desc"] <= K:
            items = leaf_items(nd)
            nns.extend(items)
            if verbose:
                print("     pop  leaf %d prio=%+.3f -> candidates {%s}  (%d so far, budget %d)"
                      % (i, d, ", ".join(name(x) for x in items), len(nns), search_k))
        else:
            margin = nd["a"] + nd["v"][0] * q
            p1 = min(d, margin)
            p0 = min(d, -margin)
            heapq.heappush(heap, (-p1, nd["children"][1]))
            heapq.heappush(heap, (-p0, nd["children"][0]))
            if verbose:
                thr = -nd["a"] / nd["v"][0]
                print("     pop  split %d (x vs %.3f) prio=%+.3f: margin=%+.3f -> push right prio %+.3f, left prio %+.3f"
                      % (i, thr, d, margin, p1, p0))
    cands = sorted(set(nns))
    scored = sorted((abs(VALUES[j] - q), j) for j in cands)
    top = [j for _, j in scored[:n]]
    if verbose:
        print("     rerank %d unique candidates exactly: %s" % (len(cands), ", ".join("%s d=%.1f" % (name(j), dd) for dd, j in scored)))
        print("     return top-%d: %s" % (n, ", ".join(name(j) for j in top)))
    return top, cands


def build(n_trees, seed, path):
    idx = AnnoyIndex(F, "euclidean")
    idx.set_seed(seed)
    for i, x in enumerate(VALUES):
        idx.add_item(i, [x])
    idx.build(n_trees, n_jobs=1)
    idx.save(path)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--query", type=float, default=9.5)
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    q, n = args.query, args.n

    print("values:", ", ".join(name(i) for i in range(len(VALUES))))
    print("f=%d euclidean; node size %d bytes; _K (items per leaf) = %d; seed=%d" % (F, NODE_SIZE, K, args.seed))
    exact = sorted(range(len(VALUES)), key=lambda j: abs(VALUES[j] - q))[:n]
    print("query %g, exact top-%d: %s" % (q, n, ", ".join(name(j) for j in exact)))

    tmp = tempfile.mkdtemp(prefix="annoy-toy-")
    for n_trees in (1, 3):
        path = os.path.join(tmp, "toy_%d.ann" % n_trees)
        idx = build(n_trees, args.seed, path)
        nodes = read_nodes(path)
        n_items = idx.get_n_items()
        # build() appends a copy of every root at the end of the node array (annoylib.h build())
        roots = list(range(len(nodes) - n_trees, len(nodes)))
        print("\n================ n_trees = %d  (file %d bytes = %d nodes: %d items + %d tree nodes + %d root copies)"
              % (n_trees, os.path.getsize(path), len(nodes), n_items, len(nodes) - n_items - n_trees, n_trees))
        for t, r in enumerate(roots):
            print("--- tree %d (root node %d, n_descendants=%d)" % (t, r, nodes[r]["n_desc"]))
            print_tree(nodes, r, n_items, is_root=True)
        for search_k in (1, 2, 3, 4):
            print("\n  query %g  n=%d  search_k=%d  (default would be n*n_trees=%d)" % (q, n, search_k, n * n_trees))
            top, cands = replay_query(nodes, roots, n_items, q, n, search_k)
            lib = idx.get_nns_by_vector([q], n, search_k=search_k)
            ok = sorted(lib) == sorted(top)
            hit = len(set(top) & set(exact))
            print("     library get_nns_by_vector -> %s  | replay %s | recall@%d = %d/%d%s"
                  % (", ".join(name(j) for j in lib), "matches" if ok else "DIFFERS", n, hit, n,
                     "" if hit == n else "  <-- MISS: %s" % ", ".join(name(j) for j in exact if j not in top)))
            assert ok, (lib, top)
        idx.unload()


if __name__ == "__main__":
    main()
