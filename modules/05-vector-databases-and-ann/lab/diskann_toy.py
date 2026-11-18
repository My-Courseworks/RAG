"""Vamana toy for M5: the module's eight 1-D vectors built with Microsoft DiskANN
(diskannpy 0.7.0), graph read back from the saved index file.

What it does
  1. Builds the eight values with `diskannpy.build_memory_index` (R=2, L=8, one
     thread) at alpha = 1.0, 1.1, 1.2, 1.3 and prints the neighbour lists the
     library wrote to disk.
  2. Replays the same build with MIRROR, a line-for-line Python transcription of
     `Index::link`, `iterate_to_fixed_point`, `occlude_list`, `prune_neighbors`
     and `inter_insert` from DiskANN tag 0.7.0 (commit df225d3), and asserts that
     MIRROR's neighbour lists equal the library's, stored order included.
     The step-by-step narration is printed from MIRROR. MIRROR is a simulation;
     the graphs it is checked against are the library's.
  3. Diffs alpha = 1.0 against alpha = 1.2 and prints the arithmetic.
  4. Traces queries hop by hop with MIRROR and checks the answers against
     `StaticMemoryIndex.search`.
  5. `--validate N`: N random small integer point sets (1-D and 2-D, several R, L,
     alpha); MIRROR must reproduce the library's graph on every one.

Run (diskannpy ships Linux x86-64 and Windows wheels only; on Apple Silicon use
the container described in supplementary/diskann/README.md):
    python -I diskann_toy.py            # toy + narration
    python -I diskann_toy.py --validate 300
"""
import os
import shutil
import struct
import sys
import tempfile

import numpy as np

try:
    import diskannpy
except ImportError:  # the mirror alone can still be imported
    diskannpy = None

NAMES = list("ABCDEFGH")
VALUES = np.array([[1], [2], [4], [7], [8], [11], [14], [15]], dtype=np.float32)
F32 = np.float32
FLT_MAX = np.finfo(np.float32).max
GRAPH_SLACK_FACTOR = F32(1.3)   # include/defaults.h:20
MAX_OCCLUSION_SIZE = 750        # include/defaults.h:13


# --------------------------------------------------------------------------
# library side
# --------------------------------------------------------------------------
def read_graph(path):
    """Parse the file written by InMemGraphStore::save_graph
    (src/in_mem_graph_store.cpp:200-230): u64 file size, u32 max degree,
    u32 start node, u64 frozen-point count, then per node u32 k + k * u32."""
    buf = open(path, "rb").read()
    size, max_degree, start, n_frozen = struct.unpack_from("<QIIQ", buf, 0)
    off, adj = 24, []
    while off < size:
        k = struct.unpack_from("<I", buf, off)[0]
        adj.append(list(struct.unpack_from(f"<{k}I", buf, off + 4)))
        off += 4 * (k + 1)
    return start, adj


def quiet(fn, *a, **kw):
    """diskannpy prints C++ progress to fd 1; silence it."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def library_build(points, R, L, alpha, keep_dir=False):
    d = tempfile.mkdtemp(prefix="vamana_toy_")
    quiet(diskannpy.build_memory_index, points, "l2", d, complexity=L,
          graph_degree=R, num_threads=1, alpha=alpha)
    start, adj = read_graph(os.path.join(d, "ann"))
    if keep_dir:
        return start, adj, d
    shutil.rmtree(d)
    return start, adj


# --------------------------------------------------------------------------
# MIRROR: Python transcription of DiskANN 0.7.0 (SIMULATION, checked against
# the library's saved graph)
# --------------------------------------------------------------------------
def sqdist(a, b):
    """DistanceL2Float::compare returns the squared L2 distance, no sqrt
    (src/distance.cpp:177-230)."""
    acc = F32(0)
    for x, y in zip(a, b):
        t = F32(x) - F32(y)
        acc = F32(acc + t * t)
    return acc


def medoid(points):
    """InMemDataStore::calculate_medoid (src/in_mem_data_store.cpp:341-390):
    the stored point nearest the centroid, first index on ties."""
    n, dim = points.shape
    centre = [F32(0)] * dim
    for i in range(n):
        for j in range(dim):
            centre[j] = F32(centre[j] + points[i, j])
    centre = [F32(c / F32(n)) for c in centre]
    best, best_d = 0, sqdist(centre, points[0])
    for i in range(1, n):
        d = sqdist(centre, points[i])
        if d < best_d:
            best, best_d = i, d
    return best


class BestL:
    """NeighborPriorityQueue (include/neighbor.h:39-135): sorted by
    (distance, id), capacity L, a cursor at the first unexpanded entry."""

    def __init__(self, cap):
        self.cap, self.data, self.cur = cap, [], 0   # entries: [dist, id, expanded]

    def insert(self, dist, nid):
        key = (dist, nid)
        if len(self.data) == self.cap and (self.data[-1][0], self.data[-1][1]) < key:
            return False
        lo, hi = 0, len(self.data)
        while lo < hi:
            mid = (lo + hi) >> 1
            if key < (self.data[mid][0], self.data[mid][1]):
                hi = mid
            elif self.data[mid][1] == nid:
                return False
            else:
                lo = mid + 1
        self.data.insert(lo, [dist, nid, False])
        if len(self.data) > self.cap:
            self.data.pop()
        if lo < self.cur:
            self.cur = lo
        return True

    def has_unexpanded(self):
        return self.cur < len(self.data)

    def closest_unexpanded(self):
        self.data[self.cur][2] = True
        pre = self.cur
        while self.cur < len(self.data) and self.data[self.cur][2]:
            self.cur += 1
        return self.data[pre]


def greedy_search(points, graph, start, q, L, log=None, name=str):
    """Index::iterate_to_fixed_point (src/index.cpp:790-974). Returns the list
    (best L, sorted) and the expanded nodes in expansion order."""
    best = BestL(L)
    visited = {start}
    best.insert(sqdist(q, points[start]), start)
    expanded = []
    while best.has_unexpanded():
        dist, n, _ = best.closest_unexpanded()
        expanded.append((dist, n))
        fresh = [v for v in graph[n] if v not in visited]
        visited.update(fresh)
        added = []
        for v in fresh:
            dv = sqdist(q, points[v])
            if best.insert(dv, v):
                added.append(v)
        if log is not None:
            log.append(dict(node=n, dist=dist, out=list(graph[n]), fresh=fresh, added=added,
                            lst=[(d, i, e) for d, i, e in best.data]))
    return [(d, i) for d, i, _ in best.data], expanded


def occlude_list(points, p, pool, alpha, R, log=None):
    """Index::occlude_list (src/index.cpp:1055-1148). `pool` is sorted by
    (squared distance to p, id). Rounds run at cur_alpha = 1, 1.2, 1.44, ...
    while cur_alpha <= alpha; a candidate is skipped in a round while its
    occlude factor d(p,c)/d(kept,c) exceeds cur_alpha."""
    alpha = F32(alpha)
    pool = pool[:MAX_OCCLUSION_SIZE]
    factor = [F32(0)] * len(pool)
    result = []
    cur_alpha = F32(1)
    while cur_alpha <= alpha and len(result) < R:
        for i, (di, ci) in enumerate(pool):
            if len(result) >= R:
                break
            if factor[i] > cur_alpha:
                continue
            was = factor[i]
            factor[i] = FLT_MAX
            if ci != p:
                result.append(ci)
                if log is not None:
                    log.append(("keep", float(cur_alpha), ci, float(di), float(was)))
            for t in range(i + 1, len(pool)):
                if factor[t] > alpha:
                    continue
                dt, ct = pool[t]
                djk = sqdist(points[ct], points[ci])
                new = FLT_MAX if djk == 0 else max(factor[t], F32(dt / djk))
                if log is not None and new > factor[t] and factor[t] != FLT_MAX:
                    log.append(("occlude", float(cur_alpha), ct, ci, float(dt), float(djk), float(new)))
                factor[t] = new
        cur_alpha = F32(cur_alpha * F32(1.2))
    return result


def prune_neighbors(points, p, pool, alpha, R, log=None):
    """Index::prune_neighbors (src/index.cpp:1158-1196), saturate_graph=false."""
    if not pool:
        return []
    pool = sorted(pool)
    return occlude_list(points, p, pool, alpha, R, log)


def inter_insert(points, graph, n, pruned, alpha, R, log=None):
    """Index::inter_insert (src/index.cpp:1199-1261): write the back-edge
    des -> n; re-prune des only when its list already holds
    floor(1.3 * R) entries."""
    cap = int(GRAPH_SLACK_FACTOR * F32(R))
    for des in pruned:
        if n in graph[des]:
            continue
        if len(graph[des]) < cap:
            graph[des].append(n)
            if log is not None:
                log.append(("append", des, n, list(graph[des])))
            continue
        pool, seen = [], set()
        for c in graph[des] + [n]:
            if c not in seen and c != des:
                pool.append((sqdist(points[des], points[c]), c))
                seen.add(c)
        sub = []
        before = list(graph[des])
        graph[des] = prune_neighbors(points, des, pool, alpha, R, sub)
        if log is not None:
            log.append(("reprune", des, n, before, sorted(pool), sub, list(graph[des])))


def mirror_build(points, R, L, alpha, steps=None):
    """Index::link (src/index.cpp:1269-1370) with one thread: empty graph,
    start = medoid, nodes visited in id order, one pass, then a cleanup pass
    over any list longer than R."""
    n = len(points)
    start = medoid(points)
    graph = [[] for _ in range(n)]
    for p in range(n):
        slog, plog, blog = [], [], []
        _, expanded = greedy_search(points, graph, start, points[p], L, slog)
        pool = [(d, i) for d, i in expanded if i != p]
        pruned = prune_neighbors(points, p, pool, alpha, R, plog)
        graph[p] = list(pruned)
        inter_insert(points, graph, p, pruned, alpha, R, blog)
        if steps is not None:
            steps.append(dict(p=p, search=slog, pool=sorted(pool), prune=plog, kept=list(pruned),
                              back=blog, graph=[list(g) for g in graph]))
    cleanup = []
    for p in range(n):
        if len(graph[p]) > R:
            pool = [(sqdist(points[p], points[c]), c) for c in dict.fromkeys(graph[p]) if c != p]
            before = list(graph[p])
            graph[p] = prune_neighbors(points, p, pool, alpha, R)
            cleanup.append((p, before, list(graph[p])))
    if steps is not None:
        steps.append(dict(cleanup=cleanup))
    return start, graph


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------
def nm(i):
    return f"{NAMES[i]}({int(VALUES[i, 0])})"


def show_graph(adj, indent="  "):
    for i, a in enumerate(adj):
        print(f"{indent}{nm(i)} -> [{', '.join(NAMES[j] for j in a)}]")


def root(d):
    return f"{float(np.sqrt(d)):g}"


def narrate(steps, R, alpha):
    for st in steps:
        if "cleanup" in st:
            print(f"\nCleanup pass: {st['cleanup'] if st['cleanup'] else 'no list is longer than R, nothing to do'}")
            continue
        p = st["p"]
        print(f"\n{p + 1}. insert {nm(p)}")
        for h in st["search"]:
            lst = " ".join(f"{NAMES[i]}{root(d)}{'*' if e else ''}" for d, i, e in h["lst"])
            print(f"   search: expand {NAMES[h['node']]} (d={root(h['dist'])}), out-list "
                  f"[{', '.join(NAMES[j] for j in h['out'])}], new: "
                  f"[{', '.join(NAMES[j] for j in h['fresh'])}] -> list [{lst}]  (* = expanded)")
        print(f"   visited set, sorted by distance to {NAMES[p]}: "
              f"[{', '.join(f'{NAMES[i]} {root(d)}' for d, i in st['pool'])}]")
        for ev in st["prune"]:
            if ev[0] == "keep":
                _, ca, c, d, was = ev
                why = "" if was == 0 else f" (ratio {was:.4f} <= {ca:.2f})"
                print(f"   prune round alpha={ca:.2f}: keep {NAMES[c]}{why}")
            else:
                _, ca, c, k, d, djk, new = ev
                print(f"   prune round alpha={ca:.2f}:   {NAMES[c]} vs kept {NAMES[k]}: "
                      f"d({NAMES[p]},{NAMES[c]})^2 / d({NAMES[k]},{NAMES[c]})^2 = {d:g}/{djk:g} = {new:.4f}"
                      f"{'  -> behind ' + NAMES[k] if new > 1 else '  -> not behind'}")
        print(f"   {NAMES[p]} -> [{', '.join(NAMES[j] for j in st['kept'])}]")
        for ev in st["back"]:
            if ev[0] == "append":
                _, des, n, after = ev
                print(f"   back-edge {NAMES[des]} -> {NAMES[n]}: room, appended; "
                      f"{NAMES[des]} -> [{', '.join(NAMES[j] for j in after)}]")
            else:
                _, des, n, before, pool, sub, after = ev
                print(f"   back-edge {NAMES[des]} -> {NAMES[n]}: list full "
                      f"[{', '.join(NAMES[j] for j in before)}], re-prune "
                      f"[{', '.join(f'{NAMES[i]} {root(d)}' for d, i in pool)}] from {NAMES[des]}")
                for s in sub:
                    if s[0] == "keep":
                        print(f"      round alpha={s[1]:.2f}: keep {NAMES[s[2]]}"
                              + ("" if s[4] == 0 else f" (ratio {s[4]:.4f} <= {s[1]:.2f})"))
                    else:
                        _, ca, c, k, d, djk, new = s
                        print(f"      round alpha={ca:.2f}:   {NAMES[c]} vs kept {NAMES[k]}: "
                              f"{d:g}/{djk:g} = {new:.4f}{'  -> behind ' + NAMES[k] if new > 1 else '  -> not behind'}")
                print(f"      {NAMES[des]} -> [{', '.join(NAMES[j] for j in after)}]")
        print("   graph: " + "  ".join(f"{NAMES[i]}>{''.join(NAMES[j] for j in g)}"
                                       for i, g in enumerate(st["graph"]) if g))


def trace_query(points, adj, start, q, L, k):
    log = []
    best, expanded = greedy_search(points, adj, start, np.array([q], dtype=np.float32), L, log)
    print(f"\nquery {q:g}, L={L}, k={k}: start at {nm(start)}")
    for hop, h in enumerate(log, 1):
        lst = " ".join(f"{NAMES[i]}{root(d)}{'*' if e else ''}" for d, i, e in h["lst"])
        print(f"   hop {hop}: expand {NAMES[h['node']]} (d={root(h['dist'])}), out-list "
              f"[{', '.join(NAMES[j] for j in h['out'])}], unseen "
              f"[{', '.join(NAMES[j] for j in h['fresh'])}], admitted "
              f"[{', '.join(NAMES[j] for j in h['added'])}] -> list [{lst}]")
    print(f"   stop: every entry in the list is expanded. {len(log)} expansions, "
          f"{sum(len(h['fresh']) for h in log) + 1} distance computations of 8.")
    print(f"   answer: {[NAMES[i] for _, i in best[:k]]}")
    return [i for _, i in best[:k]], [float(d) for d, _ in best[:k]]


# --------------------------------------------------------------------------
def toy():
    R, L = 2, 8
    print(f"diskannpy {__import__('importlib.metadata').metadata.version('diskannpy')}, "
          f"R={R}, L={L}, l2, num_threads=1")
    graphs = {}
    for alpha in (1.0, 1.1, 1.2, 1.3):
        start, adj, d = library_build(VALUES, R, L, alpha, keep_dir=True)
        graphs[alpha] = (start, adj, d)
        print(f"\nLIBRARY graph, alpha={alpha}: start node {nm(start)}, "
              f"{sum(len(a) for a in adj)} directed edges")
        show_graph(adj)
        m_start, m_adj = mirror_build(VALUES, R, L, alpha)
        assert m_start == start and m_adj == adj, f"MIRROR != library at alpha={alpha}"
        print("  MIRROR reproduces this graph exactly (start node and ordered lists).")

    for a, b in ((1.0, 1.1), (1.2, 1.3)):
        same = graphs[a][1] == graphs[b][1]
        print(f"\nalpha={a} vs alpha={b}: library graphs identical = {same} "
              f"(rounds run at 1, 1.2, 1.44, ...; values in between round down)")

    e10 = {(i, j) for i, a in enumerate(graphs[1.0][1]) for j in a}
    e12 = {(i, j) for i, a in enumerate(graphs[1.2][1]) for j in a}
    print("\nalpha=1.0 vs alpha=1.2 (library):")
    print("  only at 1.2:", sorted(f"{NAMES[i]}->{NAMES[j]}" for i, j in e12 - e10))
    print("  only at 1.0:", sorted(f"{NAMES[i]}->{NAMES[j]}" for i, j in e10 - e12))
    print("  threshold on true distance: a farther same-side candidate at b survives a kept neighbour at a when")
    for al in (1.2,):
        s = float(np.sqrt(np.float32(al)))
        print(f"    code  (squared L2): b^2/(b-a)^2 <= {al}  <=>  b >= {s / (s - 1):.2f} a   (sqrt({al}) = {s:.4f})")
        print(f"    paper (L2):         alpha (b-a) >  b       <=>  b >  {al / (al - 1):.2f} a")

    for alpha in (1.0, 1.2):
        print(f"\n================ MIRROR narration, alpha={alpha} (graph checked against the library) ================")
        steps = []
        mirror_build(VALUES, R, L, alpha, steps)
        narrate(steps, R, alpha)

    # query traces, answers checked against the library
    for alpha in (1.0, 1.2):
        start, adj, d = graphs[alpha]
        idx = quiet(diskannpy.StaticMemoryIndex, index_directory=d, num_threads=1, initial_search_complexity=8)
        print(f"\n================ query traces on the alpha={alpha} library graph ================")
        for q, Ls, k in ((1.6, 2, 1), (14.4, 2, 1), (3.2, 1, 1), (3.2, 2, 1)):
            ids, dists = trace_query(VALUES, adj, start, q, Ls, k)
            r = idx.search(np.array([q], dtype=np.float32), k_neighbors=k, complexity=Ls)
            lib_ids = [int(x) for x in r.identifiers]
            print(f"   library answer: {[NAMES[i] for i in lib_ids]}, distances {[float(x) for x in r.distances]} "
                  f"(squared L2){'  == MIRROR' if lib_ids == ids else '  != MIRROR'}")
            assert lib_ids == ids
    for _, _, d in graphs.values():
        shutil.rmtree(d)


def validate(n_cases):
    rng = np.random.default_rng(7)
    bad = 0
    for case in range(n_cases):
        dim = int(rng.integers(1, 3))
        n = int(rng.integers(6, 41))
        span = 60 if dim == 1 else 25
        pts = set()
        while len(pts) < n:
            pts.add(tuple(int(x) for x in rng.integers(0, span, dim)))
        pts = np.array(list(pts), dtype=np.float32)
        rng.shuffle(pts)
        R = int(rng.choice([2, 3, 4, 6, 8]))
        L = int(rng.choice([R, R + 1, 2 * R, 10, 50]))
        alpha = float(rng.choice([1.0, 1.1, 1.2, 1.3, 1.44, 1.5, 2.0]))
        start, adj = library_build(pts, R, L, alpha)
        m_start, m_adj = mirror_build(pts, R, L, alpha)
        if (start, adj) != (m_start, m_adj):
            bad += 1
            print(f"MISMATCH case {case}: dim={dim} n={n} R={R} L={L} alpha={alpha}")
    print(f"validate: {n_cases - bad}/{n_cases} random builds reproduced exactly by MIRROR")
    return bad


if __name__ == "__main__":
    if diskannpy is None:
        sys.exit("diskannpy is not importable here; see supplementary/diskann/README.md for the container.")
    if len(sys.argv) > 2 and sys.argv[1] == "--validate":
        sys.exit(1 if validate(int(sys.argv[2])) else 0)
    toy()
