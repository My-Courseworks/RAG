"""20 one-dimensional points, M=2, ef_construction=3: what hnswlib builds, narrated.

Two halves:

1. Ground truth. hnswlib builds the index (l2, dim 1, M=2, ef_construction=3, one thread,
   points added in a shuffled order) and we read its neighbour lists out of the pickled state,
   layer by layer, the way lab/hnsw_toy_check.py does.
2. A Python re-implementation of hnswlib's addPoint (greedy descent, searchBaseLayer,
   getNeighborsByHeuristic2, mutuallyConnectNewElement) that logs every pool change, every
   keep/reject decision and every prune, so the walk can be narrated. It uses the level
   sequence hnswlib drew (element_levels). Distance ties are frequent with 20 integers in
   [1, 40], and hnswlib resolves them by whatever order its std::priority_queue happens to
   hold, so the simulation models libc++'s heap operations exactly (class PQ) instead of
   sorting. It is checked against hnswlib's neighbour lists, order included, on every layer,
   for ef_construction=3 and for ef_construction=200.

Usage (from the repo root, inside the venv that has hnswlib + numpy):
    python -I modules/05-vector-databases-and-ann/lab/hnsw_toy_20_ef3.py          # markdown for the doc
    python -I modules/05-vector-databases-and-ann/lab/hnsw_toy_20_ef3.py --scan   # how the seeds were picked
"""
import struct
import sys

import numpy as np
import hnswlib

N = 20
NAMES = [chr(ord("A") + i) for i in range(N)]
M = 2
EF_SMALL = 3
EF_BIG = 200

# Picked by --scan (see scan() below): first (hnsw random_seed with a readable hierarchy,
# shuffle seed) pair where ef_construction=3 and ef_construction=200 build different graphs
# and the simulation reproduces hnswlib's lists, order included, for both.
HNSW_SEED = 244
SHUFFLE_SEED = 1

# libc++ changed pop_heap in LLVM 15 (Floyd sift-down + sift-up). Both are modelled; --scan
# reports which one reproduces hnswlib's stored list order on this machine.
HEAP_VARIANT = "floyd"


# ----------------------------------------------------------------------------- points
def draw_points(shuffle_seed):
    rng = np.random.default_rng(shuffle_seed)
    return [int(v) for v in rng.choice(np.arange(1, 41), size=N, replace=False)]


# ----------------------------------------------------------------------------- hnswlib
def build(values, ef_c, seed):
    idx = hnswlib.Index(space="l2", dim=1)
    idx.init_index(max_elements=N, M=M, ef_construction=ef_c, random_seed=seed)
    idx.set_num_threads(1)
    idx.add_items(np.array(values, dtype=np.float32).reshape(-1, 1), np.arange(N))
    return idx


def state_of(idx):
    st = idx.__getstate__()
    if isinstance(st, tuple):
        for part in st:
            if isinstance(part, dict) and "data_level0" in part:
                return part
    return st


def dump_graph(idx):
    """-> ({layer: {internal_id: [neighbour ids in stored order]}}, state)."""
    st = state_of(idx)
    n = st["cur_element_count"]
    sde, slpe = st["size_data_per_element"], st["size_links_per_element"]
    lab = {int(i): int(e) for e, i in zip(st["label_lookup_external"], st["label_lookup_internal"])}
    assert all(lab[i] == i for i in range(n)), "labels were given in insertion order; internal id == label"
    d0, ll, lv = st["data_level0"], st["link_lists"], list(st["element_levels"])
    adj = {0: {}}
    for i in range(n):
        off = i * sde
        cnt = struct.unpack_from("H", d0, off)[0]
        adj[0][i] = list(struct.unpack_from(f"{cnt}I", d0, off + 4))
    pos = 0
    for i in range(n):
        for l in range(1, lv[i] + 1):
            off = pos + (l - 1) * slpe
            cnt = struct.unpack_from("H", ll, off)[0]
            adj.setdefault(l, {})[i] = list(struct.unpack_from(f"{cnt}I", ll, off + 4))
        pos += slpe * lv[i]
    return adj, st


# ----------------------------------------------------------------------------- std::priority_queue
class PQ:
    """std::priority_queue<T, std::vector<T>, Comp> as libc++ implements it.
    `less(a, b)` is Comp: the top is the element nothing compares greater than (a max-heap)."""

    def __init__(self, less, variant):
        self.a, self.less, self.variant = [], less, variant

    def __len__(self):
        return len(self.a)

    def top(self):
        return self.a[0]

    def push(self, x):
        self.a.append(x)
        self._sift_up(len(self.a))

    def _sift_up(self, length):
        a, less = self.a, self.less
        if length > 1:
            parent = (length - 2) // 2
            last = length - 1
            if less(a[parent], a[last]):
                t = a[last]
                while True:
                    a[last] = a[parent]
                    last = parent
                    if parent == 0:
                        break
                    parent = (parent - 1) // 2
                    if not less(a[parent], t):
                        break
                a[last] = t

    def pop(self):
        a, n = self.a, len(self.a)
        if n > 1:
            if self.variant == "floyd":
                top = a[0]
                hole = self._floyd_sift_down(n)
                last = n - 1
                if hole == last:
                    a[hole] = top
                else:
                    a[hole] = a[last]
                    a[last] = top
                    self._sift_up(hole + 1)
            else:
                a[0], a[n - 1] = a[n - 1], a[0]
                self._sift_down(n - 1)
        a.pop()

    def _floyd_sift_down(self, n):
        a, less = self.a, self.less
        hole = child_i = child = 0
        while True:
            child_i += child + 1
            child = 2 * child + 1
            if child + 1 < n and less(a[child_i], a[child_i + 1]):
                child_i += 1
                child += 1
            a[hole] = a[child_i]
            hole = child_i
            if child > (n - 2) // 2:
                return hole

    def _sift_down(self, n):
        a, less = self.a, self.less
        start = child = 0
        if n < 2 or (n - 2) // 2 < child:
            return
        child = 2 * child + 1
        child_i = child
        if child + 1 < n and less(a[child_i], a[child_i + 1]):
            child_i += 1
            child += 1
        if less(a[child_i], a[start]):
            return
        top = a[start]
        while True:
            a[start] = a[child_i]
            start = child_i
            if (n - 2) // 2 < child:
                break
            child = 2 * child + 1
            child_i = child
            if child + 1 < n and less(a[child_i], a[child_i + 1]):
                child_i += 1
                child += 1
            if less(a[child_i], top):
                break
        a[start] = top


def by_first(a, b):          # hnswlib's CompareByFirst
    return a[0] < b[0]


def by_pair(a, b):           # std::less<std::pair<dist_t, tableint>>
    return a < b


# ----------------------------------------------------------------------------- simulation
class Sim:
    """hnswlib's addPoint, single-threaded, no deletions, with an event log per insert."""

    def __init__(self, values, levels, ef_c, variant=HEAP_VARIANT):
        self.v = values
        self.levels = levels
        self.ef = max(ef_c, M)          # hnswlib: ef_construction_ = max(ef_construction, M_)
        self.maxM, self.maxM0 = M, 2 * M
        self.links = {}                 # (node, layer) -> list in hnswlib storage order
        self.ep, self.maxlevel = None, -1
        self.variant = variant
        self.logs = []

    # l2 space is squared distance; in 1-D |a-b| orders identically (ties included), so narrate with |a-b|
    def d(self, a, b):
        return abs(self.v[a] - self.v[b])

    def nm(self, i):
        return f"{NAMES[i]}({self.v[i]})"

    def pool_str(self, pool):
        return "[" + " ".join(f"{NAMES[i]}{dd}" for dd, i in sorted(pool)) + "]"

    def insert(self, c):
        lvl = self.levels[c]
        log = {"node": c, "level": lvl, "upper": [], "layers": [], "new_entry": False, "first": False}
        self.logs.append(log)
        for l in range(lvl + 1):
            self.links[(c, l)] = []
        if self.ep is None:
            self.ep, self.maxlevel = c, lvl
            log["first"] = log["new_entry"] = True
            return
        cur, maxlevelcopy = self.ep, self.maxlevel
        if lvl < maxlevelcopy:
            curdist = self.d(c, cur)
            for layer in range(maxlevelcopy, lvl, -1):
                hops = []
                changed = True
                while changed:
                    changed = False
                    seen = [(nb, self.d(c, nb)) for nb in self.links[(cur, layer)]]
                    before = cur
                    for nb, dd in seen:
                        if dd < curdist:
                            curdist, cur, changed = dd, nb, True
                    hops.append({"at": before, "list": seen, "moved_to": cur if changed else None, "curdist": curdist})
                log["upper"].append({"layer": layer, "hops": hops})
        for layer in range(min(lvl, maxlevelcopy), -1, -1):
            lg = {"layer": layer, "start": cur, "steps": [], "stop": None}
            top = self.search_base_layer(cur, c, layer, lg)
            cur = self.connect(c, top, layer, lg)
            log["layers"].append(lg)
        if lvl > maxlevelcopy:
            self.ep, self.maxlevel = c, lvl
            log["new_entry"] = True

    def search_base_layer(self, ep, q, layer, lg):
        ef = self.ef
        top = PQ(by_first, self.variant)      # (dist, id): result pool, top = worst
        cset = PQ(by_first, self.variant)     # (-dist, id): candidates, top = nearest unexpanded
        d0 = self.d(q, ep)
        top.push((d0, ep))
        lower = d0
        cset.push((-d0, ep))
        visited = {ep}
        lg["start_dist"] = d0
        while len(cset):
            cd, cn = -cset.top()[0], cset.top()[1]
            if cd > lower and len(top) == ef:
                lg["stop"] = {"rule": "early", "cand": cn, "cd": cd, "lower": lower,
                              "unexpanded": sorted(n for _, n in cset.a)}
                break
            cset.pop()
            step = {"expand": cn, "cd": cd, "events": []}
            for nb in self.links[(cn, layer)]:
                if nb in visited:
                    continue
                visited.add(nb)
                dd = self.d(q, nb)
                if len(top) < ef or lower > dd:
                    cset.push((-dd, nb))
                    top.push((dd, nb))
                    evicted = None
                    if len(top) > ef:
                        evicted = top.top()
                        top.pop()
                    lower = top.top()[0]
                    step["events"].append({"nb": nb, "d": dd, "added": True, "evicted": evicted,
                                           "pool": list(top.a), "lower": lower})
                else:
                    step["events"].append({"nb": nb, "d": dd, "added": False, "lower": lower})
            lg["steps"].append(step)
        else:
            lg["stop"] = {"rule": "exhausted"}
        lg["pool"] = list(top.a)
        return top

    def heuristic(self, top, Mkeep, ref, decisions=None):
        """getNeighborsByHeuristic2, in place on the PQ `top` of (dist_to_ref, id)."""
        if len(top) < Mkeep:
            if decisions is not None:
                for dd, i in sorted(top.a):
                    decisions.append({"cand": i, "d": dd, "kept": True, "why": "fewer than M candidates: keep all"})
            return
        qc = PQ(by_pair, self.variant)        # (-dist, id): top = nearest, larger id first on ties
        while len(top):
            qc.push((-top.top()[0], top.top()[1]))
            top.pop()
        ret = []
        while len(qc):
            if len(ret) >= Mkeep:
                if decisions is not None:
                    for nd, i in sorted(qc.a, reverse=True):
                        decisions.append({"cand": i, "d": -nd, "kept": False, "why": "M already kept"})
                break
            cur = qc.top()
            dq = -cur[0]
            qc.pop()
            blocker = None
            for kept in ret:
                dk = self.d(kept[1], cur[1])
                if dk < dq:
                    blocker = (kept[1], dk)
                    break
            if blocker is None:
                ret.append(cur)
            if decisions is not None:
                decisions.append({"cand": cur[1], "d": dq, "kept": blocker is None, "why": blocker})
        for cur in ret:
            top.push((-cur[0], cur[1]))

    def connect(self, c, top, layer, lg):
        Mcurmax = self.maxM if layer > 0 else self.maxM0
        lg["select"] = []
        self.heuristic(top, M, c, lg["select"])
        selected = []
        while len(top):
            selected.append(top.top()[1])
            top.pop()
        next_ep = selected[-1]                 # farthest popped first, so the last is the nearest
        self.links[(c, layer)] = list(selected)
        lg["edges"] = list(selected)
        lg["prunes"] = []
        for nb in selected:
            lst = self.links[(nb, layer)]
            if len(lst) < Mcurmax:
                lst.append(c)
            else:
                cands = PQ(by_first, self.variant)
                cands.push((self.d(c, nb), c))
                for x in lst:
                    cands.push((self.d(x, nb), x))
                decisions = []
                self.heuristic(cands, Mcurmax, nb, decisions)
                new = []
                while len(cands):
                    new.append(cands.top()[1])
                    cands.pop()
                lg["prunes"].append({"node": nb, "before": list(lst), "after": list(new), "decisions": decisions})
                self.links[(nb, layer)] = new
        return next_ep

    def adjacency(self):
        adj = {}
        for (n, l), lst in self.links.items():
            adj.setdefault(l, {})[n] = list(lst)
        return adj


def run_sim(values, levels, ef_c, variant=HEAP_VARIANT):
    sim = Sim(values, levels, ef_c, variant)
    for c in range(N):
        sim.insert(c)
    return sim


def same_sets(adj_a, adj_b):
    if set(adj_a) != set(adj_b):
        return False
    for l in adj_a:
        if set(adj_a[l]) != set(adj_b[l]):
            return False
        for n in adj_a[l]:
            if set(adj_a[l][n]) != set(adj_b[l][n]):
                return False
    return True


def same_order(adj_a, adj_b):
    return same_sets(adj_a, adj_b) and all(adj_a[l][n] == adj_b[l][n] for l in adj_a for n in adj_a[l])


def edge_set(adj, layer):
    """Directed pairs on a layer."""
    return {(a, b) for a, lst in adj.get(layer, {}).items() for b in lst}


# ----------------------------------------------------------------------------- scan
def readable(levels):
    top = max(levels)
    n1 = sum(1 for l in levels if l >= 1)
    n2 = sum(1 for l in levels if l >= 2)
    ep = levels.index(top)
    return top in (2, 3) and 4 <= n1 <= 6 and n2 <= 2 and ep >= 3


def evaluate(values, hnsw_seed, variant=HEAP_VARIANT):
    """Build with hnswlib at ef=3 and ef=200 and simulate both. Returns per-ef results plus
    whether the two hnswlib graphs differ."""
    out = {}
    for ef in (EF_SMALL, EF_BIG):
        idx = build(values, ef, hnsw_seed)
        adj, st = dump_graph(idx)
        levels = list(st["element_levels"])
        sim = run_sim(values, levels, ef, variant)
        out[ef] = {"adj": adj, "st": st, "levels": levels, "sim": sim, "idx": idx,
                   "sets": same_sets(sim.adjacency(), adj), "order": same_order(sim.adjacency(), adj)}
    out["differ"] = not same_sets(out[EF_SMALL]["adj"], out[EF_BIG]["adj"])
    return out


def interest(sim):
    """Count the events the doc wants to show in an ef=3 build."""
    evict = refuse = early = prune = 0
    for lg in sim.logs:
        for lay in lg["layers"]:
            for stp in lay["steps"]:
                for e in stp["events"]:
                    if e["added"] and e["evicted"] is not None:
                        evict += 1
                    elif not e["added"]:
                        refuse += 1
            if lay["stop"] and lay["stop"]["rule"] == "early":
                early += 1
            prune += len(lay["prunes"])
    return evict, refuse, early, prune


def scan(max_seed=3000, max_shuffle=30):
    print(f"1. hnsw random_seed 1..{max_seed}: readable level pattern = top level 2 or 3, 4..6 nodes on "
          f"layer 1, <=2 on layer 2, entry point not among the first 3 inserts")
    probe = draw_points(0)
    readable_seeds = []
    for seed in range(1, max_seed + 1):
        lv = list(state_of(build(probe, EF_SMALL, seed))["element_levels"])
        if readable(lv):
            readable_seeds.append((seed, lv))
    print(f"   {len(readable_seeds)} readable seeds; first ten: {[s for s, _ in readable_seeds[:10]]}")
    print(f"2. for the first 10 readable seeds x shuffle seeds 0..{max_shuffle - 1}: build at ef=3 and ef=200, "
          f"simulate with both heap variants, compare")
    tally = {v: {"sets": 0, "order": 0} for v in ("floyd", "classic")}
    tried = 0
    first = None
    for seed, lv in readable_seeds[:10]:
        for sh in range(max_shuffle):
            tried += 1
            values = draw_points(sh)
            res = {v: evaluate(values, seed, v) for v in tally}
            for v in tally:
                tally[v]["sets"] += res[v][EF_SMALL]["sets"] and res[v][EF_BIG]["sets"]
                tally[v]["order"] += res[v][EF_SMALL]["order"] and res[v][EF_BIG]["order"]
            r = res[HEAP_VARIANT]
            ok = r[EF_SMALL]["order"] and r[EF_BIG]["order"]
            if first is None and ok and r["differ"]:
                first = (seed, sh, interest(r[EF_SMALL]["sim"]))
    print(f"   {tried} (seed, shuffle) pairs x 2 ef values each")
    for v in tally:
        print(f"   heap variant {v:8s}: simulation == hnswlib as sets in {tally[v]['sets']}/{tried}, "
              f"with identical stored order in {tally[v]['order']}/{tried}")
    if first:
        seed, sh, (ev, rf, ea, pr) = first
        print(f"3. first pair (scan order) where ef=3 and ef=200 differ and the {HEAP_VARIANT} simulation matches both: "
              f"HNSW_SEED={seed} SHUFFLE_SEED={sh}  (ef=3 build: {ev} evictions, {rf} refused admissions, "
              f"{ea} early stops, {pr} prunes)")
    else:
        print("3. no pair found")


# ----------------------------------------------------------------------------- markdown
def fmt_edges(adj, layer, values):
    es = edge_set(adj, layer)
    out = []
    for a, b in sorted(es, key=lambda e: (values[e[0]], values[e[1]])):
        if (b, a) in es:
            if values[a] < values[b]:
                out.append(f"{NAMES[a]}–{NAMES[b]}")
        else:
            out.append(f"{NAMES[a]}→{NAMES[b]}")
    return out


def true_nearest(values, i):
    return min((j for j in range(N) if j != i), key=lambda j: (abs(values[j] - values[i]), j))


def query_misses(values, idx, k, ef_query=200):
    """knn_query on every x.3 for x in 0..39 (no query is ever equidistant from two integers),
    compared with brute force. -> [(q, wanted labels, got labels)]"""
    idx.set_ef(ef_query)
    out = []
    for x in range(40):
        q = x + 0.3
        want = sorted(range(N), key=lambda i: abs(values[i] - q))[:k]
        got, _ = idx.knn_query(np.array([[q]], dtype=np.float32), k=k)
        got = [int(g) for g in got[0]]
        if set(got) != set(want):
            out.append((q, want, got))
    return out


def render(values, r, hnsw_seed, shuffle_seed):
    sim3 = r[EF_SMALL]["sim"]
    st = r[EF_SMALL]["st"]
    levels = r[EF_SMALL]["levels"]
    adj3, adj200 = r[EF_SMALL]["adj"], r[EF_BIG]["adj"]
    nm = sim3.nm
    L = []
    p = L.append

    p("<!-- BEGIN GENERATED by lab/hnsw_toy_20_ef3.py; hand-edit only outside these markers -->")
    p("")
    p("### Parameters")
    p("")
    p("| Parameter | Value | Where it comes from |")
    p("|---|---|---|")
    p(f"| `M` | {st['M']} | `init_index(M=2)`; links the new node keeps per layer (`getNeighborsByHeuristic2(…, M)`) |")
    p(f"| `Mmax` (list cap, layers ≥ 1) | {st['max_M']} | `maxM_ = M` |")
    p(f"| `Mmax0` (list cap, layer 0) | {st['max_M0']} | `maxM0_ = 2M` |")
    p(f"| `ef_construction` effective | {st['ef_construction']} | `max(ef_construction, M) = max(3, 2)` |")
    p(f"| `mL` | {st['mult']:.4f} | `1/ln M` |")
    p(f"| hnswlib `random_seed` | {hnsw_seed} | drives the level draws only; chosen by `--scan` for a readable hierarchy |")
    p(f"| shuffle seed (values and order) | {shuffle_seed} | `numpy.random.default_rng({shuffle_seed}).choice(1..40, 20, replace=False)` |")
    p(f"| entry point | {nm(st['enterpoint_node'])} | the first node to draw the highest level |")
    p(f"| top level | {st['max_level']} | |")
    p("")
    p("### Points, in insertion order")
    p("")
    p("| Insert # | " + " | ".join(str(i + 1) for i in range(N)) + " |")
    p("|---|" + "---|" * N)
    p("| Label | " + " | ".join(NAMES) + " |")
    p("| Value | " + " | ".join(str(v) for v in values) + " |")
    p("| Top layer | " + " | ".join(f"**{l}**" if l == max(levels) else str(l) for l in levels) + " |")
    p("")
    order = sorted(range(N), key=lambda i: values[i])
    p("The line the points live on, sorted by value: " + " · ".join(f"{NAMES[i]}{values[i]}" for i in order))
    p("")
    p("### Step-by-step build, `ef_construction = 3`")
    p("")
    p("Notation: `X(7)` is point X with value 7; `d=` is the distance to the point being inserted; a pool is "
      "written `[G3 F5 E11]`, nearest first, its last entry is the *worst*. Events where `ef_construction = 3` "
      "changed the walk are tagged **[ef=3]**; prunes of an existing node's list are tagged **[prune]**.")
    p("")
    for lg in sim3.logs:
        c, lvl = lg["node"], lg["level"]
        p(f"#### {c + 1}. {nm(c)}, level {lvl}" + (" — new entry point" if lg["new_entry"] else ""))
        p("")
        line = " · ".join(f"**{NAMES[i]}{values[i]}**" if i == c else f"{NAMES[i]}{values[i]}"
                          for i in order if i <= c)
        p(f"- Line so far: {line}")
        if lg["first"]:
            p(f"- Graph empty. {NAMES[c]} becomes the entry point, with a copy on "
              + (f"layers 0..{lvl}" if lvl else "layer 0") + " and no links.")
            p("")
            continue
        # upper layers: merge consecutive layers where the current node has no neighbours
        ups = lg["upper"]
        k = 0
        while k < len(ups):
            up = ups[k]
            if len(up["hops"]) == 1 and not up["hops"][0]["list"]:
                j = k
                while j + 1 < len(ups) and len(ups[j + 1]["hops"]) == 1 and not ups[j + 1]["hops"][0]["list"] \
                        and ups[j + 1]["hops"][0]["at"] == up["hops"][0]["at"]:
                    j += 1
                at = up["hops"][0]["at"]
                layers = f"Layer {up['layer']}" if j == k else f"Layers {up['layer']}–{ups[j]['layer']}"
                p(f"- {layers} (above {NAMES[c]}'s top; greedy, pool of 1): at {NAMES[at]} d={up['hops'][0]['curdist']}, "
                  f"no neighbours there. Slide down {NAMES[at]}'s pole.")
                k = j + 1
                continue
            parts = []
            for h in up["hops"]:
                lst = ", ".join(f"{nm(n)} d={dd}" for n, dd in h["list"]) or "no neighbours"
                if h["moved_to"] is not None:
                    parts.append(f"at {NAMES[h['at']]}: list {{{lst}}} → **hop to {NAMES[h['moved_to']]}** (d={h['curdist']})")
                else:
                    parts.append(f"at {NAMES[h['at']]}: list {{{lst}}} → nothing nearer than d={h['curdist']}")
            end = up["hops"][-1]["at"]
            p(f"- Layer {up['layer']} (above {NAMES[c]}'s top; greedy, pool of 1): " + "; ".join(parts) + f". Slide down {NAMES[end]}'s pole.")
            k += 1
        for lay in lg["layers"]:
            l = lay["layer"]
            p(f"- Layer {l}, pool of {sim3.ef}, start {nm(lay['start'])} d={lay['start_dist']} → pool `[{NAMES[lay['start']]}{lay['start_dist']}]`")
            for stp in lay["steps"]:
                evs = []
                for e in stp["events"]:
                    if e["added"]:
                        s = f"{nm(e['nb'])} d={e['d']} → `{sim3.pool_str(e['pool'])}`"
                        if e["evicted"] is not None:
                            s += f" **[ef=3]** pool full: evicted {NAMES[e['evicted'][1]]}({e['evicted'][0]})"
                        evs.append(s)
                    else:
                        evs.append(f"{nm(e['nb'])} d={e['d']} ≥ worst {e['lower']} and pool full: not admitted **[ef=3]**")
                p(f"  - expand {nm(stp['expand'])} (d={stp['cd']}): " + ("; ".join(evs) if evs else "no unvisited neighbours"))
            sp = lay["stop"]
            if sp["rule"] == "early":
                extra = ""
                if len(sp["unexpanded"]) > 1:
                    extra = f" (also never expanded: {', '.join(NAMES[n] for n in sp['unexpanded'] if n != sp['cand'])})"
                p(f"  - **stopped [ef=3]**: nearest unexpanded {nm(sp['cand'])} d={sp['cd']} is farther than worst-in-pool {sp['lower']} and the pool is full{extra}")
            else:
                p("  - stopped: nothing left to expand")
            sel = []
            for dcs in lay["select"]:
                if dcs["kept"]:
                    sel.append(f"keep {nm(dcs['cand'])} ({dcs['d']})" + (f" [{dcs['why']}]" if isinstance(dcs["why"], str) else ""))
                elif isinstance(dcs["why"], tuple):
                    k, dk = dcs["why"]
                    sel.append(f"reject {nm(dcs['cand'])}: closer to {NAMES[k]} ({dk}) than to {NAMES[c]} ({dcs['d']}): behind {NAMES[k]}")
                else:
                    sel.append(f"reject {nm(dcs['cand'])} ({dcs['d']}): M=2 already kept")
            p("  - select (M=2): " + "; ".join(sel))
            p("  - edges written: " + ", ".join(f"{NAMES[c]}–{NAMES[e]}" for e in lay["edges"]))
            for pr in lay["prunes"]:
                cap = sim3.maxM if l > 0 else sim3.maxM0
                dec = []
                for dcs in pr["decisions"]:
                    if dcs["kept"]:
                        dec.append(f"keep {NAMES[dcs['cand']]}({dcs['d']})")
                    elif isinstance(dcs["why"], tuple):
                        k, dk = dcs["why"]
                        dec.append(f"drop {NAMES[dcs['cand']]}({dcs['d']}): closer to {NAMES[k]} ({dk}) than to {NAMES[pr['node']]} ({dcs['d']})")
                    else:
                        dec.append(f"drop {NAMES[dcs['cand']]}({dcs['d']}): cap {cap} already kept")
                p(f"  - **[prune]** {nm(pr['node'])}'s layer-{l} list is at cap {cap} ({{{', '.join(NAMES[x] for x in pr['before'])}}}); "
                  f"heuristic over {NAMES[c]} + that list, distances measured from {NAMES[pr['node']]}, keep ≤ {cap}: "
                  + "; ".join(dec) + f" → {{{', '.join(NAMES[x] for x in pr['after'])}}}")
                lost = [x for x in pr["before"] if x not in pr["after"]]
                if lost:
                    p(f"    - {', '.join(NAMES[x] for x in lost)} still list{'s' if len(lost) == 1 else ''} {NAMES[pr['node']]}: "
                      f"{'that edge is' if len(lost) == 1 else 'those edges are'} now one-way")
        p("")

    p("### Final graph (`ef_construction = 3`), as read back from hnswlib")
    p("")
    p("`X–Y` is a mutual edge (each lists the other); `X→Y` is one-way (X lists Y; Y dropped X in a prune).")
    p("")
    for l in sorted(adj3, reverse=True):
        nodes = sorted(adj3[l], key=lambda i: values[i])
        p(f"- Layer {l} ({len(nodes)} node{'s' if len(nodes) > 1 else ''}: {' '.join(NAMES[i] for i in nodes)}): "
          + (", ".join(fmt_edges(adj3, l, values)) or "no edges"))
    p("")
    p("Per-node lists in stored order, sorted by value:")
    p("")
    p("| Node | Value | Top layer | Layer 0 | Layer 1 | Layer 2 | Layer 3 |")
    p("|---|---|---|---|---|---|---|")
    for i in order:
        row = [NAMES[i], str(values[i]), str(levels[i])]
        for l in range(4):
            lst = adj3.get(l, {}).get(i)
            row.append("" if lst is None else "{" + ", ".join(NAMES[x] for x in lst) + "}")
        p("| " + " | ".join(row) + " |")
    p("")
    p("```mermaid")
    p("flowchart TB")
    for l in sorted(adj3, reverse=True):
        nodes = sorted(adj3[l], key=lambda i: values[i])
        p(f'    subgraph L{l}["Layer {l} — {len(nodes)} of {N} vectors"]')
        p("        direction LR")
        for i in nodes:
            p(f'        {NAMES[i]}{l}(("{NAMES[i]} {values[i]}"))')
        es = edge_set(adj3, l)
        for a, b in sorted(es, key=lambda e: (values[e[0]], values[e[1]])):
            if (b, a) in es:
                if values[a] < values[b]:
                    p(f"        {NAMES[a]}{l} --- {NAMES[b]}{l}")
            else:
                p(f"        {NAMES[a]}{l} --> {NAMES[b]}{l}")
        p("    end")
    for i in order:
        if levels[i] >= 1:
            p("    " + " -.-> ".join(f"{NAMES[i]}{l}" for l in range(levels[i], -1, -1)))
    ep = st["enterpoint_node"]
    p("    classDef entry stroke:#e8590c,stroke-width:3px")
    p(f"    class {NAMES[ep]}{levels[ep]} entry")
    p("```")
    p("")

    p("### Verification: simulation vs hnswlib")
    p("")
    for ef in (EF_SMALL, EF_BIG):
        rr = r[ef]
        p(f"- `ef_construction = {ef}`: simulated per-node neighbour sets equal hnswlib's on every layer: "
          f"**{'yes' if rr['sets'] else 'NO'}**; stored list order identical too: **{'yes' if rr['order'] else 'no'}** "
          f"(heap model: libc++ `{HEAP_VARIANT}`)")
    p("")

    p("### `ef_construction = 3` vs `ef_construction = 200`: same points, same order, same levels")
    p("")
    if r["differ"]:
        for l in sorted(set(adj3) | set(adj200), reverse=True):
            e3, e200 = edge_set(adj3, l), edge_set(adj200, l)
            only3 = sorted(e3 - e200, key=lambda e: (values[e[0]], values[e[1]]))
            only200 = sorted(e200 - e3, key=lambda e: (values[e[0]], values[e[1]]))
            if not only3 and not only200:
                p(f"- Layer {l}: identical")
                continue
            p(f"- Layer {l}: only with ef=3: {{{', '.join(f'{NAMES[a]}→{NAMES[b]}' for a, b in only3) or '—'}}}; "
              f"only with ef=200: {{{', '.join(f'{NAMES[a]}→{NAMES[b]}' for a, b in only200) or '—'}}}")
        p("")
        p("The ef=200 graph, for reference:")
        p("")
        for l in sorted(adj200, reverse=True):
            p(f"- Layer {l}: " + (", ".join(fmt_edges(adj200, l, values)) or "no edges"))
    else:
        p("- Identical on every layer.")
    p("")
    p("#### Does each node's layer-0 list contain its true nearest neighbour (by value)?")
    p("")
    misses = []
    for i in range(N):
        nn = true_nearest(values, i)
        if nn not in adj3[0][i]:
            misses.append((i, nn, i in adj3[0][nn], adj3[0][i]))
    if misses:
        p("| Node | True nearest | In node's own list? | Reverse edge? | Node's layer-0 list (ef=3) | Node's layer-0 list (ef=200) |")
        p("|---|---|---|---|---|---|")
        for i, nn, rev, out_l in misses:
            p(f"| {nm(i)} | {nm(nn)} (d={abs(values[nn] - values[i])}) | no | "
              f"{'yes (one-way ' + NAMES[nn] + '→' + NAMES[i] + ')' if rev else 'no'} | "
              f"{{{', '.join(NAMES[x] for x in out_l)}}} | {{{', '.join(NAMES[x] for x in adj200[0][i])}}} |")
        p("")
        p(f"ef=3 graph: nodes missing their true nearest neighbour: {len(misses)} of {N} ({', '.join(NAMES[i] for i, *_ in misses)}).")
    else:
        p(f"ef=3 graph: nodes missing their true nearest neighbour: 0 of {N}.")
    m200 = [i for i in range(N) if true_nearest(values, i) not in adj200[0][i]]
    p(f"ef=200 graph, same check: {len(m200)} of {N}" + (f" ({', '.join(NAMES[i] for i in m200)})." if m200 else "."))
    p("")
    ev, rf, ea, pr = interest(sim3)
    p(f"Event counts in the ef=3 build: {ev} evictions from a full pool, {rf} refused admissions, {ea} early stops, {pr} prunes.")
    p("")
    p("#### Query-time check: `knn_query` with `ef = 200` on each index, 40 queries at 0.3, 1.3, …, 39.3")
    p("")
    p("| Index built with | k | Wrong answers | Which |")
    p("|---|---|---|---|")
    for ef in (EF_SMALL, EF_BIG):
        for k in (1, 2):
            miss = query_misses(values, r[ef]["idx"], k)
            which = "; ".join(f"q={q}: true {{{', '.join(nm(i) for i in want)}}}, got {{{', '.join(nm(i) for i in got)}}}"
                              for q, want, got in miss) or "—"
            p(f"| `ef_construction = {ef}` | {k} | {len(miss)} of 40 | {which} |")
    p("")
    p("<!-- END GENERATED -->")
    return "\n".join(L)


def main():
    if "--scan" in sys.argv:
        scan()
        return
    values = draw_points(SHUFFLE_SEED)
    r = evaluate(values, HNSW_SEED)
    for ef in (EF_SMALL, EF_BIG):
        if not r[ef]["sets"]:
            sys.exit(f"simulation does not reproduce hnswlib at ef_construction={ef}; fix the simulation")
    print(render(values, r, HNSW_SEED, SHUFFLE_SEED))


if __name__ == "__main__":
    main()
