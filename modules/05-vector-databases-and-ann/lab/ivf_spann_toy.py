"""SIMULATION of Microsoft SPANN (SPTAG) on the course's eight 1-D values.

This is NOT a run of SPTAG. SPTAG could not be built on this machine (no Boost,
no cmake, no SWIG). Every step below mirrors a named piece of the C++ source at
commit cafdd0abf261a8b26fbbb5ece132a46c823572f5 (microsoft/SPTAG, 2026-09-29):

  balanced k-means ........ AnnService/inc/Core/Common/BKTree.h
                              KmeansAssign      L230-341  (dist + lambda*count penalty)
                              RefineLambda      L136-153
                              RefineCenters     L155-207  (mean; empty -> farthest pt of biggest)
                              InitCenters       L347-369  (3 random tries, keep best)
                              TryClustering     L372-443  (<=100 iters, then snap each
                                                           centre to its closest data point L421-424)
                              DynamicFactorSelect L446-495 (lambdaFactor sweep 1e-3..1e3)
  BKT build ............... BKTree.h BuildTrees L579-668 (leaf if n <= BKTLeafSize)
  head selection .......... src/Core/SPANN/SPANNIndex.cpp
                              SelectHeadAdjustOptions       L733-776
                              SelectHeadDynamicallyInternal L778-824
                              SelectHeadDynamically         L826-899
  posting assignment ...... src/Core/VectorIndex.cpp ApproximateRNG L1140-1250 (RNG test L1210-1222)
  posting cut ............. inc/Core/SPANN/ExtraStaticSearcher.h L781-846
  query ................... SPANNIndex.cpp Index::SearchIndex L293-395 (MaxDistRatio prune L324-330)
                            ExtraStaticSearcher.h ProcessPosting L120-129 (dedupe + full-vector dist)

Distances are squared L2, as SPTAG computes them for DistCalcMethod::L2.
Run:  python -I ivf_spann_toy.py
"""
from __future__ import annotations

import math
import numpy as np

VALUES = np.array([1, 2, 4, 7, 8, 11, 14, 15], dtype=np.float32)
NAMES = list("ABCDEFGH")
MAXDIST = np.float32(np.finfo(np.float32).max)


def d2(a: float, b: float) -> float:
    return float((a - b) * (a - b))


# --------------------------------------------------------------------------
# Balanced k-means (BKTree.h TryClustering and helpers), 1-D specialisation
# --------------------------------------------------------------------------
def kmeans_assign(x, idx, centers, counts, lam, update_centers):
    """BKTree.h KmeansAssign L230-341. Returns dict of the per-cluster accumulators."""
    K = len(centers)
    label = {}
    new_counts = np.zeros(K, dtype=int)
    new_centers = np.zeros(K)
    weighted = np.zeros(K)
    cluster_dist = np.full(K, -MAXDIST if update_centers else MAXDIST, dtype=float)
    cluster_idx = np.full(K, -1, dtype=int)
    total = 0.0
    for i in idx:
        best_k, best = 0, float("inf")
        for k in range(K):
            dist = d2(x[i], centers[k]) + lam * counts[k]
            if dist < best:
                best_k, best = k, dist
        label[i] = best_k
        new_counts[best_k] += 1
        weighted[best_k] += best
        total += best
        if update_centers:
            new_centers[best_k] += x[i]
            if best > cluster_dist[best_k]:          # track FARTHEST point
                cluster_dist[best_k] = best
                cluster_idx[best_k] = i
        else:
            if best <= cluster_dist[best_k]:         # track CLOSEST point
                cluster_dist[best_k] = best
                cluster_idx[best_k] = i
    return dict(total=total, label=label, new_counts=new_counts, new_centers=new_centers,
                weighted=weighted, cluster_dist=cluster_dist, cluster_idx=cluster_idx)


def refine_lambda(a, counts, size):
    """BKTree.h RefineLambda L136-153."""
    maxc = -1
    maxn = 0
    for k in range(len(counts)):
        if counts[k] > maxn and a["new_counts"][k] > 0:
            maxc, maxn = k, counts[k]
    if maxc < 0:
        return 0.0
    avg = a["weighted"][maxc] / a["new_counts"][maxc]
    lam = (a["cluster_dist"][maxc] - avg) / size
    return max(lam, 0.0)


def refine_centers(x, a, centers, counts):
    """BKTree.h RefineCenters L155-207 (1-D, L2)."""
    K = len(centers)
    maxc, maxn = -1, 0
    for k in range(K):
        if counts[k] > maxn and a["new_counts"][k] > 0 and a["cluster_idx"][k] >= 0 \
                and d2(x[a["cluster_idx"][k]], centers[k]) > 1e-6:
            maxc, maxn = k, counts[k]
    new = np.zeros(K)
    diff = 0.0
    for k in range(K):
        if counts[k] == 0:
            new[k] = x[a["cluster_idx"][maxc]] if maxc != -1 else centers[k]
        else:
            new[k] = a["new_centers"][k] / counts[k]
        diff += d2(new[k], centers[k])
    return new, diff


def try_clustering(x, idx, K, rng, samples=1000, lambda_factor=100.0, log=None):
    """BKTree.h TryClustering L372-443. Returns (centers_as_data_indices, labels, count_std)."""
    idx = list(idx)
    n = len(idx)
    batch = min(samples, n)
    # InitCenters L347-369: 3 random inits (draws WITH replacement), keep lowest total dist
    adjusted_lambda, best_total, best_centers, counts = 0.0, float("inf"), None, np.zeros(K, dtype=int)
    for _ in range(3):
        centers = np.array([x[idx[rng.integers(0, n)]] for _ in range(K)], dtype=float)
        a = kmeans_assign(x, idx[:batch], centers, np.zeros(K, dtype=int), 0.0, True)
        if a["total"] < best_total:
            best_total, best_centers, counts = a["total"], centers.copy(), a["new_counts"].copy()
            adjusted_lambda = refine_lambda(a, counts, batch)
    original_lambda = 1.0 * 1.0 / lambda_factor / batch          # GetBase<float>()==1, L383
    lam = min(adjusted_lambda, original_lambda)
    centers = best_centers
    min_total, no_improve = float("inf"), 0
    for it in range(100):
        rng.shuffle(idx)
        a = kmeans_assign(x, idx[:batch], centers, counts, lam, True)
        counts = a["new_counts"].copy()
        if a["total"] < min_total:
            min_total, no_improve = a["total"], 0
        else:
            no_improve += 1
        centers, diff = refine_centers(x, a, centers, counts)
        if diff < 1e-3 or no_improve >= 5:
            break
    # final: lambda=0, no centre update -> cluster_idx = CLOSEST point; snap centres onto data (L419-424)
    a = kmeans_assign(x, idx, centers, counts, 0.0, False)
    snapped = centers.copy()
    snapped_idx = [-1] * K
    for k in range(K):
        if a["cluster_idx"][k] != -1:
            snapped[k] = x[a["cluster_idx"][k]]
            snapped_idx[k] = int(a["cluster_idx"][k])
    a = kmeans_assign(x, idx, snapped, counts, 0.0, False)
    counts = a["new_counts"]
    avg = n / K
    count_std = math.sqrt(sum((c - avg) ** 2 for c in counts) / K) / avg
    if log is not None:
        log.append(f"      lambdaFactor={lambda_factor:g}: lambda=min({original_lambda:.4g},{adjusted_lambda:.4g})"
                   f"={lam:.4g}, iters={it + 1}, counts={counts.tolist()}, std/avg={count_std:.3f}, "
                   f"heads={[NAMES[i] if i >= 0 else '-' for i in snapped_idx]}")
    return snapped_idx, a["label"], count_std


def dynamic_factor_select(x, idx, K, seed, log):
    """BKTree.h DynamicFactorSelect L446-495: sweep lambdaFactor, keep lowest count std."""
    best_f, best_std = 100.0, float("inf")
    f = 0.001
    while f <= 1000.0 + 1e-3:
        rng = np.random.default_rng(seed)
        _, _, cs = try_clustering(x, idx, K, rng, lambda_factor=f, log=log)
        if cs < best_std:
            best_f, best_std = f, cs
        f *= 10
    return best_f


# --------------------------------------------------------------------------
# BKT build + SelectHeadDynamically (SPANNIndex.cpp L778-899)
# --------------------------------------------------------------------------
class Node:
    def __init__(self, centerid):
        self.centerid = centerid      # data id, or N for the root (BKTree.h L614)
        self.children = []


def build_bkt(x, idx, K, leaf_size, lambda_factor, rng, node):
    """BKTree.h BuildTrees L617-664 (single tree, dynamicK on)."""
    n = len(idx)
    if n <= leaf_size:
        for i in idx:
            node.children.append(Node(int(i)))
        return
    dk = max(2, min(n // leaf_size + 1, K))
    snapped_idx, label, _ = try_clustering(x, idx, dk, rng, lambda_factor=lambda_factor)
    groups = [[i for i in idx if label[i] == k] for k in range(dk)]
    if sum(1 for g in groups if g) <= 1:
        for i in idx:
            node.children.append(Node(int(i)))
        return
    for k, g in enumerate(groups):
        if not g:
            continue
        child = Node(snapped_idx[k])                  # cluster centre = a real data point
        node.children.append(child)
        rest = [i for i in g if i != snapped_idx[k]]  # L656-659: centre stored at node, rest recurse
        if rest:
            build_bkt(x, rest, K, leaf_size, lambda_factor, rng, child)


def select_internal(node, root_centerid, select_th, split_th, split_factor, selected):
    """SPANNIndex.cpp SelectHeadDynamicallyInternal L778-824."""
    children = []
    size = 1
    for c in node.children:
        cs = select_internal(c, root_centerid, select_th, split_th, split_factor, selected)
        if cs > 0:
            children.append((c, cs))
            size += cs
    if size >= select_th:
        if node.centerid < root_centerid:
            selected.append(node.centerid)
        if size > split_th:
            children.sort(key=lambda p: -p[1])
            cnt = int(math.ceil(size / split_factor) + 0.5)
            for c, _ in children[:cnt]:
                selected.append(c.centerid)
        return 0
    return size


def select_head_dynamically(root, n, ratio, select_th, split_th, split_factor, log):
    """SPANNIndex.cpp SelectHeadDynamically L826-899: search (select,split) thresholds for |heads|/N ~ ratio."""
    best = (select_th, split_th)
    min_diff = 100.0
    for sel in range(2, select_th + 1):
        l, r = split_factor, split_th
        while l < r - 1:
            mid = (l + r) // 2
            s = []
            select_internal(root, root.centerid, sel, mid, split_factor, s)
            s = sorted(set(s))
            diff = len(s) / n - ratio
            log.append(f"      SelectThreshold={sel} SplitThreshold={mid}: {len(s)} heads, diff={diff * 100:+.1f}%")
            if abs(diff) < min_diff:
                min_diff, best = abs(diff), (sel, mid)
            if diff > 0:
                l = mid
            else:
                r = mid
    s = []
    select_internal(root, root.centerid, best[0], best[1], split_factor, s)
    return sorted(set(s)), best


# --------------------------------------------------------------------------
# Posting assignment with RNG replication (VectorIndex.cpp ApproximateRNG L1140-1250)
# --------------------------------------------------------------------------
def assign_postings(x, heads, replica_count, rng_factor, internal_result_num, log):
    postings = {h: [] for h in heads}
    for i in range(len(x)):
        if i in heads:                                   # ExcludeHead=true: heads are not in any posting
            continue
        cand = sorted(heads, key=lambda h: d2(x[i], x[h]))[:internal_result_num]
        accepted = []
        why = []
        for h in cand:
            if len(accepted) >= replica_count:
                break
            dxh = d2(x[i], x[h])
            ok = True
            for j in accepted:
                hh = d2(x[h], x[j])
                if rng_factor * hh < dxh:            # L1217: reject if an accepted head is closer to h than x is
                    ok = False
                    why.append(f"{NAMES[h]} rejected: d2({NAMES[h]},{NAMES[j]})={hh:g} < d2(x,{NAMES[h]})={dxh:g}")
                    break
            if ok:
                accepted.append(h)
                why.append(f"{NAMES[h]} accepted (d2={dxh:g})")
        for h in accepted:
            postings[h].append(i)
        log.append(f"      {NAMES[i]}={x[i]:g}: " + "; ".join(why))
    return postings


def posting_cut(postings, x, limit):
    """ExtraStaticSearcher.h L781-846: keep only the `limit` closest elements per posting."""
    for h, lst in postings.items():
        lst.sort(key=lambda i: (d2(x[i], x[h]), i))     # EdgeCompare order: (node, distance, tonode)
        del lst[limit:]
    return postings


# --------------------------------------------------------------------------
# Query (SPANNIndex.cpp L293-395 + ExtraStaticSearcher.h ProcessPosting)
# --------------------------------------------------------------------------
def spann_search(x, heads, postings, q, k, search_internal_result_num, max_dist_ratio, log):
    head_res = sorted(heads, key=lambda h: d2(q, x[h]))[:search_internal_result_num]   # head index (exact here)
    limit = d2(q, x[head_res[0]]) * max_dist_ratio
    opened = []
    for h in head_res:                                   # L324-330
        if limit > 0.1 and d2(q, x[h]) > limit:
            log.append(f"      head {NAMES[h]} pruned: d2={d2(q, x[h]):g} > limitDist={limit:g}")
            break
        opened.append(h)
    cands = {h: d2(q, x[h]) for h in opened}             # heads themselves are results (L335-336)
    scanned = 0
    seen = set()
    for h in opened:                                     # one posting read per head
        for i in postings[h]:
            scanned += 1
            if i in seen:                                # m_deduper L125
                continue
            seen.add(i)
            cands[i] = d2(q, x[i])
    ranked = sorted(cands, key=lambda i: (cands[i], i))[:k]
    log.append(f"      heads from head-index: {[NAMES[h] for h in head_res]}, opened: {[NAMES[h] for h in opened]}, "
               f"posting elements scanned: {scanned}, candidates: {sorted(NAMES[i] for i in cands)}")
    return ranked


# --------------------------------------------------------------------------
# Plain IVF (FAISS-style Lloyd's k-means on means, single assignment) for contrast
# --------------------------------------------------------------------------
def plain_ivf(x, nlist, seed, iters=25):
    rng = np.random.default_rng(seed)
    cent = np.array(rng.choice(x, nlist, replace=False), dtype=float)
    for _ in range(iters):
        lab = np.array([int(np.argmin([(v - c) ** 2 for c in cent])) for v in x])
        cent = np.array([x[lab == k].mean() if np.any(lab == k) else cent[k] for k in range(nlist)])
    order = np.argsort(cent)
    cent = cent[order]
    lab = np.array([int(np.where(order == l)[0][0]) for l in lab])
    return cent, lab


def plain_ivf_search(x, cent, lab, q, k, nprobe):
    probe = np.argsort([(q - c) ** 2 for c in cent])[:nprobe]
    cands = [i for i in range(len(x)) if lab[i] in probe]
    return sorted(cands, key=lambda i: ((q - x[i]) ** 2, i))[:k], probe


def fmt(ids, x):
    return ", ".join(f"{NAMES[i]}={x[i]:g}" for i in ids)


def main():
    x = VALUES
    n = len(x)
    seed = 0
    print("SIMULATION of SPANN on A..H =", ", ".join(f"{NAMES[i]}={x[i]:g}" for i in range(n)))
    print("(mirrors SPTAG @ cafdd0a; NOT a run of the real library; squared-L2 distances)\n")

    # ---- Step 1: SelectHeadAdjustOptions
    want_heads = 3
    ratio = want_heads / n
    bkt_k = min(32, want_heads)                        # L752-757: BKTKmeansK capped at head count
    select_th, split_th, split_factor = 6, 25, 5       # defaults (ParameterDefinitionList.h L57-59)
    print(f"Step 1  SelectHeadAdjustOptions: Count={want_heads} -> Ratio={ratio:.3f}; BKTKmeansK 32->{bkt_k}; "
          f"SelectThreshold={select_th}, SplitThreshold={split_th}, SplitFactor={split_factor}")

    # ---- Step 2a: faithful defaults (BKTLeafSize=8) -> root is a leaf
    log = []
    root = Node(n)
    build_bkt(x, list(range(n)), bkt_k, 8, 100.0, np.random.default_rng(seed), root)
    heads_default, th = select_head_dynamically(root, n, ratio, select_th, split_th, split_factor, log)
    print(f"\nStep 2a BKT with default BKTLeafSize=8: N=8 <= 8 so the root is ONE leaf with 8 children; "
          f"no k-means runs at all.")
    print(f"        SelectHeadDynamically threshold search picked (Select,Split)={th} -> "
          f"{len(heads_default)} heads: {fmt(heads_default, x)}  (wanted 3; the subtree-size rule "
          f"cannot produce 3 from a flat tree, and which 2 it picks is tree order, i.e. arbitrary)")

    # ---- Step 2b: BKTLeafSize=1 so the root actually clusters; DynamicFactorSelect for lambda
    print(f"\nStep 2b BKTLeafSize=1 (so clustering happens). DynamicFactorSelect sweep at the root (K={bkt_k}):")
    sweep = []
    best_f = dynamic_factor_select(x, list(range(n)), bkt_k, seed, sweep)
    print("\n".join(sweep))
    print(f"        best lambdaFactor = {best_f:g} (lowest count std/avg)")
    rng = np.random.default_rng(seed)
    snapped_idx, label, _ = try_clustering(x, list(range(n)), bkt_k, rng, lambda_factor=best_f)
    print(f"        root split: centres snapped to data points -> {fmt(snapped_idx, x)}")
    for k in range(bkt_k):
        members = [i for i in range(n) if label[i] == k]
        print(f"          cluster {k}: head {NAMES[snapped_idx[k]]}={x[snapped_idx[k]]:g}, members {fmt(members, x)}")
    root = Node(n)
    build_bkt(x, list(range(n)), bkt_k, 1, best_f, np.random.default_rng(seed), root)
    log = []
    heads, th = select_head_dynamically(root, n, ratio, select_th, split_th, split_factor, log)
    print(f"        SelectHeadDynamically on that tree -> (Select,Split)={th}, {len(heads)} heads: {fmt(heads, x)}")
    if len(heads) != want_heads:
        print(f"        NOTE: still not {want_heads}; the subtree-size selector is tuned for N>>SelectThreshold. "
              f"For the rest of the toy we take the 3 root-level cluster centres as heads "
              f"(the rule that produces heads at scale: balanced k-means centre snapped to a data point).")
        heads = sorted(snapped_idx)
    print(f"\nHEADS = {fmt(heads, x)}  (real data points, kept in RAM in the head index; "
          f"ExcludeHead=true so they are NOT stored in any posting)")

    # ---- Step 3: posting assignment with ReplicaCount=2, RNGFactor=1.0
    replica, rng_factor, internal = 2, 1.0, 64
    print(f"\nStep 3  ApproximateRNG posting assignment: ReplicaCount={replica}, RNGFactor={rng_factor}, "
          f"InternalResultNum={internal} (candidates from head index; only {len(heads)} exist)")
    log = []
    postings = assign_postings(x, heads, replica, rng_factor, internal, log)
    print("\n".join(log))
    vec_info = 1 * 4 + 4                                 # d=1 float + int id (ExtraStaticSearcher.h L673)
    limit = 3 * 4096 // vec_info                         # PostingPageLimit=3 pages
    postings = posting_cut(postings, x, limit)
    print(f"        posting cut: PostingPageLimit=3 pages x 4096 B / {vec_info} B per element = {limit} max per list; nothing cut")
    for h in heads:
        print(f"        posting[{NAMES[h]}={x[h]:g}] = [{fmt(postings[h], x)}]  ({len(postings[h])} elems, "
              f"{len(postings[h]) * vec_info} B -> 1 page on SSD)")
    replicated = [i for i in range(n) if i not in heads and sum(i in p for p in postings.values()) > 1]
    print(f"        replicated values (in 2 postings): {fmt(replicated, x) or 'none'}")
    singles = [i for i in range(n) if i not in heads and sum(i in p for p in postings.values()) == 1]
    print(f"        single-posting values (2nd head failed RNG test): {fmt(singles, x) or 'none'}")

    # ---- Step 4: plain IVF for contrast
    cent, lab = plain_ivf(x, 3, seed)
    print(f"\nStep 4  Plain IVF (Lloyd's k-means, nlist=3, means not data points, single assignment):")
    print(f"        centroids = {[round(float(c), 3) for c in cent]}")
    for k in range(3):
        print(f"        list {k} (c={cent[k]:.3f}): {fmt([i for i in range(n) if lab[i] == k], x)}")

    # ---- Step 5: queries
    for q, k in ((9.5, 2), (12.0, 1)):
        exact = sorted(range(n), key=lambda i: (d2(q, x[i]), i))[:k]
        print(f"\nStep 5  Query q={q}, k={k}. Exact: {fmt(exact, x)}")
        for sirn in (1, 2):
            for mdr in (10000.0, 1.5):
                log = []
                res = spann_search(x, heads, postings, q, k, sirn, mdr, log)
                hit = "HIT " if set(res) == set(exact) or (k == 1 and d2(q, x[res[0]]) == d2(q, x[exact[0]])) else "MISS"
                print(f"   SPANN SearchInternalResultNum={sirn} MaxDistRatio={mdr:g}: {hit} -> {fmt(res, x)}")
                print("\n".join(log))
        for nprobe in (1, 2):
            res, probe = plain_ivf_search(x, cent, lab, q, k, nprobe)
            hit = "HIT " if set(res) == set(exact) or (k == 1 and d2(q, x[res[0]]) == d2(q, x[exact[0]])) else "MISS"
            print(f"   plain IVF nprobe={nprobe}: {hit} -> {fmt(res, x)}  (lists opened: {probe.tolist()})")


if __name__ == "__main__":
    main()
