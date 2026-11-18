"""FreshVamana freshness test for M5: does recall hold across delete / consolidate /
insert cycles, and does alpha > 1 matter?

Uses diskannpy.DynamicMemoryIndex (DiskANN tag 0.7.0): `batch_insert`,
`mark_deleted` (lazy delete), `consolidate_delete` (Index::consolidate_deletes).
Same 50-topic generator as lab/diskann_scale.py. Each cycle:
    1. mark a random `--frac` of the live points deleted,
    2. consolidate_delete(),
    3. insert the same number of points the index has never seen,
    4. recall@10 against brute-force L2 over the points live right now.
The schedule (which points die, which arrive) is identical for every alpha.

Two diagnostics are printed with the recall:
  - worst topic: the lowest per-topic recall@10 at the largest L (the corpus is
    50 separated topics and every search starts at one node, so a topic whose
    bridging edges are lost shows up here before it shows in the mean);
  - between step 1 and step 2 (lazy deletes pending), how many deleted ids come
    back at L = 10 and at L = 50, and how many result rows at L = 10 are not
    sorted by distance. `search_with_tags` writes only the live entries of the
    list (src/index.cpp:2187-2217) and the binding ignores the count it returns
    (python/src/dynamic_memory_index.cpp:116-150), so a row with fewer than k
    live entries keeps whatever the output buffer held.

Run inside the container described in supplementary/diskann/README.md:
    python -I diskann_fresh.py [--n 100000] [--cycles 20] [--frac 0.05] [--alphas 1.2,1.0]
Synthetic data: the absolute numbers are not LogiTrack's.
"""
import argparse
import os
import sys
import time

import numpy as np

D, NQ, K = 256, 1000, 10
R, L_BUILD = 32, 100
LS = (10, 15, 20, 30, 50)


def make_data(total):
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, total + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((total + NQ, D), dtype=np.float32)
    return x[:total], x[total:], topic[total:]


def ground_truth(xb, live_idx, xq):
    sub = xb[live_idx]
    bn = (sub * sub).sum(1)
    out = np.empty((len(xq), K), dtype=np.int64)
    for s in range(0, len(xq), 100):
        d = bn[None, :] - 2.0 * (xq[s:s + 100] @ sub.T)
        part = np.argpartition(d, K, axis=1)[:, :K]
        order = np.argsort(np.take_along_axis(d, part, 1), axis=1)
        out[s:s + 100] = live_idx[np.take_along_axis(part, order, 1)]
    return out


def quiet(fn, *a, **kw):
    sys.stdout.flush()
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    saved2 = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.dup2(saved2, 2)
        os.close(saved)
        os.close(saved2)
        os.close(devnull)


def recalls(idx, xq, gt, qtopic, dead=None):
    """Tags are point index + 1 (tag 0 is reserved by the library). Returns mean
    recall per L, the worst per-topic recall at the largest L, and (when `dead`
    is given) deleted ids returned at L=10 / L=50 and unsorted rows at L=10."""
    out, diag = {}, {}
    for L in LS:
        res = quiet(idx.batch_search, xq, k_neighbors=K, complexity=L, num_threads=0)
        ids = np.asarray(res.identifiers).astype(np.int64) - 1
        per_q = np.array([len(set(a.tolist()) & set(b.tolist())) / K for a, b in zip(ids, gt)])
        out[L] = float(per_q.mean())
        if L == max(LS):
            diag["worst_topic"] = float(min(per_q[qtopic == t].mean() for t in np.unique(qtopic)))
        if dead is not None and L in (10, 50):
            diag[f"dead{L}"] = int(np.isin(ids, dead).sum())
            if L == 10:
                d = np.asarray(res.distances)
                diag["unsorted10"] = int((np.diff(d, axis=1) < 0).any(axis=1).sum())
    return out, diag


def run(alpha, xb, xq, qtopic, n, cycles, per_cycle, seed):
    import diskannpy
    rng = np.random.default_rng(seed)
    idx = quiet(diskannpy.DynamicMemoryIndex, distance_metric="l2", vector_dtype=np.float32, dimensions=D,
                max_vectors=n + per_cycle, complexity=L_BUILD, graph_degree=R, alpha=alpha, num_threads=0,
                initial_search_complexity=max(LS))
    live = np.zeros(len(xb), dtype=bool)
    t = time.perf_counter()
    rc = quiet(idx.batch_insert, xb[:n], np.arange(1, n + 1, dtype=np.uint32), num_threads=0)
    assert int(np.count_nonzero(rc)) == 0, "initial inserts failed"
    live[:n] = True
    build_s = time.perf_counter() - t
    gt = ground_truth(xb, np.flatnonzero(live), xq)
    rec, dg = recalls(idx, xq, gt, qtopic)
    print(f"\nalpha={alpha}  R={R} L_build={L_BUILD}  n={n}  {per_cycle} deleted + {per_cycle} inserted per cycle "
          f"({100 * per_cycle / n:.0f}%)  built by batch_insert in {build_s:.0f} s")
    print(" cycle  " + " ".join(f"L={L:<5d}" for L in LS)
          + f" worst-topic(L={max(LS)}) | pending deletes: dead@L10 dead@L50 unsorted-rows@L10 | consolidate-s insert-s")
    print(f"   0   " + " ".join(f"{rec[L]:.4f} " for L in LS) + f"    {dg['worst_topic']:.3f}")
    rows = [dict(cycle=0, **{f"L{L}": rec[L] for L in LS})]
    nxt = n
    for c in range(1, cycles + 1):
        victims = rng.choice(np.flatnonzero(live), per_cycle, replace=False)
        for v in victims:
            idx.mark_deleted(int(v) + 1)
        live[victims] = False
        # lazy delete: deleted points still steer the walk but must not be returned
        gt_mid = ground_truth(xb, np.flatnonzero(live), xq)
        _, mid = recalls(idx, xq, gt_mid, qtopic, victims)
        t = time.perf_counter()
        quiet(idx.consolidate_delete)
        cons_s = time.perf_counter() - t
        new = np.arange(nxt, nxt + per_cycle)
        nxt += per_cycle
        t = time.perf_counter()
        rc = quiet(idx.batch_insert, xb[new], (new + 1).astype(np.uint32), num_threads=0)
        ins_s = time.perf_counter() - t
        assert int(np.count_nonzero(rc)) == 0, f"cycle {c}: {int(np.count_nonzero(rc))} inserts failed"
        live[new] = True
        gt = ground_truth(xb, np.flatnonzero(live), xq)
        rec, dg = recalls(idx, xq, gt, qtopic)
        rows.append(dict(cycle=c, **{f"L{L}": rec[L] for L in LS}))
        print(f"  {c:2d}   " + " ".join(f"{rec[L]:.4f} " for L in LS) + f"    {dg['worst_topic']:.3f}"
              + f"            |          {mid['dead10']:5d}    {mid['dead50']:5d}      {mid['unsorted10']:5d}"
              + f"           |     {cons_s:5.1f}      {ins_s:5.1f}")
    print("  change cycle 0 -> last: " + "  ".join(f"L={L}: {rows[-1][f'L{L}'] - rows[0][f'L{L}']:+.4f}" for L in LS))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--alphas", default="1.2,1.0")
    ap.add_argument("--seed", type=int, default=1, help="seed of the delete schedule")
    a = ap.parse_args()
    per_cycle = int(a.n * a.frac)
    xb, xq, qtopic = make_data(a.n + a.cycles * per_cycle)
    print(f"diskannpy DynamicMemoryIndex freshness test: {a.cycles} cycles, D={D}, NQ={NQ}, K={K}, "
          f"cpus={os.cpu_count()}")
    for alpha in [float(s) for s in a.alphas.split(",")]:
        run(alpha, xb, xq, qtopic, a.n, a.cycles, per_cycle, seed=a.seed)


if __name__ == "__main__":
    main()
