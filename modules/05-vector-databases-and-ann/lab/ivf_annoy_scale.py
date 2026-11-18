"""Annoy at scale: 187,043 x 256, n_trees in {10, 50, 100}, search_k sweep.

Same data generator as every other provider in this module (50 Gaussian topics,
sigma 0.35). Reports recall@10 vs brute-force L2, ms/query (single-threaded,
one get_nns_by_vector call per query), item-add time, build time, and index
file size, then the smallest (n_trees, search_k) reaching recall >= 0.95.

Run:  python -I ivf_annoy_scale.py [--n-jobs -1] [--trees 10,50,100] [--search-k 10,100,1000,3000,5000,10000,50000]
"""
import argparse
import os
import tempfile
import time

import numpy as np
from annoy import AnnoyIndex

N, D, NQ, K = 187_043, 256, 1000, 10


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def brute_force_topk(xb, xq, k, chunk=20_000):
    xb_sq = (xb.astype(np.float64) ** 2).sum(1)
    out = np.empty((len(xq), k), dtype=np.int64)
    for s in range(0, len(xq), chunk):
        q = xq[s:s + chunk].astype(np.float64)
        d = xb_sq[None, :] - 2.0 * q @ xb.astype(np.float64).T  # + ||q||^2, constant per row
        idx = np.argpartition(d, k, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(d[rows, idx], axis=1)
        out[s:s + chunk] = idx[rows, order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1, help="build threads; -1 = all cores (trees per thread then depend on core count)")
    ap.add_argument("--trees", default="10,50,100")
    ap.add_argument("--search-k", default="10,100,1000,3000,5000,10000,50000")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    trees = [int(t) for t in args.trees.split(",")]
    search_ks = [int(s) for s in args.search_k.split(",")]

    xb, xq = make_data()
    t0 = time.perf_counter()
    gt = brute_force_topk(xb, xq, K)
    print("data %d x %d, %d queries; brute-force ground truth in %.1fs" % (N, D, NQ, time.perf_counter() - t0))
    node_bytes = 16 + 4 * D  # Euclidean node: n_descendants, a, children[2], v[D]
    leaf_cap = (node_bytes - 8) // 4
    print("euclidean node = %d bytes; _K (items per leaf) = %d; build threads = %s"
          % (node_bytes, leaf_cap, os.cpu_count() if args.n_jobs == -1 else args.n_jobs))

    xq_lists = [row.tolist() for row in xq]
    tmp = tempfile.mkdtemp(prefix="annoy-scale-")
    rows = []
    first_95 = None
    for n_trees in trees:
        idx = AnnoyIndex(D, "euclidean")
        idx.set_seed(args.seed)
        t0 = time.perf_counter()
        for i in range(N):
            idx.add_item(i, xb[i])
        t_add = time.perf_counter() - t0
        t0 = time.perf_counter()
        idx.build(n_trees, n_jobs=args.n_jobs)
        t_build = time.perf_counter() - t0
        path = os.path.join(tmp, "scale_%d.ann" % n_trees)
        idx.save(path)
        size = os.path.getsize(path)
        n_nodes = size // node_bytes
        print("\nn_trees=%d: add %.1fs, build %.1fs, file %.1f MB = %d nodes (%d items + %d tree nodes), %.2fx raw vectors"
              % (n_trees, t_add, t_build, size / 1e6, n_nodes, N, n_nodes - N, size / (N * D * 4)))
        for qi in range(50):  # warm the mmapped file before timing
            idx.get_nns_by_vector(xq_lists[qi], K, search_k=search_ks[-1])
        print("  %-9s %-10s %-9s" % ("search_k", "recall@10", "ms/query"))
        for sk in search_ks:
            hits = 0
            t0 = time.perf_counter()
            for qi in range(NQ):
                res = idx.get_nns_by_vector(xq_lists[qi], K, search_k=sk)
                hits += len(set(res) & set(gt[qi].tolist()))
            ms = (time.perf_counter() - t0) * 1000.0 / NQ
            recall = hits / (NQ * K)
            print("  %-9d %-10.4f %-9.3f" % (sk, recall, ms))
            rows.append((n_trees, sk, recall, ms, t_build, size))
            if recall >= 0.95 and first_95 is None:
                first_95 = (n_trees, sk, recall, ms)
        idx.unload()

    print("\nsummary (n_trees, search_k, recall@10, ms/query, build s, file MB):")
    for r in rows:
        print("  %4d %6d %.4f %8.3f %7.1f %8.1f" % (r[0], r[1], r[2], r[3], r[4], r[5] / 1e6))
    if first_95:
        print("smallest (n_trees, search_k) reaching recall@10 >= 0.95 in sweep order: n_trees=%d search_k=%d (recall %.4f, %.3f ms/query)" % first_95)
    else:
        print("no (n_trees, search_k) in the sweep reached recall@10 >= 0.95")
    best = [r for r in rows if r[2] >= 0.95]
    if best:
        b = min(best, key=lambda r: r[3])
        print("fastest config with recall >= 0.95: n_trees=%d search_k=%d (recall %.4f, %.3f ms/query)" % (b[0], b[1], b[2], b[3]))


if __name__ == "__main__":
    main()
