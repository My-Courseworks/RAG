"""SIMULATION of ScaNN `tree(num_leaves, L).score_brute_force()` (pure IVF, no AH)
on the course's shared 187k x 256 generator. Reports recall@10 vs brute force
for a sweep of num_leaves_to_search.

NOT a run of the real library (no macOS wheel; see ivf_scann_toy.py docstring for
the source lines mirrored, commit bb18d9b0cb460fee1a080fa2eeb4109fe3805275).
What is mirrored: training on a 100k reservoir sample (expected_sample_size
default), RANDOM_INITIALIZATION, 12 Lloyd iterations with the min_cluster_size
clamp and random re-seeding of small clusters, single-level flat tree, exact
nearest-centroid assignment of every point (no database spilling), top-L centre
pick per query, exact scan inside opened leaves.
What is NOT mirrored: ScaNN's SIMD kernels (ms/query here is numpy cost, not
ScaNN latency), float32 assignment arithmetic, the MTRandom seed stream,
asymmetric hashing and reordering.
Run:  python -I ivf_scann_scale.py [--nlist 4096] [--iters 12]
"""

import argparse
import time

import numpy as np

N, D, NQ, K = 187_043, 256, 1000, 10
SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TRAINING_SAMPLE_SIZE = 100_000   # scann_builder.py:158 expected_sample_size
MIN_PARTITION_SIZE = 50          # scann_builder.py:159
CONVERGENCE_EPS = 1e-5
PERTURBATION = 1e-7


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def nearest_centers(x, centers, chunk=8192):
    """argmin and min of squared L2 to centers, chunked over x."""
    cn = (centers ** 2).sum(1)
    assign = np.empty(len(x), dtype=np.int32)
    dmin = np.empty(len(x), dtype=np.float32)
    for s in range(0, len(x), chunk):
        xb = x[s:s + chunk]
        d = cn[None, :] - 2.0 * (xb @ centers.T)
        a = d.argmin(1)
        assign[s:s + chunk] = a
        dmin[s:s + chunk] = d[np.arange(len(xb)), a] + (xb ** 2).sum(1)
    return assign, np.maximum(dmin, 0.0)


def kmeans_scann(sample, k, iters, min_cluster_size_cfg, rng):
    """Mirror of GmmUtils::ComputeKmeansClustering (see toy script for line refs)."""
    n = len(sample)
    if n < k:
        raise ValueError(f"Number of points ({n}) is less than the number of clusters ({k}).")
    min_cluster_size = min(min_cluster_size_cfg, n // k)
    centers = sample[rng.choice(n, size=k, replace=False)].astype(np.float32)
    old_means = np.full(k, -1.0)
    reinit_total = 0
    for it in range(iters + 1):
        assign, dmin = nearest_centers(sample, centers)
        sizes = np.bincount(assign, minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            new_means = np.bincount(assign, weights=dmin, minlength=k) / sizes
        small = sizes < min_cluster_size
        converged = (not small.any()) and np.all(np.abs(new_means - old_means) <= CONVERGENCE_EPS)
        print(f"  iter {it:2d}: mean within-cluster sqdist={np.nanmean(new_means):.3f} "
              f"small clusters(<{min_cluster_size})={int(small.sum())} empty={int((sizes == 0).sum())}"
              + ("  converged" if converged else ""), flush=True)
        if converged or it == iters:
            break
        old_means = new_means
        sums = np.zeros((k, D), dtype=np.float64)
        np.add.at(sums, assign, sample)
        nonzero = sizes > 0
        centers[nonzero] = (sums[nonzero] / sizes[nonzero, None]).astype(np.float32)
        # RandomReinitializeCenters: park each small cluster next to a big one
        big_points = np.where(~small[assign])[0]
        for c in np.where(small)[0]:
            ridx = big_points[rng.integers(0, len(big_points))]
            big = assign[ridx]
            centers[c] = centers[big] + PERTURBATION * (sample[ridx] - centers[big])
            reinit_total += 1
    return centers, reinit_total


def ground_truth(xb, xq, chunk=20_000):
    qn = (xq ** 2).sum(1)
    best_d = np.full((len(xq), K), np.inf, dtype=np.float32)
    best_i = np.zeros((len(xq), K), dtype=np.int64)
    for s in range(0, len(xb), chunk):
        b = xb[s:s + chunk]
        d = qn[:, None] - 2.0 * (xq @ b.T) + (b ** 2).sum(1)[None, :]
        cand_d = np.concatenate([best_d, d], 1)
        cand_i = np.concatenate([best_i, np.arange(s, s + len(b))[None, :].repeat(len(xq), 0)], 1)
        sel = np.argpartition(cand_d, K, axis=1)[:, :K]
        best_d = np.take_along_axis(cand_d, sel, 1)
        best_i = np.take_along_axis(cand_i, sel, 1)
    return best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()
    nlist = args.nlist

    print("SIMULATION of ScaNN tree(num_leaves=%d).score_brute_force()  (numpy, not the real library)" % nlist)
    xb, xq = make_data()
    t0 = time.perf_counter()
    gt = ground_truth(xb, xq)
    print(f"ground truth: {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    # partitioner_factory_base.cc:34-62: reservoir sample of expected_sample_size
    sample_idx = rng.choice(N, size=min(TRAINING_SAMPLE_SIZE, N), replace=False)
    centers, reinits = kmeans_scann(xb[sample_idx], nlist, args.iters, MIN_PARTITION_SIZE, rng)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    assign, _ = nearest_centers(xb, centers)          # database tokenization, 1 token/point
    t_assign = time.perf_counter() - t0
    sizes = np.bincount(assign, minlength=nlist)
    order = np.argsort(assign, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    print(f"build: train {t_train:.1f}s (+{t_assign:.1f}s assign), small-cluster re-seeds during training={reinits}")
    print(f"leaf sizes: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()} empty={(sizes == 0).sum()}")
    bytes_idx = centers.nbytes + order.nbytes + bounds.nbytes
    print(f"index memory (centroids + inverted lists, excl. raw vectors): {bytes_idx / 1e6:.1f} MB; "
          f"raw float32 vectors kept for exact leaf scan: {xb.nbytes / 1e6:.1f} MB")

    qn = (xq ** 2).sum(1)
    cn = (centers ** 2).sum(1)
    dcent = cn[None, :] - 2.0 * (xq @ centers.T)       # query tokenization, all centroids
    cent_order = np.argsort(dcent, axis=1)
    bn = (xb ** 2).sum(1)

    print(f"\n{'L':>4} {'recall@10':>10} {'ms/query':>9} {'cands/query':>12}")
    first95 = None
    for L in SWEEP:
        if L > nlist:
            break
        hits = 0
        ncand = 0
        t0 = time.perf_counter()
        for qi in range(NQ):
            leaves = cent_order[qi, :L]
            cand = np.concatenate([order[bounds[l]:bounds[l + 1]] for l in leaves])
            ncand += len(cand)
            d = bn[cand] - 2.0 * (xb[cand] @ xq[qi])
            top = cand[np.argpartition(d, min(K, len(cand) - 1))[:K]] if len(cand) > K else cand
            hits += len(np.intersect1d(top, gt[qi]))
        ms = (time.perf_counter() - t0) * 1000 / NQ
        recall = hits / (NQ * K)
        if first95 is None and recall >= 0.95:
            first95 = L
        print(f"{L:>4} {recall:>10.4f} {ms:>9.3f} {ncand / NQ:>12.0f}")
    print(f"\nsmallest num_leaves_to_search with recall@10 >= 0.95: {first95}")


if __name__ == "__main__":
    main()
