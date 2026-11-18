"""SIMULATION of ScaNN's tree (k-means partitioning) + brute-force leaf scoring
on the course's eight 1-D values.

This is NOT a run of the real library. `pip install scann` ships manylinux
x86_64/aarch64 wheels only (no macOS), so the toy mirrors the C++ instead.
Source read at google-research/google-research commit
bb18d9b0cb460fee1a080fa2eeb4109fe3805275, directory `scann/`:

  scann/scann_ops/py/scann_builder.py:153-227   .tree() defaults -> proto
  scann/partitioning/partitioner_factory_base.cc:34-70  training sample draw
  scann/trees/kmeans_tree/kmeans_tree_node.cc:164-269   KMeansTreeNode::Train
  scann/utils/gmm_utils.cc:1006-1038            RandomInitializeCenters
  scann/utils/gmm_utils.cc:1068-1299            ComputeKmeansClustering loop
  scann/utils/gmm_utils.cc:1438-1516            RecomputeCentroidsSimple
  scann/utils/gmm_utils.cc:1633-1689            RandomReinitializeCenters
  scann/trees/kmeans_tree/kmeans_tree_node.cc:129-160   top-L centre pick
  scann/tree_x_hybrid/tree_ah_hybrid_residual.cc:999-1060  per-leaf scan

Deviations from the real thing: numpy RNG instead of MTRandom(seed + kDeterministicSeed);
distances in float64 (ScaNN assigns in float32 under UNBALANCED_FLOAT32).
Run:  python -I ivf_scann_toy.py
"""

import numpy as np

# ---- builder defaults (scann_builder.py:154-167) ---------------------------
NUM_LEAVES = 3                 # num_children  (FAISS nlist)
NUM_LEAVES_TO_SEARCH = (1, 2)  # query_spilling.max_spill_centers (FAISS nprobe)
TRAINING_ITERATIONS = 12       # max_clustering_iterations
MIN_PARTITION_SIZE = 50        # min_cluster_size (clamped below)
CONVERGENCE_EPS = 1e-5         # clustering_convergence_tolerance (proto default)
PERTURBATION = 1e-7            # GmmUtils::Options::perturbation
SEED = 0                       # clustering_seed proto default

LABELS = list("ABCDEFGH")
VALUES = np.array([1, 2, 4, 7, 8, 11, 14, 15], dtype=np.float64)
# ScaNN takes a 2-D (N, D) array; we pad to D=2 as (x, 0) so the geometry is 1-D.
X = np.stack([VALUES, np.zeros_like(VALUES)], axis=1)


def sq_l2(a, b):
    """Squared L2 between every row of a and every row of b -> (len(a), len(b))."""
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)


def kmeans_scann(x, k, iters, min_cluster_size_cfg, rng, log):
    """Mirror of GmmUtils::ComputeKmeansClustering with UNBALANCED_FLOAT32
    assignment, RANDOM_INITIALIZATION and RANDOM_REASSIGNMENT of small clusters."""
    n = len(x)
    if n < k:  # gmm_utils.cc:1105-1109
        raise ValueError(f"Number of points ({n}) is less than the number of clusters ({k}).")
    # gmm_utils.cc:1111-1112 — min size is clamped to N/k
    min_cluster_size = min(min_cluster_size_cfg, n // k)
    log(f"min_cluster_size = min({min_cluster_size_cfg}, {n}//{k}) = {min_cluster_size}")

    # RandomInitializeCenters (gmm_utils.cc:1006-1038): k distinct random points
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    log(f"init centres (random points): {centers[:, 0].tolist()}")

    old_means = np.full(k, -1.0)
    for it in range(iters + 1):  # Seq(max_iterations + 1), gmm_utils.cc:1139
        d = sq_l2(x, centers)
        assign = d.argmin(1)
        sizes = np.bincount(assign, minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            new_means = np.bincount(assign, weights=d[np.arange(n), assign], minlength=k) / sizes
        # convergence test, gmm_utils.cc:1239-1251
        converged = all(abs(new_means[c] - old_means[c]) <= CONVERGENCE_EPS and sizes[c] >= min_cluster_size
                        for c in range(k))
        log(f"iter {it}: centres={np.round(centers[:, 0], 4).tolist()} sizes={sizes.tolist()}"
            + ("  -> converged" if converged else ""))
        if converged or it == iters:
            break
        old_means = new_means
        # RecomputeCentroidsSimple (gmm_utils.cc:1438-1516): plain mean per cluster
        for c in range(k):
            if sizes[c] > 0:
                centers[c] = x[assign == c].mean(0)
        # RandomReinitializeCenters (gmm_utils.cc:1633-1689): a cluster below
        # min_cluster_size is moved next to a big cluster's centre.
        for c in range(k):
            if sizes[c] >= min_cluster_size:
                continue
            while True:
                ridx = rng.integers(0, n)
                big = assign[ridx]
                if sizes[big] >= min_cluster_size:
                    break
            centers[c] = centers[big] + PERTURBATION * (x[ridx] - centers[big])
            log(f"   reinit small cluster {c} (size {sizes[c]}) next to centre {big}")
    return centers, assign


def search(x, centers, assign, q, leaves_to_search, k_final, log):
    """KMeansTree::Tokenize (flat tree, FIXED_NUMBER_OF_CENTERS) then exact scan
    of each opened leaf, merged into one top-k (score_brute_force path)."""
    dc = sq_l2(q[None], centers)[0]
    order = np.argsort(dc)
    opened = order[:leaves_to_search]  # PostprocessDistancesForSpilling top-L
    log(f"   distances to centres: {np.round(dc, 3).tolist()} -> open leaves {opened.tolist()}")
    cand = np.where(np.isin(assign, opened))[0]
    dq = sq_l2(q[None], x[cand])[0]
    top = cand[np.argsort(dq)][:k_final]
    return opened, top


def main():
    log = print
    log("SIMULATION of ScaNN tree + score_brute_force (not the real library)")
    log(f"data (x,0): {dict(zip(LABELS, VALUES.tolist()))}")
    rng = np.random.default_rng(SEED)
    centers, assign = kmeans_scann(X, NUM_LEAVES, TRAINING_ITERATIONS, MIN_PARTITION_SIZE, rng, log)

    log("\nleaves (datapoints_by_token):")
    for c in range(NUM_LEAVES):
        members = [LABELS[i] for i in range(len(X)) if assign[i] == c]
        log(f"   leaf {c}: centre x={centers[c, 0]:.4f}  members={members}")

    for qx in (9.5, 5.5):
        q = np.array([qx, 0.0])
        exact = np.argsort(sq_l2(q[None], X)[0])[:2]
        log(f"\nquery x={qx}: exact top-2 = {[LABELS[i] for i in exact]}")
        for L in NUM_LEAVES_TO_SEARCH:
            log(f" leaves_to_search={L}")
            opened, top = search(X, centers, assign, q, L, 2, log)
            got = [LABELS[i] for i in top]
            miss = [LABELS[i] for i in exact if i not in top]
            log(f"   returned {got}" + (f"  MISS {miss}" if miss else "  (matches exact)"))


if __name__ == "__main__":
    main()
