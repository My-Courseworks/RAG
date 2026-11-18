"""SIMULATION of NVIDIA cuVS IVF-Flat on the course's eight 1-D values.

cuVS has no CPU backend; the real library cannot run on this Mac (no NVIDIA
GPU, only manylinux wheels). This file re-implements, in numpy, the exact
control flow of `ivf_flat::build` + `ivf_flat::search` as read at cuVS commit
aac896693db9f45683053bff236379c2449de80d (branch release/26.12):

  cpp/src/neighbors/ivf_flat/ivf_flat_build.cuh      build()     L391-446
  cpp/src/cluster/detail/kmeans_balanced.cuh         build_hierarchical L1521-1721
                                                     build_clusters     L1236-1293
                                                     balancing_em_iters L1122-1234
                                                     adjust_centers     L957-1080
                                                     adjust_centers_kernel L858-915
                                                     arrange_fine_clusters L1296-1354
  cpp/src/neighbors/ivf_flat/ivf_flat_search.cuh     search_impl L41-315

Everything printed below is produced by this simulation, not by cuVS.
Run:  python -I lab/ivf_cuvs_toy.py
"""

from __future__ import annotations

import numpy as np

VALUES = np.array([1, 2, 4, 7, 8, 11, 14, 15], dtype=np.float32)
NAMES = "ABCDEFGH"
N_LISTS = 3

# kmeans_balanced.cuh L1013-1016: primes used to derive the donor-point stride
K_PRIMES = [29, 71, 113, 173, 229, 281, 349, 409, 463, 541, 601, 659, 733, 809,
            863, 941, 1013, 1069, 1151, 1223, 1291, 1373, 1451, 1511, 1583, 1657,
            1733, 1811, 1889, 1987, 2053, 2129, 2213, 2287, 2357, 2423, 2531,
            2617, 2687, 2741]
WARP_SIZE = 32


def fmt(xs) -> str:
    return "[" + ", ".join(f"{x:g}" for x in np.atleast_1d(xs)) + "]"


def names_of(idx) -> str:
    return "{" + ", ".join(f"{NAMES[i]}={VALUES[i]:g}" for i in idx) + "}" if len(idx) else "{}"


# ----------------------------------------------------------------------------
# kmeans_balanced.cuh building blocks
# ----------------------------------------------------------------------------
def predict(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """E-step: nearest centre by L2 (predict_core, L135-215). Ties -> lowest id,
    matching the argmin convention of the fused L2-NN kernel."""
    d = np.abs(x[:, None] - centers[None, :])
    return d.argmin(axis=1)


def calc_centers_and_sizes(x, labels, k):
    """M-step (L394-496). Empty cluster -> sum/0 -> 0 via raft::div_checkzero_op."""
    sizes = np.bincount(labels, minlength=k)
    sums = np.bincount(labels, weights=x, minlength=k)
    centers = np.where(sizes > 0, sums / np.maximum(sizes, 1), 0.0)
    return centers.astype(np.float64), sizes


def adjust_centers(x, centers, labels, sizes, lower_tol, upper_tol,
                   centroid_offset, balancing_step, log, verbose=True):
    """adjust_centers (L957-1080) with the default SizeSorted donor selection,
    plus the donor-point pick of adjust_centers_kernel (L858-915).
    Returns True if any centre moved (then labels must be recomputed)."""
    n_rows, k = len(x), len(centers)
    average = n_rows / k
    lower, upper = average * lower_tol, average * upper_tol
    order = sorted((int(sizes[c]), c) for c in range(k))  # std::sort on (size, id)
    pairs = []
    for p in range(k // 2):
        small_size, small = order[p]
        large_size, large = order[k - 1 - p]
        if small == large or large_size == 0:
            break
        if small_size >= lower and large_size <= upper:
            break
        pairs.append((small, large))
    if not pairs:
        if verbose:
            log(f"      adjust_centers: sizes={fmt(sizes)} avg={average:.3f} "
                f"lower={lower:.3f} upper={upper:.1f} -> nothing underfull/overfull")
        return False
    # L1035-1039: stride = kPrimes[balancing_step ...] that does not divide n_rows
    i = balancing_step
    while True:
        ofst = K_PRIMES[i % len(K_PRIMES)]
        i += 1
        if n_rows % ofst != 0:
            break
    for pair_id, (receiver, donor) in enumerate(pairs):
        # kernel L879-893: lanes j=0..31 probe candidates until one carries the donor label
        donor_point = None
        for attempt in range(0, n_rows, WARP_SIZE):
            cands = [(ofst * (attempt + j + 1) + pair_id) % n_rows for j in range(WARP_SIZE)]
            hits = [c for c in cands if labels[c] == donor]
            if hits:
                donor_point = hits[0]  # __ffs(mask): lowest lane wins
                break
        assert donor_point is not None
        new_c = centers[donor] + centroid_offset * (x[donor_point] - centers[donor])
        log(f"      adjust_centers: cluster {receiver} (size {sizes[receiver]}) is underfull "
            f"(< {lower:.3f}); re-seeded next to donor cluster {donor} (size {sizes[donor]}): "
            f"{centers[donor]:g} + {centroid_offset}*({x[donor_point]:g} - {centers[donor]:g}) "
            f"= {new_c:.4g}  [donor point {NAMES[donor_point]}, stride prime {ofst}]")
        centers[receiver] = new_c
    return True


def balancing_em_iters(x, centers, k, n_iters, pullback, lower_tol, upper_tol,
                       centroid_offset, log, balance=True):
    """balancing_em_iters (L1122-1234). `balance=False` turns it into plain Lloyd's."""
    labels = predict(x, centers)
    _, sizes = calc_centers_and_sizes(x, labels, k)
    counter = pullback
    it = 0
    prev = None
    quiet = []  # iterations with no change, collapsed in the log
    while it < n_iters:
        did_adjust = False
        if it > 0 and balance:
            did_adjust = adjust_centers(x, centers, labels, sizes, lower_tol, upper_tol,
                                        centroid_offset, it, log, verbose=prev is None)
        if did_adjust:
            if counter >= pullback:  # L1186-1189: every pullback-th adjustment adds an iteration
                counter -= pullback
                n_iters += 1
            counter += 1
        labels = predict(x, centers)                       # E
        centers[:], sizes = calc_centers_and_sizes(x, labels, k)  # M
        state = (tuple(labels), tuple(np.round(centers, 6)))
        if state == prev and not did_adjust:
            quiet.append(it)
        else:
            if quiet:
                log(f"    iters {quiet[0]}..{quiet[-1]}: unchanged (converged; balancing finds nothing to fix)")
                quiet = []
            log(f"    iter {it}: labels={fmt(labels)} centers={fmt(np.round(centers, 4))} sizes={fmt(sizes)}")
        prev = state
        it += 1
    if quiet:
        log(f"    iters {quiet[0]}..{quiet[-1]}: unchanged (converged; balancing finds nothing to fix)")
    return centers, labels, sizes


def build_clusters(x, k, params, log, balance=True):
    """build_clusters (L1236-1293): labels[i] = i mod k, centres from those labels, then EM.
    NB: this is cuVS's 'random' init -- it is fully deterministic, no seed involved."""
    labels0 = np.arange(len(x)) % k
    centers, sizes = calc_centers_and_sizes(x, labels0, k)
    log(f"    init labels (i mod {k}) = {fmt(labels0)} -> centers {fmt(np.round(centers, 4))}")
    return balancing_em_iters(x, centers, k, params["n_iters"], pullback=2,
                              lower_tol=params["lower"], upper_tol=params["upper"],
                              centroid_offset=params["offset"], log=log, balance=balance)


def arrange_fine_clusters(n_clusters, n_meso, n_rows, meso_sizes):
    """arrange_fine_clusters (L1296-1354): fine clusters per mesocluster, proportional to size."""
    nums = []
    lists_rem, rows_rem = n_clusters, n_rows
    for i in range(n_meso):
        if i < n_meso - 1:
            if meso_sizes[i] == 0:
                s = 0
            else:
                s = max(int(lists_rem * meso_sizes[i] / rows_rem + 0.5), 1)
        else:
            s = lists_rem
        nums.append(s)
        lists_rem -= s
        rows_rem -= meso_sizes[i]
    return nums


def build_hierarchical(x, k, params, log):
    """build_hierarchical (L1521-1721) = kmeans::fit for balanced_params."""
    n_meso = min(k, int(np.sqrt(k) + 0.5))
    log(f"  n_mesoclusters = min({k}, round(sqrt({k}))) = {n_meso}")
    log(f"  [1] mesoclusters: build_clusters on {len(x)} training points, k={n_meso}")
    _, meso_labels, meso_sizes = build_clusters(x, n_meso, params, log)
    nums = arrange_fine_clusters(k, n_meso, len(x), meso_sizes)
    log(f"  [2] fine clusters per mesocluster (proportional to size {fmt(meso_sizes)}): {fmt(nums)}")
    centers = []
    for m in range(n_meso):
        pts = np.flatnonzero(meso_labels == m)
        if nums[m] == 0:
            continue
        log(f"      mesocluster {m} = {names_of(pts)} -> build_clusters k={nums[m]}")
        c, _, _ = build_clusters(x[pts], nums[m], params, log)
        centers.extend(c.tolist())
    centers = np.array(centers, dtype=np.float64)
    n_iters = max(params["n_iters"] // 10, 2)
    log(f"  [3] fine-tune all {k} centres on all training points: n_iters=max({params['n_iters']}//10,2)={n_iters}, "
        f"pullback=5, start={fmt(centers)}")
    centers, labels, sizes = balancing_em_iters(
        x, centers, k, n_iters, pullback=5, lower_tol=params["lower"],
        upper_tol=params["upper"], centroid_offset=params["offset"], log=log)
    return centers


# ----------------------------------------------------------------------------
# ivf_flat::build / extend / search
# ----------------------------------------------------------------------------
IVF_FLAT_KMEANS = dict(n_iters=20, lower=0.333, upper=3.0, offset=0.01)  # balanced_params defaults


def ivf_flat_build(values, n_lists, trainset_fraction, log):
    """ivf_flat_build.cuh build() L391-446 + extend() L158-376 (add_data_on_build=true)."""
    n = len(values)
    assert n >= n_lists, "RAFT_EXPECTS(n_rows >= params.n_lists)"
    ratio = max(1, n // max(int(trainset_fraction * n), n_lists))   # L417-418
    train_idx = np.arange(0, n, ratio)[: n // ratio]                   # L423-428: strided copy_matrix
    log(f"  trainset_ratio = max(1, {n} // max({trainset_fraction}*{n}, {n_lists})) = {ratio}; "
        f"training rows = every {ratio}th row = {names_of(train_idx)}  ('TODO: a proper sampling', L422)")
    centers = build_hierarchical(values[train_idx].astype(np.float64), n_lists, IVF_FLAT_KMEANS, log)
    # extend(): predict() every dataset row against the frozen centres (adaptive_centers=false)
    labels = predict(values.astype(np.float64), centers)
    lists = [np.flatnonzero(labels == c) for c in range(n_lists)]
    log(f"  extend(): assign all {n} rows to nearest centre (centres stay frozen, adaptive_centers=false)")
    return centers, lists


def ivf_flat_search(values, centers, lists, q, k, n_probes, log):
    """ivf_flat_search.cuh: coarse GEMM+select_k over centres (L107-190), then scan probed lists."""
    n_probes = min(n_probes, len(centers))                 # L335
    coarse = np.abs(centers - q)
    probes = np.argsort(coarse, kind="stable")[:n_probes]
    cand = np.concatenate([lists[p] for p in probes]) if n_probes else np.array([], int)
    d = np.abs(values[cand] - q)
    order = np.argsort(d, kind="stable")[:k]
    found = cand[order]
    # kOutOfBoundsRecord padding when the probed lists hold fewer than k rows (ivf_common.cuh L32)
    log(f"    n_probes={n_probes}: coarse dists {fmt(np.round(coarse, 3))} -> open list(s) {fmt(probes)} "
        f"= {'+'.join(names_of(lists[p]) for p in probes)}")
    log(f"      scan {len(cand)} vectors -> top-{k}: {names_of(found)} dists {fmt(d[order])}"
        + (f"  (+{k - len(found)} kOutOfBoundsRecord)" if len(found) < k else ""))
    return found


def exact(values, q, k):
    return np.argsort(np.abs(values - q), kind="stable")[:k]


def main() -> None:
    out = []
    log = out.append
    print("cuVS IVF-Flat toy -- SIMULATION (cuVS cannot run on this Mac; numpy mirror of the")
    print("balanced k-means + IVF-Flat code paths cited in the module header)\n")
    print(f"data: {names_of(range(len(VALUES)))}, n_lists={N_LISTS}, metric=L2, dim=1\n")

    for frac in (0.5, 1.0):
        out.clear()
        print("=" * 78)
        print(f"BUILD with kmeans_trainset_fraction={frac}" + ("  (cuVS default)" if frac == 0.5 else ""))
        print("=" * 78)
        centers, lists = ivf_flat_build(VALUES, N_LISTS, frac, log)
        print("\n".join(out))
        print(f"\n  => centres {fmt(np.round(centers, 4))}")
        for c, l in enumerate(lists):
            print(f"     list {c}: {names_of(l)}  (size {len(l)})")

        for q, k in ((9.5, 3), (5.5, 2)):
            out.clear()
            ex = exact(VALUES, q, k)
            print(f"\n  SEARCH q={q}, k={k}; exact top-{k} = {names_of(ex)}")
            for npb in (1, 2):
                got = ivf_flat_search(VALUES, centers, lists, q, k, npb, log)
                miss = sorted(set(ex) - set(got))
                verdict = "exact" if not miss else f"MISS {names_of(miss)}"
                log(f"      -> {verdict}")
            print("\n".join(out))
        print()

    # ------------------------------------------------------------------
    print("=" * 78)
    print("BALANCED vs PLAIN LLOYD'S on all 8 values, k=3, identical init (labels = i mod 3)")
    print("(build_clusters path: this is also what trains the mesoclusters and PQ codebooks)")
    print("=" * 78)
    x = VALUES.astype(np.float64)
    for balance in (False, True):
        out.clear()
        print(("\n-- cuVS balanced k-means (adjust_centers ON, lower=0.333, upper=3.0, offset=0.01)"
               if balance else "\n-- plain Lloyd's (adjust_centers OFF)"))
        centers, labels, sizes = build_clusters(x, N_LISTS, IVF_FLAT_KMEANS, log, balance=balance)
        print("\n".join(out))
        print(f"   final centres {fmt(np.round(centers, 4))}, sizes {fmt(sizes)}")
        for c in range(N_LISTS):
            print(f"     list {c}: {names_of(np.flatnonzero(labels == c))}")


if __name__ == "__main__":
    main()
