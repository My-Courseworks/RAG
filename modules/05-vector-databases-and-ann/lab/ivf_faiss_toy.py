"""IVF toy on the course's eight 1-D values, run on the real FAISS library.

A..H = 1, 2, 4, 7, 8, 11, 14, 15 ; nlist = 3 ; IndexIVFFlat with an IndexFlatL2
coarse quantizer. Everything printed is read back from the live index:
centroids via quantizer.reconstruct_n, bins via invlists.list_size / get_ids /
get_codes, scan counts via faiss.cvar.indexIVF_stats.

Run:  python -I ivf_faiss_toy.py
Deps: faiss-cpu==1.16.0, numpy
"""

import numpy as np
import faiss

LABELS = list("ABCDEFGH")
VALUES = np.array([1, 2, 4, 7, 8, 11, 14, 15], dtype=np.float32)
NLIST = 3
SEED = 1234  # ClusteringParameters::seed default (faiss/Clustering.h:50)


def hr(title):
    print(f"\n=== {title} ===")


def build_index():
    xb = VALUES.reshape(-1, 1)
    quantizer = faiss.IndexFlatL2(1)
    index = faiss.IndexIVFFlat(quantizer, 1, NLIST, faiss.METRIC_L2)
    index.cp.seed = SEED  # explicit, same as the default
    # 8 points for 3 centroids is below min_points_per_centroid (39): FAISS
    # prints "WARNING clustering 8 points to 3 centroids: please provide at
    # least 117 training points" to stderr (Clustering.cpp:179-187) and
    # continues. nx (8) > k (3) so it is not the "just copy" corner case.
    index.train(xb)
    index.add(xb)
    return index, xb


def replay_kmeans():
    """Same k-means FAISS ran inside train(): Clustering(d=1, k=3) on an
    IndexFlatL2 assigner (IndexIVF.cpp:171-176). Exposes iteration_stats."""
    clus = faiss.Clustering(1, NLIST)
    clus.seed = SEED
    clus.niter = 10  # Level1Quantizer lowers niter from 25 to 10 (IndexIVF.cpp:49)
    assigner = faiss.IndexFlatL2(1)
    clus.train(VALUES.reshape(-1, 1), assigner)
    cents = faiss.vector_to_array(clus.centroids)
    stats = clus.iteration_stats
    rows = [(stats.at(i).obj, stats.at(i).imbalance_factor, stats.at(i).nsplit)
            for i in range(stats.size())]
    return cents, rows


def read_bins(index):
    bins = []
    for i in range(NLIST):
        n = index.invlists.list_size(i)
        ids = faiss.rev_swig_ptr(index.invlists.get_ids(i), n).copy()
        codes = faiss.rev_swig_ptr(index.invlists.get_codes(i),
                                   n * index.code_size).copy()
        vals = codes.view(np.float32)
        bins.append((ids, vals))
    return bins


def search(index, q, k, nprobe):
    params = faiss.SearchParametersIVF(nprobe=nprobe)
    stats = faiss.cvar.indexIVF_stats
    stats.reset()
    D, I = index.search(np.array([[q]], dtype=np.float32), k, params=params)
    return D[0], I[0], stats.nlist, stats.ndis


def exact(q, k):
    d = (VALUES - q) ** 2
    order = np.argsort(d, kind="stable")[:k]
    return d[order], order


def fmt_ids(ids):
    return "{" + ", ".join(f"{LABELS[i]}={VALUES[i]:g}" for i in ids) + "}"


def main():
    np.set_printoptions(precision=3, suppress=True)
    print(f"faiss {faiss.__version__}, numpy {np.__version__}")
    print("data:", ", ".join(f"{l}={v:g}" for l, v in zip(LABELS, VALUES)))

    hr("Step 1  train: k-means (Lloyd's, random init, seed 1234, niter 10 for IVF)")
    index, xb = build_index()
    cents = index.quantizer.reconstruct_n(0, NLIST).ravel()
    print("centroids read from quantizer.reconstruct_n:", cents)
    print(f"cp defaults: niter={index.cp.niter} nredo={index.cp.nredo} "
          f"min_points_per_centroid={index.cp.min_points_per_centroid} "
          f"max_points_per_centroid={index.cp.max_points_per_centroid} "
          f"spherical={index.cp.spherical} seed={index.cp.seed}")
    print(f"by_residual={index.by_residual} (IVFFlat stores raw floats) "
          f"code_size={index.code_size} bytes  nprobe default={index.nprobe}")

    replay_cents, rows = replay_kmeans()
    print("replay via faiss.Clustering gives", replay_cents,
          "-> identical:", np.allclose(np.sort(replay_cents), np.sort(cents)))
    print("iteration  objective(sum sq dist)  imbalance  nsplit")
    for i, (obj, imb, nsplit) in enumerate(rows):
        print(f"  {i:2d}        {obj:7.3f}                {imb:.3f}      {nsplit}")
    print(f"  ({len(rows)} iterations ran; early stop when objective stops "
          f"changing, Clustering.cpp:452-468)")

    hr("Step 2  assign + store: each value -> nearest centroid, appended to that list")
    bins = read_bins(index)
    for i, (ids, vals) in enumerate(bins):
        print(f"  list {i}: centroid {cents[i]:6.3f}  size {len(ids)}  "
              f"members {fmt_ids(ids)}")
    print("  (ids are the insertion order 0..7 because add() was called "
          "without xids; list = quantizer.assign, IndexIVF.cpp:318-324)")

    hr("Step 3  a query near a bin boundary")
    order = np.argsort(cents)
    sorted_c = cents[order]
    boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2
    print("centroid midpoints (Voronoi boundaries in 1-D):", boundaries)

    for q, k in [(9.5, 2), (5.5, 2), (9.5, 1)]:
        hr(f"query q={q}  k={k}")
        dq = np.abs(cents - q)
        print("  |q - centroid| per list:", dq,
              "-> probe order:", list(np.argsort(dq)))
        De, Ie = exact(q, k)
        print(f"  exact top-{k}: {fmt_ids(Ie)}  sq-dists {De}")
        for nprobe in (1, 2, 3):
            D, I, nl, nd = search(index, q, k, nprobe)
            got = [i for i in I if i >= 0]
            miss = sorted(set(Ie) - set(got))
            print(f"  nprobe={nprobe}: lists scanned={nl} distances={nd} "
                  f"-> {fmt_ids(got)}  sq-dists {D[:len(got)]}"
                  + (f"   MISSED {fmt_ids(miss)}" if miss else "   == exact"))

    hr("Step 4  the nprobe dial as FAISS expects you to use it")
    print("  index.nprobe is a plain attribute (query-time); "
          "ParameterSpace sets it from a string:")
    ps = faiss.ParameterSpace()
    ps.initialize(index)
    names = [ps.parameter_ranges.at(i).name
             for i in range(ps.parameter_ranges.size())]
    print("  ParameterSpace.initialize found tunables:", names,
          "(nprobe values stop below nlist: AutoTune.cpp:370-377)")
    ps.set_index_parameters(index, "nprobe=2")
    print("  after set_index_parameters('nprobe=2'): index.nprobe =", index.nprobe)
    queries = np.arange(0.0, 16.01, 0.25)
    print(f"  manual recall sweep, k=2, {len(queries)} queries on a 0..16 grid "
          f"(step 0.25); ties broken like the exact scan:")
    for nprobe in (1, 2, 3):
        hits, misses = 0, []
        for q in queries:
            _, I, _, _ = search(index, float(q), 2, nprobe)
            De, Ie = exact(float(q), 2)
            # accept any id whose exact distance equals the k-th exact distance
            ok = {i for i in range(len(VALUES)) if (VALUES[i] - q) ** 2 <= De[-1] + 1e-6}
            h = len(set(I) & ok)
            hits += h
            if h < 2:
                misses.append(float(q))
        tot = 2 * len(queries)
        print(f"    nprobe={nprobe}: recall@2 = {hits}/{tot} = {hits / tot:.3f}"
              + (f"   misses at q in {misses}" if misses else ""))


if __name__ == "__main__":
    main()
