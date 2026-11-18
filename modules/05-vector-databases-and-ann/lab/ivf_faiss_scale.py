"""IVF scale test on the course's shared generator, run on the real FAISS library.

N=187,043 x D=256 (50 Gaussian topics, sigma 0.35), 1000 queries, k=10, L2.
Builds:
  A. IVF4096,Flat        (IndexIVFFlat, raw floats, by_residual=False)
  B. IVF4096,PQ32x8      (IndexIVFPQ, 32 bytes/vector, by_residual=True)
  C. IVF1024,Flat        (nlist/nprobe interaction)
  D. IVF4096,PQ32x8,RFlat (PQ shortlist re-ranked with full vectors, k_factor sweep)
Sweeps nprobe in 1..256, reports recall@10 vs brute-force numpy ground truth,
ms/query (batched, all threads), build time, serialized size, fraction of the
database actually scanned (from faiss.cvar.indexIVF_stats.ndis). Then runs
faiss.ParameterSpace().explore on index A to show FAISS's own recall-time
frontier (OperatingPoints).

Run:  python -I ivf_faiss_scale.py        (~2-4 min on an M-series Mac)
Deps: faiss-cpu==1.16.0, numpy
"""

import time

import numpy as np
import faiss

N, D, NQ, K = 187_043, 256, 1000, 10
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return np.ascontiguousarray(x[:N]), np.ascontiguousarray(x[N:])


def ground_truth(xb, xq, k, chunk=50):
    """Brute-force L2 top-k with numpy, in query chunks."""
    xb_sq = (xb.astype(np.float64) ** 2).sum(1)
    out = np.empty((len(xq), k), dtype=np.int64)
    for i0 in range(0, len(xq), chunk):
        q = xq[i0:i0 + chunk].astype(np.float64)
        d = xb_sq[None, :] - 2.0 * q @ xb.T.astype(np.float64) + (q ** 2).sum(1)[:, None]
        part = np.argpartition(d, k, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(d[rows, part], axis=1)
        out[i0:i0 + chunk] = part[rows, order]
    return out


def recall_at_k(I, gt):
    return float(np.mean([len(set(I[i]) & set(gt[i])) / gt.shape[1]
                          for i in range(len(gt))]))


def build(factory, xb):
    index = faiss.index_factory(D, factory, faiss.METRIC_L2)
    t0 = time.perf_counter()
    index.train(xb)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    index.add(xb)
    t_add = time.perf_counter() - t0
    size_bytes = faiss.serialize_index(index).nbytes
    return index, t_train, t_add, size_bytes


def list_size_stats(index):
    sizes = np.array([index.invlists.list_size(i) for i in range(index.nlist)])
    return sizes.min(), int(np.median(sizes)), sizes.max(), int((sizes == 0).sum())


def sweep(index, xq, gt, label):
    stats = faiss.cvar.indexIVF_stats
    rows = []
    first_95 = None
    print(f"\n--- {label}: nlist={index.nlist} code_size={index.code_size} B "
          f"by_residual={index.by_residual} threads={faiss.omp_get_max_threads()}")
    print(f"{'nprobe':>6} {'recall@10':>9} {'ms/query':>9} {'%db scanned':>11}")
    for nprobe in NPROBES:
        if nprobe > index.nlist:
            break
        params = faiss.SearchParametersIVF(nprobe=nprobe)
        index.search(xq[:50], K, params=params)  # warm-up
        stats.reset()
        t0 = time.perf_counter()
        _, I = index.search(xq, K, params=params)
        dt = (time.perf_counter() - t0) * 1000 / len(xq)
        rec = recall_at_k(I, gt)
        scanned = 100.0 * stats.ndis / (stats.nq * N)
        rows.append((nprobe, rec, dt, scanned))
        if first_95 is None and rec >= 0.95:
            first_95 = nprobe
        print(f"{nprobe:>6} {rec:>9.4f} {dt:>9.3f} {scanned:>10.2f}%")
    print(f"smallest nprobe with recall@10 >= 0.95: {first_95}")
    return rows, first_95


def autotune(index, xq, gt):
    """FAISS's own way to pick nprobe: ParameterSpace.explore over the
    recall@10 criterion, keeping only Pareto-optimal (time, recall) points."""
    crit = faiss.IntersectionCriterion(len(xq), K)
    crit.set_groundtruth(None, gt.astype(np.int64))
    ps = faiss.ParameterSpace()
    ps.initialize(index)
    ps.n_experiments = 0  # 0 = exhaustive over the (small) nprobe grid
    t0 = time.perf_counter()
    ops = ps.explore(index, xq, crit)  # python wrapper allocates OperatingPoints
    print(f"\n--- ParameterSpace.explore on IVF4096,Flat ({time.perf_counter() - t0:.1f}s), "
          f"tunables: {[ps.parameter_ranges.at(i).name for i in range(ps.parameter_ranges.size())]}")
    print(f"{'parameters':>12} {'recall@10':>9} {'ms/query':>9}   (Pareto-optimal points only)")
    opt = ops.optimal_pts
    for i in range(opt.size()):
        op = opt.at(i)
        if op.perf == 0:  # the (0 recall, 0 time) sentinel OperatingPoints starts with
            continue
        print(f"{op.key:>12} {op.perf:>9.4f} {op.t * 1000 / len(xq):>9.3f}")


def refine_sweep(xb, xq, gt, nprobe):
    """Rescoring: IVFPQ shortlist of k*k_factor, re-ranked with the stored
    full vectors (IndexRefineFlat, factory suffix ',RFlat')."""
    index, t_train, t_add, size_b = build("IVF4096,PQ32x8,RFlat", xb)
    print(f"\n### D IVF4096,PQ32x8,RFlat: train {t_train:.1f}s  add {t_add:.1f}s  "
          f"serialized {size_b / 2**20:.1f} MiB ({size_b / N:.1f} B/vector: PQ codes + full fp32 copy)")
    print(f"--- D nprobe fixed at {nprobe}, sweep k_factor (candidates re-ranked = 10*k_factor)")
    print(f"{'k_factor':>8} {'recall@10':>9} {'ms/query':>9}")
    for kf in (1, 2, 4, 8, 16, 32, 64):
        params = faiss.IndexRefineSearchParameters(
            k_factor=kf, base_index_params=faiss.SearchParametersIVF(nprobe=nprobe))
        index.search(xq[:50], K, params=params)
        t0 = time.perf_counter()
        _, I = index.search(xq, K, params=params)
        dt = (time.perf_counter() - t0) * 1000 / len(xq)
        print(f"{kf:>8} {recall_at_k(I, gt):>9.4f} {dt:>9.3f}")


def main():
    print(f"faiss {faiss.__version__}, numpy {np.__version__}, "
          f"omp threads {faiss.omp_get_max_threads()}")
    xb, xq = make_data()
    t0 = time.perf_counter()
    gt = ground_truth(xb, xq, K)
    print(f"ground truth (numpy brute force): {time.perf_counter() - t0:.1f}s")

    results = {}
    for label, factory in [("A IVF4096,Flat", "IVF4096,Flat"),
                           ("B IVF4096,PQ32x8", "IVF4096,PQ32x8"),
                           ("C IVF1024,Flat", "IVF1024,Flat")]:
        index, t_train, t_add, size_b = build(factory, xb)
        mn, med, mx, empty = list_size_stats(index)
        print(f"\n### {label}: train {t_train:.1f}s  add {t_add:.1f}s  "
              f"serialized {size_b / 2**20:.1f} MiB  ({size_b / N:.1f} B/vector)")
        print(f"    list sizes min/median/max = {mn}/{med}/{mx}, empty lists = {empty}; "
              f"training set {N} vs k*max_points_per_centroid = "
              f"{index.nlist * index.cp.max_points_per_centroid} "
              f"({'subsampled' if N > index.nlist * index.cp.max_points_per_centroid else 'no subsampling'}), "
              f"k*min_points = {index.nlist * index.cp.min_points_per_centroid}")
        if isinstance(index, faiss.IndexIVFPQ):
            print(f"    PQ: M={index.pq.M} nbits={index.pq.nbits} dsub={index.pq.dsub} "
                  f"ksub={index.pq.ksub}; use_precomputed_table={index.use_precomputed_table} "
                  f"polysemous_ht={index.polysemous_ht}; "
                  f"encoder trained on {index.train_encoder_num_vectors()} residuals")
        results[label] = sweep(index, xq, gt, label)
        if label.startswith("A"):
            autotune(index, xq, gt)
        if label.startswith("B"):
            refine_sweep(xb, xq, gt, nprobe=64)

    print("\n=== summary: smallest nprobe for recall@10 >= 0.95 ===")
    for label, (_, first) in results.items():
        print(f"  {label}: nprobe={first}")


if __name__ == "__main__":
    main()
