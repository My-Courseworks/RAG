"""Vamana / DiskANN scale test for M5, on the same generator the IVF comparison
used (supplementary/ivf/README.md), with hnswlib as the baseline.

Stages (pick with --stage, default all):
  mem   in-memory Vamana (diskannpy.build_memory_index + StaticMemoryIndex)
        R = 32 and 64 at alpha = 1.2, plus R = 32 at alpha = 1.0; build L = 100;
        sweep search L. Also counts expansions per query by walking the graph
        the library saved (MIRROR search, a Python transcription of
        iterate_to_fixed_point; labelled as such in the output).
  hnsw  hnswlib, M = 16 and 32, ef_construction = 200, sweep ef.
  disk  DiskANN SSD index (build_disk_index + StaticDiskIndex), R = 64, L = 100,
        at 32, 64, 128 and 256 PQ bytes per vector in RAM; sweep L, beam width
        and node cache; 4 KB reads per query measured from /proc/self/io (diskannpy 0.7.0 computes QueryStats
        in C++ and discards them, python/src/static_disk_index.cpp:57-104).

Run inside the container described in supplementary/diskann/README.md:
    python -I diskann_scale.py --work /work [--stage mem|hnsw|disk|all]
Timings are single-thread ms/query, best of 3 passes over the 1,000 queries.
Synthetic data: the absolute numbers are not LogiTrack's.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

N, D, NQ, K = 187_043, 256, 1000, 10


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def ground_truth(xb, xq, k=K):
    bn = (xb * xb).sum(1)
    out = np.empty((len(xq), k), dtype=np.int64)
    for s in range(0, len(xq), 100):
        q = xq[s:s + 100]
        d = bn[None, :] - 2.0 * (q @ xb.T)
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, 1), axis=1)
        out[s:s + 100] = np.take_along_axis(part, order, 1)
    return out


def recall(ids, gt):
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / gt.shape[1] for a, b in zip(ids, gt)]))


def best_of(fn, reps=3):
    best, res = None, None
    for _ in range(reps):
        t = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t
        best = dt if best is None or dt < best else best
    return best, res


def quiet(fn, *a, **kw):
    sys.stdout.flush()
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def read_graph(path):
    buf = open(path, "rb").read()
    size, max_degree, start, _ = struct.unpack_from("<QIIQ", buf, 0)
    arr = np.frombuffer(buf, dtype="<u4", offset=24, count=(size - 24) // 4)
    adj, off = [], 0
    while off < len(arr):
        k = int(arr[off])
        adj.append(arr[off + 1: off + 1 + k])
        off += k + 1
    return start, adj, max_degree


def mirror_search(xb, adj, start, q, L):
    """MIRROR of iterate_to_fixed_point (src/index.cpp:790-974): returns the
    best-L ids and the number of expansions and distance computations."""
    def dist(ids):
        diff = xb[ids] - q
        return np.einsum("ij,ij->i", diff, diff)
    data = [[float(dist(np.array([start]))[0]), int(start), False]]
    visited = {int(start)}
    cur = exp = cmps = 0
    while cur < len(data):
        data[cur][2] = True
        n = data[cur][1]
        while cur < len(data) and data[cur][2]:
            cur += 1
        exp += 1
        fresh = [int(v) for v in adj[n] if int(v) not in visited]
        if not fresh:
            continue
        visited.update(fresh)
        cmps += len(fresh)
        for dv, v in zip(dist(np.array(fresh)).tolist(), fresh):
            if len(data) == L and (data[-1][0], data[-1][1]) < (dv, v):
                continue
            lo, hi = 0, len(data)
            while lo < hi:
                mid = (lo + hi) >> 1
                if (dv, v) < (data[mid][0], data[mid][1]):
                    hi = mid
                else:
                    lo = mid + 1
            data.insert(lo, [dv, v, False])
            if len(data) > L:
                data.pop()
            if lo < cur:
                cur = lo
    return [d[1] for d in data], exp, cmps


def stage_mem(xb, xq, gt, work, out):
    import diskannpy
    rows = []
    for R, alpha in ((32, 1.2), (64, 1.2), (32, 1.0)):
        d = os.path.join(work, f"mem_R{R}_a{alpha}")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        t = time.perf_counter()
        quiet(diskannpy.build_memory_index, xb, "l2", d, complexity=100, graph_degree=R,
              num_threads=0, alpha=alpha)
        build_s = time.perf_counter() - t
        gsize = os.path.getsize(os.path.join(d, "ann"))
        start, adj, maxdeg = read_graph(os.path.join(d, "ann"))
        deg = np.array([len(a) for a in adj])
        idx = quiet(diskannpy.StaticMemoryIndex, index_directory=d, num_threads=1, initial_search_complexity=200)
        print(f"\nVamana in-memory R={R} L_build=100 alpha={alpha}: build {build_s:.1f} s (all cores), "
              f"graph file {gsize / 2**20:.1f} MiB = {gsize / N:.0f} B/node, "
              f"degree mean {deg.mean():.1f} max {deg.max()} (max_observed {maxdeg}), start node {start}")
        print("   L   recall@10   ms/query(1 thread)   MIRROR expansions/query   MIRROR distance comps/query   MIRROR top-10 == library")
        for L in (10, 20, 30, 50, 75, 100, 150, 200):
            dt, res = best_of(lambda: idx.batch_search(xq, k_neighbors=K, complexity=L, num_threads=1))
            ids = np.asarray(res.identifiers)
            rec = recall(ids, gt)
            nprobe = 200
            exps, cmps, agree = [], [], 0
            for qi in range(nprobe):
                top, e, c = mirror_search(xb, adj, start, xq[qi], L)
                exps.append(e)
                cmps.append(c)
                agree += int(top[:K] == ids[qi].tolist())
            row = dict(index="vamana", R=R, alpha=alpha, L=L, recall=rec, ms=1000 * dt / NQ,
                       expansions=float(np.mean(exps)), cmps=float(np.mean(cmps)), mirror_agree=agree / nprobe,
                       build_s=build_s, graph_bytes=gsize, mean_degree=float(deg.mean()))
            rows.append(row)
            print(f" {L:4d}     {rec:.4f}        {row['ms']:.3f}               {row['expansions']:.1f}"
                  f"                     {row['cmps']:.0f}                      {agree}/{nprobe}")
        del idx
    out["mem"] = rows


def stage_hnsw(xb, xq, gt, work, out):
    import hnswlib
    rows = []
    for M in (16, 32):
        idx = hnswlib.Index(space="l2", dim=D)
        idx.init_index(max_elements=N, M=M, ef_construction=200, random_seed=100)
        t = time.perf_counter()
        idx.add_items(xb, np.arange(N))
        build_s = time.perf_counter() - t
        path = os.path.join(work, f"hnsw_M{M}.bin")
        idx.save_index(path)
        fsize = os.path.getsize(path)
        graph_bytes = fsize - N * D * 4
        os.remove(path)
        idx.set_num_threads(1)
        print(f"\nhnswlib M={M} ef_construction=200: build {build_s:.1f} s (all cores), "
              f"index file minus vectors {graph_bytes / 2**20:.1f} MiB = {graph_bytes / N:.0f} B/node")
        print("  ef   recall@10   ms/query(1 thread)")
        for ef in (10, 20, 30, 50, 75, 100, 150, 200):
            idx.set_ef(ef)
            dt, res = best_of(lambda: idx.knn_query(xq, k=K))
            rec = recall(res[0], gt)
            rows.append(dict(index="hnsw", M=M, ef=ef, recall=rec, ms=1000 * dt / NQ, build_s=build_s,
                             graph_bytes=graph_bytes))
            print(f" {ef:4d}     {rec:.4f}        {1000 * dt / NQ:.3f}")
    out["hnsw"] = rows


def read_bytes():
    for line in open("/proc/self/io"):
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


def stage_disk(xb, xq, gt, work, out):
    import diskannpy
    R, L_build = 64, 100
    out["disk"] = []
    for pq_bytes in (32, 64, 128, 256):
        d = os.path.join(work, f"disk_R64_pq{pq_bytes}")
        stamp = d + ".build_seconds"
        if os.path.exists(stamp) and os.path.exists(os.path.join(d, "ann_disk.index")):
            build_s = float(open(stamp).read())          # reuse the index from an earlier run
        else:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            # num_pq_chunks = floor(search_memory_maximum_bytes / N)  (src/disk_utils.cpp:1298)
            search_gb = (pq_bytes + 0.5) * N / 2**30
            t = time.perf_counter()
            quiet(diskannpy.build_disk_index, xb, "l2", d, complexity=L_build, graph_degree=R,
                  search_memory_maximum=search_gb, build_memory_maximum=8.0, num_threads=0, pq_disk_bytes=0)
            build_s = time.perf_counter() - t
            open(stamp, "w").write(str(build_s))
        files = {f: os.path.getsize(os.path.join(d, f)) for f in sorted(os.listdir(d))}
        print(f"\nDiskANN disk index R={R} L_build={L_build}, {pq_bytes} PQ bytes per vector in RAM: "
              f"build {build_s:.1f} s (all cores)")
        for f, s in files.items():
            print(f"   {f:38s} {s:>12,d} B")
        disk_file = files["ann_disk.index"]
        meta = np.fromfile(os.path.join(d, "ann_disk.index"), dtype="<u8", offset=8, count=9)
        max_node_len, nodes_per_sector = int(meta[3]), int(meta[4])
        pq_file = files["ann_pq_compressed.bin"]
        print(f"   on-disk node = {max_node_len} B, {nodes_per_sector} nodes per 4,096 B sector; "
              f"SSD file {disk_file / 2**20:.1f} MiB; PQ codes in RAM {pq_file / 2**20:.2f} MiB "
              f"({(pq_file - 8) / N:.0f} B/vector); raw vectors {N * D * 4 / 2**20:.1f} MiB")
        rows = []
        for cache in (0, 10_000):
            # num_threads=1 here deadlocks at load in 0.7.0: use_medoids_data_as_centroids borrows the
            # only per-thread scratch and read_nodes then waits for a second one (fixed upstream after
            # the release, commits fd93a04a and 6f2691c7). Load with 4; each search still runs on 1.
            idx = quiet(diskannpy.StaticDiskIndex, index_directory=d, num_threads=4,
                        num_nodes_to_cache=cache, cache_mechanism=2 if cache else 0)
            print(f"   cached nodes = {cache}")
            print("      L   beam   recall@10   ms/query(1 thread)   4KB reads/query (/proc/self/io)")
            for L, W in ((10, 2), (20, 2), (30, 2), (50, 2), (75, 2), (100, 2), (200, 2), (50, 1), (50, 4), (50, 8)):
                b0 = read_bytes()
                t = time.perf_counter()
                res = idx.batch_search(xq, k_neighbors=K, complexity=L, num_threads=1, beam_width=W)
                dt = time.perf_counter() - t
                reads = (read_bytes() - b0) / 4096 / NQ
                rec = recall(np.asarray(res.identifiers), gt)
                rows.append(dict(index="diskann", R=R, pq_bytes=pq_bytes, cache=cache, L=L, beam=W, recall=rec,
                                 ms=1000 * dt / NQ, reads_4k=reads))
                print(f"    {L:4d}   {W:3d}     {rec:.4f}        {1000 * dt / NQ:.3f}               {reads:.1f}")
            del idx
        out["disk"].append(dict(pq_bytes=pq_bytes, rows=rows, files=files, build_s=build_s,
                                max_node_len=max_node_len, nodes_per_sector=nodes_per_sector))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--stage", default="all", choices=["mem", "hnsw", "disk", "all"])
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    xb, xq = make_data()
    t = time.perf_counter()
    gt = ground_truth(xb, xq)
    print(f"N={N} D={D} NQ={NQ} K={K}; brute-force L2 ground truth in {time.perf_counter() - t:.1f} s; "
          f"cpus={os.cpu_count()}")
    out = {}
    if a.stage in ("mem", "all"):
        stage_mem(xb, xq, gt, a.work, out)
    if a.stage in ("hnsw", "all"):
        stage_hnsw(xb, xq, gt, a.work, out)
    if a.stage in ("disk", "all"):
        stage_disk(xb, xq, gt, a.work, out)
    with open(os.path.join(a.work, f"scale_{a.stage}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
