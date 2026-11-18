"""Knowhere IVF_FLAT toy (via Milvus Lite) for the RAG course, module 5.

Eight 1-D values A..H = 1, 2, 4, 7, 8, 11, 14, 15, nlist = 3, query 9.5.
Milvus accepts dim=1 FLOAT_VECTOR, so the values are used as-is.

Three parts, each labelled in the output:
  PART 1  REAL        Milvus Lite on exactly the 8 values, nlist=3. Knowhere does
                      not expose centroids, so bins are inferred by probing each
                      value with nprobe=1 and a large limit: the ids returned are
                      exactly the members of the one list nearest that value.
  PART 2  SIMULATION  FAISS Clustering (Lloyd's, random init, niter=25, seed=1234)
                      on the 8 values with k=3, then IVF search at nprobe 1 vs 2.
                      Mirrors thirdparty/faiss/faiss/Clustering.cpp (random init
                      = rand_perm then first k rows; Lloyd's updates; l.100-125
                      warn when nx < k*39) and IndexIVF.cpp search_preassigned
                      (l.655-677). numpy's permutation differs from FAISS's
                      mt19937, so the initial pick differs; the converged
                      centroids are what the course compares.
  PART 3  REAL        Milvus Lite with each value replicated 150x with +-0.01
                      jitter (1200 rows), nlist=3. Shows the same probing
                      technique on a segment that clears Milvus's 1024-row
                      indexing threshold with 3*39 = 117 <= 1200 training points.

Why the legacy ORM API: Milvus Lite 3.2.2 silently drops `nlist` when it
arrives flattened from pymilvus 3.0.3's MilvusClient.prepare_index_params()
and trains with Knowhere's default nlist=128. Collection.create_index with a
nested {"params": {"nlist": 3}} is honoured.

Usage: python -I ivf_knowhere_toy.py [--db /path/to/toy.db]
"""
import argparse
import logging
import os
import sys
import warnings

import numpy as np

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*ORM-style PyMilvus API.*")  # Lite 3.2.2 needs the ORM path for nlist
from pymilvus import (  # noqa: E402
    Collection, CollectionSchema, DataType, FieldSchema, connections, utility,
)

VALUES = [1.0, 2.0, 4.0, 7.0, 8.0, 11.0, 14.0, 15.0]
LABELS = "ABCDEFGH"
QUERY = 9.5
NLIST = 3


def faiss_style_kmeans_1d(x, k, niter=25, seed=1234):
    """SIMULATION of faiss::Clustering::train_encoded with init_method=RANDOM."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    cents = x[perm[:k]].copy()
    for it in range(niter):
        assign = np.argmin(np.abs(x[:, None] - cents[None, :]), axis=1)
        new = np.array([x[assign == j].mean() if np.any(assign == j) else cents[j] for j in range(k)])
        if np.allclose(new, cents):
            break
        cents = new
    assign = np.argmin(np.abs(x[:, None] - cents[None, :]), axis=1)
    return cents, assign, it + 1


def ivf_search_1d(x, cents, assign, q, nprobe, k):
    """IndexIVF.cpp search_preassigned: rank centroids, scan the nprobe nearest lists exactly."""
    order = np.argsort(np.abs(cents - q))[:nprobe]
    cand = [i for i in range(len(x)) if assign[i] in order]
    cand.sort(key=lambda i: abs(x[i] - q))
    return [int(j) for j in order], cand[:k]


def build(name, ids, vecs, nlist):
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("v", DataType.FLOAT_VECTOR, dim=1),
    ])
    col = Collection(name, schema)
    col.insert([ids, vecs])
    col.flush()
    col.create_index("v", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": nlist}})
    utility.wait_for_index_building_complete(name, "v")
    col.load()
    return col


def search(col, q, nprobe, limit):
    res = col.search([[q]], "v", {"metric_type": "L2", "params": {"nprobe": nprobe}}, limit=limit)
    return [(h.id, h.distance) for h in res[0]]


def infer_bins(col, values, owner, limit):
    """Probe each value at nprobe=1; the returned ids are the members of its nearest list."""
    bins = {}
    for v in values:
        members = frozenset(owner[i] for i, _ in search(col, v, 1, limit))
        bins.setdefault(members, []).append(v)
    return sorted(bins, key=min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.getcwd(), "milvus_knowhere_toy.db"))
    args = ap.parse_args()
    for f in (args.db, args.db + ".lock"):
        if os.path.exists(f):
            os.remove(f)
    connections.connect(uri=args.db)
    print("server:", utility.get_server_version())
    x = np.array(VALUES)
    exact3 = sorted(range(len(x)), key=lambda i: abs(x[i] - QUERY))[:3]

    print("\n" + "=" * 72)
    print("PART 1  REAL  Milvus Lite, exactly the 8 values, IVF_FLAT nlist=3")
    print("=" * 72)
    col = build("toy8", list(range(8)), [[v] for v in VALUES], NLIST)
    print(f"1. index params echoed by Lite: {col.indexes[0].params}; "
          f"progress={utility.index_building_progress('toy8', 'v')}")
    print("   (stderr above shows FAISS's own warning: 8 points for 3 centroids < 3*39 = 117.)")
    bins = infer_bins(col, VALUES, list(range(8)), 8)
    print(f"2. lists inferred by probing each value at nprobe=1: {len(bins)}")
    for j, m in enumerate(bins):
        vals = [x[i] for i in sorted(m)]
        print(f"   list {j}: {[LABELS[i] + '=' + f'{x[i]:g}' for i in sorted(m)]}  centroid = mean = {np.mean(vals):.3f}")
    print(f"3. exact top-3 for query {QUERY}: {[f'{LABELS[i]}={x[i]:g}' for i in exact3]}")
    for nprobe in (1, 2, 3):
        got = [i for i, _ in search(col, QUERY, nprobe, 3)]
        missed = [f"{LABELS[i]}={x[i]:g}" for i in exact3 if i not in got]
        print(f"4. nprobe={nprobe}: top-3 = {[f'{LABELS[i]}={x[i]:g}' for i in got]}  missed: {missed or 'none'}")
    for nprobe in (1, 2, 3):
        got = [i for i, _ in search(col, QUERY, nprobe, 8)]
        print(f"5. nprobe={nprobe}, limit=8: {len(got)} rows reachable -> "
              f"{[LABELS[i] for i in got]}")

    print("\n" + "=" * 72)
    print("PART 2  SIMULATION  FAISS Clustering + IVF search on the 8 values, k=3")
    print("=" * 72)
    cents, assign, iters = faiss_style_kmeans_1d(x, NLIST)
    order = np.argsort(cents)
    remap = {old: new for new, old in enumerate(order)}
    cents = cents[order]
    assign = np.array([remap[a] for a in assign])
    print(f"1. Lloyd's converged in {iters} iterations (niter cap 25, seed 1234, random init).")
    print("2. centroids: " + ", ".join(f"c{j}={c:.3f}" for j, c in enumerate(cents)))
    for j in range(NLIST):
        members = [f"{LABELS[i]}={x[i]:g}" for i in range(len(x)) if assign[i] == j]
        print(f"3. list {j} (centroid {cents[j]:.3f}): {members}")
    print(f"4. exact top-3 for query {QUERY}: {[f'{LABELS[i]}={x[i]:g}' for i in exact3]}")
    for nprobe in (1, 2):
        probed, got = ivf_search_1d(x, cents, assign, QUERY, nprobe, 3)
        missed = [f"{LABELS[i]}={x[i]:g}" for i in exact3 if i not in got]
        print(f"5. nprobe={nprobe}: open lists {probed} "
              f"(|centroid - q| = {[round(abs(cents[j] - QUERY), 3) for j in probed]}) -> "
              f"{[f'{LABELS[i]}={x[i]:g}' for i in got]}  missed: {missed or 'none'}")

    print("\n" + "=" * 72)
    print("PART 3  REAL  Milvus Lite, 8 values x 150 jittered copies = 1200 rows, IVF_FLAT nlist=3")
    print("=" * 72)
    rng = np.random.default_rng(0)
    vecs, owner = [], []
    for _ in range(150):
        for i, v in enumerate(VALUES):
            vecs.append([float(v + rng.uniform(-0.01, 0.01))])
            owner.append(i)
    col = build("toy1200", list(range(len(vecs))), vecs, NLIST)
    print(f"1. progress={utility.index_building_progress('toy1200', 'v')}  (1200 >= 3*39, no FAISS warning)")
    bins = infer_bins(col, VALUES, owner, 1200)
    print(f"2. lists inferred by probing each value at nprobe=1: {len(bins)}")
    for j, m in enumerate(bins):
        vals = [VALUES[i] for i in sorted(m)]
        print(f"   list {j}: {[LABELS[i] + '=' + f'{VALUES[i]:g}' for i in sorted(m)]}  centroid ~ {np.mean(vals):.3f}")
    exact = {i for i, _ in search(col, QUERY, NLIST, 10)}
    print(f"3. exact top-10 for {QUERY} (nprobe=nlist=3) comes from values "
          f"{sorted({VALUES[owner[i]] for i in exact})}")
    for nprobe in (1, 2):
        got = {i for i, _ in search(col, QUERY, nprobe, 10)}
        print(f"4. nprobe={nprobe}: top-10 from values {sorted({VALUES[owner[i]] for i in got})}, "
              f"recall@10 = {len(got & exact) / 10:.2f}")
    connections.disconnect("default")


if __name__ == "__main__":
    sys.exit(main())
