"""Knowhere IVF_FLAT (via Milvus Lite) scale test for the RAG course, module 5.

Same data generator as every other provider in this series:
187,043 x 256 float32 base vectors from 50 Gaussian topics, 1,000 queries, k=10.

Usage:
    python -I ivf_knowhere_scale.py [--nlist 4096] [--db /path/to/milvus_scale.db]

Prints: insert and build time, index state, recall@10 and ms/query for nprobe in
1..256, Knowhere's own Size() estimate, the on-disk size of the Lite data dir,
and the smallest nprobe reaching recall >= 0.95.

Why the legacy ORM API (connections / Collection) and not MilvusClient:
  Milvus Lite 3.2.2 silently ignores `nlist` when it arrives flattened the way
  pymilvus 3.0.3's MilvusClient.prepare_index_params() sends it (every form:
  params={"nlist": n}, nlist=n, both) and trains with Knowhere's default
  nlist=128. Collection.create_index(field, {"params": {"nlist": n}}) sends the
  nested JSON string and is honoured. Verified by Knowhere's own stderr line
  `WARNING clustering 1200 points to 128 centroids` and by list sizes.

Knowhere facts this script relies on (commit 560fc87, src/index/ivf/ivf.cc):
  - IVF_FLAT search (l.1159-1172): nprobe passed straight to FAISS, max_codes=0,
    no ensure_topk_full (that knob only applies to IVF_FLAT_CC / IVF_SQ_CC / SCANN).
  - Size() for IndexIVFFlat (l.299-301): (nb + nlist) * (code_size + 8) bytes.
"""
import argparse
import logging
import os
import subprocess
import sys
import warnings
import time

import numpy as np

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*ORM-style PyMilvus API.*")  # Lite 3.2.2 needs the ORM path for nlist
from pymilvus import (  # noqa: E402
    Collection, CollectionSchema, DataType, FieldSchema, connections, utility,
)

N, D, NQ, K = 187_043, 256, 1000, 10
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def brute_force_topk(xb, xq, k, q_chunk=100):
    """Exact L2 top-k, chunked over queries so the distance block stays ~150 MB."""
    xb64 = xb.astype(np.float64)
    xb_sq = (xb64 ** 2).sum(1)
    out = np.empty((len(xq), k), dtype=np.int64)
    for s in range(0, len(xq), q_chunk):
        q = xq[s:s + q_chunk].astype(np.float64)
        d = (q ** 2).sum(1)[:, None] + xb_sq[None, :] - 2.0 * q @ xb64.T
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, 1), axis=1)
        out[s:s + q_chunk] = np.take_along_axis(part, order, 1)
    return out


def recall_at_k(found, truth):
    hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth))
    return hits / (len(truth) * truth.shape[1])


def dir_size_mib(path):
    try:
        out = subprocess.run(["du", "-sk", path], capture_output=True, text=True, check=True).stdout
        return int(out.split()[0]) / 1024
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--db", default=os.path.join(os.getcwd(), "milvus_knowhere_scale.db"))
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    print(f"Knowhere IVF_FLAT via Milvus Lite  N={N} D={D} NQ={NQ} K={K} nlist={args.nlist}", flush=True)
    xb, xq = make_data()

    t0 = time.time()
    truth = brute_force_topk(xb, xq, K)
    print(f"ground truth (numpy brute force): {time.time() - t0:.1f}s", flush=True)

    for f in (args.db, args.db + ".lock"):
        if os.path.exists(f):
            os.remove(f)
    connections.connect(uri=args.db)
    print("server:", utility.get_server_version())
    name = f"ivf_scale_nlist{args.nlist}"
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("v", DataType.FLOAT_VECTOR, dim=D),
    ])
    col = Collection(name, schema)

    t0 = time.time()
    for s in range(0, N, args.batch):
        e = min(s + args.batch, N)
        col.insert([list(range(s, e)), xb[s:e].tolist()])
        if (s // args.batch) % 20 == 0:
            print(f"  inserted {e:>7}/{N}  ({time.time() - t0:.0f}s)", flush=True)
    col.flush()
    t_insert = time.time() - t0
    print(f"insert + flush: {t_insert:.1f}s  num_entities={col.num_entities}", flush=True)

    t0 = time.time()
    col.create_index("v", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": args.nlist}})
    utility.wait_for_index_building_complete(name, "v")
    t_build = time.time() - t0
    prog = utility.index_building_progress(name, "v")
    print(f"index build: {t_build:.1f}s  progress={prog}  params echoed={col.indexes[0].params}", flush=True)
    col.load()

    try:
        segs = utility.get_query_segment_info(name)
        print("segments:", [(sg.segmentID, sg.num_rows, sg.index_name) for sg in segs])
    except Exception as e:  # Lite does not implement this RPC
        print(f"segments: not available from Milvus Lite ({type(e).__name__})")

    size_est = (N + args.nlist) * (D * 4 + 8)
    print(f"Knowhere Size() estimate for IVF_FLAT: {size_est / 2**20:.1f} MiB "
          f"(raw vectors alone {N * D * 4 / 2**20:.1f} MiB)")
    print(f"on-disk Lite data dir: {dir_size_mib(args.db):.0f} MiB (includes raw rows + index)")

    print(f"\n{'nprobe':>6} | {'recall@10':>9} | {'ms/query (batch 100)':>20} | {'ms/query (single)':>17}")
    print("-" * 64)
    first_95 = None
    for nprobe in NPROBES:
        sp = {"metric_type": "L2", "params": {"nprobe": nprobe}}
        found = []
        t0 = time.time()
        for s in range(0, NQ, 100):
            res = col.search(xq[s:s + 100].tolist(), "v", sp, limit=K)
            found.extend([[h.id for h in r] for r in res])
        ms_batch = (time.time() - t0) * 1000 / NQ
        t0 = time.time()
        n_single = 100
        for i in range(n_single):
            col.search([xq[i].tolist()], "v", sp, limit=K)
        ms_single = (time.time() - t0) * 1000 / n_single
        rec = recall_at_k(np.array(found), truth)
        if first_95 is None and rec >= 0.95:
            first_95 = nprobe
        print(f"{nprobe:>6} | {rec:>9.4f} | {ms_batch:>20.3f} | {ms_single:>17.3f}", flush=True)

    print(f"\nsmallest nprobe with recall@10 >= 0.95: {first_95}")
    print(f"build time {t_build:.1f}s, insert {t_insert:.1f}s, nlist {args.nlist}")
    connections.disconnect("default")


if __name__ == "__main__":
    sys.exit(main())
