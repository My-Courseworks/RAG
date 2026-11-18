"""M5 worked example, row 7: IVF-PQ, nlist=4096, nprobe=32, 96 sub-quantizers x 8 bits (96 B codes).

Step A checks whether Milvus Lite really builds IVF_PQ (ORM Collection.create_index path): it builds the
index, prints describe_index, then searches at nprobe = 1 / 32 / 4096. A real IVF index loses recall at
nprobe = 1; a silent FLAT fallback gives ~1.0 at every nprobe. Step B is FAISS IndexIVFPQ, in process.

Usage: WORKED_WORK=... python -I worked_example_ivfpq.py milvus|faiss
"""
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worked_example_common import (K, N_CORPUS, latency_summary, load_corpus, recall_at_k,  # noqa: E402
                                   time_queries, update_results, work_dir)

NLIST, NPROBE, PQ_M, PQ_NBITS = 4096, 32, 96, 8


def rss():
    return psutil.Process().memory_info().rss


def milvus():
    import pymilvus
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
    x, q, gt = load_corpus()
    true10 = gt["top100"][:, :K]
    db = work_dir() / "milvus_ivfpq.db"
    for p in work_dir().glob("milvus_ivfpq.db*"):
        if p.is_file():
            p.unlink()
    connections.connect(alias="default", uri=str(db))
    schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True),
                               FieldSchema("vec", DataType.FLOAT_VECTOR, dim=x.shape[1])])
    coll = Collection("dbpedia", schema)
    t0 = time.time()
    for s in range(0, N_CORPUS, 5000):
        e = min(s + 5000, N_CORPUS)
        coll.insert([list(range(s, e)), np.asarray(x[s:e])])
    coll.flush()
    t_insert = time.time() - t0
    params = {"index_type": "IVF_PQ", "metric_type": "IP", "params": {"nlist": NLIST, "m": PQ_M, "nbits": PQ_NBITS}}
    t0 = time.time()
    try:
        coll.create_index("vec", params)
    except pymilvus.exceptions.MilvusException as e:
        import milvus_lite
        res = {"pymilvus": pymilvus.__version__, "milvus_lite": getattr(milvus_lite, "__version__", "?"),
               "requested": params, "insert_s": round(t_insert, 1), "error": str(e),
               "verdict": "Milvus Lite rejects IVF_PQ outright (no silent fallback in this version); row 7 uses FAISS"}
        print("verdict:", res["verdict"], "|", e)
        update_results("milvus_lite_ivfpq_check", res)
        connections.disconnect("default")
        return
    utility.wait_for_index_building_complete("dbpedia")
    t_index = time.time() - t0
    described = [ {"field": i.field_name, "params": i.params} for i in coll.indexes]
    print("describe_index:", described)
    coll.load()
    res = {"pymilvus": pymilvus.__version__, "requested": params, "described": described,
           "insert_s": round(t_insert, 1), "create_index_s": round(t_index, 1), "searches": {}}
    try:
        import milvus_lite
        res["milvus_lite"] = getattr(milvus_lite, "__version__", "?")
    except Exception:
        pass
    for nprobe in (1, 32, 4096):
        def one(v, nprobe=nprobe):
            r = coll.search([v], "vec", {"metric_type": "IP", "params": {"nprobe": nprobe}}, limit=K)
            return [h.id for h in r[0]]
        ids, lat = time_queries(one, q, warmup=20)
        res["searches"][f"nprobe{nprobe}"] = {"recall_at_10": round(recall_at_k(ids, true10), 4), **latency_summary([lat])}
        print(f"  milvus lite nprobe={nprobe}: {res['searches'][f'nprobe{nprobe}']}")
    r1 = res["searches"]["nprobe1"]["recall_at_10"]
    res["verdict"] = ("IVF_PQ honoured (recall drops at nprobe=1)" if r1 < 0.9
                      else "looks like FLAT fallback (recall ~1.0 at nprobe=1)")
    print("verdict:", res["verdict"])
    db_bytes = sum(p.stat().st_size for p in work_dir().rglob("milvus_ivfpq.db*") if p.is_file())
    res["db_file_bytes"] = db_bytes
    update_results("milvus_lite_ivfpq_check", res)
    connections.disconnect("default")


def faiss_run():
    import faiss
    x, q, gt = load_corpus()
    true10 = gt["top100"][:, :K]
    xs = np.ascontiguousarray(np.asarray(x))
    gc.collect()
    rss0 = rss()
    quant = faiss.IndexFlatIP(xs.shape[1])
    index = faiss.IndexIVFPQ(quant, xs.shape[1], NLIST, PQ_M, PQ_NBITS, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    index.train(xs)
    t_train = time.time() - t0
    t0 = time.time()
    index.add(xs)
    t_add = time.time() - t0
    rss_built = rss()
    blob = faiss.serialize_index(index)
    res = {"faiss": faiss.__version__, "threads": faiss.omp_get_max_threads(),
           "index": f"IVF{NLIST},PQ{PQ_M}x{PQ_NBITS} (IndexIVFPQ, inner product, by_residual={bool(index.by_residual)})",
           "train_s": round(t_train, 1), "add_s": round(t_add, 1), "build_s": round(t_train + t_add, 1),
           "serialized_index_bytes": len(blob),
           "rss_delta_build_bytes": rss_built - rss0,
           "arithmetic": {"codes_bytes": N_CORPUS * PQ_M * PQ_NBITS // 8, "ids_bytes": N_CORPUS * 8,
                          "coarse_centroids_bytes": NLIST * xs.shape[1] * 4,
                          "pq_codebooks_bytes": PQ_M * (1 << PQ_NBITS) * (xs.shape[1] // PQ_M) * 4},
           "searches": {}}
    res["arithmetic"]["total_bytes"] = sum(res["arithmetic"].values())
    del blob

    def one(v):
        _, i = index.search(v[None, :], K)
        return i[0].tolist()
    for nprobe in (8, 32, 128, 512):
        index.nprobe = nprobe
        runs = []
        for _ in range(2 if nprobe == NPROBE else 1):
            ids, lat = time_queries(one, q)
            runs.append(lat)
        t0 = time.time()
        index.search(q, K)
        qps = len(q) / (time.time() - t0)
        res["searches"][f"nprobe{nprobe}"] = {"recall_at_10": round(recall_at_k(ids, true10), 4),
                                              **latency_summary(runs), "batch_qps_1000": round(qps, 1)}
        print(f"  faiss IVFPQ nprobe={nprobe}: {res['searches'][f'nprobe{nprobe}']}")
    # Rescore against the float32 originals (IndexRefineFlat): the PQ list as candidates, exact re-rank.
    index.nprobe = NPROBE
    refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(xs))
    for kf in (4, 10):
        refine.k_factor = kf
        def one_r(v):
            _, i = refine.search(v[None, :], K)
            return i[0].tolist()
        ids, lat = time_queries(one_r, q)
        res["searches"][f"nprobe{NPROBE}_refineflat_k{kf}"] = {"recall_at_10": round(recall_at_k(ids, true10), 4),
                                                               **latency_summary([lat]),
                                                               "note": "adds the 1.15 GB float32 originals to RAM"}
        print(f"  faiss IVFPQ nprobe={NPROBE} + RefineFlat k_factor={kf}: {res['searches'][f'nprobe{NPROBE}_refineflat_k{kf}']}")
    # Exact search in process (row 1 engine-free reference).
    flat = faiss.IndexFlatIP(xs.shape[1])
    flat.add(xs)
    def one_f(v):
        _, i = flat.search(v[None, :], K)
        return i[0].tolist()
    runs = []
    for _ in range(2):
        ids, lat = time_queries(one_f, q)
        runs.append(lat)
    res["faiss_flat_exact"] = {"recall_at_10": round(recall_at_k(ids, true10), 4), **latency_summary(runs),
                               "bytes": N_CORPUS * xs.shape[1] * 4}
    print("  faiss IndexFlatIP exact:", res["faiss_flat_exact"])
    update_results("faiss_ivfpq", res)


if __name__ == "__main__":
    {"milvus": milvus, "faiss": faiss_run}[sys.argv[1]]()
