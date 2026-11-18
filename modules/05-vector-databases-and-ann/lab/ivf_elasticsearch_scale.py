"""Elasticsearch DiskBBQ (`bbq_disk`) scale test for the RAG course, module 5.

Runs against a real single-node Elasticsearch 9.2.0 (REST on localhost:9200):

    docker run -d --name es-ivf -m 4g -p 9200:9200 -e discovery.type=single-node \
        -e xpack.security.enabled=false -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
        docker.elastic.co/elasticsearch/elasticsearch:9.2.0

Data generator is the one shared by every provider in the course (N=187,043, D=256).
`bbq_disk` is parameterised by vectors-per-cluster (`cluster_size`, min 64), not by
`nlist`; the course's nlist=4096 would need cluster_size ~ 46 which ES rejects, so the
closest allowed is cluster_size=64 (nlist ~ 2900). The sweep dial is `visit_percentage`
(percent of vectors to score per shard), which we derive from an `nprobe`-equivalent as
nprobe * cluster_size / N * 100.

Usage:  python -I ivf_elasticsearch_scale.py [--es http://localhost:9200] [--cluster-sizes 64,183]
"""

import argparse
import json
import sys
import time

import numpy as np
import orjson
import requests

N, D, NQ, K = 187_043, 256, 1000, 10
NPROBE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def ground_truth(xb, xq, k=K, chunk=100):
    """Brute-force L2 top-k. Returns (NQ, k) int array of base indices."""
    xb_sq = (xb.astype(np.float32) ** 2).sum(1)
    out = np.empty((len(xq), k), dtype=np.int64)
    for s in range(0, len(xq), chunk):
        q = xq[s : s + chunk]
        d = xb_sq[None, :] - 2.0 * (q @ xb.T) + (q**2).sum(1)[:, None]
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, 1), axis=1)
        out[s : s + chunk] = np.take_along_axis(part, order, 1)
    return out


def recall_at_k(found, gt):
    return float(np.mean([len(set(f) & set(g)) / len(g) for f, g in zip(found, gt)]))


class ES:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.s = requests.Session()

    def req(self, method, path, body=None, raw=None, **kw):
        headers = {"Content-Type": "application/json"}
        if raw is not None:
            headers["Content-Type"] = "application/x-ndjson"
        r = self.s.request(method, self.base + path, data=raw if raw is not None else (orjson.dumps(body) if body is not None else None), headers=headers, **kw)
        if r.status_code >= 300:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:800]}")
        return r.json() if r.text else {}


def build_index(es, name, xb, cluster_size):
    es.s.delete(es.base + "/" + name)
    es.req(
        "PUT",
        f"/{name}",
        {
            # compound_format=false keeps per-extension file sizes visible (.cenivf/.clivf/.mivf/.vec)
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "index.compound_format": False},
            "mappings": {
                "properties": {
                    "v": {
                        "type": "dense_vector",
                        "dims": D,
                        "index": True,
                        "similarity": "l2_norm",
                        "index_options": {"type": "bbq_disk", "cluster_size": cluster_size},
                    }
                }
            },
        },
    )
    t0 = time.time()
    batch = 1000
    for s in range(0, len(xb), batch):
        lines = []
        for i in range(s, min(s + batch, len(xb))):
            lines.append(orjson.dumps({"index": {"_id": i}}))
            lines.append(orjson.dumps({"v": xb[i]}, option=orjson.OPT_SERIALIZE_NUMPY))
        resp = es.req("POST", f"/{name}/_bulk", raw=b"\n".join(lines) + b"\n")
        if resp.get("errors"):
            bad = [it for it in resp["items"] if "error" in it["index"]][:2]
            raise RuntimeError(f"bulk errors: {bad}")
        if (s // batch) % 40 == 0:
            print(f"  indexed {s + batch:>7}/{len(xb)}  {time.time() - t0:6.1f}s", flush=True)
    es.req("POST", f"/{name}/_refresh")
    t_bulk = time.time() - t0
    # Every flush/merge re-runs hierarchical k-means on its own segment; force a single
    # segment so the index is one IVF partition like a FAISS index.
    t1 = time.time()
    es.req("POST", f"/{name}/_forcemerge?max_num_segments=1", timeout=3600)
    es.req("POST", f"/{name}/_refresh")
    t_merge = time.time() - t1
    segs = es.req("GET", f"/{name}/_segments")
    nseg = sum(len(sh[0]["segments"]) for sh in segs["indices"][name]["shards"].values())
    stats = es.req("GET", f"/{name}/_stats/store,segments?include_segment_file_sizes=true")
    prim = stats["indices"][name]["primaries"]
    files = {k: v["size_in_bytes"] for k, v in prim["segments"].get("file_sizes", {}).items()}
    return {"t_bulk_s": t_bulk, "t_forcemerge_s": t_merge, "segments": nseg, "store_bytes": prim["store"]["size_in_bytes"], "file_sizes": files}


def search(es, name, xq, gt, visit_pct, oversample, num_candidates=None):
    found, took_ms = [], 0.0
    t0 = time.time()
    for q in xq:
        knn = {"field": "v", "query_vector": q.tolist(), "k": K, "rescore_vector": {"oversample": oversample}}
        if visit_pct is not None:
            knn["visit_percentage"] = visit_pct
        if num_candidates is not None:
            knn["num_candidates"] = num_candidates
        r = es.req("POST", f"/{name}/_search", {"knn": knn, "size": K, "_source": False})
        took_ms += r["took"]
        found.append([int(h["_id"]) for h in r["hits"]["hits"]])
    wall = (time.time() - t0) * 1000 / len(xq)
    return recall_at_k(found, gt), took_ms / len(xq), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", default="http://localhost:9200")
    ap.add_argument("--cluster-sizes", default="64,183", help="vectors per cluster; 64 ~ nlist 2922, 183 ~ nlist 1022")
    ap.add_argument("--nq", type=int, default=NQ)
    ap.add_argument("--out", default=None, help="optional JSON results path")
    args = ap.parse_args()

    es = ES(args.es)
    info = es.req("GET", "/")
    print(f"Elasticsearch {info['version']['number']} (lucene {info['version']['lucene_version']})")

    xb, xq = make_data()
    xq = xq[: args.nq]
    t0 = time.time()
    gt = ground_truth(xb, xq)
    print(f"ground truth: brute-force L2 top-{K} for {len(xq)} queries in {time.time() - t0:.1f}s")

    results = {}
    for cs in [int(c) for c in args.cluster_sizes.split(",")]:
        name = f"ivf-scale-cs{cs}"
        nlist_equiv = round(N / cs)
        print(f"\n=== bbq_disk cluster_size={cs}  (nlist equivalent ~ {nlist_equiv}) ===")
        b = build_index(es, name, xb, cs)
        print(f"build: bulk {b['t_bulk_s']:.1f}s + forcemerge(k-means on one segment) {b['t_forcemerge_s']:.1f}s; segments={b['segments']}")
        print(f"store {b['store_bytes'] / 1e6:.1f} MB; files (MB): " + ", ".join(f"{k}={v / 1e6:.2f}" for k, v in sorted(b["file_sizes"].items(), key=lambda kv: -kv[1])))
        # Top-level `knn` runs per shard with k = num_candidates (KnnSearchBuilder.toQueryBuilder, 9.2.0 L501),
        # so the pool that gets rescored with raw floats is ceil(num_candidates * oversample).
        # oversample=0 disables rescoring (pure 1-bit ranking); 3 is the bbq_disk default; 15 is ES's default num_candidates (1.5k).
        configs = [("1bit only: oversample=0, nc=100", 0, 100), ("defaults: oversample=3, nc=15", 3, 15), ("oversample=3, nc=100", 3, 100), ("oversample=3, nc=500", 3, 500)]
        rows = []
        hdr = f"{'nprobe~':>7} {'visit%':>8} |" + " |".join(f" {'recall':>6} {'took':>5} {'wall':>5}" for _ in configs)
        print(hdr)
        print(" " * 18 + "  ".join(f"[{c[0]}]" for c in configs))
        for npb in NPROBE_SWEEP:
            pct = npb * cs / N * 100
            cells = [search(es, name, xq, gt, pct, ov, num_candidates=nc) for _, ov, nc in configs]
            rows.append({"nprobe_equiv": npb, "visit_percentage": pct, "cells": cells})
            print(f"{npb:>7} {pct:>8.3f} |" + " |".join(f" {r:>6.4f} {t:>5.2f} {w:>5.2f}" for r, t, w in cells))
        # ES default dial: visit_percentage unset -> dynamic: log10(N)^2 * max(num_candidates, 5k) vectors per shard
        rd, td, wd = search(es, name, xq, gt, None, 3, num_candidates=100)
        dyn_expected = round(np.log10(N) ** 2 * max(100, 5 * K))
        print(f"default dial (visit_percentage unset, num_candidates=100 -> ~{dyn_expected} vectors = {dyn_expected / N * 100:.2f}%, oversample=3): recall@10={rd:.4f} took={td:.2f}ms wall={wd:.2f}ms")
        for ci, (label, _, _) in enumerate(configs):
            hit = next((r for r in rows if r["cells"][ci][0] >= 0.95), None)
            print(f"smallest nprobe-equivalent reaching recall>=0.95 [{label}]: " + (f"{hit['nprobe_equiv']} (visit_percentage={hit['visit_percentage']:.3f}%)" if hit else "none in sweep"))
        results[cs] = {"build": b, "configs": [c[0] for c in configs], "rows": rows, "default_dial": [rd, td, wd]}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
