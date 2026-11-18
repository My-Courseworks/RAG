"""Shared helpers for the M5 worked example (recall@10 / latency / memory on real ada-002 vectors).

Paths come from env vars so the scripts never hard-code a scratch directory:
  WORKED_DATA  directory holding the downloaded parquet shards (untrusted data, read-only)
  WORKED_WORK  directory for derived arrays (corpus.npy, queries.npy, gt.npz, ...)
"""
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "supplementary" / "worked-example" / "results.json"

N_CORPUS = 187_043
N_QUERIES = 1_000
DIM = 1536
K = 10
FILTER_SEED = 20240501
FILTER_COUNT = 374  # round(0.002 * 187_043) = 374 -> 0.19996 %


def work_dir() -> Path:
    p = Path(os.environ["WORKED_WORK"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_corpus():
    w = work_dir()
    x = np.load(w / "corpus.npy", mmap_mode="r")
    q = np.load(w / "queries.npy")
    gt = np.load(w / "gt.npz")
    return x, q, gt


def recall_at_k(found_ids, true_ids, k=K):
    """Mean |found[:k] ∩ true[:k]| / k over queries. Rows of true_ids shorter than k are allowed."""
    hits = []
    for f, t in zip(found_ids, true_ids):
        t = [i for i in list(t)[:k] if i >= 0]
        if not t:
            continue
        hits.append(len(set(list(f)[:k]) & set(t)) / len(t))
    return float(np.mean(hits))


def time_queries(search_one, queries, warmup=100):
    """Single client, one query at a time. Returns (ids_per_query, latencies_ms)."""
    for q in queries[:warmup]:
        search_one(q)
    ids, lat = [], []
    for q in queries:
        t0 = time.perf_counter()
        r = search_one(q)
        lat.append((time.perf_counter() - t0) * 1000)
        ids.append(r)
    return ids, lat


def pct(lat, p):
    return float(np.percentile(np.asarray(lat), p))


def latency_summary(runs):
    """runs: list of latency lists (one per repeat). Reports each run and the median of run p50/p99."""
    per = [{"p50_ms": round(pct(r, 50), 3), "p99_ms": round(pct(r, 99), 3), "mean_ms": round(float(np.mean(r)), 3)} for r in runs]
    return {
        "runs": per,
        "p50_ms": round(statistics.median(r["p50_ms"] for r in per), 3),
        "p99_ms": round(statistics.median(r["p99_ms"] for r in per), 3),
    }


def update_results(section: str, payload):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[section] = payload
    RESULTS.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def hnsw_arithmetic(n, dim, m, bytes_per_dim=4, extra_code_bytes=0):
    """In-file arithmetic: vectors + layer-0 links (2m neighbours x 4 B) + upper layers (~1/m of nodes, m links)."""
    vec = n * dim * bytes_per_dim
    links0 = n * 2 * m * 4
    upper = n * (1 / m) * m * 4 * 1.0  # ~n/m nodes above layer 0, m links each -> ~n*4 B; negligible
    codes = n * extra_code_bytes
    return {"vectors_bytes": vec, "links_bytes": int(links0 + upper), "codes_bytes": codes,
            "total_bytes": int(vec + links0 + upper + codes)}
