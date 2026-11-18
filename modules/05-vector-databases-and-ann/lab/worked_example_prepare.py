"""Step 1 of the M5 worked example: real ada-002 vectors -> corpus, queries, exact ground truth, filters.

Corpus: Hugging Face KShivendu/dbpedia-entities-openai-1M (MIT), revision
af9b8869cc2d8debbd254d77737865bb09a2067f, shards train-00000..00004-of-00026.
Rows are taken in shard order: the first 187,043 are the corpus, the next 1,000 the queries.

Usage:
  WORKED_DATA=<dir with the 5 parquet shards> WORKED_WORK=<derived dir> python -I worked_example_prepare.py
"""
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worked_example_common import (FILTER_COUNT, FILTER_SEED, K, N_CORPUS, N_QUERIES, DIM,  # noqa: E402
                                   update_results, work_dir)

REVISION = "af9b8869cc2d8debbd254d77737865bb09a2067f"
SHARDS = [
    "train-00000-of-00026-3c7b99d1c7eda36e.parquet",
    "train-00001-of-00026-2b24035a6390fdcb.parquet",
    "train-00002-of-00026-b05ce48965853dad.parquet",
    "train-00003-of-00026-d116c3c239aa7895.parquet",
    "train-00004-of-00026-5c2bcfc39e1019cd.parquet",
]
# Real-metadata filter candidates: DBpedia title disambiguators. The one closest to 0.2 % is used.
TITLE_SUFFIXES = ["(album)", "(film)", "(band)", "(song)", "(musician)", "(footballer)", "(politician)",
                  "(novel)", "(TV series)", "(video game)", "(magazine)", "(river)", "(actor)", "(singer)"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(data_dir, n_needed):
    vecs, titles, ids, shard_info = [], [], [], []
    got = 0
    for name in SHARDS:
        p = Path(data_dir) / name
        t = pq.read_table(p, columns=["_id", "title", "openai"])
        shard_info.append({"file": name, "rows": t.num_rows, "sha256": sha256(p)})
        take = min(t.num_rows, n_needed - got)
        t = t.slice(0, take)
        emb = t.column("openai").combine_chunks()
        flat = emb.values.to_numpy(zero_copy_only=False).astype(np.float32)
        vecs.append(flat.reshape(take, -1))
        titles += t.column("title").to_pylist()
        ids += t.column("_id").to_pylist()
        got += take
        if got >= n_needed:
            break
    return np.concatenate(vecs), titles, ids, shard_info


def exact_topk(x, q, k, mask=None, chunk=50_000):
    """Exact cosine top-k (vectors are pre-normalised, so dot product)."""
    best_s = np.full((len(q), k), -np.inf, dtype=np.float32)
    best_i = np.full((len(q), k), -1, dtype=np.int64)
    for s in range(0, len(x), chunk):
        xs = np.asarray(x[s:s + chunk])
        sc = q @ xs.T
        if mask is not None:
            sc[:, ~mask[s:s + chunk]] = -np.inf
        cat_s = np.concatenate([best_s, sc], axis=1)
        cat_i = np.concatenate([best_i, np.broadcast_to(np.arange(s, s + len(xs)), sc.shape)], axis=1)
        top = np.argpartition(-cat_s, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(cat_s, top, 1)
        best_i = np.take_along_axis(cat_i, top, 1)
    order = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_i, order, 1), np.take_along_axis(best_s, order, 1)


def main():
    t0 = time.time()
    x_all, titles, ids, shard_info = read_rows(os.environ["WORKED_DATA"], N_CORPUS + N_QUERIES)
    assert x_all.shape == (N_CORPUS + N_QUERIES, DIM), x_all.shape
    norms = np.linalg.norm(x_all, axis=1)
    print(f"read {x_all.shape} in {time.time() - t0:.1f}s; norm min/mean/max {norms.min():.6f} {norms.mean():.6f} {norms.max():.6f}")
    x_all /= norms[:, None]
    x, q = x_all[:N_CORPUS], np.ascontiguousarray(x_all[N_CORPUS:])
    w = work_dir()
    np.save(w / "corpus.npy", x)
    np.save(w / "queries.npy", q)

    # Simulated filter: exactly FILTER_COUNT random corpus vectors, fixed seed. Real vectors, simulated attribute.
    rng = np.random.default_rng(FILTER_SEED)
    sim = np.zeros(N_CORPUS, dtype=bool)
    sim[rng.choice(N_CORPUS, FILTER_COUNT, replace=False)] = True

    # Real-metadata filter: the title disambiguator whose share is closest to 0.2 %.
    tc = titles[:N_CORPUS]
    counts = {s: sum(t.endswith(s) for t in tc) for s in TITLE_SUFFIXES}
    print("title-suffix counts:", counts)
    suffix = min(counts, key=lambda s: abs(counts[s] - FILTER_COUNT))
    real = np.array([t.endswith(suffix) for t in tc])
    np.save(w / "filter_sim.npy", sim)
    np.save(w / "filter_real.npy", real)

    t1 = time.time()
    gt_i, gt_s = exact_topk(x, q, 100)
    t_gt = time.time() - t1
    gs_i, _ = exact_topk(x, q, K, mask=sim)
    gr_i, _ = exact_topk(x, q, K, mask=real)
    np.savez(w / "gt.npz", top100=gt_i, top100_scores=gt_s, sim_top10=gs_i, real_top10=gr_i)

    # Query-to-filter correlation: how many of a query's exact unfiltered top-100 pass each filter.
    in_sim = sim[gt_i].sum(1)
    in_real = real[gt_i].sum(1)
    summary = {
        "dataset": "KShivendu/dbpedia-entities-openai-1M",
        "url": "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M",
        "license": "MIT (dataset card); source text: BeIR/dbpedia-entity",
        "revision": REVISION,
        "shards": shard_info,
        "corpus_rows": [0, N_CORPUS - 1],
        "query_rows": [N_CORPUS, N_CORPUS + N_QUERIES - 1],
        "first_corpus_id": ids[0], "last_corpus_id": ids[N_CORPUS - 1],
        "first_query_id": ids[N_CORPUS], "last_query_id": ids[-1],
        "norm_before_normalising": {"min": float(norms.min()), "mean": float(norms.mean()), "max": float(norms.max())},
        "distance": "cosine (vectors L2-normalised; dot product)",
        "exact_gt_numpy_seconds_1000_queries": round(t_gt, 2),
        "filter_simulated": {"seed": FILTER_SEED, "count": int(sim.sum()), "share": float(sim.mean())},
        "filter_real": {"rule": f"title endswith '{suffix}'", "count": int(real.sum()), "share": float(real.mean()),
                        "candidates": counts},
        "filter_correlation": {
            "sim_mean_in_unfiltered_top100": float(in_sim.mean()),
            "real_mean_in_unfiltered_top100": float(in_real.mean()),
            "real_queries_with_any_in_top100": int((in_real > 0).sum()),
        },
    }
    update_results("corpus", summary)
    for k_, v in summary.items():
        print(k_, ":", v)


if __name__ == "__main__":
    main()
