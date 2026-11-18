"""M5 worked example, row 8: a ~0.2 % filter applied after an HNSW search (pgvector), and the fixes.

pgvector HNSW m=16, ef_construction=100 (matching the Qdrant rows). Filter columns have no B-tree index at
first, so the planner can only walk the HNSW index and drop non-matching rows (post-filter). Then:
  - hnsw.ef_search raised to its 1,000 maximum,
  - hnsw.iterative_scan = strict_order / relaxed_order (pgvector >= 0.8),
  - a B-tree on the filter column, letting the planner pre-filter and sort exactly.
Recall is against exact search over the filtered subset (prepare.py).

Usage: WORKED_WORK=... python -I worked_example_pgvector.py
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worked_example_common import (K, N_CORPUS, latency_summary, load_corpus, recall_at_k,  # noqa: E402
                                   time_queries, update_results, work_dir)

IMAGE = "pgvector/pgvector:0.8.7-pg16"
CONTAINER = "m5we-pg"
PORT = 55432
DSN = f"postgresql://postgres:pw@localhost:{PORT}/postgres"


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def start():
    subprocess.run(f"docker rm -f {CONTAINER}", shell=True, capture_output=True)
    sh(f"docker run -d --name {CONTAINER} -e POSTGRES_PASSWORD=pw -p {PORT}:5432 --shm-size=4g {IMAGE} "
       "-c shared_buffers=3GB -c maintenance_work_mem=3GB -c max_parallel_maintenance_workers=7 -c work_mem=64MB")
    for _ in range(120):
        try:
            with psycopg.connect(DSN, connect_timeout=2) as c:
                c.execute("select 1")
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("postgres not ready")


def main():
    x, q, gt = load_corpus()
    w = work_dir()
    sim, band = np.load(w / "filter_sim.npy"), np.load(w / "filter_real.npy")
    start()
    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
    pgver = conn.execute("SHOW server_version").fetchone()[0]
    conn.execute("CREATE TABLE items (id int PRIMARY KEY, sim int, band int, emb vector(1536))")
    t0 = time.time()
    with conn.cursor() as cur, cur.copy("COPY items (id, sim, band, emb) FROM STDIN WITH (FORMAT BINARY)") as cp:
        cp.set_types(["int4", "int4", "int4", "vector"])
        for i in range(N_CORPUS):
            cp.write_row((i, int(sim[i]), int(band[i]), np.asarray(x[i])))
    t_load = time.time() - t0
    conn.execute("VACUUM ANALYZE items")
    t0 = time.time()
    conn.execute("CREATE INDEX items_hnsw ON items USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 100)")
    t_index = time.time() - t0
    sizes = dict(zip(["table_bytes", "table_total_bytes", "hnsw_index_bytes"], conn.execute(
        "SELECT pg_relation_size('items'), pg_total_relation_size('items'), pg_relation_size('items_hnsw')").fetchone()))
    print(f"pgvector {ver} on PG {pgver}: load {t_load:.1f}s, hnsw build {t_index:.1f}s, sizes {sizes}")
    res = {"image": IMAGE, "pgvector": ver, "postgres": pgver, "load_s": round(t_load, 1),
           "hnsw_build_s": round(t_index, 1), "hnsw": "m=16, ef_construction=100, vector_cosine_ops",
           "settings": "shared_buffers=3GB, maintenance_work_mem=3GB, max_parallel_maintenance_workers=7",
           "sizes": sizes, "arithmetic": {"vectors_bytes": N_CORPUS * (1536 * 4 + 8),
                                          "hnsw_layer0_links_bytes": N_CORPUS * 2 * 16 * 6},
           "searches": {}, "plans": {}}
    S = res["searches"]
    true10 = gt["top100"][:, :K]
    # Warm the index and heap into shared buffers once.
    conn.execute("SELECT count(*) FROM items WHERE emb IS NOT NULL")

    def run(label, sql, true, settings, repeats=1):
        for s in settings:
            conn.execute(s)
        cur = conn.cursor()
        # prepare=False: a prepared statement keeps the plan made under the settings of its first run
        # (planner GUCs do not invalidate cached plans), which silently turns a forced plan back into a seq scan.
        def one(v):
            return [r[0] for r in cur.execute(sql, (v,), prepare=False).fetchall()]
        runs, ids = [], None
        for _ in range(repeats):
            ids, lat = time_queries(one, q, warmup=50)
            runs.append(lat)
        out = {"settings": settings, "recall_at_10": round(recall_at_k(ids, true), 4),
               "mean_rows_returned": round(float(np.mean([len(i) for i in ids])), 2),
               "queries_with_zero_rows": int(sum(len(i) == 0 for i in ids)), **latency_summary(runs)}
        plan = "\n".join(r[0] for r in conn.execute("EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, BUFFERS) " + sql, (q[0],)).fetchall())
        res["plans"][label] = plan
        S[label] = out
        print(f"  {label:<44} recall@10={out['recall_at_10']:.4f} rows={out['mean_rows_returned']:5.2f} "
              f"p50={out['p50_ms']:.2f} p99={out['p99_ms']:.2f}  | {' / '.join(l.strip() for l in plan.splitlines()[:4])}")

    base = ["SET hnsw.iterative_scan = off", "SET hnsw.max_scan_tuples = 20000"]
    unf = "SELECT id FROM items ORDER BY emb <=> %s LIMIT 10"
    for ef in (40, 64, 256):
        run(f"unfiltered_ef{ef}", unf, true10, base + [f"SET hnsw.ef_search = {ef}"], repeats=2 if ef == 64 else 1)
    # Without any index on the filter column the planner still avoids the HNSW index: it estimates ~400
    # matching rows, so a sequential scan + top-10 sort (exact) is cheaper. Measure that default first, then
    # force the post-filter plan (Index Scan using items_hnsw + Filter) with enable_seqscan/bitmapscan off.
    for field, key in (("sim", "sim_top10"), ("band", "real_top10")):
        sql = f"SELECT id FROM items WHERE {field} = 1 ORDER BY emb <=> %s LIMIT 10"
        run(f"{field}_default_planner_no_btree", sql, gt[key], base + ["SET hnsw.ef_search = 40"])
    force = ["SET enable_seqscan = off", "SET enable_bitmapscan = off"]
    for field, key in (("sim", "sim_top10"), ("band", "real_top10")):
        sql = f"SELECT id FROM items WHERE {field} = 1 ORDER BY emb <=> %s LIMIT 10"
        true = gt[key]
        for ef in (40, 64, 1000):
            run(f"{field}_postfilter_ef{ef}", sql, true, base + force + [f"SET hnsw.ef_search = {ef}"], repeats=2 if ef == 40 else 1)
        for mode in ("strict_order", "relaxed_order"):
            for mst in (20000, 200000):
                run(f"{field}_iterative_{mode}_ef40_max{mst}", sql, true,
                    force + [f"SET hnsw.iterative_scan = {mode}", f"SET hnsw.max_scan_tuples = {mst}", "SET hnsw.ef_search = 40"])
    conn.execute("RESET enable_seqscan")
    conn.execute("RESET enable_bitmapscan")
    t0 = time.time()
    conn.execute("CREATE INDEX items_sim ON items (sim)")
    conn.execute("CREATE INDEX items_band ON items (band)")
    conn.execute("ANALYZE items")
    res["btree_build_s"] = round(time.time() - t0, 1)
    for field, key in (("sim", "sim_top10"), ("band", "real_top10")):
        sql = f"SELECT id FROM items WHERE {field} = 1 ORDER BY emb <=> %s LIMIT 10"
        run(f"{field}_with_btree_planner_ef40", sql, gt[key], base + ["SET hnsw.ef_search = 40"], repeats=2)
    res["docker_stats_mem_usage"] = sh(f"docker stats --no-stream --format '{{{{.MemUsage}}}}' {CONTAINER}").strip()
    update_results("pgvector", res)
    conn.close()


if __name__ == "__main__":
    main()
