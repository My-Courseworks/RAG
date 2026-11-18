"""pgvector ivfflat scale test: 187,043 x 256 float32, lists in {187, 4096, 1024}, probes sweep.

Runs a REAL Postgres + pgvector via the `pgserver` wheel (bundles PG 16 + pgvector 0.6.2 as of
pgserver 0.1.4). Same data generator as every other provider in the course (seed 0).
Ground truth is brute-force L2 top-10 in numpy. Results are appended to --out as JSON after
each `lists` value so a long 4096 build cannot lose the earlier rows.

    python -I ivf_pgvector_scale.py --lists 187 4096 1024 [--pgdata DIR] [--out results.json]
"""

import argparse
import json
import os
import tempfile
import time

import numpy as np
import pgserver
import psycopg
from pgvector.psycopg import register_vector

N, D, NQ, K = 187_043, 256, 1000, 10
PROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def make_data():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:], topic[:N], topic[N:]


def ground_truth(xb, xq, chunk=20_000):
    """Exact L2 top-K ids per query (0-based), chunked matmul."""
    qn = (xq * xq).sum(1)[:, None]
    best_d = np.full((NQ, K), np.inf, dtype=np.float32)
    best_i = np.zeros((NQ, K), dtype=np.int64)
    for s in range(0, N, chunk):
        b = xb[s:s + chunk]
        d = qn + (b * b).sum(1)[None, :] - 2.0 * xq @ b.T
        cand_d = np.concatenate([best_d, d], axis=1)
        cand_i = np.concatenate([best_i, np.arange(s, s + len(b))[None, :].repeat(NQ, 0)], axis=1)
        part = np.argpartition(cand_d, K - 1, axis=1)[:, :K]
        best_d = np.take_along_axis(cand_d, part, 1)
        best_i = np.take_along_axis(cand_i, part, 1)
    return best_i


def load_table(conn, xb, topic):
    n = conn.execute("SELECT count(*) FROM items").fetchone()[0] if conn.execute(
        "SELECT to_regclass('items')").fetchone()[0] else -1
    if n == N:
        print(f"table items already holds {N} rows; reusing")
        return
    conn.execute("DROP TABLE IF EXISTS items")
    conn.execute(f"CREATE TABLE items (id int PRIMARY KEY, topic int, v vector({D}))")
    t0 = time.perf_counter()
    with conn.cursor() as cur, cur.copy("COPY items (id, topic, v) FROM STDIN WITH (FORMAT BINARY)") as cp:
        cp.set_types(["int4", "int4", "vector"])
        for i in range(N):
            cp.write_row((i, int(topic[i]), xb[i]))
    print(f"COPY {N} rows: {time.perf_counter()-t0:.1f}s; table {conn.execute('SELECT pg_size_pretty(pg_relation_size(%s))', ('items',)).fetchone()[0]}")


def build_index(conn, lists):
    conn.execute("DROP INDEX IF EXISTS items_ivf")
    conn.execute("SET maintenance_work_mem = '8GB'")          # Elkan lower-bound matrix is samples x lists floats
    conn.execute("SET max_parallel_maintenance_workers = 4")  # assignment/sort phase only; k-means is serial
    t0 = time.perf_counter()
    conn.execute(f"CREATE INDEX items_ivf ON items USING ivfflat (v vector_l2_ops) WITH (lists = {lists})")
    dt = time.perf_counter() - t0
    size = conn.execute("SELECT pg_relation_size('items_ivf')").fetchone()[0]
    return dt, size


def sweep(conn, xq, gt, lists):
    conn.execute("SET enable_seqscan = off")  # keep the planner on the index even at probes ~ lists
    rows = []
    for probes in PROBES:
        if probes > lists:
            break
        conn.execute(f"SET ivfflat.probes = {probes}")
        hits = 0
        t0 = time.perf_counter()
        for qi in range(NQ):
            got = conn.execute("SELECT id FROM items ORDER BY v <-> %s LIMIT %s", (xq[qi], K)).fetchall()
            hits += len(set(g[0] for g in got) & set(gt[qi].tolist()))
        ms = (time.perf_counter() - t0) * 1000 / NQ
        rec = hits / (NQ * K)
        rows.append({"lists": lists, "probes": probes, "recall@10": round(rec, 4), "ms/query": round(ms, 2)})
        print(f"  lists={lists:5d} probes={probes:4d} recall@10={rec:.4f} ms/query={ms:7.2f}")
    return rows


def filter_demo(conn, xq, topic_q, lists):
    """Post-filtering: WHERE is applied to the rows the probed lists yielded, so a selective
    filter starves LIMIT 10. Query 0's own topic vs a different topic."""
    conn.execute("SET enable_seqscan = off")
    out = {}
    other = int((topic_q[0] + 1) % 50)
    for probes in (1, 4, 16, 64):
        conn.execute(f"SET ivfflat.probes = {probes}")
        same = conn.execute("SELECT count(*) FROM (SELECT id FROM items WHERE topic = %s ORDER BY v <-> %s LIMIT 10) s",
                            (int(topic_q[0]), xq[0])).fetchone()[0]
        diff = conn.execute("SELECT count(*) FROM (SELECT id FROM items WHERE topic = %s ORDER BY v <-> %s LIMIT 10) s",
                            (other, xq[0])).fetchone()[0]
        out[probes] = (same, diff)
        print(f"  filter demo lists={lists} probes={probes}: rows returned for LIMIT 10 -> same-topic {same}, other-topic {diff}")
    try:
        conn.execute("SET ivfflat.iterative_scan = relaxed_order")
        conn.execute("SET ivfflat.probes = 1")
        it = conn.execute("SELECT count(*) FROM (SELECT id FROM items WHERE topic = %s ORDER BY v <-> %s LIMIT 10) s",
                          (other, xq[0])).fetchone()[0]
        print(f"  iterative_scan=relaxed_order probes=1 other-topic rows: {it}")
        conn.execute("SET ivfflat.iterative_scan = off")
        out["iterative"] = it
    except psycopg.Error as e:
        print(f"  ivfflat.iterative_scan unavailable in this pgvector build: {str(e).strip()}")
        out["iterative"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", type=int, nargs="+", default=[187, 4096, 1024])
    ap.add_argument("--pgdata", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.pgdata is None:
        args.pgdata = tempfile.mkdtemp(prefix="pgdata-ivf-scale-")
    if args.out is None:
        args.out = os.path.join(args.pgdata, "ivf_pgvector_scale_results.json")

    xb, xq, topic_b, topic_q = make_data()
    t0 = time.perf_counter()
    gt = ground_truth(xb, xq)
    print(f"ground truth (numpy brute force): {time.perf_counter()-t0:.1f}s")

    srv = pgserver.get_server(args.pgdata)
    results = {"env": {}, "sweep": [], "builds": {}, "filter": {}}
    try:
        with psycopg.connect(srv.get_uri(), autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            results["env"] = {
                "postgres": conn.execute("SHOW server_version").fetchone()[0],
                "pgvector": conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0],
            }
            print(results["env"])
            load_table(conn, xb, topic_b)
            for lists in args.lists:
                print(f"\n== lists={lists} (pgvector README guidance for N={N}: rows/1000 ~ {N//1000}) ==")
                dt, size = build_index(conn, lists)
                print(f"  build {dt:.1f}s, index size {size/2**20:.1f} MiB")
                results["builds"][lists] = {"build_s": round(dt, 1), "index_mib": round(size / 2**20, 1)}
                results["sweep"] += sweep(conn, xq, gt, lists)
                results["filter"][lists] = filter_demo(conn, xq, topic_q, lists)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)
            print("\n| lists | probes | recall@10 | ms/query |\n|---|---|---|---|")
            for r in results["sweep"]:
                print(f"| {r['lists']} | {r['probes']} | {r['recall@10']:.4f} | {r['ms/query']:.2f} |")
            for lists in args.lists:
                ok = [r for r in results["sweep"] if r["lists"] == lists and r["recall@10"] >= 0.95]
                print(f"lists={lists}: smallest probes with recall>=0.95: {ok[0]['probes'] if ok else 'none in sweep'}")
            print(f"results written to {args.out}")
    finally:
        srv.cleanup()


if __name__ == "__main__":
    main()
