"""pgvector ivfflat toy: eight 1-D values, lists = 3, probes = 1 vs 2.

Runs a REAL Postgres + pgvector (bundled by the `pgserver` wheel) — not a simulation.
pgvector's k-means seeds from Postgres' per-backend PRNG (`pg_global_prng_state`,
src/ivfkmeans.c InitCenters), which SQL cannot seed, so the index is rebuilt
`--builds` times and the partitions that k-means produced are tabulated. The
most frequent partition is then narrated step by step. Centroids are read from
the index's list page on disk (block 1, struct IvfflatListData in src/ivfflat.h)
because pgvector has no SQL introspection for them.

    python -I ivf_pgvector_toy.py [--builds 25] [--pgdata DIR]
"""

import argparse
import collections
import os
import struct
import sys
import tempfile
import time

import pgserver
import psycopg

VALUES = {"A": 1, "B": 2, "C": 4, "D": 7, "E": 8, "F": 11, "G": 14, "H": 15}
LISTS = 3
QUERIES = [9.5, 5.5]
K = 3


def read_centroids(pgdata_dir, conn, index_name):
    """Parse block 1 (IVFFLAT_HEAD_BLKNO) of the index file: one IvfflatListData per list."""
    conn.execute("CHECKPOINT")
    relpath = conn.execute("SELECT pg_relation_filepath(%s)", (index_name,)).fetchone()[0]
    datadir = conn.execute("SHOW data_directory").fetchone()[0]
    with open(os.path.join(datadir, relpath), "rb") as f:
        f.seek(8192)
        page = f.read(8192)
    # PageHeaderData is 24 bytes; pd_lower (uint16 at offset 12) ends the line-pointer array.
    pd_lower = struct.unpack_from("<H", page, 12)[0]
    centroids = []
    for off in range(24, pd_lower, 4):
        lp = struct.unpack_from("<I", page, off)[0]
        lp_off, lp_flags, lp_len = lp & 0x7FFF, (lp >> 15) & 3, lp >> 17
        if lp_flags != 1:  # LP_NORMAL
            continue
        start_page, insert_page = struct.unpack_from("<II", page, lp_off)
        dim = struct.unpack_from("<h", page, lp_off + 8 + 4)[0]  # Vector: vl_len_(4) dim(2) unused(2) x[]
        xs = struct.unpack_from("<%df" % dim, page, lp_off + 8 + 8)
        centroids.append((start_page, xs[0]))
    # Lists are laid out in centroid order 0..lists-1; sort by startPage to recover that order.
    centroids.sort()
    return [c for _, c in centroids]


def bins_at_probes1(conn):
    """probes=1 at a value's own position returns exactly the list it lives in."""
    conn.execute("SET ivfflat.probes = 1")
    conn.execute("SET enable_seqscan = off")
    bins = {}
    for name, v in VALUES.items():
        rows = conn.execute("SELECT v FROM toy ORDER BY v <-> %s::vector LIMIT 8", (f"[{v}]",)).fetchall()
        bins[name] = tuple(sorted(int(float(r[0].strip("[]"))) for r in rows))
    return bins


def topk(conn, q, probes, k=K):
    conn.execute(f"SET ivfflat.probes = {probes}")
    conn.execute("SET enable_seqscan = off")
    rows = conn.execute("SELECT v, v <-> %s::vector FROM toy ORDER BY 2 LIMIT %s", (f"[{q}]", k)).fetchall()
    return [(int(float(r[0].strip("[]"))), float(r[1])) for r in rows]


def exact_topk(q, k=K):
    return sorted(((abs(v - q), v) for v in VALUES.values()))[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", type=int, default=25)
    ap.add_argument("--pgdata", default=None, help="Postgres data dir (default: a fresh temp dir)")
    args = ap.parse_args()
    if args.pgdata is None:
        args.pgdata = tempfile.mkdtemp(prefix="pgdata-ivf-toy-")

    srv = pgserver.get_server(args.pgdata)
    try:
        with psycopg.connect(srv.get_uri(), autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
            pgver = conn.execute("SHOW server_version").fetchone()[0]
            print(f"PostgreSQL {pgver}, pgvector {ver}")
            conn.execute("DROP TABLE IF EXISTS toy")
            conn.execute("CREATE TABLE toy (id text PRIMARY KEY, v vector(1))")
            for name, v in VALUES.items():
                conn.execute("INSERT INTO toy VALUES (%s, %s::vector)", (name, f"[{v}]"))

            partitions = collections.Counter()
            samples = {}
            notices = []
            conn.add_notice_handler(lambda d: notices.append(d.message_primary))
            # pgvector warns only when fewer samples than lists (src/ivfbuild.c ComputeCenters).
            conn.execute("CREATE INDEX toy_ivf ON toy USING ivfflat (v vector_l2_ops) WITH (lists = 10)")
            print(f"lists=10 on 8 rows -> NOTICE: {notices}")
            # custom GUCs appear only once the extension's .so is loaded by this backend
            print("ivfflat GUCs (name, default, min, max):", conn.execute(
                "SELECT name, boot_val, min_val, max_val FROM pg_settings WHERE name LIKE 'ivfflat.%'").fetchall())
            conn.execute("DROP INDEX toy_ivf")
            notices.clear()
            for b in range(args.builds):
                conn.execute("DROP INDEX IF EXISTS toy_ivf")
                conn.execute(f"CREATE INDEX toy_ivf ON toy USING ivfflat (v vector_l2_ops) WITH (lists = {LISTS})")
                cents = read_centroids(args.pgdata, conn, "toy_ivf")
                bins = bins_at_probes1(conn)
                part = tuple(sorted(set(bins.values())))
                partitions[part] += 1
                samples.setdefault(part, (cents, bins))
            print(f"lists={LISTS} on 8 rows -> NOTICE: {[n for n in notices if 'little data' in n] or '(none)'}")
            print(f"\nPartitions produced by {args.builds} rebuilds (k-means++ seed is per-backend, not SQL-seedable):")
            for part, n in partitions.most_common():
                print(f"  {n:3d}x  {[list(p) for p in part]}  centroids={[round(c, 3) for c in samples[part][0]]}")

            # Narrate the most common partition: rebuild until we get it (bounded).
            target = partitions.most_common(1)[0][0]
            for _ in range(200):
                conn.execute("DROP INDEX IF EXISTS toy_ivf")
                conn.execute(f"CREATE INDEX toy_ivf ON toy USING ivfflat (v vector_l2_ops) WITH (lists = {LISTS})")
                cents = read_centroids(args.pgdata, conn, "toy_ivf")
                bins = bins_at_probes1(conn)
                if tuple(sorted(set(bins.values()))) == target:
                    break

            print("\n=== Step-by-step on the most common partition ===")
            print(f"1. Data: {VALUES}")
            print(f"2. Sample: lists*50={LISTS*50} -> floored to 10000 -> capped at max tuples; all 8 rows are samples.")
            print(f"3. k-means++ init + Elkan k-means (<=500 iters) on 8 samples, {LISTS} centers.")
            print(f"   Centroids read from index block 1: {[round(c, 4) for c in cents]}")
            inv = collections.defaultdict(list)
            for name, v in VALUES.items():
                inv[bins[name]].append(name)
            for i, c in enumerate(cents):
                members = [n for n, v in VALUES.items() if min(range(len(cents)), key=lambda j: abs(v - cents[j])) == i]
                print(f"4. list {i}: centroid {c:.4f} <- {members} = {[VALUES[m] for m in members]}")
            print(f"   (verified via probes=1 self-queries: {sorted(set(bins.values()))})")
            for q in QUERIES:
                order = sorted(range(len(cents)), key=lambda j: abs(q - cents[j]))
                print(f"\n5. Query q={q}: centroid distances {[(i, round(abs(q-cents[i]),3)) for i in order]}")
                ex = exact_topk(q)
                print(f"   exact top-{K}: {[(v, d) for d, v in ex]}")
                for probes in (1, 2, 3):
                    got = topk(conn, q, probes)
                    opened = order[:probes]
                    miss = [v for _, v in ex if v not in [g for g, _ in got]]
                    print(f"   probes={probes}: lists opened {opened} -> top-{K} {got}"
                          + (f"  MISS {miss}" if miss else "  (exact)"))
            conn.execute("DROP TABLE toy")
    finally:
        srv.cleanup()


if __name__ == "__main__":
    main()
