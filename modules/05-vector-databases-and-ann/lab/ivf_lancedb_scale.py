"""IVF scale test on LanceDB (Lance IVF_FLAT and IVF_PQ), module 5 supplementary.

Same generator as every other provider in supplementary/ivf/: N=187_043 points in
256-D around 50 Gaussian topic centres, 1000 queries, L2, recall@10 against a
brute-force ground truth. Builds IVF_FLAT with 4096 partitions, then IVF_PQ with
32 sub-vectors (8 dims each, 8 bits), sweeps `nprobes` and `refine_factor`, and
prints Markdown tables that supplementary/ivf/lancedb.md quotes.

Run (see the "Reproduce" section of the markdown for the venv):
    python -I lab/ivf_lancedb_scale.py /path/to/scratch/dir [--nlist1024]

Deterministic in the data; Lance's k-means seeds from the OS
(rust/lance-index/src/vector/kmeans.rs `KMeansParams::rng`, seed=None), so the
partitioning and therefore the recall figures move a little from run to run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import IvfFlat, IvfPq

N, D, NQ, K = 187_043, 256, 1000, 10
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def make_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((50, D), dtype=np.float32)
    topic = rng.integers(0, 50, N + NQ)
    x = centres[topic] + 0.35 * rng.standard_normal((N + NQ, D), dtype=np.float32)
    return x[:N], x[N:]


def brute_force_top_k(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """Exact L2 top-k ids, chunked over the base so the distance matrix fits in RAM."""
    xb_sq = (xb * xb).sum(1)
    best_d = np.full((len(xq), k), np.inf, dtype=np.float32)
    best_i = np.zeros((len(xq), k), dtype=np.int64)
    step = 20_000
    for start in range(0, len(xb), step):
        chunk = xb[start : start + step]
        d = xb_sq[start : start + step][None, :] - 2.0 * xq @ chunk.T  # + |q|^2, constant per row
        cand_d = np.concatenate([best_d, d], axis=1)
        cand_i = np.concatenate([best_i, np.arange(start, start + len(chunk))[None, :].repeat(len(xq), 0)], axis=1)
        order = np.argpartition(cand_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(cand_d, order, 1)
        best_i = np.take_along_axis(cand_i, order, 1)
    return best_i


def du_bytes(path: str) -> int:
    out = subprocess.run(["du", "-sk", path], capture_output=True, text=True, check=True).stdout
    return int(out.split()[0]) * 1024


def index_dir(table_dir: str) -> str:
    idx_root = os.path.join(table_dir, "_indices")
    uuids = sorted(os.listdir(idx_root), key=lambda u: os.path.getmtime(os.path.join(idx_root, u)))
    return os.path.join(idx_root, uuids[-1])


def recall_at_k(found: list[list[int]], gt: np.ndarray) -> float:
    hits = sum(len(set(f) & set(g.tolist())) for f, g in zip(found, gt))
    return hits / (len(gt) * K)


def partitions_in_index(tbl, q: np.ndarray) -> str:
    """`partitions_ranked=N` from analyze_plan: the partition count the index was really built with
    (Lance may train fewer than requested, see supported_num_partitions in rust/lance/src/index/vector.rs)."""
    plan = tbl.search(q).nprobes(1).limit(K).select(["id"]).analyze_plan()
    tokens = plan.replace(",", " ").split()
    for i, token in enumerate(tokens):
        if token.startswith("partitions_ranked="):
            value = token.split("=")[1]
            # datafusion prints large counts with a unit suffix ("4.10 K"); keep it verbatim
            return f"{value} {tokens[i + 1]}" if i + 1 < len(tokens) and tokens[i + 1] in ("K", "M") else value
    return "?"


def sweep(tbl, xq: np.ndarray, gt: np.ndarray, label: str, refine_factor: int | None) -> list[tuple]:
    rows = []
    for q in xq[:100]:  # warm the OS page cache and the index cache before timing
        tbl.search(q).nprobes(64).limit(K).select(["id"]).to_arrow()
    for nprobes in [None] + NPROBES:
        found = []
        t0 = time.perf_counter()
        for q in xq:
            qb = tbl.search(q).limit(K).select(["id"])
            if nprobes is not None:
                qb = qb.nprobes(nprobes)
            if refine_factor is not None:
                qb = qb.refine_factor(refine_factor)
            found.append(qb.to_arrow().column("id").to_pylist())
        ms = (time.perf_counter() - t0) * 1000 / len(xq)
        r = recall_at_k(found, gt)
        rows.append((label, "auto" if nprobes is None else nprobes, refine_factor, r, ms))
        print(f"  {label:<28} nprobes={'auto' if nprobes is None else nprobes:>4}  refine={refine_factor!s:<4} "
              f"recall@10={r:.4f}  {ms:7.2f} ms/query", flush=True)
    return rows


def first_reaching(rows: list[tuple], target: float = 0.95) -> str:
    for _, nprobes, _, r, _ in rows:
        if nprobes != "auto" and r >= target:
            return str(nprobes)
    return "none up to 256"


def build(tbl, table_dir: str, config, label: str, probe: np.ndarray) -> tuple[float, int, int, str]:
    t0 = time.perf_counter()
    tbl.create_index("vector", config=config, replace=True)
    build_s = time.perf_counter() - t0
    idir = index_dir(table_dir)
    size = du_bytes(idir)
    stats = tbl.index_stats(tbl.list_indices()[0].name)
    nparts = partitions_in_index(tbl, probe)
    print(f"[{label}] build {build_s:.1f} s, index dir {size/1e6:.1f} MB ({idir}), partitions built={nparts}, "
          f"stats={stats}", flush=True)
    return build_s, size, stats.num_indexed_rows, nparts


def main() -> None:
    scratch = sys.argv[1]
    do_1024 = "--nlist1024" in sys.argv
    db_dir = os.path.join(scratch, "ivf_lancedb_scale_db")
    shutil.rmtree(db_dir, ignore_errors=True)

    xb, xq = make_data()
    t0 = time.perf_counter()
    gt = brute_force_top_k(xb, xq, K)
    print(f"ground truth (numpy brute force): {time.perf_counter() - t0:.1f} s", flush=True)

    db = lancedb.connect(db_dir)
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(xb.ravel()), D)
    t0 = time.perf_counter()
    tbl = db.create_table("scale", pa.table({"id": pa.array(np.arange(N, dtype=np.int64)), "vector": vectors}))
    print(f"table write: {time.perf_counter() - t0:.1f} s, {tbl.count_rows()} rows", flush=True)
    table_dir = os.path.join(db_dir, "scale.lance")
    raw_mb = du_bytes(os.path.join(table_dir, "data")) / 1e6
    print(f"raw data files: {raw_mb:.1f} MB (float32 payload = {N * D * 4 / 1e6:.1f} MB)", flush=True)

    results: list[tuple] = []
    builds: list[tuple] = []

    cfg = IvfFlat(distance_type="l2", num_partitions=4096, max_iterations=50, sample_rate=256)
    b = build(tbl, table_dir, cfg, "IVF_FLAT nlist=4096", xq[0])
    builds.append(("IVF_FLAT nlist=4096", *b))
    results += sweep(tbl, xq, gt, "IVF_FLAT nlist=4096", None)

    cfg = IvfPq(distance_type="l2", num_partitions=4096, num_sub_vectors=32, num_bits=8,
                max_iterations=50, sample_rate=256)
    b = build(tbl, table_dir, cfg, "IVF_PQ nlist=4096 m=32", xq[0])
    builds.append(("IVF_PQ nlist=4096 m=32", *b))
    results += sweep(tbl, xq, gt, "IVF_PQ m=32 refine=None", None)
    results += sweep(tbl, xq, gt, "IVF_PQ m=32 refine=10", 10)

    if do_1024:
        cfg = IvfFlat(distance_type="l2", num_partitions=1024, max_iterations=50, sample_rate=256)
        b = build(tbl, table_dir, cfg, "IVF_FLAT nlist=1024", xq[0])
        builds.append(("IVF_FLAT nlist=1024", *b))
        results += sweep(tbl, xq, gt, "IVF_FLAT nlist=1024", None)

    print("\n### Builds\n")
    print("| Index | Partitions requested -> built | Build time (s) | Index dir on disk (MB) | Indexed rows |")
    print("|---|---|---|---|---|")
    for label, s, size, rows, nparts in builds:
        requested = label.split("nlist=")[1].split()[0]
        print(f"| {label} | {requested} -> {nparts} | {s:.1f} | {size/1e6:.1f} | {rows:,} |")
    print(f"\nRaw table data files: {raw_mb:.1f} MB.")

    print("\n### Sweep (recall@10 / ms per query, 1000 queries, one at a time through the Python client)\n")
    labels = []
    for r in results:
        if r[0] not in labels:
            labels.append(r[0])
    header = "| nprobes | " + " | ".join(labels) + " |"
    print(header)
    print("|---|" + "---|" * len(labels))
    for nprobes in ["auto"] + NPROBES:
        cells = []
        for label in labels:
            match = [r for r in results if r[0] == label and r[1] == nprobes]
            cells.append(f"{match[0][3]:.3f} / {match[0][4]:.1f} ms" if match else "")
        print(f"| {nprobes} | " + " | ".join(cells) + " |")
    print()
    for label in labels:
        rows = [r for r in results if r[0] == label]
        print(f"- smallest nprobes with recall@10 >= 0.95, {label}: {first_reaching(rows)}")


if __name__ == "__main__":
    main()
