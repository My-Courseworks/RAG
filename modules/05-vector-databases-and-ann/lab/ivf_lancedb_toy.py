"""IVF toy on LanceDB (Lance IVF_FLAT), module 5 supplementary: the course's eight
1-D values A..H = 1, 2, 4, 7, 8, 11, 14, 15 in three partitions.

LanceDB accepts d=1 for IVF_FLAT, so this runs the real index, not a simulation.
Lance's k-means seeds from the OS (rust/lance-index/src/vector/kmeans.rs,
`KMeansParams::rng`: `seed: None` -> `SmallRng::from_os_rng()`), and neither
lancedb nor the Lance Python API expose the seed, so the partitioning differs
between runs. The script therefore rebuilds the index until it lands on the
partitioning the markdown narrates ({A,B,C} {D,E,F} {G,H}), reports how many
tries that took and every other partitioning it saw on the way, and then reads
every number below from the live index.

Centroids are not exposed through the API; they are recovered as the mean of each
partition's members, which is exactly what Lance's Lloyd's update stores
(`KMeansAlgoFloat::to_kmeans`, kmeans.rs), and the recovered centroids are checked
for Voronoi consistency against the routing the index actually performed.

Run:
    python -I lab/ivf_lancedb_toy.py /path/to/scratch/dir
"""

from __future__ import annotations

import os
import shutil
import sys

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import IvfFlat

VALUES = [1, 2, 4, 7, 8, 11, 14, 15]
LABELS = list("ABCDEFGH")
TARGET = (("A", "B", "C"), ("D", "E", "F"), ("G", "H"))
MAX_TRIES = 40


def build(tbl) -> None:
    tbl.create_index("vector", config=IvfFlat(distance_type="l2", num_partitions=3, max_iterations=50,
                                               sample_rate=256), replace=True)


def read_partitions(tbl) -> list[tuple[str, ...]]:
    """One nprobes=1 query per value: with min == max nprobes Lance's probe policy is
    Fixed, so the result is exactly the contents of that value's own partition."""
    seen: dict[tuple[str, ...], None] = {}
    for v in VALUES:
        members = tuple(sorted(r["id"] for r in tbl.search([float(v)]).nprobes(1).limit(len(VALUES)).to_list()))
        seen.setdefault(members, None)
    return sorted(seen, key=lambda m: VALUES[LABELS.index(m[0])])


def centroid(members: tuple[str, ...]) -> float:
    return sum(VALUES[LABELS.index(m)] for m in members) / len(members)


def wcss(parts: list[tuple[str, ...]]) -> float:
    return sum((VALUES[LABELS.index(m)] - centroid(p)) ** 2 for p in parts for m in p)


def fmt_parts(parts) -> str:
    return " ".join("{" + ",".join(p) + "}" for p in parts)


def exact(q: float, k: int) -> list[tuple[str, float]]:
    return sorted(((l, (v - q) ** 2) for l, v in zip(LABELS, VALUES)), key=lambda t: (t[1], t[0]))[:k]


def main() -> None:
    scratch = sys.argv[1]
    db_dir = os.path.join(scratch, "ivf_lancedb_toy_db")
    shutil.rmtree(db_dir, ignore_errors=True)
    db = lancedb.connect(db_dir)
    tbl = db.create_table("toy", pa.table({
        "id": LABELS,
        "vector": pa.array([[float(v)] for v in VALUES], pa.list_(pa.float32(), 1)),
    }))

    seen: dict[tuple, int] = {}
    for attempt in range(1, MAX_TRIES + 1):
        build(tbl)
        parts = read_partitions(tbl)
        key = tuple(parts)
        seen[key] = seen.get(key, 0) + 1
        if key == TARGET:
            break
    else:
        print(f"did not reach the narrated partitioning in {MAX_TRIES} tries; narrating the last one")

    print(f"### Build (attempt {attempt} of at most {MAX_TRIES}; OS-seeded k-means)\n")
    print("Partitionings seen while rebuilding (random init, 1 redo, Lloyd's to tolerance 1e-4):\n")
    print("| Partitioning | Centroids | WCSS | Times seen |")
    print("|---|---|---|---|")
    for k, n in sorted(seen.items(), key=lambda kv: (wcss(list(kv[0])), kv[0])):
        print(f"| {fmt_parts(k)} | {', '.join(f'{centroid(p):.3f}' for p in k)} | {wcss(list(k)):.2f} | {n} |")
    print()
    print("index:", tbl.list_indices()[0].index_type, tbl.index_stats(tbl.list_indices()[0].name))
    print()

    cents = [centroid(p) for p in parts]
    print("### Step 1: centroids (mean of members, as Lance's Lloyd's update stores them)\n")
    for i, (p, c) in enumerate(zip(parts, cents)):
        print(f"- partition {i}: members {{{','.join(p)}}} -> centroid {c:.4f}")
    print()

    print("### Step 2: assignment = nearest centroid (argmin, first minimum wins)\n")
    print("| Value | d^2 to each centroid | Partition |")
    print("|---|---|---|")
    ok = True
    for l, v in zip(LABELS, VALUES):
        d2 = [(v - c) ** 2 for c in cents]
        nearest = d2.index(min(d2))
        actual = next(i for i, p in enumerate(parts) if l in p)
        ok &= nearest == actual
        print(f"| {l}={v} | {', '.join(f'{d:.3f}' for d in d2)} | {actual} |")
    print(f"\nVoronoi consistency of recovered centroids with the index's routing: {'OK' if ok else 'MISMATCH'}\n")

    for q in (5.5, 9.5):
        print(f"### Query q={q}, k=3\n")
        # Lance ranks centroids with f32 squared L2 (find_partitions, argmin keeps the first minimum)
        c32 = np.array(cents, dtype=np.float32)
        d2 = ((np.float32(q) - c32) ** 2).tolist()
        order = sorted(range(len(cents)), key=lambda i: (d2[i], i))
        print("- centroid d^2 (f32): " + ", ".join(f"partition {i}: {d2[i]:.6f}" for i in range(len(cents))))
        print(f"- probe order: {order}")
        ex = exact(q, 3)
        print(f"- exact top-3: {[(l, round(d, 2)) for l, d in ex]}")
        for n in (1, 2, 3):
            rows = tbl.search([q]).nprobes(n).limit(3).to_list()
            got = [(r["id"], round(r["_distance"], 2)) for r in rows]
            missed = [l for l, _ in ex if l not in {g for g, _ in got}]
            predicted = fmt_parts([parts[i] for i in order[:n]])
            # the partitions the index really opened: at nprobes=1 every returned row is from one partition
            observed = sorted({next(i for i, p in enumerate(parts) if g in p) for g, _ in got})
            agree = observed == sorted(order[:n]) if n == 1 else set(observed) <= set(order[:n])
            print(f"- nprobes={n}: opens {predicted} -> returns {got}"
                  + (f"  MISS {missed}" if missed else "  (exact)")
                  + (f"  only {len(got)} rows: the probed partitions hold fewer than k" if len(got) < 3 else "")
                  + ("" if agree else f"  [routing disagrees with the f32 prediction: rows came from {observed}]"))
        rows = tbl.search([q]).limit(3).to_list()
        got = [(r["id"], round(r["_distance"], 2)) for r in rows]
        print(f"- nprobes unset (Auto probing): returns {got}")
        print()

    print("### Plans (explain_plan)\n")
    print("nprobes(1):")
    print(tbl.search([5.5]).nprobes(1).limit(3).explain_plan(True))
    print("nprobes unset:")
    print(tbl.search([5.5]).limit(3).explain_plan(True))

    idx_root = os.path.join(db_dir, "toy.lance", "_indices")
    print("### Index files on disk\n")
    for uuid in sorted(os.listdir(idx_root)):
        for f in sorted(os.listdir(os.path.join(idx_root, uuid))):
            print(f"- _indices/{uuid}/{f}: {os.path.getsize(os.path.join(idx_root, uuid, f))} bytes")


if __name__ == "__main__":
    main()
