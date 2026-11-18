"""Elasticsearch DiskBBQ (`bbq_disk`) toy for the RAG course, module 5.

Two parts:

1. SIMULATION of Elasticsearch 9.2.0's HierarchicalKMeans + SOAR + IVF search on the
   course's eight 1-D values A..H = 1,2,4,7,8,11,14,15 with "nlist = 3" (targetSize=3).
   ES itself refuses cluster_size < 64 and collapses <= cluster_size vectors into ONE
   centroid, so three clusters can only be shown by re-implementing its algorithm.
   Mirrors (tag v9.2.0, commit 25d88452371273dd27356c98598287b669a03eae):
     server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/HierarchicalKMeans.java
       cluster()            L56-88   clusterAndSplit()  L90-165
     server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/KMeansLocal.java
       pickInitialCentroids L54-70   stepLloyd L72-134  getBestCentroid L189-212
       assignSpilled (SOAR) L214-300 doCluster L337-352 cluster L354-386
     libs/simdvec/.../internal/vectorization/DefaultESVectorUtilSupport.java soarDistance L154-164
     server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/IVFVectorsReader.java
       search() L258-332 (visitRatio -> maxVectorVisited, list loop)
   java.util.Random(42) is re-implemented bit-exactly so the reservoir seeding matches.
   Distances are exact floats: 1-bit BBQ codes make no sense at d=1 (see markdown).

2. REAL RUN (if an ES 9.2 is reachable at --es): index the same 8 values as dims=1
   `bbq_disk`, force-merge, query 9.5. Shows what ES actually does with 8 vectors.

Usage: python -I ivf_elasticsearch_toy.py [--es http://localhost:9200] [--no-es]
"""

import argparse
import math
import sys

VALUES = {"A": 1.0, "B": 2.0, "C": 4.0, "D": 7.0, "E": 8.0, "F": 11.0, "G": 14.0, "H": 15.0}
NAMES = list(VALUES)
X = [[v] for v in VALUES.values()]  # 1-D vectors
QUERIES = [[9.5], [5.5]]  # 9.5 as in the course; 5.5 sits on the c0/c1 boundary
TARGET_SIZE = 3  # ES "cluster_size" (vectors per cluster) -> nlist ~ N/3 = 3
MAX_ITER = 6  # HierarchicalKMeans.MAX_ITERATIONS_DEFAULT (v9.2.0)
SAMPLES_PER_CLUSTER = 64  # HierarchicalKMeans.SAMPLES_PER_CLUSTER_DEFAULT
MAXK = 128  # HierarchicalKMeans.MAXK (also clustersPerNeighborhood)
SOAR_LAMBDA = 1.0  # HierarchicalKMeans.DEFAULT_SOAR_LAMBDA
SOAR_MIN_DISTANCE = 1e-16  # KMeansLocal.SOAR_MIN_DISTANCE
NO_SOAR = -1


class JavaRandom:
    """Bit-exact java.util.Random (48-bit LCG) so pickInitialCentroids seeds match ES."""

    MULT, ADD, MASK = 0x5DEECE66D, 0xB, (1 << 48) - 1

    def __init__(self, seed):
        self.seed = (seed ^ self.MULT) & self.MASK

    def next(self, bits):
        self.seed = (self.seed * self.MULT + self.ADD) & self.MASK
        r = self.seed >> (48 - bits)
        return r - (1 << 32) if r >= (1 << 31) else r

    def next_double(self):
        return ((self.next(26) << 27) + self.next(27)) * (1.0 / (1 << 53))

    def next_int(self, bound):
        r = self.next(31)
        m = bound - 1
        if bound & m == 0:
            return (bound * r) >> 31
        u = r
        while True:
            r = u % bound
            if u - r + m < (1 << 31):  # Java: while (u - r + m < 0) overflow check
                return r
            u = self.next(31)


def sqdist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def pick_initial_centroids(vectors, k):
    """KMeansLocal.pickInitialCentroids L54-70: first k vectors, then reservoir replacement."""
    rnd = JavaRandom(42)
    cents = [list(v) for v in vectors[:k]]
    for i in range(k, len(vectors)):
        if rnd.next_double() < k * (1.0 / i):
            c = rnd.next_int(k)
            cents[c] = list(vectors[i])
    return cents


def step_lloyd(vectors, centroids, assignments, log):
    """KMeansLocal.stepLloyd L72-134 (no neighborhoods): assign, then recompute only changed centroids."""
    changed_set = set()
    changed = False
    for i, v in enumerate(vectors):
        # getBestCentroid L189-212: strict '<' so ties go to the lower centroid ordinal
        best = min(range(len(centroids)), key=lambda c: (sqdist(v, centroids[c]), c))
        if assignments[i] != best:
            if assignments[i] != -1:
                changed_set.add(assignments[i])
            changed_set.add(best)
            assignments[i] = best
            changed = True
    if changed:
        for c in changed_set:
            members = [vectors[i] for i in range(len(vectors)) if assignments[i] == c]
            if members:
                centroids[c] = [sum(m[d] for m in members) / len(members) for d in range(len(members[0]))]
    log.append(f"      centroids={[round(c[0], 3) for c in centroids]} assignments={assignments}")
    return changed


def kmeans_local_cluster(vectors, centroids, assignments, max_iter, log):
    """KMeansLocal.cluster L354-386 (sampleSize >= n here, so no sampling and no final pass)."""
    if len(centroids) == 1:
        for i in range(len(assignments)):
            assignments[i] = 0
        return
    for it in range(max_iter):
        log.append(f"    Lloyd iteration {it + 1}:")
        if not step_lloyd(vectors, centroids, assignments, log):
            log.append("      no assignment changed -> stop")
            break


def cluster_and_split(vectors, target_size, log, depth=0):
    """HierarchicalKMeans.clusterAndSplit L90-165. Returns (centroids, assignments)."""
    pad = "  " * depth
    n = len(vectors)
    if n <= target_size:
        return [], []
    k = min(max(int((n + target_size / 2.0) / float(target_size)), 2), MAXK)  # L95
    m = min(k * SAMPLES_PER_CLUSTER, n)  # L96 (sample size; here m == n)
    log.append(f"{pad}clusterAndSplit(n={n}, targetSize={target_size}): k = clamp(int((n + targetSize/2)/targetSize), 2, 128) = {k}; sample m = {m}")
    centroids = pick_initial_centroids(vectors, k)
    log.append(f"{pad}  pickInitialCentroids (java.util.Random(42) reservoir) -> {[c[0] for c in centroids]}")
    assignments = [-1] * n
    kmeans_local_cluster(vectors, centroids, assignments, MAX_ITER, log)

    counts = [assignments.count(c) for c in range(len(centroids))]
    log.append(f"{pad}  cluster sizes = {counts}")
    effective = [c for c in range(len(centroids)) if counts[c] > 0]
    if len(effective) == 1:
        return [centroids[effective[0]]], [0] * n

    # L129-162: split clusters with 100*count > 134*targetSize, remove empty ones
    removed = 0
    for c, count in enumerate(counts):
        adj = c - removed
        if 100 * count > 134 * target_size:
            log.append(f"{pad}  cluster {adj} has {count} > 1.34*{target_size} vectors -> recurse")
            idx = [i for i in range(n) if assignments[i] == adj]
            sub_c, sub_a = cluster_and_split([vectors[i] for i in idx], target_size, log, depth + 1)
            if sub_c:
                base = len(centroids)
                centroids[adj] = sub_c[0]
                centroids.extend(sub_c[1:])
                for j, i in enumerate(idx):
                    if sub_a[j] != 0:
                        assignments[i] = sub_a[j] + base - 1
        elif count == 0:
            log.append(f"{pad}  cluster {adj} is empty -> removed")
            del centroids[adj]
            assignments = [a - 1 if a > adj else a for a in assignments]
            removed += 1
    return centroids, assignments


def soar_distance(v, c, residual, lam, rnorm):
    """DefaultESVectorUtilSupport.soarDistance L154-164: ||v-c||^2 + lambda * ((v-c).(v-c1))^2 / ||v-c1||^2."""
    dsq = sqdist(v, c)
    proj = sum((v[i] - c[i]) * residual[i] for i in range(len(v)))
    return dsq + lam * proj * proj / rnorm


def assign_spilled(vectors, centroids, assignments, lam, log):
    """KMeansLocal.assignSpilled L214-300 with neighborhoods == null (k <= clustersPerNeighborhood)."""
    spilled = [NO_SOAR] * len(vectors)
    for i, v in enumerate(vectors):
        c1 = assignments[i]
        rnorm = sqdist(v, centroids[c1])
        if rnorm <= SOAR_MIN_DISTANCE:
            log.append(f"    {NAMES[i]}={v[0]:>4}: sits on its centroid -> no SOAR assignment")
            continue
        residual = [v[d] - centroids[c1][d] for d in range(len(v))]
        cands = [c for c in range(len(centroids)) if c != c1]
        scores = {c: soar_distance(v, centroids[c], residual, lam, rnorm) for c in cands}
        best = min(cands, key=lambda c: (scores[c], c))
        spilled[i] = best
        log.append(f"    {NAMES[i]}={v[0]:>4}: primary c{c1} ({centroids[c1][0]:.3f}); soar={{" + ", ".join(f"c{c}: {scores[c]:.2f}" for c in cands) + f"}} -> spill to c{best}")
    return spilled


def hierarchical_kmeans(vectors, target_size, log):
    """HierarchicalKMeans.cluster L56-88."""
    n = len(vectors)
    if n <= target_size:  # L63-77: single mean centroid, no k-means at all
        mean = [sum(v[d] for v in vectors) / n for d in range(len(vectors[0]))]
        return [mean], [0] * n, [NO_SOAR] * n
    centroids, assignments = cluster_and_split(vectors, target_size, log)
    spilled = [NO_SOAR] * n
    if 1 < len(centroids) < n:  # L81-85: final KMeansLocal pass, neighbor-aware + SOAR
        local_sample = min(len(centroids) * SAMPLES_PER_CLUSTER // 2, n)
        log.append(f"final KMeansLocal pass (sample={local_sample}, clustersPerNeighborhood={MAXK}): {len(centroids)} centroids <= {MAXK} -> no neighborhoods, plain Lloyd then SOAR")
        kmeans_local_cluster(vectors, centroids, assignments, MAX_ITER, log)
        log.append(f"  SOAR spill (lambda={SOAR_LAMBDA}):")
        spilled = assign_spilled(vectors, centroids, assignments, SOAR_LAMBDA, log)
    return centroids, assignments, spilled


def l2_score(d2):
    return 1.0 / (1.0 + d2)  # ES l2_norm similarity -> score


def ivf_search(query, centroids, postings, visit_ratio, k, log):
    """IVFVectorsReader.search L258-332: flat centroid scan, then open lists while
    maxVectorVisited > expectedDocs (or nothing collected yet). expectedDocs counts SOAR copies."""
    n = len(X)
    max_visited = int(2.0 * visit_ratio * n)  # L289: x2 because SOAR can make us visit a vector twice
    order = sorted(range(len(centroids)), key=lambda c: (sqdist(query, centroids[c]), c))
    log.append(f"  centroid scan (flat, {len(centroids)} <= 16*16 so no parent layer): nearest-first = {[f'c{c} ({centroids[c][0]:.3f})' for c in order]}")
    log.append(f"  visit_percentage={visit_ratio * 100:.1f}% -> maxVectorVisited = int(2 * {visit_ratio} * {n}) = {max_visited}")
    top = {}
    expected = 0
    opened = []
    for c in order:
        if not (max_visited > expected or len(top) < k):
            break
        opened.append(c)
        expected += len(postings[c])
        for i in postings[c]:
            top[i] = l2_score(sqdist(query, X[i]))  # 1-bit BBQ scoring replaced by exact (d=1)
        log.append(f"  open c{c}: posting = {[NAMES[i] for i in postings[c]]} (len {len(postings[c])}); expectedDocs={expected}")
    ranked = sorted(top, key=lambda i: -top[i])[:k]
    return opened, ranked


def simulation():
    log = []
    print("=" * 78)
    print("PART 1 - SIMULATION of ES 9.2.0 HierarchicalKMeans + SOAR + IVF search (d=1, targetSize=3)")
    print("=" * 78)
    print(f"values: {', '.join(f'{k}={int(v)}' for k, v in VALUES.items())}; queries = {[q[0] for q in QUERIES]}")
    centroids, assignments, spilled = hierarchical_kmeans(X, TARGET_SIZE, log)
    print("\nStep 1-3  train (hierarchical k-means), assign, SOAR:")
    print("\n".join(log))

    postings = {c: [i for i in range(len(X)) if assignments[i] == c or spilled[i] == c] for c in range(len(centroids))}
    print("\nStep 4  posting lists (primary + SOAR copies):")
    for c in range(len(centroids)):
        prim = [NAMES[i] for i in range(len(X)) if assignments[i] == c]
        spill = [NAMES[i] for i in range(len(X)) if spilled[i] == c]
        print(f"  c{c} centroid={centroids[c][0]:.3f}: primary={prim} soar_copies={spill} -> stored {len(postings[c])} codes")
    total = sum(len(p) for p in postings.values())
    print(f"  total codes stored = {total} for {len(X)} vectors ({total / len(X):.2f}x; SOAR costs disk, each code is 1 bit/dim + 14 B corrections)")

    k = 3
    for QUERY in QUERIES:
        exact = sorted(range(len(X)), key=lambda i: (abs(X[i][0] - QUERY[0]), i))[:k]
        print(f"\nStep 5  search query={QUERY[0]}, k={k}; exact top-{k} = {[NAMES[i] for i in exact]} (distances {[abs(X[i][0] - QUERY[0]) for i in exact]})")
        n = len(X)
        # visit ratios chosen so maxVectorVisited covers exactly the first list / the first two lists
        order = sorted(range(len(centroids)), key=lambda c: (sqdist(QUERY, centroids[c]), c))
        one = len(postings[order[0]]) / (2.0 * n)
        two = (len(postings[order[0]]) + len(postings[order[1]])) / (2.0 * n)
        for label, vr in ((f"'nprobe=1' (visit_percentage={one * 100:.1f})", one), (f"'nprobe=2' (visit_percentage={two * 100:.1f})", two)):
            log = []
            opened, ranked = ivf_search(QUERY, centroids, postings, vr, k, log)
            print(f"\n  {label}:")
            print("\n".join(log))
            missed = [NAMES[i] for i in exact if i not in ranked]
            print(f"  -> returned {[NAMES[i] for i in ranked]} ; exact {[NAMES[i] for i in exact]} ; " + (f"MISS {missed}" if missed else "no miss"))
    print("\n  rescoring: ES default rescore_vector.oversample=3 -> ES asks the IVF for ceil(3k)=9 candidates, re-scores them with raw floats, keeps k. It cannot recover a vector whose list was never opened.")
    return centroids, assignments, spilled


def real_es(base):
    import requests

    print("\n" + "=" * 78)
    print(f"PART 2 - REAL Elasticsearch at {base}: dims=1 bbq_disk, 8 docs")
    print("=" * 78)
    s = requests.Session()
    try:
        info = s.get(base).json()
    except Exception as e:  # noqa: BLE001
        print(f"ES not reachable ({e}); skipping real run")
        return
    print(f"ES {info['version']['number']}")
    name = "ivf-toy-1d"
    s.delete(f"{base}/{name}")
    r = s.put(
        f"{base}/{name}",
        json={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {"v": {"type": "dense_vector", "dims": 1, "index": True, "similarity": "l2_norm", "index_options": {"type": "bbq_disk", "cluster_size": 64}}}},
        },
    )
    print(f"create index dims=1 bbq_disk cluster_size=64 -> {r.status_code} {r.text[:120]}")
    r = s.put(f"{base}/{name}-bad", json={"mappings": {"properties": {"v": {"type": "dense_vector", "dims": 1, "index": True, "index_options": {"type": "bbq_disk", "cluster_size": 3}}}}})
    print(f"create index with cluster_size=3 -> {r.status_code}: {r.json()['error']['reason'][:100]}")
    lines = []
    for k_, v in VALUES.items():
        lines.append('{"index":{"_id":"%s"}}' % k_)
        lines.append('{"v":[%s]}' % v)
    s.post(f"{base}/{name}/_bulk?refresh=true", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"})
    s.post(f"{base}/{name}/_forcemerge?max_num_segments=1")
    s.post(f"{base}/{name}/_refresh")
    print("indexed A..H, force-merged to 1 segment. 8 vectors <= cluster_size (64) -> HierarchicalKMeans.cluster L63-77: ONE mean centroid (8.0), one posting list of 8.")
    for vp in (100.0, 1.0):
        r = s.post(f"{base}/{name}/_search", json={"knn": {"field": "v", "query_vector": QUERIES[0], "k": 3, "visit_percentage": vp}, "size": 3, "_source": False}).json()
        hits = [(h["_id"], round(h["_score"], 4)) for h in r["hits"]["hits"]]
        print(f"knn query 9.5, k=3, visit_percentage={vp}: {hits}  (score = 1/(1+d^2); exact top-3 = E, F, D)")
    print("With a single list every visit_percentage opens it (the loop always opens at least one list), so recall is 1.0 - the toy cannot show a miss on real ES.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", default="http://localhost:9200")
    ap.add_argument("--no-es", action="store_true")
    args = ap.parse_args()
    simulation()
    if not args.no_es:
        real_es(args.es)


if __name__ == "__main__":
    sys.exit(main())
