"""M5 worked example, Qdrant server (Docker) rows: exact, HNSW m/ef, int8 + rescore, binary + rescore, filters.

One fresh container per phase so memory numbers belong to one collection:
  build (indexing disabled during upload, then enabled and timed until green)
  -> docker restart (drops the build's heap and page cache) -> warm-up -> 2 latency runs -> memory.

Usage (needs prepare.py output in $WORKED_WORK):
  WORKED_WORK=... python -I worked_example_qdrant.py m16      # rows 1-3 + filters
  WORKED_WORK=... python -I worked_example_qdrant.py m48      # row 4
  WORKED_WORK=... python -I worked_example_qdrant.py int8     # row 5 + sweep
  WORKED_WORK=... python -I worked_example_qdrant.py binary   # row 6 + sweep
"""
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worked_example_common import (K, N_CORPUS, latency_summary, load_corpus, recall_at_k,  # noqa: E402
                                   time_queries, update_results, work_dir, hnsw_arithmetic)

IMAGE = "qdrant/qdrant:v1.20.1"
CONTAINER = "m5we-qdrant"
COLL = "dbpedia"
EF_CONSTRUCT = 100
REPEATS = 3

PHASES = {
    "m16": dict(m=16, on_disk=False, quant=None),
    "m48": dict(m=48, on_disk=False, quant=None),
    "int8": dict(m=16, on_disk=True,
                 quant=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
                     type=models.ScalarType.INT8, always_ram=True))),
    "binary": dict(m=16, on_disk=True,
                   quant=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))),
}


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def start_container():
    subprocess.run(f"docker rm -f {CONTAINER}", shell=True, capture_output=True)
    sh(f"docker run -d --name {CONTAINER} -p 6333:6333 -p 6334:6334 {IMAGE}")
    wait_ready()


def wait_ready(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            out = subprocess.run("curl -sf localhost:6333/readyz", shell=True, capture_output=True, text=True)
            if out.returncode == 0:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("qdrant not ready")


def wait_green(client, timeout=3600, accept_grey=False):
    """GREEN = optimizers idle. After a restart Qdrant reports GREY (optimizations not yet triggered) until the
    next update, so accept_grey is used only right after docker restart."""
    ok = {models.CollectionStatus.GREEN} | ({models.CollectionStatus.GREY} if accept_grey else set())
    t0 = time.time()
    while True:
        info = client.get_collection(COLL)
        if (info.status in ok and info.optimizer_status == models.OptimizersStatusOneOf.OK
                and (info.indexed_vectors_count or 0) >= N_CORPUS):
            return time.time() - t0, info
        if time.time() - t0 > timeout:
            raise RuntimeError(f"not green: {info}")
        time.sleep(0.25)


def memory_snapshot():
    stats = sh(f"docker stats --no-stream --format '{{{{.MemUsage}}}}' {CONTAINER}").strip()
    status = sh(f"docker exec {CONTAINER} sh -c 'for p in /proc/[0-9]*; do grep -qx qdrant $p/comm 2>/dev/null && cat $p/status; done; true'")
    proc = {k: int(v.split()[0]) * 1024 for k, v in re.findall(r"^(VmRSS|RssAnon|RssFile|RssShmem|VmHWM):\s+(.+)$", status, re.M)}
    smaps = sh(f"docker exec {CONTAINER} sh -c 'for p in /proc/[0-9]*; do grep -qx qdrant $p/comm 2>/dev/null && cat $p/smaps; done; true'")
    rss_by_map, cur = defaultdict(int), "?"
    for line in smaps.splitlines():
        head = line.split()
        if head and re.match(r"^[0-9a-f]+-[0-9a-f]+$", head[0]):
            path = head[5] if len(head) > 5 else "[anon]"
            path = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27}", "<seg>", path)
            cur = re.sub(r"\d+", "N", path.split(f"/{COLL}/")[-1].split("/segments/<seg>/")[-1])
        elif head and head[0] == "Rss:":
            rss_by_map[cur] += int(head[1]) * 1024
    metrics = sh("curl -s localhost:6333/metrics")
    jem = {k: int(float(v)) for k, v in re.findall(r"^(memory_\w+_bytes) (\S+)$", metrics, re.M)}
    files = sh(f"docker exec {CONTAINER} find /qdrant/storage/collections/{COLL} -type f -printf '%s %p\\n'")
    by_kind = defaultdict(int)
    for line in files.splitlines():
        size, path = line.split(" ", 1)
        name = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27}", "<seg>", path.split(f"/{COLL}/", 1)[1])
        name = re.sub(r"\d+", "N", name.split("/segments/<seg>/")[-1])
        by_kind[name] += int(size)
    return {
        "docker_stats_mem_usage": stats,
        "proc_status_bytes": proc,
        "jemalloc_metrics_bytes": jem,
        "smaps_rss_bytes_by_mapping": dict(sorted(rss_by_map.items(), key=lambda kv: -kv[1])[:12]),
        "storage_bytes_total": sum(by_kind.values()),
        "storage_bytes_by_file": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])[:25]),
    }


def search_path_counts():
    """Which search path each segment took (telemetry counters): unfiltered_hnsw, filtered_small_cardinality
    (payload index + exact scan), filtered_large_cardinality (filtered graph walk), filtered_acorn, ..."""
    t = json.loads(sh("curl -s 'localhost:6333/telemetry?details_level=10'"))["result"]
    local = t["collections"]["collections"][0]["shards"][0]["local"]
    agg = defaultdict(int)
    for seg in local["segments"]:
        for vi in seg["vector_index_searches"]:
            for k, v in vi.items():
                if isinstance(v, dict):
                    agg[k] += v.get("count", 0)
    segs = [(s["info"]["segment_type"], s["info"]["num_points"]) for s in local["segments"]]
    return {"counts": dict(agg), "segments": segs}


def build(client, cfg, x, payloads):
    client.create_collection(
        COLL,
        vectors_config=models.VectorParams(size=x.shape[1], distance=models.Distance.COSINE, on_disk=cfg["on_disk"]),
        hnsw_config=models.HnswConfigDiff(m=cfg["m"], ef_construct=EF_CONSTRUCT),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),  # no HNSW during upload
        quantization_config=cfg["quant"],
    )
    for field in ("sim", "band"):
        client.create_payload_index(COLL, field, models.PayloadSchemaType.INTEGER, wait=True)
    t0 = time.time()
    client.upload_collection(COLL, vectors=x, payload=payloads, ids=range(len(x)), batch_size=256, parallel=4, wait=True)
    t_upload = time.time() - t0
    t0 = time.time()
    client.update_collection(COLL, optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10000))
    t_index, info = wait_green(client)
    return {"upload_s": round(t_upload, 1), "index_build_s": round(t_index, 1),
            "segments": info.segments_count, "indexed_vectors": info.indexed_vectors_count,
            "config": json.loads(info.config.model_dump_json())}


def make_search(client, params, flt=None):
    def one(q):
        r = client.query_points(COLL, query=q, limit=K, search_params=params, query_filter=flt,
                                with_payload=False, with_vectors=False)
        return [p.id for p in r.points]
    return one


def measure(client, qlist, true_ids, params, flt=None, repeats=REPEATS, label=""):
    runs, ids = [], None
    for _ in range(repeats):
        ids, lat = time_queries(make_search(client, params, flt), qlist)
        runs.append(lat)
    out = {"recall_at_10": round(recall_at_k(ids, true_ids), 4), **latency_summary(runs)}
    print(f"  {label:<48} recall@10={out['recall_at_10']:.4f}  p50={out['p50_ms']:.2f} ms  p99={out['p99_ms']:.2f} ms")
    return out


def batch_qps(client, qlist, params, batch=100):
    reqs = [models.QueryRequest(query=q, limit=K, params=params) for q in qlist]
    client.query_batch_points(COLL, requests=reqs[:batch])  # warm
    t0 = time.time()
    for s in range(0, len(reqs), batch):
        client.query_batch_points(COLL, requests=reqs[s:s + batch])
    return round(len(reqs) / (time.time() - t0), 1)


def run_phase(phase):
    cfg = PHASES[phase]
    x, q, gt = load_corpus()
    w = work_dir()
    sim, band = np.load(w / "filter_sim.npy"), np.load(w / "filter_real.npy")
    # Filter fields exist only on matching points, so no payload block is large enough to get extra HNSW links
    # at the default full_scan_threshold, and the unfiltered graph is the plain one.
    payloads = [({"sim": 1} if s else {}) | ({"band": 1} if b else {}) for s, b in zip(sim, band)]
    qlist = [v.tolist() for v in q]
    true10 = gt["top100"][:, :K]

    start_container()
    client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True, timeout=600)
    print(f"[{phase}] building ...")
    b = build(client, cfg, np.asarray(x), payloads)
    print(f"  upload {b['upload_s']} s, index build {b['index_build_s']} s, segments {b['segments']}")
    time.sleep(10)
    wait_green(client)  # still idle 10 s later: no follow-up merge in flight
    tel = json.loads(sh("curl -s 'localhost:6333/telemetry?details_level=10'"))["result"]
    local = tel["collections"]["collections"][0]["shards"][0]["local"]
    b["optimizer_telemetry"] = local["optimizations"]
    b["segments_before_restart"] = [(sg["info"]["segment_type"], sg["info"]["num_points"],
                                     sg["config"]["vector_data"][""]["index"]) for sg in local["segments"]]
    print("  optimizer log:", [(e.get("name"), e.get("status"), e.get("start_at"), e.get("end_at")) for e in local["optimizations"].get("log", [])])
    print("  segments before restart:", b["segments_before_restart"])
    sh(f"docker restart {CONTAINER}")
    wait_ready()
    client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True, timeout=600)
    # After a restart Qdrant reports GREY until an update triggers the optimizers; an optimization the
    # restart interrupted leaves a non-indexed (plain) segment that is brute-forced on every query.
    # Trigger the optimizers and require every non-empty segment to be HNSW-indexed before measuring.
    client.update_collection(COLL, optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10000))
    time.sleep(2)
    wait_green(client)
    segs = search_path_counts()["segments"]
    assert all(t == "indexed" for t, n in segs if n > 0), segs
    b["segments_measured"] = segs
    mem_after_restart = memory_snapshot()

    res = {"phase": phase, "image": IMAGE, "m": cfg["m"], "ef_construct": EF_CONSTRUCT,
           "original_vectors_on_disk": cfg["on_disk"], "build": b, "searches": {}}
    S = res["searches"]
    if cfg["quant"] is None:
        efs = [64, 256]
        if phase == "m16":
            S["exact"] = measure(client, qlist, true10, models.SearchParams(exact=True), label="exact=True (brute force in Qdrant)")
        for ef in efs:
            S[f"ef{ef}"] = measure(client, qlist, true10, models.SearchParams(hnsw_ef=ef), label=f"hnsw ef={ef}")
            S[f"ef{ef}"]["batch_qps_100"] = batch_qps(client, qlist, models.SearchParams(hnsw_ef=ef))
    else:
        for ef in (64, 256):
            for rescore in (True, False):
                for os_ in (1.0, 2.0, 4.0):
                    p = models.SearchParams(hnsw_ef=ef, quantization=models.QuantizationSearchParams(
                        ignore=False, rescore=rescore, oversampling=os_))
                    key = f"ef{ef}_os{os_:g}_rescore{'On' if rescore else 'Off'}"
                    S[key] = measure(client, qlist, true10, p, repeats=REPEATS if (ef == 64 and os_ == 2.0) else 1, label=key)
        main = models.SearchParams(hnsw_ef=64, quantization=models.QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0))
        S["ef64_os2_rescoreOn"]["batch_qps_100"] = batch_qps(client, qlist, main)
        S["ignore_quantization_ef64"] = measure(client, qlist, true10, models.SearchParams(
            hnsw_ef=64, quantization=models.QuantizationSearchParams(ignore=True)), label="ef=64, quantization ignored (originals)")
    res["search_paths_unfiltered"] = search_path_counts()
    info = client.get_collection(COLL)
    res["status_after_measurement"] = {"status": str(info.status), "optimizer": str(info.optimizer_status),
                                       "segments": info.segments_count}
    res["memory_after_restart"] = mem_after_restart
    res["memory_after_queries"] = memory_snapshot()

    quant_bytes = {"int8": 1536, "binary": 1536 // 8}.get(phase, 0)
    res["arithmetic"] = hnsw_arithmetic(N_CORPUS, x.shape[1], cfg["m"], extra_code_bytes=quant_bytes)
    update_results(f"qdrant_{phase}", res)
    if phase == "m16":
        res["filters"] = filter_tests(client, qlist, gt)
        update_results(f"qdrant_{phase}", res)
    print(json.dumps({k: res[k] for k in ("memory_after_restart", "memory_after_queries", "arithmetic")}, indent=1))


def filter_tests(client, qlist, gt):
    out = {}
    for field, key in (("sim", "sim_top10"), ("band", "real_top10")):
        flt = models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=1))])
        true = gt[key]
        out[f"{field}_default_planner_ef64"] = measure(client, qlist, true, models.SearchParams(hnsw_ef=64), flt,
                                                       label=f"filter {field}: default planner, ef=64")
        out[f"{field}_exact"] = measure(client, qlist, true, models.SearchParams(exact=True), flt, repeats=1,
                                        label=f"filter {field}: exact")
    out["search_paths_before_rebuild"] = search_path_counts()
    print("  search paths:", out["search_paths_before_rebuild"])
    # Force the filtered graph walk: full_scan_threshold below the filtered cardinality (374 x 6 KB = 2.2 MB).
    # Changing hnsw_config rebuilds the index; 10 KB is the API minimum (~1.7 vectors at 1,536-d); with it the 374-point payload block also gets its own links.
    t0 = time.time()
    client.update_collection(COLL, hnsw_config=models.HnswConfigDiff(full_scan_threshold=10))
    time.sleep(2)
    t_rebuild, _ = wait_green(client)
    out["forced_graph_rebuild_s"] = round(time.time() - t0, 1)
    for field, key in (("sim", "sim_top10"), ("band", "real_top10")):
        flt = models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=1))])
        true = gt[key]
        for ef in (64, 256):
            out[f"{field}_forced_graph_ef{ef}"] = measure(client, qlist, true, models.SearchParams(hnsw_ef=ef), flt,
                                                          repeats=1, label=f"filter {field}: forced graph walk, ef={ef}")
        out[f"{field}_forced_graph_acorn_ef64"] = measure(
            client, qlist, true, models.SearchParams(hnsw_ef=64, acorn=models.AcornSearchParams(enable=True)), flt,
            repeats=1, label=f"filter {field}: forced graph walk + ACORN, ef=64")
    out["unfiltered_after_threshold_change_ef64"] = measure(client, qlist, gt["top100"][:, :K], models.SearchParams(hnsw_ef=64),
                                                            repeats=1, label="unfiltered after rebuild, ef=64")
    out["search_paths_after_rebuild"] = search_path_counts()
    print("  search paths:", out["search_paths_after_rebuild"])
    return out


if __name__ == "__main__":
    run_phase(sys.argv[1])
